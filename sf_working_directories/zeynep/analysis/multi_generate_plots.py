import json
import os
import pathlib
import sys
from glob import glob

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

mpl.rcParams["pdf.fonttype"] = 42
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams["font.size"] = 9


from sample_factory.utils.utils import experiment_dir, get_file_names, get_folder_names, log


def _ensure_parent(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        log.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        log.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)


def get_masked_occu(tmp):
    occu_display = np.ones_like(tmp)  # all white
    occu_display[np.isnan(tmp)] = np.nan  # mark obstacles

    masked_occu = np.ma.masked_invalid(occu_display)
    # print(masked_occu)
    cmap_obst = plt.cm.Greys.copy()
    cmap_obst.set_bad(color=blocked_colour)  # NaN → black (obstacles)
    return masked_occu, cmap_obst


def plot_place_fields(seq0_xya, experiment, experiment_subname, epoch_name):
    fig, ax = plt.subplots(4, 4, figsize=(8, 6), dpi=150)
    ax = ax.flatten()
    for i in range(hippo_n_feature):  # np.nonzero(SI_mlp2>0.2)[0]:
        ax[i].set_title(f"sequence{i}")
        tmp = seq0_xya[:, :, i].T
        # tmp[tmp<1e-30]=np.nan
        line = ax[i].imshow(tmp, extent=[100, 2000, 100, 2000], origin="lower", cmap="viridis")
        ax[i].set_axis_off()
        fig.colorbar(line, ax=ax[i], label="bar")
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_place_field.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


def plot_occupancy(occu_xy, experiment, experiment_subname, epoch_name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    # ax = ax.flatten()
    # for i in range(hippo_n_feature):#np.nonzero(SI_mlp2>0.2)[0]:
    ax.set_title(f"Occupancy")
    tmp = occu_xy[:, :].T
    tmp[tmp < 1e-30] = np.nan
    masked_occu, cmap_obst = get_masked_occu(tmp)
    ax.imshow(
        masked_occu,
        origin="lower",
        cmap=cmap_obst,
        extent=[xbound[0], xbound[1], ybound[0], ybound[1]],
        alpha=1.0,
        zorder=0,
    )
    line = ax.imshow(tmp, extent=[100, 2000, 100, 2000], origin="lower", cmap="viridis")
    ax.set_axis_off()
    fig.colorbar(line, ax=ax, label="bar")
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_occupancy.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


def plot_entire_traj(occu_xy, pddata, experiment, experiment_subname, epoch_name, colordata):
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    tmp = occu_xy[:, :].T
    tmp[tmp < 1e-30] = np.nan
    masked_occu, cmap_obst = get_masked_occu(tmp)

    ax2.imshow(
        masked_occu,
        origin="lower",
        cmap=cmap_obst,
        extent=[xbound[0], xbound[1], ybound[0], ybound[1]],
        alpha=1.0,
        zorder=0,
    )
    # ax = ax.flatten()
    # for i in range(4):#np.nonzero(SI_mlp2>0.2)[0]:
    # ax[i].set_title(f"sequence{i}")
    line = colored_line_between_pts(
        pddata["x"][start:stop], pddata["y"][start:stop], c=colordata, ax=ax2, linewidth=1, cmap="nipy_spectral"
    )
    fig2.colorbar(line, ax=ax2, label="12th env_frame")
    # plt.axis("equal")

    # ax2.set_xlim(xbound)
    # ax2.set_ylim(ybound)
    ax2.set_title("Individual Trajectory")
    # plt.axis("equal")

    plt.xlim(xbound)
    plt.ylim(ybound)
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_entire_traj.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


def plot_individual_traj(occu_xy, pddata, experiment, experiment_subname, epoch_name, colordata):
    traj_length = 900
    colordata2 = np.linspace(0, traj_length - 1, traj_length - 2)
    fig2, ax2 = plt.subplots(5, int((pddata["num_traj"].to_numpy()[-1] - 1) / 5), figsize=(22, 10), dpi=150)
    ax2 = ax2.flatten()

    tmp = occu_xy[:, :].T
    tmp[tmp < 1e-30] = np.nan
    masked_occu, cmap_obst = get_masked_occu(tmp)

    for i in range(int((pddata["num_traj"].to_numpy()[-1] - 1) / 5) * 5):
        ax2[i].imshow(
            masked_occu,
            origin="lower",
            cmap=cmap_obst,
            extent=[xbound[0], xbound[1], ybound[0], ybound[1]],
            alpha=1.0,
            zorder=0,
        )
        line = colored_line_between_pts(
            pddata["x"][i * traj_length : (i + 1) * traj_length - 1],
            pddata["y"][i * traj_length : (i + 1) * traj_length - 1],
            c=colordata2,
            ax=ax2[i],
            linewidth=1.2,
            cmap="nipy_spectral",
        )

        ax2[i].set_xlim(xbound)
        ax2[i].set_ylim(ybound)
        ax2[i].set_aspect("equal", adjustable="box")
        # Remove ticks but keep visible border
        ax2[i].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Make sure the axes border is visible and styled clearly
        for spine in ax2[i].spines.values():
            spine.set_visible(True)
            spine.set_color(blocked_colour)
            spine.set_linewidth(0.75)

    fig2.colorbar(line, ax=ax2, label="12th env_frame")
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_individual_traj.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


def plot_corr(seq0_xya, experiment, experiment_subname, epoch_name):
    # correlation of PFs between enjoys for each layer
    corr = pd.DataFrame(seq0_xya.reshape(-1, 16)).corr()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), dpi=150)
    im = ax.imshow(corr, cmap="coolwarm", vmax=1, vmin=0)
    ax.set_axis_off()
    ax.set_title(f"Epoch {i if i<7 else 'new'}", fontsize=8)
    # Label just one of them (optional)
    ax.set_xlabel("DG feature index")
    ax.set_ylabel("DG feature index")

    # --- add a single colorbar on the right ---
    # 1) create a new axes to the right of all subplots
    cbar_ax = fig.add_axes([0.96, 0.25, 0.01, 0.55])
    #    [left, bottom, width, height] in fractions of figure size

    # 2) draw the colorbar based on the last 'im' (they’re all the same scale)
    fig.colorbar(im, cax=cbar_ax, label="Correlation")

    # tidy up and save
    fig.tight_layout(rect=[0, 0, 0.94, 1], pad=0.15)  # leave space on the right for the colorbar
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_correlation.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


hippo_n_feature = 16
length = 71

rootpath = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam"

experiment = "InternalRewardSeparateReward3DG_"
experiment_path = pathlib.Path(rootpath) / experiment  # / "telemetry"

experiment_subnames = get_folder_names(experiment_path)
log.debug(f"Experiment subnames for experiment {experiment}: {experiment_subnames}")

### COLOLR Data
blocked_colour = "#C7C7C7"


for i in range(len(experiment_subnames)):
    log.info(i)
    telemetry_path = experiment_path / experiment_subnames[i] / "telemetry"
    if os.path.exists(telemetry_path):
        # log.warn("Path exists!")
        epoch_names = get_folder_names(telemetry_path)
        log.debug(f"Epoch names for experiment {experiment_subnames[i]}: {epoch_names}")
        # epoch_names = epoch_names[3:]

        for j in range(len(epoch_names)):
            telemetry_list_csv = glob(str(telemetry_path / epoch_names[j] / "*.csv"))
            telemetry_list_h5 = glob(str(telemetry_path / epoch_names[j] / "*.h5"))
            log.debug(telemetry_list_csv)
            log.debug(telemetry_list_h5)

            pddata = pd.read_csv(telemetry_list_csv[0])
            with h5py.File(telemetry_list_h5[0]) as h5:
                activations = {k: h5[k][...] for k in h5}
            log.debug(f"Shapes pdata: {pddata.shape}, activations: {activations['core'].shape}")
            log.debug(f"Shapes X and Y: {pddata['x'].shape, pddata['y'].shape}")

            tmp_act = activations
            sids = tmp_act["core"][:, :-13:length]

            xbound = (100, 2000)
            ybound = (100, 2000)
            grain = 19
            occu_xy = np.histogramdd(
                (pddata["x"], pddata["y"]),
                (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1)),
                density=False,
            )[0]

            grain = 19
            seq0_xya = np.zeros((grain, grain, hippo_n_feature))
            for k in range(hippo_n_feature):
                print(k)
                seq0_xya[:, :, k] = np.histogramdd(
                    (pddata["x"], pddata["y"]),
                    (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1)),
                    weights=sids[:, k],
                    density=False,
                )[0]
            log.info(
                f"Plotting for experiment: {experiment}, sub_experiment: {experiment_subnames[i]}, epoch: {epoch_names[j]}"
            )
            plot_place_fields(seq0_xya, experiment, experiment_subnames[i], epoch_names[j])
            plot_occupancy(occu_xy, experiment, experiment_subnames[i], epoch_names[j])
            start = 0
            stop = pddata.shape[0] - 1

            colordata = np.linspace(start, stop - 1, stop - start - 1)
            plot_entire_traj(occu_xy, pddata, experiment, experiment_subnames[i], epoch_names[j], colordata)
            if int((pddata["num_traj"].to_numpy()[-1] - 1) / 5) > 0:
                plot_individual_traj(occu_xy, pddata, experiment, experiment_subnames[i], epoch_names[j], colordata)
