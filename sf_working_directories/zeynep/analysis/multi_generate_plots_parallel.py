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


def compute_correlation_kernel(X, max_dx=5, max_dy=5, demean=True):
    """
    Computes a spatial correlation kernel from a 3D array X (H, W, F).

    Parameters:
        X: ndarray of shape (H, W, F)
        max_dx, max_dy: maximum spatial shifts in x and y
        demean: if True, subtract mean and normalize variance (z-score)

    Returns:
        kernel: 2D array of shape (2*max_dx+1, 2*max_dy+1) containing average correlation.
    """
    H, W, F = X.shape
    kernel = np.full((2 * max_dx + 1, 2 * max_dy + 1), np.nan)

    if demean:
        X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)

    for dx in range(-max_dx, max_dx + 1):
        for dy in range(-max_dy, max_dy + 1):
            x1s, x1e = max(0, -dx), min(H, H - dx)
            y1s, y1e = max(0, -dy), min(W, W - dy)

            A = X[x1s:x1e, y1s:y1e]
            B = X[x1s + dx : x1e + dx, y1s + dy : y1e + dy]

            if demean:
                # z-scored → Pearson correlation is just dot product
                corr = np.sum(A * B, axis=-1) / F
            else:
                # use cosine similarity: (A·B) / (||A|| * ||B||)
                A_norm = np.linalg.norm(A, axis=-1)
                B_norm = np.linalg.norm(B, axis=-1)
                dot = np.sum(A * B, axis=-1)
                corr = dot / (A_norm * B_norm + 1e-8)

            kernel[dx + max_dx, dy + max_dy] = np.nanmean(corr)

    return kernel


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
        fig.colorbar(line, ax=ax[i])
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


def plot_kernel(Spatial_kernel, n_epochs, experiment, experiment_subname):
    layer_names = list(Spatial_kernel.keys())
    n_layers = len(layer_names)
    # if n_epochs==1:
    #     n_epochs += 1

    # Create the grid of axes
    fig, ax = plt.subplots(n_layers, n_epochs, figsize=(5.5, n_layers * 0.65), dpi=300, constrained_layout=False)

    # Loop through layers (rows) and epochs (cols)
    for i, key in enumerate(layernames):
        for j in range(n_epochs):
            print(j)
            tmp = Spatial_kernel[key][j].copy()
            tmp[xygrain - 1, xygrain - 1] = np.nan
            im = ax[i, j].imshow(tmp[1:-1, 1:-1], origin="lower", vmin=0, vmax=1, cmap="coolwarm")
            ax[i, j].set_axis_off()
        ax[i, 0].text(-30, 15, displayname_dict[key], rotation=90, va="center", ha="center", fontsize=9)
        # # Left‐side row label
        # ax[i, 0].set_axis_on()
        # ax[i, 0].set_ylabel(key, rotation=0, labelpad=40, va='center')

    # Bottom‐left only: set tick‐labels and axis on
    ax[-1, 0].set_axis_on()
    ticks = np.arange(0, xygrain * 2 - 3, xygrain - 2)
    labels = ticks - xygrain + 2
    ax[-1, 0].set_xticks(ticks)
    ax[-1, 0].set_xticklabels(labels)
    ax[-1, 0].set_yticks(ticks)
    ax[-1, 0].set_yticklabels(labels)
    ax[-1, 0].set_xlabel(r"$\Delta x$", labelpad=0)
    ax[-1, 0].set_ylabel(r"$\Delta y$", labelpad=-4)

    # Column titles
    for j in range(n_epochs):
        # title = f"Epoch {j}" if j < n_epochs-1 else "Epoch new"
        title = j
        ax[0, j].set_title(title, pad=6, fontsize=9)
    ax[0, 0].set_title("Epoch 0", pad=6, fontsize=9)
    ax[0, -1].set_title("new", pad=6, fontsize=9)

    # Add a single horizontal colorbar below all subplots
    # Use the last image handle 'im' for the color mapping
    cax = fig.add_axes([0.97, 0.13, 0.02, 0.80])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        im,
        cax=cax,
        ax=ax,
        orientation="vertical",
        fraction=0.05,  # width of colorbar as fraction of total width
        pad=0.08,  # space between subplots and colorbar
    )
    cbar.set_label("Pearson correlation of PVs", labelpad=5)

    # Adjust layout so labels/colorbar aren't clipped
    fig.tight_layout(rect=[0, 0, 0.96, 1], pad=0.01)
    base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
    file_name = experiment + "_" + experiment_subname + "__kernel.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


hippo_n_feature = 16
length = 71
xygrain = 19
layernames = ["seq0", "seqall", "mlp0", "mlp2"]
displayname_dict = {"seq0": "DG", "seqall": "CA3", "mlp0": "Decoder 1", "mlp2": "Decoder 2"}

rootpath = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam"

experiment = "RandomTest_"
experiment_path = pathlib.Path(rootpath) / experiment  # / "telemetry"

experiment_subnames = get_folder_names(experiment_path)
log.debug(f"Experiment subnames for experiment {experiment}: {experiment_subnames}")

### COLOLR Data
blocked_colour = "#C7C7C7"


xbound = (100, 2000)
ybound = (100, 2000)


for i in range(len(experiment_subnames)):
    log.info(i)
    telemetry_path = experiment_path / experiment_subnames[i] / "telemetry"
    if os.path.exists(telemetry_path):
        # log.warn("Path exists!")
        epoch_names = get_folder_names(telemetry_path)
        log.debug(f"Epoch names for experiment {experiment_subnames[i]}: {epoch_names}")
        # epoch_names = epoch_names[3:]
        folders_containing_data = []
        for k in range(len(epoch_names)):
            folders_containing_data.append(str(telemetry_path / epoch_names[k]))
        print(len(folders_containing_data))

        ## aggregating data
        fields_enjoys = []
        PFs_enjoys = []
        SI_enjoys = []
        percent_active = []
        percent_active_all = []
        pddata_enjoys = []
        for analysispath in folders_containing_data:

            pddata = pd.read_csv(glob(str(analysispath + "/*.csv"))[0])
            with h5py.File(glob(str(analysispath + "/*.h5"))[0]) as h5:
                activations = {k: h5[k][...] for k in h5}
            pddata["rot_pi"] = pddata["rot_y"] * np.pi / 180
            pddata_enjoys.append(pddata)

            # activations['core']=torch.cat(activations['core'], dim=0).numpy()
            def SpatialInformation(firing, occu, per_spike=False):
                p_occu = occu / occu.flatten().sum()
                fr_x = firing / occu
                fr_mean = firing.flatten().sum() / occu[:].sum()
                out = np.nansum((fr_x * p_occu * np.log2(fr_x / fr_mean)).flatten())
                if per_spike:
                    out /= fr_mean
                return out

            ## calculate SIs
            xbound = (100, 2000)
            ybound = (100, 2000)
            grain = 19
            occu_xya = np.histogramdd(
                (pddata["x"], pddata["y"], pddata["rot_pi"]),
                (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                density=False,
            )[0]

            tmp_act = activations

            grain = 19
            hla_xya = np.zeros((grain, grain, 12, 128))
            for l in range(128):
                hla_xya[:, :, :, l] = np.histogramdd(
                    (pddata["x"], pddata["y"], pddata["rot_pi"]),
                    (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                    weights=tmp_act["decoder.mlp.2"][:, l] * (tmp_act["decoder.mlp.2"][:, l] > 0),
                    density=False,
                )[0]
            hla_xya = hla_xya[:, :, :, :] * (hla_xya[:, :, :, :] > 0)

            # grain=19
            hla0_xya = np.zeros((grain, grain, 12, 128))
            for l in range(128):
                hla0_xya[:, :, :, l] = np.histogramdd(
                    (pddata["x"], pddata["y"], pddata["rot_pi"]),
                    (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                    weights=tmp_act["decoder.mlp.0"][:, l] * (tmp_act["decoder.mlp.0"][:, l] > 0),
                    density=False,
                )[0]
            hla0_xya = hla0_xya[:, :, :, :] * (hla0_xya[:, :, :, :] > 0)

            if tmp_act["core"].shape[1] - 13 != 16 * length:
                print("size mismatch!!!!")
                break
            sids = tmp_act["core"][:, :-13:length]
            perc = (sids > 0).mean(0)
            percent_active.append(perc)
            percent_active_all.append((sids.sum(1) > 0).mean())

            # grain=19
            seq0_xya = np.zeros((grain, grain, 12, 16))
            for l in range(16):
                seq0_xya[:, :, :, l] = np.histogramdd(
                    (pddata["x"], pddata["y"], pddata["rot_pi"]),
                    (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                    weights=sids[:, l],
                    density=False,
                )[0]

            # grain=19
            seqall_xya = np.zeros((grain, grain, 12, 16 * length))
            for l in range(16 * length):
                seqall_xya[:, :, :, l] = np.histogramdd(
                    (pddata["x"], pddata["y"], pddata["rot_pi"]),
                    (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                    weights=tmp_act["core"][:, l],
                    density=False,
                )[0]

            # fields = {"seq0":seq0_xya,"seqall":seqall_xya,"occu":occu_xya}
            fields = {"mlp0": hla0_xya, "mlp2": hla_xya, "seq0": seq0_xya, "seqall": seqall_xya, "occu": occu_xya}
            fields_enjoys.append(fields)

            SIs = dict()
            for key in fields:
                if key != "occu":
                    SIs[key] = np.array(
                        [
                            SpatialInformation(fields[key][:, :, :, i], fields["occu"])
                            for i in range(fields[key].shape[-1])
                        ]
                    )
            SI_enjoys.append(SIs)
            PFs = dict()
            for key in fields:
                if key != "occu":
                    PFs[key] = np.nansum(fields[key], -2) / np.nansum(fields["occu"][:, :, :, None], -2)
            PFs_enjoys.append(PFs)
            # SI_mlp2=np.array([SpatialInformation(hla_xya[:,:,:,i],occu_xya) for i in range(128)])
            # SI_seq0=np.array([SpatialInformation(seq0_xya[:,:,:,i],occu_xya) for i in range(16)])
            # SI_seqall=np.array([SpatialInformation(seqall_xya[:,:,:,i],occu_xya) for i in range(16*15)])

        Spatial_kernel = dict()
        for key in PFs_enjoys[0]:
            tmpPFs_enjoys = np.array([x[key] for x in PFs_enjoys])
            # tmpPFs_enjoys=tmpPFs_enjoys.reshape(tmpPFs_enjoys.shape[0],-1,tmpPFs_enjoys.shape[-1])
            tmp_kernels = np.zeros((len(tmpPFs_enjoys), 2 * xygrain - 1, 2 * xygrain - 1))
            for epoch in range(len(tmpPFs_enjoys)):
                tmp = tmpPFs_enjoys[epoch, :, :, :]

                tmp_kernels[epoch, :, :] = compute_correlation_kernel(tmp, max_dx=xygrain - 1, max_dy=xygrain - 1)
            Spatial_kernel[key] = tmp_kernels

        print(len(epoch_names))
        for j in range(len(epoch_names)):
            log.info(
                f"Plotting for experiment: {experiment}, sub_experiment: {experiment_subnames[i]}, epoch: {epoch_names[j]}"
            )
            plot_place_fields(PFs_enjoys[j]["seq0"], experiment, experiment_subnames[i], epoch_names[j])
            plot_occupancy(
                np.nansum(fields_enjoys[j]["occu"][:, :, :], -1), experiment, experiment_subnames[i], epoch_names[j]
            )
            plot_corr(PFs_enjoys[j]["seq0"], experiment, experiment_subnames[i], epoch_names[j])
            start = 0
            stop = pddata_enjoys[j].shape[0] - 1

            colordata = np.linspace(start, stop - 1, stop - start - 1)
            plot_entire_traj(
                np.nansum(fields_enjoys[j]["occu"][:, :, :], -1),
                pddata_enjoys[j],
                experiment,
                experiment_subnames[i],
                epoch_names[j],
                colordata,
            )
            if int((pddata_enjoys[j]["num_traj"].to_numpy()[-1] - 1) / 5) > 0:
                plot_individual_traj(
                    np.nansum(fields_enjoys[j]["occu"][:, :, :], -1),
                    pddata_enjoys[j],
                    experiment,
                    experiment_subnames[i],
                    epoch_names[j],
                    colordata,
                )
        plot_kernel(Spatial_kernel, len(epoch_names), experiment, experiment_subnames[i])
    else:
        log.warning(f"PATH DOES NOT EXIST")
