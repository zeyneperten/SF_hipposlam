import json
import math as math
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


import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def get_masked_occu(tmp):
    occu_display = np.ones_like(tmp)  # all white
    occu_display[np.isnan(tmp)] = np.nan  # mark obstacles

    masked_occu = np.ma.masked_invalid(occu_display)
    # print(masked_occu)
    cmap_obst = plt.cm.Greys.copy()
    cmap_obst.set_bad(color=blocked_colour)  # NaN → black (obstacles)
    return masked_occu, cmap_obst


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
    file_name = experiment + "_" + experiment_subname + "_" + epoch_name + "_place_field_unnormalized.png"
    picture_path = pathlib.Path(base_picture_path) / experiment / file_name
    _ensure_parent(picture_path)
    log.debug(f"Picture path: {picture_path}")
    plt.savefig(picture_path)


# def animate_all_trajectories(
#     pddata,
#     occu_xy,
#     xbound,
#     ybound,
#     blocked_colour,
#     traj_length=900,
#     trail=200,
#     fps=30,
#     cmap_name="nipy_spectral",
#     outfile="all_trajectories.mp4",
#     dpi=150,
#     interval=20,
#     get_masked_occu=None
# ):
#     """
#     Create one movie where each trajectory is displayed one after another
#     with fading trails. The canvas resets for each trajectory.

#     Parameters
#     ----------
#     pddata : pandas DataFrame
#         Must contain "x", "y", and "num_traj".
#     occu_xy : 2D occupancy grid for background.
#     xbound, ybound : tuple/list
#         Axis limits.
#     blocked_colour : color
#         Color of the axis border.
#     traj_length : int
#         Number of points per trajectory.
#     trail : int
#         Number of previous points to keep with fading.
#     fps : int
#         Frames per second for output video.
#     cmap_name : str
#         Colormap name for trajectory.
#     outfile : str
#         Output filename (.mp4 or .gif).
#     dpi : int
#         Resolution.
#     interval : int
#         Delay between frames in ms (affects preview only).
#     get_masked_occu : function
#         Function that returns (masked_occu, cmap_obst).

#     Returns
#     -------
#     ani : matplotlib.animation.FuncAnimation
#     """

#     # ============================================================
#     # Extract number of trajectories
#     # ============================================================
#     n_traj = int(pddata["num_traj"].to_numpy()[-1] - 1)

#     # ============================================================
#     # 1. Extract all trajectories into a list
#     # ============================================================
#     trajectories = []
#     for i in range(n_traj):
#         start = i * traj_length
#         end   = start + traj_length
#         x = pddata["x"][start:end].to_numpy()
#         y = pddata["y"][start:end].to_numpy()
#         trajectories.append((x, y))

#     # ============================================================
#     # 2. Build figure and axis
#     # ============================================================
#     fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

#     # ============================================================
#     # 3. Prepare background occupancy map
#     # ============================================================
#     tmp = occu_xy[:,:].T
#     tmp[tmp < 1e-30] = np.nan
#     masked_occu, cmap_obst = get_masked_occu(tmp)

#     ax.imshow(
#         masked_occu,
#         origin='lower',
#         cmap=cmap_obst,
#         extent=[xbound[0], xbound[1], ybound[0], ybound[1]],
#         alpha=1.0,
#         zorder=0
#     )

#     ax.set_xlim(xbound)
#     ax.set_ylim(ybound)
#     ax.set_aspect('equal')

#     ax.tick_params(bottom=False, left=False,
#                    labelbottom=False, labelleft=False)

#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color(blocked_colour)
#         spine.set_linewidth(0.75)

#     # ============================================================
#     # 4. Prepare scatter plot (updated each frame)
#     # ============================================================
#     scat = ax.scatter([], [], s=8)

#     # fading color table
#     cmap = cm.get_cmap(cmap_name)
#     color_table = cmap(np.linspace(0, 1, trail))

#     # ============================================================
#     # 5. Timeline mapping (frames → trajectory + local index)
#     # ============================================================
#     frames_per_traj = traj_length
#     total_frames = frames_per_traj * n_traj

#     def find_traj(global_frame):
#         return global_frame // frames_per_traj

#     def local_frame(global_frame):
#         return global_frame % frames_per_traj

#     # ============================================================
#     # 6. Animation update() function
#     # ============================================================
#     def update(frame):
#         traj_idx = find_traj(frame)
#         local = local_frame(frame)
#         x, y = trajectories[traj_idx]

#         # Reset scatter at start of new trajectory
#         if local == 0:
#             np.empty((0, 2))

#         start_idx = max(0, local - trail)
#         X = x[start_idx:local]
#         Y = y[start_idx:local]

#         pts = np.column_stack([X, Y])

#         L = len(X)
#         if L == 0:
#             scat.set_offsets(np.empty((0, 2)))
#             scat.set_color([])
#             return scat,

#         alphas = np.linspace(0.05, 1.0, L)
#         cols = color_table[-L:].copy()
#         cols[:, -1] = alphas  # apply fading

#         scat.set_offsets(pts)
#         scat.set_color(cols)

#         return scat,
#     log.debug(f'FuncAnimation')
#     # ============================================================
#     # 7. Build animation
#     # ============================================================
#     ani = animation.FuncAnimation(
#         fig,
#         update,
#         frames=total_frames,
#         interval=interval,
#         blit=True
#     )
#     log.debug(f'Writing file')
#     # ============================================================
#     # 8. Save the movie
#     # ============================================================
#     ani.save(outfile, fps=fps, dpi=dpi, writer="ffmpeg")
#     log.debug(f'All done!')

#     return ani


def animate_all_trajectories(
    pddata,
    occu_xy,
    xbound,
    ybound,
    blocked_colour,
    traj_length=900,
    trail=200,
    fps=30,
    cmap_name="nipy_spectral",
    outfile="all_trajectories.mp4",
    dpi=150,
    interval=20,
    frameskip=1,  # <--- NEW: skip every n frames (1 = no skip)
    get_masked_occu=None,
):
    """
    Create one movie where each trajectory is displayed one after another
    with fading trails. The canvas resets for each trajectory.

    frameskip : int
        Keep only every `frameskip`-th frame from each trajectory
        while keeping FPS the same. This shortens the *video duration*.
    """

    # ------------------------------------------------------------
    # Determine number of trajectories (fix off-by-one)
    # ------------------------------------------------------------
    n_traj = int(pddata["num_traj"].to_numpy()[-1] - 1)

    # ------------------------------------------------------------
    # Extract all trajectories
    # ------------------------------------------------------------
    trajectories = []
    for i in range(n_traj):
        start = i * traj_length
        end = start + traj_length
        x = pddata["x"].to_numpy()[start:end]
        y = pddata["y"].to_numpy()[start:end]
        trajectories.append((x, y))

    # ------------------------------------------------------------
    # Figure, axes, background
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    tmp = occu_xy.T.copy()
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
    dummy_line = LineCollection([], linewidths=0, alpha=0)
    ax.add_collection(dummy_line)

    ax.set_xlim(xbound)
    ax.set_ylim(ybound)
    ax.set_aspect("equal")
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(blocked_colour)
        spine.set_linewidth(0.75)

    # ------------------------------------------------------------
    # LineCollection object
    # ------------------------------------------------------------
    linecoll = None

    # Precompute colors
    cmap = cm.get_cmap(cmap_name)
    color_table = cmap(np.linspace(0, 1, trail))

    # ------------------------------------------------------------
    # Frame mapping (with frameskip)
    # ------------------------------------------------------------
    # Original number of frames per trajectory
    frames_per_traj = traj_length

    # New effective number of frames after skipping
    frames_per_traj_eff = math.ceil(frames_per_traj / frameskip)

    # Total frames to animate
    total_frames = frames_per_traj_eff * n_traj

    def find_traj(global_frame):
        return global_frame // frames_per_traj_eff

    def local_frame(global_frame):
        # map back into original trajectory frame index
        return (global_frame % frames_per_traj_eff) * frameskip

    # ------------------------------------------------------------
    # Animation update function
    # ------------------------------------------------------------
    def update(frame):
        nonlocal linecoll

        traj_idx = find_traj(frame)
        local = local_frame(frame)

        x, y = trajectories[traj_idx]

        # Start of new trajectory
        if local == 0:
            if linecoll is not None:
                linecoll.remove()
                linecoll = None
            return (dummy_line,)  # <<< always return an Artist

        # Extract trail segment
        start_idx = max(0, local - trail)
        X = x[start_idx:local]
        Y = y[start_idx:local]

        if len(X) < 2:
            if linecoll is not None:
                linecoll.remove()
                linecoll = None
            return (dummy_line,)  # <<< must return artist

        # Remove old line
        if linecoll is not None:
            linecoll.remove()
            linecoll = None

        # Alpha fading
        fade = np.linspace(0.05, 1.0, len(X) - 1)

        base = np.array([0.0, 0.0, 0.0, 1.0])

        linecoll = colored_line_between_pts(
            X,
            Y,
            fade,
            ax,
            colors=[base],
            cmap=None,
            linewidth=2,
        )

        # Set RGBA colors manually
        segs = linecoll.get_segments()
        colors = np.tile(base, (len(segs), 1))
        colors[:, -1] = fade
        linecoll.set_color(colors)

        return (linecoll,)

    # ------------------------------------------------------------
    # Build animation
    # ------------------------------------------------------------
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=True)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    ani.save(outfile, fps=fps, dpi=dpi, writer="ffmpeg")

    return ani


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.collections import LineCollection


def animate_all_trajectories_with_neurons(
    pddata,
    occu_xy,
    sids,
    xbound,
    ybound,
    blocked_colour,
    traj_length=900,
    trail=200,
    fps=30,
    cmap_name="nipy_spectral",
    outfile="all_trajectories.mp4",
    dpi=150,
    interval=20,
    frameskip=1,
    get_masked_occu=None,
):
    """
    Create a movie where each trajectory is displayed one after another
    with fading trails and persistent neuron activity dots.

    Parameters
    ----------
    pddata : pandas DataFrame
        Must contain "x", "y", and "num_traj".
    sids : ndarray (t, n)
        Neuron activity: 0 = silent, >0 = active
    occu_xy : 2D occupancy grid for background.
    xbound, ybound : tuple/list
        Axis limits.
    blocked_colour : color
        Color of the axis border.
    traj_length : int
        Number of points per trajectory.
    trail : int
        Number of previous points to keep with fading.
    fps : int
        Frames per second for output video.
    cmap_name : str
        Colormap for trajectory fading.
    outfile : str
        Output filename (.mp4 or .gif).
    dpi : int
        Resolution.
    interval : int
        Delay between frames in ms (preview only).
    frameskip : int
        Only draw every nth frame to shorten the video.
    get_masked_occu : function
        Function that returns (masked_occu, cmap_obst).

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    # ============================================================
    # Extract number of trajectories
    # ============================================================
    n_traj = int(pddata["num_traj"].to_numpy()[-1] - 1)

    # ============================================================
    # 1. Extract all trajectories into a list
    # ============================================================
    trajectories = []
    for i in range(n_traj):
        start = i * traj_length
        end = start + traj_length
        x = pddata["x"].to_numpy()[start:end]
        y = pddata["y"].to_numpy()[start:end]
        trajectories.append((x, y))

    # ============================================================
    # 2. Build figure and axis
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    # ============================================================
    # 3. Prepare background occupancy map
    # ============================================================
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

    ax.set_xlim(xbound)
    ax.set_ylim(ybound)
    ax.set_aspect("equal")

    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(blocked_colour)
        spine.set_linewidth(0.75)

    # ============================================================
    # 4. Prepare neuron scatter
    # ============================================================
    neuron_scat = ax.scatter([], [], s=25, c=[], cmap=cmap_name, zorder=5, alpha=0.5)
    neuron_scat.set_clim(0, sids.shape[1])
    # color_idx =

    # ============================================================
    # 5. Prepare line collection variables
    # ============================================================
    linecoll = None

    # Create dummy line for consistent blitting
    dummy_line = LineCollection([], linewidths=0, alpha=0)
    ax.add_collection(dummy_line)

    # ============================================================
    # 6. Timeline mapping
    # ============================================================
    frames_per_traj = traj_length // frameskip
    total_frames = frames_per_traj * n_traj

    def find_traj(global_frame):
        return global_frame // frames_per_traj

    def local_frame(global_frame):
        return (global_frame % frames_per_traj) * frameskip

    # ============================================================
    # 7. Prepare color table for fading
    # ============================================================
    # cmap = cm.get_cmap(cmap_name)
    # color_table = cmap(np.linspace(0, 1, trail))

    # ============================================================
    # 8. Update function
    # ============================================================
    persistent_offsets = np.empty((0, 2))  # Will store all previously active neuron positions
    persistent_colors = np.array([])  # Optional: store corresponding colors

    def update(frame):
        nonlocal linecoll, persistent_offsets, persistent_colors
        imaging_neurons = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        imaging_neurons = np.array([1, 5, 6, 11, 15])

        traj_idx = find_traj(frame)
        local = local_frame(frame)
        x, y = trajectories[traj_idx]

        # Reset trail at start of trajectory
        if local == 0:
            if linecoll is not None:
                linecoll.remove()
                linecoll = None

        # Extract fading trail segment
        start_idx = max(0, local - trail)
        X = x[start_idx:local]
        Y = y[start_idx:local]

        # Update LineCollection trail
        if len(X) >= 2:
            if linecoll is not None:
                linecoll.remove()
                linecoll = None

            fade = np.linspace(0.05, 1.0, len(X) - 1)
            base = np.array([0.36, 0.12, 0.15, 1.0])  # white base color

            linecoll = colored_line_between_pts(
                X,
                Y,
                fade,
                ax,
                colors=[base],
                cmap=None,
                linewidth=2,
            )

            # Apply per-segment alpha
            segments = linecoll.get_segments()
            colors = np.tile(base, (len(segments), 1))
            colors[:, -1] = fade
            linecoll.set_color(colors)
        else:
            linecoll = None

        # -----------------------------
        # Update neuron scatter
        # -----------------------------
        global_frame_idx = traj_idx * traj_length + local
        # print(global_frame_idx, sids.shape[0])
        # Get currently active neurons
        active_idx = np.where(sids[global_frame_idx : global_frame_idx + frameskip, :] > 0)[0]
        draw_idx = []
        for i in active_idx:
            if i in imaging_neurons:
                draw_idx.append(i)
        # active_idx = [i if i in imaging_neurons else -1 for i in active_idx]
        # print(active_idx)

        if len(draw_idx) > 0:
            # All active neurons share the trajectory position at this frame
            x_pos = pddata["x"].to_numpy()[global_frame_idx]
            y_pos = pddata["y"].to_numpy()[global_frame_idx]

            new_offsets = np.column_stack([np.full(len(draw_idx), x_pos), np.full(len(draw_idx), y_pos)])

            # Accumulate positions and colors
            persistent_offsets = np.vstack([persistent_offsets, new_offsets])
            persistent_colors = np.hstack([persistent_colors, draw_idx])

        # Update scatter
        neuron_scat.set_offsets(persistent_offsets)
        neuron_scat.set_array(persistent_colors if len(persistent_colors) > 0 else np.array([]))

        # Return artists
        # artists = [neuron_scat]
        artists = []
        if linecoll is not None:
            artists.append(dummy_line)
        else:
            artists.append(dummy_line)
        return artists

    # ============================================================
    # 9. Build animation
    # ============================================================
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=True)

    # ============================================================
    # 10. Save the movie
    # ============================================================
    log.debug(f"Writing file {outfile}")
    ani.save(outfile, fps=fps, dpi=dpi, writer="ffmpeg")
    log.debug(f"All done!")

    return ani


hippo_n_feature = 16
length = 71
xygrain = 19
layernames = ["seq0", "seqall", "mlp0", "mlp2"]
displayname_dict = {"seq0": "DG", "seqall": "CA3", "mlp0": "Decoder 1", "mlp2": "Decoder 2"}

rootpath = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam"

experiment = "InternalRewardSeparateReward3DG_"
experiment_path = pathlib.Path(rootpath) / experiment  # / "telemetry"

experiment_subnames = sorted(get_folder_names(experiment_path))
log.debug(f"Experiment subnames for experiment {experiment}: {experiment_subnames}")

### COLOLR Data
blocked_colour = "#C7C7C7"


xbound = (100, 2000)
ybound = (100, 2000)

exp_id = 2
sub_id = 1


# for i in range(len(experiment_subnames)):
for i in range(1, 2):
    i = exp_id
    log.info(i)
    telemetry_path = experiment_path / experiment_subnames[i] / "telemetry"
    if os.path.exists(telemetry_path):
        # log.warn("Path exists!")
        epoch_names = sorted(get_folder_names(telemetry_path))
        log.debug(f"Epoch names for experiment {experiment_subnames[i]}: {epoch_names}")
        # epoch_names = epoch_names[3:]
        folders_containing_data = []
        for k in range(len(epoch_names)):
            folders_containing_data.append(str(telemetry_path / epoch_names[k]))
        # folders_containing_data = sorted(folders_containing_data)
        print(len(folders_containing_data))

        ## aggregating data
        fields_enjoys = []
        PFs_enjoys = []
        SI_enjoys = []
        percent_active = []
        percent_active_all = []
        pddata_enjoys = []
        sids_enjoys = []
        for analysispath in [folders_containing_data[sub_id]]:

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
            sids_enjoys.append(sids)
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
                if key != "occu" and key != "seq0":
                    PFs[key] = np.nansum(fields[key], -2) / np.nansum(fields["occu"][:, :, :, None], -2)
                if key == "seq0":
                    PFs[key] = np.nansum(fields[key], -2)
            PFs_enjoys.append(PFs)

        base_picture_path = "/work/classic/fr_js1764-sample_factory/pictures"
        file_name = experiment + "_" + experiment_subnames[i] + "_" + epoch_names[sub_id] + "movie.mp4"
        video_path = pathlib.Path(base_picture_path) / experiment / file_name
        # t = number of frames (global time steps), n = number of neurons
        t, n = pddata_enjoys[0].shape[0], 50  # e.g., 50 neurons

        # Sparse array: mostly zeros
        sids_test = np.zeros((t, n))

        # Randomly activate some neurons at random times
        np.random.seed(42)
        for _ in range(20000):  # activate 200 random events
            time_idx = np.random.randint(0, t)
            neuron_idx = np.random.randint(0, n)
            sids_test[time_idx, neuron_idx] = np.random.uniform(0.5, 1.0)

        log.debug(sids_test[500:550, :])
        ani = animate_all_trajectories_with_neurons(
            pddata_enjoys[0],
            np.nansum(fields_enjoys[0]["occu"][:, :, :], -1),
            sids_enjoys[0],
            xbound,
            ybound,
            blocked_colour,
            traj_length=900,
            trail=100,
            fps=30,
            outfile=video_path,
            dpi=150,
            frameskip=3,
            get_masked_occu=get_masked_occu,
        )
        log.debug(f"Animation Done")
        # plot_place_fields(PFs_enjoys[0]["seq0"], experiment, experiment_subnames[i], epoch_names[sub_id])
        # log.debug(np.where(sids_enjoys[0]>0))

    else:
        log.warning(f"PATH DOES NOT EXIST")
