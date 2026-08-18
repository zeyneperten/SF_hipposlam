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


def occu_entropy(occu_trials):
    norm_occu = occu_trials / (np.nansum(np.nansum(occu_trials, axis=-1), axis=-1)[:, np.newaxis, np.newaxis])
    entropy_bits_occu = -np.nansum(np.nansum(norm_occu * np.log(norm_occu), axis=-1), axis=-1)
    return entropy_bits_occu


def gini(x):
    """Compute Gini coefficient of a nonnegative 1D array."""
    x = np.ravel(x).astype(float)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.nancumsum(x)
    return (n + 1 - 2 * np.nansum(cumx) / cumx[-1]) / n


def mean_gini_per_neuron(A):
    """Compute the mean Gini coefficient across f neurons."""
    f = A.shape[2]
    ginis = [gini(A[..., i]) for i in range(f)]
    return ginis


def mean_entropy_per_neuron(A):
    """
    Compute the mean *normalized* Shannon entropy across f neurons.
    Normalized so that uniform activity = 1, perfectly localized = 0.
    """
    n_bins = A.shape[0] * A.shape[1]
    f = A.shape[2]
    entropies = []
    for i in range(f):
        p = A[..., i].ravel().astype(float)
        p_sum = np.nansum(p)
        if p_sum == 0:
            entropies.append(0.0)
            continue
        p /= p_sum
        p = p[p > 0]
        H = -np.nansum(p * np.log(p))  # entropy in nats
        H_max = np.log(n_bins)  # maximum possible entropy
        entropies.append(H / H_max)  # normalized to [0, 1]
    return entropies


def gini_of_combined_activity(A):
    """Normalize each neuron map, sum across neurons, then compute Gini."""
    normed = np.zeros_like(A, dtype=float)
    for i in range(A.shape[2]):
        s = np.nansum(A[..., i])
        if s > 0:
            normed[..., i] = A[..., i] / s
    combined = np.nansum(normed, axis=2)
    return gini(combined)


def participation_ratio(C):
    """Compute the participation ratio of an NxN correlation (or covariance) matrix."""
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.linalg.eigvalsh(C)  # use eigvalsh for symmetric matrices
    return (np.sum(eigvals) ** 2) / np.sum(eigvals**2)


hippo_n_feature = 16
length = 71
xygrain = 19
layernames = ["seq0", "seqall", "mlp0", "mlp2"]
displayname_dict = {"seq0": "DG", "seqall": "CA3", "mlp0": "Decoder 1", "mlp2": "Decoder 2"}

rootpath = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam"

experiment = "InternalRewardSeparateReward3DG_"
experiment_path = pathlib.Path(rootpath) / experiment  # / "telemetry"

experiment_subnames = get_folder_names(experiment_path, sort_folders=True)
log.debug(f"Experiment subnames for experiment {experiment}: {experiment_subnames}")

### COLOLR Data
blocked_colour = "#C7C7C7"


xbound = (100, 2000)
ybound = (100, 2000)


traj_length = 900
num_traj = int(np.floor(50000 / traj_length))


traj_entropy = np.zeros((len(experiment_subnames), num_traj))
pf_pr = np.zeros(len(experiment_subnames))
pf_gini = np.zeros((len(experiment_subnames), 16))
pf_entropy = np.zeros((len(experiment_subnames), 16))
pf_pop_gini = np.zeros(len(experiment_subnames))
mean_firing_rate = np.zeros(len(experiment_subnames))
rms_firing_rate = np.zeros(len(experiment_subnames))
percent_active = np.zeros(len(experiment_subnames))
percent_active_all = np.zeros(len(experiment_subnames))


for i in range(len(experiment_subnames)):
    log.info(i)
    telemetry_path = experiment_path / experiment_subnames[i] / "telemetry"
    if os.path.exists(telemetry_path):
        # log.warn("Path exists!")
        epoch_names = [get_folder_names(telemetry_path)[-1]]
        log.debug(f"Epoch names for experiment {experiment_subnames[i]}: {epoch_names}. Length: {len(epoch_names)}")
        # epoch_names = epoch_names[3:]
        folders_containing_data = []
        for k in range(len(epoch_names)):
            print(k)
            folders_containing_data.append(str(telemetry_path / epoch_names[k]))
        print(len(folders_containing_data))
        log.debug(folders_containing_data)

        ## aggregating data
        fields_enjoys = []
        PFs_enjoys = []
        SI_enjoys = []
        tmp_percent_active = []
        tmp_percent_active_all = []
        pddata_enjoys = []
        tmp_mean_firing_rate = []
        tmp_rms_firing_rate = []
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
            grain = 4
            occu_xya = np.histogramdd(
                (pddata["x"], pddata["y"], pddata["rot_pi"]),
                (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1), np.linspace(-np.pi, np.pi, 13)),
                density=False,
            )[0]

            tmp_act = activations

            # grain=19
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
            tmp_percent_active.append(perc)
            tmp_percent_active_all.append((sids.sum(1) > 0).mean())
            print(tmp_percent_active)
            print(tmp_percent_active_all)

            # tmp_mean_firing_rate.append(activations['core'])

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
        # trajectory entropy
        percent_active[i] = tmp_percent_active[-1].mean()
        percent_active_all[i] = tmp_percent_active_all[-1]
        grain_traj_entropy = 2
        occu_xya_tmp = np.zeros((num_traj, grain_traj_entropy, grain_traj_entropy))
        for j in range(num_traj):
            occu_xya_tmp[j] = np.histogramdd(
                (
                    pddata_enjoys[0]["x"][j * traj_length : (j + 1) * traj_length - 1],
                    pddata_enjoys[0]["y"][j * traj_length : (j + 1) * traj_length - 1],
                ),
                (np.linspace(*xbound, grain_traj_entropy + 1), np.linspace(*ybound, grain_traj_entropy + 1)),
                density=False,
            )[0]
        # traj entropy
        traj_entropy[i, :] = occu_entropy(occu_xya_tmp)
        # traj_entropy.append(occu_entropy(occu_xya_tmp))

        # participation ratio
        corr = pd.DataFrame(fields_enjoys[0]["seq0"].reshape(-1, 16)).corr()
        pf_pr[i] = participation_ratio(corr)
        # pf_pr.append(participation_ratio(corr))

        A = PFs_enjoys[0]["seq0"]
        # pf gini
        pf_gini[i, :] = mean_gini_per_neuron(A)
        # pf_gini.append(mean_gini_per_neuron(A))
        # pf entropy
        pf_entropy[i, :] = mean_entropy_per_neuron(A)
        # pf_entropy.append(mean_entropy_per_neuron(A))
        # pf population gini
        pf_pop_gini[i] = gini_of_combined_activity(A)
        # pf_pop_gini.append(gini_of_combined_activity(A))
    else:
        print("PATH DOES NOT EXIST!")

print(f"Trajectory entropy: {list(traj_entropy)}")
print(f"Population ratio: {list(pf_pr)}")
print(f"place field GINI: {list(pf_gini)}")
print(f"place field Entropy: {list(pf_entropy)}")
print(f"population GINI: {list(pf_pop_gini)}")
print(f"Percent Active {percent_active}")
print(f"Percent Active All {percent_active_all}")
