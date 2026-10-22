"""
Statistical analysis of hippocampal / DG place cell selectivity across 20 runs (4 conditions x 5 seeds).

Analyzes:
1. Which place neurons (0-15) activate preferentially for Context 1 (Left) vs. Context 2 (Right).
2. The number and percentage of neurons in each category:
   - Left-selective (SI > 0.2)
   - Right-selective (SI < -0.2)
   - Both / Non-selective (|SI| <= 0.2)
   - Silent / Inactive
3. Aggregates mean ± SEM across seeds for each of the 4 conditions.
4. Generates publication-ready figures:
   - Stacked bar chart of neuron percentages across conditions
   - 16-neuron selectivity heatmap across all runs
   - Consistency summary table
"""

import argparse
import glob
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Set publication style
mpl.rcParams["font.sans-serif"] = "DejaVu Sans"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Condition display order & colors
CONDITION_ORDER = [
    "DG: Sigmoid",
    "DG: Concat",
    "Decoder + DG: Concat+Sigmoid",
    "Decoder",
]

CATEGORY_COLORS = {
    "Left Selective": "#2b5c8f",  # Deep blue
    "Right Selective": "#800020",  # Bordeaux / dark red
    "Both / Non-selective": "#2e7d32",  # Green
    "Silent / Inactive": "#cccccc",  # Neutral gray
}


# ==============================================================================
# RUN SPECIFICATIONS
# ==============================================================================
RUNS_CONFIG: List[Dict] = [
    # 1. DG: Sigmoid
    {
        "setup": "DG: Sigmoid",
        "seed": 1111,
        "expname": "02_encoderonly_norew_INSTR_see_1111_l.rat_2e-05_D.c.mod_sigmoid_20260907_155908_875328",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_",
    },
    {
        "setup": "DG: Sigmoid",
        "seed": 2222,
        "expname": "08_encoderonly_norew_INSTR_see_2222_l.rat_2e-05_D.c.mod_sigmoid_20260907_155919_680496",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_",
    },
    {
        "setup": "DG: Sigmoid",
        "seed": 3333,
        "expname": "14_encoderonly_norew_INSTR_see_3333_l.rat_2e-05_D.c.mod_sigmoid_20260907_160159_927031",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_",
    },
    {
        "setup": "DG: Sigmoid",
        "seed": 4444,
        "expname": "20_encoderonly_norew_INSTR_see_4444_l.rat_2e-05_D.c.mod_sigmoid_20260907_161320_772435",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_",
    },
    {
        "setup": "DG: Sigmoid",
        "seed": 5555,
        "expname": "26_encoderonly_norew_INSTR_see_5555_l.rat_2e-05_D.c.mod_sigmoid_20260907_161454_787947",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/encoderonly/run1_grid/train_dir/encoderonly_norew_INSTR/encoderonly_norew_INSTR_",
    },
    # 3. Decoder + DG: Concat+Sigmoid
    {
        "setup": "Decoder + DG: Concat+Sigmoid",
        "seed": 1111,
        "expname": "12_sigmoid_norew_INSTR_see_1111_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_094245_926033",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_",
    },
    {
        "setup": "Decoder + DG: Concat+Sigmoid",
        "seed": 2222,
        "expname": "30_sigmoid_norew_INSTR_see_2222_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_094306_102492",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_",
    },
    {
        "setup": "Decoder + DG: Concat+Sigmoid",
        "seed": 3333,
        "expname": "48_sigmoid_norew_INSTR_see_3333_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_141651_075658",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_",
    },
    {
        "setup": "Decoder + DG: Concat+Sigmoid",
        "seed": 4444,
        "expname": "66_sigmoid_norew_INSTR_see_4444_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260822_204500_870830",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_",
    },
    {
        "setup": "Decoder + DG: Concat+Sigmoid",
        "seed": 5555,
        "expname": "84_sigmoid_norew_INSTR_see_5555_n.i.coe_200_r.sca_0.1_l.rat_2e-05_20260823_155030_501136",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/sigmoid_run2_grid/train_dir/sigmoid_norew_INSTR/sigmoid_norew_INSTR_",
    },
    # 4. Decoder
    {
        "setup": "Decoder",
        "seed": 1111,
        "expname": "00_decoderonly_norew_INSTR_see_1111_l.rat_2e-05_20260907_161942_499591",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_",
    },
    {
        "setup": "Decoder",
        "seed": 2222,
        "expname": "02_decoderonly_norew_INSTR_see_2222_l.rat_2e-05_20260907_162304_426501",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_",
    },
    {
        "setup": "Decoder",
        "seed": 3333,
        "expname": "04_decoderonly_norew_INSTR_see_3333_l.rat_2e-05_20260907_175351_994268",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_",
    },
    {
        "setup": "Decoder",
        "seed": 4444,
        "expname": "06_decoderonly_norew_INSTR_see_4444_l.rat_2e-05_20260907_175351_994273",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_",
    },
    {
        "setup": "Decoder",
        "seed": 5555,
        "expname": "08_decoderonly_norew_INSTR_see_5555_l.rat_2e-05_20260907_183146_803859",
        "train_dir": "/work/classic/fr_ze12-data/ymaze_CTRL/norew_INSTR/new_modulate/decoderonly/run1/train_dir/decoderonly_norew_INSTR/decoderonly_norew_INSTR_",
    },
]


def resolve_telemetry_files(
    train_dir: str, expname: str, telemetry_base_dir: str = "/work/classic/fr_ze12-data2/telemetry_plots"
) -> Tuple[Optional[str], Optional[str]]:
    """Locates the latest pose_*.csv and activations_*.h5 files for a run."""
    # 1. Check in separate telemetry_base_dir
    candidate_dirs = [
        os.path.join(telemetry_base_dir, expname, "telemetry"),
        os.path.join(telemetry_base_dir, expname),
        os.path.join(train_dir, expname, "telemetry"),
        os.path.join(train_dir, expname),
    ]

    for d in candidate_dirs:
        if os.path.isdir(d):
            csvs = sorted(glob.glob(os.path.join(d, "pose_*.csv")))
            h5s = sorted(glob.glob(os.path.join(d, "activations_*.h5")))
            if csvs and h5s:
                return csvs[-1], h5s[-1]

    # 2. Search with prefix pattern (stripping timestamp)
    prefix = expname.split("_2026")[0] if "_2026" in expname else expname
    matches = (
        glob.glob(os.path.join(telemetry_base_dir, f"{prefix}*", "telemetry"))
        + glob.glob(os.path.join(telemetry_base_dir, f"{prefix}*"))
        + glob.glob(os.path.join(train_dir, f"{prefix}*", "telemetry"))
    )
    if not matches:
        parent_dir = os.path.dirname(train_dir.rstrip("/"))
        matches = glob.glob(os.path.join(parent_dir, "*", f"{prefix}*", "telemetry"))

    for m in matches:
        csvs = sorted(glob.glob(os.path.join(m, "pose_*.csv")))
        h5s = sorted(glob.glob(os.path.join(m, "activations_*.h5")))
        if csvs and h5s:
            return csvs[-1], h5s[-1]

    return None, None


# ==============================================================================
# CONTEXT EXTRACTION & SINGLE RUN ANALYSIS
# ==============================================================================
def extract_context_series(pddata: pd.DataFrame) -> pd.Series:
    """
    Extracts or infers context labels (Context 1 = Left, Context 2 = Right)
    for each frame in pddata. Handles CSVs with or without a 'context' column.
    """
    if "context" in pddata.columns:
        return pddata["context"]

    # 1. Check if context is embedded in the 'info' JSON column
    if "info" in pddata.columns and len(pddata) > 0:
        try:
            first_val = pddata["info"].iloc[0]
            first_dict = json.loads(first_val) if isinstance(first_val, str) else first_val
            if isinstance(first_dict, dict):
                for key in ["context", "INSTR", "instruction", "goal"]:
                    if key in first_dict:
                        return pddata["info"].apply(
                            lambda s: (
                                json.loads(s).get(key)
                                if isinstance(s, str)
                                else (s.get(key) if isinstance(s, dict) else 1)
                            )
                        )
        except Exception:
            pass

    # 2. Infer Left (1) vs Right (2) per trajectory from maze coordinates
    # In ymaze, trials are indexed by 'num_traj'. In each trial, the agent starts at the stem
    # and navigates into either the Left arm (lower x) or the Right arm (higher x).
    if "num_traj" in pddata.columns and "x" in pddata.columns and len(pddata) > 0:
        mid_x = (pddata["x"].min() + pddata["x"].max()) / 2.0
        traj_context = {}
        for traj_id, group in pddata.groupby("num_traj"):
            min_x = group["x"].min()
            max_x = group["x"].max()
            # If the trajectory moved further left from center than right:
            if (mid_x - min_x) > (max_x - mid_x):
                traj_context[traj_id] = 1  # Left
            else:
                traj_context[traj_id] = 2  # Right
        return pddata["num_traj"].map(traj_context).fillna(1).astype(int)

    # 3. Fallback if num_traj is somehow missing: split by mid_x directly
    if "x" in pddata.columns and len(pddata) > 0:
        mid_x = (pddata["x"].min() + pddata["x"].max()) / 2.0
        return (pddata["x"] > mid_x).astype(int) + 1

    return pd.Series(np.ones(len(pddata), dtype=int))


def analyze_single_run_selectivity(
    csv_path: str,
    h5_path: str,
    hippo_n_features: int = 16,
    stride: int = 71,
    si_threshold: float = 0.2,
    active_threshold: float = 1e-4,
    xbound: Tuple[float, float] = (100, 2000),
    ybound: Tuple[float, float] = (100, 2000),
    grain: int = 19,
) -> Dict:
    """
    Computes firing rates, spatial selectivity index (SI), and classifications
    for all 16 hippocampal features in a single run.
    """
    # 1. Load pose records
    pddata = pd.read_csv(csv_path)

    # 2. Load neural activations
    with h5py.File(h5_path, "r") as h5:
        if "core" in h5:
            core_act = h5["core"][...]
        else:
            k0 = list(h5.keys())[0]
            core_act = h5[k0][...]

    # Align lengths if needed
    min_len = min(len(pddata), len(core_act))
    if min_len == 0:
        return {
            "n_left": 0,
            "n_right": 0,
            "n_both": 0,
            "n_silent": hippo_n_features,
            "pct_left": 0.0,
            "pct_right": 0.0,
            "pct_both": 0.0,
            "pct_silent": 100.0,
            "left_neurons": [],
            "right_neurons": [],
            "both_neurons": [],
            "silent_neurons": list(range(hippo_n_features)),
            "si_values": np.zeros(hippo_n_features),
            "classifications": ["Silent / Inactive"] * hippo_n_features,
            "neuron_stats": [],
        }

    pddata = pddata.iloc[:min_len].copy()
    sids = core_act[:min_len, : (hippo_n_features * stride) : stride]
    if sids.ndim == 1:
        sids = sids[:, np.newaxis]

    # 3. Separate by context (Context 1 = Left, Context 2 = Right)
    context_series = extract_context_series(pddata)
    pddata["context"] = context_series

    unique_contexts = sorted(context_series.unique())
    if len(unique_contexts) >= 2:
        # Determine which context value corresponds to Left (lower average x) vs Right (higher average x)
        m_x0 = pddata.loc[context_series == unique_contexts[0], "x"].mean()
        m_x1 = pddata.loc[context_series == unique_contexts[1], "x"].mean()
        if m_x0 <= m_x1:
            c_left, c_right = unique_contexts[0], unique_contexts[1]
        else:
            c_left, c_right = unique_contexts[1], unique_contexts[0]
        mask1 = (context_series == c_left).values
        mask2 = (context_series == c_right).values
    elif len(unique_contexts) == 1:
        mask1 = (context_series == unique_contexts[0]).values
        mask2 = np.zeros(len(pddata), dtype=bool)
    else:
        mask1 = np.ones(len(pddata), dtype=bool)
        mask2 = np.zeros(len(pddata), dtype=bool)

    # 4. Spatial bins and occupancy
    bins = (np.linspace(*xbound, grain + 1), np.linspace(*ybound, grain + 1))
    if np.any(mask1):
        occu1 = np.histogramdd((pddata.loc[mask1, "x"], pddata.loc[mask1, "y"]), bins, density=False)[0]
    else:
        occu1 = np.zeros((grain, grain))

    if np.any(mask2):
        occu2 = np.histogramdd((pddata.loc[mask2, "x"], pddata.loc[mask2, "y"]), bins, density=False)[0]
    else:
        occu2 = np.zeros((grain, grain))

    # 5. Compute selectivity per neuron
    neuron_stats = []
    si_values = []
    classifications = []

    left_neurons = []
    right_neurons = []
    both_neurons = []
    silent_neurons = []

    for i in range(hippo_n_features):
        if np.any(mask1):
            act1 = np.histogramdd(
                (pddata.loc[mask1, "x"], pddata.loc[mask1, "y"]), bins, weights=sids[mask1, i], density=False
            )[0]
            rate_map1 = np.divide(act1, occu1, out=np.zeros_like(act1), where=occu1 > 0)
            r1_mean = float(np.mean(sids[mask1, i]))
            p1_peak = float(np.max(rate_map1))
        else:
            rate_map1 = np.zeros((grain, grain))
            r1_mean = 0.0
            p1_peak = 0.0

        if np.any(mask2):
            act2 = np.histogramdd(
                (pddata.loc[mask2, "x"], pddata.loc[mask2, "y"]), bins, weights=sids[mask2, i], density=False
            )[0]
            rate_map2 = np.divide(act2, occu2, out=np.zeros_like(act2), where=occu2 > 0)
            r2_mean = float(np.mean(sids[mask2, i]))
            p2_peak = float(np.max(rate_map2))
        else:
            rate_map2 = np.zeros((grain, grain))
            r2_mean = 0.0
            p2_peak = 0.0

        r_all_mean = float(np.mean(sids[:, i])) if len(sids) > 0 else 0.0

        # Selectivity Index based on mean firing rate
        denom = r1_mean + r2_mean + 1e-6
        si = (r1_mean - r2_mean) / denom
        si_values.append(si)

        # Classification
        is_active = (r_all_mean > active_threshold) or (max(p1_peak, p2_peak) > active_threshold)

        if not is_active:
            cat = "Silent / Inactive"
            silent_neurons.append(i)
        elif si >= si_threshold:
            cat = "Left Selective"
            left_neurons.append(i)
        elif si <= -si_threshold:
            cat = "Right Selective"
            right_neurons.append(i)
        else:
            cat = "Both / Non-selective"
            both_neurons.append(i)

        classifications.append(cat)
        neuron_stats.append(
            {
                "feature_idx": i,
                "r1_mean": r1_mean,
                "r2_mean": r2_mean,
                "p1_peak": p1_peak,
                "p2_peak": p2_peak,
                "SI": si,
                "category": cat,
            }
        )

    # Summary counts and percentages
    n_left = len(left_neurons)
    n_right = len(right_neurons)
    n_both = len(both_neurons)
    n_silent = len(silent_neurons)
    total_n = hippo_n_features

    pct_left = (n_left / total_n) * 100.0
    pct_right = (n_right / total_n) * 100.0
    pct_both = (n_both / total_n) * 100.0
    pct_silent = (n_silent / total_n) * 100.0

    return {
        "n_left": n_left,
        "n_right": n_right,
        "n_both": n_both,
        "n_silent": n_silent,
        "pct_left": pct_left,
        "pct_right": pct_right,
        "pct_both": pct_both,
        "pct_silent": pct_silent,
        "left_neurons": left_neurons,
        "right_neurons": right_neurons,
        "both_neurons": both_neurons,
        "silent_neurons": silent_neurons,
        "si_values": np.array(si_values),
        "classifications": classifications,
        "neuron_stats": neuron_stats,
    }


# ==============================================================================
# BATCH ANALYSIS OVER ALL 20 RUNS
# ==============================================================================
def run_full_statistical_analysis(
    runs: List[Dict] = RUNS_CONFIG,
    output_dir: str = "place_cell_analysis_results",
    si_threshold: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes full selectivity quantification across all runs, aggregates SEM across seeds,
    and saves publication figures and tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_records = []
    si_matrix = []  # (20, 16)
    row_labels = []

    print(f"\n{'='*75}\nRunning Place Cell Selectivity Analysis Across {len(runs)} Runs...\n{'='*75}")

    for idx, r_cfg in enumerate(runs):
        setup_name = r_cfg["setup"]
        seed = r_cfg["seed"]
        expname = r_cfg["expname"]
        train_dir = r_cfg["train_dir"]

        csv_p, h5_p = resolve_telemetry_files(train_dir, expname)

        if not csv_p or not h5_p:
            print(f"[{idx:02d}] MISSING TELEMETRY for {setup_name} (seed={seed})")
            continue

        print(f"[{idx:02d}] Processing {setup_name} (seed={seed})...")
        res = analyze_single_run_selectivity(csv_p, h5_p, si_threshold=si_threshold)
        print(
            f"      -> Left: {res['n_left']} ({res['pct_left']:.1f}%), Right: {res['n_right']} ({res['pct_right']:.1f}%), Both: {res['n_both']} ({res['pct_both']:.1f}%), Silent: {res['n_silent']} ({res['pct_silent']:.1f}%)"
        )

        run_records.append(
            {
                "Setup": setup_name,
                "Seed": seed,
                "Expname": expname,
                "Left_Count": res["n_left"],
                "Right_Count": res["n_right"],
                "Both_Count": res["n_both"],
                "Silent_Count": res["n_silent"],
                "Left_Pct": res["pct_left"],
                "Right_Pct": res["pct_right"],
                "Both_Pct": res["pct_both"],
                "Silent_Pct": res["pct_silent"],
                "Selective_Pct": res["pct_left"] + res["pct_right"],
                "Left_Neurons": str(res["left_neurons"]),
                "Right_Neurons": str(res["right_neurons"]),
                "Both_Neurons": str(res["both_neurons"]),
                "Silent_Neurons": str(res["silent_neurons"]),
            }
        )

        si_matrix.append(res["si_values"])
        row_labels.append(f"{setup_name} (seed {seed})")

    df_runs = pd.DataFrame(run_records)
    if df_runs.empty:
        print("[Notice] No telemetry files found on disk to analyze. Make sure to run generate_telemetry.py first.")
        return df_runs, pd.DataFrame()

    # Save per-run results
    runs_csv_path = os.path.join(output_dir, "per_run_selectivity.csv")
    df_runs.to_csv(runs_csv_path, index=False)
    print(f"\nSaved per-run results to {runs_csv_path}")

    # Aggregate across seeds (Mean ± SEM) per Setup
    summary_records = []
    for setup in CONDITION_ORDER:
        sub = df_runs[df_runs["Setup"] == setup]
        n_seeds = len(sub)
        if n_seeds == 0:
            continue

        def get_mean_sem(series):
            m = series.mean()
            sem = series.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
            return m, sem

        l_m, l_sem = get_mean_sem(sub["Left_Pct"])
        r_m, r_sem = get_mean_sem(sub["Right_Pct"])
        b_m, b_sem = get_mean_sem(sub["Both_Pct"])
        s_m, s_sem = get_mean_sem(sub["Silent_Pct"])
        tot_m, tot_sem = get_mean_sem(sub["Selective_Pct"])

        summary_records.append(
            {
                "Setup": setup,
                "Seeds": n_seeds,
                "Left_Pct_Mean": l_m,
                "Left_Pct_SEM": l_sem,
                "Right_Pct_Mean": r_m,
                "Right_Pct_SEM": r_sem,
                "Both_Pct_Mean": b_m,
                "Both_Pct_SEM": b_sem,
                "Silent_Pct_Mean": s_m,
                "Silent_Pct_SEM": s_sem,
                "Selective_Pct_Mean": tot_m,
                "Selective_Pct_SEM": tot_sem,
            }
        )

    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "summary_by_condition.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated condition summary to {summary_csv_path}")

    # ==============================================================================
    # PLOTTING
    # ==============================================================================
    plot_stacked_percentages(df_summary, output_dir)
    if si_matrix:
        plot_si_heatmap(np.array(si_matrix), row_labels, output_dir)

    return df_runs, df_summary


def plot_stacked_percentages(df_summary: pd.DataFrame, output_dir: str):
    """Generates horizontal stacked percentage bar plot across conditions."""
    if df_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 2.5), dpi=300)

    setups = df_summary["Setup"].tolist()
    y = np.arange(len(setups))
    height = 0.52

    left_means = df_summary["Left_Pct_Mean"].values
    right_means = df_summary["Right_Pct_Mean"].values
    both_means = df_summary["Both_Pct_Mean"].values
    silent_means = df_summary["Silent_Pct_Mean"].values

    left_sems = df_summary["Left_Pct_SEM"].values
    right_sems = df_summary["Right_Pct_SEM"].values

    # Horizontal Stacking
    p1 = ax.barh(
        y,
        left_means,
        height,
        label="Left Selective",
        color=CATEGORY_COLORS["Left Selective"],
        edgecolor="white",
        linewidth=0.8,
    )
    p2 = ax.barh(
        y,
        right_means,
        height,
        left=left_means,
        label="Right Selective",
        color=CATEGORY_COLORS["Right Selective"],
        edgecolor="white",
        linewidth=0.8,
    )
    p3 = ax.barh(
        y,
        both_means,
        height,
        left=left_means + right_means,
        label="Both / Non-selective",
        color=CATEGORY_COLORS["Both / Non-selective"],
        edgecolor="white",
        linewidth=0.8,
    )
    p4 = ax.barh(
        y,
        silent_means,
        height,
        left=left_means + right_means + both_means,
        label="Silent / Inactive",
        color=CATEGORY_COLORS["Silent / Inactive"],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title("Place Cell Selectivity by Condition", fontsize=11, pad=10, fontweight="medium")
    ax.set_xlabel("Neuron Proportion (%)", fontsize=9.5)
    ax.set_xlim(0, 102)
    ax.set_yticks(y)
    ax.set_yticklabels(setups, fontsize=8.5)
    ax.invert_yaxis()  # Top-down order matching CONDITION_ORDER
    ax.tick_params(axis="both", labelsize=8.5)

    # Style
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#222222")

    ax.legend(frameon=False, fontsize=8.0, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    svg_path = os.path.join(output_dir, "place_cell_selectivity_stacked.svg")
    pdf_path = os.path.join(output_dir, "place_cell_selectivity_stacked.pdf")
    png_path = os.path.join(output_dir, "place_cell_selectivity_stacked.png")

    save_kwargs = dict(facecolor="white", edgecolor="none", transparent=False, pad_inches=0.08, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", **save_kwargs)
    fig.savefig(pdf_path, dpi=300, **save_kwargs)
    fig.savefig(png_path, dpi=300, **save_kwargs)
    plt.close(fig)
    print(f"Saved horizontal stacked bar figure to {svg_path} (also {pdf_path} and {png_path})")


def plot_si_heatmap(si_matrix: np.ndarray, row_labels: List[str], output_dir: str):
    """Plots Selectivity Index heatmap across all runs and all 16 features."""
    fig, ax = plt.subplots(figsize=(6.5, max(4.0, len(row_labels) * 0.25)), dpi=300)

    # Diverging colormap: Blue for Left (SI > 0), Orange/Red for Right (SI < 0)
    im = ax.imshow(si_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_title("Context Selectivity Index (SI) across 16 Features", fontsize=10.5, pad=8)
    ax.set_xlabel("Hippocampal Feature Index (0 - 15)", fontsize=9.5)
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(np.arange(16), fontsize=8)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Selectivity Index (SI)\n[Right <- 0 -> Left]", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    svg_path = os.path.join(output_dir, "place_cell_si_heatmap.svg")
    pdf_path = os.path.join(output_dir, "place_cell_si_heatmap.pdf")
    png_path = os.path.join(output_dir, "place_cell_si_heatmap.png")

    save_kwargs = dict(facecolor="white", edgecolor="none", transparent=False, pad_inches=0.08, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", **save_kwargs)
    fig.savefig(pdf_path, dpi=300, **save_kwargs)
    fig.savefig(png_path, dpi=300, **save_kwargs)
    plt.close(fig)
    print(f"Saved SI heatmap to {svg_path} (also {pdf_path} and {png_path})")


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze place cell selectivity across 20 experimental runs.")
    parser.add_argument(
        "--output_dir", type=str, default="place_cell_analysis_results", help="Directory to save figures and CSVs."
    )
    parser.add_argument("--si_thresh", type=float, default=0.2, help="Threshold for selectivity index (default: 0.2).")
    args = parser.parse_args()

    run_full_statistical_analysis(output_dir=args.output_dir, si_threshold=args.si_thresh)


if __name__ == "__main__":
    main()
