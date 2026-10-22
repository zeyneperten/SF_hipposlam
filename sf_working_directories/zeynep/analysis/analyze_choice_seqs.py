import json
import os
import pathlib

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")  # Prevents headless X11 GUI issues
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PUBLICATION STANDARD STYLING (figure_making_functions.ipynb)
# ==============================================================================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.labelsize"] = 7.5
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["lines.linewidth"] = 0.9
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["savefig.bbox"] = "tight"

# ==============================================================================
# 2. PATHS & EPISODE PARAMETERS
# ==============================================================================
# csv_path = (
#    "/work/classic/fr_ze12-data2/outputs/telemetry_plots/08_encoderonly_norew_INSTR_see_2222_l.rat_2e-05_D.c.mod_sigmoid/telemetry/pose_20261013_171929.csv"
# )
# output_dir = "/work/classic/fr_ze12-data2/ymaze_CTRL/08_encoderonly_norew_INSTR_see_2222_l.rat_2e-05_D.c.mod_sigmoid"

csv_path = "/work/classic/fr_ze12-data2/outputs/telemetry_plots/42_noINSTR_see_5555_r.sca_1.0_l.rat_2e-05/telemetry/pose_20261021_160031.csv"
output_dir = "/work/classic/fr_ze12-data2/ymaze_CTRL/42_noINSTR_see_5555_r.sca_1.0_l.rat_2e-05"

pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

print("Starting choice streak analysis...", flush=True)
print(f"Loading from : {csv_path}", flush=True)
print(f"Saving to    : {output_dir}\n", flush=True)

BLOCK_SIZE = 15
MAX_TRIALS = 75

# ==============================================================================
# 3. FAST PARSE TELEMETRY
# ==============================================================================
df = pd.read_csv(csv_path)
print(f"Total raw steps loaded: {len(df):,}", flush=True)

trials = []
trial_counter = 0

info_vals = df["info"].tolist() if "info" in df.columns else []
frame_vals = df["frame"].tolist() if "frame" in df.columns else list(range(len(df)))
ctx_col = df["context"].tolist() if "context" in df.columns else None
instr_col = df["instr"].tolist() if "instr" in df.columns else None

for idx, info_val in enumerate(info_vals):
    if not isinstance(info_val, str) or "{" not in info_val:
        continue

    try:
        info = json.loads(info_val)
    except Exception:
        continue

    hi_hit = bool(info.get("highrew_hit", False))
    hi_miss = bool(info.get("highrew_miss", False))
    lo_hit = bool(info.get("lowrew_hit", False))
    lo_miss = bool(info.get("lowrew_miss", False))

    if hi_hit or hi_miss or lo_hit or lo_miss:
        trial_counter += 1
        chose_high = int(hi_hit or hi_miss)
        rewarded = int(hi_hit or lo_hit)

        ctx = 1
        if ctx_col is not None and not pd.isna(ctx_col[idx]):
            ctx = int(ctx_col[idx])
        elif instr_col is not None and not pd.isna(instr_col[idx]):
            ctx = int(instr_col[idx])
        else:
            for k in ["context", "inst_block", "INSTR", "instr"]:
                val = info.get(k)
                if val is not None:
                    try:
                        ctx = int(val)
                        break
                    except (ValueError, TypeError):
                        pass

        trials.append(
            {
                "trial_idx": trial_counter,
                "frame": int(frame_vals[idx]),
                "context": ctx,
                "chose_high_arm": chose_high,
                "rewarded": rewarded,
                "outcome": (
                    "High-Rew Hit (+10)"
                    if hi_hit
                    else "High-Rew Miss (0)" if hi_miss else "Low-Rew Hit (+10)" if lo_hit else "Low-Rew Miss (0)"
                ),
            }
        )

trial_df = pd.DataFrame(trials)
total_extracted = len(trial_df)
print(f"Extracted completed choice trials: {total_extracted}", flush=True)

# Clip to exactly 75 trials (Episode 1)
if total_extracted > MAX_TRIALS:
    print(f"[Info] Clipping to first {MAX_TRIALS} trials (Episode 1: 5 blocks x 15 trials).", flush=True)
    trial_df = trial_df.iloc[:MAX_TRIALS].copy()

n_trials = len(trial_df)
num_blocks = n_trials // BLOCK_SIZE

trial_df["block"] = ((trial_df["trial_idx"] - 1) // BLOCK_SIZE) + 1
trial_df["trial_in_block"] = ((trial_df["trial_idx"] - 1) % BLOCK_SIZE) + 1

trials_csv_path = os.path.join(output_dir, "trial_by_trial_choices.csv")
trial_df.to_csv(trials_csv_path, index=False)
print(f"Saved trial table: {trials_csv_path}\n", flush=True)

# ==============================================================================
# 4. 15-TRIAL BLOCK & STREAK CALCULATIONS
# ==============================================================================
choices = trial_df["chose_high_arm"].to_numpy()
rewarded = trial_df["rewarded"].to_numpy()
raw_contexts = trial_df["context"].to_numpy()

overall_acc = np.mean(choices) * 100


def compute_streaks(c_arr):
    streaks, curr = [], 0
    for c in c_arr:
        if c == 1:
            curr += 1
        else:
            if curr > 0:
                streaks.append(curr)
            curr = 0
    if curr > 0:
        streaks.append(curr)
    return np.array(streaks) if streaks else np.array([0])


block_info = []
for b in range(num_blocks):
    b_start = b * BLOCK_SIZE
    b_end = (b + 1) * BLOCK_SIZE
    b_choices = choices[b_start:b_end]
    b_ctx = int(pd.Series(raw_contexts[b_start:b_end]).mode()[0])
    b_acc = np.mean(b_choices) * 100
    b_streaks = compute_streaks(b_choices)

    block_info.append(
        {
            "block_num": b + 1,
            "start_trial": b_start + 1,
            "end_trial": b_end,
            "context": b_ctx,
            "accuracy": b_acc,
            "max_streak": np.max(b_streaks),
        }
    )

transitions = []
print(
    f"{'Block':<7} | {'Trials':<11} | {'Context':<8} | {'Status':<16} | {'Accuracy':<10} | {'Max Streak'}", flush=True
)
print("-" * 72, flush=True)

for i, b in enumerate(block_info):
    if i == 0:
        status = "START"
    else:
        prev_ctx = block_info[i - 1]["context"]
        curr_ctx = b["context"]
        flipped = prev_ctx != curr_ctx
        status = f"INVERTED ({prev_ctx}->{curr_ctx})" if flipped else f"UNCHANGED ({curr_ctx}->{curr_ctx})"
        transitions.append(
            {
                "boundary_trial": b["start_trial"] - 0.5,
                "trial_num": b["start_trial"] - 1,
                "flipped": flipped,
                "from_ctx": prev_ctx,
                "to_ctx": curr_ctx,
            }
        )

    print(
        f"Block {b['block_num']:<2} | {b['start_trial']:>2}-{b['end_trial']:<6} | Context {b['context']} | {status:<16} | {b['accuracy']:>6.1f}%    | {b['max_streak']} trials",
        flush=True,
    )

print("-" * 72, flush=True)
print(f"Overall Accuracy: {overall_acc:.1f}%\n", flush=True)

# ==============================================================================
# 5. COMPACT PUBLICATION FIGURE
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(3, 3), dpi=300)

trial_indices = np.arange(1, n_trials + 1)
ctx_colors = {1: "#3498db", 2: "#e67e22"}
xticks = sorted(list(set([1] + [b * BLOCK_SIZE for b in range(1, num_blocks + 1)] + [n_trials])))


def add_decorations(ax):
    for b in block_info:
        c = ctx_colors.get(b["context"], "#cccccc")
        ax.axvspan(b["start_trial"] - 0.5, b["end_trial"] + 0.5, color=c, alpha=0.06)

    for trans in transitions:
        line_col = "#c0392b" if trans["flipped"] else "#7f8c8d"
        line_style = "--" if trans["flipped"] else ":"
        ax.axvline(trans["boundary_trial"], color=line_col, linestyle=line_style, lw=0.75, zorder=2)


# Panel a: Trial raster
ax1 = axes[0]
add_decorations(ax1)

# Display block accuracy directly above each block
for b in block_info:
    mid_x = (b["start_trial"] + b["end_trial"]) / 2
    ax1.text(mid_x, 0.30, f"{b['accuracy']:.0f}%", ha="center", va="bottom", fontsize=4.5, color="#555555")

# ax1.step(trial_indices, choices, where="mid", color="#555555", lw=0.7, alpha=0.4, zorder=1)
rew_mask = rewarded == 1
unrew_mask = rewarded == 0
s1 = ax1.scatter(trial_indices[rew_mask], choices[rew_mask], color="#27ae60", s=2, label="Reward +10", zorder=3)
s2 = ax1.scatter(
    trial_indices[unrew_mask],
    choices[unrew_mask],
    facecolors="none",
    edgecolors="#c0392b",
    s=2,
    lw=0.8,
    label="No Reward",
    zorder=3,
)

# Highlight color patches for the legend
patch_c1 = mpatches.Patch(facecolor="#3498db", alpha=0.35, edgecolor="none", label="Block R")
patch_c2 = mpatches.Patch(facecolor="#e67e22", alpha=0.35, edgecolor="none", label="Block L")

ax1.set_xticks(xticks)
ax1.set_xticklabels([])
ax1.set_yticks([0, 1])
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks)
ax1.set_yticklabels(["Low arm", "High arm"])
ax1.set_xlabel("Trial")
ax1.set_xlim(0.5, n_trials + 0.5)

# Left: title; Right: overall accuracy
ax1.set_title("a. Trial choices", loc="left", fontsize=8.5, pad=2)
# ax1.text(n_trials + 0.5, 1.3, f"Overall accuracy: {overall_acc:.1f}%", fontsize=7, color="#555555")

ax1.legend(
    handles=[patch_c1, patch_c2, s1, s2],
    frameon=False,
    bbox_to_anchor=[1.02, 1],
    loc="upper left",
    handletextpad=0.3,
    borderaxespad=0.2,
    markerscale=2.5,
)
ax1.spines[["top", "right"]].set_visible(False)

# Panel b: Cumulative choices
ax2 = axes[1]
add_decorations(ax2)
ax2.plot(trial_indices, np.cumsum(choices), color="#2980b9", lw=1.1, label=f"High arm \n(Accuracy: {overall_acc:.0f}%)")
ax2.plot(trial_indices, np.cumsum(1 - choices), color="#7f8c8d", lw=1.1, linestyle="--", label="Low arm")
ax2.set_xticks(xticks)
ax2.set_xlim(0.5, n_trials + 0.5)
ax2.set_xlabel("Trial")
ax2.set_ylabel("Choices")
ax2.set_title("b. Cumulative choices", loc="left", fontsize=8.5, pad=2)
ax2.legend(
    frameon=False, bbox_to_anchor=[1.02, 1], loc="upper left", handletextpad=0.2, borderaxespad=0.2, fontsize=7.5
)
ax2.spines[["top", "right"]].set_visible(False)

# Panel c: Consecutive streaks histogram
ax3 = axes[2]
streaks = compute_streaks(choices)
max_s = max(int(np.max(streaks)), 1)
bins = np.arange(0.5, max_s + 1.5, 1)
ax3.hist(streaks, bins=bins, color="#27ae60", edgecolor="none", rwidth=0.75)
med = np.median(streaks)
ax3.axvline(med, color="#e67e22", linestyle="--", lw=0.9, label=f"Median ({int(med)})")
ax3.set_xlabel("Sequence length")
ax3.set_ylabel("Count")
ax3.set_title("c. Length distribution", loc="left", fontsize=8.5, pad=2)
ax3.legend(frameon=False, bbox_to_anchor=[1.02, 1], loc="upper left", handletextpad=0.2, borderaxespad=0.2)
ax3.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=0.4, h_pad=0.7)

save_kwargs = dict(dpi=300, transparent=True, bbox_inches="tight")
png_path = os.path.join(output_dir, "choice_streaks_analysis.png")
svg_path = os.path.join(output_dir, "choice_streaks_analysis.svg")
pdf_path = os.path.join(output_dir, "choice_streaks_analysis.pdf")

plt.savefig(png_path, **save_kwargs)
plt.savefig(svg_path, format="svg", **save_kwargs)
plt.savefig(pdf_path, **save_kwargs)
plt.close(fig)

print(f"Saved publication figures to:\n  {png_path}\n  {svg_path}\n  {pdf_path}", flush=True)
