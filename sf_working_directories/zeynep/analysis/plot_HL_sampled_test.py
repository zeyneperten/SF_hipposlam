import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Models to compare
MODELS = {
    "Xiao Baseline (Additive Q)": "train_dir/xiao_pilot_seed7",
    "Multiplicative Q (tau=0.2)": "train_dir/multiplicative_q_seed7",
    "Multiplicative PG": "train_dir/multiplicative_pg_seed7",
    "MLP + PG": "train_dir/mlp_pg_seed7_v2",
}

# Color palette
COLORS = {
    "Xiao Baseline (Additive Q)": "#7F7F7F",  # Gray
    "Multiplicative Q (tau=0.2)": "#1F77B4",  # Blue
    "Multiplicative PG": "#2CA02C",  # Green
    "MLP + PG": "#D62728",  # Red
}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

session_accuracies = {}
omission_rates = {}
entropy_values = {}

for name, path in MODELS.items():
    eval_file = Path(path) / "evaluation.json"
    if not eval_file.is_file():
        print(f"Skipping {name}: {eval_file} not found")
        continue

    with open(eval_file) as f:
        data = json.load(f)

    sampled = data.get("sampled", {})
    color = COLORS.get(name, "black")

    # -------------------------------------------------------------
    # Panel 1: Post-Reversal Adaptation Curve (Sampled)
    # -------------------------------------------------------------
    actual_curve = [
        item["high_arm_accuracy"]
        for item in sampled.get("actual_switch_curve", [])
        if item.get("high_arm_accuracy") is not None
    ]
    if actual_curve:
        axes[0, 0].plot(
            range(len(actual_curve)), actual_curve, marker="o", linewidth=2.2, label=name, color=color, alpha=0.9
        )

    # -------------------------------------------------------------
    # Panel 2: Within-Block Evidence Accumulation (Sampled)
    # -------------------------------------------------------------
    nominal_curve = [
        item["high_arm_accuracy"]
        for item in sampled.get("nominal_block_curve", [])
        if item.get("high_arm_accuracy") is not None
    ]
    if nominal_curve:
        axes[0, 1].plot(
            range(len(nominal_curve)), nominal_curve, marker="s", linewidth=2.2, label=name, color=color, alpha=0.9
        )

    # -------------------------------------------------------------
    # Data Collection for Panels 3 & 4
    # -------------------------------------------------------------
    clean_name = name.replace(" (Champion)", "").replace(" (Additive Q)", "")

    # Omission switch rate: P(Switch | No Reward on same high arm)
    omission_rate = sampled.get("arm_change_after_omission_same_context")
    if omission_rate is not None:
        omission_rates[clean_name] = omission_rate

    # Entropy of decisions
    entropy = sampled.get("mode_entropy")
    if entropy is not None:
        entropy_values[clean_name] = entropy

    # Per-session accuracies across the 128 held-out sessions
    sessions = sampled.get("sessions", [])
    if sessions:
        session_accuracies[clean_name] = [s["high_arm_accuracy"] for s in sessions if "high_arm_accuracy" in s]

# -----------------------------------------------------------------
# Panel 3: Omission Sensitivity (P(Switch after Unrewarded Good Arm))
# -----------------------------------------------------------------
if omission_rates:
    names = list(omission_rates.keys())
    rates = list(omission_rates.values())
    x_pos = np.arange(len(names))
    bar_colors = [COLORS.get(k, "#333") for k in MODELS.keys() if any(n in k for n in names)]

    bars = axes[1, 0].bar(x_pos, rates, width=0.45, color=bar_colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    axes[1, 0].set_ylabel("P(Switch Arm | Omission)", fontsize=11)
    axes[1, 0].set_title("Omission Response (False-Switch Sensitivity)", fontsize=12)
    axes[1, 0].axhline(
        0.20, color="gray", linestyle="--", alpha=0.7, label="Ideal Bayesian Evidence Threshold (~0.20)"
    )
    axes[1, 0].set_ylim(0, 0.6)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right")

    # Add numeric labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2, yval + 0.015, f"{yval:.2%}", ha="center", va="bottom", fontsize=9
        )

# -----------------------------------------------------------------
# Panel 4: Distribution across 128 Held-Out Sessions (Reliability)
# -----------------------------------------------------------------
if session_accuracies:
    labels = list(session_accuracies.keys())
    data_to_plot = [session_accuracies[l] for l in labels]

    box = axes[1, 1].boxplot(data_to_plot, patch_artist=True, labels=labels)
    for patch, key in zip(box["boxes"], MODELS.keys()):
        patch.set_facecolor(COLORS.get(key, "#ddd"))
        patch.set_alpha(0.7)

    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    axes[1, 1].set_ylabel("Sampled High-Arm Accuracy", fontsize=11)
    axes[1, 1].set_title("Cross-Session Reliability (128 Held-Out Sessions)", fontsize=12)
    axes[1, 1].axhline(0.5, color="gray", linestyle=":", label="Chance (50%)")
    axes[1, 1].axhline(0.775, color="purple", linestyle="--", alpha=0.8, label="WS-LS Baseline (77.5%)")
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="lower right")

# Formatting Panels 1 & 2
axes[0, 0].set_title("Post-Reversal Adaptation Curve (Sampled Policy)", fontsize=12)
axes[0, 0].set_xlabel("Trials Since Actual Reversal", fontsize=11)
axes[0, 0].set_ylabel("P(High Arm Choice)", fontsize=11)
axes[0, 0].axhline(0.5, color="gray", linestyle=":", label="Chance (50%)")
axes[0, 0].set_ylim(0.2, 1.0)
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend(loc="lower right")

axes[0, 1].set_title("Nominal Block Evidence Accumulation (Sampled Policy)", fontsize=12)
axes[0, 1].set_xlabel("Trial in Block (0 to 14)", fontsize=11)
axes[0, 1].set_ylabel("P(High Arm Choice)", fontsize=11)
axes[0, 1].axhline(0.5, color="gray", linestyle=":", label="Chance (50%)")
axes[0, 1].set_ylim(0.2, 1.0)
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("sampled_policy_deep_dive.svg", dpi=300)
plt.savefig("sampled_policy_deep_dive.pdf", dpi=300)
print("Saved Sampled Policy Deep-Dive to sampled_policy_deep_dive.png")