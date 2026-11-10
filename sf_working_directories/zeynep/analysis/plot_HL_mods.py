import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "Xiao Baseline (Additive Q)": "train_dir/xiao_pilot_seed7",
    "Multiplicative Q (tau=0.2)": "train_dir/multiplicative_q_seed7",
    "Multiplicative PG": "train_dir/multiplicative_pg_seed7",
    "MLP + PG (Best)": "train_dir/mlp_pg_seed7_v2",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for name, path in MODELS.items():
    eval_file = Path(path) / "evaluation.json"
    if not eval_file.is_file():
        print(f"Skipping {name}: {eval_file} not found")
        continue
    with open(eval_file) as f:
        data = json.load(f)

    # 1. Post-reversal switch curve (Sampled)
    sampled_curve = [
        item["high_arm_accuracy"]
        for item in data.get("sampled", {}).get("actual_switch_curve", [])
        if item.get("high_arm_accuracy") is not None
    ]
    if sampled_curve:
        axes[0, 0].plot(range(len(sampled_curve)), sampled_curve, marker="o", label=name)

    # 2. Post-reversal switch curve (Greedy)
    greedy_curve = [
        item["high_arm_accuracy"]
        for item in data.get("greedy", {}).get("actual_switch_curve", [])
        if item.get("high_arm_accuracy") is not None
    ]
    if greedy_curve:
        axes[0, 1].plot(range(len(greedy_curve)), greedy_curve, marker="s", label=name)

    # 3. Nominal block progression (Greedy)
    nominal_curve = [
        item["high_arm_accuracy"]
        for item in data.get("greedy", {}).get("nominal_block_curve", [])
        if item.get("high_arm_accuracy") is not None
    ]
    if nominal_curve:
        axes[1, 0].plot(range(len(nominal_curve)), nominal_curve, marker="^", label=name)

# 4. Bar chart: Overall High-Arm Accuracy (Sampled vs Greedy)
model_names = []
sampled_accs = []
greedy_accs = []

for name, path in MODELS.items():
    eval_file = Path(path) / "evaluation.json"
    if not eval_file.is_file():
        continue
    with open(eval_file) as f:
        data = json.load(f)
    clean_name = name.replace(" (Best)", "").replace(" (Additive Q)", "")
    model_names.append(clean_name)
    sampled_accs.append(data["sampled"]["high_arm_accuracy"])
    greedy_accs.append(data["greedy"]["high_arm_accuracy"])

x = np.arange(len(model_names))
width = 0.35

if len(model_names) > 0:
    axes[1, 1].bar(x - width / 2, sampled_accs, width, label="Sampled", color="#4C72B0")
    axes[1, 1].bar(x + width / 2, greedy_accs, width, label="Greedy", color="#DD8452")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=20, ha="right")
    axes[1, 1].set_ylabel("High Arm Accuracy")
    axes[1, 1].set_title("Overall Accuracy: Sampled vs Greedy Gap")
    axes[1, 1].set_ylim([0.45, 0.95])
    axes[1, 1].axhline(0.775, color="gray", linestyle="--", label="WS-LS (77.5%)")
    axes[1, 1].legend()

# Formatting
axes[0, 0].set_title("Actual Switch Curve (Sampled Policy)")
axes[0, 0].set_xlabel("Trials Since Actual Reversal")
axes[0, 0].set_ylabel("P(High Arm Choice)")
axes[0, 0].axhline(0.5, color="gray", linestyle=":")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title("Actual Switch Curve (Greedy Policy)")
axes[0, 1].set_xlabel("Trials Since Actual Reversal")
axes[0, 1].set_ylabel("P(High Arm Choice)")
axes[0, 1].axhline(0.5, color="gray", linestyle=":")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_title("Nominal Block Progression (Greedy Policy)")
axes[1, 0].set_xlabel("Trial in Block (0-14)")
axes[1, 0].set_ylabel("P(High Arm Choice)")
axes[1, 0].axhline(0.5, color="gray", linestyle=":")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reversal_curves_comparison.png", dpi=300)
print("Saved comparison plot to reversal_curves_comparison.png")