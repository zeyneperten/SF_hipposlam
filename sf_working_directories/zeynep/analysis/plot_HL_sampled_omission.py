import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. MODEL DEFINITIONS & PALETTE
# ==============================================================================
MODELS = {
    'Xiao Baseline (Additive Q)': 'train_dir/xiao_pilot_seed7',
    'Multiplicative Q (tau=0.2)': 'train_dir/multiplicative_q_seed7',
    'Multiplicative PG': 'train_dir/multiplicative_pg_seed7',
    'MLP + PG': 'train_dir/mlp_pg_seed7_v2',
}

COLORS = {
    'Xiao Baseline (Additive Q)': '#7F7F7F',  # Gray
    'Multiplicative Q (tau=0.2)': '#1F77B4',  # Blue
    'Multiplicative PG': '#2CA02C',  # Green
    'MLP + PG': '#D62728',  # Red
}


# ==============================================================================
# 2. TRIAL-BY-TRIAL OMISSION LATENCY ANALYZER
# ==============================================================================
def compute_omission_switch_latency(trials_df: pd.DataFrame):
  """Computes how many trials / consecutive losses it takes to switch arms."""
  sessions = trials_df['session'].unique()
  session_mean_consec_losses = []
  all_consec_losses = []

  for sess in sessions:
    sdf = trials_df[trials_df['session'] == sess].sort_values('trial')
    rows = sdf.to_dict('records')

    consec_losses = 0
    current_arm = None
    sess_losses = []

    for r in rows:
      chosen = r['chosen_arm']
      rew = r['reward']

      if current_arm is None:
        current_arm = chosen
        consec_losses = 1 if rew == 0 else 0
        continue

      if chosen != current_arm:
        # Agent switched arms! Record how many consecutive misses triggered it
        if consec_losses > 0:
          sess_losses.append(consec_losses)
          all_consec_losses.append(consec_losses)
        current_arm = chosen
        consec_losses = 1 if rew == 0 else 0
      else:
        # Stayed on the same arm
        if rew == 0:
          consec_losses += 1
        else:
          # Win reset: a reward confirms the arm is good
          consec_losses = 0

    if sess_losses:
      session_mean_consec_losses.append(np.mean(sess_losses))

  mean_val = np.mean(all_consec_losses) if all_consec_losses else np.nan
  sem_val = (
      np.std(session_mean_consec_losses, ddof=1)
      / np.sqrt(len(session_mean_consec_losses))
      if len(session_mean_consec_losses) > 1
      else 0.0
  )

  return {
      'mean': mean_val,
      'sem': sem_val,
      'per_session': session_mean_consec_losses,
  }


# ==============================================================================
# 3. DATA EXTRACTION
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

session_accuracies = {}
omission_latencies = {}  # Stores consecutive omissions before switch

for name, path in MODELS.items():
  model_dir = Path(path)
  eval_file = model_dir / 'evaluation.json'
  trials_file = model_dir / 'sampled_trials.csv'

  if not eval_file.is_file():
    print(f'Skipping {name}: {eval_file} not found')
    continue

  with open(eval_file) as f:
    data = json.load(f)

  sampled = data.get('sampled', {})
  color = COLORS.get(name, 'black')
  clean_name = name.replace(' (Champion)', '').replace(' (Additive Q)', '')

  # -----------------------------------------------------------------
  # Panel 1: Post-Reversal Adaptation Curve
  # -----------------------------------------------------------------
  actual_curve = [
      item['high_arm_accuracy']
      for item in sampled.get('actual_switch_curve', [])
      if item.get('high_arm_accuracy') is not None
  ]
  if actual_curve:
    axes[0, 0].plot(
        range(len(actual_curve)),
        actual_curve,
        marker='o',
        linewidth=2.2,
        label=name,
        color=color,
        alpha=0.9,
    )

  # -----------------------------------------------------------------
  # Panel 2: Within-Block Evidence Accumulation
  # -----------------------------------------------------------------
  nominal_curve = [
      item['high_arm_accuracy']
      for item in sampled.get('nominal_block_curve', [])
      if item.get('high_arm_accuracy') is not None
  ]
  if nominal_curve:
    axes[0, 1].plot(
        range(len(nominal_curve)),
        nominal_curve,
        marker='s',
        linewidth=2.2,
        label=name,
        color=color,
        alpha=0.9,
    )

  # -----------------------------------------------------------------
  # Panel 3 Data: Omission Switch Latency (from sampled_trials.csv)
  # -----------------------------------------------------------------
  if trials_file.is_file():
    tdf = pd.read_csv(trials_file)
    omission_latencies[clean_name] = compute_omission_switch_latency(tdf)
  else:
    print(
        f'[Warning] {trials_file} not found. Panel 3 requires'
        ' sampled_trials.csv.'
    )

  # -----------------------------------------------------------------
  # Panel 4 Data: Per-Session Accuracy across 128 Held-Out Sessions
  # -----------------------------------------------------------------
  sessions = sampled.get('sessions', [])
  if sessions:
    session_accuracies[clean_name] = [
        s['high_arm_accuracy'] for s in sessions if 'high_arm_accuracy' in s
    ]

# ==============================================================================
# 4. PANEL 3: TRIALS TO SWITCH AFTER OMISSION (DETAILED IN-DEPTH PLOT)
# ==============================================================================
if omission_latencies:
  names = list(omission_latencies.keys())
  means = [omission_latencies[n]['mean'] for n in names]
  sems = [omission_latencies[n]['sem'] for n in names]
  x_pos = np.arange(len(names))
  bar_colors = [
      COLORS.get(k, '#333')
      for k in MODELS.keys()
      if any(clean_k in k for clean_k in names)
  ]

  bars = axes[1, 0].bar(
      x_pos,
      means,
      yerr=sems,
      capsize=4,
      width=0.45,
      color=bar_colors,
      alpha=0.85,
      edgecolor='black',
      error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
  )

  axes[1, 0].set_xticks(x_pos)
  axes[1, 0].set_xticklabels(names, rotation=20, ha='right', fontsize=10)
  axes[1, 0].set_ylabel('Consecutive Losses Before Switch', fontsize=11)
  axes[1, 0].set_title(
      'Trials Required to Switch After Omissions (Mean ± SEM)', fontsize=12
  )

  # Bayesian Benchmark (~2.3 consecutive misses in 85/15 task) & WS-LS Baseline (1.0)
  axes[1, 0].axhline(
      2.30,
      color='green',
      linestyle='--',
      alpha=0.8,
      linewidth=1.3,
      label='Ideal Bayesian Switch Threshold (~2.3)',
  )
  axes[1, 0].axhline(
      1.0,
      color='gray',
      linestyle=':',
      alpha=0.8,
      linewidth=1.2,
      label='WS-LS Baseline (1.0)',
  )

  axes[1, 0].set_ylim(0, 3.5)
  axes[1, 0].grid(axis='y', linestyle=':', alpha=0.6)
  axes[1, 0].legend(loc='upper right', fontsize=8.5)

  # Numeric value labels on top of each bar
  for bar, mean_val, sem_val in zip(bars, means, sems):
    axes[1, 0].text(
      bar.get_x() + bar.get_width() / 2,
      mean_val + sem_val + 0.08,
      f"{mean_val:.2f} trials",
      ha="center",
      va="bottom",
      fontsize=9.5,
      fontweight="bold",
    )

# ==============================================================================
# 5. PANEL 4: CROSS-SESSION RELIABILITY BOXPLOT
# ==============================================================================
if session_accuracies:
  labels = list(session_accuracies.keys())
  data_to_plot = [session_accuracies[l] for l in labels]

  box = axes[1, 1].boxplot(data_to_plot, patch_artist=True, labels=labels)
  for patch, key in zip(box['boxes'], MODELS.keys()):
    patch.set_facecolor(COLORS.get(key, '#ddd'))
    patch.set_alpha(0.7)

  axes[1, 1].set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
  axes[1, 1].set_ylabel('Sampled High-Arm Accuracy', fontsize=11)
  axes[1, 1].set_title(
      'Cross-Session Reliability (128 Held-Out Sessions)', fontsize=12
  )
  axes[1, 1].axhline(0.5, color='gray', linestyle=':', label='Chance (50%)')
  axes[1, 1].axhline(
      0.775,
      color='purple',
      linestyle='--',
      alpha=0.8,
      label='WS-LS Baseline (77.5%)',
  )
  axes[1, 1].grid(axis='y', linestyle=':', alpha=0.6)
  axes[1, 1].legend(loc='lower right', fontsize=8.5)

# Formatting Panels 1 & 2
axes[0, 0].set_title(
    'Post-Reversal Adaptation Curve (Sampled Policy)', fontsize=12
)
axes[0, 0].set_xlabel('Trials Since Actual Reversal', fontsize=11)
axes[0, 0].set_ylabel('P(High Arm Choice)', fontsize=11)
axes[0, 0].axhline(0.5, color='gray', linestyle=':', label='Chance (50%)')
axes[0, 0].set_ylim(0.2, 1.0)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].legend(loc='lower right', fontsize=8.5)

axes[0, 1].set_title(
    'Nominal Block Evidence Accumulation (Sampled Policy)', fontsize=12
)
axes[0, 1].set_xlabel('Trial in Block (0 to 14)', fontsize=11)
axes[0, 1].set_ylabel('P(High Arm Choice)', fontsize=11)
axes[0, 1].axhline(0.5, color='gray', linestyle=':', label='Chance (50%)')
axes[0, 1].set_ylim(0.2, 1.0)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].legend(loc='lower right', fontsize=8.5)

plt.tight_layout()
plt.savefig('sampled_policy_deep_dive_omission_latency.svg', dpi=300)
plt.savefig('sampled_policy_deep_dive_omission_latency.pdf', dpi=300)
plt.savefig('sampled_policy_deep_dive_omission_latency.png', dpi=300)
plt.show()
print('Saved updated figure to sampled_policy_deep_dive_omission_latency.png')