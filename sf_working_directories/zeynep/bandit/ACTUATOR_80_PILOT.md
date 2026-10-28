# Standalone HL RNN bandit task: 80% actuator pilot

28 October 2026. This exploratory pilot compares perfect execution with an
actuator that independently chooses the wrong arm on 20% of completed trials.
The RNN observes its commanded mode and binary reward, without execution feedback.

## Matched experiment

Source: `32a95b2fe009b08b490300b63cce190d17584cc5`, clean working tree.
Each condition trained fresh seeds 7, 17 and 27 for 600,000 completed choices
and 9,500 optimizer updates. All other settings were the runbook defaults:
Q loss, temperature 1, 16 hidden units, eight-trial replay, Adam learning rate
0.0002, 80/20 reward, 15-trial blocks, five blocks per session, switch
probability 0.66 and right-high initialization. Validation ran every 100 session
batches; final testing used the last checkpoint, without test-based selection.

Each checkpoint was evaluated under both actuator conditions, frozen, on
128 sessions of 75 trials with test seed 2000007 (320 actual reversals).
Task lotteries and actuator draws were paired across conditions; action and
reward histories could differ because evaluation was closed-loop. The three
perfect-actuator retrainings reproduced every original pilot model tensor
bit-for-bit. PyTorch 2.9.1 CPU was used with one thread per process.

## Results

Means across the three training seeds, using **greedy** decisions:

| Training actuator | Test actuator | Commanded better arm | Executed better arm | Reward rate |
| --- | --- | ---: | ---: | ---: |
| 100% | 100% | 77.09% | 77.09% | 66.05% |
| 100% | 80% | 71.66% | 63.08% | 57.97% |
| 80% | 100% | 62.58% | 62.58% | 57.60% |
| 80% | 80% | 61.44% | 56.89% | 54.38% |

Individual seeds trained and tested at 80% actuator success:

| Seed | Greedy commanded better arm | Greedy executed better arm | Greedy reward | Sampled executed better arm |
| --- | ---: | ---: | ---: | ---: |
| 7 | 61.47% | 56.97% | 55.01% | 51.45% |
| 17 | 58.44% | 55.19% | 52.56% | 51.71% |
| 27 | 64.41% | 58.51% | 55.57% | 51.59% |

At 80% execution reliability, greedy memory reset scored 58.44% commanded and
55.19% executed accuracy. Win-stay/lose-shift scored 66.29% commanded and 60.25%
executed accuracy (56.67% reward), exceeding all three noisy-trained RNNs.
Random execution accuracy was 50.03%. The hidden-context oracle commanded the
better arm on every trial, with 80.60% executed accuracy and 67.81% reward;
its theoretical expectations are 80% and 68%, respectively. These baselines
share the held-out task streams; their repeated values are not independent seeds.

Under 80% training and testing, sampled executed accuracy averaged 51.58% and
reward 50.94% at temperature 1. The calibrated-Q reference is about 55.3%
executed accuracy and 53.2% reward if current context were known perfectly;
it is not a bound on arbitrary learned scores.

The noisy-trained models did not show reliable post-switch recovery at this
budget: mean greedy commanded accuracy was 49.5% on the fifth post-reversal
choice and 55.3% on the fifteenth. Perfect-trained models tested with noise
reached 81.4% by the fifteenth choice. The noisy-trained seed 17's greedy
policy always commanded right, matching the memory-reset baseline. A higher
overall score need not indicate repeated switch tracking: seed 7 started with
right commands but commanded left in every session from trial 20 onward.
With seven-trial blocks and randomized initial contingency, noisy-trained
greedy commanded accuracy was only 48.5–49.2%. A higher
accuracy immediately after a reversal can result from poor pre-switch inference,
so early switch-curve values alone should not be read as faster adaptation.

The cross-condition results suggest a training deficit at this budget beyond
the unavoidable 20% execution errors. They do not establish permanent inability
to learn block switches or explain the full transfer experiment. With execution
unobserved, the noisy task is equivalent from the manager's perspective to
command rewards 68/32 rather than 80/20: weaker evidence and a smaller Q-value
gap are expected. Three seeds on one held-out environment set are a pilot,
not a population-level reliability claim.

## Reproduce and inspect

From the repository root, use a fresh output directory for each run:

```bash
for rate in 1.0 0.8; do
  for seed in 7 17 27; do
    python -m sf_working_directories.zeynep.bandit.train \
      --output train_dir/hl_bandit/actuator${rate}_seed${seed} \
      --seed "$seed" --actuator-success-rate "$rate" \
      --session-batches 500 --num-envs 16 --eval-sessions 128 --eval-every 100
  done
done

# Frozen cross-condition example: perfect-trained model, noisy execution.
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/seed7_perfect_to_noisy --seed 7 \
  --evaluate-checkpoint train_dir/hl_bandit/actuator1.0_seed7/manager.pt \
  --actuator-success-rate 0.8 --eval-sessions 128
```

Local raw outputs and checkpoints are saved at
`/home/xiaoxiong/SFgit/SF_hipposlam_hl_bandit/train_dir/hl_bandit_actuator80/`,
with directories `train100_seed{7,17,27}`, `train80_seed{7,17,27}`, and frozen
cross-evaluations suffixed `_eval80` or `_eval100`. The local
`pilot_summary.json` records checkpoint SHA-256 digests and per-seed policy
metrics. Configurations retain their original temporary output paths and clean
source revision. Raw runs are ignored by Git; this result note is published.

Validation: 107 standalone/history/checkpoint tests passed, plus eight existing
rollout tests and two initialization/resume tests. The known pre-existing
rollout test with a stale unscaled-mode expectation remains excluded. Tests
cover wrong-arm reward attribution, paired RNG streams, command-only recurrent
feedback and replay credit, WSLS information parity, actuator-aware oracle
regret, Q/policy CLI round trips, and checkpoint compatibility across actuator
settings. No full simulator training was run for this pilot.
