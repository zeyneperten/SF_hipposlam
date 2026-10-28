# Standalone high-level RNN bandit

This setup asks whether the existing high-level RNN can infer changing reward
contingencies when every selected mode reliably executes one arm choice. It
removes vision, DG, CA3, the motor decoder, navigation, PPO, PBT, and the simulator.
It reuses `HighLevelContextRNN_Learner`, the full core's trial-history replay,
and the existing Q-regression or policy-gradient loss. PyTorch and NumPy are
the only runtime dependencies; run commands from the SF_hipposlam repository root.

## Run the reference condition

```bash
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/seed7 --seed 7
```

The output directory must be new. Defaults train 500 session batches of 16
parallel, independent 75-trial sessions: **600,000 completed choices and 9,500
optimizer updates**. Adam updates every four completed trials per environment,
with a shorter final update. Each outcome appears once as a loss target;
past outcomes also supply recurrent history. This is synchronous trial-level
optimization, not an emulation of Sample Factory's frame-level update frequency.

Defaults match the transfer launcher and `ymaze_80_final.lua` at base revision
`769af6734e7b1b91b9979117cc5ebfd3903a4d29`:

| Component | Default |
| --- | --- |
| Modes / hidden units | 2 / 16, existing tanh RNNCell |
| Recurrent input | Previous mode one-hot and binary reward |
| Replay gradient horizon | Last 8 completed trials |
| Loss | Selected-mode immediate-reward squared error, `hl_is_policy=False` |
| Optimizer | Adam, learning rate 0.0002, epsilon 0.000001, gradient clipping 1 |
| Q sampling | Softmax, temperature 1 |
| Task | 80% versus 20% reward; right arm initially high |
| Schedule | 15 trials per nominal block; 0.66 probability of reversing at a boundary |
| Session | 5 nominal blocks; memory resets after 75 outcomes |

The task is a hidden-state bandit. The model never receives block identity,
boundary flags, trial count, current better arm, or reward probabilities.
Those fields are diagnostic metadata only. A nominal boundary can leave the
contingency unchanged. Reward uses the Lua task's complementary winning-arm
lottery for the default 80/20 pair; the manager's target is 0/1, independent of
the maze's +10 game score and low-level `reward_scale`.

Mode 0 executes left and mode 1 right by default. `--mode-to-arm 1 0` reverses
that convention. This is a perfect actuator assumption for the diagnostic;
it neither supervises a worker nor establishes learned mode meanings in the
full system. There are no failures to initiate, reach an endpoint, or return.

## Controls and evaluation

Use a separate output directory for each run. Change one factor at a time:

```bash
# Basic optimization control: stationary deterministic rewards.
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/stationary --switch-probability 0 --p-high 1 --p-low 0

# Switching without reward noise; retain the stochastic boundary rule.
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/deterministic --p-high 1 --p-low 0

# Existing policy-gradient alternative, same RNN and observations.
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/policy --loss policy

# Frozen checkpoint evaluation, optionally changing only Q sampling temperature.
python -m sf_working_directories.zeynep.bandit.train \
  --output train_dir/hl_bandit/eval_tau025 \
  --evaluate-checkpoint train_dir/hl_bandit/seed7/manager.pt --eval-tau 0.25
```

`--loss policy` uses the full system's existing batch-mean reward baseline and
0.05 entropy coefficient. Its logits do not use `--tau`; `--eval-tau` is rejected
for policy checkpoints. No alternative critic or supervised block target is added.

Every run evaluates **sampled** and **greedy** policies in separate closed-loop
sessions, along with greedy memory reset, random, win-stay/lose-shift, and an
oracle better-arm reference. The oracle alone reads the hidden contingency.
Memory reset restores the same initial-trigger state before every choice,
removing both recent outcome input and accumulated recurrent history. It is a
memory-use diagnostic, not a separately trained memoryless baseline.

At correct Q-values 0.8 and 0.2, temperature 1 yields only
$1/(1+\exp(-0.6))\approx0.646$ better-arm sampling even with perfect current
context knowledge. Its expected reward is about 0.588. This is a calibrated-Q
reference, not a hard bound on an imperfect model's logits, and does not mean
the hidden context can actually be known before observing post-switch evidence.
Greedy accuracy and the sampled-versus-greedy gap therefore answer different questions.

Validation uses held-out sessions at fixed checkpoints, with seed `seed+1000000`.
The final frozen test uses `--test-seed` (default 2000007), independently of
training and validation. Final results always use the last checkpoint; no best
test checkpoint is selected. Evaluation preserves the training RNG and weights.
Policies share environmental random streams within an evaluation. Their choices
can generate different observed rewards.

`generalization.json` additionally evaluates half/double nominal block lengths
(7 and 30 by default) with randomized initial contingency. These are deliberate
schedule changes, not full-task equivalence claims. For separate noise tests,
evaluate the same checkpoint with different `--p-high` and `--p-low`.

Useful outputs:

- `config.json`: exact arguments, observation/actuator contract, source revision,
  dirty status and library versions.
- `learning.csv` / `learning.jsonl`: completed choices, updates, loss, gradient
  norm, training accuracy/reward and wall time.
- `validation.json`: sampled and greedy learning curves at saved checkpoints.
- `evaluation.json`: final metrics, per-session summaries, curves indexed by
  trials since an **actual reversal**, and a separate nominal-block curve.
- `*_trials.csv`: frozen-test choices, rewards, scores/probabilities and hidden
  evaluator metadata, enough to recompute adaptation and omission responses.
- `generalization.json`, `summary.json`, `manager.pt`, and intermediate manager
  checkpoints. Checkpoints contain weights and compatibility metadata only.

Stable accuracy means the second half of each nominal block. Switch curves omit
the initial block and censor a curve at the next actual switch. Missing switch
bins contain `null` and a zero count. Per-session measurements are not independent
training seeds. Repeat seeds before making a learning or reliability claim.

## Compatibility with full runs

The full core and this runner call the same `TrialHistory` helper. The extraction
preserves history packing and `core.hl_learner.*` state-dictionary names, including
the full core's optional representation inputs. Initialization reproduces the
first trigger's zero-mode/zero-reward RNN update, which is not a training target.
History is saved before appending the current outcome, so a reward cannot leak
into its own predicted value. The final completed bandit trial is credited before
reset; the maze's endpoint/return timing is intentionally absent.

The standalone setting requires `hl_representations=False`. A representation-
enabled checkpoint is rejected rather than silently filling decoder inputs with
zeros. Compatibility means the architecture, state layout and supported weights
are shared; it does not imply representation-enabled experiments are equivalent.

To initialize from a full high-level checkpoint, place its saved `config.json`
beside the run's `checkpoint_p0` folder and use `--init-checkpoint PATH` with
matching `--hidden-size`, `--history-len`, `--loss` and `--tau`. Alternatively,
`--evaluate-checkpoint PATH` builds the model from checkpoint metadata. Both
`core.hl_learner.*` and `actor_core.hl_learner.*` are accepted. The loader requires
configuration because tensor shapes alone cannot identify Q versus policy semantics.
It extracts manager weights only; optimizer, recurrent state and training progress
are not resumed. Full-checkpoint mode meanings remain unverified.

To initialize the manager in a **new** full maze run, add this to its existing
launcher, keeping a matching architecture and retaining that launcher's computed
full `rnn_size`:

```bash
--hl_init_checkpoint=/absolute/path/to/manager.pt \
--hl_K=2 --hl_d_H=16 --hl_history_len=8 \
--hl_is_policy=False --hl_tau=1.0 --hl_representations=False
```

It composes with `--oracle_init_checkpoint`: controller weights load first,
then manager weights. The target run's existing checkpoint takes precedence
when resuming. All non-manager tensors are preserved by manager initialization.
The controller must empirically implement the saved mode-to-arm mapping; matching
tensor shapes does not establish behavioral transfer. Validate forced modes before
interpreting a transferred manager's reward.

## Tests

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_high_level_bandit_environment.py \
  tests/test_high_level_trial_history.py \
  tests/test_high_level_bandit_checkpoints.py \
  tests/test_high_level_bandit_training.py
```

Environment/training tests require pytest in addition to NumPy/PyTorch. Tests
that instantiate the full wrapper require the existing Sample Factory Python
dependencies, but no running simulator. Coverage includes event timing,
outcome/action attribution, recurrent replay/gradient parity, old checkpoint
keys, final-trial accounting, seed isolation and manager-only transfer.
