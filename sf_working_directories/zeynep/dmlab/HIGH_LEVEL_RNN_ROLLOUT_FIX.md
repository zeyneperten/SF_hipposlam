# High-level RNN rollout and trial-history fix

This note explains the changes to `exp_ymaze_HLRNN.py` for Zeynep. The high-level
model still acts from its recurrent hidden state $h$. The added trial records are
used **only by the learner** to train recurrent updates across 64-frame rollout
boundaries; the Q head does not read them during acting.

## What was wrong

1. Sample Factory carries `rnn_states` between rollouts, but backpropagation
   stops at `recurrence=64`. A reward update near the end of one rollout could
   have no later high-level loss in the same training chunk, so that update
   received no useful gradient from the next trial.
2. The actor's sampled mode was not saved as a trajectory field. During learner
   replay, the high-level module sampled again. The reconstructed decoder input
   could differ from the mode used for the recorded low-level action, and a
   later reward in the chunk could be assigned to the replayed mode.
3. Sampling was tied to `torch.is_grad_enabled()`. Sample Factory's inference
   workers collect training rollouts without gradients, so this made the actor
   choose greedily during data collection.
4. The experiment's name said `Q_grid`, but `--hl_is_policy=True` selected its
   policy-gradient loss. This run now explicitly uses the intended Q regression.

A negative `hl_loss` in that earlier policy run is possible without numerical
error: the policy objective subtracts $0.05$ times the mode entropy. With four
uniform modes and zero advantage, it is $-0.05\log 4 \approx -0.0693$. A finite
Q-regression loss is a masked mean of squared errors and cannot be negative. If
this experiment reports a negative high-level loss after switching to
`--hl_is_policy=False`, check the effective launch arguments and run directory.

## Data and timing

At a center outcome event, the old mode $z_{k-1}$ earned the latched reward
$r_{k-1}$. The actor records the pair $(z_{k-1},r_{k-1})$, updates $h$, samples
$z_k$, and passes $z_k$ to the low-level decoder. The first center event has no
previously chosen mode, so it does not enter the completed-trial history or the
high-level loss.

The actor's recurrent state now contains, in order:

```text
base_core_state | high_level_h | last N completed trial records | current_z
```

Each record contains the hidden state **before** that reward update, the mode,
the reward, and a validity flag. Records shift only at completed-trial events.
They persist across 64-frame rollouts and are cleared when Sample Factory
reports a real environment `done`. The actor's Q head still receives only
`high_level_h`; the record fields cannot directly tell it which arm to choose.

For each outcome in learner replay, the learner starts from the oldest saved
pre-event hidden state, unrolls the RNN through up to `hl_history_len` earlier
trial outcomes, and predicts the value of the **actor's** mode for the current
reward. This gives the RNN-cell parameters a gradient through earlier reward
updates even when those events occurred before the current 64-frame rollout.
The starting hidden state is detached, so the trial history is a bounded
truncated-backpropagation window. No second PPO implementation is involved.

## Code changes

| File | Change |
| --- | --- |
| `sample_factory/algo/utils/shared_buffers.py` | Allocate `hl_z` for the high-level core only. |
| `custom_actor_critic.py` | Save the post-decision mode from `new_rnn_states` as `hl_z` alongside the actor's actions; preserve the configured mode sampling rule for value-only bootstrap calls. |
| `sample_factory/algo/learning/learner.py` | Pass recorded `hl_z` and training-only trial records into the packed recurrent learner input. |
| `custom_core.py` | Carry the bounded trial history, unroll it at outcome events, use recorded modes during replay, reject an incorrect configured state size, and summarize valid outcome count and raw trial reward scale. |
| `custom_highlevelRNN.py` | Separate reward-state updates from mode sampling; sample or choose greedily using an explicit setting rather than autograd state. |
| `custom_params.py` | Add `hl_history_len`, `hl_deterministic`, and `hl_tau`. |
| `exp_ymaze_HLRNN.py` | Set eight historical trials, update `rnn_size` from 1166 to 1342, select Q regression, and make the existing temperature of 1 explicit. |

The learner also passes its `valids` flag through the packed sequence. The
high-level event mask now excludes invalid samples, matching the ordinary PPO
losses. Computing the high-level loss inside the core is functional here: the
learner reads `last_hl_loss` after the core forward pass, adds it to the total
loss, and calls `backward()` on that total. The mutable attribute is a coupling
to the current one-core-forward-per-minibatch flow; if that flow changes, return
the auxiliary loss explicitly from the forward pass instead.

For this configuration, eight records use $8(16+4+2)=176$ state elements.
`hl_deterministic=False` is the training default. Set it to `True` only when
greedy high-level evaluation is intended; actor inference mode alone no longer
changes exploration. Value-only bootstrap uses the same mode-sampling rule as
acting. A greedy bootstrap with a sampling actor would bias the value target
at trial boundaries; one sampled mode is an unbiased but noisy estimate of
the mode-averaged next value. Other runs
using the high-level core must update their
`rnn_size` to account for `hl_history_len` or the core will raise a size error.

## Validation and limits

`tests/test_high_level_rollout.py` checks sampling under `no_grad`, temperature
configuration, history continuity across separate forward calls, a nonzero
recurrent gradient through an earlier trial, and decoder replay under a
recorded mode. These are unit
checks. Run them with `python -m pytest -q tests/test_high_level_rollout.py`.
An end-to-end DeepMind Lab run is still needed to inspect event timing,
reward alignment, mode use, and learning curves. The configured level
`ymaze_vol5_INSTR_HL` is not checked into this branch, so its Lua termination
behavior cannot be verified here. The Python wrapper only reports `done` when
DeepMind Lab stops running.

The new state layout is incompatible with old high-level checkpoints. Start a
new experiment rather than resuming a 1166-element state. Saved pre-event
hidden states were produced by actor parameters at collection time; if policy
lag is large, reconstructed histories can differ from those states. Keep the
window short, inspect policy lag, and compare against a `hl_history_len=1`
control. The Q loss still uses only the chosen mode and immediate completed
trial reward. It does not implement multi-trial return or off-policy
correction.

## Training throughput and reward scale probe (2026-10-05)

The launch uses 75 trials per environment episode, four policies, 32 total
environments, `batch_size=2048`, `num_batches_per_epoch=2`, `num_epochs=1`,
`rollout=recurrence=64`, learning rate $2\times 10^{-4}$, and gradient clipping
at 1. Each completed trial gives one high-level supervised Q target. Assuming
workers are evenly distributed, one policy collects from about eight
environments. If a trial averages $F$ agent steps, a 2048-step minibatch has
about $2048/F$ high-level targets; $F$ must be measured from a real run.
Seventy-five trials cap the targets in **one episode**, not over training: the
weights are shared across episodes and environments. The Q loss averages over
outcome events, so sparse events do not automatically shrink its magnitude,
but a minibatch with zero valid outcomes gives no high-level update.

`--reward_scale=0.1` applies to Sample Factory's PPO reward stream. The Q
target and RNN reward input come separately from the unscaled
`prev_trial_reward` observation. The Python observation space declares that
value in $[0,1]$, but only a real run can confirm what the Lua level emits.
If it is $0/1$, the Q target scale is reasonable; if it is much larger, the
high-level squared loss and shared gradient clipping can dominate the PPO
update. Log the actual outcome reward range before changing either scale.

The current high-level sampling temperature is $\tau=1$. The new `--hl_tau`
setting permits a controlled sweep without changing this default. With one
mode valued at 1 and three valued at 0, even a perfect Q head samples the best
mode with probability $e^{1/\tau}/(e^{1/\tau}+3)\approx 0.475$ at $\tau=1$.
If values are only 0.1 apart, that probability is about 0.269. Greedy
evaluation is different. Lowering temperature early can also prevent rarely
sampled modes from being discovered, so it needs a controlled exploration
comparison rather than a silent default change.

`tests/probe_high_level_training.py` is a reproducible standalone check of the
actual high-level RNN cell, eight-event history reconstruction, Q loss, Adam
learning rate, and gradient clipping. It models eight parallel 75-trial
environments per policy and one optimizer update after each four trials per
environment. It excludes DeepMind Lab, frame timing, PPO, the decoder, actor
lag, and reward noise, so its episode count is **not** a prediction for Y-maze.
Run from the repository root, for example:

```bash
PYTHONPATH=. python tests/probe_high_level_training.py --task=stationary --best-mode=2 --tau=1 --episodes=20
PYTHONPATH=. python tests/probe_high_level_training.py --task=stationary --best-mode=2 --tau=0.25 --episodes=20
PYTHONPATH=. python tests/probe_high_level_training.py --task=latent --tau=0.25 --episodes=300
```

With seed 7 and $0/1$ rewards, the stationary mode-2 task reached sampled
success 0.14 after the first 75-trial cohort and 0.43 by cohort 20 at
$\tau=1$, despite Q loss falling to 0.0025. At $\tau=0.25$, it reached 0.06
after the first cohort and 0.96 by cohort 20. In the harder task with the
rewarding mode switching between 0 and 1 each episode, sampled success was
0.50 at cohort 100 and 0.94 at cohort 200 for $\tau=0.25$. These runs show
that low Q loss alone does not establish useful sampled choices, and that one
75-trial episode is too little evidence for a learning-speed judgment.

On this CPU with one PyTorch thread, one 32-sample history reconstruction plus
backward pass took about 0.21 ms with one event versus 0.73 ms with eight
events. This isolates the extra high-level computation; it is not an
end-to-end environment throughput benchmark. For the real run, track valid
outcome count per minibatch, the raw `prev_trial_reward` distribution, Q loss,
mode selection entropy, and both sampled and greedy trial success against
environment steps. The learner now exposes `hl/valid_outcomes`,
`hl/trial_reward_mean`, and `hl/trial_reward_abs_max` in its sampled training
summaries. These measurements determine whether more data, a changed
temperature schedule, or a different reward normalization is warranted.
