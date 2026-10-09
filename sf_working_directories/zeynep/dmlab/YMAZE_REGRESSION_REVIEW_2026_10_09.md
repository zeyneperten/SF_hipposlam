# Y-maze high-level regression review

- **Date:** 2026-10-09
- **Reviewed branch:** `zeyneperten/SF_hipposlam:ymaze` at `8515197`.

The earlier rollout and diversity changes were merged in PR #1. This review
checked the merged code and the two subsequent commits. The current
`exp_ymaze_HLRNN.py` launch has the diversity reward commented out, so that
reward cannot explain results from this exact configuration.

## Confirmed defects corrected here

1. **Value bootstrap chose a different high-level policy.** The actor sampled
   a mode with `hl_deterministic=False`, while `values_only=True` forced the
   best-scoring mode. At an outcome observation the critic therefore estimated
   the next state under a greedy mode even though the next trial's actor would
   sample. The wrapper now preserves the configured sampling rule. A small
   probe with two equally likely modes produced mode 1 on $50\%$ of actor
   calls but $0\%$ of bootstrap calls before the correction.
2. **Chance guesses earned a positive diversity reward.** The classifier
   reward clipped each sample's negative evidence to zero. In a probe where
   independent random guesses were $49.5\%$ accurate, the mean reward was
   $49.3\%$ of its maximum. The bounded evidence now retains its sign; wrong
   predictions give a small negative term, and chance-level guesses have
   approximately zero mean. For `--hl_diversity_reward_coef=0.01`, each
   trial-end term is bounded by $\pm 0.01$.
3. **Oracle mode modulation received the wrong slice.** The high-level core
   appends `hl_K` values even when it selects the mode from an oracle block.
   The additive and FiLM decoder factory instead reserved only two values
   when `oracle_context=True`. With `hl_K=4`, oracle modes 0 and 1 had zeros
   in those last two positions, so the intended modulation path received no
   mode signal. The decoder now reserves `hl_K` values in both oracle and
   learned runs. This changes additive/FiLM oracle decoder parameter shapes;
   start a fresh run for those configurations.

## Checks and remaining limits

The diversity tests, high-level rollout checks, and new decoder width check
pass in a synthetic Python environment. Python compilation and a whitespace
check pass. A complete actor–critic or DeepMind Lab rollout was not run:
this machine lacks `torchvision` and the configured `ymaze_vol5_*` Lua levels
are absent from the repository.

Two additional observations need a separate decision or dataset check:

- The current high-level launch sets `--max_grad_norm=0.0`, and the latest
  learner commit removed the explicit finite-gradient guard. A nonfinite
  gradient would now reach `optimizer.step()`. This is a confirmed code path,
  but whether it occurs in a real run is unknown.
- `hl/reward_achieved_mode{k}` is assembled from the mode at the **last frame**
  of the learner minibatch and that frame's `prev_trial_reward`. At an outcome
  the mode has already switched, so this metric can attribute the old trial's
  reward to the next mode. Use the outcome-masked Q-loss inputs to calculate
  per-mode reward statistics before interpreting these plots.

The final observation at a real environment `done` may be a reset image, and
the optional diversity reward deliberately excludes it. Verify `outcome_event`
timing and whether the visual trial-end image distinguishes arms in a real
DeepMind Lab rollout before enabling that reward.
