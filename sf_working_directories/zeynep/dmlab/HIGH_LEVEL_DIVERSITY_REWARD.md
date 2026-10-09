# Optional trial-end mode diversity reward

This note explains the auxiliary reward added for Zeynep's Y-maze high-level
RNN experiment. Its purpose is to encourage the low-level controller to reach
different trial-end states under different high-level modes $z$. The feature is
off by default. The current `exp_ymaze_HLRNN.py` launch also leaves it off;
set `--hl_diversity_reward_coef=0.01` to test an auxiliary reward whose
magnitude is at most $0.01$ per completed trial.

## What happens at a trial boundary

For an action at frame $t$, the actor has already recorded the mode $z_t$ that
controlled its decoder. If the next observation $s_{t+1}$ has an
`outcome_event` pulse, the learner classifies the image in $s_{t+1}$ and gives
a bonus to **that action's** reward. This includes an event in the last
observation of a 64-frame rollout, because the rollout buffer holds $T+1$
observations. The label for classifier training is the mode in the recurrent
state **before** the outcome event; the core chooses the next trial's mode
after processing that event.

The classifier has its own small CNN. It sees only the trial-end visual
observation. It does not see $z$, `prev_trial_reward`, `outcome_event`, the
high-level hidden state, or the policy encoder features. An optional
`--hl_diversity_include_chosen_arm=True` also supplies the reached arm if the
trial-end image is indistinguishable after the agent returns to the center.
That option is off in the experiment until the level's observation timing is
confirmed.

Let $q_\phi(z\mid s)$ be the classifier probability and let
$\bar q_\phi(z)$ be its average prediction over valid outcome states in the
current learner batch. The bonus is

$$
r_t^{\mathrm{div}} = \beta\,\mathbf{1}[\mathrm{outcome}_{t+1}]
\operatorname{clip}\!\left(
\frac{\log q_\phi(z_t\mid s_{t+1})-\log \bar q_\phi(z_t)}{\log K},
-1,1\right),
$$

with $\beta=0.01$ and $K=\texttt{hl_K}=4$ in the opt-in configuration. The
term lies between $-0.01$ and $0.01$: incorrect predictions can give a small
penalty. Keeping signed evidence matters because clipping each negative value
to zero paid a positive average bonus even when the classifier guessed at
chance. The batch marginal removes the easy reward
for predicting the most common mode from class frequency alone: a classifier
whose output is identical for every reached state gives zero bonus. A batch
with fewer than two represented modes also gives zero bonus. The classifier
learns with cross entropy, weighted to give each represented mode equal total
weight in its minibatch.

The bonus is added to Sample Factory's already scaled **frame reward** before
GAE and PPO return calculation. With this experiment's `--reward_scale=0.1`,
$0.01$ is one tenth of a scaled environment reward of $1$. The high-level
Q-regression target remains the original `prev_trial_reward` from the level;
the auxiliary bonus trains the low-level controller through PPO. The
classifier loss is added to the learner's total loss with weight $0.1$ and
updates only the classifier parameters, which are registered on the actor
critic before optimizer creation. Its loss is computed in the learner, where
the trial-end observations and pre-event modes are available.

## Files and switches

| File | Change |
| --- | --- |
| `high_level_diversity.py` | Define the optional CNN, class-balanced loss, and bounded trial-end bonus. |
| `custom_actor_critic.py` | Attach the classifier only when the bonus coefficient is positive. |
| `custom_params.py` | Add the three `hl_diversity_*` options; default bonus coefficient is zero. |
| `sample_factory/algo/learning/learner.py` | Add the bonus before GAE and the classifier loss to the existing optimization step, guarded by the coefficient. The rollout reward tensor is cloned so shared experience is untouched. |
| `experiments/exp_ymaze_HLRNN.py` | Contains an example `--hl_diversity_reward_coef=0.01` switch, currently commented out. |
| `tests/test_high_level_diversity.py` | Check reward timing, boundary handling, masking, prior-only predictions, and classifier gradients. |

The learner logs `hl/diversity_classifier_loss`,
`hl/diversity_classifier_accuracy`, `hl/diversity_event_count`, and
`hl/diversity_bonus_mean`. The last value averages over **all frames** in the
minibatch, so it can be small when trials are long. For a baseline, launch
the same experiment with `--hl_diversity_reward_coef=0`; then the classifier
is not created and the standard learner path is unchanged. No Sample Factory
buffer format or default model is modified by this feature.
The existing `hl/hl_loss` metric still describes the high-level Q loss alone;
the classifier has its own loss metric. Because the enabled actor critic has
new parameters, start a fresh run rather than loading a checkpoint saved
without the classifier.

## Validation and limits

Run the focused checks with
`python -m unittest tests.test_high_level_diversity -v`. They use synthetic
observations and do not require DeepMind Lab. Syntax and whitespace checks
were also run on the changed files. In a separate 32-sample synthetic image
probe, a classifier learned two visibly distinct end states in 40 Adam steps:
accuracy rose to $1.0$ and mean normalized evidence rose from $0$ to $1.0$.
This checks the classifier path, not Y-maze learning speed. The configured Lua level
`ymaze_vol5_INSTR_HL` is absent from this branch, so an actual rollout is
still needed to check whether the visual outcome observation retains path
information. If it does not, the visual-only classifier should stay near
chance and its bonus near zero; use the optional reached-arm input only if
that is the intended definition of terminal state. A terminal environment
`done` may replace the final observation with a reset observation, so such a
transition is excluded from the bonus. Ordinary trial boundaries inside a
75-trial episode remain eligible.

This bonus can reward distinct end states, but it does not prove that the
controller took a distinct route along the entire trial. Compare outcome
images, mode frequencies, reached arms, and the environmental task reward in
the first real run. If all modes collapse to one, the zero bonus is expected;
mode exploration must still come from the high-level sampler.

## Regression correction (2026-10-09)

The original positive-only clipping gave almost half the maximum bonus in a
10,000-sample probe where mode guesses were independent of the true mode and
accuracy was $49.5\%$. Keeping the signed term made that chance-level signal
average to approximately zero. The new unit check covers an exactly balanced
chance-level example. This does not replace an end-to-end check of actual
trial-end images.
