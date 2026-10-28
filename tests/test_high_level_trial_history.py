"""Shared trial replay preserves the full model's state and credit assignment."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sf_working_directories.zeynep.dmlab.custom_core import HighLevelRNNWrapperCore
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import HighLevelContextRNN_Learner, TrialHistory


def make_core(representations=False, is_policy=False):
    cfg = SimpleNamespace(
        Hippo_n_feature=2,
        Hippo_R=1,
        Hippo_L=2,
        hl_K=2,
        hl_d_H=3,
        hl_history_len=3,
        hl_d_rep=4,
        hl_representations=representations,
        hl_is_policy=is_policy,
        hl_deterministic=True,
        oracle_context=False,
        number_instruction_coef=1.0,
    )
    return HighLevelRNNWrapperCore(cfg, input_size=9)


def reference_append(layout, history, h, z, reward, mask, trigger, outcome):
    """Row-wise specification of the flat records used before extraction."""
    rows = []
    for b in range(history.size(0)):
        if mask[b] and z[b].sum() > 0.5:
            fields = [h[b].detach(), z[b].detach(), reward[b : b + 1]]
            if layout.representations:
                fields += [trigger[b].detach(), outcome[b].detach()]
            fields.append(reward.new_ones(1))
            rows.append(torch.cat([history[b, layout.event_width :], *fields]))
        else:
            rows.append(history[b])
    return torch.stack(rows)


def reference_scores(layout, learner, history, fallback):
    """Explicit per-row unroll, independent of the helper's masking/update API."""
    scores = []
    for b, records in enumerate(history.reshape(-1, layout.history_len, layout.event_width)):
        completed = records[records[:, -1] > 0.5]
        if len(completed) == 0:
            h = fallback[b]
        else:
            h = completed[0, : layout.d_H]
            for record in completed:
                # z, reward, and optional representations occupy every field
                # between the saved hidden state and the validity flag.
                h = learner.rnn_cell(record[layout.d_H : -1], h)
        scores.append(learner.head(h))
    return torch.stack(scores)


def example_history(layout):
    events = torch.randn(3, layout.history_len, layout.event_width)
    events[:, :, layout.d_H : layout.d_H + layout.K] = F.one_hot(
        torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), layout.K
    ).float()
    events[:, :, -1] = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    return events.flatten(1)


@pytest.mark.parametrize("representations", [False, True])
def test_record_layout_shift_masks_and_detachment(representations):
    torch.manual_seed(12)
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=representations, d_rep=4)
    assert layout.event_width == (15 if representations else 7)
    history = torch.zeros(3, layout.history_size)
    reference = history.clone()
    for t in range(5):
        h = torch.randn(3, 3, requires_grad=True)
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True)
        reward = torch.randn(3, requires_grad=True)
        trigger = torch.randn(3, 4, requires_grad=True)
        outcome = torch.randn(3, 4, requires_grad=True)
        mask = torch.tensor([True, t % 2 == 0, True])
        history = layout.append_event(history, h, z, reward, mask, trigger, outcome)
        reference = reference_append(layout, reference, h, z, reward, mask, trigger, outcome)
        torch.testing.assert_close(history, reference, rtol=0, atol=0)
    assert history[2].count_nonzero() == 0  # Initial trigger has no previous trial.
    history.sum().backward()
    assert h.grad is None
    assert z.grad is None
    assert trigger.grad is None
    assert outcome.grad is None
    torch.testing.assert_close(reward.grad, torch.tensor([1.0, 1.0, 0.0]))


@pytest.mark.parametrize("representations", [False, True])
@pytest.mark.parametrize("is_policy", [False, True])
def test_replay_matches_reference_scores_and_parameter_gradients(representations, is_policy, monkeypatch):
    torch.manual_seed(23)
    core = make_core(representations, is_policy)
    layout = core.trial_history
    learner = core.hl_learner
    reference = deepcopy(learner)
    history = example_history(layout)
    fallback = torch.randn(3, layout.d_H, requires_grad=True)
    reference_fallback = fallback.detach().clone().requires_grad_()

    def fail_sampling(*args, **kwargs):
        raise AssertionError("Trial replay must use saved modes, never sample new modes")

    monkeypatch.setattr(learner, "sample_mode", fail_sampling)
    actual = core.scores_from_history(history, fallback)
    expected = reference_scores(layout, reference, history, reference_fallback)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for parameter, reference_parameter in zip(learner.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    torch.testing.assert_close(fallback.grad, reference_fallback.grad)
    assert fallback.grad[:2].count_nonzero() == 0
    assert fallback.grad[2].abs().sum() > 0  # Empty history uses the caller's state.


@pytest.mark.parametrize("representations", [False, True])
def test_replay_gradient_reaches_oldest_completed_trial(representations):
    torch.manual_seed(34)
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=representations, d_rep=4)
    learner = HighLevelContextRNN_Learner(K=2, d_H=3, representations=representations, d_rep=4)
    history = example_history(layout)[1:2].detach().requires_grad_()
    scores = layout.scores_from_history(learner, history, torch.zeros(1, 3))
    scores.square().sum().backward()
    gradients = history.grad.reshape(1, layout.history_len, layout.event_width)
    assert gradients[0, 0, layout.d_H + layout.K].abs() > 0  # Reward two records back.
    assert gradients[0, 1].count_nonzero() == 0  # Invalid record is skipped.
    assert learner.rnn_cell.weight_hh.grad.abs().sum() > 0


@pytest.mark.parametrize("representations", [False, True])
@pytest.mark.parametrize("is_policy", [False, True])
def test_full_replay_scores_precede_current_reward_and_preserve_actor_mode(representations, is_policy, monkeypatch):
    torch.manual_seed(45)
    core = make_core(representations, is_policy)
    if representations:
        core.set_decoder(SimpleNamespace(mlp=torch.nn.Sequential(torch.nn.Linear(core.get_out_size(), core.d_rep))))
    state = torch.zeros(1, core.get_core_state_size())
    history = example_history(core.trial_history)[:1]
    state[:, core.history_start : core.history_start + core.history_size] = history
    state[:, -core.K :] = torch.tensor([[1.0, 0.0]])
    actor_mode = torch.tensor([[0.0, 1.0]])

    def fail_sampling(*args, **kwargs):
        raise AssertionError("Full learner replay must not resample actor modes")

    monkeypatch.setattr(core.hl_learner, "sample_mode", fail_sampling)
    scored = []
    hook = core.hl_learner.head.register_forward_hook(
        lambda module, inputs, output: scored.append(output.detach().clone()) if output.requires_grad else None
    )
    for reward in (0.0, 1.0):
        frame = torch.zeros(1, 9)
        frame[:, -2] = 1.0
        frame[:, -1] = reward
        replay_input = torch.cat((frame, history, actor_mode, torch.ones(1, 1)), dim=-1).unsqueeze(0)
        output, new_state = core(replay_input, state)
        torch.testing.assert_close(output[0, :, -core.K :], actor_mode)
        torch.testing.assert_close(new_state[:, -core.K :], actor_mode)
        last_record = new_state[:, core.history_start : core.history_start + core.history_size].reshape(
            1, core.history_len, core.event_width
        )[:, -1]
        assert last_record[0, core.d_H + core.K] == reward
    hook.remove()
    assert len(scored) == 2
    torch.testing.assert_close(scored[0], scored[1], rtol=0, atol=0)


@pytest.mark.parametrize("representations", [False, True])
def test_full_core_checkpoint_keys_and_flat_state_remain_compatible(representations):
    core = make_core(representations)
    expected = {
        "high_level_rnn_stage1.rnn_cell.weight_ih",
        "high_level_rnn_stage1.rnn_cell.weight_hh",
        "high_level_rnn_stage1.rnn_cell.bias_ih",
        "high_level_rnn_stage1.rnn_cell.bias_hh",
        "hl_learner.rnn_cell.weight_ih",
        "hl_learner.rnn_cell.weight_hh",
        "hl_learner.rnn_cell.bias_ih",
        "hl_learner.rnn_cell.bias_hh",
        "hl_learner.head.weight",
        "hl_learner.head.bias",
    }
    checkpoint = core.state_dict()
    assert set(checkpoint) == expected
    rep_width = 8 if representations else 0
    assert core.history_start == 7 + 3 + rep_width
    assert core.get_core_state_size() == 7 + 3 + rep_width + 3 * (3 + 2 + 2 + rep_width) + 2
    standalone = HighLevelContextRNN_Learner(K=2, d_H=3, representations=representations, d_rep=4)
    standalone.load_state_dict(
        {key[len("hl_learner.") :]: value for key, value in checkpoint.items() if key.startswith("hl_learner.")},
        strict=True,
    )
    clone = make_core(representations)
    clone.load_state_dict(checkpoint, strict=True)


def test_representation_history_requires_both_representations():
    layout = TrialHistory(K=2, d_H=3, history_len=3, representations=True, d_rep=4)
    with pytest.raises(ValueError, match="e_trigger and e_outcome"):
        layout.append_event(
            torch.zeros(1, layout.history_size),
            torch.zeros(1, 3),
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="at least one"):
        TrialHistory(K=2, d_H=3, history_len=0)
