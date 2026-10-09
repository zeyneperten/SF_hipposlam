"""Regression checks for high-level choices and trial history at rollout edges."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sample_factory.algo.utils.shared_buffers import policy_output_shapes
from sf_working_directories.zeynep.dmlab.custom_actor_critic import HighLevel_LossWrapper
from sf_working_directories.zeynep.dmlab.custom_core import HighLevelRNNWrapperCore
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import (
    HighLevelContextRNN_Learner, high_level_policy_loss, q_loss,
)


def make_core():
    cfg = SimpleNamespace(
        Hippo_n_feature=2, Hippo_R=1, Hippo_L=2,
        hl_K=2, hl_d_H=3, hl_history_len=2,
        hl_is_policy=False, hl_deterministic=True, oracle_context=False,
    )
    return HighLevelRNNWrapperCore(cfg, input_size=9)


def frame(outcome=0.0, reward=0.0):
    # Six base features followed by chosen_arm, outcome_event, prev_trial_reward.
    x = torch.zeros(1, 9)
    x[:, -2] = outcome
    x[:, -1] = reward
    return x


def test_actor_sampling_does_not_depend_on_autograd():
    model = HighLevelContextRNN_Learner(K=2, d_H=3, deterministic=False)
    logits = torch.zeros(1, 2)
    torch.manual_seed(4)
    with torch.no_grad():
        choices = [model.sample_mode(logits)[1].item() for _ in range(100)]
    assert set(choices) == {0, 1}
    model.deterministic = True
    with torch.no_grad():
        assert all(model.sample_mode(logits)[1].item() == 0 for _ in range(20))


def test_sampling_temperature_reaches_high_level_model():
    cfg = SimpleNamespace(
        Hippo_n_feature=2, Hippo_R=1, Hippo_L=2,
        hl_K=2, hl_d_H=3, hl_history_len=2, hl_tau=0.25,
        hl_is_policy=False, hl_deterministic=False, oracle_context=False,
    )
    core = HighLevelRNNWrapperCore(cfg, input_size=9)
    assert core.hl_learner.tau == 0.25


def test_actor_mode_is_a_recorded_policy_output():
    cfg = SimpleNamespace(core_name="BypassSS_HighLevelRNN", hl_K=2, double_value=False)
    names = [name for name, _ in policy_output_shapes(cfg, 1, 2)]
    assert "hl_z" in names

    class DummyActor:
        def __init__(self):
            self.cfg = cfg
            self.core = object()

        def forward(self):
            return {"values": torch.zeros(1),
                    "new_rnn_states": torch.tensor([[10.0, 1.0, 0.0]])}

        def summaries(self):
            return {}

    actor = HighLevel_LossWrapper(DummyActor())
    assert torch.equal(actor.forward()["hl_z"], torch.tensor([[1.0, 0.0]]))


def test_value_bootstrap_uses_training_mode_distribution():
    class DummyActor:
        def __init__(self):
            self.cfg = SimpleNamespace(core_name="other")
            self.core = SimpleNamespace(hl_learner=HighLevelContextRNN_Learner(K=2, d_H=3))

        def forward(self, values_only=False):
            z, _ = self.core.hl_learner.sample_mode(torch.zeros(1, 2))
            return {"values": z[:, 1], "new_rnn_states": z}

        def summaries(self):
            return {}

    actor = HighLevel_LossWrapper(DummyActor())
    torch.manual_seed(4)
    with torch.no_grad():
        bootstrap_values = [actor.forward(values_only=True)["values"].item() for _ in range(100)]
    assert set(bootstrap_values) == {0.0, 1.0}

    actor.core.hl_learner.deterministic = True
    with torch.no_grad():
        assert all(actor.forward(values_only=True)["values"].item() == 0.0 for _ in range(10))


def test_trial_history_survives_chunk_boundary_and_trains_rnn_cell():
    core = make_core()
    state = torch.zeros(1, core.get_core_state_size())
    with torch.no_grad():
        # First center trigger makes the initial choice. There is no previous
        # trial to record yet.
        _, state = core(frame(outcome=1), state)
        assert state[:, core.history_start:core.history_start + core.history_size].sum() == 0
        # Next trigger occurs in a later rollout and records the completed trial.
        _, state = core(frame(outcome=1, reward=1), state)
    history = state[:, core.history_start:core.history_start + core.history_size]
    assert history.reshape(1, 2, core.event_width)[0, -1, -1].item() == 1

    core.zero_grad()
    scores = core.scores_from_history(history, state[:, core.base_state_size:core.history_start])
    loss = (scores[0, 0] - 1.0).square()
    loss.backward()
    assert core.hl_learner.rnn_cell.weight_ih.grad.abs().sum() > 0


def test_replay_uses_recorded_actor_mode_for_decoder():
    core = make_core()
    state = torch.zeros(1, core.get_core_state_size())
    with torch.no_grad():
        _, state = core(frame(outcome=1), state)

    actor_z = F.one_hot(1 - state[:, -core.K:].argmax(dim=-1), core.K).float()
    history = state[:, core.history_start:core.history_start + core.history_size]
    replay_input = torch.cat((frame(outcome=1, reward=1), history, actor_z,
                              torch.ones(1, 1)), dim=-1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(replay_input.unsqueeze(0), [1])
    output, _ = core(packed, state)
    assert torch.equal(output.data[:, -core.K:], actor_z)


def test_replay_loss_reaches_prior_trial_update():
    core = make_core()
    state = torch.zeros(1, core.get_core_state_size())
    with torch.no_grad():
        _, state = core(frame(outcome=1), state)
        _, state = core(frame(outcome=1, reward=1), state)

    history = state[:, core.history_start:core.history_start + core.history_size]
    actor_z = state[:, -core.K:]
    replay_input = torch.cat((frame(outcome=1, reward=0), history, actor_z,
                              torch.ones(1, 1)), dim=-1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(replay_input.unsqueeze(0), [1])
    core.zero_grad()
    core(packed, state)
    core.last_hl_loss.backward()
    assert core.hl_learner.rnn_cell.weight_ih.grad.abs().sum() > 0


def test_invalid_transition_does_not_contribute_to_high_level_loss():
    core = make_core()
    state = torch.zeros(1, core.get_core_state_size())
    with torch.no_grad():
        _, state = core(frame(outcome=1), state)
    history = state[:, core.history_start:core.history_start + core.history_size]
    actor_z = state[:, -core.K:]
    replay_input = torch.cat((frame(outcome=1, reward=1), history, actor_z,
                              torch.zeros(1, 1)), dim=-1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(replay_input.unsqueeze(0), [1])
    core(packed, state)
    assert core.last_hl_loss.item() == 0.0


def test_only_policy_objective_can_be_negative():
    scores = torch.zeros(1, 1, 4)
    chosen = torch.zeros(1, 1, dtype=torch.long)
    rewards = torch.ones(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    policy, _ = high_level_policy_loss(scores, chosen, rewards, mask)
    assert policy.item() < 0  # Entropy bonus with zero advantage.
    assert q_loss(scores, chosen, rewards, mask).item() >= 0
