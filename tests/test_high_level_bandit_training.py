"""Reward attribution, evaluation isolation, and complete-session accounting."""

import json
import math

import numpy as np
import pytest
import torch

from sf_working_directories.zeynep.bandit import train as training_module
from sf_working_directories.zeynep.bandit.environment import BanditConfig
from sf_working_directories.zeynep.bandit.train import evaluate, initial_hidden, main, trial_loss
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import HighLevelContextRNN_Learner, TrialHistory


def test_initial_trigger_matches_full_model():
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    expected = model.rnn_cell(torch.zeros(4, 3), torch.zeros(4, 3))
    assert torch.equal(initial_hidden(model, 4), expected)
    assert not initial_hidden(model, 4).requires_grad


@pytest.mark.parametrize("is_policy", [False, True])
def test_loss_credits_recorded_choice_without_sampling(is_policy):
    model = HighLevelContextRNN_Learner(K=2, d_H=3, is_policy=is_policy)
    history = TrialHistory(2, 3, 2)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    samples = [
        (torch.zeros(2, history.history_size), initial_hidden(model, 2), torch.tensor([0, 1]), torch.tensor([1.0, 0.0]))
    ]
    rng = torch.get_rng_state().clone()
    loss = trial_loss(model, history, samples)
    assert torch.equal(rng, torch.get_rng_state())
    loss.backward()
    assert model.head.bias.grad[0] < 0  # increase score of mode that earned reward
    assert model.head.bias.grad[1] >= 0


def test_evaluation_preserves_weights_rng_and_training_mode():
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    task = BanditConfig(block_length=3, blocks_per_session=2)
    first, rows = evaluate(model, task, 8, 99)
    second, repeated = evaluate(model, task, 8, 99)
    assert first == second and rows == repeated
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_q_temperature_ceiling_is_separate_from_greedy_accuracy():
    model = HighLevelContextRNN_Learner(K=2, d_H=3, tau=1.0)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([0.2, 0.8]))
    task = BanditConfig(block_length=10, blocks_per_session=1)
    sampled, _ = evaluate(model, task, 600, 44, "sampled")
    greedy, _ = evaluate(model, task, 20, 44, "greedy")
    assert abs(sampled["high_arm_accuracy"] - 1 / (1 + math.exp(-0.6))) < 0.025
    assert greedy["high_arm_accuracy"] == 1
    assert sampled["actual_switches"] == 0
    assert all(item["n"] == 0 for item in sampled["actual_switch_curve"])


def test_actual_switch_curve_counts_reversals_not_nominal_boundaries():
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    task = BanditConfig(block_length=3, blocks_per_session=3, switch_probability=1)
    summary, rows = evaluate(model, task, 4, 44, "oracle")
    assert summary["actual_switches"] == 8
    assert all(point["n"] == 8 for point in summary["actual_switch_curve"])
    assert all(point["high_arm_accuracy"] == 1 for point in summary["actual_switch_curve"])
    assert all(row["since_actual_switch"] == -1 for row in rows if row["trial"] < 3)


def test_evaluation_recurrent_input_uses_commanded_mode_when_actuator_always_fails(monkeypatch):
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([1.0, 0.0]))
    original_update = model.update_state
    inputs = []

    def record_update(event, reward, hidden, mode):
        inputs.append((reward.clone(), mode.clone()))
        return original_update(event, reward, hidden, mode)

    monkeypatch.setattr(model, "update_state", record_update)
    task = BanditConfig(actuator_success_rate=0.0, p_high=1.0, p_low=0.0, block_length=3, blocks_per_session=1)
    _, rows = evaluate(model, task, 2, 91, "greedy")
    assert len(inputs) == task.session_trials + 1
    assert not inputs[0][0].any() and not inputs[0][1].any()
    assert all(row["mode"] == row["commanded_arm"] == 0 and row["chosen_arm"] == 1 for row in rows)
    for reward, mode in inputs[1:]:
        torch.testing.assert_close(mode, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        torch.testing.assert_close(reward, torch.ones(2))


def test_training_and_replay_credit_commanded_mode_when_actuator_always_fails(tmp_path, monkeypatch):
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    original_update = model.update_state
    original_loss = training_module.trial_loss
    recurrent_modes, credited_modes, credited_rewards = [], [], []

    def choose_mode_zero(scores):
        indices = torch.zeros(len(scores), dtype=torch.long)
        return torch.nn.functional.one_hot(indices, num_classes=2).float(), indices

    def record_update(event, reward, hidden, mode):
        recurrent_modes.append(mode[event & (mode.sum(-1) > 0)].clone())
        return original_update(event, reward, hidden, mode)

    def record_loss(learner, history, samples):
        credited_modes.extend(sample[2].clone() for sample in samples)
        credited_rewards.extend(sample[3].clone() for sample in samples)
        return original_loss(learner, history, samples)

    monkeypatch.setattr(model, "sample_mode", choose_mode_zero)
    monkeypatch.setattr(model, "update_state", record_update)
    monkeypatch.setattr(training_module, "trial_loss", record_loss)
    monkeypatch.setattr(
        training_module,
        "evaluate",
        lambda *args: (
            {
                "high_arm_accuracy": 1.0,
                "commanded_high_arm_accuracy": 0.0,
                "reward_rate": 1.0,
                "actual_switches": 0,
            },
            [],
        ),
    )
    args = training_module.parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--num-envs",
            "2",
            "--session-batches",
            "1",
            "--trials-per-update",
            "2",
        ]
    )
    task = BanditConfig(actuator_success_rate=0.0, p_high=1.0, p_low=0.0, block_length=3, blocks_per_session=1)
    accounting = training_module.train(args, model, task, tmp_path)
    assert accounting["completed_trials"] == 6 and accounting["optimizer_updates"] == 2
    modes = torch.cat(recurrent_modes)
    assert len(modes) >= 6  # Includes the live recurrent updates and history replay.
    torch.testing.assert_close(modes, torch.tensor([1.0, 0.0]).expand_as(modes))
    torch.testing.assert_close(torch.cat(credited_modes), torch.zeros(6, dtype=torch.long))
    torch.testing.assert_close(torch.cat(credited_rewards), torch.ones(6))
    row = json.loads((tmp_path / "learning.jsonl").read_text())
    assert row["high_arm_accuracy"] == 1.0 and row["commanded_high_arm_accuracy"] == 0.0
    assert row["actuator_success_rate"] == 0.0


@pytest.mark.parametrize("mapping", [(0, 1), (1, 0)])
def test_win_stay_lose_shift_uses_previous_command_not_executed_arm(mapping):
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    task = BanditConfig(
        actuator_success_rate=0.0,
        mode_to_arm=mapping,
        p_high=1.0,
        p_low=0.0,
        block_length=4,
        blocks_per_session=2,
        switch_probability=1.0,
    )
    _, rows = evaluate(model, task, 8, 44, "win_stay_lose_shift")
    for session in range(8):
        trajectory = [row for row in rows if row["session"] == session]
        for previous, current in zip(trajectory, trajectory[1:]):
            expected = previous["commanded_arm"] if previous["reward"] else 1 - previous["commanded_arm"]
            assert current["commanded_arm"] == expected
            assert current["chosen_arm"] == 1 - expected
            assert current["changed_command"] == (previous["reward"] == 0)


@pytest.mark.parametrize("success_rate", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("mapping", [(0, 1), (1, 0)])
def test_oracle_uses_best_command_and_has_zero_attainable_regret(success_rate, mapping):
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    task = BanditConfig(
        actuator_success_rate=success_rate,
        mode_to_arm=mapping,
        block_length=3,
        blocks_per_session=3,
        switch_probability=1.0,
    )
    summary, rows = evaluate(model, task, 256, 77, "oracle")
    expected_accuracy = max(success_rate, 1 - success_rate)
    assert summary["expected_regret_per_trial"] == pytest.approx(0, abs=1e-12)
    assert abs(summary["high_arm_accuracy"] - expected_accuracy) < 0.035
    assert summary["commanded_high_arm_accuracy"] == float(success_rate >= 0.5)
    for row in rows:
        expected_command = row["high_arm"] if success_rate >= 0.5 else 1 - row["high_arm"]
        assert row["commanded_arm"] == expected_command
        assert mapping[row["mode"]] == expected_command
        assert row["expected_regret"] == pytest.approx(0, abs=1e-12)
    if 0 < success_rate < 1:
        assert summary["perfect_actuator_regret_per_trial"] > 0


def test_command_and_execution_accuracy_are_reported_separately_with_actuator_noise():
    model = HighLevelContextRNN_Learner(K=2, d_H=3)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([0.0, 1.0]))
    task = BanditConfig(actuator_success_rate=0.8, block_length=10, blocks_per_session=1)
    summary, rows = evaluate(model, task, 600, 43, "greedy")
    assert summary["commanded_high_arm_accuracy"] == 1.0
    assert summary["stable_half_block_commanded_accuracy"] == 1.0
    assert abs(summary["high_arm_accuracy"] - 0.8) < 0.025
    assert summary["actuator_success_rate"] == summary["high_arm_accuracy"]
    assert abs(summary["reward_rate"] - 0.68) < 0.025
    assert summary["expected_regret_per_trial"] == pytest.approx(0.0)
    assert summary["perfect_actuator_regret_per_trial"] == pytest.approx(0.6 * (1 - summary["high_arm_accuracy"]))
    assert all(point["commanded_high_arm_accuracy"] == 1.0 for point in summary["nominal_block_curve"])
    assert all(session["commanded_high_arm_accuracy"] == 1.0 for session in summary["sessions"])
    assert any(row["chosen_arm"] != row["commanded_arm"] for row in rows)


@pytest.mark.parametrize("loss", ["q", "policy"])
@pytest.mark.parametrize("success_rate", [1.0, 0.8])
def test_cli_flushes_final_trial_and_checkpoint_evaluates(tmp_path, loss, success_rate):
    output = tmp_path / "run"
    main(
        [
            "--output",
            str(output),
            "--session-batches",
            "2",
            "--num-envs",
            "2",
            "--block-length",
            "3",
            "--blocks-per-session",
            "1",
            "--eval-sessions",
            "2",
            "--trials-per-update",
            "2",
            "--eval-every",
            "2",
            "--loss",
            loss,
            "--actuator-success-rate",
            str(success_rate),
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["training"]["completed_trials"] == 12
    assert summary["training"]["optimizer_updates"] == 4
    assert (output / "manager.pt").is_file()
    assert json.loads((output / "config.json").read_text())["task"]["actuator_success_rate"] == success_rate
    for checkpoint in output.glob("manager*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["extra"]["task"]["actuator_success_rate"] == success_rate
        assert "actuator_success_rate" not in payload["metadata"]  # Task provenance is not an architecture constraint.
    evaluation = tmp_path / "evaluation"
    main(
        [
            "--output",
            str(evaluation),
            "--evaluate-checkpoint",
            str(output / "manager.pt"),
            "--block-length",
            "3",
            "--blocks-per-session",
            "1",
            "--eval-sessions",
            "2",
            "--actuator-success-rate",
            str(success_rate),
        ]
    )
    assert json.loads((evaluation / "summary.json").read_text())["held_out"] == summary["held_out"]
    changed_actuator = tmp_path / "changed_actuator"
    other_rate = 0.8 if success_rate == 1.0 else 1.0
    main(
        [
            "--output",
            str(changed_actuator),
            "--evaluate-checkpoint",
            str(output / "manager.pt"),
            "--block-length",
            "3",
            "--blocks-per-session",
            "1",
            "--eval-sessions",
            "2",
            "--actuator-success-rate",
            str(other_rate),
        ]
    )
    assert json.loads((changed_actuator / "config.json").read_text())["task"]["actuator_success_rate"] == other_rate
    assert json.loads((changed_actuator / "summary.json").read_text())["training"] is None
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])


def test_import_does_not_require_sample_factory_or_dmlab():
    import subprocess
    import sys

    script = (
        "import sys; import sf_working_directories.zeynep.bandit.train; "
        "assert not any(k.startswith(('sample_factory', 'deepmind_lab', 'torchvision')) for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("extra", [["--eval-tau", "0.25"], ["--num-envs", "1", "--trials-per-update", "1"]])
def test_invalid_policy_settings_fail_before_creating_output(tmp_path, extra):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        main(["--output", str(output), "--loss", "policy"] + extra)
    assert not output.exists()
