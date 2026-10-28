"""Reward attribution, evaluation isolation, and complete-session accounting."""

import json
import math

import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("loss", ["q", "policy"])
def test_cli_flushes_final_trial_and_checkpoint_evaluates(tmp_path, loss):
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
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["training"]["completed_trials"] == 12
    assert summary["training"]["optimizer_updates"] == 4
    assert (output / "manager.pt").is_file()
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
        ]
    )
    assert json.loads((evaluation / "summary.json").read_text())["held_out"] == summary["held_out"]
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
