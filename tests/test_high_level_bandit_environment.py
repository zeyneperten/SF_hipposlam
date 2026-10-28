"""Reward-process checks independent of DeepMind Lab and neural networks."""

import numpy as np
import pytest

from sf_working_directories.zeynep.bandit.environment import BanditConfig, BlockSwitchBandit


def test_default_session_has_exactly_five_fifteen_trial_blocks():
    env = BlockSwitchBandit(BanditConfig(), num_envs=3, seed=11)
    assert env.session_trials == 75
    np.testing.assert_array_equal(env.high_arm, [1, 1, 1])
    previous_high = env.high_arm
    for trial in range(75):
        reward, info = env.step(np.zeros(3, dtype=np.int64))
        assert reward.dtype == np.float32
        assert all(value.shape == (3,) for value in info.values())
        np.testing.assert_array_equal(info["trial"], trial)
        np.testing.assert_array_equal(info["block_index"], trial // 15)
        np.testing.assert_array_equal(info["trial_in_block"], trial % 15)
        np.testing.assert_array_equal(info["nominal_boundary"], trial > 0 and trial % 15 == 0)
        np.testing.assert_array_equal(info["actual_switch"], info["high_arm"] != previous_high)
        np.testing.assert_array_equal(info["done"], trial == 74)
        previous_high = info["high_arm"]
    assert env.done and env.trial_index == 75
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0, 0])
    env.reset()
    assert not env.done and env.trial_index == 0
    np.testing.assert_array_equal(env.high_arm, [1, 1, 1])


def test_deterministic_switches_have_correct_reward_and_first_trial_alignment():
    cfg = BanditConfig(p_high=1.0, p_low=0.0, block_length=2, blocks_per_session=3, switch_probability=1.0)
    env = BlockSwitchBandit(cfg, num_envs=2, seed=7)
    for trial, expected_high in enumerate([1, 1, 0, 0, 1, 1]):
        np.testing.assert_array_equal(env.high_arm, expected_high)
        reward, info = env.step([0, 1])
        np.testing.assert_array_equal(reward, [1 - expected_high, expected_high])
        np.testing.assert_array_equal(info["high_arm"], expected_high)
        np.testing.assert_array_equal(info["actual_switch"], trial in (2, 4))
        np.testing.assert_array_equal(info["nominal_boundary"], trial in (2, 4))


def test_nominal_boundaries_without_switches_and_mode_permutation():
    cfg = BanditConfig(
        p_high=1.0,
        p_low=0.0,
        block_length=1,
        blocks_per_session=3,
        switch_probability=0.0,
        initial_high_arm="left",
        mode_to_arm=(1, 0),
    )
    env = BlockSwitchBandit(cfg, num_envs=2, seed=1)
    for trial in range(3):
        reward, info = env.step([0, 1])
        np.testing.assert_array_equal(reward, [0, 1])
        np.testing.assert_array_equal(info["chosen_arm"], [1, 0])
        assert not info["actual_switch"].any()
        np.testing.assert_array_equal(info["nominal_boundary"], trial > 0)


def test_complementary_lottery_has_exactly_one_winning_arm():
    cfg = BanditConfig()
    left = BlockSwitchBandit(cfg, num_envs=64, seed=31)
    right = BlockSwitchBandit(cfg, num_envs=64, seed=31)
    for _ in range(cfg.session_trials):
        left_reward, left_info = left.step(np.zeros(64, dtype=np.int64))
        right_reward, right_info = right.step(np.ones(64, dtype=np.int64))
        np.testing.assert_array_equal(left_reward + right_reward, 1)
        np.testing.assert_array_equal(left_info["high_arm"], right_info["high_arm"])


@pytest.mark.parametrize("p_high,p_low", [(0.8, 0.2), (0.9, 0.4)])
def test_reward_marginals_and_boundary_hazard(p_high, p_low):
    cfg = BanditConfig(p_high=p_high, p_low=p_low, block_length=1, blocks_per_session=2)
    env = BlockSwitchBandit(cfg, num_envs=40000, seed=8)
    choices = np.tile([0, 1], 20000)
    reward, info = env.step(choices)
    assert abs(reward[choices == 1].mean() - p_high) < 0.015
    assert abs(reward[choices == 0].mean() - p_low) < 0.015
    _, boundary_info = env.step(choices)
    assert abs(boundary_info["actual_switch"].mean() - 0.66) < 0.015
    np.testing.assert_allclose(info["reward_probability"], np.where(choices == 1, p_high, p_low))


def test_random_initialization_reset_seed_and_metadata_are_independent():
    cfg = BanditConfig(initial_high_arm="random", block_length=1, blocks_per_session=1)
    env = BlockSwitchBandit(cfg, num_envs=100, seed=18)
    first_high = env.high_arm
    assert set(first_high) == {0, 1}
    reward, info = env.step(np.zeros(100, dtype=np.int64))
    info["high_arm"][:] = 9
    np.testing.assert_array_equal(env.high_arm, first_high)
    env.reset(seed=18)
    np.testing.assert_array_equal(env.high_arm, first_high)
    repeated_reward, _ = env.step(np.zeros(100, dtype=np.int64))
    np.testing.assert_array_equal(repeated_reward, reward)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_high": float("nan")},
        {"p_low": -0.1},
        {"p_high": 0.1, "p_low": 0.2},
        {"switch_probability": 1.01},
        {"switch_probability": True},
        {"block_length": 0},
        {"block_length": 2.5},
        {"blocks_per_session": False},
        {"initial_high_arm": "unknown"},
        {"mode_to_arm": (0, 0)},
        {"mode_to_arm": (False, True)},
        {"mode_to_arm": None},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BanditConfig(**kwargs)


@pytest.mark.parametrize("num_envs", [0, -1, 1.5, True])
def test_invalid_environment_count_is_rejected(num_envs):
    with pytest.raises(ValueError):
        BlockSwitchBandit(BanditConfig(), num_envs=num_envs)


@pytest.mark.parametrize("modes", [[0], [0, 2], [0.0, 1.0], [True, False], [[0, 1]]])
def test_invalid_actions_do_not_advance_environment(modes):
    env = BlockSwitchBandit(BanditConfig(), num_envs=2, seed=19)
    control = BlockSwitchBandit(BanditConfig(), num_envs=2, seed=19)
    with pytest.raises(ValueError):
        env.step(modes)
    assert env.trial_index == 0
    reward, _ = env.step([0, 1])
    control_reward, _ = control.step([0, 1])
    np.testing.assert_array_equal(reward, control_reward)
