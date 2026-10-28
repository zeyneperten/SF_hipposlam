"""Trial-level version of the Y-maze reward process, without navigation.

Arm indices are 0=left and 1=right; mode indices remain arbitrary actions. A
fixed mode-to-arm permutation defines the intended arm; an optional actuator
error executes the opposite arm. This is not a claim that the full controller
has learned that mapping. Hidden task state and executed arms are returned
only as diagnostic metadata and must never be passed to the learned policy.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BanditConfig:
    """Defaults match ``game_scripts/levels/ymaze_80_final.lua``.

    There are five nominal 15-trial blocks, initially right-high. At each
    inter-block boundary the high arm flips with probability 0.66. A nominal
    boundary therefore need not be an actual contingency switch.
    """

    p_high: float = 0.8
    p_low: float = 0.2
    block_length: int = 15
    blocks_per_session: int = 5
    switch_probability: float = 0.66
    initial_high_arm: str = "right"
    mode_to_arm: Tuple[int, int] = (0, 1)
    actuator_success_rate: float = 1.0

    def __post_init__(self):
        for name in ("p_high", "p_low", "switch_probability", "actuator_success_rate"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite probability in [0, 1]")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability in [0, 1]")
        if self.p_high < self.p_low:
            raise ValueError("p_high must be at least p_low")
        for name in ("block_length", "blocks_per_session"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_high_arm not in ("left", "right", "random"):
            raise ValueError("initial_high_arm must be 'left', 'right', or 'random'")
        try:
            mapping = tuple(self.mode_to_arm)
        except TypeError as error:
            raise ValueError("mode_to_arm must be a permutation of (0, 1)") from error
        if (
            len(mapping) != 2
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in mapping)
            or set(mapping) != {0, 1}
        ):
            raise ValueError("mode_to_arm must be a permutation of (0, 1)")
        object.__setattr__(self, "mode_to_arm", mapping)

    @property
    def session_trials(self) -> int:
        return self.block_length * self.blocks_per_session


class BlockSwitchBandit:
    """Vectorized independent sessions with synchronized trial boundaries.

    ``step`` returns binary manager rewards and per-trial diagnostic arrays.
    When probabilities sum to one, a single lottery chooses the winning arm,
    matching the Lua task's complementary potential outcomes. For a custom
    non-complementary pair, only the chosen arm's Bernoulli outcome is sampled.

    This removes travel time, failed initiation, and timeouts. Motor errors
    independently flip the commanded arm with probability
    ``1 - actuator_success_rate``. A step always represents exactly one
    completed trial. Sessions never reset
    implicitly; the caller must reset recurrent state together with ``reset``.
    """

    def __init__(self, config: BanditConfig, num_envs: int = 1, seed: Optional[int] = None):
        if not isinstance(config, BanditConfig):
            raise TypeError("config must be a BanditConfig")
        if isinstance(num_envs, (bool, np.bool_)) or not isinstance(num_envs, Integral) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.config = config
        self.num_envs = int(num_envs)
        self._seed_rngs(seed)
        self._mode_to_arm = np.asarray(config.mode_to_arm, dtype=np.int64)
        self.reset()

    @property
    def session_trials(self) -> int:
        return self.config.session_trials

    @property
    def done(self) -> bool:
        return self.trial_index >= self.session_trials

    @property
    def high_arm(self) -> np.ndarray:
        """Hidden contingency for evaluation controls only, returned as a copy."""
        return self._high_arm.copy()

    def _seed_rngs(self, seed: Optional[int]) -> None:
        # Keep the original default_rng(seed) reward/switch stream unchanged.
        # A child stream isolates actuator errors from hidden task randomness.
        seed_sequence = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(seed_sequence)
        self._actuator_rng = np.random.default_rng(seed_sequence.spawn(1)[0])

    def reset(self, seed: Optional[int] = None) -> None:
        """Begin fresh sessions; continue both RNG streams unless reseeded."""
        if seed is not None:
            self._seed_rngs(seed)
        self.trial_index = 0
        if self.config.initial_high_arm == "random":
            self._high_arm = self._rng.integers(0, 2, size=self.num_envs, dtype=np.int64)
        else:
            initial_arm = int(self.config.initial_high_arm == "right")
            self._high_arm = np.full(self.num_envs, initial_arm, dtype=np.int64)
        self._actual_switch = np.zeros(self.num_envs, dtype=bool)
        self._nominal_boundary = False

    def step(self, mode_indices) -> Tuple[np.ndarray, dict]:
        """Complete one trial for each environment.

        Metadata describes the decision just made, not the next trial. Both
        ``actual_switch`` and ``nominal_boundary`` are false on session trial
        zero and mark the first trial after a subsequent boundary. ``done``
        marks the final completed trial; its reward is still a valid target.
        ``commanded_arm`` is the mode's intended arm; ``chosen_arm`` is the
        executed arm. ``command_reward_probability`` marginalizes actuator
        error, while ``reward_probability`` conditions on the executed arm.
        """
        if self.done:
            raise RuntimeError("Session is complete; call reset() before step()")
        modes = np.asarray(mode_indices)
        if modes.shape != (self.num_envs,):
            raise ValueError(f"mode_indices must have shape ({self.num_envs},)")
        if not np.issubdtype(modes.dtype, np.integer) or np.any((modes < 0) | (modes > 1)):
            raise ValueError("mode_indices must contain integer mode indices 0 or 1")

        commanded_arm = self._mode_to_arm[modes]
        actuator_success = self._actuator_rng.random(self.num_envs) < self.config.actuator_success_rate
        chosen_arm = np.where(actuator_success, commanded_arm, 1 - commanded_arm)
        intended_probability = np.where(commanded_arm == self._high_arm, self.config.p_high, self.config.p_low)
        opposite_probability = np.where(commanded_arm == self._high_arm, self.config.p_low, self.config.p_high)
        command_reward_probability = (
            self.config.actuator_success_rate * intended_probability
            + (1.0 - self.config.actuator_success_rate) * opposite_probability
        )
        reward_probability = np.where(chosen_arm == self._high_arm, self.config.p_high, self.config.p_low)
        lottery = self._rng.random(self.num_envs)
        if np.isclose(self.config.p_high + self.config.p_low, 1.0, rtol=0.0, atol=1e-12):
            winning_arm = np.where(lottery < self.config.p_high, self._high_arm, 1 - self._high_arm)
            reward = (chosen_arm == winning_arm).astype(np.float32)
        else:
            reward = (lottery < reward_probability).astype(np.float32)

        metadata = {
            "trial": np.full(self.num_envs, self.trial_index, dtype=np.int64),
            "high_arm": self._high_arm.copy(),
            "commanded_arm": commanded_arm.copy(),
            "chosen_arm": chosen_arm.copy(),
            "actuator_success": actuator_success,
            "block_index": np.full(self.num_envs, self.trial_index // self.config.block_length, dtype=np.int64),
            "trial_in_block": np.full(self.num_envs, self.trial_index % self.config.block_length, dtype=np.int64),
            "actual_switch": self._actual_switch.copy(),
            "nominal_boundary": np.full(self.num_envs, self._nominal_boundary, dtype=bool),
            "reward_probability": reward_probability,
            "command_reward_probability": command_reward_probability,
            "done": np.full(self.num_envs, self.trial_index + 1 == self.session_trials, dtype=bool),
        }
        self.trial_index += 1
        self._nominal_boundary = not self.done and self.trial_index % self.config.block_length == 0
        if self._nominal_boundary:
            self._actual_switch = self._rng.random(self.num_envs) < self.config.switch_probability
            self._high_arm = np.where(self._actual_switch, 1 - self._high_arm, self._high_arm)
        else:
            self._actual_switch = np.zeros(self.num_envs, dtype=bool)
        return reward, metadata
