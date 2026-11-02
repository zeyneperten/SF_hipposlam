#!/usr/bin/env python3
"""Discrete Reinforcement Learning Test Run for High-Level RNN.

This test simulates the Y-maze probabilistic reversal learning task without
requiring the 3D environment, low-level core, or Sample Factory.

Task Specifications:
- 2 Arms: Right (1) and Left (2).
- Probabilistic Rewards:
    * High-reward arm pays r=1.0 with 80% probability, r=0.0 with 20%.
    * Low-reward arm pays r=1.0 with 20% probability, r=0.0 with 80%.
- Block Structure & Contingency Reversals:
    * 15 trials per block (matching ymaze_80_final.lua).
    * At the end of every block, contingencies switch with 66% probability
      (shift_roll <= 0.66) and remain the same with 34% probability.
- High-Level RNN Inputs:
    * current choice / chosen_arm (previous choice made)
    * reward / prev_trial_reward (outcome received: 1.0 or 0.0)
    * inferred mod / z_prev (one-hot latent mode vector)
- Efficiency Metrics:
    * Choice accuracy (% High-Reward Arm selections)
    * Reversal adaptation curve (accuracy on trials 1..15 post-switch)
    * Flexibility Index: (P(R|Base) - P(R|Rev)) / (P(R|Base) + P(R|Rev))
    * Win-Stay / Lose-Shift (WSLS) behavioral profile
    * Mode specialization and Q-value / policy entropy

Usage:
    # Run from repository root or ymaze directory:
    python tests/test_high_level_rnn_discrete.py --episodes 50
    python tests/test_high_level_rnn_discrete.py --is-policy --episodes 50
    python -m unittest tests/test_high_level_rnn_discrete.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

# Ensure ymaze directory is on sys.path for workspace module imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ymaze_dir = os.path.abspath(os.path.join(_script_dir, ".."))
_cwd_ymaze = os.path.join(os.getcwd(), "ymaze")
for _p in (_ymaze_dir, _cwd_ymaze, os.getcwd()):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False



# ==============================================================================
# 1. Discrete Environment (Y-Maze Reversal Learning Task)
# ==============================================================================

class DiscreteYMazeBanditEnv:
    """Discrete 2-armed reversal bandit simulating the Y-Maze environment.

    Matches the exact contingency logic in `ymaze_80_final.lua`:
    - 15 trials per block.
    - Global 80/20 reward lottery.
    - 66% probability of contingency reversal per block boundary.
    - Tracks choices, rewards, and ground truth context for evaluation.
    """

    ARM_RIGHT = 1
    ARM_LEFT = 2

    def __init__(
        self,
        trials_per_block: int = 15,
        p_reward_high: float = 0.80,
        p_reward_low: float = 0.20,
        p_switch: float = 0.66,
        seed: Optional[int] = None,
    ):
        self.trials_per_block = trials_per_block
        self.p_reward_high = p_reward_high
        self.p_reward_low = p_reward_low
        self.p_switch = p_switch
        self.rng = random.Random(seed)

        # Internal state
        self.current_high_arm = self.ARM_RIGHT
        self.block_count = 1
        self.trial_in_block = 1
        self.total_trials = 0
        self.trials_since_switch = 0
        self.contingency_switched = False

        # Previous step information
        self.prev_choice = 0
        self.prev_reward = 0.0

        # Behavioral counters for flexibility index
        self.base_right_count = 0
        self.base_left_count = 0
        self.rev_right_count = 0
        self.rev_left_count = 0

    def reset(self, start_arm: Optional[int] = None):
        """Reset environment to trial 1 of block 1."""
        if start_arm in (self.ARM_RIGHT, self.ARM_LEFT):
            self.current_high_arm = start_arm
        else:
            self.current_high_arm = self.rng.choice([self.ARM_RIGHT, self.ARM_LEFT])

        self.block_count = 1
        self.trial_in_block = 1
        self.total_trials = 0
        self.trials_since_switch = 0
        self.contingency_switched = False

        self.prev_choice = 0
        self.prev_reward = 0.0

        self.base_right_count = 0
        self.base_left_count = 0
        self.rev_right_count = 0
        self.rev_left_count = 0

        return self._get_obs()

    def _get_obs(self) -> Dict[str, float]:
        """Generate observation matching the agent's inputs."""
        return {
            "outcome_event": 1.0,
            "prev_trial_reward": float(self.prev_reward),
            "chosen_arm": float(self.prev_choice),
            "inst_block": 1.0 if self.current_high_arm == self.ARM_RIGHT else 2.0,
            "high_reward_arm": self.current_high_arm,
            "trial_in_block": self.trial_in_block,
            "block_count": self.block_count,
            "trials_since_switch": self.trials_since_switch,
            "contingency_switched": self.contingency_switched,
        }

    def step(self, choice: int) -> Tuple[Dict[str, float], float, bool, Dict]:
        """Execute a discrete arm choice (1=Right, 2=Left).

        Returns:
            obs: observation dict for next trial
            reward: immediate reward obtained (1.0 or 0.0)
            done: False (infinite horizon / controlled by runner)
            info: diagnostic metadata
        """
        assert choice in (self.ARM_RIGHT, self.ARM_LEFT), f"Invalid choice: {choice}"

        is_high_arm = (choice == self.current_high_arm)

        # 80/20 probabilistic lottery
        roll = self.rng.random()
        if is_high_arm:
            reward = 1.0 if roll < self.p_reward_high else 0.0
        else:
            reward = 1.0 if roll < self.p_reward_low else 0.0

        # Update baseline vs reversal counters (for Flexibility Index)
        if self.current_high_arm == self.ARM_RIGHT:
            if choice == self.ARM_RIGHT:
                self.base_right_count += 1
            else:
                self.base_left_count += 1
        else:
            if choice == self.ARM_RIGHT:
                self.rev_right_count += 1
            else:
                self.rev_left_count += 1

        info = {
            "choice": choice,
            "reward": reward,
            "is_high_arm": is_high_arm,
            "high_reward_arm": self.current_high_arm,
            "trial_in_block": self.trial_in_block,
            "block_count": self.block_count,
            "trials_since_switch": self.trials_since_switch,
        }

        # Advance trial and check block transition
        self.prev_choice = choice
        self.prev_reward = reward
        self.total_trials += 1
        self.trials_since_switch += 1
        self.contingency_switched = False

        if self.trial_in_block >= self.trials_per_block:
            # End of block reached! Roll for switch (66% probability)
            self.block_count += 1
            self.trial_in_block = 1
            shift_roll = self.rng.random()
            if shift_roll <= self.p_switch:
                # Contingency inversion
                self.current_high_arm = (
                    self.ARM_LEFT if self.current_high_arm == self.ARM_RIGHT else self.ARM_RIGHT
                )
                self.contingency_switched = True
                self.trials_since_switch = 0
        else:
            self.trial_in_block += 1

        obs = self._get_obs()
        return obs, reward, False, info

    def get_flexibility_index(self) -> float:
        """Calculate Flexibility Index = (P_base - P_rev) / (P_base + P_rev)."""
        total_base = self.base_right_count + self.base_left_count
        p_base = (self.base_right_count / total_base) if total_base > 0 else 0.0

        total_rev = self.rev_right_count + self.rev_left_count
        p_rev = (self.rev_right_count / total_rev) if total_rev > 0 else 0.0

        denom = p_base + p_rev
        if total_base > 0 and total_rev > 0 and denom > 0:
            return (p_base - p_rev) / denom
        return 0.0


# ==============================================================================
# 1B. Vectorized Session Bandit (Supervisor's Exact Evaluation Environment)
# ==============================================================================

@dataclass(frozen=True)
class BanditConfig:
    """Exact task configuration matching supervisor's standalone pilot."""
    p_high: float = 0.8
    p_low: float = 0.2
    block_length: int = 15
    blocks_per_session: int = 5
    switch_probability: float = 0.66
    initial_high_arm: str = "right"
    mode_to_arm: Tuple[int, ...] = (0, 1)

    def __post_init__(self):
        for name in ("p_high", "p_low", "switch_probability"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if self.p_high < self.p_low:
            raise ValueError("p_high must be >= p_low")
        for name in ("block_length", "blocks_per_session"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_high_arm not in ("left", "right", "random"):
            raise ValueError("initial_high_arm must be 'left', 'right', or 'random'")

    @property
    def session_trials(self) -> int:
        return self.block_length * self.blocks_per_session


class BlockSwitchBandit:
    """Vectorized independent sessions with synchronized trial boundaries matching supervisor's pilot."""

    def __init__(self, config: BanditConfig, num_envs: int = 1, seed: Optional[int] = None):
        self.config = config
        self.num_envs = int(num_envs)
        if HAS_NUMPY:
            self._rng = np.random.default_rng(seed)
            self._mode_to_arm = np.asarray(config.mode_to_arm, dtype=np.int64)
        else:
            self._rng = random.Random(seed)
            self._mode_to_arm = list(config.mode_to_arm)
        self.reset()

    @property
    def session_trials(self) -> int:
        return self.config.session_trials

    @property
    def done(self) -> bool:
        return self.trial_index >= self.session_trials

    @property
    def high_arm(self):
        if HAS_NUMPY:
            return self._high_arm.copy()
        return list(self._high_arm)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            if HAS_NUMPY:
                self._rng = np.random.default_rng(seed)
            else:
                self._rng = random.Random(seed)
        self.trial_index = 0
        if HAS_NUMPY:
            if self.config.initial_high_arm == "random":
                self._high_arm = self._rng.integers(0, 2, size=self.num_envs, dtype=np.int64)
            else:
                initial_arm = int(self.config.initial_high_arm == "right")
                self._high_arm = np.full(self.num_envs, initial_arm, dtype=np.int64)
            self._actual_switch = np.zeros(self.num_envs, dtype=bool)
        else:
            if self.config.initial_high_arm == "random":
                self._high_arm = [self._rng.randint(0, 1) for _ in range(self.num_envs)]
            else:
                self._high_arm = [int(self.config.initial_high_arm == "right")] * self.num_envs
            self._actual_switch = [False] * self.num_envs
        self._nominal_boundary = False

    def step(self, mode_indices):
        if self.done:
            raise RuntimeError("Session is complete; call reset() before step()")
        if HAS_NUMPY:
            modes = np.asarray(mode_indices)
            chosen_arm = self._mode_to_arm[modes]
            reward_prob = np.where(chosen_arm == self._high_arm, self.config.p_high, self.config.p_low)
            lottery = self._rng.random(self.num_envs)
            if np.isclose(self.config.p_high + self.config.p_low, 1.0, atol=1e-12):
                winning_arm = np.where(lottery < self.config.p_high, self._high_arm, 1 - self._high_arm)
                reward = (chosen_arm == winning_arm).astype(np.float32)
            else:
                reward = (lottery < reward_prob).astype(np.float32)

            metadata = {
                "trial": np.full(self.num_envs, self.trial_index, dtype=np.int64),
                "high_arm": self._high_arm.copy(),
                "chosen_arm": chosen_arm.copy(),
                "block_index": np.full(self.num_envs, self.trial_index // self.config.block_length, dtype=np.int64),
                "trial_in_block": np.full(self.num_envs, self.trial_index % self.config.block_length, dtype=np.int64),
                "actual_switch": self._actual_switch.copy(),
                "nominal_boundary": np.full(self.num_envs, self._nominal_boundary, dtype=bool),
                "reward_probability": reward_prob,
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
        else:
            raise NotImplementedError("NumPy is required for vectorized BlockSwitchBandit.")


# ==============================================================================
# 2. High-Level RNN Architecture & Loss Functions
# ==============================================================================

if HAS_TORCH:
    # Attempt import from existing workspace codebase, or use identical definition
    try:
        from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import (
            HighLevelContextRNN_Learner as WorkspaceHighLevelRNN,
            high_level_policy_loss as workspace_policy_loss,
            q_loss as workspace_q_loss,
        )
        HAS_WORKSPACE_MODULE = True
    except (ImportError, ModuleNotFoundError):
        HAS_WORKSPACE_MODULE = False


    class HighLevelContextRNN(nn.Module):
        """Unified High-Level RNN for Contextual Bandit Reversal Learning.

        Directly implements SimGaY §16, §18, supporting both Q-learning and
        Policy Gradient formulations.
        """

        def __init__(
            self,
            K: int = 4,
            d_H: int = 16,
            tau: float = 1.0,
            is_policy: bool = False,
            deterministic: bool = False,
            include_choice_input: bool = False,
            mixed_selectivity: bool = False,
            cell_type: str = "rnn",
        ):
            super().__init__()
            self.K = K
            self.d_H = d_H
            self.tau = tau
            self.is_policy = is_policy
            self.deterministic = deterministic
            self.include_choice_input = include_choice_input
            self.mixed_selectivity = mixed_selectivity
            self.cell_type = cell_type.lower()

            # Input dimension:
            # - If mixed_selectivity: z_prev * r (K dims) + z_prev * (1 - r) (K dims) = 2*K
            # - Baseline: z_prev (K) + reward (1) = K + 1
            # - Optionally + previous choice one-hot (2 dims)
            base_dim = (2 * K) if mixed_selectivity else (K + 1)
            input_dim = base_dim + (2 if include_choice_input else 0)

            if self.cell_type == "gru":
                self.rnn_cell = nn.GRUCell(input_size=input_dim, hidden_size=d_H)
            else:
                self.rnn_cell = nn.RNNCell(input_size=input_dim, hidden_size=d_H, nonlinearity="tanh")
            self.head = nn.Linear(d_H, K)

        def sample_mode(self, outputs: Tensor) -> Tuple[Tensor, Tensor]:
            """Sample mode z from Q-scores (via Boltzmann) or policy logits."""
            if self.is_policy:
                log_pi = F.log_softmax(outputs, dim=-1)
            else:
                log_pi = F.log_softmax(outputs / self.tau, dim=-1)

            if self.deterministic:
                z_index = log_pi.argmax(dim=-1)
            else:
                z_index = torch.distributions.Categorical(logits=log_pi).sample()

            z_onehot = F.one_hot(z_index, num_classes=self.K).float()
            return z_onehot, z_index

        def update_state(
            self,
            outcome_mask: Tensor,
            prev_trial_reward: Tensor,
            h_high: Tensor,
            z_prev: Tensor,
            chosen_arm: Optional[Tensor] = None,
        ) -> Tensor:
            """Recurrent state update at trial outcome events."""
            if prev_trial_reward.dim() == 1:
                r_col = prev_trial_reward.unsqueeze(-1)
            else:
                r_col = prev_trial_reward

            if self.mixed_selectivity:
                # Conjunction of choice and outcome (mixed selectivity):
                # z * r (won with chosen mode), z * (1 - r) (lost with chosen mode)
                inputs = [z_prev * r_col, z_prev * (1.0 - r_col)]
            else:
                inputs = [z_prev, r_col]

            if self.include_choice_input and chosen_arm is not None:
                # Arm choice 1 -> [1, 0], Arm choice 2 -> [0, 1]
                choice_idx = torch.clamp(chosen_arm.long() - 1, 0, 1)
                choice_onehot = F.one_hot(choice_idx, num_classes=2).float()
                inputs.append(choice_onehot)

            rnn_in = torch.cat(inputs, dim=-1)
            h_candidate = self.rnn_cell(rnn_in, h_high)
            h_new = torch.where(outcome_mask[:, None], h_candidate, h_high)
            return h_new

        def forward(
            self,
            outcome_mask: Tensor,
            prev_trial_reward: Tensor,
            h_high: Tensor,
            z_prev: Tensor,
            chosen_arm: Optional[Tensor] = None,
        ):
            # 1. Recurrent state update
            h_high_new = self.update_state(outcome_mask, prev_trial_reward, h_high, z_prev, chosen_arm)

            # 2. Score estimation (Q-values or Policy Logits)
            scores = self.head(h_high_new)

            # 3. Sample new mode
            z_candidate, z_candidate_index = self.sample_mode(scores)
            z_new = torch.where(outcome_mask[:, None], z_candidate, z_prev)

            z_index_prev = z_prev.argmax(dim=-1)
            z_index_new = torch.where(outcome_mask, z_candidate_index, z_index_prev)

            new_state = {
                "high_level_h": h_high_new,
                "current_z": z_new,
                "current_z_index": z_index_new,
            }

            return h_high_new, z_new, scores, new_state


    def discrete_q_loss(
        scores: Tensor,
        z_indices: Tensor,
        rewards: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Mean squared error between predicted Q-value for chosen mode and reward."""
        q_selected = scores.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(-1)
        error = (q_selected - rewards) ** 2
        if mask is not None:
            mask_f = mask.float()
            return (error * mask_f).sum() / (mask_f.sum() + 1e-8)
        return error.mean()


    def discrete_policy_loss(
        logits: Tensor,
        z_indices: Tensor,
        rewards: Tensor,
        mask: Optional[Tensor] = None,
        entropy_coef: float = 0.05,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Policy gradient loss with baseline and entropy regularization."""
        if mask is None:
            mask = torch.ones_like(rewards, dtype=torch.bool)
        mask_f = mask.float()
        valid = mask_f.sum() + 1e-8

        log_pi_all = F.log_softmax(logits, dim=-1)
        pi_all = torch.exp(log_pi_all)

        log_pi_selected = log_pi_all.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(-1)
        entropy = -(pi_all * log_pi_all).sum(dim=-1)

        baseline = (rewards * mask_f).sum() / valid
        advantage = (rewards - baseline).detach()

        pg_loss = -(advantage * log_pi_selected)
        total_loss = pg_loss - (entropy_coef * entropy)
        final_loss = (total_loss * mask_f).sum() / valid

        metrics = {
            "baseline": baseline.item(),
            "advantage": (advantage * mask_f).sum().item() / valid.item(),
            "entropy": (entropy * mask_f).sum().item() / valid.item(),
        }
        return final_loss, metrics


# ==============================================================================
# 3. Mode-to-Choice Mapping (Simulating Discrete Action Execution)
# ==============================================================================

class ModeActionMapper:
    """Maps high-level mode z into discrete arm choice (Right=1, Left=2).

    Supports:
    - K=2: Mode 0 -> Right, Mode 1 -> Left.
    - K=4: Modes {0, 1} -> Right, Modes {2, 3} -> Left (overcomplete mode set).
    - Optional execution noise (reliability) to simulate low-level execution noise.
    """

    def __init__(self, K: int = 4, reliability: float = 1.0, seed: Optional[int] = None):
        self.K = K
        self.reliability = reliability
        self.rng = random.Random(seed)

        # Default mapping partition
        if K == 2:
            self.mapping = {0: DiscreteYMazeBanditEnv.ARM_RIGHT, 1: DiscreteYMazeBanditEnv.ARM_LEFT}
        elif K == 4:
            self.mapping = {
                0: DiscreteYMazeBanditEnv.ARM_RIGHT,
                1: DiscreteYMazeBanditEnv.ARM_RIGHT,
                2: DiscreteYMazeBanditEnv.ARM_LEFT,
                3: DiscreteYMazeBanditEnv.ARM_LEFT,
            }
        else:
            # General even split
            half = K // 2
            self.mapping = {
                k: (DiscreteYMazeBanditEnv.ARM_RIGHT if k < half else DiscreteYMazeBanditEnv.ARM_LEFT)
                for k in range(K)
            }

    def mode_to_choice(self, mode_idx: int) -> int:
        intended_arm = self.mapping.get(mode_idx, DiscreteYMazeBanditEnv.ARM_RIGHT)
        if self.reliability < 1.0 and self.rng.random() > self.reliability:
            # Execution error / exploration slip
            return (
                DiscreteYMazeBanditEnv.ARM_LEFT
                if intended_arm == DiscreteYMazeBanditEnv.ARM_RIGHT
                else DiscreteYMazeBanditEnv.ARM_RIGHT
            )
        return intended_arm


# ==============================================================================
# 4. Discrete RL Test Runner & Efficiency Evaluator
# ==============================================================================

class HighLevelRNNTester:
    """Runs discrete RL training and evaluation for the High-Level RNN."""

    def __init__(
        self,
        K: int = 4,
        d_H: int = 16,
        tau: float = 1.0,
        lr: float = 1e-3,
        is_policy: bool = False,
        deterministic: bool = False,
        include_choice_input: bool = False,
        mixed_selectivity: bool = False,
        cell_type: str = "rnn",
        num_envs: int = 1,
        trials_per_block: int = 15,
        p_reward_high: float = 0.80,
        p_reward_low: float = 0.20,
        p_switch: float = 0.66,
        reliability: float = 1.0,
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)

        self.K = K
        self.d_H = d_H
        self.tau = tau
        self.is_policy = is_policy
        self.deterministic = deterministic
        self.trials_per_block = trials_per_block
        self.num_envs = max(1, num_envs)
        self.mixed_selectivity = mixed_selectivity
        self.cell_type = cell_type

        self.env = DiscreteYMazeBanditEnv(
            trials_per_block=trials_per_block,
            p_reward_high=p_reward_high,
            p_reward_low=p_reward_low,
            p_switch=p_switch,
            seed=seed,
        )

        self.mapper = ModeActionMapper(K=K, reliability=reliability, seed=seed)

        if self.K == 2:
            mode_to_arm = (0, 1)
        else:
            half = self.K // 2
            mode_to_arm = tuple([0] * half + [1] * (self.K - half))

        self.task = BanditConfig(
            p_high=p_reward_high,
            p_low=p_reward_low,
            block_length=trials_per_block,
            switch_probability=p_switch,
            mode_to_arm=mode_to_arm,
        )

        if HAS_TORCH:
            self.model = HighLevelContextRNN(
                K=K,
                d_H=d_H,
                tau=tau,
                is_policy=is_policy,
                deterministic=deterministic,
                include_choice_input=include_choice_input,
                mixed_selectivity=mixed_selectivity,
                cell_type=cell_type,
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = None
            self.optimizer = None

    def run_simulation(
        self,
        episodes: int = 50,
        blocks_per_episode: int = 10,
        bptt_len: int = 15,
        verbose: bool = True,
        eval_sessions: int = 128,
        test_seed: int = 2_000_007,
        eval_tau: Optional[float] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Run training simulation over multiple multi-block episodes.

        Returns comprehensive telemetry and efficiency metrics.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to run the neural High-Level RNN simulation.")

        trials_per_episode = blocks_per_episode * self.trials_per_block
        history_log = []
        num_envs = self.num_envs

        envs = [
            DiscreteYMazeBanditEnv(
                trials_per_block=self.trials_per_block,
                p_reward_high=self.env.p_reward_high,
                p_reward_low=self.env.p_reward_low,
                p_switch=self.env.p_switch,
                seed=(self.seed + e * 1000) if self.seed is not None else None,
            )
            for e in range(num_envs)
        ]

        # Global metrics accumulators
        switch_adaptation = defaultdict(list)  # trial_since_switch -> list of is_high_arm (0 or 1)
        wsls_stats = {"win_stay": 0, "win_total": 0, "lose_shift": 0, "lose_total": 0}
        mode_usage = defaultdict(int)

        if verbose:
            print("=" * 78)
            print("HIGH-LEVEL RNN DISCRETE RL TEST RUN")
            print("=" * 78)
            print(f"Task: 2-Armed Probabilistic Reversal Bandit ({self.env.p_reward_high*100:.0f}% / {self.env.p_reward_low*100:.0f}%)")
            print(f"Block Length: {self.trials_per_block} trials | Switch Probability: {self.env.p_switch*100:.0f}% per block")
            print(f"High-Level Model: d_H={self.d_H}, K={self.K}, Cell={self.cell_type.upper()}, Mode={('Policy-Gradient' if self.is_policy else 'Q-Learning')}, Tau={self.tau}")
            print(f"Features: Mixed-Selectivity (Action x Outcome) = {self.mixed_selectivity} | Choice Input = {self.model.include_choice_input if self.model else False}")
            print(f"Vectorized Environments: {num_envs} parallel envs | Episode: {blocks_per_episode} blocks ({trials_per_episode} trials) x {episodes} episodes")
            print("-" * 78)
            print(f"{'Ep':>3} | {'Reward Rate':>11} | {'% High Arm':>10} | {'Flexibility':>11} | {'Loss':>8} | {'Entropy':>7} | {'Q-Gap':>7}")
            print("-" * 78)

        for ep in range(episodes):
            for env in envs:
                env.reset()
            h = torch.zeros(num_envs, self.d_H)
            z_prev = torch.zeros(num_envs, self.K)
            z_prev[:, 0] = 1.0  # Initial default mode
            prev_choices = torch.ones(num_envs)
            prev_rewards = torch.zeros(num_envs)

            ep_rewards = []
            ep_high_arm = []
            ep_losses = []
            ep_entropies = []
            ep_q_gaps = []

            # Unroll buffers for BPTT
            buffer_scores = []
            buffer_z = []
            buffer_rewards = []

            for t in range(trials_per_episode):
                # 1. Model step
                mask = torch.ones(num_envs, dtype=torch.bool)
                h, z_new, scores, _ = self.model(mask, prev_rewards, h, z_prev, prev_choices)

                mode_indices = z_new.argmax(dim=-1)
                for m in mode_indices.tolist():
                    mode_usage[m] += 1

                # Telemetry: Entropy & Q-gap
                if self.is_policy:
                    probs = F.softmax(scores, dim=-1)
                else:
                    probs = F.softmax(scores / self.tau, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
                ep_entropies.append(entropy)

                if self.K >= 2:
                    top2 = scores.topk(k=2, dim=-1).values
                    q_gap = (top2[:, 0] - top2[:, 1]).mean().item()
                else:
                    q_gap = 0.0
                ep_q_gaps.append(q_gap)

                # 2. Map mode to choice and step environments in parallel
                step_choices = [self.mapper.mode_to_choice(m) for m in mode_indices.tolist()]
                step_rewards = []
                for e in range(num_envs):
                    c_e = step_choices[e]
                    _, r_e, _, info_e = envs[e].step(c_e)
                    step_rewards.append(r_e)

                    # Track WSLS
                    if t > 0:
                        stayed = (c_e == int(prev_choices[e].item()))
                        if prev_rewards[e].item() > 0.0:
                            wsls_stats["win_total"] += 1
                            if stayed:
                                wsls_stats["win_stay"] += 1
                        else:
                            wsls_stats["lose_total"] += 1
                            if not stayed:
                                wsls_stats["lose_shift"] += 1

                    # Track reversal adaptation
                    dist = info_e["trials_since_switch"]
                    if dist < self.trials_per_block:
                        switch_adaptation[dist].append(1.0 if info_e["is_high_arm"] else 0.0)

                    # Accumulate episode metrics
                    ep_rewards.append(r_e)
                    ep_high_arm.append(1.0 if info_e["is_high_arm"] else 0.0)

                r_tensor = torch.tensor(step_rewards, dtype=torch.float32)

                # Store for gradient update
                buffer_scores.append(scores)
                buffer_z.append(mode_indices)
                buffer_rewards.append(r_tensor)

                # Update state for next step
                z_prev = z_new
                prev_choices = torch.tensor(step_choices, dtype=torch.float32)
                prev_rewards = r_tensor

                # 3. Recurrent BPTT update
                if (t + 1) % bptt_len == 0 or t == trials_per_episode - 1:
                    batch_scores = torch.cat(buffer_scores, dim=0)
                    batch_z = torch.cat(buffer_z, dim=0)
                    batch_rewards = torch.cat(buffer_rewards, dim=0)

                    if self.is_policy:
                        loss, _ = discrete_policy_loss(batch_scores, batch_z, batch_rewards)
                    else:
                        loss = discrete_q_loss(batch_scores, batch_z, batch_rewards)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    ep_losses.append(loss.item())

                    # Detach hidden state at BPTT boundary
                    h = h.detach()
                    buffer_scores.clear()
                    buffer_z.clear()
                    buffer_rewards.clear()

            # Episode summary
            mean_rew = sum(ep_rewards) / len(ep_rewards)
            mean_high = (sum(ep_high_arm) / len(ep_high_arm)) * 100.0
            flex_idx = sum(e.get_flexibility_index() for e in envs) / num_envs
            mean_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
            mean_ent = sum(ep_entropies) / len(ep_entropies)
            mean_gap = sum(ep_q_gaps) / len(ep_q_gaps)

            history_log.append({
                "episode": ep + 1,
                "reward_rate": mean_rew,
                "pct_high_arm": mean_high,
                "flexibility": flex_idx,
                "loss": mean_loss,
                "entropy": mean_ent,
                "q_gap": mean_gap,
            })

            # Print progress periodically or on early episodes
            should_print = (
                ep in (0, 1, 4, 9)
                or (ep + 1) % max(1, episodes // 10) == 0
                or ep == episodes - 1
            )
            if verbose and should_print:
                print(
                    f"{ep+1:3d} | {mean_rew:11.3f} | {mean_high:9.1f}% | "
                    f"{flex_idx:11.3f} | {mean_loss:8.4f} | {mean_ent:7.3f} | {mean_gap:7.3f}"
                )

        # Compute aggregate final efficiency metrics
        last_n = max(1, episodes // 5)
        recent_log = history_log[-last_n:]
        avg_recent_high = sum(x["pct_high_arm"] for x in recent_log) / len(recent_log)
        avg_recent_rew = sum(x["reward_rate"] for x in recent_log) / len(recent_log)
        avg_recent_flex = sum(x["flexibility"] for x in recent_log) / len(recent_log)

        win_stay_rate = (wsls_stats["win_stay"] / wsls_stats["win_total"]) if wsls_stats["win_total"] > 0 else 0.0
        lose_shift_rate = (wsls_stats["lose_shift"] / wsls_stats["lose_total"]) if wsls_stats["lose_total"] > 0 else 0.0

        adaptation_curve = {
            dist: (sum(vals) / len(vals))
            for dist, vals in sorted(switch_adaptation.items())
            if len(vals) > 0
        }

        # Run supervisor's exact held-out frozen evaluation suite
        supervisor_eval = None
        if HAS_TORCH and HAS_NUMPY:
            supervisor_eval = self.evaluate_supervisor_protocol(
                num_sessions=eval_sessions,
                test_seed=test_seed,
                eval_tau=eval_tau,
                output=output_dir,
                verbose=verbose,
            )

        results = {
            "history": history_log,
            "final_pct_high_arm": avg_recent_high,
            "final_reward_rate": avg_recent_rew,
            "final_flexibility": avg_recent_flex,
            "win_stay_rate": win_stay_rate,
            "lose_shift_rate": lose_shift_rate,
            "adaptation_curve": adaptation_curve,
            "mode_usage": dict(mode_usage),
            "supervisor_evaluation": supervisor_eval,
        }

        if verbose:
            self._print_efficiency_report(results)

        return results

    def evaluate_supervisor_protocol(
        self,
        num_sessions: int = 128,
        test_seed: int = 2_000_007,
        eval_tau: Optional[float] = None,
        output: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """Run supervisor's exact frozen held-out evaluation suite across 6 policies."""
        if not HAS_TORCH or not HAS_NUMPY:
            raise RuntimeError("PyTorch and NumPy are required for the evaluation suite.")

        out_path = Path(output) if output else None
        results = evaluation_suite(
            model=self.model,
            task=self.task,
            num_sessions=num_sessions,
            seed=test_seed,
            eval_tau=eval_tau,
            output=out_path,
        )
        if verbose:
            print_supervisor_benchmark_report(results, self.task, model=self.model, eval_tau=eval_tau)
        return results

    def _print_efficiency_report(self, results: Dict):
        """Prints a comprehensive diagnostic and efficiency summary."""
        print("-" * 78)
        print("EFFICIENCY & ADAPTATION SUMMARY REPORT (Online Training Trajectory)")
        print("-" * 78)
        print(f"Final High-Reward Arm Choice: {results['final_pct_high_arm']:.1f}%  (Chance = 50%, Optimal ~85-90%)")
        print(f"Final Mean Reward Rate:       {results['final_reward_rate']:.3f}   (Ceiling = 0.80 minus switch costs)")
        print(f"Final Flexibility Index:      {results['final_flexibility']:.3f}   (Target > 0.60; ymaze_80_final metric)")
        print(f"Win-Stay Rate:                {results['win_stay_rate']*100:.1f}%")
        print(f"Lose-Shift Rate:              {results['lose_shift_rate']*100:.1f}%")
        print()
        print("Reversal Adaptation Curve (P(High Arm) vs. Trials Since Block Switch):")
        curve = results["adaptation_curve"]
        for trial_idx in range(self.trials_per_block):
            val = curve.get(trial_idx, 0.0) * 100.0
            bar = "#" * int(val / 2.5)
            print(f"  Trial {trial_idx+1:2d}: {val:5.1f}% | {bar}")
        print("-" * 78)


# ==============================================================================
# 4B. Supervisor's Exact Evaluation Protocol (Held-Out Benchmark Suite)
# ==============================================================================

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict], task: BanditConfig, num_sessions: int) -> Dict:
    def mean(field, subset=None):
        values = [r[field] for r in (rows if subset is None else subset) if r.get(field) is not None]
        return float(np.mean(values)) if values else None

    def curve(field, limit):
        result = []
        for lag in range(limit):
            subset = [r for r in rows if r[field] == lag]
            result.append(
                {
                    "lag": lag,
                    "n": len(subset),
                    "high_arm_accuracy": mean("high_arm_choice", subset),
                    "reward_rate": mean("reward", subset),
                }
            )
        return result

    stable = [r for r in rows if r["trial_in_block"] >= task.block_length // 2]
    omissions = [r for r in rows if r["same_context_omission"]]
    sessions = []
    for session in range(num_sessions):
        subset = [r for r in rows if r["session"] == session]
        sessions.append(
            {
                "session": session,
                "high_arm_accuracy": mean("high_arm_choice", subset),
                "reward_rate": mean("reward", subset),
            }
        )
    return {
        "num_sessions": num_sessions,
        "completed_trials": len(rows),
        "actual_switches": sum(1 for r in rows if r["actual_switch"]),
        "high_arm_accuracy": mean("high_arm_choice"),
        "reward_rate": mean("reward"),
        "expected_regret_per_trial": mean("expected_regret"),
        "stable_half_block_accuracy": mean("high_arm_choice", stable),
        "mode0_fraction": float(np.mean([r["mode"] == 0 for r in rows])) if HAS_NUMPY else 0.5,
        "mode_entropy": mean("mode_entropy"),
        "arm_change_after_omission_same_context": mean("changed_arm", omissions),
        "actual_switch_curve": curve("since_actual_switch", task.block_length),
        "nominal_block_curve": curve("trial_in_block", task.block_length),
        "sessions": sessions,
    }


if HAS_TORCH:
    def initial_hidden(model: nn.Module, num_envs: int) -> Tensor:
        """Match the full core's initial trigger, with no completed trial to credit."""
        parameter = next(model.parameters())
        h = parameter.new_zeros(num_envs, model.d_H)
        z = parameter.new_zeros(num_envs, model.K)
        event = torch.ones(num_envs, dtype=torch.bool, device=parameter.device)
        with torch.no_grad():
            return model.update_state(event, parameter.new_zeros(num_envs), h, z)

    def policy_probabilities(model: nn.Module, scores: Tensor, eval_tau: Optional[float] = None) -> Tensor:
        """Returns action probabilities, respecting is_policy and temperature settings."""
        if model.is_policy:
            return F.softmax(scores, dim=-1)
        tau = eval_tau if eval_tau is not None else getattr(model, "tau", 1.0)
        return F.softmax(scores / tau, dim=-1)

    @torch.no_grad()
    def evaluate(
        model: nn.Module,
        task: BanditConfig,
        num_sessions: int,
        seed: int,
        policy: str = "sampled",
        reset_memory: bool = False,
        eval_tau: Optional[float] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """Frozen independent sessions; hidden task metadata only enters metrics.

        All policies get the exact same environmental random stream. Sampled and greedy
        are separate closed-loop runs, since each induces its own reward history.
        """
        env = BlockSwitchBandit(task, num_sessions, seed)
        device = next(model.parameters()).device
        event = torch.ones(num_sessions, dtype=torch.bool, device=device)
        h = initial_hidden(model, num_sessions)
        rng = np.random.default_rng(seed + 1)
        mode_to_arm = np.asarray(task.mode_to_arm)
        arm_to_mode = np.array([np.where(mode_to_arm == a)[0][0] for a in range(2)])
        previous_arm = rng.integers(0, 2, num_sessions)
        previous_reward = np.ones(num_sessions)
        previous_high = np.full(num_sessions, -1)
        switch_age = np.full(num_sessions, -1)
        rows = []
        was_training = model.training
        model.eval()

        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed + 2)
                for trial in range(env.session_trials):
                    if reset_memory:
                        h = initial_hidden(model, num_sessions)
                    scores = model.head(h)
                    probabilities = policy_probabilities(model, scores, eval_tau=eval_tau)
                    if policy == "sampled":
                        choices = torch.distributions.Categorical(probs=probabilities).sample().cpu().numpy()
                    elif policy == "greedy":
                        choices = scores.argmax(-1).cpu().numpy()
                    elif policy == "random":
                        choices = rng.integers(0, model.K, num_sessions)
                    elif policy == "win_stay_lose_shift":
                        choices = arm_to_mode[np.where(previous_reward > 0, previous_arm, 1 - previous_arm)]
                    elif policy == "oracle":
                        choices = arm_to_mode[env.high_arm]
                    else:
                        raise ValueError(f"Unknown evaluation policy: {policy}")

                    reward, info = env.step(choices)
                    switch_age = np.where(info["actual_switch"], 0, np.where(switch_age >= 0, switch_age + 1, -1))
                    probs = probabilities.cpu().numpy()
                    values = scores.cpu().numpy()
                    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).cpu().numpy()

                    for index in range(num_sessions):
                        high = int(info["high_arm"][index])
                        chosen = int(info["chosen_arm"][index])
                        rows.append(
                            {
                                "session": index,
                                "trial": trial,
                                "block_index": int(info["block_index"][index]),
                                "trial_in_block": int(info["trial_in_block"][index]),
                                "nominal_boundary": bool(info["nominal_boundary"][index]),
                                "actual_switch": bool(info["actual_switch"][index]),
                                "since_actual_switch": int(switch_age[index]),
                                "mode": int(choices[index]),
                                "chosen_arm": chosen,
                                "high_arm": high,
                                "reward": float(reward[index]),
                                "high_arm_choice": int(chosen == high),
                                "expected_regret": float(task.p_high - info["reward_probability"][index]),
                                "mode0_probability": float(probs[index, 0]) if policy in ("sampled", "greedy") else None,
                                "score0": float(values[index, 0]) if policy in ("sampled", "greedy") else None,
                                "score1": float(values[index, 1]) if policy in ("sampled", "greedy") and model.K > 1 else None,
                                "mode_entropy": float(entropy[index]) if policy in ("sampled", "greedy") else None,
                                "same_context_omission": bool(
                                    trial > 0 and previous_reward[index] == 0 and previous_high[index] == high
                                ),
                                "changed_arm": bool(trial > 0 and chosen != previous_arm[index]),
                            }
                        )

                    z = F.one_hot(torch.as_tensor(choices, device=device), num_classes=model.K).float()
                    h = model.update_state(event, torch.as_tensor(reward, device=device), h, z)
                    previous_arm = info["chosen_arm"]
                    previous_reward = reward
                    previous_high = info["high_arm"]
        finally:
            model.train(was_training)

        return summarize(rows, task, num_sessions), rows

    def evaluation_suite(
        model: nn.Module,
        task: BanditConfig,
        num_sessions: int = 128,
        seed: int = 2_000_007,
        eval_tau: Optional[float] = None,
        output: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """Runs the supervisor's exact frozen benchmark evaluation suite across 6 policies."""
        result = {}
        for policy, reset in [
            ("sampled", False),
            ("greedy", False),
            ("greedy_memory_reset", True),
            ("random", False),
            ("win_stay_lose_shift", False),
            ("oracle", False),
        ]:
            summary, rows = evaluate(
                model=model,
                task=task,
                num_sessions=num_sessions,
                seed=seed,
                policy="greedy" if policy == "greedy_memory_reset" else policy,
                reset_memory=reset,
                eval_tau=eval_tau,
            )
            result[policy] = summary
            if output is not None and HAS_NUMPY:
                out_p = Path(output)
                out_p.mkdir(parents=True, exist_ok=True)
                write_csv(out_p / f"{policy}_trials.csv", rows)

        if output is not None:
            out_p = Path(output)
            out_p.mkdir(parents=True, exist_ok=True)
            write_json(out_p / "evaluation.json", result)

        return result
else:
    def initial_hidden(*args, **kwargs):
        raise RuntimeError("PyTorch is required for initial_hidden.")

    def policy_probabilities(*args, **kwargs):
        raise RuntimeError("PyTorch is required for policy_probabilities.")

    def evaluate(*args, **kwargs):
        raise RuntimeError("PyTorch is required for evaluate.")

    def evaluation_suite(*args, **kwargs):
        raise RuntimeError("PyTorch is required for evaluation_suite.")


def print_supervisor_benchmark_report(
    eval_results: Dict[str, Dict],
    task: BanditConfig,
    model: Optional[nn.Module] = None,
    eval_tau: Optional[float] = None,
):
    """Renders the comprehensive benchmark comparison report and reversal curves."""
    print()
    print("=" * 86)
    print("SUPERVISOR FROZEN EVALUATION BENCHMARK SUITE (Exact Held-Out Protocol)")
    print("=" * 86)
    num_sessions = eval_results["sampled"]["num_sessions"]
    switches = eval_results["sampled"]["actual_switches"]
    trials = eval_results["sampled"]["completed_trials"]
    print(f"Sessions: {num_sessions} | Completed Trials: {trials} | Actual Reversals Experienced: {switches}")
    if model is not None:
        mode_str = "Policy-Gradient" if model.is_policy else f"Q-Learning (tau={eval_tau or model.tau})"
        print(f"Model Configuration: K={model.K}, d_H={model.d_H}, Mode={mode_str}, Mixed-Selectivity={model.mixed_selectivity}")
    print("-" * 86)
    print(f"{'Evaluation Policy':<28} | {'Better-Arm %':>12} | {'Reward Rate':>11} | {'Regret':>8} | {'Stable Acc':>10} | {'Mode 0 %':>8}")
    print("-" * 86)

    display_names = [
        ("sampled", "Sampled (Exploration)"),
        ("greedy", "Greedy (Exploitation)"),
        ("greedy_memory_reset", "Greedy (Memory Reset)"),
        ("random", "Random (50/50 Baseline)"),
        ("win_stay_lose_shift", "Win-Stay/Lose-Shift"),
        ("oracle", "Oracle (Upper Bound)"),
    ]

    for key, label in display_names:
        res = eval_results.get(key)
        if not res:
            continue
        acc = f"{res['high_arm_accuracy'] * 100:.1f}%" if res['high_arm_accuracy'] is not None else "N/A"
        rew = f"{res['reward_rate']:.3f}" if res['reward_rate'] is not None else "N/A"
        regret = f"{res['expected_regret_per_trial']:.3f}" if res['expected_regret_per_trial'] is not None else "N/A"
        stable = f"{res['stable_half_block_accuracy'] * 100:.1f}%" if res['stable_half_block_accuracy'] is not None else "N/A"
        m0 = f"{res['mode0_fraction'] * 100:.1f}%" if res['mode0_fraction'] is not None else "N/A"
        print(f"{label:<28} | {acc:>12} | {rew:>11} | {regret:>8} | {stable:>10} | {m0:>8}")

    print("-" * 86)
    print()
    print("Reversal Adaptation Curve (P(Better Arm) vs. Lags Post-Actual Switch):")
    print(f"{'Lag':>4} | {'Trial':>5} | {'Sampled':>9} | {'Greedy':>9} | {'WSLS':>9} | {'Greedy Adaptation Visual Profile'}")
    print("-" * 86)

    sampled_curve = {c["lag"]: c["high_arm_accuracy"] for c in eval_results["sampled"]["actual_switch_curve"]}
    greedy_curve = {c["lag"]: c["high_arm_accuracy"] for c in eval_results["greedy"]["actual_switch_curve"]}
    wsls_curve = {c["lag"]: c["high_arm_accuracy"] for c in eval_results["win_stay_lose_shift"]["actual_switch_curve"]}

    for lag in range(task.block_length):
        s_val = sampled_curve.get(lag)
        g_val = greedy_curve.get(lag)
        w_val = wsls_curve.get(lag)
        s_str = f"{s_val*100:6.1f}%" if s_val is not None else "  N/A  "
        g_str = f"{g_val*100:6.1f}%" if g_val is not None else "  N/A  "
        w_str = f"{w_val*100:6.1f}%" if w_val is not None else "  N/A  "

        bar_len = int((g_val or 0.0) * 30)
        bar = "#" * bar_len
        marker = " (Perseveration)" if lag == 0 else (" (Crossover!)" if (g_val or 0.0) >= 0.5 and (greedy_curve.get(lag-1, 0.0) or 0.0) < 0.5 else "")
        print(f"{lag:4d} | {lag+1:5d} | {s_str:>9} | {g_str:>9} | {w_str:>9} | {bar:<30}{marker}")

    print("-" * 86)


# ==============================================================================
# 5. Unit Tests
# ==============================================================================

import unittest


class TestDiscreteHighLevelRNN(unittest.TestCase):
    """Automated unit test suite verifying components and integration."""

    def test_01_environment_lottery_statistics(self):
        """Verify 80/20 lottery and 66% reversal probabilities empirically."""
        env = DiscreteYMazeBanditEnv(trials_per_block=15, p_reward_high=0.80, p_reward_low=0.20, p_switch=0.66, seed=123)
        obs = env.reset(start_arm=DiscreteYMazeBanditEnv.ARM_RIGHT)

        high_arm_rewards = []
        low_arm_rewards = []
        for _ in range(500):
            high_arm = env.current_high_arm
            # Step high arm
            _, r_high, _, _ = env.step(high_arm)
            high_arm_rewards.append(r_high)
            # Step opposite (low) arm
            low_arm = DiscreteYMazeBanditEnv.ARM_LEFT if env.current_high_arm == DiscreteYMazeBanditEnv.ARM_RIGHT else DiscreteYMazeBanditEnv.ARM_RIGHT
            _, r_low, _, _ = env.step(low_arm)
            low_arm_rewards.append(r_low)

        p_high = sum(high_arm_rewards) / len(high_arm_rewards)
        p_low = sum(low_arm_rewards) / len(low_arm_rewards)

        self.assertAlmostEqual(p_high, 0.80, delta=0.06, msg="High arm reward deviates from 80%")
        self.assertAlmostEqual(p_low, 0.20, delta=0.06, msg="Low arm reward deviates from 20%")

    def test_02_block_switching_contingency(self):
        """Verify that blocks switch contingencies with approximately 66% frequency."""
        env = DiscreteYMazeBanditEnv(trials_per_block=5, p_switch=0.66, seed=456)
        env.reset()

        switches = 0
        total_blocks = 200
        for _ in range(total_blocks * 5):
            _, _, _, info = env.step(DiscreteYMazeBanditEnv.ARM_RIGHT)
            if env.contingency_switched:
                switches += 1

        empirical_switch_prob = switches / total_blocks
        self.assertAlmostEqual(empirical_switch_prob, 0.66, delta=0.08, msg="Block switch rate deviates from 66%")

    def test_03_high_level_rnn_forward_and_shapes(self):
        """Verify forward pass shapes across both Q-learning and Policy Gradient."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")

        for is_policy in (False, True):
            model = HighLevelContextRNN(K=4, d_H=16, is_policy=is_policy, include_choice_input=True)
            mask = torch.ones(2, dtype=torch.bool)
            prev_r = torch.tensor([1.0, 0.0])
            h = torch.zeros(2, 16)
            z_prev = F.one_hot(torch.tensor([0, 1]), num_classes=4).float()
            c_prev = torch.tensor([1.0, 2.0])

            h_new, z_new, scores, state = model(mask, prev_r, h, z_prev, c_prev)

            self.assertEqual(h_new.shape, (2, 16))
            self.assertEqual(z_new.shape, (2, 4))
            self.assertEqual(scores.shape, (2, 4))
            self.assertEqual(state["current_z_index"].shape, (2,))

    def test_04_gradient_flow_through_recurrent_cells(self):
        """Verify end-to-end backprop computes gradients for both W_ih and W_hh."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")

        model = HighLevelContextRNN(K=2, d_H=8, is_policy=False)
        h = torch.zeros(1, 8)
        z = F.one_hot(torch.tensor([0]), num_classes=2).float()
        r = torch.tensor([1.0])
        mask = torch.ones(1, dtype=torch.bool)

        h_next, z_next, scores, _ = model(mask, r, h, z)
        loss = discrete_q_loss(scores, torch.tensor([0]), r)

        model.zero_grad()
        loss.backward()

        self.assertIsNotNone(model.rnn_cell.weight_ih.grad)
        self.assertIsNotNone(model.rnn_cell.weight_hh.grad)
        self.assertGreater(model.rnn_cell.weight_ih.grad.abs().sum().item(), 0.0)

    def test_05_smoke_test_discrete_runner(self):
        """Quick 2-episode simulation smoke test to guarantee end-to-end operation."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")

        tester = HighLevelRNNTester(
            K=2,
            d_H=8,
            trials_per_block=5,
            p_reward_high=0.80,
            p_reward_low=0.20,
            p_switch=0.66,
            seed=7,
        )
        res = tester.run_simulation(episodes=2, blocks_per_episode=2, bptt_len=5, verbose=False)
        self.assertEqual(len(res["history"]), 2)
        self.assertIn("final_pct_high_arm", res)
        self.assertIn("adaptation_curve", res)

    def test_06_supervisor_evaluation_protocol(self):
        """Verify the supervisor's frozen evaluation suite runs across all 6 policies."""
        if not HAS_TORCH or not HAS_NUMPY:
            self.skipTest("PyTorch and NumPy required")

        model = HighLevelContextRNN(K=2, d_H=8, is_policy=True, mixed_selectivity=True)
        task = BanditConfig(block_length=5, blocks_per_session=2)
        suite = evaluation_suite(model, task, num_sessions=2, seed=100)
        self.assertEqual(len(suite), 6)
        for pol in ("sampled", "greedy", "greedy_memory_reset", "random", "win_stay_lose_shift", "oracle"):
            self.assertIn(pol, suite)
            self.assertIn("high_arm_accuracy", suite[pol])
            self.assertIn("actual_switch_curve", suite[pol])


# ==============================================================================
# 6. Command Line Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Discrete Reinforcement Learning Test Run for High-Level RNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=30, help="Number of training episodes")
    parser.add_argument("--blocks-per-episode", type=int, default=8, help="Number of 15-trial blocks per episode")
    parser.add_argument("--trials-per-block", type=int, default=15, help="Number of trials in each block")
    parser.add_argument("--p-reward-high", type=float, default=0.80, help="Reward probability for the high arm")
    parser.add_argument("--p-reward-low", type=float, default=0.20, help="Reward probability for the low arm")
    parser.add_argument("--p-switch", type=float, default=0.66, help="Switch probability per block boundary")
    parser.add_argument("--K", type=int, default=4, help="Number of high-level modes")
    parser.add_argument("--d-H", type=int, default=16, help="High-level RNN hidden state dimension")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for Boltzmann Q exploration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--bptt-len", type=int, default=15, help="BPTT unroll length in trials")
    parser.add_argument("--is-policy", action="store_true", help="Use Policy Gradient instead of Q-learning")
    parser.add_argument("--deterministic", action="store_true", help="Greedy mode selection during rollout")
    parser.add_argument("--include-choice-input", action="store_true", help="Feed explicit previous arm choice to RNN")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments for low-variance updates")
    parser.add_argument("--mixed-selectivity", action="store_true", help="Provide conjunctive (action x outcome) representation")
    parser.add_argument("--cell-type", type=str, default="rnn", choices=["rnn", "gru"], help="Recurrent cell type (rnn or gru)")
    parser.add_argument("--reliability", type=float, default=1.0, help="Low-level execution reliability (1.0 = flawless)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval-sessions", type=int, default=128, help="Number of frozen sessions for held-out evaluation suite (default: 128)")
    parser.add_argument("--test-seed", type=int, default=2_000_007, help="Seed for frozen held-out evaluation suite (default: 2000007)")
    parser.add_argument("--eval-tau", type=float, default=None, help="Evaluation-only temperature override (optional)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run the frozen evaluation suite directly")
    parser.add_argument("--eval-output", type=str, default="", help="Optional directory to save evaluation.json and trial CSVs")
    parser.add_argument("--save-plot", type=str, default="", help="Optional file path to save diagnostic figure")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("[!] PyTorch is not available in the current Python environment.")
        print("    To run the neural High-Level RNN simulation, activate your conda or virtual environment:")
        print("        conda activate sf2")
        print("        python tests/test_high_level_rnn_discrete.py")
        sys.exit(1)

    tester = HighLevelRNNTester(
        K=args.K,
        d_H=args.d_H,
        tau=args.tau,
        lr=args.lr,
        is_policy=args.is_policy,
        deterministic=args.deterministic,
        include_choice_input=args.include_choice_input,
        mixed_selectivity=args.mixed_selectivity,
        cell_type=args.cell_type,
        num_envs=args.num_envs,
        trials_per_block=args.trials_per_block,
        p_reward_high=args.p_reward_high,
        p_reward_low=args.p_reward_low,
        p_switch=args.p_switch,
        reliability=args.reliability,
        seed=args.seed,
    )

    if args.eval_only:
        tester.evaluate_supervisor_protocol(
            num_sessions=args.eval_sessions,
            test_seed=args.test_seed,
            eval_tau=args.eval_tau,
            output=args.eval_output,
            verbose=True,
        )
        return

    results = tester.run_simulation(
        episodes=args.episodes,
        blocks_per_episode=args.blocks_per_episode,
        bptt_len=args.bptt_len,
        verbose=True,
        eval_sessions=args.eval_sessions,
        test_seed=args.test_seed,
        eval_tau=args.eval_tau,
        output_dir=args.eval_output,
    )

    if args.save_plot:
        try:
            import matplotlib.pyplot as plt

            history = results["history"]
            eps = [h["episode"] for h in history]
            pct_high = [h["pct_high_arm"] for h in history]
            rew_rate = [h["reward_rate"] for h in history]
            flex = [h["flexibility"] for h in history]

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("High-Level RNN Reversal Learning Efficiency", fontsize=14, fontweight="bold")

            # 1. Choice Accuracy
            axs[0, 0].plot(eps, pct_high, color="navy", lw=2, label="% High-Reward Arm")
            axs[0, 0].axhline(50, color="gray", linestyle="--", label="Chance (50%)")
            axs[0, 0].axhline(80, color="green", linestyle=":", label="Optimum (~80%)")
            axs[0, 0].set_ylabel("% High-Reward Arm")
            axs[0, 0].set_xlabel("Episode")
            axs[0, 0].set_title("Choice Accuracy")
            axs[0, 0].legend()
            axs[0, 0].grid(True, alpha=0.3)

            # 2. Reward Rate & Flexibility
            axs[0, 1].plot(eps, rew_rate, color="teal", lw=2, label="Mean Reward")
            axs[0, 1].plot(eps, flex, color="crimson", lw=1.5, linestyle="--", label="Flexibility Index")
            axs[0, 1].set_ylabel("Rate / Index")
            axs[0, 1].set_xlabel("Episode")
            axs[0, 1].set_title("Reward Rate & Flexibility")
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)

            # 3. Reversal Adaptation Trajectory
            curve = results["adaptation_curve"]
            dists = list(range(args.trials_per_block))
            probs = [curve.get(d, 0.0) * 100.0 for d in dists]
            axs[1, 0].plot(dists, probs, marker="o", color="darkorange", lw=2)
            axs[1, 0].axhline(50, color="gray", linestyle="--")
            axs[1, 0].set_ylabel("% High-Reward Arm")
            axs[1, 0].set_xlabel("Trials Since Block Switch")
            axs[1, 0].set_title("Post-Switch Adaptation Trajectory")
            axs[1, 0].grid(True, alpha=0.3)

            # 4. Mode Usage Breakdown
            modes = sorted(results["mode_usage"].keys())
            counts = [results["mode_usage"][m] for m in modes]
            axs[1, 1].bar([f"Mode {m}" for m in modes], counts, color="purple", alpha=0.7)
            axs[1, 1].set_ylabel("Selection Count")
            axs[1, 1].set_title("Mode Selection Distribution")
            axs[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=200)
            print(f"[+] Saved diagnostic plot to: {args.save_plot}")
        except Exception as e:
            print(f"[!] Could not save plot: {e}")


if __name__ == "__main__":
    main()
