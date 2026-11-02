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

import argparse
import math
import os
import random
import sys
from collections import defaultdict
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

        results = {
            "history": history_log,
            "final_pct_high_arm": avg_recent_high,
            "final_reward_rate": avg_recent_rew,
            "final_flexibility": avg_recent_flex,
            "win_stay_rate": win_stay_rate,
            "lose_shift_rate": lose_shift_rate,
            "adaptation_curve": adaptation_curve,
            "mode_usage": dict(mode_usage),
        }

        if verbose:
            self._print_efficiency_report(results)

        return results

    def _print_efficiency_report(self, results: Dict):
        """Prints a comprehensive diagnostic and efficiency summary."""
        print("-" * 78)
        print("EFFICIENCY & ADAPTATION SUMMARY REPORT")
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

    results = tester.run_simulation(
        episodes=args.episodes,
        blocks_per_episode=args.blocks_per_episode,
        bptt_len=args.bptt_len,
        verbose=True,
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
