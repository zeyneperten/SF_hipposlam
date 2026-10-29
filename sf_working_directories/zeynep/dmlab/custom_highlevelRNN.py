# SimGaY §16 decomposition + §18 forward-pass pseudocode
# Q-style contextual bandit

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sample_factory.utils.utils import log

# HighLevelContextRNN  —  §16, §18


class HighLevelContextRNN_Stage1(nn.Module):
    """
    Stage 1: Event-driven latching ONLY.
    No learned head. No z_k sampling, random for now. Just verifies that the timing works.
    """

    def __init__(self, K: int = 4, d_H: int = 16):
        super().__init__()
        self.K = K

        # We keep the RNN just to make sure the tensor shapes and updates don't crash,
        # but we ignore its output for now.
        self.rnn_cell = nn.RNNCell(input_size=K + 1, hidden_size=d_H, nonlinearity="tanh")

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, inst_block=None):

        # 1. Background RNN update (just testing mechanics)
        rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)
        h_candidate = self.rnn_cell(rnn_in, h_high)

        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,  # update only at outcome events (at trigger)
            h_high,  # keep previous hidden state otherwise
        )

        # MODE SELECTION
        # e.g. [0,0,1,0] -> mode 2 is selected. With randomization that "1" mode is selected randomly
        if inst_block is not None:
            # If inst_block is provided, we can use it to select a mode deterministically
            # For example, if inst_block is 1, we select mode 0; if it's 2, we select mode 1, and so on.
            oracle_idx = torch.clamp(inst_block - 1, 0, 1)  # Ensure the index is within bounds
            #log.info(f"Using oracle Z.")
            z_candidate = F.one_hot(oracle_idx, num_classes=self.K).float()
        else:
            # Otherwise, we can randomly select a mode for testing purposes
            # This case Z has no correlation with where the actual reward is. Low-level policy considers z as "noise".
            B = z_prev.size(0)
            random_indices = torch.randint(0, self.K, (B,), device=z_prev.device)
            #log.info(f"Randomly selecting modes.")
            z_candidate = F.one_hot(random_indices, num_classes=self.K).float()

        # 3. Latch the new mode only at outcome events
        # Keep z_prev unless it is at trigger (outcome event) then update to z_candidate

        should_update = outcome_mask | (
            z_prev.sum(dim=-1) < 0.5
        )  # Update if it's an outcome event or if z_prev is all zeros (initial state)

        z_new = torch.where(
            should_update[:, None],
            z_candidate,
            z_prev,
        )

        # Passing the mode to decoder is handeled in the wrapper
        return h_high_new, z_new


# QHead  —  §16, §9.2


class QHead(nn.Module):
    def __init__(self, d_H: int, K: int, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.head = nn.Linear(d_H, K)

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, d_H)  →  q: (B, K)"""
        q = self.head(h)  # current estimate of expected reward for each mode | THIS IS LEARNED BY THE NETWORK
        return q  # output raw expected rewards.


# sample_mode  —  §9, §18 step 4


def sample_mode(
    scores: Tensor, tau: float = 1.0, deterministic: bool = False
) -> Tuple[Tensor, Tensor]:  # scale the raw values by temperature tau to control exploration vs exploitation
    """
    scores: (B, K)
    → z_onehot: (B, K)   one-hot e(z_k)
    → z_index:  (B,)     integer index  — §17 current_z_index
    """
    K = scores.size(-1)
    log_pi = F.log_softmax(
        scores / tau, dim=-1
    )  # convert raw Q-values to log-probabilities (distribution) for sampling

    if deterministic:
        z_index = log_pi.argmax(dim=-1)
    else:
        z_index = torch.distributions.Categorical(
            logits=log_pi
        ).sample()  # actor exploration does not depend on autograd
    z_onehot = F.one_hot(
        z_index, num_classes=K
    ).float()  # return binary one-hot vector of the sampled mode index | RNN and DECODER RECIEVE IT
    return z_onehot, z_index


# Logging helpers  —  §22


def compute_log_dict(
    scores_or_logits: Tensor,  # (B, K)  Q-values
    z_index: Tensor,  # (B,)    selected mode index
    h_high_new: Tensor,  # (B, d_H)
    reward_prev: Tensor,  # (B,)
    tau: float,
    chosen_arm: Tensor = None,  # (B,) optional, for logging
    K=4,
    is_policy: bool = False,  # if False, it does Q-learning, if True, it does policy gradient
) -> Dict[str, Tensor]:
    """
    Recommended logging quantities (computed at decision points only).
    High-level process:
        z_k, scores/Q vector, selection entropy, h_k, r_k
    Q-head specific:
        all Q_H(h_k, z), selected-Q vs reward, gap between top two Q-values
    """
    if is_policy:
        pi = F.softmax(scores_or_logits, dim=-1)  # (B, K)
    else:
        pi = F.softmax(scores_or_logits / tau, dim=-1)  # (B, K)

    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)  # (B,)  selection entropy

    selected = scores_or_logits.gather(dim=-1, index=z_index.unsqueeze(-1)).squeeze(  # (B,)
        -1
    )  # selected mode's score/Q-value

    top2 = scores_or_logits.topk(k=2, dim=-1).values  # (B, 2)
    gap = top2[:, 0] - top2[:, 1]  # (B,)  gap top1 - top2

    if is_policy:
        log_dict = {"entropy": entropy, "h": h_high_new, "logit_selected": selected, "logit_gap": gap}
    else:
        log_dict = {"entropy": entropy, "h": h_high_new, "q_selected": selected, "q_gap": gap}

    went_right = (chosen_arm == 1.0) if chosen_arm is not None else None
    went_left = (chosen_arm == 2.0) if chosen_arm is not None else None

    for k in range(K):

        if is_policy:
            log_dict[f"logit_mode{k}"] = scores_or_logits[:, k]
        else:
            log_dict[f"qval_mode{k}"] = scores_or_logits[
                :, k
            ]  # if you plot it together with reward_achieved they must overlap.

        log_dict[f"prob_mode{k}"] = pi[:, k]  # predicted probability to select this mode

        went_mode_k = z_index == k

        log_dict[f"selection_rate_mode{k}"] = (
            went_mode_k.float().mean()
        )  # fraction of times mode k was ACTUALLY selected (emprical)

        if went_mode_k.sum() > 0 and chosen_arm is not None:
            log_dict[f"reward_achieved_mode{k}"] = (
                reward_prev[went_mode_k].float().mean()
            )  # average reward achieved when mode k was selected

            prob_R = went_right[went_mode_k].float().mean()
            prob_L = went_left[went_mode_k].float().mean()

            log_dict[f"prob_R_mode{k}"] = prob_R
            log_dict[f"prob_L_mode{k}"] = prob_L

    return log_dict


class HighLevelContextRNN_QLearning(nn.Module):
    """
    HighLevelContextRNN
    ├── RNNCell  (vanilla tanh)    §2
    └── QHead(K)                   §16

    K   = 4          overcomplete mode set
    d_H = 16         hidden dim (could be 8, 16, 32)
    input_dim = K+1  (z_prev one-hot + reward scalar)  §16
    """

    def __init__(self, K: int = 4, d_H: int = 16, tau: float = 1.0):
        super().__init__()
        self.K = K
        self.d_H = d_H
        self.tau = tau

        self.rnn_cell = nn.RNNCell(input_size=K + 1, hidden_size=d_H, nonlinearity="tanh")

        self.q_head = QHead(
            d_H=d_H, K=K
        )  # convert hidden state to Q-values and log-probabilities for sampling a new mode

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, chosen_arm):

        # 1. Candidate high-level state update  —  §18 step 1
        rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)  # (B, K+1)
        h_candidate = self.rnn_cell(rnn_in, h_high)  # (B, d_H)

        # 2. Tick only at outcome events  —  §18 step 2
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,
            h_high,
        )  # (B, d_H)

        # 3. Produce high-level scores from updated context state  —  §18 step 3
        q_scores = self.q_head(h_high_new)  # (B, K) each

        # 4. Sample a new mode only at decision events  —  §18 step 4
        z_candidate, z_candidate_index = sample_mode(q_scores)  # (B, K)

        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )  # (B, K)

        # §17: persist z_index for logging and loss computation
        z_index_prev = z_prev.argmax(dim=-1)  # (B,)
        z_index_new = torch.where(
            outcome_mask,
            z_candidate_index,
            z_index_prev,
        )

        # 5. z_new goes to existing low-level controller  —  §18 step 5
        # (caller passes z_new into the decoder)

        new_state = {
            "high_level_h": h_high_new,
            "current_z": z_new,
            "current_z_index": z_index_new,
        }

        # §22: logging quantities (only meaningful at decision points)
        log_dict = compute_log_dict(
            q_scores, z_index_new, h_high_new, prev_trial_reward, self.tau, chosen_arm, K=self.K
        )

        return h_high_new, z_new, q_scores, new_state, log_dict


# Q loss  —  §10 Option C, §19
# compare these raw expected rewards (Q) to actual obtained rewards
def q_loss(
    scores: Tensor,  # (T, B, K)
    z_indices: Tensor,  # (T, B)
    rewards: Tensor,  # (T, B)
    decision_mask: Tensor,  # (T, B)  bool — 1 only at decision points
) -> Tensor:
    """
    L_Q = Σ_t d_t (Q_H(h_t, z_t) - r_t)^2  /  (Σ_t d_t + ε)   §19
    """
    q_selected = scores.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(-1)  # (T, B)
    error = (q_selected - rewards) ** 2
    mask = decision_mask.float()
    return (error * mask).sum() / (mask.sum() + 1e-8)


### SECOND OPTION ###


# Policy Gradient — §9.1 and §10
class HighLevelContextRNN_Policy(nn.Module):
    def __init__(self, d_H: int, K: int, tau: float = 1.0):
        super().__init__()
        self.K = K
        self.d_H = d_H
        self.tau = tau

        # Keep your exact same RNN cell from the Q-learning version
        self.rnn_cell = nn.RNNCell(input_size=K + 1, hidden_size=d_H, nonlinearity="tanh")

        # Replace QHead with the new PolicyHead
        self.policy_head = PolicyHead(d_H=d_H, K=K)

    def forward(
        self, outcome_mask, prev_trial_reward, h_high: Tensor, z_prev: Tensor, chosen_arm
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        1. Update state: h_k
        2. Get logits:   l_k
        3. Sample mode:  z_k
        """

        # 1. Update the recurrent state exactly like you did before
        rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)  # (B, K+1)
        h_candidate = self.rnn_cell(rnn_in, h_high)  # (B, d_H)

        # 2. Tick only at outcome events  —  §18 step 2
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,
            h_high,
        )

        # 2. Pass the new hidden state to the Policy Head to get logits
        logits = self.policy_head(h_high_new)

        # 3. Sample the mode using the new sample_mode function
        z_candidate, z_candidate_index = sample_mode_policy(logits)

        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )  # (B, K)

        z_index_prev = z_prev.argmax(dim=-1)  # (B,)
        z_index_new = torch.where(
            outcome_mask,
            z_candidate_index,
            z_index_prev,
        )

        new_state = {
            "high_level_h": h_high_new,
            "current_z": z_new,
            "current_z_index": z_index_new,
        }

        log_dict = compute_log_dict(logits, z_index_new, h_high_new, prev_trial_reward, self.tau, chosen_arm, K=self.K)

        # Return 'logits' instead of 'scores' so Sample Factory saves them to the trajectory buffer!
        return h_high_new, z_new, logits, new_state, log_dict


# Policy-logit formulation  —  §9.1


class PolicyHead(nn.Module):
    def __init__(self, d_H: int, K: int):
        super().__init__()
        # l_k = W_pi * h_k + b_pi  —  §9.1
        self.head = nn.Linear(d_H, K)

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, d_H)  →  logits: (B, K)"""
        logits = self.head(h)
        return logits


# sample_mode  —  §9.1


def sample_mode_policy(logits: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
    """
    logits: (B, K)  — denoted as l_k in §9.1
    → z_onehot: (B, K)   one-hot e(z_k)
    → z_index:  (B,)     integer index

    z_k ~ pi_H(.|h_k)  —  §9.1
    """
    K = logits.size(-1)

    if deterministic:
        z_index = logits.argmax(dim=-1)
    else:
        # z_k ~ pi_H(.|h_k), including under torch.no_grad() on actors
        z_index = torch.distributions.Categorical(logits=logits).sample()

    z_onehot = F.one_hot(z_index, num_classes=K).float()
    return z_onehot, z_index


def high_level_policy_loss(
    logits: Tensor, z_indices: Tensor, rewards: Tensor, decision_mask: Tensor, entropy_coef: float = 0.05
) -> Tuple[Tensor, dict]:
    """
    L_H = -(R_k - b_k) * log(pi(z_k|h_k))
    Returns: (total_loss, dict_of_metrics)
    """
    mask_float = (
        decision_mask.float()
    )  # use the mask to multiply all non-decision points by zero, so they don't contribute to the loss
    valid_transitions = mask_float.sum() + 1e-8  # to avoid division by zero in case there are no valid transitions

    log_pi_all = F.log_softmax(logits, dim=-1)  # convert logits to log-probabilities
    pi_all = torch.exp(log_pi_all)  # convert logits to probabilities

    log_pi_selected = log_pi_all.gather(dim=-1, index=z_indices.unsqueeze(-1)).squeeze(
        -1
    )  # get the log-probabiltiy of the mode that was actually selected (z_k) for each batch element
    entropy = -(pi_all * log_pi_all).sum(dim=-1)

    # 1. The Baseline (b_k)
    # Average rewards the HL RNN got in this batch.
    #b_k = (rewards * mask_float).sum() / valid_transitions
    # instead have a fixed b_k = 0.5, so a win gives +0.5 and a loss gives -0.5
    b_k = 0.5

    # 2. The Advantage (R_k - b_k)
    # Difference between the actual reward and the baseline. This tells us how much better or worse the selected action was compared to the average.
    advantages = rewards - b_k
    advantages = advantages.detach()

    # Losses
    pg_loss = -(advantages * log_pi_selected)
    total_loss = pg_loss - (entropy_coef * entropy)

    final_loss = (total_loss * mask_float).sum() / valid_transitions

    # --- NEW: Create a dictionary of these specific Policy Gradient metrics ---
    metrics = {
        "hl/pg_baseline": float(b_k),
        "hl/pg_advantage": (advantages * mask_float).sum().item() / valid_transitions.item(),
        "hl/pg_loss": (pg_loss * mask_float).sum().item() / valid_transitions.item(),
    }

    return final_loss, metrics


### UNIFIED FOR Q-LEARNING AND POLICY GRADIENTS ###


class HighLevelContextRNN_Learner(nn.Module):
    """
    Unified High-Level RNN that seamlessly handles BOTH Q-Learning and Policy Gradients
    based on the 'is_policy' flag.
    """

    def __init__(
        self,
        K: int = 4,
        d_H: int = 16,
        tau: float = 1.0,
        is_policy: bool = False,
        deterministic: bool = False,
        representations: bool = False,
        d_rep: int = 64,
        mixed_selection: bool = False,  
    ):
        super().__init__()
        self.K = K
        self.d_H = d_H
        self.tau = tau
        self.is_policy = is_policy
        self.deterministic = deterministic
        self.representations = representations  # If True, the model will get representations from decoder 1st layer
        self.d_rep = d_rep  # Dimension of the representation vector
        self.mixed_selection = mixed_selection  # If True, the model will use a mix of deterministic and stochastic selection

        mode_dim = (2 * K) if self.mixed_selection else (K + 1) # 2 * K won and lost modes, otherwise z and r additively
        if self.representations:
            input_dim = mode_dim+ (2 * self.d_rep)  # z_prev + prev_reward + 2 * representation size = 2 + 1 + 128 = 131
        else:
            input_dim = mode_dim  # z_prev + prev_reward

        self.rnn_cell = nn.RNNCell(input_size=input_dim, hidden_size=d_H, nonlinearity="tanh")

        # This acts as BOTH the QHead and the PolicyHead
        self.head = nn.Linear(d_H, K)

    def sample_mode(self, outputs: Tensor) -> Tuple[Tensor, Tensor]:
        # Policy logits use raw outputs, Q-Learning uses temperature scaling
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

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, chosen_arm, e_trigger=None, e_outcome=None):
        # 1. Update State
        h_high_new = self.update_state(outcome_mask, prev_trial_reward, h_high, z_prev, e_trigger, e_outcome)

        # 2. Get Outputs (Q-Scores OR Logits)
        outputs = self.head(h_high_new)

        # 3. Sample and Latch
        z_candidate, z_candidate_index = self.sample_mode(outputs)

        z_new = torch.where(outcome_mask[:, None], z_candidate, z_prev)

        z_index_prev = z_prev.argmax(dim=-1)
        z_index_new = torch.where(outcome_mask, z_candidate_index, z_index_prev)

        new_state = {
            "high_level_h": h_high_new,
            "current_z": z_new,
            "current_z_index": z_index_new,
        }

        log_dict = compute_log_dict(
            outputs,
            z_index_new,
            h_high_new,
            prev_trial_reward,
            self.tau,
            chosen_arm,
            K=self.K,
            is_policy=self.is_policy,
        )

        return h_high_new, z_new, outputs, new_state, log_dict

    def update_state(self, outcome_mask, prev_trial_reward, h_high, z_prev, e_trigger=None, e_outcome=None):
        """Update reward history without choosing a new mode (used during replay)."""
        r_col = prev_trial_reward[:, None] if prev_trial_reward.dim() == 1 else prev_trial_reward # turn the 1D reward into a column vector if needed
        
        if self.mixed_selection:
            z_won = z_prev * r_col  # element-wise multiplication to get the "won" mode
            z_lost = z_prev * (1 - r_col)  # element-wise multiplication to get the "lost" mode
            mode_inputs = [z_won, z_lost]
        else:
            mode_inputs = [z_prev, r_col]

        if self.representations:
            if e_trigger is None or e_outcome is None:
                raise ValueError("representations=True requires both e_trigger and e_outcome")

            rnn_in = torch.cat(mode_inputs + [e_trigger, e_outcome], dim=-1)
        else:
            rnn_in = torch.cat(mode_inputs, dim=-1)

        h_candidate = self.rnn_cell(rnn_in, h_high)
        h_high_new = torch.where(outcome_mask[:, None], h_candidate, h_high)
        return h_high_new
