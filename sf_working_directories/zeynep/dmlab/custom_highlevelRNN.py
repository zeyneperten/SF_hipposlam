# SimGaY §16 decomposition + §18 forward-pass pseudocode
# Q-style contextual bandit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


# QHead  —  §16, §9.2

class QHead(nn.Module):
    def __init__(self, d_H: int, K: int, tau: float = 1.0):
        super().__init__()
        self.tau  = tau
        self.head = nn.Linear(d_H, K)

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, d_H)  →  q: (B, K)"""
        q      = self.head(h) # current estimate of expected reward for each mode | THIS IS LEARNED BY THE NETWORK
        return q


# sample_mode  —  §9, §18 step 4

def sample_mode(scores: Tensor, tau: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    scores: (B, K)
    → z_onehot: (B, K)   one-hot e(z_k)
    → z_index:  (B,)     integer index  — §17 current_z_index
    """
    K = log_pi.size(-1)
    log_pi = F.log_softmax(scores / tau, dim=-1) # convert raw Q-values to log-probabilities (distribution) for sampling 
    if torch.is_grad_enabled():
        z_index = torch.distributions.Categorical(logits=log_pi).sample() # from distribution, sample a discrete mode index z_k
    else:
        z_index = log_pi.argmax(dim=-1)
    z_onehot = F.one_hot(z_index, num_classes=K).float() # return binary one-hot vector of the sampled mode index | RNN and DECODER RECIEVE IT
    return z_onehot, z_index 


# Logging helpers  —  §22

def compute_log_dict(
    scores:       Tensor,   # (B, K)  Q-values
    z_index:      Tensor,   # (B,)    selected mode index
    h_high_new:   Tensor,   # (B, d_H)
    reward_prev:  Tensor,   # (B,)
    tau:          float,
) -> Dict[str, Tensor]:
    """
    Recommended logging quantities (computed at decision points only).
    High-level process:
        z_k, scores/Q vector, selection entropy, h_k, r_k
    Q-head specific:
        all Q_H(h_k, z), selected-Q vs reward, gap between top two Q-values
    """
    pi      = F.softmax(scores / tau, dim=-1)               # (B, K)
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)         # (B,)  selection entropy
    q_selected = scores.gather(                              # (B,)
        dim=-1, index=z_index.unsqueeze(-1)
    ).squeeze(-1)
    top2       = scores.topk(k=2, dim=-1).values            # (B, 2)
    q_gap      = top2[:, 0] - top2[:, 1]                   # (B,)  gap top1 - top2

    return {
        # §22 high-level process
        "hl/z_index":         z_index,           # sampled z_k
        "hl/scores":          scores,             # all Q vector
        "hl/entropy":         entropy,            # selection entropy
        "hl/h":               h_high_new,         # hidden state h_k
        "hl/reward_prev":     reward_prev,        # trial reward r_k
        # §22 Q-head specific
        "hl/q_selected":      q_selected,         # Q of chosen mode
        "hl/q_gap":           q_gap,              # gap between top two Q-values
        # §22 mode probabilities (for context adaptation plots)
        "hl/mode_probs":      pi,                 # π_H(z | h_k) for all z
    }


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
        self.rnn_cell = nn.RNNCell(input_size=K + 1, hidden_size=d_H, nonlinearity='tanh')

    def forward(self, outcome_mask, prev_trial_reward, h_high, z_prev, inst_block = None):

        # 1. Background RNN update (just testing mechanics)
        rnn_in = torch.cat([z_prev, prev_trial_reward[:, None]], dim=-1)
        h_candidate = self.rnn_cell(rnn_in, h_high)
        
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,
            h_high,
        )

        # MODE SELECTION
        if inst_block is not None:
            # If inst_block is provided, we can use it to select a mode deterministically
            oracle_idx = torch.clamp(inst_block - 1, 0, 1)  # Ensure the index is within bounds
            z_candidate = F.one_hot(oracle_idx, num_classes=self.K).float()
        else:
            # Otherwise, we can randomly select a mode for testing purposes
            B = z_prev.size(0)
            random_indices = torch.randint(0, self.K, (B,), device=z_prev.device)
            z_candidate = F.one_hot(random_indices, num_classes=self.K).float()

        # 3. Latch the new mode only at outcome events
        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )

        # Passing the mode to decoder is handeled in the wrapper

        return h_high_new, z_new


class HighLevelContextRNN_QLearning(nn.Module):
    """
    HighLevelContextRNN
    ├── RNNCell  (vanilla tanh)    §2
    └── QHead(K)                   §16

    K   = 4          overcomplete mode set
    d_H = 16         hidden dim (could be 8, 16, 32)
    input_dim = K+1  (z_prev one-hot + reward scalar)  §16
    """

    def __init__(self, low_level_model, K: int = 4, d_H: int = 16, tau: float = 1.0):
        super().__init__()
        self.K   = K
        self.d_H = d_H
        self.tau = tau

        self.rnn_cell    = nn.RNNCell(input_size=K + 1,
                                      hidden_size=d_H,
                                      nonlinearity='tanh')
        
        self.choice_head = QHead(d_H=d_H, K=K) # convert hidden state to Q-values and log-probabilities for sampling a new mode

        self.low_level_model = low_level_model  # reference to the existing low-level controller (decoder) that will receive the sampled mode z_new

    def forward(
        self,
        obs:             Dict[str, Tensor],
        recurrent_state: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        obs keys required:
            "outcome_event"      (B,)   bool — 1 when trial just ended # add this new instantenous observation when its back to center
            "prev_trial_reward"  (B,)   scalar reward of the previous trial # the current reward_input is instantenous, we need one that is maintained across the trial, so we use the previous trial's reward to update the high-level state

        recurrent_state keys:
            "high_level_h"  (B, d_H)   slow hidden state
            "current_z"     (B, K)     latched mode one-hot

        Returns
        -------
        z_new       (B, K)  held mode for this frame  →  goes to decoder
        q_values    (B, K)  all Q-values              →  used for loss
        new_state   dict    updated recurrent state
        """

        h_high = recurrent_state["high_level_h"]   # (B, d_H)
        z_prev = recurrent_state["current_z"]       # (B, K)

        outcome_mask = obs["outcome_event"].bool()           # (B,)
        reward_prev  = obs["prev_trial_reward"]  # (B, 1)

        # 1. Candidate high-level state update  —  §18 step 1
        rnn_in      = torch.cat([z_prev, reward_prev[:,None]], dim=-1)  # (B, K+1)
        h_candidate = self.rnn_cell(rnn_in, h_high)             # (B, d_H)

        # 2. Tick only at outcome events  —  §18 step 2
        h_high_new = torch.where(
            outcome_mask[:, None],
            h_candidate,
            h_high,
        )                                                        # (B, d_H)

        # 3. Produce high-level scores from updated context state  —  §18 step 3
        q_scores = self.choice_head(h_high_new)         # (B, K) each

        # 4. Sample a new mode only at decision events  —  §18 step 4
        z_candidate, z_candidate_index = sample_mode(q_scores)                       # (B, K)

        z_new = torch.where(
            outcome_mask[:, None],
            z_candidate,
            z_prev,
        )                                                        # (B, K)

        # §17: persist z_index for logging and loss computation
        z_index_prev = recurrent_state["current_z_index"]     # (B,)
        z_index_new  = torch.where(
            outcome_mask,
            z_candidate_index,
            z_index_prev,
        )
        
        # 5. z_new goes to existing low-level controller  —  §18 step 5
        # (caller passes z_new into the decoder)
        low_level_out = self.low_level_model(obs, z_new)

        new_state = {
            "high_level_h": h_high_new,
            "current_z":    z_new,
            "current_z_index": z_index_new,
        }

        # §22: logging quantities (only meaningful at decision points)
        log_dict = compute_log_dict(
            q_scores, z_index_new, h_high_new, reward_prev, self.tau
        )

        return low_level_out, q_scores, new_state, log_dict


# Q loss  —  §10 Option C, §19

def q_loss(
    scores:        Tensor,   # (T, B, K)
    z_indices:     Tensor,   # (T, B)
    rewards:       Tensor,   # (T, B)
    decision_mask: Tensor,   # (T, B)  bool — 1 only at decision points
) -> Tensor:
    """
    L_Q = Σ_t d_t (Q_H(h_t, z_t) - r_t)^2  /  (Σ_t d_t + ε)   §19
    """
    q_selected = scores.gather(
        dim=-1, index=z_indices.unsqueeze(-1)
    ).squeeze(-1)                                   # (T, B)
    error = (q_selected - rewards) ** 2
    mask  = decision_mask.float()
    return (error * mask).sum() / (mask.sum() + 1e-8)