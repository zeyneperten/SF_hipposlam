import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class ContextRNN(nn.Module):
    """
    Maintains an explicit 2D value state [v_left, v_right] with decay:
        v_t = alpha * v_{t-1} + reward_pulse * choice_prob
    followed by an optional small recurrent layer and readout heads to DG and Decoder.

    Rescola-Wagner style value update is used to ensure that the model can learn to maintain a memory of which arm was rewarded, even if the reward is delayed.
    The context vector is then used to modulate the DG layer and provide context to the decoder.
    """
    def __init__(
        self,
        ca3_dim: int,
        dg_dim: int,
        hidden_size: int = 6,
        decoder_context_dim: int = 6, # keep it same as hidden_size
        dg_strength: float = 0.1, # how much to modulate DG with context, keep it low so that the model doesn't fully rely on context and still learn
        initial_decay: float = 0.95,      # Initial alpha decay rate (e.g. 0.95 per step)
        learn_decay: bool = True,          # If True, alpha is learned via backprop
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dg_strength = dg_strength
        self.value_dim = 2  # [value_left, value_right]

        # 1. Project CA3 state to 2D arm evidence [p_left, p_right]
        # Softmax ensures it acts as probabilities of being in left vs right arm
        self.ca3_to_choice = nn.Linear(ca3_dim, self.value_dim)

        # 2. Value decay factor alpha = sigmoid(alpha_logit) in (0, 1)
        # Store the unconstrained alpha_logit and apply sigmoid in forward pass to ensure alpha is in (0, 1)
        # During training the gradient descent will optimize alpha, can discover the exact decay rate that matches block duration.
        # logit of 0.95 is ~ 2.94
        init_logit = torch.log(torch.tensor(initial_decay) / (1.0 - torch.tensor(initial_decay)))
        if learn_decay:
            self.alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_buffer("alpha_logit", init_logit)

        # 3. Optional small RNN/GRU on top of the 2D value to model richer dynamics
        # Input to RNN: [v_left, v_right, reward_pulse] -> 3 dims
        self.rnn_cell = nn.GRUCell(
            input_size=self.value_dim + 1,
            hidden_size=hidden_size,
        )

        # 4. Readout heads
        # Decoder gets the full hidden state (or [value, hidden])
        self.context_to_decoder = nn.Linear(hidden_size, decoder_context_dim)
        # DG gets modulation vector
        self.context_to_dg = nn.Linear(hidden_size, dg_dim)

        # Start DG modulation near unmodulated baseline
        nn.init.normal_(self.context_to_dg.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.context_to_dg.bias)

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def initial_hidden(
        self, batch_size: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            v_init: (B, 2) zeros for [v_left, v_right]
            h_init: (B, hidden_size) zeros for GRU
        """
        v_init = torch.zeros(batch_size, self.value_dim, device=device, dtype=dtype)
        h_init = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return v_init, h_init

    def forward(
        self,
        reward_input: Tensor,
        ca3_output: Tensor,
        v_prev: Tensor,
        h_prev: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            reward_input: (B, 1) scalar reward pulse (1 at reward, 0 otherwise)
            ca3_output:  (B, ca3_dim) CA3 shift-register state
            v_prev:      (B, 2) previous [v_left, v_right]
            h_prev:      (B, hidden_size) previous GRU state

        Returns:
            decoder_context: (B, decoder_context_dim) for decoder at timestep t. Calculated by passing the hidden state through a linear layer and tanh activation.
            dg_gate_next:    (B, dg_dim) for DG at timestep t+1. To multiplicatively modulate DG output.
            v_new:           (B, 2) updated value state
            h_new:           (B, hidden_size) updated GRU state
        """
        # Ensure correct shape (B, 1)
        target_shape = ca3_output.shape[:-1] + (1,)
        if reward_input.shape != target_shape:
            reward_input = reward_input.view(target_shape)
        reward_input = reward_input.to(device=ca3_output.device, dtype=ca3_output.dtype)

        # 1. Infer arm presence from CA3: choice_probs has shape (B, 2)
        # softmax: ensure these two values are in (0, 1) and sum to 1, representing probabilities of being in left vs right arm
        # Biological intuition: CA3's sequence of place fields naturally separates trajectories. The linear projection learns to recognize the spatial fingerprint of each arm.
        choice_probs = torch.softmax(self.ca3_to_choice(ca3_output), dim=-1)

        # 2. Compute arm-specific reward:
        # If reward_input = 1 and choice is Left [1, 0] -> reward_arm = [1, 0]
        # If no reward (reward_input = 0) -> reward_arm = [0, 0]
        reward_arm = reward_input * choice_probs

        # 3. EXPLICIT VALUE DECAY UPDATE:
        # v_t = alpha * v_{t-1} + reward_arm
        v_new = self.alpha * v_prev + reward_arm

        # 4. Feed updated [v_left, v_right, reward_pulse] into GRU
        rnn_in = torch.cat([v_new, reward_input], dim=-1)
        h_new = self.rnn_cell(rnn_in, h_prev) # update the hidden state based on new value and reward.

        # 5. Readout to Decoder and DG
        decoder_context = torch.tanh(self.context_to_decoder(h_new)) # tells policy head which arm has high value
        dg_gate_next = self.dg_strength * torch.tanh(self.context_to_dg(h_new)) # gating value to modulate at next step, so DG can subtly prime spatial representations for the next step without feedback loop

        return decoder_context, dg_gate_next, v_new, h_new