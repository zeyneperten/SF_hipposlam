from typing import Tuple

import torch
from torch import Tensor


def generate_shift_register_weights(
    n_feature: int,
    register_length: int,
    injection_width: int,
    *,
    device=None,
    dtype=None,
) -> Tuple[Tensor, Tensor]:
    """Create fixed DG->CA3 and CA3->CA3 shift-register matrices.

    Returns:
        W_in: shape (n_feature * register_length, n_feature)
            The fixed DG-to-CA3 input map. Each DG feature is injected into
            the first `injection_width` slots of its CA3 register.

        W_hh: shape (n_feature * register_length,
                     n_feature * register_length)
            The fixed CA3 recurrence. Each register slot receives the prior
            slot from the preceding time step.
    """
    if n_feature <= 0:
        raise ValueError("n_feature must be positive.")
    if register_length <= 0:
        raise ValueError("register_length must be positive.")
    if not 1 <= injection_width <= register_length:
        raise ValueError("injection_width must lie in [1, register_length].")

    hidden_size = n_feature * register_length
    W_in = torch.zeros(hidden_size, n_feature, device=device, dtype=dtype)
    W_hh = torch.zeros(hidden_size, hidden_size, device=device, dtype=dtype)

    for feature in range(n_feature):
        start = feature * register_length
        W_in[start : start + injection_width, feature] = 1.0

        rows = torch.arange(start + 1, start + register_length, device=device)
        cols = torch.arange(start, start + register_length - 1, device=device)
        W_hh[rows, cols] = 1.0

    return W_in, W_hh


def configure_fixed_dg_feedback_rnn(rnn, W_in: Tensor, W_hh: Tensor) -> None:
    """Configure a CustomRNN for W_hh_eff = W_hh + W_in @ W_feedback.

    `rnn.lr_column` is set to W_in and remains frozen. `rnn.lr_row` is zeroed
    and is the sole trainable parameter; it represents W_feedback.
    """
    rnn.set_fixed_weights(W_in, W_hh)

    # These should already be buffers, but the assignments make the intended
    # optimisation policy explicit if CustomRNN is changed later.
    rnn.weight_ih_l0.requires_grad_(False)
    rnn.weight_hh_l0.requires_grad_(False)
    rnn.lr_column.requires_grad_(False)
    rnn.lr_row.requires_grad_(True)


def dg_feedback_update(W_in: Tensor, W_feedback: Tensor) -> Tensor:
    """Return the DG-loop contribution Delta_W_hh = W_in @ W_feedback."""
    if W_in.ndim != 2 or W_feedback.ndim != 2:
        raise ValueError("W_in and W_feedback must both be matrices.")
    if W_in.shape[1] != W_feedback.shape[0]:
        raise ValueError(
            "Expected W_in.shape[1] == W_feedback.shape[0], got "
            f"{W_in.shape} and {W_feedback.shape}."
        )
    return W_in @ W_feedback
