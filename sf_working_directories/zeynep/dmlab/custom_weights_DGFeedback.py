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
    """Create fixed DG->CA3 and CA3->CA3 weight matrices.

    Returns:
        W_in: shape (n_feature * register_length, n_feature)
            The fixed DG->CA3 input map. Each DG feature is injected into
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
    W_in = torch.zeros(hidden_size, n_feature, device=device, dtype=dtype) # inject DG features into first injection_width slots of each CA3 register
    W_hh = torch.zeros(hidden_size, hidden_size, device=device, dtype=dtype) # move each CA3 register slot to the next slot in the next time step

    for feature in range(n_feature):
        start = feature * register_length
        W_in[start : start + injection_width, feature] = 1.0

        rows = torch.arange(start + 1, start + register_length, device=device)
        cols = torch.arange(start, start + register_length - 1, device=device)
        W_hh[rows, cols] = 1.0

    return W_in, W_hh