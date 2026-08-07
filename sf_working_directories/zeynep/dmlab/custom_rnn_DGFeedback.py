from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence


class CustomRNN(nn.Module):
    """One-layer ReLU RNN with constrained DG-feedback adaptation.

    Effective recurrent matrix:
        W_hh_eff = W_hh + W_in @ W_feedback

    W_in is stored in lr_column and is fixed. W_feedback is lr_row and is
    the only trainable parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = "relu",
        batch_first: bool = False,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if nonlinearity != "relu":
            raise ValueError("This constrained core is implemented for nonlinearity='relu'.")
        if bias:
            raise ValueError("This fixed shift-register core is designed with bias=False.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer(
            "weight_ih_l0", torch.zeros(hidden_size, input_size, **factory_kwargs)
        )
        self.register_buffer(
            "weight_hh_l0", torch.zeros(hidden_size, hidden_size, **factory_kwargs)
        )

        # Fixed B factor in Delta_W = B @ A. It is W_in / W_ih.
        self.register_buffer(
            "lr_column", torch.zeros(hidden_size, input_size, **factory_kwargs)
        )

        # Trainable A factor in Delta_W = B @ A. It is W_feedback.
        self.lr_row = nn.Parameter(
            torch.zeros(input_size, hidden_size, **factory_kwargs)
        )

    def set_fixed_weights(self, W_ih: Tensor, W_hh: Tensor) -> None:
        """Set W_in and W_hh, and tie the fixed LoRA column to W_in."""
        if W_ih.shape != self.weight_ih_l0.shape:
            raise ValueError(f"W_ih must have shape {tuple(self.weight_ih_l0.shape)}.")
        if W_hh.shape != self.weight_hh_l0.shape:
            raise ValueError(f"W_hh must have shape {tuple(self.weight_hh_l0.shape)}.")

        with torch.no_grad():
            self.weight_ih_l0.copy_(W_ih)
            self.weight_hh_l0.copy_(W_hh)
            self.lr_column.copy_(W_ih)
            self.lr_row.zero_()

    @property
    def effective_weight_hh(self) -> Tensor:
        # W_hh + W_in @ W_feedback
        return self.weight_hh_l0 + self.lr_column @ self.lr_row

    def _step(self, x_t: Tensor, h_t: Tensor) -> Tensor:
        pre_activation = x_t @ self.weight_ih_l0.T + h_t @ self.effective_weight_hh.T
        return torch.relu(pre_activation)

    def _forward_packed(self, packed: PackedSequence, hx: Optional[Tensor]):
        data, batch_sizes, sorted_indices, unsorted_indices = packed
        max_batch = int(batch_sizes[0])

        if hx is None:
            h = data.new_zeros(max_batch, self.hidden_size)
        else:
            if hx.ndim != 3 or hx.shape[0] != 1 or hx.shape[2] != self.hidden_size:
                raise ValueError(
                    "For this one-layer RNN, hx must have shape (1, batch, hidden_size)."
                )
            h = hx[0]
            if sorted_indices is not None:
                h = h.index_select(0, sorted_indices)

        outputs = []
        offset = 0
        for batch_size in batch_sizes.tolist():
            x_t = data[offset : offset + batch_size]
            h_active = self._step(x_t, h[:batch_size])
            h = torch.cat((h_active, h[batch_size:]), dim=0)
            outputs.append(h_active)
            offset += batch_size

        output = PackedSequence(
            torch.cat(outputs, dim=0), batch_sizes, sorted_indices, unsorted_indices
        )
        hidden = h.unsqueeze(0)
        if unsorted_indices is not None:
            hidden = hidden.index_select(1, unsorted_indices)
        return output, hidden

    def forward(
        self, input: Union[Tensor, PackedSequence], hx: Optional[Tensor] = None
    ):
        if isinstance(input, PackedSequence):
            return self._forward_packed(input, hx)

        unbatched = input.ndim == 2
        if unbatched:
            input = input.unsqueeze(1 if not self.batch_first else 0)
        if input.ndim != 3:
            raise ValueError("input must be a 2-D, 3-D, or PackedSequence tensor.")

        if self.batch_first:
            input = input.transpose(0, 1)
        time_steps, batch_size, input_size = input.shape
        if input_size != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_size}.")

        if hx is None:
            h = input.new_zeros(batch_size, self.hidden_size)
        else:
            if hx.shape != (1, batch_size, self.hidden_size):
                raise ValueError(
                    f"hx must have shape (1, {batch_size}, {self.hidden_size})."
                )
            h = hx[0]

        outputs = []
        for t in range(time_steps):
            h = self._step(input[t], h)
            outputs.append(h)
        output = torch.stack(outputs, dim=0)
        hidden = h.unsqueeze(0)

        if self.batch_first:
            output = output.transpose(0, 1)
        if unbatched:
            output = output.squeeze(1 if not self.batch_first else 0)
            hidden = hidden.squeeze(1)
        return output, hidden
