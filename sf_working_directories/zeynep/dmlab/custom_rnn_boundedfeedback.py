from typing import Optional, overload

import torch
from torch import _VF, Tensor
from torch.nn import RNNBase
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import torch.nn as nn

from sample_factory.utils.utils import log

class CustomRNN(RNNBase):
   
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        rank = 1,  # added the rank to allow low-rank adaptations
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        if len(args) > 1:
            self.hidden_size = args[1]
        else:
            self.hidden_size = kwargs.get("hidden_size", 0)
        if len(args) > 3:
            self.nonlinearity = args[3]
            self.rank = args[4] 
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if len(args) > 4:
            self.rank = args[4]
            args = args[:3] + args[4:]
        else:
            self.rank = kwargs.pop("rank", 1)
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        
        if len(args) > 9:
            self.device = args[9]
        else:
            self.device = kwargs.get("device", None)
        if len(args) > 10:
            self.dtype = args[10]
        else:
            self.dtype = kwargs.get("dtype", None)
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        #for layer in range(self.num_layers):
            #for direction in range(num_directions):


        super().__init__(mode, *args, **kwargs)

        # change the rnn constructor to include low rank adaptation  #Idee: mach das zum parameter aber nicht teil der offiziellen liste, damit es trainiert wird aber nicht an den c code übergeben wird
        # B
        self.lr_column = Parameter(
            torch.empty((self.hidden_size, self.rank), **factory_kwargs)
        )
        # A
        self.lr_row = Parameter(
            torch.empty((self.rank, self.hidden_size), **factory_kwargs)
        )

        self.reset_parameters()

        # add to log 
        self._feedback_forward_count = 0

        self.feedback_norm = torch.nn.LayerNorm(self.hidden_size, elementwise_affine=False)


    def set_fixed_weights(self, W_in, W_hh):
        """
        Configure fixed CA3 weights and constrained DG-feedback LoRA.

        Effective recurrence:
            W_hh_eff = W_hh + W_in @ W_feedback

        lr_column = W_in          fixed
        lr_row    = W_feedback    trainable
        """
        with torch.no_grad(): # disable gradient calculation for fixed weights
            self.weight_ih_l0.copy_(W_in)
            self.weight_hh_l0.copy_(W_hh)

            # Fixed DG -> CA3 map.
            self.lr_column.copy_(W_in)

            # Initial condition: no learned feedback yet.
            # self.lr_row.zero_() # comment out for only direction learning, since I normalize
            self.lr_row.normal_(mean=0.0, std=0.02)

        # Freeze fixed CA3 and DG -> CA3 structure.
        self.weight_ih_l0.requires_grad_(False)
        self.weight_hh_l0.requires_grad_(False)
        self.lr_column.requires_grad_(False)

        # Only learned matrix: CA3 -> DG feedback.
        self.lr_row.requires_grad_(True)

    def update_weights():
        # 
 
        pass


    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811

    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        self._feedback_forward_count += 1

        if not torch.isfinite(self.lr_row).all():
            raise RuntimeError("lr_row contains NaN or Inf")

        if not torch.isfinite(self.lr_column).all():
            raise RuntimeError("lr_column contains NaN or Inf")

        if isinstance(input, PackedSequence):
            if not torch.isfinite(input.data).all():
                raise RuntimeError("PackedSequence input contains NaN or Inf")

            return self._forward_bounded_feedback_packed(input, hx)

        if not torch.isfinite(input).all():
            raise RuntimeError("RNN tensor input contains NaN or Inf")

        return self._forward_bounded_feedback_tensor(input, hx)
    
    def _bounded_feedback_step(
        self,
        x_t,
        h_prev,
        W_in,
        W_hh_base,
    ):
        direction = F.normalize(
            self.lr_row,
            p=2.0,
            dim=None,
            eps=1e-8,
        )

        h_context = F.layer_norm(
            h_prev,
            normalized_shape=(self.hidden_size,),
        )

        raw_feedback = h_context @ direction.T
        feedback = torch.tanh(raw_feedback)

        effective_dg_input = x_t + feedback

        pre_activation = (
            effective_dg_input @ W_in.T
            + h_prev @ W_hh_base.T
        )

        if self.nonlinearity == "relu":
            h_new = F.relu(pre_activation)
        else:
            h_new = torch.tanh(pre_activation)

        return h_new, feedback

    def _forward_bounded_feedback_tensor(
        self,
        input: Tensor,
        hx: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """
        Explicit bounded-feedback recurrence for non-packed RNN input.

        Supports:
            - num_layers == 1
            - bidirectional == False
            - dropout == 0
            - tanh or relu
            - batch_first True or False
        """
        if self.num_layers != 1:
            raise NotImplementedError(
                "Bounded-feedback recurrence currently supports num_layers=1 only"
            )

        if self.bidirectional:
            raise NotImplementedError(
                "Bounded-feedback recurrence does not support bidirectional=True"
            )

        if self.dropout != 0.0:
            raise NotImplementedError(
                "Bounded-feedback recurrence does not support nonzero dropout"
            )

        if input.dim() not in (2, 3):
            raise ValueError(
                f"Expected a 2D or 3D tensor, got {input.dim()}D tensor"
            )

        original_input_was_unbatched = input.dim() == 2

        # Match standard torch.nn.RNN behavior for unbatched sequence input.
        if original_input_was_unbatched:
            batch_dim = 0 if self.batch_first else 1
            input = input.unsqueeze(batch_dim)

            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError(
                        "For unbatched 2D input, hx must be 2D; "
                        f"got hx.dim()={hx.dim()}"
                    )
                hx = hx.unsqueeze(1)

        # Work internally as [B, T, input_size].
        if self.batch_first:
            input_batched = input
        else:
            input_batched = input.transpose(0, 1)

        B, T, input_size = input_batched.shape

        if input_size != self.input_size:
            raise RuntimeError(
                f"Expected input_size={self.input_size}, got {input_size}"
            )

        if hx is None:
            h = torch.zeros(
                B,
                self.hidden_size,
                dtype=input_batched.dtype,
                device=input_batched.device,
            )
        else:
            if hx.dim() != 3:
                raise RuntimeError(
                    f"Expected hx to be 3D after batching, got hx.dim()={hx.dim()}"
                )

            if hx.shape != (1, B, self.hidden_size):
                raise RuntimeError(
                    "Invalid hx shape for bounded-feedback RNN: "
                    f"expected {(1, B, self.hidden_size)}, got {tuple(hx.shape)}"
                )

            h = hx[0]

        if not torch.isfinite(h).all():
            raise RuntimeError("Initial hidden state contains NaN or Inf")

        W_in = self.weight_ih_l0
        W_hh_base = self.weight_hh_l0

        if not torch.isfinite(W_in).all():
            raise RuntimeError("W_in contains NaN or Inf")

        if not torch.isfinite(W_hh_base).all():
            raise RuntimeError("W_hh_base contains NaN or Inf")

        outputs = []

        # One scalar Tensor per timestep.
        state_abs_max = []
        feedback_abs_max = []
        raw_feedback_abs_max = []
        preactivation_abs_max = []

        for t in range(T):
            x_t = input_batched[:, t, :]

            h_new, feedback, raw_feedback = self._bounded_feedback_step(
                x_t=x_t,
                h_prev=h,
                W_in=W_in,
                W_hh_base=W_hh_base,
            )

            if not torch.isfinite(raw_feedback).all():
                raise RuntimeError(
                    f"Raw feedback contains NaN/Inf at timestep={t}"
                )

            if not torch.isfinite(feedback).all():
                raise RuntimeError(
                    f"Bounded feedback contains NaN/Inf at timestep={t}"
                )

            if not torch.isfinite(h_new).all():
                raise RuntimeError(
                    f"CA3 hidden state contains NaN/Inf at timestep={t}; "
                    f"previous_h_abs_max={h.detach().abs().max().item():.6g}, "
                    f"feedback_abs_max={feedback.detach().abs().max().item():.6g}"
                )

            # tanh feedback must stay in [-1, 1].
            feedback_max = feedback.detach().abs().max()

            if feedback_max.item() > 1.00001:
                raise RuntimeError(
                    f"Bounded feedback exceeds tanh bound at timestep={t}: "
                    f"max_abs={feedback_max.item():.6g}"
                )

            # A tanh CA3 activation must also remain in [-1, 1].
            h_new_max = h_new.detach().abs().max()

            if self.nonlinearity == "tanh" and h_new_max.item() > 1.00001:
                raise RuntimeError(
                    f"Tanh CA3 state exceeds tanh bound at timestep={t}: "
                    f"max_abs={h_new_max.item():.6g}"
                )

            h = h_new
            outputs.append(h)

            state_abs_max.append(h_new_max)
            feedback_abs_max.append(feedback_max)
            raw_feedback_abs_max.append(raw_feedback.detach().abs().max())

        output_batched = torch.stack(outputs, dim=1)
        hidden = h.unsqueeze(0)

        if not torch.isfinite(output_batched).all():
            raise RuntimeError("Bounded-feedback RNN output contains NaN or Inf")

        if not torch.isfinite(hidden).all():
            raise RuntimeError("Bounded-feedback final hidden state contains NaN or Inf")

        if self._feedback_forward_count % 50 == 0:
            with torch.no_grad():
                state_per_t = torch.stack(state_abs_max)
                feedback_per_t = torch.stack(feedback_abs_max)
                raw_feedback_per_t = torch.stack(raw_feedback_abs_max)

                log.debug(
                    "Bounded-feedback tensor summary: "
                    f"batch={B}, timesteps={T}, "
                    f"ca3_abs_max={state_per_t.max().item():.6g}, "
                    f"feedback_abs_max={feedback_per_t.max().item():.6g}, "
                    f"raw_feedback_abs_max={raw_feedback_per_t.max().item():.6g}, "
                    f"final_hidden_abs_max={hidden.detach().abs().max().item():.6g}"
                )

                log.debug(
                    "Bounded-feedback tensor timestep maxima: "
                    f"ca3={state_per_t.cpu().tolist()}, "
                    f"feedback={feedback_per_t.cpu().tolist()}, "
                    f"raw_feedback={raw_feedback_per_t.cpu().tolist()}"
                )

        # Return to the input layout expected by torch.nn.RNN.
        if self.batch_first:
            output = output_batched
        else:
            output = output_batched.transpose(0, 1)

        if original_input_was_unbatched:
            batch_dim = 0 if self.batch_first else 1
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, hidden

    def _forward_bounded_feedback_packed(
        self,
        packed_input: PackedSequence,
        hx: Optional[Tensor],
    ) -> tuple[PackedSequence, Tensor]:
        """
        Explicit bounded-feedback recurrence for PackedSequence input.

        Supports:
            - num_layers == 1
            - bidirectional == False
            - dropout == 0
            - tanh or relu
        """
        if self.num_layers != 1:
            raise NotImplementedError(
                "Bounded-feedback recurrence currently supports num_layers=1 only"
            )

        if self.bidirectional:
            raise NotImplementedError(
                "Bounded-feedback recurrence does not support bidirectional=True"
            )

        if self.dropout != 0.0:
            raise NotImplementedError(
                "Bounded-feedback recurrence does not support nonzero dropout"
            )

        data = packed_input.data
        batch_sizes = packed_input.batch_sizes
        sorted_indices = packed_input.sorted_indices
        unsorted_indices = packed_input.unsorted_indices

        if data.dim() != 2:
            raise RuntimeError(
                f"PackedSequence data must be 2D, got shape={tuple(data.shape)}"
            )

        if data.shape[1] != self.input_size:
            raise RuntimeError(
                f"Expected packed input_size={self.input_size}, "
                f"got {data.shape[1]}"
            )

        if not torch.isfinite(data).all():
            raise RuntimeError("PackedSequence input data contains NaN or Inf")

        max_batch_size = int(batch_sizes[0].item())

        if hx is None:
            h = torch.zeros(
                max_batch_size,
                self.hidden_size,
                dtype=data.dtype,
                device=data.device,
            )
        else:
            if hx.dim() != 3:
                raise RuntimeError(
                    f"Packed hx must be 3D, got hx.dim()={hx.dim()}"
                )

            # Reorder initial hidden states to the PackedSequence sorted order.
            hx = self.permute_hidden(hx, sorted_indices)

            if hx.shape != (1, max_batch_size, self.hidden_size):
                raise RuntimeError(
                    "Invalid packed hx shape: "
                    f"expected {(1, max_batch_size, self.hidden_size)}, "
                    f"got {tuple(hx.shape)}"
                )

            h = hx[0]

        if not torch.isfinite(h).all():
            raise RuntimeError("Initial packed hidden state contains NaN or Inf")

        W_in = self.weight_ih_l0
        W_hh_base = self.weight_hh_l0

        if not torch.isfinite(W_in).all():
            raise RuntimeError("W_in contains NaN or Inf")

        if not torch.isfinite(W_hh_base).all():
            raise RuntimeError("W_hh_base contains NaN or Inf")

        outputs = []

        # One scalar Tensor per packed timestep.
        state_abs_max = []
        feedback_abs_max = []
        raw_feedback_abs_max = []

        offset = 0

        for t, batch_size_tensor in enumerate(batch_sizes):
            active_batch_size = int(batch_size_tensor.item())

            # The first active_batch_size rows are the active sorted sequences.
            x_t = data[offset : offset + active_batch_size]
            h_active = h[:active_batch_size]

            h_new, feedback, raw_feedback = self._bounded_feedback_step(
                x_t=x_t,
                h_prev=h_active,
                W_in=W_in,
                W_hh_base=W_hh_base,
            )

            if not torch.isfinite(raw_feedback).all():
                raise RuntimeError(
                    f"Packed raw feedback contains NaN/Inf at timestep={t}"
                )

            if not torch.isfinite(feedback).all():
                raise RuntimeError(
                    f"Packed bounded feedback contains NaN/Inf at timestep={t}"
                )

            if not torch.isfinite(h_new).all():
                raise RuntimeError(
                    f"Packed CA3 state contains NaN/Inf at timestep={t}; "
                    f"previous_h_abs_max={h_active.detach().abs().max().item():.6g}, "
                    f"feedback_abs_max={feedback.detach().abs().max().item():.6g}"
                )

            feedback_max = feedback.detach().abs().max()

            if feedback_max.item() > 1.00001:
                raise RuntimeError(
                    f"Packed bounded feedback exceeds tanh bound at timestep={t}: "
                    f"max_abs={feedback_max.item():.6g}"
                )

            h_new_max = h_new.detach().abs().max()

            if self.nonlinearity == "tanh" and h_new_max.item() > 1.00001:
                raise RuntimeError(
                    f"Packed tanh CA3 state exceeds tanh bound at timestep={t}: "
                    f"max_abs={h_new_max.item():.6g}"
                )

            # Only active sequence states are updated. Inactive trailing rows retain
            # their final valid state for the final hidden-state output.
            h = h.clone()
            h[:active_batch_size] = h_new

            outputs.append(h_new)

            state_abs_max.append(h_new_max)
            feedback_abs_max.append(feedback_max)
            raw_feedback_abs_max.append(raw_feedback.detach().abs().max())

            offset += active_batch_size

        if offset != data.shape[0]:
            raise RuntimeError(
                f"Packed data traversal mismatch: consumed {offset}, "
                f"but data has {data.shape[0]} rows"
            )

        output_data = torch.cat(outputs, dim=0)

        if not torch.isfinite(output_data).all():
            raise RuntimeError("Packed bounded-feedback output contains NaN or Inf")

        hidden_sorted = h.unsqueeze(0)

        if not torch.isfinite(hidden_sorted).all():
            raise RuntimeError(
                "Packed bounded-feedback final hidden state contains NaN or Inf"
            )

        # Restore original batch order for the caller.
        hidden = self.permute_hidden(hidden_sorted, unsorted_indices)

        output_packed = PackedSequence(
            output_data,
            batch_sizes,
            sorted_indices,
            unsorted_indices,
        )

        if self._feedback_forward_count % 50 == 0:
            with torch.no_grad():
                state_per_t = torch.stack(state_abs_max)
                feedback_per_t = torch.stack(feedback_abs_max)
                raw_feedback_per_t = torch.stack(raw_feedback_abs_max)

                log.debug(
                    "Bounded-feedback packed summary: "
                    f"timesteps={len(batch_sizes)}, "
                    f"max_batch={max_batch_size}, "
                    f"ca3_abs_max={state_per_t.max().item():.6g}, "
                    f"feedback_abs_max={feedback_per_t.max().item():.6g}, "
                    f"raw_feedback_abs_max={raw_feedback_per_t.max().item():.6g}, "
                    f"final_hidden_abs_max={hidden.detach().abs().max().item():.6g}"
                )

                log.debug(
                    "Bounded-feedback packed timestep maxima: "
                    f"ca3={state_per_t.cpu().tolist()}, "
                    f"feedback={feedback_per_t.cpu().tolist()}, "
                    f"raw_feedback={raw_feedback_per_t.cpu().tolist()}"
                )

        return output_packed, hidden