import numbers
import weakref
import warnings
from typing import Optional, overload

import torch
from torch import _VF, Tensor
from torch.nn import RNNBase
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

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
            self.lr_row.zero_()

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

    def forward(self, input, hx=None):  # noqa: F811
        """
        Runs the forward pass.
        """
        '''
        new_W_hh = self.weight_hh_l0 + self.lr_column @ self.lr_row
        self.weight_hh_l0 = new_W_hh
        '''
        self._update_flat_weights()

        # add low-rank adaptation to the recurrent weights
        weights = list(self._flat_weights)

        idx = self._flat_weights_names.index("weight_hh_l0")
        W_hh_base = weights[idx]
        new_W_hh = W_hh_base + self.lr_column @ self.lr_row
        # update recurrent weights to the new weights with low-rank adaptation
        weights[idx] = new_W_hh

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            # script() is unhappy when max_batch_size is different type in cond branches, so we duplicate
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx, batch_sizes)
        assert self.mode == "RNN_TANH" or self.mode == "RNN_RELU"
        if batch_sizes is None:
            if self.mode == "RNN_TANH":
                result = _VF.rnn_tanh(
                    input,
                    hx,
                    weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    hx,
                    weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
        else:
            if self.mode == "RNN_TANH":
                result = _VF.rnn_tanh(
                    input,
                    batch_sizes,
                    hx,
                    weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    batch_sizes,
                    hx,
                    weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:  # type: ignore[possibly-undefined]
            output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)