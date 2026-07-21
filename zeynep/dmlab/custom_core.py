# from asyncio.sslproto import add_flowcontrol_defaults
from email import header
from logging import warning

import torch
from torch import Tensor, nn

from sample_factory.model.core import ModelCore, ModelCoreIdentity, ModelCoreRNN
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log


class FixedRNNSequenceCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. (Here we assume it matches Hippo_n_feature.)
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)

        if self.Hippo_n_feature != input_size:
            raise Warning(f"hippo_n_feature{self.Hippo_n_feature } does not match input size {input_size}")

        # The total register length.
        self.expanded_length = self.R + self.L - 1
        # The flattened hidden state dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        # Create a one-layer RNN with ReLU activation.
        # (We use ReLU so that if inputs and state are nonnegative, the activation is effectively identity.)
        self.rnn = nn.RNN(
            input_size=self.n_feature,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="relu",
            batch_first=False,
        )

        # Create fixed weight matrices.
        # weight_ih: shape (hidden_size, input_size)
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0

        # weight_hh: shape (hidden_size, hidden_size)
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0

        # Assign the fixed weights and set biases to zero.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()

        # Freeze the weights so that they are not updated during training.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (core_output, new_rnn_states)
        """
        # Ensure hidden state is contiguous before passing to RNN.
        h0 = rnn_states.unsqueeze(0).contiguous()

        if isinstance(head_output, PackedSequence):
            output, new_hidden = self.rnn(head_output, h0)
            new_hidden = new_hidden.squeeze(0)  # (B, core_output_size)
            return output, new_hidden
        else:
            head_output = head_output.unsqueeze(0)  # (1, B, input_size)
            output, new_hidden = self.rnn(head_output, h0)
            new_hidden = new_hidden.squeeze(0)  # (B, core_output_size)
            output = output.squeeze(0)  # (B, core_output_size)
            return output, new_hidden


from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class FixedRNNWithBypassCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature
        log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length + self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        # Create a one-layer RNN with ReLU activation.
        self.rnn = nn.RNN(
            input_size=self.n_feature,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="relu",
            batch_first=False,
        )

        # Create fixed weight matrices.
        # weight_ih: shape (hidden_size, n_feature)
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0

        # weight_hh: shape (hidden_size, hidden_size)
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0

        # Assign the fixed weights and zero out biases.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()

        # Freeze RNN parameters.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (concat_output, new_rnn_states) where:
              - concat_output is the concatenation of the fixed RNN output and the bypass features.
              - new_rnn_states is the updated recurrent state.
        """
        # Prepare initial hidden state for RNN.
        # log.info(rnn_states.size())
        h0 = rnn_states.unsqueeze(0)[:, :, : self.hidden_size].contiguous()

        if isinstance(head_output, PackedSequence):
            # For PackedSequence, work on the underlying data.
            # Split into RNN and bypass parts.
            rnn_data = head_output.data[:, : self.n_feature]
            bypass_data = head_output.data[:, self.n_feature :] if self.bypass_size > 0 else None

            # Create a PackedSequence for the RNN input.
            rnn_packed = PackedSequence(
                rnn_data, head_output.batch_sizes, head_output.sorted_indices, head_output.unsorted_indices
            )
            # Run the RNN.
            rnn_output_packed, new_hidden = self.rnn(rnn_packed, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)

            # If bypass features exist, concatenate them.
            if bypass_data is not None:
                # Concatenate along the feature dimension.
                concatenated_data = torch.cat([rnn_output_packed.data, bypass_data], dim=1)

                bypass_data_packed = PackedSequence(
                    bypass_data, head_output.batch_sizes, head_output.sorted_indices, head_output.unsorted_indices
                )
                # Assume 'packed' is your PackedSequence and you used batch_first=True when packing.
                padded, lengths = pad_packed_sequence(bypass_data_packed, batch_first=True)

                # For each sequence in the batch, pick the last valid time step.
                # lengths is a tensor of the original sequence lengths.
                last_inputs = padded[torch.arange(padded.size(0)), lengths - 1, :]
                concatenated_data_hidden = torch.cat([new_hidden.data, last_inputs], dim=1)

                # # Compute indices in the packed data that correspond to the last time step of each sequence.
                # last_indices = head_output.batch_sizes.cumsum(0) - 1

                # # Use these indices to index into the bypass_data tensor.
                # last_bypass = bypass_data[last_indices,:]

                # # If the sequences were originally unsorted, restore the original order:
                # last_bypass = last_bypass[head_output.unsorted_indices,:]
                # concatenated_data_hidden = torch.cat([new_hidden.data, last_bypass], dim=1)

            else:
                concatenated_data = rnn_output_packed.data
                concatenated_data_hidden = new_hidden.data

            # Build a new PackedSequence with the concatenated data.
            concat_output = PackedSequence(
                concatenated_data,
                rnn_output_packed.batch_sizes,
                rnn_output_packed.sorted_indices,
                rnn_output_packed.unsorted_indices,
            )

            concat_hidden = PackedSequence(
                concatenated_data_hidden,
                rnn_output_packed.batch_sizes,
                rnn_output_packed.sorted_indices,
                rnn_output_packed.unsorted_indices,
            )
            return concat_output, concat_hidden
        else:
            # For Tensor input.
            # Split the input into RNN and bypass parts.
            rnn_input = head_output[:, : self.n_feature]  # shape: (B, n_feature)
            bypass_output = head_output[:, self.n_feature :] if self.bypass_size > 0 else None

            # Add sequence dimension for the RNN.
            rnn_input = rnn_input.unsqueeze(0)  # shape: (1, B, n_feature)
            rnn_output, new_hidden = self.rnn(rnn_input, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)
            rnn_output = rnn_output.squeeze(0)  # shape: (B, core_output_size)

            # Concatenate the RNN output with bypass features.
            if bypass_output is not None:
                concat_output = torch.cat([rnn_output, bypass_output], dim=1)
                concat_hidden = torch.cat([new_hidden, bypass_output], dim=1)
            else:
                concat_output = rnn_output

                concat_hidden = new_hidden

            return concat_output, concat_hidden


class SimpleSequenceCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            input_dim (int): Dimensionality of the input observation.
            R (int): Number of positions to update with the feature value.
            L (int): Number of time steps over which the shift register is built.
        """
        super().__init__(cfg)
        self.dim = input_size
        self.cfg = cfg
        if cfg.Hippo_R:
            self.R = cfg.Hippo_R
        else:
            self.R = 8
            log.info("R not set, using default: {self.R}")
        if cfg.Hippo_L:
            self.L = cfg.Hippo_L
        else:
            self.L = 48
            log.info("L not set, using default: {self.R}")
        if cfg.Hippo_n_feature:
            self.Hippo_n_feature = cfg.Hippo_n_feature
        else:
            self.Hippo_n_feature = 64
            log.info("hippo n feature not set, using default: {self.Hippo_n_feature}")

        # self.linear = nn.Linear(input_size, self.Hippo_n_feature)  # Map input to 64-dimensional features.
        self.expanded_length = self.R + self.L - 1  # Total length of the shift register.

        # self.rnn_states = torch.zeros(batch_size,self.Hippo_n_feature*self.expanded_length, device=device)
        self.core_output_size = self.Hippo_n_feature * self.expanded_length

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_dim) from the encoder
                        or a PackedSequence.
            rnn_states: Tensor of shape (B, Hippo_n_feature * expanded_length)
                        representing the flattened recurrent state.

        Returns:
            Tuple (core_output, new_rnn_states), both of shape
            (B, Hippo_n_feature * expanded_length) if the input is a single time step.
            (If a sequence is provided, the time dimension is preserved.)
        """
        # Determine if head_output is a PackedSequence or a plain tensor.
        # this supposedly also means inference mode or training mode
        is_seq = not torch.is_tensor(head_output)
        if is_seq:
            # log.info('==========packed sequence')
            # Unpack the sequence; padded_out shape: (T, B, input_dim)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded_out, lengths = nn.utils.rnn.pad_packed_sequence(head_output)

            # batch_indices=batch_sizes-1

            # unsorted_indices = getattr(head_output, 'unsorted_indices', None)

            features = padded_out
            # log.info(features.size())
            # log.info(batch_sizes[:5])
        else:
            # log.info('==========plain tensor')
            # If a plain tensor is provided, add a time dimension.
            # head_output = head_output.unsqueeze(0)  # now (1, B, input_dim)
            # lengths = [1]* head_output.size(0)
            features = head_output
            features = features.unsqueeze(0)

            batch_sizes = features.size(1)

        rnn_states = rnn_states.unsqueeze(0)
        rnn_states = rnn_states.view(
            rnn_states.size(0), rnn_states.size(1), self.Hippo_n_feature, self.expanded_length
        ).contiguous()
        output = torch.empty(
            (features.size(0), features.size(1), self.Hippo_n_feature, self.expanded_length), device=features.device
        )

        new_rnn_states = rnn_states.clone()

        ### tried to fast propagate hipposeq but it wouldn't work
        # for i in range(len(lengths)):
        #     # rnn_states = rnn_states[:,i,:,:].roll(shifts=lengths[i], dims=-1).contiguous()
        #     # # propagate hipposeq
        #     # rnn_states[:, i, :, :lengths[i]] = 0

        #     tmp_states= F.pad(rnn_states[:,i,:,:],(lengths[i],0)) #dim (0,n_feature,n_expanded)

        #     kernel=F.pad(torch.ones(self.R,device=rnn_states.device),(0,self.R-1))

        #     injection=F.conv1d(features[:,i,:].permute(1,0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0))

        #     rnn_states[:, i, :, :] = rnn_states[:, i, :, :] + injection

        #     # output (time, B, Hippo_n_feature, expanded_length)
        #     output[i,:,:,:]=new_rnn_states[0, :, :, :]
        #     rnn_states=new_rnn_states
        if is_seq:
            for i in range(features.size(0)):

                # Here, state_dim should equal self.Hippo_n_feature * expanded_length.
                # Reshape to 4D so we can perform our custom shift and injection.
                # New shape: (time, B, Hippo_n_feature, expanded_length)

                tmpind = sorted_indices[: batch_sizes[i]]

                # Shift the register one step along the last dimension.
                tmp_rnn_states = new_rnn_states[:, tmpind, :, :].roll(shifts=1, dims=-1).contiguous()
                # Zero out the newly empty slot.
                tmp_rnn_states[:, :, :, 0] = 0

                # Inject the current features into the first R positions.
                # features: (B, Hippo_n_feature) --> unsqueeze to (1, B, Hippo_n_feature, 1)
                # then expand to (1, B, Hippo_n_feature, R)
                injection = features[i, tmpind, :].unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R).contiguous()

                # log.info(tmp_rnn_states.size())
                # log.info(injection.size())
                # log.info('======size_above======')

                tmp_rnn_states[:, :, :, : self.R] = tmp_rnn_states[:, :, :, : self.R] + injection

                new_rnn_states[:, tmpind, :, :] = tmp_rnn_states[:, :, :, :]
                # output (time, B, Hippo_n_feature, expanded_length)
                output[i, :, :, :] = new_rnn_states[0, :, :, :]
        else:

            # Shift the register one step along the last dimension.
            tmp_rnn_states = new_rnn_states[:, :, :, :].roll(shifts=1, dims=-1).contiguous()
            # Zero out the newly empty slot.
            tmp_rnn_states[:, :, :, 0] = 0

            # Inject the current features into the first R positions.
            # features: (B, Hippo_n_feature) --> unsqueeze to (1, B, Hippo_n_feature, 1)
            # then expand to (1, B, Hippo_n_feature, R)
            injection = features[0, :].unsqueeze(-1).expand(tmp_rnn_states.size(0), -1, -1, self.R).contiguous()

            tmp_rnn_states[:, :, :, : self.R] = tmp_rnn_states[:, :, :, : self.R] + injection

            new_rnn_states[:, :, :, :] = tmp_rnn_states[:, :, :, :]
            # output (time, B, Hippo_n_feature, expanded_length)
            output[0, :, :, :] = new_rnn_states[0, :, :, :]

        # Flatten new_rnn_states back to 3D: (time, B, Hippo_n_feature * expanded_length)
        new_rnn_states = new_rnn_states.view(
            new_rnn_states.size(0), new_rnn_states.size(1), self.Hippo_n_feature * self.expanded_length
        ).contiguous()

        output = output.view(features.size(0), features.size(1), self.Hippo_n_feature * self.expanded_length)

        # If we added a time dimension for a single step, remove it for output consistency.
        if not is_seq:
            x = output.squeeze(0)  # shape: (B, Hippo_n_feature * expanded_length)
            new_rnn_states = new_rnn_states.squeeze(0)
        else:
            # log.info(new_rnn_states.size())
            new_rnn_states = new_rnn_states.squeeze(0)
            x = nn.utils.rnn.pack_padded_sequence(output, lengths, enforce_sorted=False)
            # x = output  # Preserve time dimension if multiple steps provided

        # log.info(x.size())

        return x, new_rnn_states


class SimpleSequenceWithBypassCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        log.warn("USING SIMPLESEQUENCEWITHBYPASSCORE")
        super().__init__(cfg)
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Case: head_output is a PackedSequence (multiple time steps)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass part from the recurrent state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We add a time dimension (of size 1) for in-loop updates.
            new_core_state = core_state.unsqueeze(0).clone()

            # We'll store the flattened core outputs for each time step.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)

            # Process each time step updating only the valid (sorted) indices.
            for t in range(T):
                valid_idx = sorted_indices[: batch_sizes[t]]
                # Extract the current core input for valid batch indices.
                curr_core = padded[t, valid_idx, : self.Hippo_n_feature]
                # Update the core state for these indices:
                # Roll the shift register by one.
                tmp_state = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # Zero the newly available slot.
                tmp_state[:, :, :, 0] = 0
                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)
                tmp_state[:, :, :, : self.R] += injection

                # progression = torch.argmax(torch.cat(((tmp_state != 0).to(dtype=torch.int), torch.ones(tmp_state.shape[:2] + (1,), dtype=torch.int)), dim=-1), dim=-1).squeeze(0)
                # value = torch.sum(torch.abs(progression.unsqueeze(0) - progression.unsqueeze(1)).to(dtype=torch.float))
                # log.info(f'Summed values: {value}')

                # Update the new core state for valid indices.
                new_core_state[:, valid_idx, :, :] = tmp_state
                # Save the flattened core state (for all batches) at time t.
                out_core[t] = new_core_state[0].view(B, self.core_output_size)
            # torch.save(out_core, "./train_dir/rnn_states.pt")
            # log.info(f'RNN states: {out_core}')
            # Compute the final core part of the new rnn state.
            final_core = new_core_state[0].view(B, self.core_output_size)
            # For bypass, we do not update it recurrently.
            # Instead, for each sequence we take the last valid bypass input.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature :].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape: (B, bypass_size)
            else:
                last_bypass = None

            # Form the new recurrent state by concatenating final core and bypass.
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([final_core, last_bypass], dim=1)
            else:
                new_rnn_state = final_core

            # For the output at each time step, the bypass features come directly from the current input.
            # We can concatenate the core output computed in the loop with the bypass part from the padded sequence.
            if self.bypass_size > 0:
                # padded_bypass has shape (T, B, bypass_size)
                padded_bypass = padded[:, :, self.Hippo_n_feature :]
                out_total = torch.cat([out_core, padded_bypass], dim=2)
            else:
                out_total = out_core

            # Repack the output using the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Case: head_output is a plain tensor (single time step)
            B = head_output.size(0)
            core_input = head_output[:, : self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature :] if self.bypass_size > 0 else None

            # Reshape the core part of the rnn state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Roll the shift register and zero the new slot.
            if torch.nonzero(core_state).shape[0] != 0:
                # log.info(f'Core State: {core_state}')
                # torch.save(core_state, "./train_dir/core_state.pt")
                new_core_state = core_state.roll(shifts=1, dims=-1)
                # log.info(f'New Core State: {new_core_state}')
                # torch.save(new_core_state, "./train_dir/new_core_state.pt")
            else:
                new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, : self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)
            # The new rnn state combines the updated core state with the current bypass input.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = out
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size


def straight_through_binary(x: Tensor, identity=True):
    x_binary = (x > 0).float()
    if identity:
        return x_binary - x.detach() + x


class SimpleSequenceWithBypassCore_binary(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

        self.refractory = getattr(cfg, "refractory", -1)

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Case: head_output is a PackedSequence (multiple time steps)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass part from the recurrent state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We add a time dimension (of size 1) for in-loop updates.
            new_core_state = core_state.unsqueeze(0).clone()

            # We'll store the flattened core outputs for each time step.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)

            # Process each time step updating only the valid (sorted) indices.
            for t in range(T):
                valid_idx = sorted_indices[: batch_sizes[t]]
                # Extract the current core input for valid batch indices.
                curr_core = padded[t, valid_idx, : self.Hippo_n_feature]
                # Update the core state for these indices:
                # Roll the shift register by one.
                tmp_state = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # Zero the newly available slot.
                tmp_state[:, :, :, 0] = 0

                if self.refractory != 0:

                    curr_core = straight_through_binary(
                        straight_through_binary(curr_core) - tmp_state[0, :, :, : self.refractory].sum(-1) / self.R
                    )  # .values)

                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)

                tmp_state[:, :, :, : self.R] += injection
                # Update the new core state for valid indices.
                new_core_state[:, valid_idx, :, :] = tmp_state
                # Save the flattened core state (for all batches) at time t.
                out_core[t] = new_core_state[0].view(B, self.core_output_size)
            # torch.save(out_core, "./train_dir/rnn_states.pt")
            # log.info(f'RNN states: {out_core}')
            # Compute the final core part of the new rnn state.
            final_core = new_core_state[0].view(B, self.core_output_size)
            # For bypass, we do not update it recurrently.
            # Instead, for each sequence we take the last valid bypass input.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature :].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape: (B, bypass_size)
            else:
                last_bypass = None

            # Form the new recurrent state by concatenating final core and bypass.
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([final_core, last_bypass], dim=1)
            else:
                new_rnn_state = final_core

            new_rnn_state = straight_through_binary(new_rnn_state)

            # For the output at each time step, the bypass features come directly from the current input.
            # We can concatenate the core output computed in the loop with the bypass part from the padded sequence.
            if self.bypass_size > 0:
                # padded_bypass has shape (T, B, bypass_size)
                padded_bypass = padded[:, :, self.Hippo_n_feature :]
                out_total = torch.cat([out_core, padded_bypass], dim=2)
            else:
                out_total = out_core

            out_total = straight_through_binary(out_total)
            # Repack the output using the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Case: head_output is a plain tensor (single time step)
            B = head_output.size(0)
            core_input = head_output[:, : self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature :] if self.bypass_size > 0 else None

            # Reshape the core part of the rnn state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Roll the shift register and zero the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0

            if self.refractory != 0:
                core_input = straight_through_binary(
                    straight_through_binary(core_input) - new_core_state[:, :, : self.refractory].sum(-1) / self.R
                )
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, : self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)
            # The new rnn state combines the updated core state with the current bypass input.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = out
            else:
                out = flat_core
                new_rnn_state = flat_core

            return straight_through_binary(out), straight_through_binary(new_rnn_state)

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size


class SimpleSequenceWithBypassCore_outdated(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Process the case where head_output is a PackedSequence.
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # Note: head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices).
            # We use the sorted_indices and batch_sizes to update only the valid batch entries.
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass state from the recurrent state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            bypass_state = rnn_states[:, self.core_output_size :] if self.bypass_size > 0 else None

            # We'll add a time dimension for the core state to update it at every time step.
            # new_core_state has shape (1, B, Hippo_n_feature, expanded_length)
            new_core_state = core_state.unsqueeze(0).clone()
            # For bypass, we clone so that we can update valid batch entries.
            if self.bypass_size > 0:
                new_bypass_state = bypass_state.clone()
            else:
                new_bypass_state = None

            # Prepare an output tensor to collect updated (core + bypass) outputs.
            # We'll first build the core part then concatenate the bypass features.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)
            out_total = torch.empty((T, B, self.total_output_size), device=padded.device)

            # Process each time step.
            for t in range(T):
                # For the current time step, only the first batch_sizes[t] entries are valid.
                valid_idx = sorted_indices[: batch_sizes[t]]
                # Extract current core input for the valid batch indices.
                curr_core = padded[t, valid_idx, : self.Hippo_n_feature]  # shape: (valid_count, Hippo_n_feature)
                # For bypass, if available, extract the corresponding features.
                if self.bypass_size > 0:
                    curr_bypass = padded[t, valid_idx, self.Hippo_n_feature :]  # shape: (valid_count, bypass_size)

                # Update the core state for valid indices:
                # a) roll the shift register one step to the right (along the last dimension)
                tmp_core = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # b) zero out the new slot (first position).
                tmp_core[:, :, :, 0] = 0
                # c) inject the current core input into the first R positions.
                # Expand current core input so it can be added to the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)
                tmp_core[:, :, :, : self.R] += injection
                # d) update the state.
                new_core_state[:, valid_idx, :, :] = tmp_core

                # Flatten the core state for all batches.
                flat_core = new_core_state[0].view(B, self.core_output_size)

                # For bypass: update only for the valid indices with the current bypass features.
                if self.bypass_size > 0:
                    new_bypass_state[valid_idx] = curr_bypass

                # Combine the core and bypass parts to form the total output.
                if self.bypass_size > 0:
                    combined = torch.cat([flat_core, new_bypass_state], dim=1)  # shape: (B, total_output_size)
                else:
                    combined = flat_core

                # Save the combined state as output at time step t.
                out_total[t] = combined
                # Also keep track of the core portion separately if needed.
                out_core[t] = flat_core

            # After processing the sequence, form the new rnn state.
            new_core_flat = new_core_state[0].view(B, self.core_output_size)
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([new_core_flat, new_bypass_state], dim=1)
            else:
                new_rnn_state = new_core_flat

            # Repack the output sequence. Note that we use the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Processing a single time step (plain tensor of shape (B, input_size)).
            B = head_output.size(0)
            core_input = head_output[:, : self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature :] if self.bypass_size > 0 else None

            # Reshape the core part of the recurrent state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Shift the register: roll along the last dimension and zero the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, : self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)

            # For bypass, simply take the current bypass features.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = torch.cat([flat_core, bypass_input], dim=1)
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size


class SimpleSequenceWithBypassCore_no_batch_ordering(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                        bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # For core processing, extract and reshape the recurrent state.
        # rnn_states shape: (B, total_output_size) = (B, core_output_size + bypass_size)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence: padded shape is (T, B, input_size).
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, _ = padded.shape

            # Separate the recurrent state for core processing.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We will update this core_state for each time step.
            new_core_state = core_state.clone()
            outputs = []
            # Process each time step.
            for t in range(T):
                # Split the input into core and bypass parts.
                curr_core = padded[t, :, : self.Hippo_n_feature]  # shape: (B, Hippo_n_feature)
                # Shift the core state (roll along the last dimension) and zero the new slot.
                new_core_state = new_core_state.roll(shifts=1, dims=-1)
                new_core_state[:, :, 0] = 0
                # Inject current features into the first R positions.
                injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
                new_core_state[:, :, : self.R] += injection
                # Flatten the core state.
                flat_core = new_core_state.view(B, self.core_output_size)
                # For bypass, if available, take the current bypass features.
                if self.bypass_size > 0:
                    curr_bypass = padded[t, :, self.Hippo_n_feature :]  # shape: (B, bypass_size)
                    out_t = torch.cat([flat_core, curr_bypass], dim=1)
                else:
                    out_t = flat_core
                outputs.append(out_t.unsqueeze(0))

            # Stack outputs along the time dimension.
            outputs = torch.cat(outputs, dim=0)  # shape: (T, B, total_output_size)
            # Repack the sequence.
            new_output = nn.utils.rnn.pack_padded_sequence(outputs, lengths, enforce_sorted=False)

            # For updating rnn_states, use the final core state and the last bypass features.
            if self.bypass_size > 0:
                # For each sequence, pick the last valid bypass input.
                last_bypass = []
                for i in range(B):
                    # lengths[i] is the number of time steps for sequence i.
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature :].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # (B, bypass_size)
                new_rnn_state = torch.cat([new_core_state.view(B, self.core_output_size), last_bypass], dim=1)
            else:
                new_rnn_state = new_core_state.view(B, self.core_output_size)
            return new_output, new_rnn_state

        else:
            # head_output is a plain tensor of shape (B, input_size) for a single time step.
            B = head_output.size(0)
            core_input = head_output[:, : self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature :] if self.bypass_size > 0 else None

            # Reshape the core part of the recurrent state.
            core_state = rnn_states[:, : self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Shift the register and zero out the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, : self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)

            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = torch.cat([flat_core, bypass_input], dim=1)
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug("get out size called: {self.total_output_size}")
        return self.total_output_size


class SS_Bypass_Forget_Core(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, "Hippo_R", 8)
        self.L = getattr(cfg, "Hippo_L", 48)
        self.Hippo_n_feature = getattr(cfg, "Hippo_n_feature", 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Previously, total output was core + bypass.
        self.total_output_size = self.core_output_size + self.bypass_size

        # New forget gate RNN parameters.
        self.forget_hidden_size = getattr(cfg, "forget_hidden_size", 10)  # can be adjusted
        # The overall recurrent state now includes:
        #   - core state: size = core_output_size,
        #   - bypass state: size = bypass_size,
        #   - forget gate RNN hidden state: size = forget_hidden_size.
        self.total_state_size = self.core_output_size + self.bypass_size + self.forget_hidden_size

        # Define the forget gate RNN (to be run in parallel to the simple sequence loop).
        # Note: We use an RNN (not RNNCell) so that we can process the entire sequence at once.
        self.forget_rnn = nn.RNN(
            input_size=self.bypass_size,
            hidden_size=self.forget_hidden_size,
            batch_first=False,  # input shape (T, B, input_size)
        )
        # A linear layer to reduce the RNN output to one scalar per time step.
        self.forget_linear = nn.Linear(self.forget_hidden_size, 1)

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_state_size) representing the flattened recurrent state.
                        The first core_output_size entries are the shift-register core state,
                        the next bypass_size entries are the last bypass input,
                        and the final forget_hidden_size entries are the hidden state for the forget gate RNN.
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Extract state parts.
        # core state is stored flattened; reshape into (B, Hippo_n_feature, expanded_length)
        core_state_flat = rnn_states[:, : self.core_output_size]
        core_state = core_state_flat.view(-1, self.Hippo_n_feature, self.expanded_length)
        # Bypass state: previous bypass input.
        bypass_state = rnn_states[:, self.core_output_size : self.core_output_size + self.bypass_size]
        # Forget RNN hidden state.
        forget_hidden = rnn_states[:, self.core_output_size + self.bypass_size :]

        if isinstance(head_output, PackedSequence):
            # Unpack the sequence: padded shape (T, B, input_size)
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, _ = padded.shape

            # Precompute forget gate for all time steps from the bypass features.
            if self.bypass_size > 0:
                # Get the bypass sequence from padded input: shape (T, B, bypass_size)
                bypass_seq = padded[:, :, self.Hippo_n_feature :]
                # Run the forget gate RNN in parallel.
                # RNN expects hidden state shape (num_layers, B, hidden_size)
                forget_rnn_out, forget_hidden_new = self.forget_rnn(
                    bypass_seq, forget_hidden.unsqueeze(0)
                )  # forget_rnn_out: (T, B, forget_hidden_size)
                # Linear transformation: (T, B, 1)
                forget_logits = self.forget_linear(forget_rnn_out)
                # Compute sigmoid activation.
                y_soft = torch.sigmoid(forget_logits)
                # Hardmax trick: form hard decision (threshold at 0.5) but keep gradient from y_soft.
                y_hard = (y_soft > 0.5).float()
                forget_gate_seq = y_hard - y_soft.detach() + y_soft  # (T, B, 1)
            else:
                # If no bypass features, use a gate of ones.
                forget_gate_seq = torch.ones(T, B, 1, device=padded.device)
                forget_hidden_new = forget_hidden.unsqueeze(0)  # keep dimensions consistent

            new_core_state = core_state.clone()
            outputs = []
            # Process each time step.
            for t in range(T):
                curr_full = padded[t, :, :]
                curr_core = curr_full[:, : self.Hippo_n_feature]  # (B, Hippo_n_feature)
                # Shift the core state: roll along the last dimension and zero the new slot.
                rolled_state = new_core_state.roll(shifts=1, dims=-1)
                rolled_state[:, :, 0] = 0
                # Apply the forget gate (broadcasting the (B,1) to (B, Hippo_n_feature, expanded_length))
                current_gate = forget_gate_seq[t]  # shape (B, 1)
                rolled_state = rolled_state * current_gate
                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
                rolled_state[:, :, : self.R] += injection
                new_core_state = rolled_state
                flat_core = new_core_state.view(B, self.core_output_size)
                # For bypass, use current bypass input.
                if self.bypass_size > 0:
                    curr_bypass = curr_full[:, self.Hippo_n_feature :]
                    out_t = torch.cat([flat_core, curr_bypass], dim=1)
                else:
                    out_t = flat_core
                outputs.append(out_t.unsqueeze(0))
            # Stack outputs along the time dimension.
            outputs = torch.cat(outputs, dim=0)  # shape: (T, B, total_output_size)
            new_output = nn.utils.rnn.pack_padded_sequence(outputs, lengths, enforce_sorted=False)

            # For updating rnn_states, update:
            #   - core state: final new_core_state (flattened),
            #   - bypass state: last valid bypass input per sequence,
            #   - forget hidden state: final state from the forget RNN.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature :].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape (B, bypass_size)
            else:
                last_bypass = torch.zeros(B, self.bypass_size, device=padded.device)
            new_rnn_state = torch.cat(
                [
                    new_core_state.view(B, self.core_output_size),
                    last_bypass,
                    forget_hidden_new.squeeze(0),  # updated forget hidden state
                ],
                dim=1,
            )
            return new_output, new_rnn_state

        else:
            # Single time step: head_output shape (B, input_size)
            B = head_output.size(0)
            curr_core = head_output[:, : self.Hippo_n_feature]
            curr_bypass = head_output[:, self.Hippo_n_feature :] if self.bypass_size > 0 else None

            if self.bypass_size > 0:
                # Process the single bypass input with the forget RNN.
                # Prepare input shape (1, B, bypass_size) and hidden shape (1, B, forget_hidden_size)
                bypass_input = curr_bypass.unsqueeze(0)
                forget_rnn_out, new_forget_hidden = self.forget_rnn(
                    bypass_input, forget_hidden.unsqueeze(0)
                )  # forget_rnn_out: (1, B, forget_hidden_size)
                forget_logits = self.forget_linear(forget_rnn_out)  # (1, B, 1)
                y_soft = torch.sigmoid(forget_logits)
                y_hard = (y_soft > 0.5).float()
                forget_gate = y_hard - y_soft.detach() + y_soft  # (1, B, 1)
                # Remove the time dimension.
                forget_gate = forget_gate.squeeze(0)  # (B, 1)
                new_forget_hidden = new_forget_hidden.squeeze(0)  # (B, forget_hidden_size)
            else:
                forget_gate = torch.ones(B, 1, device=head_output.device)
                new_forget_hidden = forget_hidden  # unchanged

            # Shift core state.
            rolled_state = core_state.roll(shifts=1, dims=-1)
            rolled_state[:, :, 0] = 0
            # Apply the forget gate.
            rolled_state = rolled_state * forget_gate
            # Inject the current core input.
            injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
            rolled_state[:, :, : self.R] += injection
            new_core_state = rolled_state
            flat_core = new_core_state.view(B, self.core_output_size)

            if self.bypass_size > 0:
                out = torch.cat([flat_core, curr_bypass], dim=1)
            else:
                out = flat_core
            new_rnn_state = torch.cat(
                [
                    flat_core,
                    (
                        curr_bypass
                        if self.bypass_size > 0
                        else torch.zeros(B, self.bypass_size, device=head_output.device)
                    ),
                    new_forget_hidden,
                ],
                dim=1,
            )
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug("get out size called: {self.total_output_size}")
        return self.total_output_size


class NoveltyCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: the Sample Factory configuration (unused here but normally provided)
            input_size (int): dimensionality of the input feature vector.
            novelty_thr (float): threshold to decide whether to add a new feature.
        """
        super().__init__(cfg)
        self.dim = input_size
        if cfg.novelty_thr:
            self.novelty_thr = cfg.novelty_thr
        else:
            self.novelty_thr = 0.4
            raise Warning("Novelty Threshold not set, using default: {0.4}")
        # These will be initialized on the first forward call
        self.N = None  # Tensor of shape (B, dim, dim)
        self.w_list = None  # List of length B; each entry is a tensor of shape (dim, num_features)

    def _initialize_state(self, batch_size, device):
        # Initialize per-sample state for a new batch.
        self.N = torch.eye(self.dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        # For each sample, we start with an empty feature bank (we use a list per sample)
        self.w_list = [None for _ in range(batch_size)]

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Tensor of shape (B, dim) – output from encoder.
            rnn_states: (ignored here; passed through unchanged)

        Returns:
            pred: Tensor of shape (B, 1) of novelty predictions.
            rnn_states: Passed through unchanged.
        """
        B = head_output.size(0)
        device = head_output.device

        # If state is not initialized or batch size has changed, initialize new state.
        if (self.N is None) or (self.N.size(0) != B):
            self._initialize_state(B, device)

        # Reshape head_output to (B, dim, 1)
        x = head_output.unsqueeze(-1)  # (B, dim, 1)

        # Compute variance for each sample: v = x^T N x  -> shape (B, 1, 1)
        v = torch.bmm(x.transpose(1, 2), torch.bmm(self.N, x))
        norm_val = torch.sqrt(v).squeeze(-1)  # shape (B, dim?) -> (B, 1) ideally; we squeeze the last dim
        # Ensure norm_val is of shape (B, 1)
        norm_val = norm_val if norm_val.dim() == 2 else norm_val.unsqueeze(-1)

        # Compute the orthogonalized x: (B, dim, 1) = N @ x / norm_val (broadcasted)
        x_orth = torch.bmm(self.N, x) / norm_val

        # Compute correlation matrix: (B, dim, dim) = x_orth @ x_orth^T
        x_corr = torch.bmm(x_orth, x_orth.transpose(1, 2))

        # Update state matrix N for each sample
        self.N = self.N - x_corr

        # Compute norm of x for each sample (using torch.norm along dim=1)
        x_norm = x.norm(dim=1, keepdim=True)  # shape (B, 1, 1)
        x_norm = x_norm.squeeze(-1)  # (B, 1)

        # Prepare a list to collect predictions for each sample
        preds = []
        # Loop over batch dimension (typically B is small, so this loop is acceptable)
        for i in range(B):
            # For sample i, check if novelty condition is met.
            # norm_val[i] is scalar; x_norm[i] is scalar.
            if (norm_val[i] / (x_norm[i] + 1e-8)) > self.novelty_thr:
                # Compute a new feature vector for sample i:
                # Here we compute: new_w = (1 - (w^T x)) * (x_orth / norm_val)
                # If there are no existing features, treat the scalar factor as 1.
                if self.w_list[i] is None:
                    w_new = x_orth[i] / norm_val[i]
                    self.w_list[i] = w_new  # shape (dim, 1)
                else:
                    # Compute average correlation of existing features with x.
                    # self.w_list[i] is (dim, num_features)
                    prod = torch.matmul(self.w_list[i].transpose(0, 1), x[i])  # shape (num_features, 1)
                    scalar = 1 - prod.mean()
                    w_new = scalar * (x_orth[i] / norm_val[i])
                    self.w_list[i] = torch.cat([self.w_list[i], w_new], dim=1)
            # Compute prediction for sample i: if no features stored, predict 0; else, use the maximum dot product.
            if self.w_list[i] is None:
                pred_i = torch.tensor(0.0, device=device)
            else:
                pred_i = torch.matmul(self.w_list[i].transpose(0, 1), x[i]).max()
            preds.append(pred_i)

        # Stack predictions to a tensor of shape (B, 1)
        pred_tensor = torch.stack(preds).unsqueeze(-1)

        return pred_tensor, rnn_states


# Example usage in a custom model:
# In your custom Actor-Critic model you would register this core:
#
#   self.core = NoveltyCore(cfg, input_size=self.encoder.get_out_size(), novelty_thr=0.4)
#
# and then in your forward() method, you would call:
#
#   core_output, new_rnn_states = self.core(encoder_output, rnn_states)
#
# This demonstrates how you can incorporate your novelty operation as the core of your model.


class ModelHipposlam(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


def make_hipposlam_core(cfg: Config, core_input_size: int) -> ModelCore:
    if cfg.core_name:
        if cfg.core_name == "simple_sequence":
            core = SimpleSequenceCore(cfg, core_input_size)
        elif cfg.core_name == "fixed_rnn":
            core = FixedRNNSequenceCore(cfg, core_input_size)
        elif cfg.core_name == "BypassFixedRNN":
            core = FixedRNNWithBypassCore(cfg, core_input_size)
        elif cfg.core_name == "BypassSS":
            core = SimpleSequenceWithBypassCore(cfg, core_input_size)
        elif cfg.core_name == "BypassSS_binary":
            core = SimpleSequenceWithBypassCore_binary(cfg, core_input_size)
        elif cfg.core_name == "Default":
            core = ModelCoreRNN(cfg, core_input_size)
    else:
        core = ModelCoreIdentity(cfg, core_input_size)

    return core
