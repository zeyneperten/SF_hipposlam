class HighLevelRNNWrapperCore(ModelCore):
    """
    Stage 1: High-Level RNN Wrapper.
    """

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.K = getattr(cfg, "hl_K", 4)
        self.d_H = getattr(cfg, "hl_d_H", 16)
        self.oracle = getattr(cfg, "oracle_context", False)

        # 1. Instantiate the base core
        # Subtract 3 because outcome_event, prev_trial_reward, and chosen_arm don't go to base
        base_input_size = input_size - 3
        if self.oracle:
            base_input_size -= 1  # Subtract 1 more for inst_block if oracle is used

        self.base_core = SimpleSequenceWithBypassCore(cfg, input_size=base_input_size)

        # The SimpleSequenceWithBypassCore state size equals its total_output_size
        self.base_state_size = self.base_core.total_output_size

        # 2. Instantiate the custom High-Level RNN
        self.high_level_rnn = HighLevelContextRNN_Stage1(K=self.K, d_H=self.d_H)
        self.high_level_rnn_Q = HighLevelContextRNN_QLearning(K=self.K, d_H=self.d_H)  # Use the Q-learning version
        self.high_level_rnn_PG = HighLevelContextRNN_Policy(K=self.K, d_H=self.d_H)  # Use the Policy Gradient version
        # 3. Sizes
        self.total_state_size = self.base_state_size + self.d_H + self.K
        self.total_output_size = self.base_core.total_output_size + self.K

    def get_core_state_size(self):
        return self.total_state_size

    def get_out_size(self):
        return self.total_output_size

    def forward(self, head_output, rnn_states):
        # --- Safely handle PackedSequences ---
        is_packed = isinstance(head_output, PackedSequence)
        if is_packed:
            head_output, lengths = pad_packed_sequence(head_output)

        is_bptt = head_output.dim() == 3

        # Extract states
        base_states = rnn_states[:, : self.base_state_size]
        h_high_prev = rnn_states[:, self.base_state_size : self.base_state_size + self.d_H]
        z_prev = rnn_states[:, self.base_state_size + self.d_H : self.base_state_size + self.d_H + self.K]

        if not is_bptt:
            # === INFERENCE PASS ===
            if self.oracle:
                # Oracle mode has 4 extra variables appended
                base_head_output = head_output[:, :-4]
                inst_block = head_output[:, -4].long()
                chosen_arm = head_output[:, -3].long()
                outcome_mask = head_output[:, -2].bool()
                prev_trial_reward = head_output[:, -1]

                # (Make sure Stage 1 forward() accepts inst_block!)
                h_high_new, z_new = self.high_level_rnn(
                    outcome_mask, prev_trial_reward, h_high_prev, z_prev, inst_block
                )
            else:
                # Q-Learning mode only has 3 extra variables appended
                base_head_output = head_output[:, :-3]
                chosen_arm = head_output[:, -3].long()
                outcome_mask = head_output[:, -2].bool()
                prev_trial_reward = head_output[:, -1]

                # (Make sure Q-Learning forward() accepts chosen_arm!)
                h_high_new, z_new, _, _, _ = self.high_level_rnn_Q(
                    outcome_mask, prev_trial_reward, h_high_prev, z_prev, chosen_arm
                )

            # Base Core Step
            base_core_out, base_states_new = self.base_core(base_head_output, base_states)

            # log.warning(f"Base Core Out Shape: {base_core_out.shape}")
            # log.warning(f"z_new Shape: {z_new.shape}")

            # Pack Outputs
            out_total = torch.cat([base_core_out, z_new], dim=-1)
            new_rnn_states = torch.cat([base_states_new, h_high_new, z_new], dim=-1)

            self.last_q_loss = torch.tensor(0.0, device=head_output.device)
            self.last_log_dict = None  # Prevent crashes during inference

            if is_packed:
                out_total = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)

            return out_total, new_rnn_states

        else:
            # === BPTT PASS ===
            T, B = head_output.shape[:2]

            h_high_t = h_high_prev
            z_t = z_prev
            base_states_t = base_states

            out_total_list = []

            if not self.oracle:
                scores_seq = []
                z_indices_seq = []
                rewards_seq = []
                mask_seq = []

            for t in range(T):
                if self.oracle:
                    base_head_output_t = head_output[t, :, :-4]
                    inst_block_t = head_output[t, :, -4].long()
                    chosen_arm_t = head_output[t, :, -3].long()
                    outcome_mask_t = head_output[t, :, -2].bool()
                    prev_trial_reward_t = head_output[t, :, -1]

                    h_high_t, z_t = self.high_level_rnn(
                        outcome_mask_t, prev_trial_reward_t, h_high_t, z_t, inst_block_t
                    )
                else:
                    base_head_output_t = head_output[t, :, :-3]
                    chosen_arm_t = head_output[t, :, -3].long()
                    outcome_mask_t = head_output[t, :, -2].bool()
                    prev_trial_reward_t = head_output[t, :, -1]

                    # 1. Grab Q-scores for OLD state
                    q_scores_old = self.high_level_rnn_Q.q_head(h_high_t)

                    scores_seq.append(q_scores_old)
                    z_indices_seq.append(z_t.argmax(dim=-1))
                    rewards_seq.append(prev_trial_reward_t)
                    mask_seq.append(outcome_mask_t)

                    # 2. Step High-Level RNN
                    h_high_t, z_t, q_scores_new, new_state_dict, log_dict_t = self.high_level_rnn_Q(
                        outcome_mask_t, prev_trial_reward_t, h_high_t, z_t, chosen_arm_t
                    )

                # Base Core Step
                base_core_out_t, base_states_t = self.base_core(base_head_output_t, base_states_t)

                out_total_t = torch.cat([base_core_out_t, z_t], dim=-1)
                out_total_list.append(out_total_t)

            out_total_seq = torch.stack(out_total_list, dim=0)
            new_rnn_states = torch.cat([base_states_t, h_high_t, z_t], dim=-1)

            # Process Q-loss ONLY if NOT oracle!
            if not self.oracle:
                scores_stacked = torch.stack(scores_seq, dim=0)
                z_indices_stacked = torch.stack(z_indices_seq, dim=0)
                rewards_stacked = torch.stack(rewards_seq, dim=0)
                mask_stacked = torch.stack(mask_seq, dim=0)

                total_q_loss = q_loss(
                    scores=scores_stacked,
                    z_indices=z_indices_stacked,
                    rewards=rewards_stacked,
                    decision_mask=mask_stacked,
                )

                self.last_q_loss = total_q_loss
                self.last_log_dict = log_dict_t
            else:
                self.last_q_loss = torch.tensor(0.0, device=head_output.device)
                self.last_log_dict = None

            if is_packed:
                out_total_seq = nn.utils.rnn.pack_padded_sequence(out_total_seq, lengths, enforce_sorted=False)

            return out_total_seq, new_rnn_states
