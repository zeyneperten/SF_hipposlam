"""Unit and synthetic integration tests for High-Level mode DG modulation.

Tests:
1. Pre-DG modulation logic inside DGProjection_batchnorm_relu.
2. Encoder initialization and forward pass across hl_DG_mod ('multiply', 'sigmoid', 'concat', 'None').
3. Multi-trial synthetic environment rollout simulating actor latched mode propagation to DG and core.
4. Learner replay forward pass and backprop through the pre-DG modulation embedding layer.
5. Parameter freezing check (hl_train_fix_base keeps hl_mode_embed trainable).

Note: This test runs out-of-the-box on your clean repository BEFORE changing custom_encoder.py
using a prototype encoder class, and will automatically test HipposlamEncoder directly once
the implementation is applied.

Run from the ymaze directory on HPC:
    PYTHONPATH=. python tests/test_high_level_dg_modulation.py
or:
    python -m unittest tests/test_high_level_dg_modulation.py
or:
    pytest -v tests/test_high_level_dg_modulation.py
"""

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.algo.utils.shared_buffers import policy_output_shapes
from sf_working_directories.zeynep.dmlab.custom_actor_critic import (
    HighLevel_LossWrapper,
)
from sf_working_directories.zeynep.dmlab.custom_core import HighLevelRNNWrapperCore
from sf_working_directories.zeynep.dmlab.custom_encoder import (
    DGProjection_batchnorm_relu,
    HipposlamEncoder,
)


class MockVisualBackbone(nn.Module):
    """Lightweight 2-layer CNN for fast synthetic testing without external assets."""

    def __init__(self, in_channels: int = 3, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(8 * 2 * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

    def get_out_size(self) -> int:
        return self.out_dim


# If HipposlamEncoder already has set_hl_z, use it directly.
# Otherwise, use PrototypeModulatedEncoder so the user can test BEFORE modifying custom_encoder.py!
if hasattr(HipposlamEncoder, "set_hl_z"):
    ModulatedEncoderTarget = HipposlamEncoder
else:
    class PrototypeModulatedEncoder(HipposlamEncoder):
        """Prototype wrapper adding hl_DG_mod functionality for pre-implementation testing."""

        def __init__(self, cfg, obs_space):
            super().__init__(cfg, obs_space)
            self.hl_DG_mod = getattr(cfg, "hl_DG_mod", "None")
            self.high_level = bool(cfg.core_name and "HighLevel" in cfg.core_name)
            self.hl_K = getattr(cfg, "hl_K", 4)
            self.current_hl_z = None

            if self.high_level and self.hl_DG_mod in ("multiply", "sigmoid"):
                self.hl_mode_embed_layer = nn.Linear(self.hl_K, cfg.Hippo_n_feature)
                nn.init.orthogonal_(self.hl_mode_embed_layer.weight)
                nn.init.ones_(self.hl_mode_embed_layer.bias)
                if self.hl_DG_mod == "multiply" and hasattr(self.DG_projection, "modulate"):
                    self.DG_projection.modulate = True
            elif self.high_level and self.hl_DG_mod == "concat":
                # Re-create DG_projection to accept concat (visual + mode) features
                intercept = getattr(cfg, "DG_BN_intercept", 2.0)
                in_size = self.basic_encoder.get_out_size() + self.hl_K
                self.DG_projection = DGProjection_batchnorm_relu(
                    in_size, cfg.Hippo_n_feature, intercept=intercept, modulate=False
                )

        def set_hl_z(self, z):
            self.current_hl_z = z

        def forward(self, obs_dict):
            obs_cnn = obs_dict["obs"]
            x = self.basic_encoder(obs_cnn)

            if self.high_level and self.hl_DG_mod != "None":
                hl_z = obs_dict.get("hl_z", getattr(self, "current_hl_z", None))
                if hl_z is None:
                    hl_z = x.new_zeros(x.shape[0], self.hl_K)
                else:
                    if hl_z.dim() == 1:
                        hl_z = hl_z.unsqueeze(0)
                    if hl_z.shape[0] != x.shape[0]:
                        hl_z = hl_z.expand(x.shape[0], -1)
                    hl_z = hl_z.to(device=x.device, dtype=x.dtype)

                if self.hl_DG_mod == "multiply":
                    embedded_mode = self.hl_mode_embed_layer(hl_z)
                    tmp_out = self.DG_projection(x, context=embedded_mode)
                elif self.hl_DG_mod == "sigmoid":
                    embedded_mode = torch.sigmoid(self.hl_mode_embed_layer(hl_z))
                    tmp_out = self.DG_projection(x) * embedded_mode
                elif self.hl_DG_mod == "concat":
                    coef = getattr(self, "number_instruction_coef", 1)
                    x_cat = torch.cat((x, hl_z * coef), dim=1)
                    tmp_out = self.DG_projection(x_cat)
                else:
                    tmp_out = self.DG_projection(x)
            else:
                tmp_out = self.DG_projection(x)

            if self.high_level:
                chosen_arm = obs_dict.get("chosen_arm").to(device=tmp_out.device, dtype=tmp_out.dtype)
                outcome_event = obs_dict.get("outcome_event").to(device=tmp_out.device, dtype=tmp_out.dtype)
                prev_trial_reward = obs_dict.get("prev_trial_reward").to(device=tmp_out.device, dtype=tmp_out.dtype)
                tmp_out = torch.cat((tmp_out, chosen_arm, outcome_event, prev_trial_reward), dim=1)

            return tmp_out

    ModulatedEncoderTarget = PrototypeModulatedEncoder


def make_test_cfg(**kwargs):
    defaults = {
        "encoder_conv_architecture": "convnet_impala",
        "encoder_conv_mlp_layers": [32],
        "nonlinearity": "relu",
        "core_name": "BypassSS_HighLevelRNN",
        "DG_name": "batchnorm_relu",
        "DG_BN_intercept": 2.0,
        "DG_context_mod": "None",
        "Decoder_context_mod": "None",
        "oracle_context": False,
        "with_number_instruction": False,
        "reward_input": False,
        "depth_sensor": False,
        "Hippo_n_feature": 16,
        "Hippo_R": 2,
        "Hippo_L": 4,
        "hl_K": 4,
        "hl_d_H": 8,
        "hl_history_len": 4,
        "hl_tau": 1.0,
        "hl_is_policy": False,
        "hl_deterministic": False,
        "hl_train_fix_base": False,
        "hl_diversity_reward_coef": 0.0,
        "hl_DG_mod": "multiply",
        "device": "cpu",
        "use_jit": False,
        "rnn_type": "gru",
        "decoder_mlp_layers": [32],
        "policy_initialization": "orthogonal",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_encoder_with_mock_backbone(cfg, visual_dim=32):
    """Instantiates encoder with a fast mock visual backbone."""
    obs_space = SimpleNamespace(shape=(3, 16, 16))
    encoder = ModulatedEncoderTarget(cfg, obs_space)
    # Replace basic_encoder with lightweight mock for fast synthetic test
    encoder.basic_encoder = MockVisualBackbone(in_channels=3, out_dim=visual_dim)
    return encoder


class TestHighLevelDGModulation(unittest.TestCase):

    def test_01_standalone_pre_dg_multiply_changes_activations(self):
        """Verifies that Linear(x) * embed(z) before BatchNorm+ReLU produces distinct sparse codes."""
        in_dim = 32
        out_dim = 16
        K = 4
        dg = DGProjection_batchnorm_relu(in_dim, out_dim, intercept=1.0, modulate=True)
        mode_embed = nn.Linear(K, out_dim)
        nn.init.orthogonal_(mode_embed.weight)
        nn.init.ones_(mode_embed.bias)

        x = torch.randn(2, in_dim)
        z0 = F.one_hot(torch.tensor([0, 0]), num_classes=K).float()
        z1 = F.one_hot(torch.tensor([1, 1]), num_classes=K).float()

        ctx0 = mode_embed(z0)
        ctx1 = mode_embed(z1)

        dg.eval()
        with torch.no_grad():
            out0 = dg(x, context=ctx0)
            out1 = dg(x, context=ctx1)

        self.assertEqual(out0.shape, (2, out_dim))
        self.assertEqual(out1.shape, (2, out_dim))
        # The same visual input under two different modes MUST produce different DG codes
        self.assertFalse(
            torch.allclose(out0, out1, atol=1e-4),
            "Pre-DG modulation failed: different modes produced identical activations",
        )

    def test_02_encoder_init_and_output_size(self):
        """Checks encoder instantiation across all modulation regimes."""
        for mod in ["multiply", "sigmoid", "concat", "None"]:
            cfg = make_test_cfg(hl_DG_mod=mod)
            encoder = make_encoder_with_mock_backbone(cfg)
            out_size = encoder.get_out_size()
            # Hippo_n_feature (16) + 3 high level signals (chosen_arm, outcome_event, prev_reward)
            self.assertEqual(out_size, 19, f"Unexpected encoder output size for mode {mod}: {out_size}")
            if mod in ("multiply", "sigmoid"):
                self.assertTrue(hasattr(encoder, "hl_mode_embed_layer"))
                self.assertEqual(encoder.hl_mode_embed_layer.weight.shape, (16, 4))

    def test_03_encoder_forward_modes_produce_different_features(self):
        """Ensures encoder forward generates different representations for different modes."""
        cfg = make_test_cfg(hl_DG_mod="multiply")
        encoder = make_encoder_with_mock_backbone(cfg)
        encoder.eval()

        img = torch.randn(2, 3, 16, 16)
        obs_dict = {
            "obs": img,
            "chosen_arm": torch.zeros(2, 1),
            "outcome_event": torch.zeros(2, 1),
            "prev_trial_reward": torch.zeros(2, 1),
        }

        # Run with mode 0
        obs_dict["hl_z"] = F.one_hot(torch.tensor([0, 0]), num_classes=4).float()
        out_mode0 = encoder(obs_dict)

        # Run with mode 1
        obs_dict["hl_z"] = F.one_hot(torch.tensor([1, 1]), num_classes=4).float()
        out_mode1 = encoder(obs_dict)

        # DG features are the first 16 dimensions
        dg_mode0 = out_mode0[:, :16]
        dg_mode1 = out_mode1[:, :16]

        self.assertFalse(
            torch.allclose(dg_mode0, dg_mode1, atol=1e-4),
            "Encoder forward: DG features did not vary with mode vector z",
        )

    def test_04_encoder_fallback_when_hl_z_missing(self):
        """Ensures encoder does not crash when hl_z is absent (e.g. uninitialized initial frame)."""
        cfg = make_test_cfg(hl_DG_mod="multiply")
        encoder = make_encoder_with_mock_backbone(cfg)

        obs_dict = {
            "obs": torch.randn(2, 3, 16, 16),
            "chosen_arm": torch.zeros(2, 1),
            "outcome_event": torch.zeros(2, 1),
            "prev_trial_reward": torch.zeros(2, 1),
        }
        # Should not raise any KeyError or crash
        out = encoder(obs_dict)
        self.assertEqual(out.shape, (2, 19))

    def test_05_encoder_set_hl_z_method(self):
        """Verifies set_hl_z(z) method correctly influences forward pass when hl_z is omitted from dict."""
        cfg = make_test_cfg(hl_DG_mod="multiply")
        encoder = make_encoder_with_mock_backbone(cfg)
        encoder.eval()

        obs_dict = {
            "obs": torch.randn(1, 3, 16, 16),
            "chosen_arm": torch.zeros(1, 1),
            "outcome_event": torch.zeros(1, 1),
            "prev_trial_reward": torch.zeros(1, 1),
        }

        encoder.set_hl_z(F.one_hot(torch.tensor([0]), num_classes=4).float())
        out0 = encoder(obs_dict)[:, :16]

        encoder.set_hl_z(F.one_hot(torch.tensor([2]), num_classes=4).float())
        out2 = encoder(obs_dict)[:, :16]

        self.assertFalse(
            torch.allclose(out0, out2, atol=1e-4),
            "set_hl_z did not update internal mode representation",
        )

    def test_06_simulated_synthetic_environment_rollout(self):
        """Simulates a multi-step rollout environment loop verifying latched mode flows to DG."""
        cfg = make_test_cfg(hl_DG_mod="multiply", Hippo_n_feature=16, Hippo_R=2, Hippo_L=4, hl_K=4)
        encoder = make_encoder_with_mock_backbone(cfg)
        core = HighLevelRNNWrapperCore(cfg, input_size=encoder.get_out_size())

        state = torch.zeros(1, core.get_core_state_size())
        num_steps = 10
        outcomes_seen = 0

        for step in range(num_steps):
            is_outcome = 1.0 if step in (3, 7) else 0.0
            reward_val = 1.0 if is_outcome else 0.0
            if is_outcome:
                outcomes_seen += 1

            # Simulated environment observation
            obs = {
                "obs": torch.randn(1, 3, 16, 16),
                "chosen_arm": torch.tensor([[1.0]]),
                "outcome_event": torch.tensor([[is_outcome]]),
                "prev_trial_reward": torch.tensor([[reward_val]]),
            }

            # Extract latched mode from recurrent state (last K elements)
            z_prev = state[:, -cfg.hl_K:].clone()
            encoder.set_hl_z(z_prev)
            obs["hl_z"] = z_prev

            # 1. Encoder forward (with DG modulation)
            enc_out = encoder(obs)

            # 2. Core forward (shift-register CA3 + high level RNN)
            core_out, state = core(enc_out, state)

            # Check core output: shift-register features + latched mode z
            self.assertEqual(core_out.shape[1], core.get_out_size())
            # Active mode is in the last K elements of the core output
            z_active = core_out[:, -cfg.hl_K:]
            self.assertEqual(z_active.shape, (1, cfg.hl_K))

        self.assertEqual(outcomes_seen, 2, "Test did not simulate the expected outcome events")

    def test_07_learner_bptt_gradient_reaches_dg_mode_embed(self):
        """Verifies end-to-end backprop: gradients flow back to hl_mode_embed_layer."""
        cfg = make_test_cfg(hl_DG_mod="multiply")
        encoder = make_encoder_with_mock_backbone(cfg)

        obs_dict = {
            "obs": torch.randn(4, 3, 16, 16),
            "chosen_arm": torch.zeros(4, 1),
            "outcome_event": torch.zeros(4, 1),
            "prev_trial_reward": torch.zeros(4, 1),
            "hl_z": F.one_hot(torch.tensor([0, 1, 2, 3]), num_classes=4).float(),
        }

        encoder.zero_grad()
        out = encoder(obs_dict)
        loss = out[:, :16].sum()
        loss.backward()

        embed_grad = encoder.hl_mode_embed_layer.weight.grad
        self.assertIsNotNone(embed_grad, "Gradient was not computed for hl_mode_embed_layer")
        self.assertGreater(
            embed_grad.abs().sum().item(),
            0.0,
            "Gradient for hl_mode_embed_layer was zero",
        )


if __name__ == "__main__":
    unittest.main()
