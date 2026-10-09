"""Controller-transfer checks that do not require DeepMind Lab."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import torch
from torch import nn

from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.learning.learner import DefaultLearner

# The focused encoder test substitutes its image backbone; torchvision is not
# needed on the lightweight test host.
if "torchvision" not in sys.modules:
    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.ModuleType("torchvision.models")
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.models"] = torchvision.models
from sf_working_directories.zeynep.dmlab import custom_encoder
from sf_working_directories.zeynep.dmlab import custom_actor_critic
from sf_working_directories.zeynep.dmlab.custom_decoder import (
    MlpDecoderAdditiveJit, MlpDecoderFiLMJit,
)
from sf_working_directories.zeynep.dmlab.oracle_transfer import (
    load_oracle_controller, prepare_oracle_transfer,
)
from sf_working_directories.zeynep.dmlab.oracle_transfer_learner import OracleTransferLearner


class ToyImageEncoder(nn.Module):
    def get_out_size(self):
        return 3

    def forward(self, image):
        return image[:, :3, 0, 0]


class ToyDepthEncoder(nn.Module):
    def __init__(self, cfg, size=10):
        super().__init__()

    def get_out_size(self):
        return 10

    def forward(self, image):
        return image[:, 0, 0, 0].unsqueeze(1).expand(-1, 10)


def encoder_cfg(mod, oracle):
    return SimpleNamespace(
        Hippo_n_feature=2, depth_sensor=True, res_h=2, res_w=2,
        DG_context_mod=mod, Decoder_context_mod="FiLM",
        oracle_context=oracle, hl_dg_from_prev_z=not oracle,
        with_number_instruction=True, number_instruction_coef=9,
        hl_K=4 if oracle else 2, reward_input=False,
        DG_lr=None, DG_temperature=None, DG_batch_q=None, DG_softmax=None,
        DG_name="batchnorm_relu", DG_BN_intercept=0.0,
        core_name="BypassSS_HighLevelRNN",
    )


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_core = nn.Linear(3, 3)
        self.hl_learner = nn.Linear(3, 2)
        self.high_level_rnn_stage1 = nn.Linear(3, 3)


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.basic_encoder = nn.Linear(3, 3)
        self.encoder.DG_projection = nn.Linear(3, 2)
        self.core = ToyCore()
        self.decoder = nn.Module()
        self.decoder.mlp = nn.Linear(3, 3)
        self.decoder.film_gamma = nn.Linear(2, 3)
        self.decoder.film_beta = nn.Linear(2, 3)
        self.action_parameterization = nn.Linear(3, 2)
        self.critic_linear = nn.Linear(3, 1)


class OracleTransferTests(unittest.TestCase):
    def test_prior_mode_matches_oracle_dg_for_every_supported_variant(self):
        obs_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 255, (4, 2, 2))})
        obs = {
            "obs": torch.ones(2, 4, 2, 2),
            custom_encoder.DMLAB_INSTRUCTIONS: torch.tensor([[1], [2]]),
            "inst_block": torch.tensor([[1], [2]]),
            "chosen_arm": torch.zeros(2, 1),
            "outcome_event": torch.zeros(2, 1),
            "prev_trial_reward": torch.zeros(2, 1),
        }
        mode = torch.eye(2)
        with patch.object(custom_encoder, "make_img_encoder", return_value=ToyImageEncoder()), \
             patch.object(custom_encoder, "DepthEncoder", ToyDepthEncoder):
            for variant in ("concat", "multiply", "sigmoid"):
                with self.subTest(variant=variant):
                    source = custom_encoder.HipposlamEncoder(encoder_cfg(variant, True), obs_space)
                    target = custom_encoder.HipposlamEncoder(encoder_cfg(variant, False), obs_space)
                    target.DG_projection.load_state_dict(source.DG_projection.state_dict())
                    if hasattr(source, "instruction_embed_layer"):
                        target.instruction_embed_layer.load_state_dict(source.instruction_embed_layer.state_dict())
                    source.eval()
                    target.eval()
                    self.assertTrue(torch.allclose(source(obs)[:, :2], target(obs, previous_mode=mode)[:, :2]))
                    self.assertTrue(torch.equal(target(obs, previous_mode=mode)[:, 12:14], mode * 9))

    def test_actor_head_uses_saved_prior_mode(self):
        class Dummy:
            cfg = SimpleNamespace(hl_dg_from_prev_z=True, hl_K=2)

            def encoder(self, obs, previous_mode=None):
                return previous_mode

        state = torch.tensor([[5.0, 1.0, 0.0], [5.0, 0.0, 1.0]])
        output = ActorCriticSharedWeights.forward_head(Dummy(), {}, state)
        self.assertTrue(torch.equal(output, state[:, -2:]))
        self.assertRaises(ValueError, ActorCriticSharedWeights.forward_head, Dummy(), {})

    def test_decoder_and_action_logits_keep_oracle_modes_zero_and_one(self):
        cfg = AttrConfig(decoder_mlp_layers=[5], use_jit=False,
                         nonlinearity="relu", context_injection_coef=1.7)
        base = torch.randn(2, 3)
        oracle_context = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        learned_context = oracle_context[:, :2]
        for decoder_class in (MlpDecoderFiLMJit, MlpDecoderAdditiveJit):
            with self.subTest(decoder=decoder_class.__name__):
                source = decoder_class(cfg, 3, context_dim=4)
                target = decoder_class(cfg, 3, context_dim=2)
                source_state = source.state_dict()
                target_state = target.state_dict()
                for key, value in source_state.items():
                    if value.ndim == 2 and value.shape[1] == 4 and target_state[key].shape[1] == 2:
                        value = value[:, :2]
                    target_state[key] = value
                target.load_state_dict(target_state)
                source.eval()
                target.eval()
                action_head = nn.Linear(5, 3)
                oracle_logits = action_head(source(torch.cat((base, oracle_context), 1)))
                learned_logits = action_head(target(torch.cat((base, learned_context), 1)))
                self.assertTrue(torch.allclose(oracle_logits, learned_logits, atol=1e-6))

    def test_config_and_weight_conversion_leave_manager_new(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            policy = run / "checkpoint_p2"
            policy.mkdir()
            checkpoint = policy / "best_0001.pth"
            source_cfg = {
                "core_name": "BypassSS_HighLevelRNN", "oracle_context": True,
                "with_number_instruction": True, "hl_K": 4, "decoder_type": "mlp",
                "actor_critic_share_weights": True, "DG_context_mod": "concat",
                "Decoder_context_mod": "FiLM", "Hippo_n_feature": 2, "Hippo_L": 3,
                "Hippo_R": 1, "depth_sensor": True, "number_instruction_coef": 200,
            }
            (run / "config.json").write_text(json.dumps(source_cfg))
            cfg = dict(source_cfg, oracle_init_checkpoint=str(checkpoint),
                       oracle_freeze_controller=True, hl_diversity_reward_coef=0.0,
                       hl_d_H=3, hl_history_len=2, cli_args={})
            cfg = SimpleNamespace(**cfg)
            actor = ToyActor()
            source = {key: value.clone() for key, value in actor.state_dict().items()}
            source["decoder.film_gamma.weight"] = torch.cat((source["decoder.film_gamma.weight"], torch.ones(3, 2)), 1)
            source["decoder.film_beta.weight"] = torch.cat((source["decoder.film_beta.weight"], torch.ones(3, 2)), 1)
            torch.save({"model": source, "optimizer": {"unused": True}, "train_step": 99}, checkpoint)
            prepare_oracle_transfer(cfg)
            self.assertEqual(cfg.hl_K, 2)
            self.assertEqual(cfg.rnn_size, 37)
            manager_before = actor.core.hl_learner.weight.detach().clone()
            copied, converted = load_oracle_controller(actor, checkpoint)
            self.assertIn("decoder.film_gamma.weight", converted)
            self.assertTrue(torch.equal(actor.decoder.film_gamma.weight, source["decoder.film_gamma.weight"][:, :2]))
            self.assertTrue(torch.equal(actor.core.hl_learner.weight, manager_before))
            self.assertIn("encoder.DG_projection.weight", copied)
            source["encoder.DG_projection.weight"] = torch.zeros(1, 1)
            torch.save({"model": source}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Incompatible oracle tensor"):
                load_oracle_controller(actor, checkpoint)

    def test_frozen_controller_keeps_batchnorm_statistics_fixed(self):
        cfg = AttrConfig(oracle_init_checkpoint="source.pth", oracle_freeze_controller=True,
                         hl_train_fix_base=False, hl_diversity_reward_coef=0.0,
                         core_name="BypassSS_HighLevelRNN")
        actor = ToyActor()
        actor.obs_normalizer = nn.BatchNorm1d(3)
        actor.encoder.bn = nn.BatchNorm1d(3)
        actor.cfg = cfg
        actor.summaries = lambda: {}
        with patch.object(custom_actor_critic, "default_make_actor_critic_func", return_value=actor):
            result = custom_actor_critic.make_hipposlam_actor_critic(cfg, None, None)
        result.train()
        self.assertFalse(result.encoder.bn.training)
        self.assertFalse(result.obs_normalizer.training)
        before = result.encoder.bn.running_mean.clone()
        result.encoder.bn(torch.randn(8, 3))
        self.assertTrue(torch.equal(before, result.encoder.bn.running_mean))
        self.assertFalse(result.action_parameterization.weight.requires_grad)
        self.assertTrue(result.critic_linear.weight.requires_grad)
        self.assertTrue(result.core.hl_learner.weight.requires_grad)

    def test_target_checkpoint_resume_skips_oracle_initializer(self):
        learner = OracleTransferLearner.__new__(OracleTransferLearner)
        learner.cfg = AttrConfig(load_checkpoint_kind="latest", oracle_init_checkpoint="missing.pth")
        with patch.object(OracleTransferLearner, "checkpoint_dir", return_value="target"), \
             patch.object(OracleTransferLearner, "get_checkpoints", return_value=["checkpoint_1.pth"]), \
             patch.object(DefaultLearner, "load_from_checkpoint", return_value=None) as resume, \
             patch("sf_working_directories.zeynep.dmlab.oracle_transfer_learner.load_oracle_controller") as transfer:
            learner.load_from_checkpoint(0)
        resume.assert_called_once_with(0, load_progress=True)
        transfer.assert_not_called()

    def test_fresh_run_loads_weights_without_source_progress(self):
        learner = OracleTransferLearner.__new__(OracleTransferLearner)
        learner.cfg = AttrConfig(load_checkpoint_kind="latest", oracle_init_checkpoint="source.pth")
        learner.actor_critic = ToyActor()
        with patch.object(OracleTransferLearner, "checkpoint_dir", return_value="target"), \
             patch.object(OracleTransferLearner, "get_checkpoints", return_value=[]), \
             patch.object(DefaultLearner, "load_from_checkpoint", return_value=None) as resume, \
             patch("sf_working_directories.zeynep.dmlab.oracle_transfer_learner.load_oracle_controller") as transfer:
            learner.load_from_checkpoint(0)
        transfer.assert_called_once_with(learner.actor_critic, "source.pth")
        resume.assert_not_called()


class AttrConfig(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


if __name__ == "__main__":
    unittest.main()
