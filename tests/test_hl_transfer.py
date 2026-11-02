import json
import tempfile
import unittest
from pathlib import Path
import torch
from types import SimpleNamespace

# Import your new transfer functions and high-level core
from sf_working_directories.zeynep.dmlab.hl_transfer import load_hl_checkpoint, prepare_hl_transfer
from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import HighLevelContextRNN_Learner


class ToyCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_core = torch.nn.Linear(10, 10)
        self.hl_learner = HighLevelContextRNN_Learner(K=2, d_H=16, tau=0.2)


class ToyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = torch.nn.Identity()
        self.encoder = torch.nn.Linear(10, 10)
        self.core = ToyCore()
        self.decoder = torch.nn.Linear(10, 10)
        self.action_parameterization = torch.nn.Linear(10, 2)
        self.critic_linear = torch.nn.Linear(10, 1)
        self.returns_normalizer = torch.nn.Identity()


class TestHLTransfer(unittest.TestCase):
    def test_hl_and_controller_weights_transferred(self):
        # 1. Create a source model with unique weights
        source_model = ToyActorCritic()
        with torch.no_grad():
            source_model.core.hl_learner.head.weight.fill_(3.14)
            source_model.core.hl_learner.rnn_cell.weight_ih.fill_(1.23)

        # 2. Save mock checkpoint and companion config.json
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "test_run"
            ckpt_dir = run_dir / "checkpoint_p0"
            ckpt_dir.mkdir(parents=True)

            ckpt_path = ckpt_dir / "checkpoint_000000001.pth"
            torch.save({"model": source_model.state_dict()}, ckpt_path)

            cfg_path = run_dir / "config.json"
            cfg_path.write_text(json.dumps({
                "core_name": "BypassSS_HighLevelRNN",
                "hl_K": 2,
                "hl_d_H": 16,
                "Hippo_n_feature": 16,
                "Hippo_L": 64,
                "Hippo_R": 8,
                "depth_sensor": True,
            }))

            # 3. Create fresh target model
            target_model = ToyActorCritic()
            self.assertFalse(torch.allclose(target_model.core.hl_learner.head.weight, torch.tensor(3.14)))

            # 4. Run your loader
            copied = load_hl_checkpoint(target_model, ckpt_path)

            # 5. Assert HLRNN weights were transferred exactly
            self.assertTrue(torch.allclose(target_model.core.hl_learner.head.weight, torch.tensor(3.14)))
            self.assertTrue(torch.allclose(target_model.core.hl_learner.rnn_cell.weight_ih, torch.tensor(1.23)))
            self.assertIn("core.hl_learner.head.weight", copied)
            self.assertIn("core.hl_learner.rnn_cell.weight_ih", copied)
            print(f"\n[PASS] Verified {len(copied)} tensors transferred successfully!")


if __name__ == "__main__":
    unittest.main()