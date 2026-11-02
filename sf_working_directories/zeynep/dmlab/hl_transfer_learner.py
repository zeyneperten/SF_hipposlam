"""Learner for High-Level RNN curriculum transfer without treating source as resume."""

from sample_factory.algo.learning.learner import DefaultLearner
from sf_working_directories.zeynep.dmlab.hl_transfer import load_hl_checkpoint


class HLTransferLearner(DefaultLearner):
    def load_from_checkpoint(self, policy_id, load_progress=True):
        # If the target experiment already has its own checkpoint, resume normally
        existing = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern="*.pth")
        if existing or not load_progress or not getattr(self.cfg, "hl_init_checkpoint", None):
            return super().load_from_checkpoint(policy_id, load_progress=load_progress)

        # Fresh run: initialize from the transferred HLRNN checkpoint
        load_hl_checkpoint(self.actor_critic, self.cfg.hl_init_checkpoint)


def make_hl_transfer_learner(cfg, env_info, policy_versions_tensor, policy_id, param_server):
    return HLTransferLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)