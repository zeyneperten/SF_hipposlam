"""Fresh-run oracle initialization without treating the source as a resume."""

from sample_factory.algo.learning.learner import DefaultLearner
from sf_working_directories.zeynep.dmlab.oracle_transfer import load_oracle_controller


class OracleTransferLearner(DefaultLearner):
    def load_from_checkpoint(self, policy_id, load_progress=True):
        existing = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern="*.pth")
        if existing or not load_progress or not getattr(self.cfg, "oracle_init_checkpoint", None):
            return super().load_from_checkpoint(policy_id, load_progress=load_progress)
        load_oracle_controller(self.actor_critic, self.cfg.oracle_init_checkpoint)


def make_oracle_transfer_learner(cfg, env_info, policy_versions_tensor, policy_id, param_server):
    return OracleTransferLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)
