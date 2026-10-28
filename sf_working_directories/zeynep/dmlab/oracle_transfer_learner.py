"""Fresh-run controller/manager initialization without importing source progress."""

from sample_factory.algo.learning.learner import DefaultLearner
from sample_factory.utils.utils import log
from sf_working_directories.zeynep.bandit.checkpoints import load_checkpoint_into
from sf_working_directories.zeynep.dmlab.oracle_transfer import load_oracle_controller


class OracleTransferLearner(DefaultLearner):
    def load_from_checkpoint(self, policy_id, load_progress=True):
        existing = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern="*.pth")
        # Dictionary-backed configurations used by callers may raise KeyError
        # for optional new fields rather than the usual AttributeError.
        get = self.cfg.get if isinstance(self.cfg, dict) else lambda key, default=None: getattr(self.cfg, key, default)
        oracle_checkpoint = get("oracle_init_checkpoint")
        hl_checkpoint = get("hl_init_checkpoint")
        if existing or not load_progress or not (oracle_checkpoint or hl_checkpoint):
            return super().load_from_checkpoint(policy_id, load_progress=load_progress)
        if oracle_checkpoint:
            load_oracle_controller(self.actor_critic, oracle_checkpoint)
        if hl_checkpoint:
            metadata = load_checkpoint_into(self.actor_critic, hl_checkpoint)
            log.warning(
                "Initialized high-level weights from %s; optimizer and progress start fresh. "
                "Source bandit mode_to_arm=%s; verify the full controller's mode-to-arm mapping empirically.",
                hl_checkpoint,
                metadata["mode_to_arm"],
            )


def make_oracle_transfer_learner(cfg, env_info, policy_versions_tensor, policy_id, param_server):
    return OracleTransferLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)
