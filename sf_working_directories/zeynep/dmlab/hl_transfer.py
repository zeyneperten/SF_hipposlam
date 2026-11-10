"""Initialize a new High-Level RNN run from a prior HLRNN checkpoint."""

import json
from pathlib import Path
import torch

from sample_factory.utils.utils import log


def source_config_path(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".pth":
        raise ValueError(f"HLRNN checkpoint must be an existing .pth file: {checkpoint}")
    if not checkpoint.parent.name.startswith("checkpoint_p"):
        raise ValueError("Checkpoint must be inside a policy-specific checkpoint_pN directory")
    for name in ("config.json", "cfg.json"):
        candidate = checkpoint.parent.parent / name
        if candidate.is_file():
            return checkpoint, candidate
    raise ValueError(f"No config.json or cfg.json found beside {checkpoint.parent}")


def prepare_hl_transfer(cfg):
    """Align architecture with the source HLRNN run before Sample Factory initializes."""
    checkpoint, config_path = source_config_path(cfg.hl_transfer_checkpoint)
    with config_path.open(encoding="utf-8") as handle:
        source = json.load(handle)

    # Validate that source was indeed an HLRNN run
    if source.get("core_name") != "BypassSS_HighLevelRNN":
        raise ValueError(
            f"Expected source core_name='BypassSS_HighLevelRNN', got '{source.get('core_name')}'"
        )

    # Inherit core structural parameters to ensure identical tensor dimensions
    structural = (
        "encoder_conv_architecture",
        "encoder_conv_mlp_layers",
        "encoder_name",
        "encoder_type",
        "decoder_mlp_layers",
        "DG_name",
        "DG_BN_intercept",
        "Hippo_n_feature",
        "Hippo_L",
        "Hippo_R",
        "depth_sensor",
        "hl_K",
        "hl_d_H",
        "hl_history_len",
        "hl_representations",
        "hl_d_rep",
        "hl_DG_mod",
        "Decoder_context_mod",
        "number_instruction_coef",
        "nonlinearity",
    )
    for key in structural:
        if key in source:
            value = int(source[key]) if key in ("Hippo_n_feature", "Hippo_L", "Hippo_R", "hl_K", "hl_d_H") else source[key]
            setattr(cfg, key, value)
            cfg.cli_args[key] = value

    cfg.hl_transfer_checkpoint = str(checkpoint)
    cfg.core_name = "BypassSS_HighLevelRNN"
    cfg.oracle_context = False
    cfg.hl_dg_from_prev_z = source.get("hl_dg_from_prev_z", True)

    # Calculate exact rnn_size (CA3 + Depth + High-Level state + History + Mode)
    base_size = cfg.Hippo_n_feature * (cfg.Hippo_L + cfg.Hippo_R - 1)
    base_size += 10 if cfg.depth_sensor else 0
    rep_size = (2 * cfg.hl_d_rep) if getattr(cfg, "hl_representations", False) else 0
    event_width = cfg.hl_d_H + cfg.hl_K + 2 + rep_size
    history_size = cfg.hl_history_len * event_width
    cfg.rnn_size = base_size + cfg.hl_d_H + rep_size + history_size + cfg.hl_K
    cfg.cli_args["rnn_size"] = cfg.rnn_size

    log.info(
        "HLRNN Transfer Prepared: %s (K=%d, d_H=%d, DG_mod=%s, rnn_size=%d)",
        checkpoint, cfg.hl_K, cfg.hl_d_H, cfg.hl_DG_mod, cfg.rnn_size
    )
    return source


def load_hl_checkpoint(actor_critic, checkpoint):
    """Copy controller and high-level RNN tensors into the new ActorCritic model."""
    checkpoint, _ = source_config_path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model", payload)
    target = actor_critic.state_dict()

    prefixes = (
        "obs_normalizer.",
        "encoder.",
        "core.base_core.",
        "core.hl_learner.",   # <--- Transfers HLRNN (RNNCell + Q/Policy Head)
        "decoder.",
        "action_parameterization.",
        "critic_linear.",
        "returns_normalizer.",
    )

    copied = {}
    for key, value in target.items():
        if not any(key.startswith(p) for p in prefixes):
            continue
        if key in source:
            if source[key].shape == value.shape:
                copied[key] = source[key]
            else:
                raise ValueError(
                    f"Shape mismatch for {key}: source {source[key].shape} vs target {value.shape}"
                )
        else:
            log.warning("Tensor %s not in source checkpoint; leaving initialized", key)

    missing, unexpected = actor_critic.load_state_dict(copied, strict=False)
    
    # hl_diversity_classifier is optional, so it is the only expected missing prefix
    allowed_missing = ("hl_diversity_classifier.",)
    real_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing)]
    if real_missing:
        log.warning("Some tensors were not found in source: %s", real_missing)

    log.info(
        "Successfully initialized %d tensors from %s (including High-Level RNN). "
        "Optimizer and step counters start fresh at 0.",
        len(copied), checkpoint
    )
    return copied