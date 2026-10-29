"""Initialize the two-mode controller from a selected oracle checkpoint."""

import json
from pathlib import Path

import torch

from sample_factory.utils.utils import log

SUPPORTED_DG = {"concat", "multiply", "sigmoid"}
SUPPORTED_DECODER = {"enter", "FiLM", "additive"}


def source_config_path(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".pth":
        raise ValueError(f"Oracle checkpoint must be an existing .pth file: {checkpoint}")
    if not checkpoint.parent.name.startswith("checkpoint_p"):
        raise ValueError("Oracle checkpoint must be inside a policy-specific checkpoint_pN directory")
    for name in ("config.json", "cfg.json"):
        candidate = checkpoint.parent.parent / name
        if candidate.is_file():
            return checkpoint, candidate
    raise ValueError(f"No config.json or cfg.json beside {checkpoint.parent}")


def prepare_oracle_transfer(cfg):
    """Apply source architecture before Sample Factory builds buffers and modules."""
    checkpoint, config_path = source_config_path(cfg.oracle_init_checkpoint)
    with config_path.open(encoding="utf-8") as handle:
        source = json.load(handle)

        # If DG_context_mod is not specified or None in source, assume sigmoid + enter + oracle_context FOR ONLY THIS CHECKPOINT
    if source.get("DG_context_mod") in (None, "None"):
        source["DG_context_mod"] = source.get("INSTR_modulation", "sigmoid") or "sigmoid"

    if source.get("Decoder_context_mod") in (None, "None"):
        source["Decoder_context_mod"] = "enter"

    if source.get("oracle_context") in (None, "None"):
        source["oracle_context"] = True

    # Legacy checkpoint normalization:
    # Older runs used `instr_modulation` instead of separate DG/Decoder flags.
    # An older run with instr_modulation="sigmoid" corresponds to:
    # DG_context_mod="sigmoid", Decoder_context_mod="enter", and oracle_context=True.
    if "instr_modulation" in source:
        mod = source["INSTR_modulation"]
        source.setdefault("DG_context_mod", mod)
        source.setdefault("Decoder_context_mod", "enter")
        source.setdefault("oracle_context", True)

        # Allow transferring from any ymaze environment to ymaze_instr_hl
    if source.get("env") is not None and source["env"] != cfg.env:
        if not (source["env"].startswith("ymaze") and cfg.env.startswith("ymaze")):
            raise ValueError(f"Incompatible oracle environment: source {source['env']!r}, target {cfg.env!r}")
        log.warning(
            "Transferring controller across ymaze environments: source '%s' -> target '%s'", source["env"], cfg.env
        )

    expected = {
        "core_name": "BypassSS",
        "with_number_instruction": True,
        "decoder_type": "mlp",
        "actor_critic_share_weights": True,
        "reward_input": False,
    }
    for key, value in expected.items():
        actual = source.get(key, False if key == "reward_input" else None)
        if actual != value:
            raise ValueError(f"Incompatible oracle checkpoint: {key} must be {value!r}, got {source.get(key)!r}")
    if source.get("DG_context_mod") not in SUPPORTED_DG:
        raise ValueError(f"Unsupported oracle DG_context_mod={source.get('DG_context_mod')!r}")
    if source.get("Decoder_context_mod") not in SUPPORTED_DECODER:
        raise ValueError(f"Unsupported oracle Decoder_context_mod={source.get('Decoder_context_mod')!r}")

    structural = (
        "encoder_conv_architecture",
        "encoder_conv_mlp_layers",
        "encoder_name",
        "encoder_type",
        "encoder_subtype",
        "decoder_mlp_layers",
        "DG_name",
        "DG_BN_intercept",
        "DG_lr",
        "DG_temperature",
        "DG_batch_q",
        "DG_softmax",
        "DG_detect",
        "DG_novelty",
        "Hippo_n_feature",
        "Hippo_L",
        "Hippo_R",
        "res_h",
        "res_w",
        "depth_sensor",
        "with_number_instruction",
        "number_instruction_coef",
        "context_injection_coef",
        "DG_context_mod",
        "Decoder_context_mod",
        "dmlab_reduced_action_set",
        "dmlab_navigation_action_set",
        "dmlab_extended_action_set",
        "env_frameskip",
        "nonlinearity",
        "normalize_input",
        "normalize_input_keys",
        "obs_scale",
        "obs_subtract_mean",
        "use_jit",
    )
    for key in structural:
        if key in source:
            value = int(source[key]) if key in ("Hippo_n_feature", "Hippo_L", "Hippo_R") else source[key]
            setattr(cfg, key, value)
            cfg.cli_args[key] = value

    cfg.oracle_init_checkpoint = str(checkpoint)
    cfg.core_name = "BypassSS_HighLevelRNN"
    cfg.oracle_context = False
    cfg.hl_dg_from_prev_z = True
    cfg.hl_K = 2
    cfg.actor_critic_share_weights = True
    cfg.DG_context_mod = "sigmoid"
    cfg.Decoder_context_mod = "enter"
    cfg.decoder_type = "mlp"
    for key in (
        "oracle_init_checkpoint",
        "core_name",
        "oracle_context",
        "hl_dg_from_prev_z",
        "hl_K",
        "actor_critic_share_weights",
        "decoder_type",
    ):
        cfg.cli_args[key] = getattr(cfg, key)

    if cfg.oracle_freeze_controller and getattr(cfg, "hl_diversity_reward_coef", 0.0) > 0:
        raise ValueError("Disable hl_diversity_reward_coef while the inherited action controller is frozen")

    # The high-level wrapper stores the fixed CA3 register, depth bypass, the
    # two-dimensional oracle-like bypass, high state, trial history, and z.
    base_size = cfg.Hippo_n_feature * (cfg.Hippo_L + cfg.Hippo_R - 1)
    base_size += 10 if cfg.depth_sensor else 0  # do not add +2
    event_width = cfg.hl_d_H + cfg.hl_K + 2
    cfg.rnn_size = base_size + cfg.hl_d_H + cfg.hl_history_len * event_width + cfg.hl_K
    cfg.cli_args["rnn_size"] = cfg.rnn_size
    log.warning(
        "Oracle transfer: %s (DG=%s, decoder=%s, scale=%s, rnn_size=%s)",
        checkpoint,
        cfg.DG_context_mod,
        cfg.Decoder_context_mod,
        cfg.number_instruction_coef,
        cfg.rnn_size,
    )
    return source


def load_oracle_controller(actor_critic, checkpoint):
    """Copy controller tensors; deliberately leave the learned manager and optimizer new."""
    checkpoint, _ = source_config_path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model")
    if not isinstance(source, dict):
        raise ValueError(f"Oracle checkpoint has no model state dictionary: {checkpoint}")
    target = actor_critic.state_dict()
    prefixes = (
        "obs_normalizer.",
        "encoder.",
        "core.base_core.",
        "decoder.",
        "action_parameterization.",
        "critic_linear.",
        "returns_normalizer.",
    )
    source_only = sorted(
        key
        for key in source
        if key.startswith(prefixes)
        and key not in target
        and not (key.startswith("encoder.instruction_embed_layer.") and "encoder.hl_mode_embed_layer.weight" in target)
    )
    if source_only:
        raise ValueError(f"Incompatible oracle controller has unexpected tensors: {source_only}")
    copied = {}
    converted = []
    for key, value in target.items():
        if not key.startswith(prefixes):
            continue

        # High-level mode modulation in encoder (hl_mode_embed_layer):
        # If present in source (e.g. from an HLRNN run), it will be loaded directly below.
        # If absent from source, inherit from the oracle's instruction_embed_layer (dropping Context 0).
        if key == "encoder.hl_mode_embed_layer.weight" and key not in source:
            if "encoder.instruction_embed_layer.weight" in source:
                old = source["encoder.instruction_embed_layer.weight"]
                if old.ndim == 2 and old.shape[0] == value.shape[0] and old.shape[1] == value.shape[1] + 1:
                    old = old[:, 1:]
                    converted.append(f"{key} (mapped from instruction_embed_layer.weight[:, 1:] -> {value.shape})")
                elif old.shape == value.shape:
                    converted.append(f"{key} (mapped from instruction_embed_layer.weight -> {value.shape})")
                else:
                    raise ValueError(
                        f"Incompatible shape for {key}: source instruction_embed_layer {tuple(old.shape)}, "
                        f"target {tuple(value.shape)}"
                    )
                copied[key] = old
                continue
            else:
                continue

        if key == "encoder.hl_mode_embed_layer.bias" and key not in source:
            if "encoder.instruction_embed_layer.bias" in source:
                old = source["encoder.instruction_embed_layer.bias"]
                if old.shape == value.shape:
                    converted.append(f"{key} (mapped from instruction_embed_layer.bias)")
                    copied[key] = old
                    continue
                else:
                    raise ValueError(
                        f"Incompatible shape for {key}: source instruction_embed_layer.bias {tuple(old.shape)}, "
                        f"target {tuple(value.shape)}"
                    )
            else:
                continue

        if key not in source:
            raise ValueError(f"Oracle checkpoint is missing controller tensor {key}")
        old = source[key]
        if old.shape != value.shape:
            # Case 1: decoder.mlp.0.weight with 3 instruction slots (1149 -> 1148)
            # Layout: [CA3 (1136) + Depth (10) | Context 0 (1146), Context 1 (1147), Context 2 (1148)]
            # Context 0 has no place fields / inactive; Contexts 1 and 2 are the active navigation behaviors.
            if (
                key == "decoder.mlp.0.weight"
                and old.ndim == 2
                and old.shape[0] == value.shape[0]
                and old.shape[1] == value.shape[1] + 1
            ):
                base_dim = value.shape[1] - 2  # 1146 (CA3 + Depth)
                # Keep base features, drop Context 0 (at index base_dim), keep Context 1 & 2
                old = torch.cat([old[:, :base_dim], old[:, base_dim + 1 :]], dim=1)
                converted.append(f"{key} (dropped slot 0, kept active contexts 1 & 2 -> {value.shape[1]})")

            # Case 2: encoder.instruction_embed_layer.weight (16, 3) -> (16, 2)
            # Drop slot 0, keep active contexts 1 & 2
            elif (
                key == "encoder.instruction_embed_layer.weight"
                and old.ndim == 2
                and old.shape[0] == value.shape[0]
                and old.shape[1] == value.shape[1] + 1
            ):
                old = old[:, 1:]
                converted.append(f"{key} (dropped slot 0, kept active contexts 1 & 2 -> {value.shape[1]})")

            # Case 3: FiLM / additive / context projection weights
            elif (
                key in {"decoder.film_gamma.weight", "decoder.film_beta.weight", "decoder.context_proj.weight"}
                and old.ndim == 2
            ):
                if old.shape[1] == 3 and value.shape[1] == 2:
                    old = old[:, 1:]
                    converted.append(f"{key} (dropped slot 0, kept active contexts 1 & 2)")
                elif old.shape[1] == 4 and value.shape[1] == 2:
                    old = old[:, :2]
                    converted.append(f"{key} (kept active contexts 0 & 1)")
                else:
                    raise ValueError(
                        f"Incompatible oracle tensor {key}: source {tuple(old.shape)}, target {tuple(value.shape)}"
                    )
            else:
                raise ValueError(
                    f"Incompatible oracle tensor {key}: source {tuple(old.shape)}, target {tuple(value.shape)}"
                )
        copied[key] = old

    missing, unexpected = actor_critic.load_state_dict(copied, strict=False)
    allowed_missing_prefixes = (
        "core.hl_learner.",
        "core.high_level_rnn_stage1.",
        "hl_diversity_classifier.",
        "encoder.hl_mode_embed_layer.",
        "core.decoder.",
    )
    if unexpected or any(not key.startswith(allowed_missing_prefixes) for key in missing):
        raise ValueError(f"Unexpected transfer state: missing={missing}, unexpected={unexpected}")
    log.warning(
        "Initialized %d controller tensors from %s; conversions: %s. "
        "High-level learner, optimizer, and progress start fresh.",
        len(copied),
        checkpoint,
        converted if converted else "none",
    )
    return copied, converted
