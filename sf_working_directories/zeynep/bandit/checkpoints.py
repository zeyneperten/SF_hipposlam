"""Weights-only manager checkpoints shared by the bandit and full environment.

The mode-to-arm map describes the ideal bandit controller only. A full learned
controller's mapping must be measured independently before transferring weights.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from sf_working_directories.zeynep.dmlab.custom_highlevelRNN import HighLevelContextRNN_Learner

FORMAT = "simgay_high_level_weights"
VERSION = 1
MODEL_FIELDS = ("K", "d_H", "is_policy", "tau", "representations", "d_rep")


def _manager(model):
    if isinstance(model, HighLevelContextRNN_Learner):
        return model, None
    if isinstance(getattr(model, "hl_learner", None), HighLevelContextRNN_Learner):
        return model.hl_learner, model
    for name in ("core", "actor_core"):
        core = getattr(model, name, None)
        if isinstance(getattr(core, "hl_learner", None), HighLevelContextRNN_Learner):
            return core.hl_learner, core
    raise ValueError("Expected HighLevelContextRNN_Learner or a full model with core/actor_core.hl_learner")


def _validate_metadata(metadata):
    metadata = dict(metadata)
    required = (*MODEL_FIELDS, "history_len", "mode_to_arm")
    missing = sorted(set(required) - metadata.keys())
    if missing:
        raise ValueError(f"High-level checkpoint metadata is missing: {missing}")
    for key in ("K", "d_H", "d_rep", "history_len"):
        if type(metadata[key]) is not int or metadata[key] < 1:
            raise ValueError(f"Checkpoint {key} must be a positive integer")
    for key in ("is_policy", "representations"):
        if type(metadata[key]) is not bool:
            raise ValueError(f"Checkpoint {key} must be a boolean")
    tau = metadata["tau"]
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Checkpoint tau must be a finite positive number")
    mapping = metadata["mode_to_arm"]
    if mapping is not None:
        if not isinstance(mapping, (list, tuple)) or len(mapping) != metadata["K"]:
            raise ValueError("Checkpoint mode_to_arm must contain one arm per mode")
        if any(type(arm) is not int or arm not in (0, 1) for arm in mapping):
            raise ValueError("Checkpoint mode_to_arm must use zero-based arms 0 and 1")
        metadata["mode_to_arm"] = list(mapping)
    return metadata


def save_checkpoint(path, model, history_len, mode_to_arm, extra=None):
    """Save manager weights and compatibility metadata, without optimizer state."""
    manager, _ = _manager(model)
    metadata = _validate_metadata(
        {
            **{key: getattr(manager, key) for key in MODEL_FIELDS},
            "history_len": history_len,
            "mode_to_arm": list(mode_to_arm) if mode_to_arm is not None else None,
        }
    )
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "weights_only": True,
        "metadata": metadata,
        "model": {key: value.detach().cpu().clone() for key, value in manager.state_dict().items()},
    }
    if extra is not None:
        payload["extra"] = extra
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _full_config(path, payload):
    for key in ("config", "cfg"):
        if isinstance(payload.get(key), Mapping):
            return payload[key]
    for directory in (path.parent.parent, path.parent):
        for filename in ("config.json", "cfg.json"):
            candidate = directory / filename
            if candidate.is_file():
                with candidate.open(encoding="utf-8") as handle:
                    return json.load(handle)
    raise ValueError(
        "Full Sample Factory checkpoint requires its config.json/cfg.json to validate "
        "high-level policy, temperature, and representation semantics"
    )


def read_checkpoint(path):
    """Return (manager state, metadata) from a standalone or full SF checkpoint.

    Full SF checkpoints retain their usual trusted-local pickle loading policy.
    Missing historical configuration flags use the full model's own defaults.
    """
    path = Path(path).expanduser()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("High-level checkpoint must contain a model state dictionary")
    state = payload["model"]
    if payload.get("format") == FORMAT:
        if payload.get("version") != VERSION or payload.get("weights_only") is not True:
            raise ValueError("Unsupported standalone high-level checkpoint version or kind")
        metadata = _validate_metadata(payload.get("metadata", {}))
    else:
        prefixes = [
            prefix
            for prefix in ("core.hl_learner.", "actor_core.hl_learner.")
            if any(key.startswith(prefix) for key in state)
        ]
        if len(prefixes) != 1:
            raise ValueError("Full checkpoint must contain exactly one core.hl_learner or actor_core.hl_learner")
        prefix = prefixes[0]
        state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        cfg = _full_config(path, payload)
        metadata = _validate_metadata(
            {
                "K": cfg.get("hl_K", 4),
                "d_H": cfg.get("hl_d_H", 16),
                "is_policy": cfg.get("hl_is_policy", False),
                "tau": cfg.get("hl_tau", 1.0),
                "representations": cfg.get("hl_representations", False),
                "d_rep": cfg.get("hl_d_rep", 64),
                "history_len": cfg.get("hl_history_len", 8),
                "mode_to_arm": cfg.get("hl_mode_to_arm"),
            }
        )
    return state, metadata


def _load_validated(model, state, metadata, expected_mode_to_arm=None):
    manager, core = _manager(model)
    mismatches = [
        f"{key}: checkpoint={metadata[key]!r}, target={getattr(manager, key)!r}"
        for key in MODEL_FIELDS
        if metadata[key] != getattr(manager, key)
    ]
    if core is not None and hasattr(core, "history_len") and metadata["history_len"] != core.history_len:
        mismatches.append(f"history_len: checkpoint={metadata['history_len']}, target={core.history_len}")
    if expected_mode_to_arm is not None and metadata["mode_to_arm"] is not None:
        if list(expected_mode_to_arm) != metadata["mode_to_arm"]:
            mismatches.append(f"mode_to_arm: checkpoint={metadata['mode_to_arm']}, target={list(expected_mode_to_arm)}")
    if mismatches:
        raise ValueError("Incompatible high-level checkpoint: " + "; ".join(mismatches))
    target = manager.state_dict()
    missing = sorted(target.keys() - state.keys())
    unexpected = sorted(state.keys() - target.keys())
    if missing or unexpected:
        raise ValueError(f"Incompatible high-level tensors: missing={missing}, unexpected={unexpected}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(
                f"Incompatible high-level tensor {key}: checkpoint={shape}, target={tuple(target[key].shape)}"
            )
    # Validate the entire state before copying, so a failed import changes nothing.
    manager.load_state_dict(state, strict=True)


def load_checkpoint_into(model, path, expected_mode_to_arm=None):
    """Initialize only manager weights; all other model parameters stay intact.

    A missing source mode-to-arm map remains unknown. Loading into the full model
    cannot establish that its navigation controller realizes the bandit mapping.
    """
    state, metadata = read_checkpoint(path)
    _load_validated(model, state, metadata, expected_mode_to_arm)
    return metadata


def build_model_from_checkpoint(path, device="cpu", expected_mode_to_arm=None):
    """Build the actual shared manager architecture from saved metadata."""
    state, metadata = read_checkpoint(path)
    model = HighLevelContextRNN_Learner(**{key: metadata[key] for key in MODEL_FIELDS}).to(device)
    _load_validated(model, state, metadata, expected_mode_to_arm)
    return model, metadata
