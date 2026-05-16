from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DA2_CONFIG = "configs/experiments/da2_vits_refinenets_output.yaml"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand environment variables in string values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return normalize_config(data)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand $VARS and ~ in strings loaded from config."""
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a dotted config key, creating intermediate dictionaries as needed."""
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and expand the final canonical training config schema."""
    updated = copy.deepcopy(config)

    retired = [key for key in ("finetune", "checkpointing") if key in updated]
    if retired:
        joined = ", ".join(retired)
        raise ValueError(
            f"Unsupported legacy config key(s): {joined}. "
            "Use final schema keys `base`, `adapter`, and `checkpoint`."
        )

    model_cfg = updated.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Config key 'model' must be a mapping with `family`.")
    family = model_cfg.get("family")
    if family not in {"da2_relative", "unet"}:
        raise ValueError("Config key 'model.family' must be one of: da2_relative, unet")

    if family == "da2_relative":
        base_cfg = updated.get("base")
        if not isinstance(base_cfg, dict):
            raise ValueError("DA2 config key 'base' must be a mapping with `trainable_scope`.")
        scope = base_cfg.get("trainable_scope")
        if scope not in {"frozen", "full", "decoder", "refinenets_output"}:
            raise ValueError(
                "DA2 config key 'base.trainable_scope' must be one of: "
                "frozen, full, decoder, refinenets_output"
            )
    elif family == "unet":
        prediction_kind = model_cfg.get("prediction_kind", "disparity")
        if prediction_kind not in {"disparity", "depth"}:
            raise ValueError("U-Net config key 'model.prediction_kind' must be one of: disparity, depth")

    checkpoint_cfg = updated.get("checkpoint", {})
    if checkpoint_cfg is not None and not isinstance(checkpoint_cfg, dict):
        raise ValueError("Config key 'checkpoint' must be a mapping when provided.")

    return expand_env_vars(updated)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping overrides into a deep-copied base mapping."""
    updated = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = copy.deepcopy(value)
    return updated


def load_augmentation_config(section: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve an experiment augmentation section to a concrete preset config.

    Experiment configs normally contain ``augmentation.preset``.  The preset is
    loaded from ``configs/augmentations/<preset>.yaml`` and any additional
    inline fields under ``augmentation`` override the preset.
    """
    section = copy.deepcopy(section or {"preset": "none"})
    if not isinstance(section, dict):
        raise ValueError("Config key 'augmentation' must be a mapping")

    preset = section.pop("preset", None)
    if preset is None:
        resolved = section
    else:
        preset_path = Path(__file__).resolve().parents[1] / "configs" / "augmentations" / f"{preset}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Unknown augmentation preset '{preset}': {preset_path}")
        with preset_path.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle) or {}
        if not isinstance(resolved, dict):
            raise ValueError(f"Augmentation preset must be a YAML mapping: {preset_path}")
        resolved = deep_update(resolved, section)

    return expand_env_vars(resolved)


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with non-None dotted-key overrides applied."""
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            set_nested(updated, key, value)
    return normalize_config(updated)
