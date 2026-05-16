from __future__ import annotations

import copy
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger("cil_depth")
DEFAULT_CONFIG = "configs/experiments/da2_vits_refinenets_output.yaml"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return expand(cfg)


def load_augmentation_config(section: dict[str, Any] | None) -> dict[str, Any]:
    section = copy.deepcopy(section or {"preset": "none"})
    preset = section.pop("preset", None)
    if preset is None:
        return expand(section)
    path = Path("configs") / "augmentations" / f"{preset}.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(section)
    return expand(cfg)


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if value is not None:
            set_nested(cfg, key, value)
    cfg["augmentation"] = load_augmentation_config(cfg.get("augmentation"))
    return expand(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("paths", {}).get("output_root", "runs"))
    name = cfg.get("experiment", {}).get("name", "depth_run")
    out = root / name / timestamp()
    out.mkdir(parents=True, exist_ok=False)
    return out


def maybe_wandb(cfg: dict[str, Any], run_dir: Path, job_type: str):
    log_cfg = cfg.get("logging", {})
    if log_cfg.get("backend", "wandb") != "wandb":
        return None
    try:
        import wandb
    except ImportError:
        LOGGER.warning("W&B requested but not installed. Run: python -m pip install -r requirements.txt")
        return None
    if not getattr(wandb.api, "api_key", None):
        LOGGER.warning("W&B requested but not logged in. Run: wandb login")
        return None
    return wandb.init(
        entity=log_cfg.get("entity"),
        project=log_cfg.get("project"),
        name=cfg.get("experiment", {}).get("name"),
        tags=cfg.get("experiment", {}).get("tags", []),
        config=cfg,
        dir=str(run_dir),
        job_type=job_type,
    )


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
