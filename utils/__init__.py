from __future__ import annotations

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
    if isinstance(cfg.get("experiment"), str):
        cfg["experiment"] = {"name": cfg["experiment"]}
    return expand(cfg)


def load_augmentation_config(section: Any | None) -> dict[str, Any]:
    if section is None:
        preset, overrides = "none", {}
    elif isinstance(section, str):
        preset, overrides = section, {}
    else:
        overrides = dict(section)
        preset = overrides.pop("preset", overrides.pop("name", "none"))
    path = Path("configs") / "augmentations" / f"{preset}.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(overrides)
    return expand(cfg)

def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if value is None:
            continue
        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
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
