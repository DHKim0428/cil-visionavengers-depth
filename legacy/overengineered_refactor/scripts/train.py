#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import DEFAULT_DA2_CONFIG, load_augmentation_config, load_yaml_config, with_overrides
from training.runner import run_training

LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical config-driven CIL depth training")
    parser.add_argument("--config", type=str, default=DEFAULT_DA2_CONFIG, help="Experiment YAML config")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment.name")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root")
    parser.add_argument("--output-root", type=str, default=None, help="Override paths.output_root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override an optional input/base checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a canonical latest.pth checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--img-size", type=int, default=None, help="Override data.image_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug cap for discovered dataset samples")
    parser.add_argument("--save-policy", type=str, default=None, choices=["trainable_only", "full_model"], help="Override checkpoint.save_policy")
    parser.add_argument("--dry-run", action="store_true", help="Validate config resolution and exit before loading data/model")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_config(args.config)
    overrides = {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "paths.output_root": args.output_root,
        "train.epochs": args.epochs,
        "train.batch_size": args.batch_size,
        "train.num_workers": args.num_workers,
        "data.image_size": args.img_size,
        "checkpoint.save_policy": args.save_policy,
    }
    config = with_overrides(config, overrides)
    config["augmentation"] = load_augmentation_config(config.get("augmentation"))
    if args.checkpoint is not None:
        config.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(args.checkpoint))
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.resume is not None:
        config.setdefault("train", {})["resume"] = os.path.expanduser(os.path.expandvars(args.resume))
    return config


def main() -> None:
    args = parse_args()
    setup_logging()
    config = resolve_config(args)
    LOGGER.info("Config      : %s", args.config)
    LOGGER.info("Experiment  : %s", config["experiment"]["name"])
    LOGGER.info("Model family: %s", config["model"]["family"])
    LOGGER.info("Prediction  : %s", config["model"].get("prediction_kind", "disparity"))
    LOGGER.info("Train view  : %s", config["data"]["views"]["train"])
    LOGGER.info("Eval view   : %s", config["data"]["views"]["eval"])
    LOGGER.info("Augmentation: %s", config.get("augmentation", {}).get("name", "none"))
    if config["model"]["family"] == "da2_relative":
        LOGGER.info("Base scope  : %s", config["base"]["trainable_scope"])
    if args.dry_run:
        LOGGER.info("Dry run complete before data/model loading.")
        return
    run_training(config)


if __name__ == "__main__":
    main()
