#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.da2 import checkpoint_path_for_encoder
from models.eval_adapters import build_eval_adapter
from training.checkpoints import checkpoint_has_config
from training.config import DEFAULT_DA2_CONFIG, expand_env_vars, load_yaml_config, set_nested
from training.depth_evaluation import (
    evaluate_depth_adapter,
    maybe_log_wandb_eval,
    now_stamp,
    select_eval_filenames,
    write_evaluation_outputs,
)

LOGGER = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CIL depth evaluation with canonical siRMSE")
    parser.add_argument("--config", type=str, default=DEFAULT_DA2_CONFIG, help="Experiment YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Optional for official DA2 configs, required for U-Net.")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment.name for output naming")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit evaluation output directory")
    parser.add_argument("--img-size", type=int, default=None, help="Override data.image_size")
    parser.add_argument("--protocol", type=str, default=None, help="Override data.eval_protocol, e.g. native_resolution/raw_infer_native")
    parser.add_argument("--split-file", type=str, default=None, help="Override data.split_file")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap selected validation samples for smoke/debug")
    parser.add_argument("--fraction", type=float, default=None, help="Subsample the validation split for smoke/debug")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/adapter/output and exit before model/data loading")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def apply_eval_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            set_nested(updated, key, value)
    return expand_env_vars(updated)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "data.image_size": args.img_size,
        "data.eval_protocol": args.protocol,
        "data.split_file": args.split_file,
    }


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = apply_eval_overrides(config, cli_overrides(args))
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.checkpoint is not None:
        config.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(args.checkpoint))
    return config


def resolve_checkpoint(config: dict[str, Any]) -> Path | None:
    paths = config.get("paths", {})
    if paths.get("checkpoint"):
        return Path(os.path.expanduser(os.path.expandvars(paths["checkpoint"])))
    family = config.get("model", {}).get("family")
    if family == "da2_relative":
        return checkpoint_path_for_encoder(paths["da2_checkpoint_dir"], config["model"]["encoder"])
    return None


def maybe_restore_checkpoint_config(config: dict[str, Any], checkpoint_path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return config
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return config
    if checkpoint_has_config(payload):
        restored = apply_cli_overrides(payload["config"], args)
        LOGGER.info("Loaded effective config from checkpoint metadata: %s", checkpoint_path)
        return restored
    return config


def resolve_output_dir(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(os.path.expanduser(os.path.expandvars(explicit)))
    paths = config.get("paths", {})
    output_root = Path(os.path.expanduser(os.path.expandvars(paths.get("output_root", "/work/scratch/$USER/cil-visionavengers-depth/checkpoints"))))
    run_name = config.get("experiment", {}).get("name", "depth_eval")
    return output_root.parent / "evaluations" / run_name / now_stamp()


def main() -> None:
    args = parse_args()
    setup_logging()

    config = apply_cli_overrides(load_yaml_config(args.config), args)
    checkpoint_path = resolve_checkpoint(config)
    config = maybe_restore_checkpoint_config(config, checkpoint_path, args)
    checkpoint_path = resolve_checkpoint(config)
    output_dir = resolve_output_dir(config, args.output_dir)

    LOGGER.info("Config      : %s", args.config)
    LOGGER.info("Experiment  : %s", config.get("experiment", {}).get("name"))
    LOGGER.info("Model family: %s", config.get("model", {}).get("family"))
    LOGGER.info("Checkpoint  : %s", checkpoint_path)
    LOGGER.info("Output dir  : %s", output_dir)
    LOGGER.info("Eval policy : canonical_sirmse_v1")

    if args.dry_run:
        LOGGER.info("Dry run complete before data/model loading.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_eval_adapter(config=config, checkpoint_path=checkpoint_path, device=device)
    data_cfg = config.get("data", {})
    filenames = select_eval_filenames(
        data_dir=data_cfg["root"],
        val_fraction=float(data_cfg.get("val_fraction", 0.05)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        split_file=data_cfg.get("split_file"),
        max_samples=data_cfg.get("max_samples"),
        fraction=args.fraction,
    )
    LOGGER.info("Adapter     : %s", adapter.metadata())
    LOGGER.info("Samples     : selected=%d", len(filenames))

    result = evaluate_depth_adapter(
        adapter=adapter,
        data_dir=data_cfg["root"],
        filenames=filenames,
        device=device,
    )
    payload = write_evaluation_outputs(
        output_dir=output_dir,
        result=result,
        config=config,
        checkpoint=checkpoint_path,
        selected_filenames=filenames,
    )
    with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    summary = payload["summary"]
    LOGGER.info(
        "siRMSE mean=%.4f median=%.4f std=%.4f samples=%d",
        summary["sirmse_mean"],
        summary["sirmse_median"],
        summary["sirmse_std"],
        summary["samples_evaluated"],
    )
    maybe_log_wandb_eval(config, output_dir, summary, disabled=args.no_wandb)


if __name__ == "__main__":
    main()
