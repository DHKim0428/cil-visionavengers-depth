#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml

from loaders import build_da2_dataloaders
from models.adapters import adapter_is_enabled, apply_adapters_from_config
from models.da2 import apply_base_trainable_scope, checkpoint_path_for_encoder, create_da2_model, extract_state_dict, load_da2_model
from training.checkpoints import (
    checkpoint_has_config,
    is_full_model_checkpoint,
    is_trainable_checkpoint,
    load_checkpoint_payload,
    restore_model_payload,
)
from training.config import DEFAULT_DA2_CONFIG, load_yaml_config, with_overrides
from da2_eval import (
    EvaluationResult,
    evaluate_loader,
    evaluate_raw_infer_native,
    resize_prediction_to_depth,
    save_visualization,
    select_filenames,
)
from da2_losses import MAX_DEPTH, MIN_DEPTH, disparity_to_depth, sirmse_eval_from_disparity

LOGGER = logging.getLogger("eval_da2")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PROTOCOLS = ("native_resolution", "legacy_square", "raw_infer_native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Depth Anything V2 evaluation on CIL depth data")
    parser.add_argument("--config", type=str, default=DEFAULT_DA2_CONFIG, help="Path to DA2 experiment YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate. Supports official DA2, best.pth, or latest.pth")
    parser.add_argument("--protocol", type=str, default=None, choices=PROTOCOLS, help="Evaluation protocol override")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment.name for output naming")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit evaluation output directory")
    parser.add_argument("--img-size", type=int, default=None, help="Override data.image_size")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for loader protocols")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap selected samples for smoke/debug evaluation")
    parser.add_argument("--fraction", type=float, default=None, help="Raw-infer random subset fraction; defaults to config data.val_fraction")
    parser.add_argument("--vis-dir", type=str, default=None, help="Optional directory for per-sample visualizations")
    parser.add_argument("--num-vis", type=int, default=0, help="Maximum visualizations to save")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/protocol/paths and exit before loading data/model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B eval logging even if config requests it")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    return apply_eval_cli_overrides(load_yaml_config(args.config), args)



def eval_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "data.image_size": args.img_size,
        "train.batch_size": args.batch_size,
        "train.num_workers": args.num_workers,
    }


def apply_eval_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = with_overrides(config, eval_cli_overrides(args))
    if args.checkpoint is not None:
        config.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(args.checkpoint))
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.protocol is not None:
        config.setdefault("data", {})["eval_protocol"] = args.protocol
    return config

def resolve_checkpoint(config: dict[str, Any]) -> Path:
    if config.get("paths", {}).get("checkpoint"):
        return Path(config["paths"]["checkpoint"])
    return checkpoint_path_for_encoder(config["paths"]["da2_checkpoint_dir"], config["model"]["encoder"])


def resolve_output_dir(config: dict[str, Any], explicit: str | None, *, create: bool = True) -> Path:
    if explicit:
        out = Path(os.path.expanduser(os.path.expandvars(explicit)))
    else:
        output_root = Path(config["paths"]["output_root"])
        out = output_root.parent / "evaluations" / config["experiment"]["name"] / now_stamp()
    if create:
        out.mkdir(parents=True, exist_ok=False)
    return out


def pipeline_for_protocol(config: dict[str, Any], protocol: str) -> str:
    if protocol == "native_resolution":
        return "dpt_native"
    if protocol == "legacy_square":
        return "legacy_square"
    if protocol == "raw_infer_native":
        return "raw_infer_native"
    raise ValueError(f"Unknown eval protocol: {protocol}")


def maybe_init_wandb(config: dict[str, Any], output_dir: Path, disabled: bool):
    if disabled:
        return None
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("backend", "wandb") != "wandb":
        return None
    try:
        import wandb
    except ImportError:
        LOGGER.warning("W&B requested but wandb is not installed. Run: python -m pip install -r requirements.txt")
        return None
    if not getattr(wandb.api, "api_key", None):
        LOGGER.warning("W&B requested but no login detected. Run `wandb login`; continuing with local eval outputs only.")
        return None
    return wandb.init(
        entity=logging_cfg.get("entity"),
        project=logging_cfg.get("project"),
        name=f"eval_{config['experiment'].get('name')}",
        tags=list(config["experiment"].get("tags", [])) + ["eval", config["data"]["eval_protocol"]],
        config=config,
        dir=str(output_dir),
        job_type="eval",
    )


def denormalize_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = (image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    rgb_uint8 = (rgb.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def save_loader_visualizations(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    vis_dir: Path,
    num_vis: int,
    pred_label: str,
) -> None:
    if num_vis <= 0:
        return
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        pred_disp = resize_prediction_to_depth(model(image), depth)
        pred_depth = disparity_to_depth(pred_disp, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        names = batch.get("name", [f"sample_{saved + i:04d}" for i in range(depth.shape[0])])
        for idx in range(depth.shape[0]):
            if saved >= num_vis:
                return
            name = str(names[idx]).replace("_rgb.png", "")
            save_visualization(
                path=vis_dir / f"{name}_vis.jpg",
                image_bgr=denormalize_image_to_bgr(image[idx]),
                gt_depth=depth[idx].detach().cpu().numpy(),
                pred_depth=pred_depth[idx].detach().cpu().numpy(),
                valid_mask=valid[idx].detach().cpu().numpy().astype(bool),
                score=sirmse_eval_from_disparity(pred_disp[idx], depth[idx], valid[idx]) or float("nan"),
                pred_label=pred_label,
            )
            saved += 1


def write_outputs(
    *,
    output_dir: Path,
    config: dict[str, Any],
    checkpoint_path: Path,
    result: EvaluationResult,
) -> dict[str, Any]:
    summary = result.summary()
    payload = {
        "summary": summary,
        "checkpoint": str(checkpoint_path),
        "config_name": config["experiment"].get("name"),
        "model": config.get("model", {}),
        "data": {
            "root": config["data"].get("root"),
            "views": config["data"].get("views"),
            "eval_protocol": config["data"].get("eval_protocol"),
            "image_size": config["data"].get("image_size"),
            "val_fraction": config["data"].get("val_fraction"),
            "split_seed": config["data"].get("split_seed"),
            "max_samples": config["data"].get("max_samples"),
        },
        "sample_names": result.sample_names,
        "scores": result.scores,
    }
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    lines = [
        f"Samples evaluated : {summary['samples_evaluated']}",
        f"siRMSE mean       : {summary['sirmse_mean']:.4f}",
        f"siRMSE median     : {summary['sirmse_median']:.4f}",
        f"siRMSE std        : {summary['sirmse_std']:.4f}",
        f"siRMSE min        : {summary['sirmse_min']:.4f}",
        f"siRMSE max        : {summary['sirmse_max']:.4f}",
        f"Checkpoint        : {checkpoint_path}",
        f"Eval protocol     : {config['data'].get('eval_protocol')}",
    ]
    (output_dir / "eval_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return payload



def build_evaluation_model(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_payload: Any,
    device: torch.device,
) -> torch.nn.Module:
    if is_trainable_checkpoint(checkpoint_payload):
        base_checkpoint = checkpoint_payload.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError("Trainable-only checkpoint is missing base_checkpoint metadata")
        model = load_da2_model(
            encoder=config["model"]["encoder"],
            da2_repo=config["paths"]["da2_repo"],
            checkpoint_path=base_checkpoint,
            map_location="cpu",
        )
        apply_base_trainable_scope(model, config["base"]["trainable_scope"])
        apply_adapters_from_config(model, config)
        restore_model_payload(model, checkpoint_payload)
        return model.to(device).eval()

    if is_full_model_checkpoint(checkpoint_payload):
        model = create_da2_model(encoder=config["model"]["encoder"], da2_repo=config["paths"]["da2_repo"])
        apply_base_trainable_scope(model, config["base"]["trainable_scope"])
        apply_adapters_from_config(model, config)
        restore_model_payload(model, checkpoint_payload)
        return model.to(device).eval()

    if adapter_is_enabled(config):
        raise ValueError(
            "Config enables adapters, but the checkpoint is an official/full base state_dict with no adapter payload. "
            "Evaluate a trainable-only adapter checkpoint or disable adapter.enabled."
        )
    model = create_da2_model(encoder=config["model"]["encoder"], da2_repo=config["paths"]["da2_repo"])
    model.load_state_dict(extract_state_dict(checkpoint_payload), strict=True)
    return model.to(device).eval()

def main() -> None:
    args = parse_args()
    setup_logging()
    config = resolve_config(args)
    protocol = config["data"].get("eval_protocol", "native_resolution")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported eval protocol '{protocol}'. Expected one of {PROTOCOLS}")
    checkpoint_path = resolve_checkpoint(config)

    if args.dry_run:
        output_dir = resolve_output_dir(config, args.output_dir, create=False)
        LOGGER.info("Config       : %s", args.config)
        LOGGER.info("Experiment   : %s", config["experiment"].get("name"))
        LOGGER.info("Checkpoint   : %s", checkpoint_path)
        LOGGER.info("Eval protocol: %s", protocol)
        LOGGER.info("Output dir   : %s", output_dir)
        LOGGER.info("Dry run complete before data/model loading.")
        return

    checkpoint_payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    if checkpoint_has_config(checkpoint_payload):
        config = apply_eval_cli_overrides(checkpoint_payload["config"], args)
        protocol = config["data"].get("eval_protocol", "native_resolution")
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unsupported eval protocol '{protocol}'. Expected one of {PROTOCOLS}")
        checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else checkpoint_path

    output_dir = resolve_output_dir(config, args.output_dir, create=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir else (output_dir / "visualizations" if args.num_vis > 0 else None)

    LOGGER.info("Config       : %s", args.config)
    LOGGER.info("Experiment   : %s", config["experiment"].get("name"))
    LOGGER.info("Checkpoint   : %s", checkpoint_path)
    LOGGER.info("Eval protocol: %s", protocol)
    LOGGER.info("Output dir   : %s", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_evaluation_model(
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        device=device,
    )

    if protocol == "raw_infer_native":
        fraction = args.fraction if args.fraction is not None else float(config["data"].get("val_fraction", 0.1))
        filenames = select_filenames(
            config["data"]["root"],
            fraction=fraction,
            max_samples=config["data"].get("max_samples"),
            seed=int(config["data"].get("split_seed", 42)),
        )
        result = evaluate_raw_infer_native(
            model=model,
            data_dir=config["data"]["root"],
            filenames=filenames,
            input_size=int(config["data"].get("image_size", 518)),
            device=device,
            vis_dir=vis_dir,
            num_vis=int(args.num_vis),
        )
    else:
        pipeline = pipeline_for_protocol(config, protocol)
        loaders = build_da2_dataloaders(
            data_dir=config["data"]["root"],
            pipeline=pipeline,
            input_size=int(config["data"]["image_size"]),
            batch_size=int(config["train"].get("batch_size", 1)),
            val_fraction=float(config["data"].get("val_fraction", 0.05)),
            split_seed=int(config["data"].get("split_seed", 42)),
            num_workers=int(config["train"].get("num_workers", 4)),
            split_file=config["data"].get("split_file"),
            max_samples=config["data"].get("max_samples"),
            pin_memory=(device.type == "cuda"),
            drop_last_train=False,
        )
        result = evaluate_loader(model, loaders.val_loader, device)
        if vis_dir is not None:
            save_loader_visualizations(
                model=model,
                loader=loaders.val_loader,
                device=device,
                vis_dir=vis_dir,
                num_vis=int(args.num_vis),
                pred_label=f"Pred depth ({protocol})",
            )

    payload = write_outputs(output_dir=output_dir, config=config, checkpoint_path=checkpoint_path, result=result)
    summary = payload["summary"]
    LOGGER.info(
        "siRMSE mean=%.4f median=%.4f std=%.4f samples=%d",
        summary["sirmse_mean"],
        summary["sirmse_median"],
        summary["sirmse_std"],
        summary["samples_evaluated"],
    )

    wandb_run = maybe_init_wandb(config, output_dir, args.no_wandb)
    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in summary.items()})
        wandb_run.summary.update(summary)
        wandb_run.summary["checkpoint"] = str(checkpoint_path)
        wandb_run.summary["eval_protocol"] = protocol
        wandb_run.finish()


if __name__ == "__main__":
    main()
