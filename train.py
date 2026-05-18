#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.data_loader import build_cil_loaders
from utils.eval import evaluate_names
from models.da2 import build_da2
from models.da2_refine import build_da2_unet_refine
from models.unet import UNetBaseline
from utils.loss import sirmse
from utils import DEFAULT_CONFIG, apply_overrides, load_config, make_run_dir, maybe_wandb, save_json, save_yaml, seed_everything, setup_logging

LOGGER = logging.getLogger("cil_depth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIL depth models")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--run-name")
    p.add_argument("--data-root")
    p.add_argument("--output-root")
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--img-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--scheduler", choices=["constant", "poly_decay"])
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--save-policy", choices=["trainable_only", "full_model"])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_model(cfg: dict[str, Any]) -> tuple[torch.nn.Module, Path | None]:
    model_name = cfg["model"]
    if model_name == "da2_unet_refine":
        return build_da2_unet_refine(cfg)
    if isinstance(model_name, str) and model_name.startswith("da2_"):
        return build_da2(cfg)
    if model_name == "unet":
        return UNetBaseline(), None
    raise ValueError(f"Unknown model: {model_name}")


def forward_on_depth_grid(model: torch.nn.Module, image: torch.Tensor, depth: torch.Tensor, invert_output: bool) -> torch.Tensor:
    pred = model(image)
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if invert_output:
        pred = 1.0 / pred.clamp_min(1e-6)
    if pred.shape[-2:] != depth.shape[-2:]:
        pred = F.interpolate(pred[:, None], size=depth.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return pred


def batch_sirmse_loss(pred_depth: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = [sirmse(pred_depth[i], depth[i], valid[i]) for i in range(depth.shape[0])]
    return torch.stack(losses).mean()


def batch_region_sirmse_loss(pred_depth: torch.Tensor, depth: torch.Tensor, loss_masks: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(depth.shape[0]):
        sample_losses = []
        sample_weights = []
        for j in range(loss_masks.shape[1]):
            if int(loss_masks[i, j].sum()) == 0 or float(loss_weights[i, j]) <= 0:
                continue
            sample_losses.append(sirmse(pred_depth[i], depth[i], loss_masks[i, j]))
            sample_weights.append(loss_weights[i, j])
        weights = torch.stack(sample_weights)
        weights = weights / weights.sum().clamp_min(1e-6)
        losses.append((torch.stack(sample_losses) * weights).sum())
    return torch.stack(losses).mean()


def save_checkpoint(path: Path, cfg: dict[str, Any], model: torch.nn.Module, optimizer, scaler, epoch: int, global_step: int, best: float, val: float, base_ckpt: Path | None, include_optimizer: bool) -> None:
    policy = cfg.get("checkpoint", {}).get("save_policy", "trainable_only")
    if policy == "trainable_only" and str(cfg["model"]).startswith("da2_"):
        payload = {"format": "trainable_only", "trainable": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, "base_checkpoint": str(base_ckpt)}
    else:
        payload = {"format": "full_model", "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    payload.update({"config": cfg, "epoch": epoch, "global_step": global_step, "best_sirmse": best, "val_sirmse": val})
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def restore_if_requested(cfg: dict[str, Any], model: torch.nn.Module, optimizer, scaler) -> tuple[int, int, float]:
    resume = cfg.get("train", {}).get("resume")
    if not resume:
        return 1, 0, float("inf")
    payload = torch.load(resume, map_location="cpu")
    if payload.get("format") == "trainable_only":
        state = model.state_dict()
        state.update(payload.get("trainable", {}))
        model.load_state_dict(state, strict=True)
    elif "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0)) + 1, int(payload.get("global_step", 0)), float(payload.get("best_sirmse", float("inf")))


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = apply_overrides(load_config(args.config), {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "paths.output_root": args.output_root,
        "paths.checkpoint": os.path.expanduser(os.path.expandvars(args.checkpoint)) if args.checkpoint else None,
        "train.resume": os.path.expanduser(os.path.expandvars(args.resume)) if args.resume else None,
        "train.epochs": args.epochs,
        "train.batch_size": args.batch_size,
        "train.num_workers": args.num_workers,
        "train.scheduler": args.scheduler,
        "train.grad_accum_steps": args.grad_accum_steps,
        "logging.log_every": args.log_every,
        "data.image_size": args.img_size,
        "data.max_samples": args.max_samples,
        "checkpoint.save_policy": args.save_policy,
    })

    LOGGER.info("Config      : %s", args.config)
    LOGGER.info("Experiment  : %s", cfg.get("experiment", {}).get("name"))
    LOGGER.info("Model       : %s", cfg.get("model"))
    LOGGER.info("Data root   : %s", cfg.get("data", {}).get("root"))
    LOGGER.info("Augmentation: %s", cfg.get("augmentation", {}).get("name", "none"))
    if args.dry_run:
        return

    seed_everything(int(cfg.get("data", {}).get("split_seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg)
    save_yaml(run_dir / "effective_config.yaml", cfg)
    wandb_run = maybe_wandb(cfg, run_dir, job_type="train")

    train_loader, _, train_names, val_names = build_cil_loaders(cfg)
    model, base_ckpt = build_model(cfg)
    model.to(device)
    invert_output = str(cfg["model"]).startswith("da2_")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts = {"total": total, "trainable": trainable, "frozen": total - trainable, "trainable_pct": 100.0 * trainable / total if total else 0.0}
    LOGGER.info("Output dir  : %s", run_dir)
    LOGGER.info("Samples     : train=%d val=%d", len(train_names), len(val_names))
    LOGGER.info("Params      : total=%s trainable=%s frozen=%s trainable=%.2f%%", f"{counts['total']:,}", f"{counts['trainable']:,}", f"{counts['frozen']:,}", counts["trainable_pct"])

    train_cfg = cfg.get("train", {})
    early_cfg = train_cfg.get("early_stopping", {}) or {}
    early_stopping = {
        "enabled": bool(early_cfg.get("enabled", False)),
        "patience": max(1, int(early_cfg.get("patience", 3))),
        "min_delta": max(0.0, float(early_cfg.get("min_delta", 0.0))),
    }
    if early_stopping["enabled"]:
        LOGGER.info(
            "Early stopping enabled: patience=%d min_delta=%g metric=val_siRMSE",
            early_stopping["patience"],
            early_stopping["min_delta"],
        )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected")
    lr, wd = float(train_cfg.get("learning_rate", 1e-4)), float(train_cfg.get("weight_decay", 0.0))
    optimizer = AdamW(params, lr=lr, weight_decay=wd) if train_cfg.get("optimizer", "adamw").lower() == "adamw" else Adam(params, lr=lr, weight_decay=wd)
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, global_step, best = restore_if_requested(cfg, model, optimizer, scaler)

    epochs = int(train_cfg.get("epochs", 10))
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(cfg.get("logging", {}).get("log_every", 50)))
    scheduler = train_cfg.get("scheduler", "constant")
    total_steps = max(1, math.ceil(len(train_loader) / grad_accum) * max(1, epochs - start_epoch + 1))
    history = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"), start=1):
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            valid = batch["valid_mask"].to(device)
            loss_masks = batch.get("loss_masks")
            loss_weights = batch.get("loss_weights")
            with torch.cuda.amp.autocast(enabled=amp):
                pred = forward_on_depth_grid(model, image, depth, invert_output)
                if loss_masks is not None and loss_weights is not None:
                    # CutMix regions get separate siRMSE normalization.
                    loss = batch_region_sirmse_loss(pred, depth, loss_masks.to(device), loss_weights.to(device)) / grad_accum
                else:
                    loss = batch_sirmse_loss(pred, depth, valid) / grad_accum
            scaler.scale(loss).backward()
            loss_value = float(loss.item() * grad_accum)
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler == "poly_decay":
                    progress = min(global_step / total_steps, 1.0)
                    for group in optimizer.param_groups:
                        group["lr"] = lr * (1.0 - progress) ** 0.9
                elif scheduler != "constant":
                    raise ValueError(f"Unknown scheduler: {scheduler}")
                if wandb_run and (global_step == 1 or global_step % log_every == 0):
                    wandb_run.log({
                        "train/step_loss": loss_value,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/epoch_progress": step / len(train_loader),
                    }, step=global_step)
            losses.append(loss_value)

        val_result = evaluate_names(model, cfg, val_names, device)
        val = val_result["summary"]["sirmse_mean"]
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_sirmse": val})
        LOGGER.info("Epoch %d/%d train_loss=%.4f val_siRMSE=%.4f", epoch, epochs, train_loss, val)
        if wandb_run:
            wandb_run.log({"train/loss": train_loss, "val/sirmse": val, "epoch": epoch}, step=global_step)

        improvement_margin = early_stopping["min_delta"] if early_stopping["enabled"] else 0.0
        improved = np.isfinite(val) and val < best - improvement_margin
        if improved:
            best = val
            epochs_without_improvement = 0
            if cfg.get("checkpoint", {}).get("keep_best", True):
                save_checkpoint(run_dir / "best.pth", cfg, model, optimizer, scaler, epoch, global_step, best, val, base_ckpt, include_optimizer=False)
        elif early_stopping["enabled"]:
            epochs_without_improvement += 1

        if cfg.get("checkpoint", {}).get("keep_latest", True):
            save_checkpoint(run_dir / "latest.pth", cfg, model, optimizer, scaler, epoch, global_step, best, val, base_ckpt, include_optimizer=bool(cfg.get("checkpoint", {}).get("save_optimizer", True)))

        if early_stopping["enabled"] and epochs_without_improvement >= early_stopping["patience"]:
            stopped_early = True
            stop_epoch = epoch
            LOGGER.info("Early stopping at epoch %d: best val_siRMSE=%.4f", epoch, best)
            break

    summary = {
        "best_sirmse": best,
        "history": history,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stopping": early_stopping,
        "elapsed_seconds": time.time() - t0,
        "output_dir": str(run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    if wandb_run:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
