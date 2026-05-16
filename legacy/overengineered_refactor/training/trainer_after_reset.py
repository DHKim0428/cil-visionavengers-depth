from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from tqdm import tqdm

from dataset.cil_depth import build_cil_loaders
from models.da2 import build_da2, load_training_checkpoint, parameter_counts, trainable_state
from models.unet import UNetBaseline
from training.losses_metrics import EPS, MAX_DEPTH, MIN_DEPTH, sirmse
from training.utils import make_run_dir, maybe_wandb, save_json, save_yaml, seed_everything

LOGGER = logging.getLogger("cil_depth")


class Trainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        seed_everything(int(cfg.get("data", {}).get("split_seed", 42)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = make_run_dir(cfg)
        save_yaml(self.run_dir / "effective_config.yaml", cfg)
        self.wandb = maybe_wandb(cfg, self.run_dir, job_type="train")

        self.train_loader, self.val_loader, self.train_names, self.val_names = build_cil_loaders(cfg)
        self.model, self.prediction_kind, self.base_checkpoint = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("train", {}).get("amp", False)) and self.device.type == "cuda")
        self.start_epoch = 1
        self.global_step = 0
        self.best = float("inf")

        resume = cfg.get("train", {}).get("resume")
        if resume:
            payload = load_training_checkpoint(self.model, resume)
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            if "scaler" in payload:
                self.scaler.load_state_dict(payload["scaler"])
            self.start_epoch = int(payload.get("epoch", 0)) + 1
            self.global_step = int(payload.get("global_step", 0))
            self.best = float(payload.get("best_sirmse", self.best))

        counts = parameter_counts(self.model) if cfg["model"]["family"] == "da2_relative" else self._parameter_counts()
        LOGGER.info("Output dir  : %s", self.run_dir)
        LOGGER.info("Device      : %s", self.device)
        LOGGER.info("Samples     : train=%d val=%d", len(self.train_names), len(self.val_names))
        LOGGER.info("Params      : total=%s trainable=%s frozen=%s trainable=%.2f%%", f"{counts['total']:,}", f"{counts['trainable']:,}", f"{counts['frozen']:,}", counts["trainable_pct"])

    def _build_model(self) -> tuple[torch.nn.Module, str, Path | None]:
        family = self.cfg["model"]["family"]
        if family == "da2_relative":
            model, ckpt = build_da2(self.cfg)
            return model, "disparity", ckpt
        if family == "unet":
            kind = self.cfg["model"].get("prediction_kind", "disparity")
            if kind not in {"disparity", "depth"}:
                raise ValueError("U-Net training supports prediction_kind: disparity or depth")
            return UNetBaseline(prediction_kind=kind), kind, None
        raise ValueError(f"Unknown model.family: {family}")

    def _parameter_counts(self) -> dict[str, float | int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable, "trainable_pct": 100.0 * trainable / total if total else 0.0}

    def _build_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters selected")
        train = self.cfg.get("train", {})
        opt = train.get("optimizer", "adamw").lower()
        lr = float(train.get("learning_rate", 1e-4))
        wd = float(train.get("weight_decay", 0.0))
        return AdamW(trainable, lr=lr, weight_decay=wd) if opt == "adamw" else Adam(trainable, lr=lr, weight_decay=wd)

    def prediction_to_depth(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if self.prediction_kind == "disparity":
            pred = torch.nan_to_num(pred, nan=EPS, posinf=1.0 / MIN_DEPTH, neginf=EPS).clamp_min(EPS)
            return (1.0 / pred).clamp(MIN_DEPTH, MAX_DEPTH)
        return torch.nan_to_num(pred, nan=MIN_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH).clamp(MIN_DEPTH, MAX_DEPTH)

    def forward_for_loss(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        pred = self.model(image)
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred[:, None], size=depth.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return pred

    def loss(self, pred: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        pred_depth = self.prediction_to_depth(pred)
        losses = [score for i in range(depth.shape[0]) if (score := sirmse(pred_depth[i], depth[i], valid[i])) is not None]
        return torch.stack(losses).mean() if losses else pred_depth.sum() * 0.0

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        scores = []
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            image = batch["image"].to(self.device)
            depth = batch["depth"].to(self.device)
            valid = batch["valid_mask"].to(self.device)
            pred_depth = self.prediction_to_depth(self.forward_for_loss(image, depth))
            for i in range(depth.shape[0]):
                score = sirmse(pred_depth[i], depth[i], valid[i])
                if score is not None:
                    scores.append(float(score.item()))
        return float(np.mean(scores)) if scores else float("nan")

    def save_checkpoint(self, name: str, epoch: int, val_sirmse: float, include_optimizer: bool) -> None:
        policy = self.cfg.get("checkpoint", {}).get("save_policy", "trainable_only")
        if policy == "trainable_only" and self.cfg["model"]["family"] == "da2_relative":
            payload = {"format": "trainable_only", "trainable": trainable_state(self.model), "base_checkpoint": str(self.base_checkpoint)}
        else:
            payload = {"format": "full_model", "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
        payload.update({"epoch": epoch, "global_step": self.global_step, "best_sirmse": self.best, "val_sirmse": val_sirmse, "config": self.cfg})
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
            payload["scaler"] = self.scaler.state_dict()
        torch.save(payload, self.run_dir / name)

    def fit(self) -> dict[str, Any]:
        train = self.cfg.get("train", {})
        epochs = int(train.get("epochs", 10))
        grad_accum = max(1, int(train.get("grad_accum_steps", 1)))
        amp = bool(train.get("amp", False)) and self.device.type == "cuda"
        history = []
        start = time.time()

        for epoch in range(self.start_epoch, epochs + 1):
            self.model.train()
            losses = []
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {epoch}/{epochs}"), start=1):
                image = batch["image"].to(self.device)
                depth = batch["depth"].to(self.device)
                valid = batch["valid_mask"].to(self.device)
                with torch.cuda.amp.autocast(enabled=amp):
                    loss = self.loss(self.forward_for_loss(image, depth), depth, valid) / grad_accum
                self.scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == len(self.train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                losses.append(float(loss.item() * grad_accum))

            val = self.evaluate()
            train_loss = float(np.mean(losses)) if losses else float("nan")
            history.append({"epoch": epoch, "train_loss": train_loss, "val_sirmse": val})
            LOGGER.info("Epoch %d/%d train_loss=%.4f val_siRMSE=%.4f", epoch, epochs, train_loss, val)
            if self.wandb:
                self.wandb.log({"train/loss": train_loss, "val/sirmse": val, "epoch": epoch}, step=self.global_step)

            if np.isfinite(val) and val < self.best:
                self.best = val
                if self.cfg.get("checkpoint", {}).get("keep_best", True):
                    self.save_checkpoint("best.pth", epoch, val, include_optimizer=False)
            if self.cfg.get("checkpoint", {}).get("keep_latest", True):
                self.save_checkpoint("latest.pth", epoch, val, include_optimizer=bool(self.cfg.get("checkpoint", {}).get("save_optimizer", True)))

        summary = {"best_sirmse": self.best, "history": history, "elapsed_seconds": time.time() - start, "output_dir": str(self.run_dir)}
        save_json(self.run_dir / "summary.json", summary)
        if self.wandb:
            self.wandb.summary.update(summary)
            self.wandb.finish()
        return summary
