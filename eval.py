#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.data_loader import rgb_names, split_names
from models.da2 import build_da2, default_checkpoint_path, load_training_checkpoint
from models.unet import UNetBaseline
from utils.loss import EPS, MAX_DEPTH, MIN_DEPTH, sirmse
from utils import DEFAULT_CONFIG, apply_overrides, load_config, maybe_wandb, save_json, save_yaml, setup_logging, timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CIL depth models with siRMSE")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint")
    p.add_argument("--run-name")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--img-size", type=int)
    p.add_argument("--protocol")
    p.add_argument("--split-file")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--fraction", type=float)
    p.add_argument("--save-images", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load_pair(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(root / name))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {root / name}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(root / name.replace("_rgb.png", "_depth.npy")).astype(np.float32)
    return image, depth


def preprocess(image: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor:
    size = int(cfg.get("data", {}).get("image_size", 518))
    family = cfg["model"]["family"]
    image = image.astype(np.float32) / 255.0
    if family == "unet" or cfg.get("data", {}).get("eval_protocol") == "legacy_square":
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        h, w = image.shape[:2]
        scale = max(size / h, size / w)
        image = cv2.resize(image, (int(np.ceil(w * scale / 14) * 14), int(np.ceil(h * scale / 14) * 14)), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    if cfg.get("data", {}).get("views", {}).get("eval", {}).get("normalize") == "imagenet" or family == "da2_relative":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
    return x[None]


def build_model(cfg: dict[str, Any], checkpoint: str | None, device: torch.device) -> tuple[torch.nn.Module, str]:
    if checkpoint:
        cfg.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(checkpoint))
    if cfg["model"]["family"] == "da2_relative":
        ckpt = Path(cfg.get("paths", {}).get("checkpoint") or default_checkpoint_path(cfg))
        payload = torch.load(ckpt, map_location="cpu")
        if isinstance(payload, dict) and payload.get("format") == "trainable_only":
            cfg.setdefault("paths", {})["checkpoint"] = payload["base_checkpoint"]
            model, _ = build_da2(cfg)
            load_training_checkpoint(model, ckpt)
        elif isinstance(payload, dict) and payload.get("format") == "full_model":
            cfg.setdefault("paths", {}).pop("checkpoint", None)
            model, _ = build_da2(cfg)
            load_training_checkpoint(model, ckpt)
        else:
            model, _ = build_da2(cfg)
        return model.to(device).eval(), "disparity"
    if cfg["model"]["family"] == "unet":
        if not checkpoint:
            raise ValueError("U-Net eval requires --checkpoint")
        kind = cfg["model"].get("prediction_kind", "disparity")
        model = UNetBaseline(prediction_kind=kind)
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload)
        return model.to(device).eval(), kind
    raise ValueError(f"Unknown model.family: {cfg['model']['family']}")


def pred_to_depth(pred: torch.Tensor, kind: str) -> torch.Tensor:
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if kind == "disparity":
        pred = torch.nan_to_num(pred, nan=EPS, posinf=1.0 / MIN_DEPTH, neginf=EPS).clamp_min(EPS)
        return (1.0 / pred).clamp(MIN_DEPTH, MAX_DEPTH)
    return torch.nan_to_num(pred, nan=MIN_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH).clamp(MIN_DEPTH, MAX_DEPTH)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = apply_overrides(load_config(args.config), {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "data.image_size": args.img_size,
        "data.eval_protocol": args.protocol,
        "data.split_file": args.split_file,
        "data.max_samples": args.max_samples,
    })
    out = Path(args.output_dir) if args.output_dir else Path(cfg.get("paths", {}).get("output_root", "runs")).parent / "evaluations" / cfg.get("experiment", {}).get("name", "eval") / timestamp()
    print(f"config={args.config}")
    print(f"model={cfg.get('model')}")
    print(f"output_dir={out}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kind = build_model(cfg, args.checkpoint, device)
    all_names = rgb_names(cfg["data"]["root"])
    _, val_names = split_names(all_names, float(cfg["data"].get("val_fraction", 0.05)), int(cfg["data"].get("split_seed", 42)), cfg["data"].get("split_file"))
    if args.fraction:
        rng = np.random.default_rng(int(cfg["data"].get("split_seed", 42)))
        idx = sorted(rng.choice(len(val_names), size=max(1, int(len(val_names) * args.fraction)), replace=False))
        val_names = [val_names[i] for i in idx]
    if args.max_samples:
        val_names = val_names[:args.max_samples]

    root = Path(cfg["data"]["root"])
    scores = []
    evaluated = []
    image_logs = []
    out.mkdir(parents=True, exist_ok=True)
    image_dir = out / "images"
    if args.save_images > 0:
        image_dir.mkdir(parents=True, exist_ok=True)

    for name in tqdm(val_names, desc="eval"):
        image, gt = load_pair(root, name)
        gt_t = torch.from_numpy(gt).float().to(device)
        if cfg["model"]["family"] == "da2_relative":
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pred = torch.from_numpy(model.infer_image(image_bgr, int(cfg["data"].get("image_size", 518)))).float().to(device)
            depth = pred_to_depth(pred, kind)
        else:
            x = preprocess(image, cfg).to(device)
            depth = pred_to_depth(model(x), kind)[0]
        if depth.shape != gt_t.shape:
            depth = F.interpolate(depth[None, None], size=gt_t.shape, mode="bilinear", align_corners=False)[0, 0]
        score = sirmse(depth, gt_t)
        if score is not None:
            scores.append(float(score.item()))
            evaluated.append(name)
            if len(image_logs) < args.save_images:
                gt_np = gt_t.detach().cpu().numpy()
                pred_np = depth.detach().cpu().numpy()
                valid = np.isfinite(gt_np) & (gt_np >= MIN_DEPTH) & (gt_np <= MAX_DEPTH)
                rgb_small = cv2.resize(image, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_AREA)
                gt_vis = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
                pred_vis = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
                if valid.any():
                    lo, hi = np.percentile(gt_np[valid], [2, 98])
                    hi = hi if hi > lo else lo + 1.0
                    gt_u8 = np.clip((gt_np - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
                    pred_u8 = np.clip((pred_np - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
                    gt_vis = cv2.cvtColor(cv2.applyColorMap(gt_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                    pred_vis = cv2.cvtColor(cv2.applyColorMap(pred_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                    gt_vis[~valid] = 0
                panel = np.concatenate([rgb_small, gt_vis, pred_vis], axis=1)
                path = image_dir / f"{Path(name).stem}_sirmse_{float(score.item()):.4f}.png"
                cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                image_logs.append(path)

    summary = {
        "sirmse_mean": float(np.mean(scores)) if scores else float("nan"),
        "sirmse_median": float(np.median(scores)) if scores else float("nan"),
        "sirmse_std": float(np.std(scores)) if scores else float("nan"),
        "samples_selected": len(val_names),
        "samples_evaluated": len(scores),
    }
    save_json(out / "eval_summary.json", {"summary": summary, "scores": scores, "selected_sample_names": val_names, "evaluated_sample_names": evaluated, "config": cfg})
    save_yaml(out / "effective_config.yaml", cfg)
    (out / "eval_summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.no_wandb:
        run = maybe_wandb(cfg, out, job_type="eval")
        if run:
            payload = {f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}
            if image_logs:
                import wandb
                payload["eval/images"] = [wandb.Image(str(path)) for path in image_logs]
            run.log(payload)
            run.finish()


if __name__ == "__main__":
    main()
