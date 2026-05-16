#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.loading import load_model_for_inference
from utils import DEFAULT_CONFIG, apply_overrides, load_config, save_json, save_yaml, setup_logging, timestamp
from utils.eval import load_rgb_depth, predict_depth_for_eval, validation_names
from utils.loss import EPS, MAX_DEPTH, MIN_DEPTH, sirmse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare current and legacy-compatible DA2 eval protocols")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint")
    p.add_argument("--run-name")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--img-size", type=int)
    p.add_argument("--split-file")
    p.add_argument("--val-fraction", type=float)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--fraction", type=float)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def summarize(scores: list[float]) -> dict[str, float | int]:
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "samples": int(arr.size),
    }


def legacy_sirmse_from_raw(raw_disp: np.ndarray, gt: np.ndarray) -> float:
    valid = np.isfinite(gt) & (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
    if int(valid.sum()) == 0:
        raise ValueError("No valid ground-truth depth pixels for legacy siRMSE")
    pred_depth = np.clip(1.0 / (raw_disp[valid].astype(np.float64) + EPS), MIN_DEPTH, MAX_DEPTH)
    gt_valid = gt[valid].astype(np.float64)
    diff = np.log(pred_depth) - np.log(gt_valid)
    return float(np.sqrt(np.mean(diff ** 2) - np.mean(diff) ** 2))


@torch.no_grad()
def current_score(model: torch.nn.Module, cfg: dict, image_rgb: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    gt_t = torch.from_numpy(gt).float().to(device)
    pred = predict_depth_for_eval(model, image_rgb, gt_t.shape, cfg, device)
    return float(sirmse(pred, gt_t).item())


@torch.no_grad()
def legacy_score(model: torch.nn.Module, image_rgb: np.ndarray, gt: np.ndarray, input_size: int) -> float:
    if not hasattr(model, "infer_image"):
        raise ValueError("legacy protocol requires a DA2 model with infer_image()")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    raw_disp = model.infer_image(bgr, input_size)
    if raw_disp.shape != gt.shape:
        raw_disp = cv2.resize(raw_disp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return legacy_sirmse_from_raw(raw_disp, gt)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = apply_overrides(load_config(args.config), {
        "experiment.name": args.run_name,
        "data.root": args.data_root,
        "data.image_size": args.img_size,
        "data.split_file": args.split_file,
        "data.val_fraction": args.val_fraction,
    })
    run_name = args.run_name or f"{cfg.get('experiment', {}).get('name', 'eval')}_protocol_compare"
    out = Path(args.output_dir) if args.output_dir else Path(cfg.get("paths", {}).get("output_root", "runs")).parent / "evaluations" / run_name / timestamp()

    names = validation_names(cfg)
    if args.fraction:
        rng = np.random.default_rng(int(cfg["data"].get("split_seed", 42)))
        idx = sorted(rng.choice(len(names), size=max(1, int(len(names) * args.fraction)), replace=False))
        names = [names[i] for i in idx]
    if args.max_samples:
        names = names[:args.max_samples]

    print(f"config={args.config}")
    print(f"model={cfg.get('model')}")
    print(f"output_dir={out}")
    print(f"samples={len(names)}")
    print("protocols=current,legacy_infer")
    if args.dry_run:
        return

    if not str(cfg["model"]).startswith("da2_"):
        raise ValueError("legacy protocol comparison is currently DA2-only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(cfg, args.checkpoint, device)
    input_size = int(cfg.get("data", {}).get("image_size", 518))
    root = Path(cfg["data"]["root"])

    current_scores: list[float] = []
    legacy_scores: list[float] = []
    rows: list[dict[str, float | str]] = []

    for name in tqdm(names, desc="compare", leave=False):
        image, gt = load_rgb_depth(root, name)
        current = current_score(model, cfg, image, gt, device)
        legacy = legacy_score(model, image, gt, input_size)
        current_scores.append(current)
        legacy_scores.append(legacy)
        rows.append({
            "name": name,
            "current": current,
            "legacy_infer": legacy,
            "legacy_minus_current": legacy - current,
        })

    diffs = [float(row["legacy_minus_current"]) for row in rows]
    summary = {
        "current": summarize(current_scores),
        "legacy_infer": summarize(legacy_scores),
        "legacy_minus_current": summarize(diffs),
        "samples_selected": len(names),
        "samples_evaluated": len(rows),
    }
    payload = {
        "summary": summary,
        "scores": rows,
        "selected_sample_names": names,
        "config": cfg,
        "protocol_notes": {
            "current": "utils.eval.predict_depth_for_eval: 1/raw.clamp_min(EPS), then torch bilinear resize align_corners=False, utils.loss.sirmse",
            "legacy_infer": "model.infer_image BGR path, raw resized by DA2 align_corners=True, then np.clip(1/(raw+EPS), MIN_DEPTH, MAX_DEPTH), legacy numpy siRMSE",
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "protocol_compare_summary.json", payload)
    save_yaml(out / "effective_config.yaml", cfg)
    (out / "protocol_compare_summary.txt").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
