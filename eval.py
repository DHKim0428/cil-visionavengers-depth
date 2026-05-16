#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.da2 import build_da2, load_training_checkpoint
from models.unet import UNetBaseline
from utils import DEFAULT_CONFIG, apply_overrides, load_config, maybe_wandb, save_json, save_yaml, setup_logging, timestamp
from utils.eval import evaluate_names, validation_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CIL depth models with siRMSE")
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
    p.add_argument("--save-images", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def build_model(cfg: dict[str, Any], checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    if checkpoint:
        cfg.setdefault("paths", {})["checkpoint"] = os.path.expanduser(os.path.expandvars(checkpoint))
    if str(cfg["model"]).startswith("da2_"):
        model_name = cfg["model"]
        ckpt = Path(cfg.get("paths", {}).get("checkpoint") or Path(cfg["paths"]["da2_checkpoint_dir"]) / f"depth_anything_v2_{model_name.removeprefix('da2_')}.pth")
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
        return model.to(device).eval()
    if cfg["model"] == "unet":
        if not checkpoint:
            raise ValueError("U-Net eval requires --checkpoint")
        model = UNetBaseline()
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload)
        return model.to(device).eval()
    raise ValueError(f"Unknown model: {cfg['model']}")


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
        "data.max_samples": args.max_samples,
    })
    out = Path(args.output_dir) if args.output_dir else Path(cfg.get("paths", {}).get("output_root", "runs")).parent / "evaluations" / cfg.get("experiment", {}).get("name", "eval") / timestamp()
    print(f"config={args.config}")
    print(f"model={cfg.get('model')}")
    print(f"output_dir={out}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)
    val_names = validation_names(cfg)
    if args.fraction:
        rng = np.random.default_rng(int(cfg["data"].get("split_seed", 42)))
        idx = sorted(rng.choice(len(val_names), size=max(1, int(len(val_names) * args.fraction)), replace=False))
        val_names = [val_names[i] for i in idx]
    if args.max_samples:
        val_names = val_names[:args.max_samples]

    out.mkdir(parents=True, exist_ok=True)
    result = evaluate_names(model, cfg, val_names, device, save_images=args.save_images, image_dir=out / "images")
    summary = result["summary"]
    save_json(out / "eval_summary.json", {"summary": summary, "scores": result["scores"], "selected_sample_names": val_names, "evaluated_sample_names": result["evaluated_sample_names"], "config": cfg})
    save_yaml(out / "effective_config.yaml", cfg)
    (out / "eval_summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.no_wandb:
        run = maybe_wandb(cfg, out, job_type="eval")
        if run:
            payload = {f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}
            if result["image_paths"]:
                import wandb
                payload["eval/images"] = [wandb.Image(str(path)) for path in result["image_paths"]]
            run.log(payload)
            run.finish()


if __name__ == "__main__":
    main()
