#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.loading import load_model_for_inference
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
    p.add_argument("--scale-depth-percentile", action="store_true")
    p.add_argument("--scale-percentile", type=float, default=99.0)
    p.add_argument("--scale-target", type=float, default=80.0)
    p.add_argument("--scale-max-clip", type=float, default=60000.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()



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
    scaling = None
    if args.scale_depth_percentile:
        scaling = {"mode": "percentile", "percentile": args.scale_percentile, "target": args.scale_target, "max_clip": args.scale_max_clip}
        print(f"scaling={scaling}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(cfg, args.checkpoint, device)
    val_names = validation_names(cfg)
    if args.fraction:
        rng = np.random.default_rng(int(cfg["data"].get("split_seed", 42)))
        idx = sorted(rng.choice(len(val_names), size=max(1, int(len(val_names) * args.fraction)), replace=False))
        val_names = [val_names[i] for i in idx]
    if args.max_samples:
        val_names = val_names[:args.max_samples]

    out.mkdir(parents=True, exist_ok=True)
    result = evaluate_names(model, cfg, val_names, device, save_images=args.save_images, image_dir=out / "images", scaling=scaling)
    summary = result["summary"]
    payload = {"summary": summary, "scores": result["scores"], "selected_sample_names": val_names, "evaluated_sample_names": result["evaluated_sample_names"], "config": cfg}
    if scaling is not None:
        payload["scaling"] = scaling
        payload["scales"] = result["scales"]
    save_json(out / "eval_summary.json", payload)
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
