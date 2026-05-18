#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import DEFAULT_CONFIG, apply_overrides, load_config, save_json, save_yaml, setup_logging, timestamp
from utils.calibration import scale_depth_percentile

DEFAULT_TEST_ROOT = "/cluster/courses/cil/monocular-depth-estimation/test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict CIL test depth maps")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint")
    p.add_argument("--model", choices=["da2_vits", "da2_vitb", "da2_vitl", "da2_unet_refine", "unet"])
    p.add_argument("--test-root", default=DEFAULT_TEST_ROOT)
    p.add_argument("--output-dir")
    p.add_argument("--submission-csv")
    p.add_argument("--run-name")
    p.add_argument("--img-size", type=int)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def test_rgb_names(test_root: str | Path, max_samples: int | None = None) -> list[str]:
    names = sorted(path.name for path in Path(test_root).glob("test_*_rgb.png"))
    if max_samples is not None:
        names = names[:max_samples]
    if not names:
        raise FileNotFoundError(f"No test_*_rgb.png files found under {test_root}")
    return names


def default_output_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("paths", {}).get("output_root", "runs")).parent
    name = cfg.get("experiment", {}).get("name", "predict")
    return root / "predictions" / name / timestamp() / "preds"


def load_rgb(path: str | Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = apply_overrides(load_config(args.config), {
        "experiment.name": args.run_name,
        "model": args.model,
        "data.image_size": args.img_size,
    })
    test_root = Path(args.test_root)
    out = Path(args.output_dir) if args.output_dir else default_output_dir(cfg)
    names = test_rgb_names(test_root, args.max_samples)

    print(f"config={args.config}")
    print(f"model={cfg.get('model')}")
    print(f"checkpoint={args.checkpoint or cfg.get('paths', {}).get('checkpoint', 'config/default')}")
    print(f"test_root={test_root}")
    print(f"output_dir={out}")
    print(f"samples={len(names)}")
    if args.submission_csv:
        print(f"submission_csv={args.submission_csv}")
    if args.dry_run:
        return

    import torch
    from tqdm import tqdm

    from models.loading import load_model_for_inference
    from utils.eval import predict_depth_for_eval
    from utils.submission import write_submission_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(cfg, args.checkpoint, device)

    scaling = {"mode": "percentile", "percentile": 99.0, "target": 80.0, "max_clip": 60000.0}
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    scales = {}
    with torch.no_grad():
        for name in tqdm(names, desc="predict"):
            image = load_rgb(test_root / name)
            pred = predict_depth_for_eval(model, image, image.shape[:2], cfg, device)
            raw_pred = pred.detach().cpu().numpy().astype(np.float32)
            pred_np, scale, percentile_value, clipped = scale_depth_percentile(raw_pred, scaling["percentile"], scaling["target"], scaling["max_clip"])
            pred_name = f"{Path(name).stem.removesuffix('_rgb')}.npy"
            pred_path = out / pred_name
            np.save(pred_path, pred_np)
            saved.append(pred_path.name)
            scales[pred_path.name] = {"scale": scale, "source_percentile": percentile_value, "clipped_pixels": clipped}

    summary = {
        "samples_selected": len(names),
        "samples_predicted": len(saved),
        "test_root": str(test_root),
        "output_dir": str(out),
        "prediction_files": saved,
        "scaling": scaling,
        "scales": scales,
        "config": cfg,
    }
    if args.submission_csv:
        summary["submission_csv"] = str(args.submission_csv)
        summary["submission_rows"] = write_submission_csv(out, args.submission_csv)

    save_json(out / "predict_summary.json", summary)
    save_yaml(out / "effective_config.yaml", cfg)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"prediction_files", "config"}}, indent=2))


if __name__ == "__main__":
    main()
