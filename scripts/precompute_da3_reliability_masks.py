import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = "/cluster/courses/cil/monocular-depth-estimation/train"
DEFAULT_OUTPUT_DIR = (
    f"/work/scratch/{os.environ.get('USER', 'student')}/"
    "cil-visionavengers-depth/teacher_masks/da3_giant_p95_img128_seed42"
)
DEFAULT_MODEL_DIR = "depth-anything/DA3-GIANT-1.1"
EPS = 1e-6


def add_da3_repo_to_path(da3_repo):
    if not da3_repo:
        return
    repo = Path(da3_repo).expanduser().resolve()
    src = repo / "src"
    for path in (repo, src):
        if path.exists():
            sys.path.insert(0, str(path))


def import_da3(da3_repo):
    add_da3_repo_to_path(da3_repo)
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Could not import depth_anything_3. Install Depth-Anything-3 in the "
            "active environment, or pass --da3_repo /path/to/Depth-Anything-3 "
            "or set DA3_REPO=/path/to/Depth-Anything-3."
        ) from exc
    return DepthAnything3


def load_da3_model(model_dir, device, da3_repo):
    DepthAnything3 = import_da3(da3_repo)
    if hasattr(DepthAnything3, "from_pretrained"):
        model = DepthAnything3.from_pretrained(model_dir)
    else:
        model = DepthAnything3(model_name=model_dir)
    return model.to(device=device).eval()


def resize_array_to_shape(array, shape, mode):
    if array.shape == shape:
        return array.astype(np.float32, copy=False)

    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor, size=shape, mode=mode)
    else:
        resized = F.interpolate(tensor, size=shape, mode=mode, align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def load_resized_depth(depth_path, img_size):
    depth = np.load(depth_path).astype(np.float32)
    return resize_array_to_shape(depth, (img_size, img_size), mode="nearest")


def run_da3_inference(model, image_path, process_res):
    kwargs = {}
    if process_res is not None and process_res > 0:
        kwargs["process_res"] = process_res
    prediction = model.inference([str(image_path)], **kwargs)
    depth = np.asarray(prediction.depth)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected DA3 depth with shape [H,W] or [N,H,W], got {depth.shape}")
    return depth.astype(np.float32)


def load_train_names(split_file, max_samples):
    split_path = Path(split_file)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_names = list(split["train_names"])
    if max_samples is not None:
        train_names = train_names[:max_samples]
    return train_names, split


def compute_reliability_mask(teacher_depth, target_depth, threshold_percentile):
    valid = (target_depth > 0) & np.isfinite(target_depth) & (teacher_depth > 0) & np.isfinite(teacher_depth)
    valid_pixels = int(valid.sum())
    if valid_pixels < 2:
        reliable = np.zeros_like(valid, dtype=bool)
        return reliable, {
            "valid_pixels": valid_pixels,
            "kept_pixels": 0,
            "removed_pixels": 0,
            "removed_valid_ratio": float("nan"),
            "threshold_absrel": float("nan"),
            "median_scale": float("nan"),
            "absrel_median": float("nan"),
            "absrel_p95": float("nan"),
            "absrel_p99": float("nan"),
        }

    teacher_values = teacher_depth[valid].astype(np.float64)
    target_values = target_depth[valid].astype(np.float64)
    teacher_median = float(np.median(teacher_values))
    target_median = float(np.median(target_values))
    if teacher_median <= 0 or target_median <= 0:
        reliable = valid.copy()
        return reliable, {
            "valid_pixels": valid_pixels,
            "kept_pixels": valid_pixels,
            "removed_pixels": 0,
            "removed_valid_ratio": 0.0,
            "threshold_absrel": float("nan"),
            "median_scale": float("nan"),
            "absrel_median": float("nan"),
            "absrel_p95": float("nan"),
            "absrel_p99": float("nan"),
        }

    scale = target_median / teacher_median
    teacher_scaled = teacher_depth.astype(np.float64) * scale
    absrel = np.zeros_like(target_depth, dtype=np.float64)
    absrel[valid] = np.abs(teacher_scaled[valid] - target_depth[valid]) / np.maximum(
        target_depth[valid], EPS
    )

    threshold = float(np.percentile(absrel[valid], threshold_percentile))
    reliable = valid & (absrel <= threshold)
    kept_pixels = int(reliable.sum())
    removed_pixels = valid_pixels - kept_pixels
    return reliable, {
        "valid_pixels": valid_pixels,
        "kept_pixels": kept_pixels,
        "removed_pixels": removed_pixels,
        "removed_valid_ratio": float(removed_pixels / max(valid_pixels, 1)),
        "threshold_absrel": threshold,
        "median_scale": float(scale),
        "absrel_median": float(np.median(absrel[valid])),
        "absrel_p95": float(np.percentile(absrel[valid], 95.0)),
        "absrel_p99": float(np.percentile(absrel[valid], 99.0)),
    }


def aggregate_rows(rows):
    if not rows:
        return {}
    valid_pixels = np.array([row["valid_pixels"] for row in rows], dtype=np.float64)
    removed_pixels = np.array([row["removed_pixels"] for row in rows], dtype=np.float64)

    def mean_finite(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    def median_finite(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")

    return {
        "num_samples": len(rows),
        "total_valid_pixels": int(valid_pixels.sum()),
        "total_removed_pixels": int(removed_pixels.sum()),
        "weighted_removed_valid_ratio": float(removed_pixels.sum() / max(valid_pixels.sum(), 1.0)),
        "mean_removed_valid_ratio": mean_finite("removed_valid_ratio"),
        "median_removed_valid_ratio": median_finite("removed_valid_ratio"),
        "median_threshold_absrel": median_finite("threshold_absrel"),
        "median_absrel_median": median_finite("absrel_median"),
        "median_absrel_p95": median_finite("absrel_p95"),
        "median_absrel_p99": median_finite("absrel_p99"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute DA3-guided training reliability masks for CIL depth labels."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--da3_repo", type=str, default=os.environ.get("DA3_REPO"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.img_size <= 0:
        raise ValueError("--img_size must be positive")
    if not (0.0 < args.threshold_percentile < 100.0):
        raise ValueError("--threshold_percentile must be between 0 and 100")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names, split_metadata = load_train_names(args.split_file, args.max_samples)
    if not train_names:
        raise ValueError("No train names found in split file")

    model = load_da3_model(args.model_dir, args.device, args.da3_repo)

    rows = []
    for index, rgb_name in enumerate(train_names):
        rgb_path = data_root / rgb_name
        depth_path = data_root / rgb_name.replace("_rgb.png", "_depth.npy")
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing RGB file listed in split: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth file for {rgb_path.name}: {depth_path}")

        target_depth = load_resized_depth(depth_path, args.img_size)
        teacher_depth = run_da3_inference(model, rgb_path, args.process_res)
        teacher_depth = resize_array_to_shape(teacher_depth, target_depth.shape, mode="bilinear")

        reliable, metrics = compute_reliability_mask(
            teacher_depth,
            target_depth,
            args.threshold_percentile,
        )

        stem = rgb_name.replace("_rgb.png", "")
        mask_path = output_dir / f"{stem}_teacher_mask.png"
        Image.fromarray((reliable.astype(np.uint8) * 255)).save(mask_path)

        row = {
            "index": index,
            "rgb_name": rgb_name,
            "mask_file": mask_path.name,
            **metrics,
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(train_names)}] {rgb_name}: "
            f"removed {row['removed_pixels']}/{row['valid_pixels']} "
            f"({row['removed_valid_ratio']:.4f})"
        )

    csv_path = output_dir / "per_sample.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "data_root": str(data_root),
        "split_file": str(Path(args.split_file)),
        "output_dir": str(output_dir),
        "img_size": args.img_size,
        "threshold_percentile": args.threshold_percentile,
        "model_dir": args.model_dir,
        "da3_repo": args.da3_repo,
        "process_res": args.process_res,
        "max_samples": args.max_samples,
        "num_train_names_in_split": len(split_metadata["train_names"]),
        "num_masks_saved": len(rows),
        "mask_meaning": "255 means keep for training loss; 0 means ignore",
        "comparison": "DA3 median-scaled to ground-truth depth, then AbsRel thresholded per image",
        "saved_predictions": False,
        "saved_error_maps": False,
    }
    summary = {
        **metadata,
        "aggregate": aggregate_rows(rows),
        "per_sample_csv": csv_path.name,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved DA3 reliability masks to {output_dir}")


if __name__ == "__main__":
    main()
