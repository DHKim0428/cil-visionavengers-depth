import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import DepthAugmentation


DEFAULT_DATA_ROOT = "/cluster/courses/cil/monocular-depth-estimation/train"
DEFAULT_OUTPUT_DIR = f"/work/scratch/{os.environ.get('USER', 'student')}/cil-visionavengers-depth/tilt_depth_change_analysis"


def resize_depth_pair(depth_path, img_size):
    depth = np.load(depth_path).astype(np.float32)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    depth_t = F.interpolate(depth_t, size=(img_size, img_size), mode="nearest").squeeze(0)
    mask_t = (depth_t > 0).float()
    return depth_t, mask_t


def compute_naive_and_geometry_depth(depth, mask, augmentation):
    _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Match DepthAugmentation.__call__: tilt_prob check consumes one random draw
    # before yaw/pitch are sampled, even when tilt_prob is 1.0.
    _ = torch.rand(())
    yaw = augmentation._sample_angle(augmentation.max_yaw_rad)
    pitch = augmentation._sample_angle(augmentation.max_pitch_rad)

    K, K_inv = augmentation._intrinsics(height, width, device, dtype)
    R = augmentation._rotation(yaw, pitch, device, dtype)
    H = K @ R @ K_inv
    H_inv = torch.linalg.inv(H)

    u_src, v_src = augmentation._inverse_warp_coordinates(H_inv, height, width, device, dtype)
    grid = augmentation._grid_sample_coordinates(u_src, v_src, height, width)

    depth_src = F.grid_sample(
        depth.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    mask_src = F.grid_sample(
        mask.unsqueeze(0),
        grid.unsqueeze(0),
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    in_bounds = (u_src >= 0.0) & (u_src <= width - 1) & (v_src >= 0.0) & (v_src <= height - 1)
    alpha = augmentation._depth_scale(R, K_inv, u_src, v_src)
    depth_geo = depth_src * alpha.unsqueeze(0)

    valid = in_bounds
    valid = valid & (alpha > augmentation.eps)
    valid = valid & (mask_src.squeeze(0) > 0.5)
    valid = valid & (depth_src.squeeze(0) > 0.0)
    valid = valid & (depth_geo.squeeze(0) > 0.0)
    valid = valid & torch.isfinite(depth_src.squeeze(0))
    valid = valid & torch.isfinite(depth_geo.squeeze(0))

    return {
        "yaw_deg": float(np.degrees(yaw)),
        "pitch_deg": float(np.degrees(pitch)),
        "depth_src": depth_src.squeeze(0),
        "depth_geo": depth_geo.squeeze(0),
        "alpha": alpha,
        "valid": valid,
    }


def summarize_change(depth_src, depth_geo, alpha, valid):
    if valid.sum().item() == 0:
        return None

    src = depth_src[valid].detach().cpu().numpy().astype(np.float64)
    geo = depth_geo[valid].detach().cpu().numpy().astype(np.float64)
    alpha_valid = alpha[valid].detach().cpu().numpy().astype(np.float64)

    rel = (geo - src) / np.maximum(src, 1e-12)
    abs_rel = np.abs(rel)

    return {
        "valid_pixels": int(valid.sum().item()),
        "valid_ratio": float(valid.float().mean().item()),
        "signed_rel_mean": float(rel.mean()),
        "signed_rel_median": float(np.median(rel)),
        "abs_rel_mean": float(abs_rel.mean()),
        "abs_rel_median": float(np.median(abs_rel)),
        "abs_rel_p90": float(np.quantile(abs_rel, 0.90)),
        "abs_rel_p95": float(np.quantile(abs_rel, 0.95)),
        "abs_rel_p99": float(np.quantile(abs_rel, 0.99)),
        "abs_rel_max": float(abs_rel.max()),
        "alpha_mean": float(alpha_valid.mean()),
        "alpha_median": float(np.median(alpha_valid)),
        "alpha_min": float(alpha_valid.min()),
        "alpha_max": float(alpha_valid.max()),
        "naive_depth_mean": float(src.mean()),
        "geometry_depth_mean": float(geo.mean()),
    }


def signed_change_to_color(rel_change, valid, limit):
    rel = np.asarray(rel_change, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    norm = np.clip(rel / max(limit, 1e-6), -1.0, 1.0)

    out = np.full((*rel.shape, 3), 40, dtype=np.uint8)
    colors = np.zeros((*rel.shape, 3), dtype=np.float32)

    neg = norm < 0
    pos = norm >= 0
    colors[neg, 2] = 255.0
    colors[neg, 1] = (1.0 + norm[neg]) * 255.0
    colors[neg, 0] = (1.0 + norm[neg]) * 255.0
    colors[pos, 0] = 255.0
    colors[pos, 1] = (1.0 - norm[pos]) * 255.0
    colors[pos, 2] = (1.0 - norm[pos]) * 255.0

    out[valid] = colors[valid].astype(np.uint8)
    return out


def save_change_map(output_path, depth_src, depth_geo, valid, limit):
    src = depth_src.detach().cpu().numpy().astype(np.float32)
    geo = depth_geo.detach().cpu().numpy().astype(np.float32)
    valid_np = valid.detach().cpu().numpy().astype(bool)
    rel = np.zeros_like(src, dtype=np.float32)
    rel[valid_np] = (geo[valid_np] - src[valid_np]) / np.maximum(src[valid_np], 1e-12)
    Image.fromarray(signed_change_to_color(rel, valid_np, limit)).save(output_path)


def aggregate_rows(rows):
    if not rows:
        return {}

    valid_pixels = np.array([row["valid_pixels"] for row in rows], dtype=np.float64)
    total_valid = float(valid_pixels.sum())

    def weighted_mean(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        return float((values * valid_pixels).sum() / max(total_valid, 1.0))

    def median(key):
        return float(np.median([row[key] for row in rows]))

    return {
        "num_samples": len(rows),
        "total_valid_pixels": int(total_valid),
        "weighted_signed_rel_mean": weighted_mean("signed_rel_mean"),
        "weighted_abs_rel_mean": weighted_mean("abs_rel_mean"),
        "median_sample_abs_rel_median": median("abs_rel_median"),
        "median_sample_abs_rel_p90": median("abs_rel_p90"),
        "median_sample_abs_rel_p95": median("abs_rel_p95"),
        "max_sample_abs_rel_max": float(max(row["abs_rel_max"] for row in rows)),
        "median_valid_ratio": median("valid_ratio"),
        "median_alpha_min": median("alpha_min"),
        "median_alpha_max": median("alpha_max"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze naive warped depth vs geometry-recomputed tilt depth."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tilt_max_yaw_deg", type=float, default=5.0)
    parser.add_argument("--tilt_max_pitch_deg", type=float, default=5.0)
    parser.add_argument("--tilt_fov_deg", type=float, default=60.0)
    parser.add_argument(
        "--save_maps",
        action="store_true",
        help="Save per-sample signed relative-change heatmaps.",
    )
    parser.add_argument(
        "--map_limit",
        type=float,
        default=0.10,
        help="Relative-change magnitude mapped to full red/blue in heatmaps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(data_root.glob("*_rgb.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png files found in {data_root}")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_samples, len(rgb_files))
    sampled_indices = rng.choice(len(rgb_files), size=sample_count, replace=False)
    sampled_indices.sort()

    torch.manual_seed(args.seed)
    augmentation = DepthAugmentation(
        tilt_mode="geometry",
        tilt_prob=1.0,
        tilt_max_yaw_deg=args.tilt_max_yaw_deg,
        tilt_max_pitch_deg=args.tilt_max_pitch_deg,
        tilt_fov_deg=args.tilt_fov_deg,
    )

    rows = []
    for output_index, dataset_index in enumerate(sampled_indices.tolist()):
        rgb_path = rgb_files[dataset_index]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth map for {rgb_path}: {depth_path}")

        depth, mask = resize_depth_pair(depth_path, args.img_size)
        result = compute_naive_and_geometry_depth(depth, mask, augmentation)
        stats = summarize_change(
            result["depth_src"],
            result["depth_geo"],
            result["alpha"],
            result["valid"],
        )
        if stats is None:
            continue

        stem = rgb_path.name.replace("_rgb.png", "")
        row = {
            "output_index": output_index,
            "dataset_index": int(dataset_index),
            "source_rgb": str(rgb_path),
            "source_depth": str(depth_path),
            "yaw_deg": result["yaw_deg"],
            "pitch_deg": result["pitch_deg"],
            **stats,
        }
        rows.append(row)

        if args.save_maps:
            map_path = output_dir / f"{output_index:03d}_{stem}_naive_vs_geometry_relchange.png"
            save_change_map(
                map_path,
                result["depth_src"],
                result["depth_geo"],
                result["valid"],
                args.map_limit,
            )
            row["relchange_map"] = map_path.name

    csv_path = output_dir / "per_sample.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "num_samples_requested": args.num_samples,
        "num_samples_analyzed": len(rows),
        "img_size": args.img_size,
        "comparison": "naive warped depth D_src vs geometry depth D_geo = D_src * alpha",
        "relative_change": "(D_geo - D_src) / D_src, measured only on valid target pixels",
        "tilt_max_yaw_deg": args.tilt_max_yaw_deg,
        "tilt_max_pitch_deg": args.tilt_max_pitch_deg,
        "tilt_fov_deg": args.tilt_fov_deg,
        "aggregate": aggregate_rows(rows),
        "per_sample_csv": csv_path.name,
        "heatmaps_saved": bool(args.save_maps),
        "heatmap_note": "blue means geometry depth smaller than naive; red means larger.",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
