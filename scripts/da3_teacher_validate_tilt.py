import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_INPUT_DIR = (
    f"/work/scratch/{os.environ.get('USER', 'student')}/"
    "cil-visionavengers-depth/debug_tilt_geometry_samples_rgb_unmasked"
)
DEFAULT_OUTPUT_DIR = (
    f"/work/scratch/{os.environ.get('USER', 'student')}/"
    "cil-visionavengers-depth/da3_teacher_validation"
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


def target_path_for_variant(input_dir, prefix, variant):
    if variant in {"geo", "d_geo", "depth_aug"}:
        return input_dir / f"{prefix}_depth_aug.npy"
    if variant == "naive":
        return input_dir / f"{prefix}_depth_naive.npy"
    if variant.startswith("naive_fov"):
        suffix = variant[len("naive_fov") :]
        return input_dir / f"{prefix}_depth_naive_fov{suffix}.npy"
    if variant.startswith("geo_fov"):
        suffix = variant[len("geo_fov") :]
        return input_dir / f"{prefix}_depth_geo_fov{suffix}.npy"
    raise ValueError(
        f"Unknown target variant: {variant}. Use naive, naive_fovXX, geo, depth_aug, or geo_fovXX."
    )


def fov_suffix_for_variant(variant):
    if variant.startswith("geo_fov"):
        return variant[len("geo_fov") :]
    if variant.startswith("naive_fov"):
        return variant[len("naive_fov") :]
    return None


def rgb_path_for_variant(input_dir, prefix, variant):
    suffix = fov_suffix_for_variant(variant)
    if suffix:
        variant_path = input_dir / f"{prefix}_rgb_aug_fov{suffix}.png"
        if variant_path.exists():
            return variant_path
    return input_dir / f"{prefix}_rgb_aug.png"


def mask_path_for_variant(input_dir, prefix, variant):
    suffix = fov_suffix_for_variant(variant)
    if suffix:
        variant_path = input_dir / f"{prefix}_mask_aug_fov{suffix}.png"
        if variant_path.exists():
            return variant_path
    return input_dir / f"{prefix}_mask_aug.png"


def discover_samples(input_dir, target_variants, max_samples=None):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    rgb_paths = sorted(input_dir.glob("*_rgb_aug.png"))
    if max_samples is not None:
        rgb_paths = rgb_paths[:max_samples]
    if not rgb_paths:
        raise FileNotFoundError(f"No *_rgb_aug.png files found in {input_dir}")

    samples = []
    for rgb_path in rgb_paths:
        prefix = rgb_path.name[: -len("_rgb_aug.png")]
        base_mask_path = input_dir / f"{prefix}_mask_aug.png"
        if not base_mask_path.exists():
            raise FileNotFoundError(f"Missing mask file for {rgb_path.name}: {base_mask_path}")
        target_paths = {}
        rgb_paths_by_variant = {}
        mask_paths_by_variant = {}
        for variant in target_variants:
            variant_rgb_path = rgb_path_for_variant(input_dir, prefix, variant)
            if not variant_rgb_path.exists():
                raise FileNotFoundError(
                    f"Missing RGB file for target variant {variant}: {variant_rgb_path}"
                )
            variant_mask_path = mask_path_for_variant(input_dir, prefix, variant)
            if not variant_mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask file for target variant {variant}: {variant_mask_path}"
                )
            target_path = target_path_for_variant(input_dir, prefix, variant)
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Missing target variant {variant} for {rgb_path.name}: {target_path}"
                )
            target_paths[variant] = target_path
            rgb_paths_by_variant[variant] = variant_rgb_path
            mask_paths_by_variant[variant] = variant_mask_path
        samples.append(
            {
                "prefix": prefix,
                "rgb_path": rgb_path,
                "mask_path": base_mask_path,
                "target_paths": target_paths,
                "rgb_paths_by_variant": rgb_paths_by_variant,
                "mask_paths_by_variant": mask_paths_by_variant,
            }
        )
    return samples


def resize_array_to_shape(array, shape):
    if array.shape == shape:
        return array.astype(np.float32, copy=False)

    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


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


def rankdata_average(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def corrcoef_safe(x, y):
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    return corrcoef_safe(rankdata_average(x), rankdata_average(y))


def silog(pred, target):
    log_diff = np.log(np.maximum(pred, EPS)) - np.log(np.maximum(target, EPS))
    value = np.mean(log_diff ** 2) - np.mean(log_diff) ** 2
    return float(np.sqrt(max(value, 0.0)))


def compute_metrics(da3_depth, target_depth, mask):
    valid = (
        (mask > 0)
        & (target_depth > 0)
        & np.isfinite(target_depth)
        & (da3_depth > 0)
        & np.isfinite(da3_depth)
    )
    if int(valid.sum()) < 2:
        return None, valid, None

    teacher = da3_depth[valid].astype(np.float64)
    target = target_depth[valid].astype(np.float64)

    teacher_median = float(np.median(teacher))
    target_median = float(np.median(target))
    if teacher_median <= 0 or target_median <= 0:
        return None, valid, None

    scale = target_median / teacher_median
    teacher_scaled = teacher * scale
    abs_rel = np.abs(teacher_scaled - target) / np.maximum(target, EPS)
    log_teacher = np.log(np.maximum(teacher_scaled, EPS))
    log_target = np.log(np.maximum(target, EPS))

    metrics = {
        "valid_pixels": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "median_scale": float(scale),
        "teacher_raw_median": teacher_median,
        "d_geo_median": target_median,
        "silog_raw": silog(teacher, target),
        "silog_scaled": silog(teacher_scaled, target),
        "absrel_scaled_mean": float(abs_rel.mean()),
        "absrel_scaled_median": float(np.median(abs_rel)),
        "absrel_scaled_p90": float(np.quantile(abs_rel, 0.90)),
        "absrel_scaled_p95": float(np.quantile(abs_rel, 0.95)),
        "absrel_scaled_p99": float(np.quantile(abs_rel, 0.99)),
        "spearman": spearman_corr(teacher_scaled, target),
        "pearson_log": corrcoef_safe(log_teacher, log_target),
    }
    return metrics, valid, scale


def depth_to_color(depth, valid):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    out = np.full((*depth.shape, 3), 40, dtype=np.uint8)
    if not valid.any():
        return out

    values = depth[valid]
    vmin = float(np.percentile(values, 1.0))
    vmax = float(np.percentile(values, 99.0))
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - vmin) / max(vmax - vmin, EPS)
    norm = np.clip(norm, 0.0, 1.0)

    stops = np.array(
        [
            [49, 54, 149],
            [69, 117, 180],
            [116, 173, 209],
            [171, 217, 233],
            [224, 243, 248],
            [254, 224, 144],
            [253, 174, 97],
            [244, 109, 67],
            [215, 48, 39],
            [165, 0, 38],
        ],
        dtype=np.float32,
    )
    scaled = norm * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int64)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    weight = (scaled - lo)[..., None]
    colors = stops[lo] * (1.0 - weight) + stops[hi] * weight
    out[valid] = colors[valid].astype(np.uint8)
    return out


def error_to_color(abs_rel, valid, limit=0.50):
    valid = np.asarray(valid, dtype=bool)
    norm = np.clip(abs_rel / max(limit, EPS), 0.0, 1.0)
    out = np.full((*abs_rel.shape, 3), 40, dtype=np.uint8)
    colors = np.zeros((*abs_rel.shape, 3), dtype=np.float32)
    colors[..., 0] = 255.0
    colors[..., 1] = (1.0 - norm) * 255.0
    colors[..., 2] = (1.0 - norm) * 255.0
    out[valid] = colors[valid].astype(np.uint8)
    return out


def add_label(image, text):
    labeled = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.array(labeled)


def save_visualization(path, rgb_path, target_depth, da3_depth, da3_scaled, valid, error_limit, target_label):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    target_shape = target_depth.shape
    if rgb.shape[:2] != target_shape:
        rgb = np.asarray(Image.fromarray(rgb).resize((target_shape[1], target_shape[0]), Image.BILINEAR))

    abs_rel = np.zeros_like(target_depth, dtype=np.float32)
    abs_rel[valid] = np.abs(da3_scaled[valid] - target_depth[valid]) / np.maximum(target_depth[valid], EPS)

    panels = [
        add_label(rgb, "Tilt RGB"),
        add_label(depth_to_color(target_depth, valid), target_label),
        add_label(depth_to_color(da3_scaled, valid), "DA3 scaled"),
        add_label(error_to_color(abs_rel, valid, limit=error_limit), "AbsRel error"),
    ]
    spacer = np.full((target_shape[0], 6, 3), 220, dtype=np.uint8)
    strip = []
    for panel in panels:
        if strip:
            strip.append(spacer)
        strip.append(panel)
    Image.fromarray(np.concatenate(strip, axis=1)).save(path, quality=92)


def aggregate_rows(rows):
    if not rows:
        return {}

    weights = np.array([row["valid_pixels"] for row in rows], dtype=np.float64)
    total_weight = float(weights.sum())

    def weighted_mean(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            return float("nan")
        return float((values[valid] * weights[valid]).sum() / max(weights[valid].sum(), 1.0))

    def sample_median(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")

    return {
        "num_samples": len(rows),
        "total_valid_pixels": int(total_weight),
        "weighted_silog_scaled": weighted_mean("silog_scaled"),
        "weighted_absrel_scaled_mean": weighted_mean("absrel_scaled_mean"),
        "median_sample_absrel_scaled_median": sample_median("absrel_scaled_median"),
        "median_sample_absrel_scaled_p90": sample_median("absrel_scaled_p90"),
        "median_sample_absrel_scaled_p95": sample_median("absrel_scaled_p95"),
        "median_sample_spearman": sample_median("spearman"),
        "weighted_pearson_log": weighted_mean("pearson_log"),
        "median_valid_ratio": sample_median("valid_ratio"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DA3 teacher depth against geometry tilt D_geo on debug samples."
    )
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--da3_repo", type=str, default=os.environ.get("DA3_REPO"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument(
        "--target_variants",
        nargs="+",
        default=["geo"],
        help=(
            "Target variants to compare: naive, naive_fov50, geo/depth_aug, "
            "geo_fov50, geo_fov60, ..."
        ),
    )
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--error_vis_limit", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    vis_dir = output_dir / "visualizations"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    target_variants = list(dict.fromkeys(args.target_variants))
    samples = discover_samples(input_dir, target_variants, args.max_samples)
    metadata_path = input_dir / "metadata.json"
    source_metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)

    model = load_da3_model(args.model_dir, args.device, args.da3_repo)

    rows = []
    da3_cache = {}
    for index, sample in enumerate(samples):
        for variant, target_path in sample["target_paths"].items():
            rgb_path = sample["rgb_paths_by_variant"][variant]
            mask_path = sample["mask_paths_by_variant"][variant]
            target_depth = np.load(target_path).astype(np.float32)

            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            if mask.shape != target_depth.shape:
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (target_depth.shape[1], target_depth.shape[0]),
                        Image.NEAREST,
                    )
                ) > 127

            rgb_key = str(rgb_path)
            if rgb_key not in da3_cache:
                da3_cache[rgb_key] = run_da3_inference(model, rgb_path, args.process_res)
            da3_depth = resize_array_to_shape(da3_cache[rgb_key], target_depth.shape)

            if args.save_predictions:
                pred_path = pred_dir / f"{sample['prefix']}_{variant}_da3_depth.npy"
                np.save(pred_path, da3_depth.astype(np.float32))

            metrics, valid, scale = compute_metrics(da3_depth, target_depth, mask)
            if metrics is None:
                continue

            da3_scaled = da3_depth * scale
            row = {
                "index": index,
                "prefix": sample["prefix"],
                "target_variant": variant,
                "rgb_aug": rgb_path.name,
                "target_depth": target_path.name,
                "mask_aug": mask_path.name,
                **metrics,
            }

            if args.save_predictions:
                row["da3_depth"] = str(pred_path.relative_to(output_dir))

            if args.save_visualizations:
                vis_path = vis_dir / f"{sample['prefix']}_{variant}_da3_vs_target.jpg"
                save_visualization(
                    vis_path,
                    rgb_path,
                    target_depth,
                    da3_depth,
                    da3_scaled,
                    valid,
                    args.error_vis_limit,
                    variant,
                )
                row["visualization"] = str(vis_path.relative_to(output_dir))

            rows.append(row)

    csv_path = output_dir / "per_sample.csv"
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_dir": args.model_dir,
        "da3_repo": args.da3_repo,
        "process_res": args.process_res,
        "target_variants": target_variants,
        "num_samples_found": len(samples),
        "num_samples_analyzed": len({row["prefix"] for row in rows}),
        "num_rows": len(rows),
        "num_unique_rgb_inferences": len(da3_cache),
        "comparison": "DA3 teacher depth vs selected tilt target variants, using each variant's RGB warp when available",
        "alignment": "per-image median scaling on valid pixels",
        "valid_pixels": "variant mask > 0, target depth > 0, DA3 finite and positive",
        "aggregate": {
            variant: aggregate_rows([row for row in rows if row["target_variant"] == variant])
            for variant in target_variants
        },
        "per_sample_csv": csv_path.name,
        "predictions_saved": bool(args.save_predictions),
        "visualizations_saved": bool(args.save_visualizations),
        "source_metadata": source_metadata,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved DA3 teacher validation to {output_dir}")


if __name__ == "__main__":
    main()
