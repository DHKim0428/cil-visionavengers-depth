import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_DATA_ROOT = "/cluster/courses/cil/monocular-depth-estimation/train"
DEFAULT_OUTPUT_DIR = (
    f"/work/scratch/{os.environ.get('USER', 'student')}/"
    "cil-visionavengers-depth/geocalib_fov"
)
EPS = 1e-6


def add_geocalib_repo_to_path(geocalib_repo):
    if not geocalib_repo:
        return
    repo = Path(geocalib_repo).expanduser().resolve()
    if repo.exists():
        sys.path.insert(0, str(repo))


def import_geocalib(geocalib_repo):
    add_geocalib_repo_to_path(geocalib_repo)
    try:
        from geocalib import GeoCalib
    except ImportError as exc:
        raise ImportError(
            "Could not import geocalib. Install GeoCalib in the active environment, "
            "or pass --geocalib_repo /path/to/GeoCalib, or set "
            "GEOCALIB_REPO=/path/to/GeoCalib."
        ) from exc
    return GeoCalib


def tensor_scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def extract_camera_values(camera, image_width, image_height):
    fx = fy = float("nan")
    width = float(image_width)
    height = float(image_height)

    if hasattr(camera, "f"):
        f = camera.f.detach().cpu().reshape(-1)
        if f.numel() >= 2:
            fx = float(f[0].item())
            fy = float(f[1].item())
        elif f.numel() == 1:
            fx = fy = float(f[0].item())

    if hasattr(camera, "size"):
        size = camera.size.detach().cpu().reshape(-1)
        if size.numel() >= 2:
            width = float(size[0].item())
            height = float(size[1].item())

    if hasattr(camera, "hfov"):
        hfov_deg = math.degrees(tensor_scalar(camera.hfov))
    else:
        hfov_deg = math.degrees(2.0 * math.atan(width / max(2.0 * fx, EPS)))

    if hasattr(camera, "vfov"):
        vfov_deg = math.degrees(tensor_scalar(camera.vfov))
    else:
        vfov_deg = math.degrees(2.0 * math.atan(height / max(2.0 * fy, EPS)))

    return width, height, fx, fy, hfov_deg, vfov_deg


def finite_or_empty(value):
    return value if np.isfinite(value) else ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute per-image GeoCalib horizontal FOV for adaptive geometry tilt diagnostics."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geocalib_repo", type=str, default=os.environ.get("GEOCALIB_REPO"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--clip_min_deg", type=float, default=35.0)
    parser.add_argument("--clip_max_deg", type=float, default=95.0)
    parser.add_argument("--weights", type=str, default="pinhole")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clip_min_deg >= args.clip_max_deg:
        raise ValueError("--clip_min_deg must be smaller than --clip_max_deg")

    rgb_files = sorted(data_root.glob("*_rgb.png"))
    if args.max_samples is not None:
        rgb_files = rgb_files[: args.max_samples]
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png files found in {data_root}")

    GeoCalib = import_geocalib(args.geocalib_repo)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = GeoCalib(weights=args.weights).to(device).eval()

    rows = []
    for index, rgb_path in enumerate(rgb_files):
        row = {
            "name": rgb_path.name,
            "width": "",
            "height": "",
            "fx": "",
            "fy": "",
            "hfov_deg_raw": "",
            "vfov_deg_raw": "",
            "tilt_fov_deg": "",
            "clipped": "",
            "status": "error",
            "error": "",
        }
        try:
            with Image.open(rgb_path) as image:
                image_width, image_height = image.size
            image_t = model.load_image(rgb_path).to(device)
            result = model.calibrate(image_t, camera_model="pinhole")
            camera = result["camera"]
            width, height, fx, fy, hfov_deg, vfov_deg = extract_camera_values(
                camera, image_width, image_height
            )
            if not np.isfinite(hfov_deg):
                raise ValueError(f"Non-finite horizontal FOV for {rgb_path.name}: {hfov_deg}")

            tilt_fov_deg = float(np.clip(hfov_deg, args.clip_min_deg, args.clip_max_deg))
            clipped = abs(tilt_fov_deg - hfov_deg) > 1e-6
            row.update(
                {
                    "width": finite_or_empty(width),
                    "height": finite_or_empty(height),
                    "fx": finite_or_empty(fx),
                    "fy": finite_or_empty(fy),
                    "hfov_deg_raw": finite_or_empty(hfov_deg),
                    "vfov_deg_raw": finite_or_empty(vfov_deg),
                    "tilt_fov_deg": tilt_fov_deg,
                    "clipped": int(clipped),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:  # keep long precomputes resumable/auditable
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"[{index + 1}/{len(rgb_files)}] {rgb_path.name}: {row['status']} {row['tilt_fov_deg']}")

    csv_path = output_dir / "geocalib_fov.csv"
    fieldnames = [
        "name",
        "width",
        "height",
        "fx",
        "fy",
        "hfov_deg_raw",
        "vfov_deg_raw",
        "tilt_fov_deg",
        "clipped",
        "status",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    fovs = np.array([float(row["tilt_fov_deg"]) for row in ok_rows], dtype=np.float64)
    raw_fovs = np.array([float(row["hfov_deg_raw"]) for row in ok_rows], dtype=np.float64)
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "geocalib_repo": args.geocalib_repo,
        "weights": args.weights,
        "device": str(device),
        "clip_min_deg": args.clip_min_deg,
        "clip_max_deg": args.clip_max_deg,
        "num_images": len(rows),
        "num_ok": len(ok_rows),
        "num_error": len(rows) - len(ok_rows),
        "num_clipped": int(sum(int(row["clipped"] or 0) for row in ok_rows)),
        "fov_column_for_tilt": "tilt_fov_deg",
        "fov_convention": "horizontal field of view in degrees, clipped per image",
        "csv": csv_path.name,
    }
    if fovs.size:
        summary["tilt_fov_deg_stats"] = {
            "min": float(np.min(fovs)),
            "p05": float(np.quantile(fovs, 0.05)),
            "median": float(np.median(fovs)),
            "mean": float(np.mean(fovs)),
            "p95": float(np.quantile(fovs, 0.95)),
            "max": float(np.max(fovs)),
        }
        summary["hfov_deg_raw_stats"] = {
            "min": float(np.min(raw_fovs)),
            "median": float(np.median(raw_fovs)),
            "mean": float(np.mean(raw_fovs)),
            "max": float(np.max(raw_fovs)),
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved GeoCalib FOV table to {csv_path}")


if __name__ == "__main__":
    main()
