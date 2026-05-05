import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import DepthAugmentation


DEFAULT_DATA_ROOT = "/cluster/courses/cil/monocular-depth-estimation/train"
DEFAULT_OUTPUT_DIR = f"/work/scratch/{os.environ.get('USER', 'student')}/cil-visionavengers-depth/debug_tilt_geometry_samples"


class RecordingDepthAugmentation(DepthAugmentation):
    """DepthAugmentation wrapper that records the sampled yaw/pitch angles."""

    def __call__(self, image, depth, mask):
        self.sampled_angles_rad = []
        return super().__call__(image, depth, mask)

    def _sample_angle(self, max_abs_rad):
        angle = super()._sample_angle(max_abs_rad)
        self.sampled_angles_rad.append(angle)
        return angle


def resize_pair(rgb_path, depth_path, img_size):
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(depth_path).astype(np.float32)

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.interpolate(
        rgb_t,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    depth_t = F.interpolate(depth_t, size=(img_size, img_size), mode="nearest").squeeze(0)

    valid_mask = (depth_t > 0).float()
    return rgb_t, depth_t, valid_mask


def tensor_rgb_to_uint8(image):
    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image_np * 255.0 + 0.5).astype(np.uint8)


def tensor_mask_to_uint8(mask):
    mask_np = mask.detach().cpu().squeeze(0).numpy() > 0.5
    return (mask_np.astype(np.uint8) * 255)


def depth_to_color(depth, mask):
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    out = np.full((*depth.shape, 3), 40, dtype=np.uint8)
    if not mask.any():
        return out

    valid_depth = depth[mask]
    vmin = float(valid_depth.min())
    vmax = float(valid_depth.max())
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[mask] = (depth[mask] - vmin) / max(vmax - vmin, 1e-6)

    # Lightweight blue-cyan-yellow-red ramp without extra plotting dependencies.
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
    scaled = np.clip(norm, 0.0, 1.0) * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int64)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    weight = (scaled - lo)[..., None]
    colors = stops[lo] * (1.0 - weight) + stops[hi] * weight
    out[mask] = colors[mask].astype(np.uint8)
    return out


def add_label(image, text):
    labeled = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return np.array(labeled)


def make_debug_strip(original_rgb, original_depth, original_mask, augmented_rgb, augmented_depth, augmented_mask):
    original_rgb_np = tensor_rgb_to_uint8(original_rgb)
    augmented_rgb_np = tensor_rgb_to_uint8(augmented_rgb)

    original_depth_np = original_depth.detach().cpu().squeeze(0).numpy()
    original_mask_np = original_mask.detach().cpu().squeeze(0).numpy() > 0.5
    augmented_depth_np = augmented_depth.detach().cpu().squeeze(0).numpy()
    augmented_mask_np = augmented_mask.detach().cpu().squeeze(0).numpy() > 0.5

    original_depth_vis = depth_to_color(original_depth_np, original_mask_np)
    augmented_depth_vis = depth_to_color(augmented_depth_np, augmented_mask_np)
    augmented_mask_vis = np.repeat(tensor_mask_to_uint8(augmented_mask)[:, :, None], 3, axis=2)

    panels = [
        add_label(original_rgb_np, "Original RGB"),
        add_label(original_depth_vis, "Original depth"),
        add_label(augmented_rgb_np, "Tilt RGB"),
        add_label(augmented_depth_vis, "Tilt depth"),
        add_label(augmented_mask_vis, "Valid mask"),
    ]
    spacer = np.full((panels[0].shape[0], 6, 3), 220, dtype=np.uint8)
    parts = []
    for panel in panels:
        if parts:
            parts.append(spacer)
        parts.append(panel)
    return np.concatenate(parts, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Save geometry-consistent tilt augmentation debug samples.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tilt_max_yaw_deg", type=float, default=5.0)
    parser.add_argument("--tilt_max_pitch_deg", type=float, default=5.0)
    parser.add_argument("--tilt_fov_deg", type=float, default=60.0)
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
    augmentation = RecordingDepthAugmentation(
        tilt_mode="geometry",
        tilt_prob=1.0,
        tilt_max_yaw_deg=args.tilt_max_yaw_deg,
        tilt_max_pitch_deg=args.tilt_max_pitch_deg,
        tilt_fov_deg=args.tilt_fov_deg,
    )

    samples = []
    for output_index, dataset_index in enumerate(sampled_indices.tolist()):
        rgb_path = rgb_files[dataset_index]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth map for {rgb_path}: {depth_path}")

        original_rgb, original_depth, original_mask = resize_pair(rgb_path, depth_path, args.img_size)
        augmented_rgb, augmented_depth, augmented_mask = augmentation(
            original_rgb.clone(),
            original_depth.clone(),
            original_mask.clone(),
        )

        stem = rgb_path.name.replace("_rgb.png", "")
        prefix = f"{output_index:03d}_{stem}"
        rgb_aug_path = output_dir / f"{prefix}_rgb_aug.png"
        depth_aug_path = output_dir / f"{prefix}_depth_aug.npy"
        mask_aug_path = output_dir / f"{prefix}_mask_aug.png"
        debug_path = output_dir / f"{prefix}_debug.jpg"

        Image.fromarray(tensor_rgb_to_uint8(augmented_rgb)).save(rgb_aug_path)
        np.save(depth_aug_path, augmented_depth.detach().cpu().squeeze(0).numpy().astype(np.float32))
        Image.fromarray(tensor_mask_to_uint8(augmented_mask)).save(mask_aug_path)

        debug_strip = make_debug_strip(
            original_rgb,
            original_depth,
            original_mask,
            augmented_rgb,
            augmented_depth,
            augmented_mask,
        )
        Image.fromarray(debug_strip).save(debug_path, quality=92)

        sampled_angles = getattr(augmentation, "sampled_angles_rad", [])
        yaw_rad = sampled_angles[0] if len(sampled_angles) > 0 else None
        pitch_rad = sampled_angles[1] if len(sampled_angles) > 1 else None
        valid_pixels = int((augmented_mask > 0.5).sum().item())

        samples.append(
            {
                "output_index": output_index,
                "dataset_index": int(dataset_index),
                "source_rgb": str(rgb_path),
                "source_depth": str(depth_path),
                "rgb_aug": rgb_aug_path.name,
                "depth_aug": depth_aug_path.name,
                "mask_aug": mask_aug_path.name,
                "debug": debug_path.name,
                "yaw_deg": None if yaw_rad is None else float(np.degrees(yaw_rad)),
                "pitch_deg": None if pitch_rad is None else float(np.degrees(pitch_rad)),
                "valid_pixels": valid_pixels,
            }
        )

    metadata = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "num_samples_requested": args.num_samples,
        "num_samples_saved": sample_count,
        "img_size": args.img_size,
        "tilt_mode": "geometry",
        "tilt_prob": 1.0,
        "tilt_max_yaw_deg": args.tilt_max_yaw_deg,
        "tilt_max_pitch_deg": args.tilt_max_pitch_deg,
        "tilt_fov_deg": args.tilt_fov_deg,
        "other_augmentations": "disabled",
        "depth_units": "meters",
        "files": {
            "rgb_aug": "Augmented RGB after geometry tilt.",
            "depth_aug": "Recomputed geometry-consistent z-depth in meters; invalid pixels are zero.",
            "mask_aug": "Valid-pixel mask after inverse warp and geometry depth checks.",
            "debug": "Side-by-side original RGB, original depth, tilt RGB, tilt depth, and valid mask.",
        },
        "samples": samples,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {sample_count} geometry tilt debug samples to {output_dir}")


if __name__ == "__main__":
    main()
