import argparse
import json
import math
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


def variant_name_for_fov(fov):
    return f"geo_fov{int(round(fov))}"


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


def compute_tilt_debug_outputs(image, depth, mask, augmentation, fov_variants):
    _, height, width = image.shape
    device = image.device
    dtype = image.dtype
    if hasattr(augmentation, "sampled_angles_rad"):
        augmentation.sampled_angles_rad = []

    # Match DepthAugmentation.__call__: the tilt probability check consumes a
    # random draw before yaw/pitch are sampled. Here tilt_prob is always 1.0,
    # but preserving the draw keeps exported samples comparable to the dataset
    # path under the same random seed.
    _ = torch.rand(())
    yaw = augmentation._sample_angle(augmentation.max_yaw_rad)
    pitch = augmentation._sample_angle(augmentation.max_pitch_rad)
    R = augmentation._rotation(yaw, pitch, device, dtype)

    fov_outputs = {}
    primary_name = variant_name_for_fov(math.degrees(augmentation.fov_rad))
    for fov in fov_variants:
        old_fov = augmentation.fov_rad
        augmentation.fov_rad = math.radians(fov)
        try:
            K, K_inv = augmentation._intrinsics(height, width, device, dtype)
        finally:
            augmentation.fov_rad = old_fov

        H = K @ R @ K_inv
        H_inv = torch.linalg.inv(H)
        u_src, v_src = augmentation._inverse_warp_coordinates(
            H_inv, height, width, device, dtype
        )
        grid = augmentation._grid_sample_coordinates(u_src, v_src, height, width)

        image_aug = F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        depth_naive = F.grid_sample(
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
        depth_geo = depth_naive * alpha.unsqueeze(0)
        valid = in_bounds
        valid = valid & (alpha > augmentation.eps)
        valid = valid & (mask_src.squeeze(0) > 0.5)
        valid = valid & (depth_naive.squeeze(0) > 0.0)
        valid = valid & (depth_geo.squeeze(0) > 0.0)
        valid = valid & torch.isfinite(depth_geo.squeeze(0))

        mask_geo = valid.unsqueeze(0).float()
        name = variant_name_for_fov(fov)
        fov_outputs[name] = {
            "image": image_aug,
            "depth_naive": depth_naive,
            "depth": depth_geo * mask_geo,
            "alpha": alpha,
            "mask": mask_geo,
            "fov_deg": float(fov),
        }

    if primary_name not in fov_outputs:
        raise ValueError(f"Primary FOV variant {primary_name} missing from {fov_variants}")

    return {
        "fov_outputs": fov_outputs,
        "primary_variant": primary_name,
        "yaw_deg": float(np.degrees(yaw)),
        "pitch_deg": float(np.degrees(pitch)),
    }


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


def make_debug_strip(
    original_rgb,
    original_depth,
    original_mask,
    augmented_rgb,
    naive_depth,
    geometry_depth,
    augmented_mask,
):
    original_rgb_np = tensor_rgb_to_uint8(original_rgb)
    augmented_rgb_np = tensor_rgb_to_uint8(augmented_rgb)

    original_depth_np = original_depth.detach().cpu().squeeze(0).numpy()
    original_mask_np = original_mask.detach().cpu().squeeze(0).numpy() > 0.5
    naive_depth_np = naive_depth.detach().cpu().squeeze(0).numpy()
    geometry_depth_np = geometry_depth.detach().cpu().squeeze(0).numpy()
    augmented_mask_np = augmented_mask.detach().cpu().squeeze(0).numpy() > 0.5

    original_depth_vis = depth_to_color(original_depth_np, original_mask_np)
    naive_depth_vis = depth_to_color(naive_depth_np, augmented_mask_np)
    geometry_depth_vis = depth_to_color(geometry_depth_np, augmented_mask_np)
    augmented_mask_vis = np.repeat(tensor_mask_to_uint8(augmented_mask)[:, :, None], 3, axis=2)

    panels = [
        add_label(original_rgb_np, "Original RGB"),
        add_label(original_depth_vis, "Original depth"),
        add_label(augmented_rgb_np, "Tilt RGB"),
        add_label(naive_depth_vis, "Naive depth"),
        add_label(geometry_depth_vis, "Geometry depth"),
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
    parser.add_argument("--save_naive", action="store_true", help="Save the naive homography-warped depth map")
    parser.add_argument(
        "--fov_variants",
        type=float,
        nargs="*",
        default=None,
        help="FOV values for additional geometry-depth variants, e.g. 50 60 70",
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
    augmentation = RecordingDepthAugmentation(
        tilt_mode="geometry",
        tilt_prob=1.0,
        tilt_max_yaw_deg=args.tilt_max_yaw_deg,
        tilt_max_pitch_deg=args.tilt_max_pitch_deg,
        tilt_fov_deg=args.tilt_fov_deg,
    )
    fov_variants = args.fov_variants if args.fov_variants else [args.tilt_fov_deg]
    if args.tilt_fov_deg not in fov_variants:
        fov_variants = [args.tilt_fov_deg] + fov_variants
    fov_variants = sorted({float(fov) for fov in fov_variants})
    primary_variant = variant_name_for_fov(args.tilt_fov_deg)
    target_variants = []
    if args.save_naive:
        target_variants.append("naive")
    target_variants.extend(variant_name_for_fov(fov) for fov in fov_variants)

    samples = []
    for output_index, dataset_index in enumerate(sampled_indices.tolist()):
        rgb_path = rgb_files[dataset_index]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth map for {rgb_path}: {depth_path}")

        original_rgb, original_depth, original_mask = resize_pair(rgb_path, depth_path, args.img_size)
        outputs = compute_tilt_debug_outputs(
            original_rgb.clone(),
            original_depth.clone(),
            original_mask.clone(),
            augmentation,
            fov_variants,
        )
        primary_output = outputs["fov_outputs"][primary_variant]
        augmented_rgb = primary_output["image"]
        augmented_depth = primary_output["depth"]
        augmented_mask = primary_output["mask"]
        naive_depth = primary_output["depth_naive"]

        stem = rgb_path.name.replace("_rgb.png", "")
        prefix = f"{output_index:03d}_{stem}"
        rgb_aug_path = output_dir / f"{prefix}_rgb_aug.png"
        depth_aug_path = output_dir / f"{prefix}_depth_aug.npy"
        depth_naive_path = output_dir / f"{prefix}_depth_naive.npy"
        mask_aug_path = output_dir / f"{prefix}_mask_aug.png"
        debug_path = output_dir / f"{prefix}_debug.jpg"

        Image.fromarray(tensor_rgb_to_uint8(augmented_rgb)).save(rgb_aug_path)
        np.save(depth_aug_path, augmented_depth.detach().cpu().squeeze(0).numpy().astype(np.float32))
        if args.save_naive:
            np.save(
                depth_naive_path,
                naive_depth.detach().cpu().squeeze(0).numpy().astype(np.float32),
            )
        fov_files = {}
        for variant_name, variant_output in outputs["fov_outputs"].items():
            fov_value = int(round(variant_output["fov_deg"]))
            is_primary_variant = variant_name == primary_variant
            depth_geo_path = output_dir / f"{prefix}_depth_geo_fov{fov_value}.npy"
            alpha_path = output_dir / f"{prefix}_alpha_fov{fov_value}.npy"
            rgb_variant_path = output_dir / f"{prefix}_rgb_aug_fov{fov_value}.png"
            mask_variant_path = output_dir / f"{prefix}_mask_aug_fov{fov_value}.png"
            naive_variant_path = output_dir / f"{prefix}_depth_naive_fov{fov_value}.npy"
            debug_variant_path = output_dir / f"{prefix}_debug_fov{fov_value}.jpg"

            np.save(
                depth_geo_path,
                variant_output["depth"].detach().cpu().squeeze(0).numpy().astype(np.float32),
            )
            np.save(alpha_path, variant_output["alpha"].detach().cpu().numpy().astype(np.float32))
            if args.save_naive:
                np.save(
                    naive_variant_path,
                    variant_output["depth_naive"]
                    .detach()
                    .cpu()
                    .squeeze(0)
                    .numpy()
                    .astype(np.float32),
                )
            if not is_primary_variant:
                Image.fromarray(tensor_rgb_to_uint8(variant_output["image"])).save(rgb_variant_path)
                Image.fromarray(tensor_mask_to_uint8(variant_output["mask"])).save(mask_variant_path)
                debug_variant_strip = make_debug_strip(
                    original_rgb,
                    original_depth,
                    original_mask,
                    variant_output["image"],
                    variant_output["depth_naive"],
                    variant_output["depth"],
                    variant_output["mask"],
                )
                Image.fromarray(debug_variant_strip).save(debug_variant_path, quality=92)
            fov_files[variant_name] = {
                "rgb": rgb_aug_path.name if is_primary_variant else rgb_variant_path.name,
                "mask": mask_aug_path.name if is_primary_variant else mask_variant_path.name,
                "depth": depth_geo_path.name,
                "alpha": alpha_path.name,
                "depth_naive": naive_variant_path.name if args.save_naive else None,
                "depth_naive_primary_alias": depth_naive_path.name
                if args.save_naive and is_primary_variant
                else None,
                "debug": debug_path.name if is_primary_variant else debug_variant_path.name,
                "fov_deg": variant_output["fov_deg"],
            }
        Image.fromarray(tensor_mask_to_uint8(augmented_mask)).save(mask_aug_path)

        debug_strip = make_debug_strip(
            original_rgb,
            original_depth,
            original_mask,
            augmented_rgb,
            naive_depth,
            augmented_depth,
            augmented_mask,
        )
        Image.fromarray(debug_strip).save(debug_path, quality=92)

        valid_pixels = int((augmented_mask > 0.5).sum().item())

        sample_record = {
            "output_index": output_index,
            "dataset_index": int(dataset_index),
            "source_rgb": str(rgb_path),
            "source_depth": str(depth_path),
            "rgb_aug": rgb_aug_path.name,
            "depth_aug": depth_aug_path.name,
            "mask_aug": mask_aug_path.name,
            "debug": debug_path.name,
            "yaw_deg": outputs["yaw_deg"],
            "pitch_deg": outputs["pitch_deg"],
            "valid_pixels": valid_pixels,
            "primary_target": primary_variant,
            "fov_variants": fov_files,
        }
        if args.save_naive:
            sample_record["depth_naive"] = depth_naive_path.name
        samples.append(sample_record)

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
        "fov_variants": fov_variants,
        "target_variants": target_variants,
        "primary_target": primary_variant,
        "strict_fov_rgb_warp": True,
        "other_augmentations": "disabled",
        "depth_units": "meters",
        "files": {
            "rgb_aug": "Augmented RGB after geometry tilt using the primary FOV.",
            "rgb_aug_fov*": "Augmented RGB after geometry tilt recomputed with the listed FOV.",
            "depth_aug": "Recomputed geometry-consistent z-depth in meters; invalid pixels are zero.",
            "depth_naive": "Naive homography-warped depth from the primary FOV warp.",
            "depth_naive_fov*": "Naive homography-warped depth recomputed with the listed FOV warp.",
            "depth_geo_fov*": "Geometry depth recomputed with the listed FOV warp.",
            "alpha_fov*": "Pixel-dependent geometry depth scale for the listed FOV.",
            "mask_aug": "Valid-pixel mask after inverse warp and geometry depth checks for the primary FOV.",
            "mask_aug_fov*": "Valid-pixel mask after inverse warp and geometry depth checks for the listed FOV.",
            "debug": "Side-by-side original RGB, original depth, tilt RGB, naive depth, geometry depth, and valid mask.",
        },
        "samples": samples,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {sample_count} geometry tilt debug samples to {output_dir}")


if __name__ == "__main__":
    main()
