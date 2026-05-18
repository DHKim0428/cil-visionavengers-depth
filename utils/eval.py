from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.data_loader import IMAGENET_MEAN, IMAGENET_STD, rgb_names, split_names
from utils.calibration import scale_depth_percentile
from utils.loss import MAX_DEPTH, MIN_DEPTH, sirmse


def load_rgb_depth(data_root: str | Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    bgr = cv2.imread(str(root / name))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {root / name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(root / name.replace("_rgb.png", "_depth.npy")).astype(np.float32)
    return rgb, depth


def preprocess_eval_image(image_rgb: np.ndarray, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    model_name = cfg["model"]
    size = int(cfg.get("data", {}).get("image_size", 518 if (str(model_name).startswith("da2_") or model_name == "unet_disp") else 128))
    input_profile = "unet"
    if str(model_name).startswith("da2_"):
        input_profile = "da2"
    elif model_name == "unet_disp" and str((cfg.get("refiner") or {}).get("conditioning", "rgb")) in {"prior", "prior_features"}:
        input_profile = "da2"
    image = image_rgb.astype(np.float32) / 255.0

    if input_profile == "da2":
        h, w = image.shape[:2]
        scale = max(size / h, size / w)
        new_w = int(np.ceil((w * scale) / 14) * 14)
        new_h = int(np.ceil((h * scale) / 14) * 14)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif input_profile == "unet":
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    if input_profile == "da2":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x[None].to(device)


def predict_depth_for_eval(model: torch.nn.Module, image_rgb: np.ndarray, gt_shape: tuple[int, int], cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    # Uses only RGB plus the required output grid shape. No GT depth values enter prediction.
    x = preprocess_eval_image(image_rgb, cfg, device)
    amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=amp):
        raw = model(x)
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0]

    if str(cfg["model"]).startswith("da2_") or cfg["model"] == "unet_disp":
        pred_inv_depth = raw[0].float()
        pred_depth = 1.0 / pred_inv_depth.clamp_min(1e-6)
    else:
        pred_depth = raw[0].float()

    if pred_depth.shape != gt_shape:
        pred_depth = F.interpolate(pred_depth[None, None], size=gt_shape, mode="bilinear", align_corners=False)[0, 0]
    return pred_depth


@torch.no_grad()
def evaluate_names(model: torch.nn.Module, cfg: dict[str, Any], names: list[str], device: torch.device, save_images: int = 0, image_dir: str | Path | None = None, scaling: dict[str, float] | None = None) -> dict[str, Any]:
    model.eval()
    root = Path(cfg["data"]["root"])
    scores = []
    evaluated = []
    image_paths = []
    scales = {}

    if save_images > 0:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

    for name in tqdm(names, desc="eval", leave=False):
        image, gt = load_rgb_depth(root, name)
        gt_t = torch.from_numpy(gt).float().to(device)
        pred = predict_depth_for_eval(model, image, gt_t.shape, cfg, device)
        if scaling is not None:
            pred_np, scale, percentile_value, clipped = scale_depth_percentile(
                pred.detach().cpu().numpy().astype(np.float32),
                scaling["percentile"],
                scaling["target"],
                scaling["max_clip"],
            )
            pred = torch.from_numpy(pred_np).to(device=device, dtype=gt_t.dtype)
            scales[name] = {"scale": scale, "source_percentile": percentile_value, "clipped_pixels": clipped}
        score = sirmse(pred, gt_t)
        scores.append(float(score.item()))
        evaluated.append(name)

        if save_images > 0 and len(image_paths) < save_images:
            valid = np.isfinite(gt) & (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
            rgb_small = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
            pred_np = pred.detach().cpu().numpy()
            gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
            pred_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
            if valid.any():
                lo, hi = np.percentile(gt[valid], [2, 98])
                hi = hi if hi > lo else lo + 1.0
                gt_u8 = np.clip((gt - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
                pred_u8 = np.clip((pred_np - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
                gt_vis = cv2.cvtColor(cv2.applyColorMap(gt_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                pred_vis = cv2.cvtColor(cv2.applyColorMap(pred_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                gt_vis[~valid] = 0
            panel = np.concatenate([rgb_small, gt_vis, pred_vis], axis=1)
            path = Path(image_dir) / f"{Path(name).stem}_sirmse_{float(score.item()):.4f}.png"
            cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            image_paths.append(path)

    summary = {
        "sirmse_mean": float(np.mean(scores)) if scores else float("nan"),
        "sirmse_median": float(np.median(scores)) if scores else float("nan"),
        "sirmse_std": float(np.std(scores)) if scores else float("nan"),
        "samples_selected": len(names),
        "samples_evaluated": len(scores),
    }
    return {
        "summary": summary,
        "scores": scores,
        "evaluated_sample_names": evaluated,
        "image_paths": image_paths,
        "scales": scales,
    }


def validation_names(cfg: dict[str, Any]) -> list[str]:
    all_names = rgb_names(cfg["data"]["root"], cfg["data"].get("max_samples"))
    _, val_names = split_names(all_names, float(cfg["data"].get("val_fraction", 0.05)), int(cfg["data"].get("split_seed", 42)), cfg["data"].get("split_file"))
    return val_names
