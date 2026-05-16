from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset.raw_cil import depth_filename_from_rgb, discover_rgb_filenames
from da2_losses import MAX_DEPTH, MIN_DEPTH, disparity_to_depth, sirmse_eval_from_disparity
from training.metrics import resize_to_match, valid_depth_mask

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class EvaluationResult:
    scores: list[float]
    sample_names: list[str]

    @property
    def count(self) -> int:
        return len(self.scores)

    def summary(self) -> dict[str, float | int]:
        if not self.scores:
            return {
                "samples_evaluated": 0,
                "sirmse_mean": float("nan"),
                "sirmse_median": float("nan"),
                "sirmse_std": float("nan"),
                "sirmse_min": float("nan"),
                "sirmse_max": float("nan"),
            }
        arr = np.array(self.scores, dtype=np.float64)
        return {
            "samples_evaluated": int(arr.size),
            "sirmse_mean": float(arr.mean()),
            "sirmse_median": float(np.median(arr)),
            "sirmse_std": float(arr.std()),
            "sirmse_min": float(arr.min()),
            "sirmse_max": float(arr.max()),
        }


def resize_prediction_to_depth(pred_disp: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Resize a DA2 disparity prediction tensor to the GT depth tensor grid."""
    return resize_to_match(pred_disp, depth, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, device: torch.device) -> EvaluationResult:
    """Evaluate transformed-tensor DA2 datasets with an explicit spatial contract."""
    model.eval()
    scores: list[float] = []
    names: list[str] = []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        pred_disp = resize_prediction_to_depth(model(image), depth)
        batch_names = batch.get("name", [""] * depth.shape[0])
        for idx in range(depth.shape[0]):
            score = sirmse_eval_from_disparity(pred_disp[idx], depth[idx], valid_mask[idx])
            if score is not None:
                scores.append(score)
                names.append(str(batch_names[idx]))
    return EvaluationResult(scores=scores, sample_names=names)


def select_filenames(
    data_dir: str | Path,
    *,
    fraction: float | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Select filenames for raw-infer evaluation, matching the old random-subset behavior."""
    filenames = discover_rgb_filenames(data_dir)
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        rng = np.random.default_rng(seed)
        count = max(1, int(len(filenames) * fraction))
        indices = np.sort(rng.choice(len(filenames), size=count, replace=False))
        filenames = [filenames[i] for i in indices]
    if max_samples is not None:
        filenames = filenames[:max_samples]
    return filenames


@torch.no_grad()
def evaluate_raw_infer_native(
    *,
    model,
    data_dir: str | Path,
    filenames: Iterable[str],
    input_size: int,
    device: torch.device,
    vis_dir: str | Path | None = None,
    num_vis: int = 0,
) -> EvaluationResult:
    """Evaluate the old zero-shot raw-image protocol through DA2 infer_image(...)."""
    root = Path(data_dir)
    if vis_dir is not None:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    scores: list[float] = []
    names: list[str] = []
    saved_vis = 0
    for rgb_name in tqdm(list(filenames), desc="Evaluating raw infer"):
        rgb_path = root / rgb_name
        depth_path = root / depth_filename_from_rgb(rgb_name)
        image_bgr = cv2.imread(str(rgb_path))
        if image_bgr is None:
            continue
        gt = np.load(depth_path).astype(np.float32)

        pred_disp = model.infer_image(image_bgr, input_size)
        if pred_disp.shape != gt.shape:
            pred_disp = cv2.resize(pred_disp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred_t = torch.from_numpy(pred_disp).to(device)
        gt_t = torch.from_numpy(gt).to(device)
        valid_t = torch.from_numpy(dataset_depth_mask(gt)).to(device)
        score = sirmse_eval_from_disparity(pred_t, gt_t, valid_t)
        if score is None:
            continue
        scores.append(score)
        names.append(rgb_name)

        if vis_dir is not None and saved_vis < num_vis:
            pred_depth = disparity_to_depth(torch.from_numpy(pred_disp)).cpu().numpy().astype(np.float32)
            save_visualization(
                path=Path(vis_dir) / f"{rgb_name.replace('_rgb.png', '')}_vis.jpg",
                image_bgr=image_bgr,
                gt_depth=gt,
                pred_depth=pred_depth,
                valid_mask=dataset_depth_mask(gt),
                score=score,
                pred_label="Pred depth (DA2 raw infer)",
            )
            saved_vis += 1
    return EvaluationResult(scores=scores, sample_names=names)


def dataset_depth_mask(depth: np.ndarray) -> np.ndarray:
    return valid_depth_mask(torch.from_numpy(depth)).cpu().numpy()


def normalize_for_vis(array: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if valid_mask is not None:
        finite &= valid_mask.astype(bool)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo = np.percentile(array[finite], 2)
    hi = np.percentile(array[finite], 98)
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def depth_to_color(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    if valid_mask is None:
        valid_mask = np.isfinite(depth)
    out = np.full((*depth.shape, 3), 40, dtype=np.uint8)
    if not valid_mask.any():
        return out
    norm = normalize_for_vis(depth, valid_mask)
    color = (cm.get_cmap("Spectral_r")(norm)[..., :3] * 255).astype(np.uint8)
    out[valid_mask.astype(bool)] = color[valid_mask.astype(bool)]
    return out


def label_image_rgb(image_rgb: np.ndarray, text: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_visualization(
    *,
    path: str | Path,
    image_bgr: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    score: float,
    pred_label: str,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != gt_depth.shape:
        image_rgb = cv2.resize(image_rgb, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_AREA)
    gt_vis = depth_to_color(gt_depth, valid_mask)
    pred_vis = depth_to_color(pred_depth, np.isfinite(pred_depth))
    strip = np.ones((gt_depth.shape[0], 8, 3), dtype=np.uint8) * 200
    combined_rgb = np.hstack(
        [
            label_image_rgb(image_rgb, "RGB"),
            strip,
            label_image_rgb(gt_vis, f"GT depth [siRMSE={score:.3f}]"),
            strip,
            label_image_rgb(pred_vis, pred_label),
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
