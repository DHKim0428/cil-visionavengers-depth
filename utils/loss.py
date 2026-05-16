from __future__ import annotations

import torch

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
EPS = 1e-6


def sirmse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor | None = None, min_pixels: int = 10) -> torch.Tensor | None:
    """Scale-invariant RMSE on GT-valid pixels.

    Keep this intentionally small: callers decide whether their model predicts
    depth or disparity and pass depth here.
    """
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze().bool()

    pred_depth = torch.nan_to_num(pred_depth, nan=MIN_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH).clamp(MIN_DEPTH, MAX_DEPTH)
    mask = torch.isfinite(gt_depth) & (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH)
    if valid_mask is not None:
        mask &= valid_mask
    if int(mask.sum()) < min_pixels:
        return None

    diff = torch.log(pred_depth[mask].clamp_min(EPS)) - torch.log(gt_depth[mask].clamp_min(EPS))
    return torch.sqrt(torch.clamp(diff.pow(2).mean() - diff.mean().pow(2), min=1e-8))
