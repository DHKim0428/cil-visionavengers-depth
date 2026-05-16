from __future__ import annotations

import torch

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
EPS = 1e-6


def sirmse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """CIL project-spec scale-invariant RMSE on valid ground-truth depth pixels."""
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    mask = torch.isfinite(gt_depth) & (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH)
    if valid_mask is not None:
        mask &= valid_mask.squeeze().bool()
    if int(mask.sum()) == 0:
        raise ValueError("No valid ground-truth depth pixels for siRMSE")

    pred = torch.nan_to_num(pred_depth[mask], nan=EPS, posinf=1.0 / EPS, neginf=EPS).clamp_min(EPS)
    gt = gt_depth[mask]
    diff = torch.log(pred) - torch.log(gt)
    diff = diff - diff.mean()
    return torch.sqrt(diff.pow(2).mean())
