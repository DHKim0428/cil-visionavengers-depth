from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
DEFAULT_EPS = 1e-6


def squeeze_channel(pred: torch.Tensor) -> torch.Tensor:
    """Remove a singleton channel dimension from dense depth/disparity tensors."""
    if pred.ndim == 4 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


def resize_to_match(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize a dense prediction to the target spatial grid.

    Accepts `[H, W]`, `[B, H, W]`, or `[B, 1, H, W]` predictions and preserves
    the non-channel batch shape of the input.
    """
    pred = squeeze_channel(pred)
    original_ndim = pred.ndim
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    if pred.ndim == 2:
        pred_4d = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred_4d = pred.unsqueeze(1)
    else:
        raise ValueError(f"Expected prediction with 2 or 3 spatial dims after squeeze, got shape {tuple(pred.shape)}")
    resized = F.interpolate(
        pred_4d,
        size=target.shape[-2:],
        mode=mode,
        align_corners=align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
    )[:, 0]
    if original_ndim == 2:
        return resized[0]
    return resized


def valid_depth_mask(
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
) -> torch.Tensor:
    """Ground-truth availability mask used by the canonical evaluator."""
    mask = torch.isfinite(gt_depth) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    return mask


def sanitize_direct_depth(
    pred_depth: torch.Tensor,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
) -> torch.Tensor:
    """Map direct-depth predictions into the valid evaluation depth range."""
    pred_depth = torch.nan_to_num(pred_depth, nan=min_depth, posinf=max_depth, neginf=min_depth)
    return pred_depth.clamp(min_depth, max_depth)


def disparity_to_depth(
    pred_disp: torch.Tensor,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Convert DA2 relative-disparity output to sanitized pseudo-depth."""
    pred_disp = squeeze_channel(pred_disp)
    pred_disp = torch.nan_to_num(pred_disp, nan=eps, posinf=1.0 / min_depth, neginf=eps)
    pred_disp = pred_disp.clamp(min=eps)
    pred_depth = 1.0 / pred_disp
    return sanitize_direct_depth(pred_depth, min_depth=min_depth, max_depth=max_depth)


def sirmse_from_depth(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = DEFAULT_EPS,
    min_pixels: int = 10,
) -> torch.Tensor | None:
    """Single-sample siRMSE using GT-valid pixels only.

    Model failures such as zero/negative predictions should be handled before
    this function by sanitizing the prediction, not by shrinking the metric mask.
    """
    pred_depth = squeeze_channel(pred_depth).squeeze()
    gt_depth = gt_depth.squeeze()
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
    pred_depth = sanitize_direct_depth(pred_depth, min_depth=min_depth, max_depth=max_depth)
    mask = valid_depth_mask(gt_depth, valid_mask, min_depth=min_depth, max_depth=max_depth)
    if int(mask.sum().item()) < min_pixels:
        return None
    diff = torch.log(pred_depth[mask].clamp(min=eps)) - torch.log(gt_depth[mask].clamp(min=eps))
    return torch.sqrt(torch.clamp(diff.pow(2).mean() - diff.mean().pow(2), min=1e-8))


def sirmse_loss_from_depth_batch(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = DEFAULT_EPS,
    min_pixels: int = 10,
) -> torch.Tensor:
    """Batch-mean differentiable siRMSE loss for direct-depth predictions."""
    pred_depth = squeeze_channel(pred_depth)
    losses: list[torch.Tensor] = []
    for batch_idx in range(pred_depth.shape[0]):
        sample_valid = valid_mask[batch_idx] if valid_mask is not None else None
        loss = sirmse_from_depth(
            pred_depth[batch_idx],
            gt_depth[batch_idx],
            sample_valid,
            min_depth=min_depth,
            max_depth=max_depth,
            eps=eps,
            min_pixels=min_pixels,
        )
        if loss is not None:
            losses.append(loss)
    if not losses:
        return pred_depth.sum() * 0.0
    return torch.stack(losses).mean()


def sirmse_from_disparity(
    pred_disp: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = DEFAULT_EPS,
    min_pixels: int = 10,
) -> torch.Tensor | None:
    """Single-sample siRMSE for DA2 disparity outputs with clamp-based sanitization."""
    pred_depth = disparity_to_depth(pred_disp, min_depth=min_depth, max_depth=max_depth, eps=eps)
    return sirmse_from_depth(
        pred_depth,
        gt_depth,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        eps=eps,
        min_pixels=min_pixels,
    )


def sirmse_loss_from_disparity_batch(
    pred_disp: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = DEFAULT_EPS,
    min_pixels: int = 10,
) -> torch.Tensor:
    """Batch-mean siRMSE loss for DA2 disparity outputs."""
    pred_depth = disparity_to_depth(pred_disp, min_depth=min_depth, max_depth=max_depth, eps=eps)
    return sirmse_loss_from_depth_batch(
        pred_depth,
        gt_depth,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        eps=eps,
        min_pixels=min_pixels,
    )
