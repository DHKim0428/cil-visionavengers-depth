from __future__ import annotations

import torch

from training.metrics import (
    MAX_DEPTH,
    MIN_DEPTH,
    disparity_to_depth,
    sirmse_from_disparity,
    sirmse_loss_from_disparity_batch,
    squeeze_channel,
)


def squeeze_prediction(pred_disp: torch.Tensor) -> torch.Tensor:
    return squeeze_channel(pred_disp)


def sirmse_loss_from_disparity(
    pred_disp: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Batch siRMSE loss for DA2 disparity predictions.

    The supervision mask is based on GT validity only.  Non-positive disparity
    predictions are clamped into a finite depth instead of being removed from
    the loss mask, matching the canonical evaluation policy.
    """
    return sirmse_loss_from_disparity_batch(
        pred_disp,
        gt_depth,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        eps=eps,
    )


@torch.no_grad()
def sirmse_eval_from_disparity(
    pred_disp: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    eps: float = 1e-6,
) -> float | None:
    """Single-sample siRMSE metric for DA2 validation/evaluation."""
    score = sirmse_from_disparity(
        pred_disp,
        gt_depth,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        eps=eps,
    )
    return None if score is None else float(score.item())


__all__ = [
    "MIN_DEPTH",
    "MAX_DEPTH",
    "disparity_to_depth",
    "squeeze_prediction",
    "sirmse_loss_from_disparity",
    "sirmse_eval_from_disparity",
]
