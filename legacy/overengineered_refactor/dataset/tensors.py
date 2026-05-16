from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Normalize a CHW RGB tensor in [0, 1] using ImageNet statistics."""
    return (image - IMAGENET_MEAN.to(image.device)) / IMAGENET_STD.to(image.device)


def to_chw_image(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def to_1hw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
