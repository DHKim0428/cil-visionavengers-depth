from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.da2 import build_da2
from models.unet import DoubleConv


class UNetRefiner(nn.Module):
    """Log-disparity residual refiner for a frozen DA2 prior."""

    def __init__(self, base_channels: int = 18) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8
        self.enc1 = DoubleConv(4, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _match_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, prior_disp: torch.Tensor) -> torch.Tensor:
        prior_disp = prior_disp.clamp_min(1e-6)
        log_prior = prior_disp[:, None].log()
        x = torch.cat([image, log_prior], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        u3 = self._match_skip(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self._match_skip(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._match_skip(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        delta = self.out_conv(d1)[:, 0]
        return prior_disp * torch.exp(delta)


class DA2UNetRefine(nn.Module):
    def __init__(self, prior: nn.Module, base_channels: int = 18) -> None:
        super().__init__()
        self.prior = prior
        self.refiner = UNetRefiner(base_channels=base_channels)
        for param in self.prior.parameters():
            param.requires_grad = False
        self.prior.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prior_disp = self.prior(image)
            if prior_disp.ndim == 4 and prior_disp.shape[1] == 1:
                prior_disp = prior_disp[:, 0]
            if prior_disp.shape[-2:] != image.shape[-2:]:
                prior_disp = F.interpolate(prior_disp[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return self.refiner(image, prior_disp)


def build_da2_unet_refine(cfg: dict[str, Any]) -> tuple[nn.Module, Path]:
    refiner_cfg = cfg.get("refiner", {}) or {}
    prior_model = refiner_cfg.get("prior_model", "da2_vits")
    prior_cfg = deepcopy(cfg)
    prior_cfg["model"] = prior_model
    prior_cfg["trainable"] = "frozen"
    prior_cfg.pop("adapter", None)
    prior, ckpt = build_da2(prior_cfg)
    model = DA2UNetRefine(prior, base_channels=int(refiner_cfg.get("base_channels", 18)))
    return model, ckpt
