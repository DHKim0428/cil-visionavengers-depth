from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.da2 import build_da2
from models.unet import DoubleConv


class UNetDispCore(nn.Module):
    """U-Net family that predicts disparity directly or refines a disparity prior."""

    def __init__(self, in_channels: int, base_channels: int = 18, residual: bool = True, feature_channels: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8
        self.residual = residual
        self.enc1 = DoubleConv(int(in_channels), c1)
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
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

        self.feature_projs = None
        if feature_channels is not None:
            if len(feature_channels) != 3:
                raise ValueError(f"feature_channels must describe f2/f3/f4, got {feature_channels!r}")
            f2_ch, f3_ch, f4_ch = (int(ch) for ch in feature_channels)
            self.feature_projs = nn.ModuleDict({
                "f2": nn.Conv2d(f2_ch, c2, kernel_size=1),
                "f3": nn.Conv2d(f3_ch, c3, kernel_size=1),
                "f4": nn.Conv2d(f4_ch, c4, kernel_size=1),
            })
            for proj in self.feature_projs.values():
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

    @staticmethod
    def _match_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, prior_disp: torch.Tensor | None = None, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        if self.residual:
            if prior_disp is None:
                raise ValueError("Residual disparity refinement requires a prior_disp tensor")
            prior_disp = prior_disp.clamp_min(1e-6)
            x = torch.cat([image, prior_disp[:, None].log()], dim=1)
        else:
            x = image

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        if self.feature_projs is not None:
            if features is None:
                raise ValueError("Feature-conditioned disparity refinement requires f2/f3/f4 tensors")
            f2, f3, f4 = features
            b = b + F.interpolate(self.feature_projs["f4"](f4.float()), size=b.shape[-2:], mode="bilinear", align_corners=False)

        u3 = self._match_skip(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        if self.feature_projs is not None:
            d3 = d3 + F.interpolate(self.feature_projs["f3"](f3.float()), size=d3.shape[-2:], mode="bilinear", align_corners=False)

        u2 = self._match_skip(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        if self.feature_projs is not None:
            d2 = d2 + F.interpolate(self.feature_projs["f2"](f2.float()), size=d2.shape[-2:], mode="bilinear", align_corners=False)

        u1 = self._match_skip(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        delta = self.out_conv(d1)[:, 0]
        if self.residual:
            return prior_disp * torch.exp(delta)
        return torch.sigmoid(delta).clamp_min(1e-6)


class UNetRefiner(UNetDispCore):
    """Log-disparity residual refiner for a frozen DA2 prior."""

    def __init__(self, base_channels: int = 18) -> None:
        base_channels = int(base_channels)
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        super().__init__(in_channels=4, base_channels=base_channels, residual=True)


class UNetDisp(nn.Module):
    """RGB-only U-Net whose sigmoid output is interpreted as disparity."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.refiner = UNetDispCore(in_channels=3, base_channels=base_channels, residual=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"UNetDisp expects RGB BCHW input, got shape {tuple(image.shape)}")
        disp = self.refiner(image)
        if disp.ndim != 3:
            raise RuntimeError(f"UNetDisp should return BHW disparity, got shape {tuple(disp.shape)}")
        return disp


class DA2UNetRefine(nn.Module):
    def __init__(self, prior: nn.Module, base_channels: int = 18, use_features: bool = False) -> None:
        super().__init__()
        self.prior = prior
        self.use_features = bool(use_features)
        feature_channels = self._feature_channels() if self.use_features else None
        self.refiner = UNetDispCore(in_channels=4, base_channels=base_channels, residual=True, feature_channels=feature_channels)
        for param in self.prior.parameters():
            param.requires_grad = False
        self.prior.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.prior.eval()
        return self

    def _feature_channels(self) -> tuple[int, int, int]:
        projects = getattr(getattr(self.prior, "depth_head", None), "projects", None)
        if projects is None or len(projects) < 4:
            raise ValueError("DA2 feature refinement expects a DPT head with four projection layers")
        return tuple(int(projects[i].out_channels) for i in (1, 2, 3))

    def _prior_with_features(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        depth_head = self.prior.depth_head
        if getattr(depth_head, "use_clstoken", False):
            raise ValueError("DA2 feature refinement currently supports the default use_clstoken=False path")
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
        out_features = self.prior.pretrained.get_intermediate_layers(
            image,
            self.prior.intermediate_layer_idx[self.prior.encoder],
            return_class_token=True,
        )
        projected = []
        for idx, feature in enumerate(out_features):
            tokens = feature[0]
            feature_map = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            feature_map = depth_head.projects[idx](feature_map)
            projected.append(depth_head.resize_layers[idx](feature_map))
        prior_disp = F.relu(depth_head(out_features, patch_h, patch_w)).squeeze(1)
        return prior_disp, (projected[1], projected[2], projected[3])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.use_features:
                prior_disp, features = self._prior_with_features(image)
            else:
                prior_disp = self.prior(image)
                features = None
            if prior_disp.ndim == 4 and prior_disp.shape[1] == 1:
                prior_disp = prior_disp[:, 0]
            if prior_disp.shape[-2:] != image.shape[-2:]:
                prior_disp = F.interpolate(prior_disp[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return self.refiner(image, prior_disp, features)


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


def build_unet_disp(cfg: dict[str, Any]) -> tuple[nn.Module, Path | None]:
    refiner_cfg = cfg.get("refiner", {}) or {}
    conditioning = str(refiner_cfg.get("conditioning", "rgb"))
    if conditioning == "rgb":
        return UNetDisp(base_channels=int(refiner_cfg.get("base_channels", 32))), None
    if conditioning in {"prior", "prior_features"}:
        prior_model = refiner_cfg.get("prior_model", "da2_vits")
        prior_cfg = deepcopy(cfg)
        prior_cfg["model"] = prior_model
        prior_cfg["trainable"] = "frozen"
        prior_cfg.pop("adapter", None)
        prior, ckpt = build_da2(prior_cfg)
        model = DA2UNetRefine(
            prior,
            base_channels=int(refiner_cfg.get("base_channels", 16)),
            use_features=conditioning == "prior_features",
        )
        return model, ckpt
    raise ValueError(f"Unknown unet_disp conditioning: {conditioning!r}")
