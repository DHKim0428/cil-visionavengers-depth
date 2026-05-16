from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PREDICTION_KINDS = {"disparity", "depth", "legacy_normalized_depth"}


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    """Canonical U-Net with configurable dense-output semantics.

    ``disparity`` and ``depth`` use a positive softplus head for new canonical
    runs. ``legacy_normalized_depth`` preserves the old sigmoid head so historical
    baseline checkpoints remain loadable/evaluable without changing their meaning.
    """

    def __init__(self, *, prediction_kind: str = "disparity", eps: float = 1e-6) -> None:
        super().__init__()
        if prediction_kind not in PREDICTION_KINDS:
            raise ValueError(
                f"Unsupported U-Net prediction_kind={prediction_kind!r}; expected one of {sorted(PREDICTION_KINDS)}"
            )
        self.prediction_kind = prediction_kind
        self.eps = eps

        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        raw = self.out_conv(d1)
        if self.prediction_kind == "legacy_normalized_depth":
            return torch.sigmoid(raw)
        return F.softplus(raw) + self.eps


def parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    pct = 100.0 * trainable / total if total else 0.0
    return {"total": total, "trainable": trainable, "frozen": frozen, "trainable_pct": pct}
