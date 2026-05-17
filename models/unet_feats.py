"""
UNetRefinementWithFeats

UNetRefinement+DA2 multi-scale feature injection
Same architecture as UnetRefinement, but with additional DA2 feature injection (f2, f3, f4) at each decoder scale.

  f4 [B, 384, 19, 19] → proj → add to bottleneck
  f3 [B, 192, 37, 37] → proj → add to dec3
  f2 [B,  96, 74, 74] → proj → add to dec2

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetRefinementWithFeats(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder 
        self.enc1      = DoubleConv(4, 16)
        self.pool1     = nn.MaxPool2d(2)
        self.enc2      = DoubleConv(16, 32)
        self.pool2     = nn.MaxPool2d(2)
        self.enc3      = DoubleConv(32, 64)
        self.pool3     = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)

        # Decoder 
        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # ── DA2 feature projection (zero-init) ───────────────────────────────
        self.proj_f4 = nn.Conv2d(384, 128, kernel_size=1)  # f4 → bottleneck
        self.proj_f3 = nn.Conv2d(192,  64, kernel_size=1)  # f3 → dec3
        self.proj_f2 = nn.Conv2d( 96,  32, kernel_size=1)  # f2 → dec2
        for proj in (self.proj_f4, self.proj_f3, self.proj_f2):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x, d_prior, f2, f3, f4):
        """
        Args:
            x:       [B, 3, H, W]  ImageNet-normalized RGB
            d_prior: [B, H, W]     DA2 disparity prior
            f2:      [B,  96, 74,  74]  DA2 DPT feature (block 6/12)
            f3:      [B, 192, 37,  37]  DA2 DPT feature (block 9/12)
            f4:      [B, 384, 19,  19]  DA2 DPT feature (block 12/12)
        Returns:
            [B, H, W]  refined disparity
        """
        # ── Encoder ───────────────────────────────────────────────────────────
        log_prior = torch.log(d_prior.unsqueeze(1).clamp(min=1e-6))
        inp = torch.cat([x, log_prior], dim=1)          # [B, 4, H, W]

        e1 = self.enc1(inp)                             # [B,  16, 518, 518]
        e2 = self.enc2(self.pool1(e1))                  # [B,  32, 259, 259]
        e3 = self.enc3(self.pool2(e2))                  # [B,  64, 129, 129]
        b  = self.bottleneck(self.pool3(e3))            # [B, 128,  64,  64]

        # f4 Injection to bottleneck
        b = b + F.interpolate(self.proj_f4(f4.float()),
                              size=b.shape[-2:], mode='bilinear', align_corners=False)

        # ── Decoder ───────────────────────────────────────────────────────────
        u3 = F.interpolate(self.up3(b),  size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))     # [B,  64, 129, 129]

        # f3 injection to dec3
        d3 = d3 + F.interpolate(self.proj_f3(f3.float()),
                                size=d3.shape[-2:], mode='bilinear', align_corners=False)

        u2 = F.interpolate(self.up2(d3), size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))     # [B,  32, 259, 259]

        # f2 injection to dec2
        d2 = d2 + F.interpolate(self.proj_f2(f2.float()),
                                size=d2.shape[-2:], mode='bilinear', align_corners=False)

        u1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))     # [B,  16, 518, 518]

        delta = self.out_conv(d1).squeeze(1)            # [B, H, W]
        return d_prior * torch.exp(delta)
