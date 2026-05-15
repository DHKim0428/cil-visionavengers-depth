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


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(32, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)          # [B,16,H,W]
        e2 = self.enc2(self.pool1(e1))   # [B,32,H/2,W/2]
        
        b = self.bottleneck(self.pool2(e2))  # [B,64,H/4,W/4]
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        out = torch.sigmoid(out)   # output in [0,1]
        return out


class UNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return torch.sigmoid(out)  # Predicts values in [0, 1]


class UNetRefinement(nn.Module):
    """
    Refines a DA-V2 disparity prior using RGB context.

    Input:  RGB image  +  log(DA-V2 disparity prior)  →  4-channel input
    Output: D_final = D_prior · exp(Δ)
            i.e. additive correction Δ in log-disparity space,
            directly aligned with the siRMSE metric.

    The output conv is zero-initialized so the network starts as
    an identity (passes the DA-V2 prior unchanged at epoch 0).
    """
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(4, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, d_prior):
        # x:       [B, 3, H, W]  ImageNet-normalized RGB
        # d_prior: [B, H, W]     DA-V2 disparity prior (positive)
        log_prior = torch.log(d_prior.unsqueeze(1).clamp(min=1e-6))  # [B,1,H,W]
        inp = torch.cat([x, log_prior], dim=1)                        # [B,4,H,W]

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        u3 = F.interpolate(self.up3(b),  size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))   # 64+64 → 64
        u2 = F.interpolate(self.up2(d3), size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))   # 32+32 → 32
        u1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))   # 16+16 → 16

        delta = self.out_conv(d1).squeeze(1)     # [B, H, W]  log-space correction
        return d_prior * torch.exp(delta)        # D_final = D_prior · exp(Δ)