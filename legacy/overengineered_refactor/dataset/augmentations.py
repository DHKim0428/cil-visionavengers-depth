from __future__ import annotations

import torch
import torch.nn.functional as F


class DepthAugmentation:
    """One small paired RGB/depth augmentation class.

    It intentionally stays boring: flip/crop/rotation affect image+depth+mask,
    color jitter affects RGB only. Geometry-heavy teacher experiments belong in
    legacy/DA3 until we deliberately re-add them.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {"name": "none"}
        paired = cfg.get("paired_spatial", {}) or {}
        rgb_only = cfg.get("rgb_only", {}) or {}

        hflip = paired.get("hflip", {}) or {}
        crop = paired.get("crop", {}) or {}
        rotation = paired.get("rotation", {}) or {}
        color = rgb_only.get("color_jitter", {}) or {}

        self.hflip_prob = float(hflip.get("prob", 0.5)) if hflip.get("enabled", False) else 0.0
        self.crop_scale_min = float(crop.get("scale_min", 1.0)) if crop.get("enabled", False) else 1.0
        self.rotation_deg = float(rotation.get("max_deg", 0.0)) if rotation.get("enabled", False) else 0.0
        self.color_prob = float(color.get("prob", 0.0)) if color.get("enabled", False) else 0.0
        self.brightness = float(color.get("brightness", 0.0))
        self.contrast = float(color.get("contrast", 0.0))
        self.saturation = float(color.get("saturation", 0.0))

    def __call__(self, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hflip_prob and torch.rand(()) < self.hflip_prob:
            image = image.flip(-1)
            depth = depth.flip(-1)
            mask = mask.flip(-1)

        if self.crop_scale_min < 1.0:
            _, h, w = image.shape
            size = int(round(min(h, w) * (self.crop_scale_min + (1.0 - self.crop_scale_min) * torch.rand(()).item())))
            top = torch.randint(0, h - size + 1, ()).item()
            left = torch.randint(0, w - size + 1, ()).item()
            image = image[:, top:top + size, left:left + size]
            depth = depth[:, top:top + size, left:left + size]
            mask = mask[:, top:top + size, left:left + size]
            image = F.interpolate(image[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            depth = F.interpolate(depth[None], size=(h, w), mode="nearest")[0]
            mask = F.interpolate(mask[None], size=(h, w), mode="nearest")[0]

        if self.rotation_deg > 0:
            angle = (torch.rand(()).item() * 2 - 1) * self.rotation_deg * torch.pi / 180.0
            c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=image.dtype, device=image.device)[None]
            grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            depth = F.grid_sample(depth[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
            mask = F.grid_sample(mask[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
            mask = (mask > 0.5).float()
            depth = depth * mask

        if self.color_prob and torch.rand(()) < self.color_prob:
            if self.brightness:
                image = image * (1.0 + (torch.rand(()).item() * 2 - 1) * self.brightness)
            if self.contrast:
                mean = image.mean(dim=(1, 2), keepdim=True)
                image = (image - mean) * (1.0 + (torch.rand(()).item() * 2 - 1) * self.contrast) + mean
            if self.saturation:
                gray = image.mean(dim=0, keepdim=True)
                image = gray + (image - gray) * (1.0 + (torch.rand(()).item() * 2 - 1) * self.saturation)
            image = image.clamp(0.0, 1.0)

        return image, depth, mask
