from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class DepthAugmentation:
    def __init__(self, cfg: dict[str, Any] | None) -> None:
        cfg = cfg or {"name": "none"}
        paired = cfg.get("paired_spatial", {}) or {}
        rgb_only = cfg.get("rgb_only", {}) or {}
        hflip = paired.get("hflip", {}) or {}
        crop = paired.get("crop", {}) or {}
        rotation = paired.get("rotation", {}) or {}
        tilt = paired.get("tilt", {}) or {}
        color = rgb_only.get("color_jitter", {}) or {}

        self.hflip_prob = float(hflip.get("prob", 0.5)) if hflip.get("enabled", False) else 0.0
        self.crop_scale_min = float(crop.get("scale_min", 1.0)) if crop.get("enabled", False) else 1.0
        self.rotation_deg = float(rotation.get("max_deg", 0.0)) if rotation.get("enabled", False) else 0.0

        self.tilt_mode = str(tilt.get("mode", "none"))
        if self.tilt_mode not in {"none", "naive", "geometry"}:
            raise ValueError(f"Unsupported tilt mode: {self.tilt_mode}")
        self.tilt_prob = float(tilt.get("prob", 0.0)) if self.tilt_mode != "none" else 0.0
        self.max_yaw_rad = float(tilt.get("max_yaw_deg", 0.0)) * np.pi / 180.0
        self.max_pitch_rad = float(tilt.get("max_pitch_deg", 0.0)) * np.pi / 180.0
        self.tilt_fov_rad = float(tilt.get("fov_deg", 60.0)) * np.pi / 180.0
        self.eps = 1e-6

        self.color_prob = float(color.get("prob", 0.0)) if color.get("enabled", False) else 0.0
        self.brightness = float(color.get("brightness", 0.0))
        self.contrast = float(color.get("contrast", 0.0))
        self.saturation = float(color.get("saturation", 0.0))

    def __call__(self, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hflip_prob and torch.rand(()) < self.hflip_prob:
            image, depth, mask = image.flip(-1), depth.flip(-1), mask.flip(-1)

        if self.crop_scale_min < 1.0:
            _, h, w = image.shape
            crop = int(round(min(h, w) * (self.crop_scale_min + (1.0 - self.crop_scale_min) * torch.rand(()).item())))
            top = torch.randint(0, h - crop + 1, ()).item()
            left = torch.randint(0, w - crop + 1, ()).item()
            image = F.interpolate(image[:, top:top + crop, left:left + crop][None], size=(h, w), mode="bilinear", align_corners=False)[0]
            depth = F.interpolate(depth[:, top:top + crop, left:left + crop][None], size=(h, w), mode="nearest")[0]
            mask = F.interpolate(mask[:, top:top + crop, left:left + crop][None], size=(h, w), mode="nearest")[0]

        if self.rotation_deg > 0:
            angle = torch.tensor((torch.rand(()).item() * 2 - 1) * self.rotation_deg * np.pi / 180.0, dtype=image.dtype, device=image.device)
            c, s = torch.cos(angle), torch.sin(angle)
            theta = torch.stack([torch.stack([c, -s, torch.zeros_like(c)]), torch.stack([s, c, torch.zeros_like(c)])])[None]
            grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            depth = F.grid_sample(depth[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
            mask = (F.grid_sample(mask[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0] > 0.5).float()
            depth = depth * mask

        if self.tilt_prob and torch.rand(()) < self.tilt_prob:
            image, depth, mask = self._tilt(image, depth, mask)

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

    def _tilt(self, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        device, dtype = image.device, image.dtype
        yaw = self._sample_angle(self.max_yaw_rad, device, dtype)
        pitch = self._sample_angle(self.max_pitch_rad, device, dtype)
        K, K_inv = self._intrinsics(height, width, device, dtype)
        R = self._rotation(yaw, pitch, device, dtype)
        H_inv = torch.linalg.inv(K @ R @ K_inv)
        u_src, v_src = self._inverse_warp_coordinates(H_inv, height, width, device, dtype)
        grid = self._grid_sample_coordinates(u_src, v_src, height, width)

        image_w = F.grid_sample(image[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        depth_w = F.grid_sample(depth[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        mask_w = F.grid_sample(mask[None], grid[None], mode="nearest", padding_mode="zeros", align_corners=True)[0] > 0.5
        in_bounds = (u_src >= 0.0) & (u_src <= width - 1) & (v_src >= 0.0) & (v_src <= height - 1)

        valid = mask_w[0] & in_bounds & (depth_w[0] > 0.0) & torch.isfinite(depth_w[0])
        if self.tilt_mode == "geometry":
            alpha = self._depth_scale(R, K_inv, u_src, v_src)
            depth_w = depth_w * alpha.clamp_min(0.0)[None]
            valid = valid & (alpha > self.eps) & (depth_w[0] > 0.0) & torch.isfinite(depth_w[0])

        mask_out = valid[None].float()
        return image_w * mask_out, depth_w * mask_out, mask_out

    def _sample_angle(self, max_abs_rad: float, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = (torch.rand((), device=device).item() * 2.0 - 1.0) * max_abs_rad
        return torch.tensor(value, device=device, dtype=dtype or torch.float32)

    def _intrinsics(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        focal = 0.5 * float(width) / np.tan(max(self.tilt_fov_rad, self.eps) * 0.5)
        cx = (float(width) - 1.0) * 0.5
        cy = (float(height) - 1.0) * 0.5
        K = torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        return K, torch.linalg.inv(K)

    @staticmethod
    def _rotation(yaw: torch.Tensor, pitch: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zero = torch.zeros((), device=device, dtype=dtype)
        one = torch.ones((), device=device, dtype=dtype)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        Ry = torch.stack([torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])])
        Rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, cp, -sp]), torch.stack([zero, sp, cp])])
        return Ry @ Rx

    @staticmethod
    def _inverse_warp_coordinates(H_inv: torch.Tensor, height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        v, u = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
        src = H_inv @ torch.stack([u, v, torch.ones_like(u)]).reshape(3, -1)
        src = src / src[2:3].clamp_min(1e-6)
        return src[0].reshape(height, width), src[1].reshape(height, width)

    @staticmethod
    def _grid_sample_coordinates(u_src: torch.Tensor, v_src: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = (2.0 * u_src / max(width - 1, 1)) - 1.0
        y = (2.0 * v_src / max(height - 1, 1)) - 1.0
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _depth_scale(R: torch.Tensor, K_inv: torch.Tensor, u_src: torch.Tensor, v_src: torch.Tensor) -> torch.Tensor:
        rays = K_inv @ torch.stack([u_src, v_src, torch.ones_like(u_src)]).reshape(3, -1)
        alpha = R[2:3] @ rays
        return alpha.reshape_as(u_src)
