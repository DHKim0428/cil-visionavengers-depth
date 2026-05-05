import os
import math
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class DepthAugmentation:
    def __init__(
        self,
        enable_hflip=False,
        hflip_prob=0.5,
        enable_rotation=False,
        rotation_deg=3.0,
        enable_crop=False,
        crop_scale_min=0.9,
        enable_color_jitter=False,
        color_jitter_prob=0.8,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        tilt_mode="none",
        tilt_prob=0.5,
        tilt_max_yaw_deg=5.0,
        tilt_max_pitch_deg=5.0,
        tilt_fov_deg=60.0,
        eps=1e-6,
    ):
        assert tilt_mode in {"none", "naive", "geometry"}, f"Unknown tilt mode: {tilt_mode}"
        self.enable_hflip = enable_hflip
        self.hflip_prob = hflip_prob
        self.enable_rotation = enable_rotation
        self.rotation_rad = math.radians(rotation_deg)
        self.enable_crop = enable_crop
        self.crop_scale_min = crop_scale_min
        self.enable_color_jitter = enable_color_jitter
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.tilt_mode = tilt_mode
        self.tilt_prob = tilt_prob
        self.max_yaw_rad = math.radians(tilt_max_yaw_deg)
        self.max_pitch_rad = math.radians(tilt_max_pitch_deg)
        self.fov_rad = math.radians(tilt_fov_deg)
        self.eps = eps

    def __call__(self, image, depth, mask):
        if self.enable_hflip and torch.rand(()) < self.hflip_prob:
            image = torch.flip(image, dims=[2])
            depth = torch.flip(depth, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if self.enable_crop:
            image, depth, mask = self._random_square_crop(image, depth, mask)

        if self.enable_rotation:
            image, depth, mask = self._random_rotation(image, depth, mask)

        if self.enable_color_jitter:
            image = self._color_jitter(image)

        if self.tilt_mode == "none" or torch.rand(()) > self.tilt_prob:
            return image, depth, mask

        _, height, width = image.shape
        yaw = self._sample_angle(self.max_yaw_rad)
        pitch = self._sample_angle(self.max_pitch_rad)

        if abs(yaw) < self.eps and abs(pitch) < self.eps:
            return image, depth, mask

        device = image.device
        dtype = image.dtype
        K, K_inv = self._intrinsics(height, width, device, dtype)
        R = self._rotation(yaw, pitch, device, dtype)

        # Pure camera rotation induces a homography from source pixels to target pixels.
        H = K @ R @ K_inv
        H_inv = torch.linalg.inv(H)

        u_src, v_src = self._inverse_warp_coordinates(H_inv, height, width, device, dtype)
        grid = self._grid_sample_coordinates(u_src, v_src, height, width)

        image_aug = F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        depth_src = F.grid_sample(
            depth.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        mask_src = F.grid_sample(
            mask.unsqueeze(0),
            grid.unsqueeze(0),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        in_bounds = (u_src >= 0.0) & (u_src <= width - 1) & (v_src >= 0.0) & (v_src <= height - 1)
        alpha = self._depth_scale(R, K_inv, u_src, v_src)

        if self.tilt_mode == "geometry":
            depth_aug = depth_src * alpha.unsqueeze(0)
            alpha_valid = alpha > self.eps
        else:
            depth_aug = depth_src
            alpha_valid = torch.ones_like(in_bounds, dtype=torch.bool)

        valid = in_bounds & alpha_valid & (mask_src.squeeze(0) > 0.5) & (depth_aug.squeeze(0) > 0.0)
        valid = valid & torch.isfinite(depth_aug.squeeze(0))

        mask_aug = valid.unsqueeze(0).float()
        depth_aug = depth_aug * mask_aug
        # image_aug = image_aug * mask_aug

        return image_aug, depth_aug, mask_aug

    def _random_square_crop(self, image, depth, mask):
        _, height, width = image.shape
        if self.crop_scale_min >= 1.0:
            return image, depth, mask

        scale = self.crop_scale_min + (1.0 - self.crop_scale_min) * torch.rand(()).item()
        crop_size = max(1, min(height, width, int(round(min(height, width) * scale))))

        if crop_size >= min(height, width):
            return image, depth, mask

        top = torch.randint(0, height - crop_size + 1, ()).item()
        left = torch.randint(0, width - crop_size + 1, ()).item()

        image = image[:, top:top + crop_size, left:left + crop_size]
        depth = depth[:, top:top + crop_size, left:left + crop_size]
        mask = mask[:, top:top + crop_size, left:left + crop_size]

        image = F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)
        return image, depth, mask

    def _random_rotation(self, image, depth, mask):
        if self.rotation_rad <= 0.0:
            return image, depth, mask

        angle = (torch.rand(()).item() * 2.0 - 1.0) * self.rotation_rad
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0)

        batch_shape = (1, image.shape[0], image.shape[1], image.shape[2])
        grid = F.affine_grid(theta, batch_shape, align_corners=False)
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)
        depth = F.grid_sample(
            depth.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        mask = (mask > 0.5).float()
        depth = depth * mask
        return image, depth, mask

    def _color_jitter(self, image):
        if torch.rand(()) >= self.color_jitter_prob:
            return image

        image = image * self._sample_factor(self.brightness)
        image_mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - image_mean) * self._sample_factor(self.contrast) + image_mean
        gray = image.mean(dim=0, keepdim=True)
        image = gray + (image - gray) * self._sample_factor(self.saturation)
        return image.clamp(0.0, 1.0)

    def _sample_angle(self, max_abs_rad):
        if max_abs_rad <= 0.0:
            return 0.0
        return (torch.rand(()).item() * 2.0 - 1.0) * max_abs_rad

    def _sample_factor(self, strength):
        if strength <= 0.0:
            return 1.0
        return 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * strength

    def _intrinsics(self, height, width, device, dtype):
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        focal = width / (2.0 * math.tan(self.fov_rad / 2.0))

        K = torch.tensor(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        return K, torch.linalg.inv(K)

    def _rotation(self, yaw, pitch, device, dtype):
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)

        R_yaw = torch.tensor(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            device=device,
            dtype=dtype,
        )
        R_pitch = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
            device=device,
            dtype=dtype,
        )
        return R_pitch @ R_yaw

    def _inverse_warp_coordinates(self, H_inv, height, width, device, dtype):
        v, u = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        ones = torch.ones_like(u)
        target_pixels = torch.stack([u, v, ones], dim=0).reshape(3, -1)
        source_pixels = H_inv @ target_pixels
        denom = source_pixels[2:3]
        safe_denom = torch.where(
            denom.abs() < self.eps,
            denom.sign().clamp(min=0.0) * 2.0 * self.eps - self.eps,
            denom,
        )
        source_pixels = source_pixels / safe_denom

        u_src = source_pixels[0].reshape(height, width)
        v_src = source_pixels[1].reshape(height, width)
        return u_src, v_src

    def _grid_sample_coordinates(self, u_src, v_src, height, width):
        x_norm = 2.0 * u_src / max(width - 1, 1) - 1.0
        y_norm = 2.0 * v_src / max(height - 1, 1) - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    def _depth_scale(self, R, K_inv, u_src, v_src):
        ones = torch.ones_like(u_src)
        source_pixels = torch.stack([u_src, v_src, ones], dim=0).reshape(3, -1)
        rays = K_inv @ source_pixels
        alpha = R[2:3, :] @ rays
        return alpha.reshape_as(u_src)


class SimpleDepthDataset(Dataset):
    def __init__(
        self,
        root,
        img_size=128,
        max_samples=None,
        enable_hflip=False,
        hflip_prob=0.5,
        enable_rotation=False,
        rotation_deg=3.0,
        enable_crop=False,
        crop_scale_min=0.9,
        enable_color_jitter=False,
        color_jitter_prob=0.8,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        tilt_mode="none",
        tilt_prob=0.5,
        tilt_max_yaw_deg=5.0,
        tilt_max_pitch_deg=5.0,
        tilt_fov_deg=60.0,
        max_depth=80.0,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.max_depth = max_depth
        self.augmentation = DepthAugmentation(
            enable_hflip=enable_hflip,
            hflip_prob=hflip_prob,
            enable_rotation=enable_rotation,
            rotation_deg=rotation_deg,
            enable_crop=enable_crop,
            crop_scale_min=crop_scale_min,
            enable_color_jitter=enable_color_jitter,
            color_jitter_prob=color_jitter_prob,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            tilt_mode=tilt_mode,
            tilt_prob=tilt_prob,
            tilt_max_yaw_deg=tilt_max_yaw_deg,
            tilt_max_pitch_deg=tilt_max_pitch_deg,
            tilt_fov_deg=tilt_fov_deg,
        )
        
        self.rgb_files = sorted(self.root.glob("*_rgb.png"))
        if max_samples is not None:
            self.rgb_files = self.rgb_files[:max_samples]
        
        assert len(self.rgb_files) > 0, f"No *_rgb.png files found in {self.root}"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        
        # Load RGB
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        
        # Load Depth
        depth = np.load(depth_path).astype(np.float32)
        
        # Resize RGB
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        rgb_t = F.interpolate(rgb_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        rgb_t = rgb_t.squeeze(0)
        
        # Resize Depth
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(depth_t, size=(self.img_size, self.img_size), mode="nearest")
        depth_t = depth_t.squeeze(0)
        
        # Valid mask: depth > 0
        valid_mask = (depth_t > 0).float()

        # Apply all train-time paired augmentations before normalization so depth
        # updates remain in meters and masks stay aligned with the supervision.
        rgb_t, depth_t, valid_mask = self.augmentation(rgb_t, depth_t, valid_mask)
        
        # Normalize depth (as in demo)
        depth_t = torch.clamp(depth_t, min=0.0, max=self.max_depth)
        depth_t = depth_t / self.max_depth
        
        return {
            "image": rgb_t,
            "depth": depth_t,
            "mask": valid_mask,
            "name": rgb_path.name
        }
