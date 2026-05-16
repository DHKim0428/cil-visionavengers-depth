import os
import math
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.augmentations import DepthAugmentation


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
        teacher_mask_dir=None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.max_depth = max_depth
        self.teacher_mask_dir = Path(teacher_mask_dir) if teacher_mask_dir else None
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
        if self.teacher_mask_dir is not None:
            stem = rgb_path.name.replace("_rgb.png", "")
            teacher_mask_path = self.teacher_mask_dir / f"{stem}_teacher_mask.png"
            if not teacher_mask_path.exists():
                raise FileNotFoundError(
                    f"Missing teacher reliability mask for {rgb_path.name}: {teacher_mask_path}"
                )

            teacher_mask = np.array(Image.open(teacher_mask_path).convert("L"), dtype=np.float32)
            teacher_mask_t = torch.from_numpy(teacher_mask).unsqueeze(0).unsqueeze(0)
            teacher_mask_t = F.interpolate(
                teacher_mask_t,
                size=(self.img_size, self.img_size),
                mode="nearest",
            ).squeeze(0)
            valid_mask = valid_mask * (teacher_mask_t > 127.0).float()
            depth_t = depth_t * valid_mask

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
