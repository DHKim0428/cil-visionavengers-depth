from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DepthAugmentation:
    def __init__(self, cfg: dict[str, Any] | None) -> None:
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


class CILDepthDataset(Dataset):
    def __init__(self, data_root: str | Path, names: list[str], view_cfg: dict[str, Any], augmentation: DepthAugmentation | None = None) -> None:
        self.root = Path(data_root)
        self.names = names
        self.view_cfg = view_cfg
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.names[idx]
        bgr = cv2.imread(str(self.root / name))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {self.root / name}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(self.root / name.replace("_rgb.png", "_depth.npy")).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= 0.001) & (depth <= 80.0)

        image, depth, valid = resize_sample(image, depth, valid, self.view_cfg)
        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        depth_t = torch.from_numpy(np.ascontiguousarray(depth))[None].float()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid))[None].float()

        if self.augmentation is not None:
            image_t, depth_t, valid_t = self.augmentation(image_t, depth_t, valid_t)
        if self.view_cfg.get("normalize") == "imagenet":
            image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD
        return {"image": image_t, "depth": depth_t[0], "valid_mask": valid_t[0] > 0.5, "name": name}


def resize_sample(image: np.ndarray, depth: np.ndarray, valid: np.ndarray, view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = view.get("resize_policy", "none")
    size = view.get("image_size")
    if policy == "none" or size is None:
        return image, depth, valid

    size = int(size)
    if policy == "square":
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        if view.get("output_grid", "native_gt") == "model_input":
            depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
        return image, depth, valid

    if policy == "dpt_lower_bound":
        h, w = image.shape[:2]
        scale = max(size / h, size / w)
        new_w = int(np.ceil(w * scale / 14) * 14)
        new_h = int(np.ceil(h * scale / 14) * 14)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if view.get("output_grid") == "model_input":
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            crop = size if view.get("crop_size") == "image_size" else view.get("crop_size")
            if crop:
                crop = int(crop)
                top, left = max(0, (new_h - crop) // 2), max(0, (new_w - crop) // 2)
                image, depth, valid = image[top:top + crop, left:left + crop], depth[top:top + crop, left:left + crop], valid[top:top + crop, left:left + crop]
        return image, depth, valid

    raise ValueError(f"Unknown resize_policy: {policy}")


def rgb_names(data_root: str | Path, max_samples: int | None = None) -> list[str]:
    names = sorted(p.name for p in Path(data_root).glob("*_rgb.png"))
    if max_samples is not None:
        names = names[:max_samples]
    if not names:
        raise FileNotFoundError(f"No *_rgb.png files found under {data_root}")
    return names


def split_names(names: list[str], val_fraction: float, seed: int, split_file: str | Path | None = None) -> tuple[list[str], list[str]]:
    if split_file and Path(split_file).exists():
        with Path(split_file).open("r", encoding="utf-8") as f:
            split = json.load(f)
        return list(split["train_names"]), list(split["val_names"])
    rng = np.random.default_rng(seed)
    indices = np.arange(len(names))
    rng.shuffle(indices)
    n_val = max(1, int(len(names) * val_fraction))
    train = [names[i] for i in sorted(indices[n_val:])]
    val = [names[i] for i in sorted(indices[:n_val])]
    if split_file:
        path = Path(split_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"split_seed": seed, "val_fraction": val_fraction, "train_names": train, "val_names": val}, f, indent=2)
    return train, val


def build_cil_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    data = cfg["data"]
    train_view = dict(data.get("views", {}).get("train", {}))
    val_view = dict(data.get("views", {}).get("eval", train_view))
    train_view.setdefault("image_size", data.get("image_size"))
    val_view.setdefault("image_size", data.get("image_size"))

    names = rgb_names(data["root"], data.get("max_samples"))
    train_names, val_names = split_names(names, float(data.get("val_fraction", 0.05)), int(data.get("split_seed", 42)), data.get("split_file"))
    augmentation = None if (cfg.get("augmentation") or {}).get("name", "none") == "none" else DepthAugmentation(cfg.get("augmentation"))

    train = cfg.get("train", {})
    train_loader = DataLoader(CILDepthDataset(data["root"], train_names, train_view, augmentation), batch_size=int(train.get("batch_size", 8)), shuffle=True, num_workers=int(train.get("num_workers", 4)), pin_memory=True, drop_last=True)
    val_loader = DataLoader(CILDepthDataset(data["root"], val_names, val_view), batch_size=int(train.get("batch_size", 8)), shuffle=False, num_workers=int(train.get("num_workers", 4)), pin_memory=True)
    return train_loader, val_loader, train_names, val_names
