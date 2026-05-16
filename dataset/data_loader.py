from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataset.data_augment import DepthAugmentation

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
        new_w = int(np.round((w * scale) / 14) * 14)
        new_h = int(np.round((h * scale) / 14) * 14)
        if new_w < size:
            new_w = int(np.ceil((w * scale) / 14) * 14)
        if new_h < size:
            new_h = int(np.ceil((h * scale) / 14) * 14)
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
