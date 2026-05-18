from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.data_augment import DepthAugmentation
from dataset.supervision import TeacherMaskSpec, load_teacher_mask, validate_teacher_masks

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CILDepthDataset(Dataset):
    def __init__(self, data_root: str | Path, names: list[str], model: str, image_size: int, training: bool, augmentation: DepthAugmentation | None = None, cutmix: dict[str, Any] | None = None, teacher_mask: TeacherMaskSpec | None = None, input_profile: str | None = None) -> None:
        self.root = Path(data_root)
        self.names = names
        self.model = model
        self.input_profile = input_profile or ("da2" if model.startswith("da2_") else "unet")
        self.image_size = int(image_size)
        self.training = training
        self.augmentation = augmentation
        self.teacher_mask = teacher_mask if training else None
        cutmix = cutmix or {}
        self.cutmix_prob = float(cutmix.get("prob", 0.0)) if cutmix.get("enabled", False) and training else 0.0
        self.cutmix_alpha = float(cutmix.get("alpha", 1.0))

    def __len__(self) -> int:
        return len(self.names)

    def _load_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, bool]:
        name = self.names[idx]
        bgr = cv2.imread(str(self.root / name))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {self.root / name}")

        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = np.load(self.root / name.replace("_rgb.png", "_depth.npy")).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= 0.001) & (depth <= 80.0)
        if self.teacher_mask is not None:
            teacher_mask = load_teacher_mask(self.teacher_mask, name)
            if teacher_mask.shape != valid.shape:
                raise ValueError(f"Teacher mask shape mismatch for {name}: got {teacher_mask.shape}, expected {valid.shape}")
            valid = valid & teacher_mask

        if self.input_profile == "da2":
            image, depth, valid = preprocess_da2_sample(image, depth, valid, self.image_size, self.training)
            normalize = True
        elif self.input_profile == "unet":
            image, depth, valid = preprocess_unet_sample(image, depth, valid, self.image_size, self.training)
            normalize = False
        else:
            raise ValueError(f"Unknown input profile for {self.model}: {self.input_profile}")

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        depth_t = torch.from_numpy(np.ascontiguousarray(depth))[None].float()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid))[None].float()
        return image_t, depth_t, valid_t, name, normalize

    def _cutmix(self, image_a: torch.Tensor, depth_a: torch.Tensor, valid_a: torch.Tensor, image_b: torch.Tensor, depth_b: torch.Tensor, valid_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = image_a.shape
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha)) if self.cutmix_alpha > 0 else 0.5
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_w = max(1, int(round(w * cut_ratio)))
        cut_h = max(1, int(round(h * cut_ratio)))
        cx = int(torch.randint(0, w, ()).item())
        cy = int(torch.randint(0, h, ()).item())
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + (cut_w + 1) // 2)
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + (cut_h + 1) // 2)

        region = torch.zeros((1, h, w), dtype=torch.bool)
        region[:, y1:y2, x1:x2] = True
        keep_a = ~region

        image = torch.where(region, image_b, image_a)
        depth = torch.where(region, depth_b, depth_a)
        mask_a = valid_a.bool() & keep_a
        mask_b = valid_b.bool() & region
        valid = (mask_a | mask_b).float()
        loss_masks = torch.stack([mask_a[0], mask_b[0]])
        weights = loss_masks.flatten(1).sum(dim=1).float()
        if float(weights.sum()) > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.tensor([1.0, 0.0])
        return image, depth, valid, loss_masks, weights

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_t, depth_t, valid_t, name, normalize = self._load_sample(idx)
        if self.augmentation is not None:
            # Apply standard paired/RGB augmentations before any sample mixing.
            image_orig, depth_orig, valid_orig = image_t, depth_t, valid_t
            image_t, depth_t, valid_t = self.augmentation(image_t, depth_t, valid_t)
            if int(valid_t.sum()) == 0 and int(valid_orig.sum()) > 0:
                image_t, depth_t, valid_t = image_orig, depth_orig, valid_orig

        loss_masks = torch.stack([valid_t[0] > 0.5, torch.zeros_like(valid_t[0], dtype=torch.bool)])
        loss_weights = torch.tensor([1.0, 0.0])
        if self.cutmix_prob and len(self.names) > 1 and torch.rand(()) < self.cutmix_prob:
            # Mix in a second training sample and keep per-region supervision masks.
            donor_idx = int(torch.randint(0, len(self.names) - 1, ()).item())
            if donor_idx >= idx:
                donor_idx += 1
            image_b, depth_b, valid_b, donor_name, _ = self._load_sample(donor_idx)
            if self.augmentation is not None:
                # Keep donor augmentation independent from the anchor sample.
                image_orig, depth_orig, valid_orig = image_b, depth_b, valid_b
                image_b, depth_b, valid_b = self.augmentation(image_b, depth_b, valid_b)
                if int(valid_b.sum()) == 0 and int(valid_orig.sum()) > 0:
                    image_b, depth_b, valid_b = image_orig, depth_orig, valid_orig
            image_t, depth_t, valid_t, loss_masks, loss_weights = self._cutmix(image_t, depth_t, valid_t, image_b, depth_b, valid_b)
            name = f"{name}|cutmix|{donor_name}"

        if normalize:
            image_t = (image_t - IMAGENET_MEAN) / IMAGENET_STD

        sample: dict[str, torch.Tensor | str] = {"image": image_t, "depth": depth_t[0], "valid_mask": valid_t[0] > 0.5, "name": name}
        if self.cutmix_prob:
            sample["loss_masks"] = loss_masks
            sample["loss_weights"] = loss_weights
        return sample


def preprocess_da2_sample(image: np.ndarray, depth: np.ndarray, valid: np.ndarray, image_size: int, training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # DA2 follows the official DPT-style input convention: resize the shorter side up
    # to at least image_size and make both dimensions divisible by 14. During training
    # depth/mask are put on the same cropped model grid as the image. During eval, GT
    # stays native-resolution and the prediction is resized back to GT before siRMSE.
    h, w = image.shape[:2]
    scale = max(image_size / h, image_size / w)
    new_w = int(np.round((w * scale) / 14) * 14)
    new_h = int(np.round((h * scale) / 14) * 14)
    if new_w < image_size:
        new_w = int(np.ceil((w * scale) / 14) * 14)
    if new_h < image_size:
        new_h = int(np.ceil((h * scale) / 14) * 14)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if not training:
        return image, depth, valid

    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    top, left = max(0, (new_h - image_size) // 2), max(0, (new_w - image_size) // 2)
    return image[top:top + image_size, left:left + image_size], depth[top:top + image_size, left:left + image_size], valid[top:top + image_size, left:left + image_size]


def preprocess_unet_sample(image: np.ndarray, depth: np.ndarray, valid: np.ndarray, image_size: int, training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # U-Net keeps the original simple square-input baseline. Training uses square GT
    # because the loss is computed on the model grid. Eval keeps native GT and resizes
    # the U-Net prediction back to native GT before siRMSE, matching DA2 eval semantics.
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if not training:
        return image, depth, valid
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return image, depth, valid


def rgb_names(data_root: str | Path, max_samples: int | None = None) -> list[str]:
    names = sorted(p.name for p in Path(data_root).glob("*_rgb.png"))
    if max_samples is not None:
        names = names[:max_samples]
    if not names:
        raise FileNotFoundError(f"No *_rgb.png files found under {data_root}")
    return names


def default_split_file(val_fraction: float, seed: int) -> Path | None:
    percent = val_fraction * 100.0
    rounded = int(round(percent))
    if abs(percent - rounded) > 1e-6:
        return None
    return Path(__file__).resolve().parents[1] / "configs" / "splits" / f"cil_depth_val_{rounded:02d}pct_seed{seed}.json"


def split_names_from_file(path: Path, names: list[str]) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    available = set(names)
    if "train_names" in split and "val_names" in split:
        train = list(split["train_names"])
        val = list(split["val_names"])
        if set(train).issubset(available) and set(val).issubset(available):
            return train, val
        raise ValueError(f"Split file {path} is not compatible with the selected dataset names")
    if "val_indices" in split:
        val_indices = sorted(int(i) for i in split["val_indices"])
        if val_indices and val_indices[-1] >= len(names):
            raise ValueError(f"Split file {path} expects at least {val_indices[-1] + 1} samples, got {len(names)}")
        val_set = set(val_indices)
        train = [name for i, name in enumerate(names) if i not in val_set]
        val = [names[i] for i in val_indices]
        return train, val
    raise ValueError(f"Split file {path} must contain train_names/val_names or val_indices")


def split_names(names: list[str], val_fraction: float, seed: int, split_file: str | Path | None = None) -> tuple[list[str], list[str]]:
    explicit_split = split_file is not None
    candidate = Path(split_file) if split_file else default_split_file(val_fraction, seed)
    if candidate and candidate.exists():
        try:
            return split_names_from_file(candidate, names)
        except ValueError:
            if explicit_split:
                raise

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
    model = cfg["model"]
    image_size = int(data.get("image_size", 518 if (str(model).startswith("da2_") or model == "unet_disp") else 128))

    names = rgb_names(data["root"], data.get("max_samples"))
    train_names, val_names = split_names(names, float(data.get("val_fraction", 0.05)), int(data.get("split_seed", 42)), data.get("split_file"))
    aug_cfg = cfg.get("augmentation") or {}
    augmentation = None if aug_cfg.get("name", "none") == "none" else DepthAugmentation(aug_cfg)
    cutmix = (aug_cfg.get("mix", {}) or {}).get("cutmix", {}) or {}
    teacher_mask = TeacherMaskSpec.from_config((cfg.get("supervision") or {}).get("teacher_mask"))
    if teacher_mask is not None:
        validate_teacher_masks(teacher_mask, data["root"], train_names)

    train = cfg.get("train", {})
    input_profile = "unet"
    if str(model).startswith("da2_"):
        input_profile = "da2"
    elif model == "unet_disp" and str((cfg.get("refiner") or {}).get("conditioning", "rgb")) in {"prior", "prior_features"}:
        input_profile = "da2"

    train_loader = DataLoader(CILDepthDataset(data["root"], train_names, model, image_size, training=True, augmentation=augmentation, cutmix=cutmix, teacher_mask=teacher_mask, input_profile=input_profile), batch_size=int(train.get("batch_size", 8)), shuffle=True, num_workers=int(train.get("num_workers", 4)), pin_memory=True, drop_last=True)
    val_loader = DataLoader(CILDepthDataset(data["root"], val_names, model, image_size, training=False, input_profile=input_profile), batch_size=int(train.get("batch_size", 8)), shuffle=False, num_workers=int(train.get("num_workers", 4)), pin_memory=True)
    return train_loader, val_loader, train_names, val_names
