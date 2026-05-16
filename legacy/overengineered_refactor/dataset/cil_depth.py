from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.augmentations import DepthAugmentation
from dataset.tensors import normalize_imagenet, to_1hw, to_chw_image
from dataset.raw_cil import load_rgb_depth
from dataset.supervision import TeacherMaskSpec, compose_valid_mask, dataset_valid_mask, load_teacher_mask
from dataset.transform import Crop, PrepareForNet, Resize

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
RESIZE_POLICIES = {"none", "square", "dpt_lower_bound"}
OUTPUT_GRIDS = {"native_gt", "model_input"}
NORMALIZATIONS = {"none", "imagenet"}


@dataclass(frozen=True)
class CILDepthViewSpec:
    """One configurable RGB/depth view emitted by the canonical CIL dataset."""

    image_size: int | None
    resize_policy: str
    output_grid: str
    normalize: str
    crop_size: int | None = None

    def __post_init__(self) -> None:
        if self.resize_policy not in RESIZE_POLICIES:
            raise ValueError(f"Unsupported resize_policy={self.resize_policy!r}; expected one of {sorted(RESIZE_POLICIES)}")
        if self.output_grid not in OUTPUT_GRIDS:
            raise ValueError(f"Unsupported output_grid={self.output_grid!r}; expected one of {sorted(OUTPUT_GRIDS)}")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(f"Unsupported normalize={self.normalize!r}; expected one of {sorted(NORMALIZATIONS)}")
        if self.resize_policy != "none" and self.image_size is None:
            raise ValueError(f"resize_policy={self.resize_policy!r} requires image_size")
        if self.output_grid == "model_input" and self.resize_policy == "none" and self.crop_size is not None:
            raise ValueError("crop_size with resize_policy='none' is not supported")

    @classmethod
    def from_config(cls, data_config: dict[str, Any], view: str) -> "CILDepthViewSpec":
        views = data_config.get("views") or {}
        if view not in views:
            raise KeyError(f"data.views.{view} must be defined for canonical CIL depth data")
        view_cfg = views[view] or {}
        image_size = view_cfg.get("image_size", data_config.get("image_size"))
        crop_size = view_cfg.get("crop_size")
        if crop_size == "image_size":
            crop_size = image_size
        return cls(
            image_size=None if image_size is None else int(image_size),
            resize_policy=str(view_cfg.get("resize_policy", "none")),
            output_grid=str(view_cfg.get("output_grid", "native_gt")),
            normalize=str(view_cfg.get("normalize", "none")),
            crop_size=None if crop_size is None else int(crop_size),
        )


class CILDepthDataset(Dataset):
    """Canonical CIL RGB/depth dataset with one metric-depth output contract.

    Every view returns:

    - ``image``: CHW float tensor after configurable preprocessing;
    - ``depth``: metric depth in meters on the configured output grid;
    - ``valid_mask``: GT-valid supervision mask on the same grid as ``depth``;
    - ``name``: RGB filename.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        filenames: list[str],
        spec: CILDepthViewSpec,
        augmentation: DepthAugmentation | None = None,
        teacher_mask: TeacherMaskSpec | str | Path | None = None,
        min_depth: float = MIN_DEPTH,
        max_depth: float = MAX_DEPTH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.filenames = list(filenames)
        self.spec = spec
        self.augmentation = augmentation
        self.teacher_mask = teacher_mask
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.prepare = PrepareForNet()
        self.crop = Crop(spec.crop_size) if spec.crop_size is not None else None
        self.dpt_resize = None
        if spec.resize_policy == "dpt_lower_bound":
            assert spec.image_size is not None
            self.dpt_resize = Resize(
                width=spec.image_size,
                height=spec.image_size,
                resize_target=(spec.output_grid == "model_input"),
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rgb_name = self.filenames[idx]
        image, depth = load_rgb_depth(self.data_dir, rgb_name)
        image = image.astype(np.float32) / 255.0
        sample: dict[str, np.ndarray] = {
            "image": image,
            "depth": depth,
            "dataset_valid_mask": dataset_valid_mask(depth, self.min_depth, self.max_depth),
        }
        # Raw-depth masks live on the native GT grid, so they must enter before
        # any paired spatial transform. Plain paths retain the old behavior for
        # compatibility with non-canonical callers.
        if not isinstance(self.teacher_mask, TeacherMaskSpec) or self.teacher_mask.grid == "raw_depth":
            teacher_mask = load_teacher_mask(self.teacher_mask, rgb_name)
            if teacher_mask is not None:
                sample["teacher_mask"] = teacher_mask

        sample = self._apply_view(sample)
        # Square masks are explicitly tied to the post-view square target grid.
        # The factory validator guarantees that only square/model_input views
        # with the matching size reach this branch.
        if isinstance(self.teacher_mask, TeacherMaskSpec) and self.teacher_mask.grid == "square":
            teacher_mask = load_teacher_mask(self.teacher_mask, rgb_name)
            if teacher_mask is not None:
                sample["teacher_mask"] = teacher_mask
        valid_mask = compose_valid_mask(sample["dataset_valid_mask"], sample.get("teacher_mask"))

        image_t = self._image_to_tensor(sample["image"])
        depth_t = to_1hw(sample["depth"])
        valid_t = to_1hw(valid_mask.astype(np.float32))

        if self.augmentation is not None:
            image_t, depth_t, valid_t = self.augmentation(image_t, depth_t, valid_t)

        if self.spec.normalize == "imagenet":
            image_t = normalize_imagenet(image_t)

        return {
            "image": image_t,
            "depth": depth_t.squeeze(0),
            "valid_mask": valid_t.squeeze(0) > 0.5,
            "name": rgb_name,
        }

    def _apply_view(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.spec.resize_policy == "none":
            return sample
        if self.spec.resize_policy == "square":
            assert self.spec.image_size is not None
            return self._square_resize(sample, self.spec.image_size)
        if self.spec.resize_policy == "dpt_lower_bound":
            assert self.dpt_resize is not None
            sample = self.dpt_resize(sample)
            sample = self.prepare(sample)
            if self.crop is not None:
                sample = self.crop(sample)
            return sample
        raise AssertionError(f"unreachable resize_policy: {self.spec.resize_policy}")

    def _square_resize(self, sample: dict[str, np.ndarray], image_size: int) -> dict[str, np.ndarray]:
        sample = dict(sample)
        sample["image"] = cv2.resize(sample["image"], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if self.spec.output_grid == "model_input":
            sample["depth"] = cv2.resize(sample["depth"], (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            sample["dataset_valid_mask"] = cv2.resize(
                sample["dataset_valid_mask"].astype(np.uint8),
                (image_size, image_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if "teacher_mask" in sample:
                sample["teacher_mask"] = cv2.resize(
                    sample["teacher_mask"].astype(np.uint8),
                    (image_size, image_size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        return sample

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected RGB image with HWC shape, got {image.shape}")
        # DPT preprocessing already converts HWC -> CHW via PrepareForNet.
        if image.shape[0] == 3 and image.shape[-1] != 3:
            return torch.from_numpy(np.ascontiguousarray(image)).float()
        return to_chw_image(image)
