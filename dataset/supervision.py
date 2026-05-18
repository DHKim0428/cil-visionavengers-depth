from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TeacherMaskSpec:
    directory: Path
    grid: str
    metadata: dict[str, Any]

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "TeacherMaskSpec | None":
        cfg = cfg or {}
        if not cfg.get("enabled", False):
            return None
        if not cfg.get("dir"):
            raise ValueError("supervision.teacher_mask.enabled=true requires `dir`")

        directory = Path(cfg["dir"])
        metadata = load_teacher_mask_metadata(directory)
        grid = str(cfg.get("grid", metadata.get("grid", "")))
        if grid != "raw_depth":
            raise ValueError("This training path currently supports only raw_depth teacher masks")
        if metadata.get("grid") != grid:
            raise ValueError(f"Teacher-mask grid mismatch: config={grid!r}, metadata={metadata.get('grid')!r}")
        if metadata.get("size") is not None:
            raise ValueError("raw_depth teacher masks must have metadata size=null")

        for key in ("threshold_percentile", "model_dir", "process_res"):
            expected = cfg.get(key)
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(
                    f"Teacher-mask {key} mismatch: config={expected!r}, metadata={metadata.get(key)!r}"
                )
        return cls(directory=directory, grid=grid, metadata=metadata)


def teacher_mask_path(mask_dir: str | Path, rgb_name: str) -> Path:
    stem = rgb_name.replace("_rgb.png", "")
    return Path(mask_dir) / f"{stem}_teacher_mask.png"


def load_teacher_mask_metadata(mask_dir: str | Path) -> dict[str, Any]:
    path = Path(mask_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Teacher-mask directory requires metadata.json: {path}")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = {
        "grid",
        "size",
        "split_file",
        "threshold_percentile",
        "model_dir",
        "process_res",
        "num_masks_saved",
        "mask_meaning",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Teacher-mask metadata missing required fields: {missing}")
    return metadata


def load_teacher_mask(spec: TeacherMaskSpec, rgb_name: str) -> np.ndarray:
    path = teacher_mask_path(spec.directory, rgb_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher reliability mask for {rgb_name}: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read teacher reliability mask: {path}")
    return mask > 127


def validate_teacher_masks(spec: TeacherMaskSpec, data_root: str | Path, train_names: list[str], shape_check_limit: int = 16) -> dict[str, Any]:
    if not train_names:
        raise ValueError("Cannot validate teacher masks without train_names")
    if int(spec.metadata.get("num_masks_saved", -1)) < len(train_names):
        raise ValueError(
            "Teacher-mask artifact has fewer masks than the training split: "
            f"{spec.metadata.get('num_masks_saved')} < {len(train_names)}"
        )

    missing = [name for name in train_names if not teacher_mask_path(spec.directory, name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Teacher-mask directory is missing {len(missing)} training masks; first missing: {missing[0]}"
        )

    data_root = Path(data_root)
    checked = train_names[: max(0, shape_check_limit)]
    for name in checked:
        depth = np.load(data_root / name.replace("_rgb.png", "_depth.npy"))
        mask = load_teacher_mask(spec, name)
        if mask.shape != depth.shape:
            raise ValueError(
                f"Teacher mask shape mismatch for {name}: got {mask.shape}, expected {depth.shape}"
            )

    return {"validated": True, "grid": spec.grid, "num_train_masks": len(train_names), "shape_checks": len(checked)}
