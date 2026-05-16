from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset.raw_cil import depth_filename_from_rgb, sample_stem_from_rgb

TEACHER_MASK_GRIDS = {"raw_depth", "square"}


@dataclass(frozen=True)
class TeacherMaskSpec:
    """Canonical teacher-mask contract loaded from config + artifact metadata."""

    directory: Path
    grid: str
    size: int | None
    metadata: dict[str, Any]
    summary: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TeacherMaskSpec | None":
        cfg = config or {}
        if not cfg.get("enabled", False):
            return None
        if not cfg.get("dir"):
            raise ValueError("supervision.teacher_mask.enabled=true requires `dir`")
        grid = str(cfg.get("grid", ""))
        if grid not in TEACHER_MASK_GRIDS:
            raise ValueError(
                "supervision.teacher_mask.grid must be one of: raw_depth, square"
            )
        size = cfg.get("size")
        if grid == "square":
            if size is None:
                raise ValueError("square teacher masks require supervision.teacher_mask.size")
            size = int(size)
            if size <= 0:
                raise ValueError("supervision.teacher_mask.size must be positive")
        elif size is not None:
            raise ValueError("raw_depth teacher masks must use size: null")

        directory = Path(cfg["dir"])
        metadata = load_teacher_mask_metadata(directory)
        summary = load_teacher_mask_summary(directory)
        metadata_grid = metadata.get("grid")
        metadata_size = metadata.get("size")
        if metadata_grid != grid:
            raise ValueError(
                f"Teacher-mask config grid={grid!r} does not match artifact metadata grid={metadata_grid!r}"
            )
        if metadata_size != size:
            raise ValueError(
                f"Teacher-mask config size={size!r} does not match artifact metadata size={metadata_size!r}"
            )
        for key in ("threshold_percentile", "model_dir", "process_res"):
            expected = cfg.get(key)
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(
                    f"Teacher-mask config {key}={expected!r} does not match "
                    f"artifact metadata {key}={metadata.get(key)!r}"
                )
        return cls(directory=directory, grid=grid, size=size, metadata=metadata, summary=summary)

    def export_metadata(self) -> dict[str, Any]:
        payload = {
            "dir": str(self.directory),
            "grid": self.grid,
            "size": self.size,
            "artifact": self.metadata,
        }
        if self.summary is not None:
            payload["summary"] = self.summary
        return payload


def dataset_valid_mask(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Return the native dataset-valid mask for a depth map."""
    return np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)


def teacher_mask_path(mask_dir: str | Path, rgb_filename: str) -> Path:
    return Path(mask_dir) / f"{sample_stem_from_rgb(rgb_filename)}_teacher_mask.png"


def load_teacher_mask_metadata(mask_dir: str | Path) -> dict[str, Any]:
    path = Path(mask_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical teacher-mask directory requires metadata.json: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
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
    if metadata["grid"] not in TEACHER_MASK_GRIDS:
        raise ValueError(f"Unsupported teacher-mask metadata grid={metadata['grid']!r}")
    return metadata


def load_teacher_mask_summary(mask_dir: str | Path) -> dict[str, Any] | None:
    path = Path(mask_dir) / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_teacher_mask(
    teacher_mask: TeacherMaskSpec | str | Path | None,
    rgb_filename: str,
) -> np.ndarray | None:
    """Load one binary teacher reliability mask from a canonical/legacy source."""
    if teacher_mask is None:
        return None
    mask_dir = teacher_mask.directory if isinstance(teacher_mask, TeacherMaskSpec) else Path(teacher_mask)
    mask_path = teacher_mask_path(mask_dir, rgb_filename)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing teacher reliability mask for {rgb_filename}: {mask_path}"
        )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read teacher reliability mask: {mask_path}")
    return mask > 127


def validate_teacher_mask_spec(
    *,
    spec: TeacherMaskSpec,
    data_dir: str | Path,
    train_names: list[str],
    train_view_spec,
    shape_check_limit: int | None = 32,
) -> dict[str, Any]:
    """Validate mask coverage, metadata, split coupling, and expected grid shape."""
    if not train_names:
        raise ValueError("Cannot validate teacher masks without train_names")
    missing = [name for name in train_names if not teacher_mask_path(spec.directory, name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Teacher-mask directory is missing {len(missing)} training masks; first missing: {missing[0]}"
        )
    if int(spec.metadata.get("num_masks_saved", -1)) < len(train_names):
        raise ValueError(
            "Teacher-mask metadata num_masks_saved is smaller than the requested training split: "
            f"{spec.metadata.get('num_masks_saved')} < {len(train_names)}"
        )

    artifact_split_file = spec.metadata.get("split_file")
    if artifact_split_file:
        split_path = Path(artifact_split_file)
        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as handle:
                split_payload = json.load(handle)
            artifact_train = set(split_payload.get("train_names", []))
            absent = [name for name in train_names if name not in artifact_train]
            if absent:
                raise ValueError(
                    "Teacher-mask artifact split does not cover the requested training split; "
                    f"first absent sample: {absent[0]}"
                )

    if spec.grid == "square":
        if train_view_spec.output_grid != "model_input" or train_view_spec.resize_policy != "square":
            raise ValueError(
                "square teacher masks are only compatible with square/model_input training views; "
                f"got resize_policy={train_view_spec.resize_policy!r}, output_grid={train_view_spec.output_grid!r}"
            )
        if train_view_spec.image_size != spec.size:
            raise ValueError(
                "square teacher-mask size does not match training target grid: "
                f"mask size={spec.size}, train image_size={train_view_spec.image_size}"
            )

    checked = train_names if shape_check_limit is None else train_names[:shape_check_limit]
    data_root = Path(data_dir)
    for name in checked:
        mask = load_teacher_mask(spec, name)
        assert mask is not None
        if spec.grid == "square":
            expected_shape = (spec.size, spec.size)
        else:
            depth = np.load(data_root / depth_filename_from_rgb(name))
            expected_shape = depth.shape
        if mask.shape != expected_shape:
            raise ValueError(
                f"Teacher mask shape mismatch for {name}: got {mask.shape}, expected {expected_shape}"
            )
    return {
        "validated": True,
        "coverage_train_names": len(train_names),
        "shape_checks": len(checked),
        "grid": spec.grid,
        "size": spec.size,
    }


def compose_valid_mask(
    dataset_mask: np.ndarray,
    teacher_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compose the final supervision mask after all paired spatial transforms."""
    if teacher_mask is None:
        return dataset_mask.astype(bool)
    if teacher_mask.shape != dataset_mask.shape:
        raise ValueError(
            "Teacher mask and dataset mask must share the same spatial shape before "
            f"composition, got {teacher_mask.shape} and {dataset_mask.shape}"
        )
    return dataset_mask.astype(bool) & teacher_mask.astype(bool)
