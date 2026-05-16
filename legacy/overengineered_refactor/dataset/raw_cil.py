from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def discover_rgb_filenames(data_dir: str | Path, max_samples: int | None = None) -> list[str]:
    """Return sorted CIL RGB filenames, optionally truncated for debug runs."""
    root = Path(data_dir)
    filenames = sorted(path.name for path in root.glob("*_rgb.png"))
    if max_samples is not None:
        filenames = filenames[:max_samples]
    if not filenames:
        raise FileNotFoundError(f"No *_rgb.png files found in {root}")
    return filenames


def depth_filename_from_rgb(rgb_filename: str) -> str:
    if not rgb_filename.endswith("_rgb.png"):
        raise ValueError(f"Expected an *_rgb.png filename, got: {rgb_filename}")
    return rgb_filename.replace("_rgb.png", "_depth.npy")


def sample_stem_from_rgb(rgb_filename: str) -> str:
    if not rgb_filename.endswith("_rgb.png"):
        raise ValueError(f"Expected an *_rgb.png filename, got: {rgb_filename}")
    return rgb_filename[: -len("_rgb.png")]


def load_rgb_depth(data_dir: str | Path, rgb_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one raw RGB/depth pair without model-specific preprocessing."""
    root = Path(data_dir)
    rgb_path = root / rgb_filename
    depth_path = root / depth_filename_from_rgb(rgb_filename)

    image = cv2.imread(str(rgb_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {rgb_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path).astype(np.float32)
    return image, depth
