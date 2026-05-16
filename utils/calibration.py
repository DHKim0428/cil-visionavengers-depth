from __future__ import annotations

import numpy as np


def scale_depth_percentile(depth: np.ndarray, percentile: float = 99.0, target: float = 80.0, max_clip: float = 60000.0) -> tuple[np.ndarray, float, float, int]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        raise ValueError("Prediction has no finite positive depth values for percentile scaling")
    value = float(np.percentile(depth[valid], percentile))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid prediction percentile p{percentile}: {value}")
    scale = float(target / value)
    scaled = (depth * scale).astype(np.float32, copy=False)
    clipped = int((scaled > max_clip).sum())
    if clipped:
        scaled = np.minimum(scaled, max_clip).astype(np.float32, copy=False)
    return scaled, scale, value, clipped
