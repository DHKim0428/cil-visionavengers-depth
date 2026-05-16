from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    return base64.b64encode(compressed).decode("utf-8")


def write_submission_csv(pred_dir: str | Path, out_csv: str | Path) -> int:
    pred_dir = Path(pred_dir)
    out_csv = Path(out_csv)
    rows = []
    for pred_path in sorted(pred_dir.glob("test_*.npy")):
        depth = np.load(pred_path)
        idx = pred_path.stem.split("_")[-1]
        rows.append({"id": f"test_{idx}_depth", "Depths": encode_depth(depth)})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "Depths"]).to_csv(out_csv, index=False)
    return len(rows)
