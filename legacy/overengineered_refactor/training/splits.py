from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_or_load_split(
    sample_names: list[str],
    val_fraction: float,
    seed: int,
    split_file: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Create or load a deterministic filename-based train/val split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n_total = len(sample_names)
    if n_total < 2:
        raise ValueError(f"Need at least two samples to create a split, got {n_total}")

    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(
            f"Invalid split: n_total={n_total}, val_fraction={val_fraction} gives "
            f"n_train={n_train}, n_val={n_val}"
        )

    names_set = set(sample_names)
    split_path = Path(split_file) if split_file is not None else None

    if split_path is not None and split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)

        train_names = list(split["train_names"])
        val_names = list(split["val_names"])
        missing = [name for name in train_names + val_names if name not in names_set]
        if missing:
            raise ValueError(
                f"Split file {split_path} references {len(missing)} missing samples. "
                f"First missing sample: {missing[0]}"
            )
        return train_names, val_names

    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_names = [sample_names[i] for i in sorted(train_indices)]
    val_names = [sample_names[i] for i in sorted(val_indices)]

    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split_seed": seed,
            "val_fraction": val_fraction,
            "n_total": n_total,
            "train_names": train_names,
            "val_names": val_names,
        }
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return train_names, val_names
