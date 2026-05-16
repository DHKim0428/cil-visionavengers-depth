from __future__ import annotations

from pathlib import Path

from dataset.factory import CILDepthDataLoaders, build_cil_depth_dataloaders

# Backward-compatible alias for transition DA2 entrypoints.  Canonical callers
# should use build_cil_depth_dataloaders directly with composable config fields.
DA2DataLoaders = CILDepthDataLoaders


def build_da2_dataloaders(
    *,
    data_dir: str | Path,
    pipeline: str,
    input_size: int,
    batch_size: int,
    val_fraction: float,
    split_seed: int,
    num_workers: int,
    split_file: str | Path | None = None,
    max_samples: int | None = None,
    teacher_mask_dir: str | Path | None = None,
    augmentation_config: dict | None = None,
    pin_memory: bool = True,
    drop_last_train: bool = True,
) -> DA2DataLoaders:
    """Compatibility wrapper around the canonical composable CIL factory.

    New code should not branch on ``legacy_square``/``dpt_native`` directly;
    these names are retained only so the transition DA2 evaluator remains usable.
    """
    if pipeline == "legacy_square":
        views = {
            "train": {
                "resize_policy": "square",
                "output_grid": "model_input",
                "crop_size": None,
                "normalize": "imagenet",
            },
            "eval": {
                "resize_policy": "square",
                "output_grid": "model_input",
                "crop_size": None,
                "normalize": "imagenet",
            },
        }
    elif pipeline == "dpt_native":
        views = {
            "train": {
                "resize_policy": "dpt_lower_bound",
                "output_grid": "model_input",
                "crop_size": input_size,
                "normalize": "imagenet",
            },
            "eval": {
                "resize_policy": "dpt_lower_bound",
                "output_grid": "native_gt",
                "crop_size": None,
                "normalize": "imagenet",
            },
        }
    else:
        raise ValueError(f"Unknown DA2 compatibility pipeline {pipeline!r}")

    return build_cil_depth_dataloaders(
        data_config={
            "dataset": "cil_depth",
            "root": str(data_dir),
            "image_size": input_size,
            "val_fraction": val_fraction,
            "split_seed": split_seed,
            "split_file": None if split_file is None else str(split_file),
            "max_samples": max_samples,
            "views": views,
        },
        train_config={"batch_size": batch_size, "num_workers": num_workers},
        augmentation_config=augmentation_config,
        teacher_mask_dir=teacher_mask_dir,
        pin_memory=pin_memory,
        drop_last_train=drop_last_train,
    )
