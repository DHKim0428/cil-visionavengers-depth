from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from dataset.augmentations import depth_augmentation_from_config
from dataset.cil_depth import CILDepthDataset, CILDepthViewSpec
from dataset.raw_cil import discover_rgb_filenames
from dataset.supervision import TeacherMaskSpec, validate_teacher_mask_spec
from training.splits import make_or_load_split


@dataclass(frozen=True)
class CILDepthDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: CILDepthDataset
    val_dataset: CILDepthDataset
    train_names: list[str]
    val_names: list[str]
    train_spec: CILDepthViewSpec
    eval_spec: CILDepthViewSpec
    teacher_mask_validation: dict[str, Any] | None = None


def build_cil_depth_dataloaders(
    *,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    augmentation_config: dict[str, Any] | None = None,
    teacher_mask_spec: TeacherMaskSpec | None = None,
    teacher_mask_dir: str | Path | None = None,
    pin_memory: bool = True,
    drop_last_train: bool = True,
) -> CILDepthDataLoaders:
    """Build canonical CIL depth train/val loaders from composable config fields."""
    if data_config.get("dataset") != "cil_depth":
        raise ValueError(f"Unsupported dataset={data_config.get('dataset')!r}; expected 'cil_depth'")

    data_dir = data_config["root"]
    sample_names = discover_rgb_filenames(data_dir, max_samples=data_config.get("max_samples"))
    train_names, val_names = make_or_load_split(
        sample_names=sample_names,
        val_fraction=float(data_config.get("val_fraction", 0.05)),
        seed=int(data_config.get("split_seed", 42)),
        split_file=data_config.get("split_file"),
    )
    train_spec = CILDepthViewSpec.from_config(data_config, "train")
    eval_spec = CILDepthViewSpec.from_config(data_config, "eval")
    train_augmentation = depth_augmentation_from_config(augmentation_config)
    teacher_mask_validation = None
    if teacher_mask_spec is not None:
        teacher_mask_validation = validate_teacher_mask_spec(
            spec=teacher_mask_spec,
            data_dir=data_dir,
            train_names=train_names,
            train_view_spec=train_spec,
        )
    teacher_mask = teacher_mask_spec if teacher_mask_spec is not None else teacher_mask_dir

    train_dataset = CILDepthDataset(
        data_dir=data_dir,
        filenames=train_names,
        spec=train_spec,
        augmentation=train_augmentation,
        teacher_mask=teacher_mask,
    )
    val_dataset = CILDepthDataset(
        data_dir=data_dir,
        filenames=val_names,
        spec=eval_spec,
    )

    batch_size = int(train_config.get("batch_size", 1))
    num_workers = int(train_config.get("num_workers", 0))
    eval_batch_size = 1 if eval_spec.output_grid == "native_gt" else batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return CILDepthDataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_names=train_names,
        val_names=val_names,
        train_spec=train_spec,
        eval_spec=eval_spec,
        teacher_mask_validation=teacher_mask_validation,
    )
