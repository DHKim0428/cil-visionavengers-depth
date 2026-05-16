#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.cil_depth import CILDepthViewSpec
from dataset.raw_cil import discover_rgb_filenames
from dataset.supervision import TeacherMaskSpec, validate_teacher_mask_spec
from training.config import DEFAULT_DA2_CONFIG, load_yaml_config
from training.splits import make_or_load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a teacher-mask artifact against a canonical CIL train view.")
    parser.add_argument("--config", type=str, default=DEFAULT_DA2_CONFIG, help="Experiment config with supervision.teacher_mask enabled")
    parser.add_argument("--shape-check-limit", type=int, default=32, help="How many masks to open for shape checks; use 0 for all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    spec = TeacherMaskSpec.from_config(config.get("supervision", {}).get("teacher_mask", {}))
    if spec is None:
        raise ValueError("Config must enable supervision.teacher_mask for validation")

    data_cfg = config["data"]
    names = discover_rgb_filenames(data_cfg["root"], max_samples=data_cfg.get("max_samples"))
    train_names, _ = make_or_load_split(
        sample_names=names,
        val_fraction=float(data_cfg.get("val_fraction", 0.05)),
        seed=int(data_cfg.get("split_seed", 42)),
        split_file=data_cfg.get("split_file"),
    )
    train_spec = CILDepthViewSpec.from_config(data_cfg, "train")
    shape_limit = None if args.shape_check_limit == 0 else args.shape_check_limit
    result = validate_teacher_mask_spec(
        spec=spec,
        data_dir=data_cfg["root"],
        train_names=train_names,
        train_view_spec=train_spec,
        shape_check_limit=shape_limit,
    )
    print(json.dumps({"artifact": spec.export_metadata(), "validation": result}, indent=2))


if __name__ == "__main__":
    main()
