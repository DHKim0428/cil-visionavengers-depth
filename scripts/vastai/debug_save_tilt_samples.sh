#!/bin/bash

REPO_DIR="${REPO_DIR:-/workspace/cil-visionavengers-depth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/workspace/cil-visionavengers-depth}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/monocular-depth-estimation/train}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/debug/tilt_geometry_samples}"
SEED="${SEED:-42}"

mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

ARGS=(
    scripts/debug_save_tilt_samples.py
    --data_root "$DATA_ROOT"
    --output_dir "$OUTPUT_DIR"
)

python "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/debug_save_tilt_samples.log"