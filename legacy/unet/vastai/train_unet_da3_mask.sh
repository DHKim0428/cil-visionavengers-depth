#!/bin/bash

REPO_DIR="${REPO_DIR:-/workspace/cil-visionavengers-depth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/workspace/cil-visionavengers-depth}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/monocular-depth-estimation/train}"
SPLIT_FILE="${SPLIT_FILE:-$SCRATCH_ROOT/splits/unet_seed42.json}"
MASK_PERCENTILE="${MASK_PERCENTILE:-95}"
MASK_LABEL="${MASK_PERCENTILE//./p}"
TEACHER_MASK_DIR="${TEACHER_MASK_DIR:-$SCRATCH_ROOT/teacher_masks/da3_giant_p${MASK_LABEL}_img128_seed42}"
RUN_NAME="${RUN_NAME:-unet_da3_mask_p${MASK_LABEL}_${SLURM_JOB_ID:-local}}"

mkdir -p "$SCRATCH_ROOT/checkpoints/$RUN_NAME" "$SCRATCH_ROOT/logs" "$SCRATCH_ROOT/splits"
cd "$REPO_DIR"

python train.py \
    --data_root "$DATA_ROOT" \
    --save_dir "$SCRATCH_ROOT/checkpoints/$RUN_NAME" \
    --split_file "$SPLIT_FILE" \
    --split_seed 42 \
    --img_size 128 \
    --batch_size 8 \
    --num_epochs "${NUM_EPOCHS:-10}" \
    --lr 1e-3 \
    --val_split 0.20 \
    --tilt_mode none \
    --teacher_mask_dir "$TEACHER_MASK_DIR" \
    2>&1 | tee "$SCRATCH_ROOT/logs/$RUN_NAME.log"