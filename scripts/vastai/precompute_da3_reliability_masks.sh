#!/bin/bash

REPO_DIR="${REPO_DIR:-/workspace/cil-visionavengers-depth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/workspace/cil-visionavengers-depth}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/monocular-depth-estimation/train}"
SPLIT_FILE="${SPLIT_FILE:-$SCRATCH_ROOT/splits/unet_seed42.json}"
GRID="${GRID:-square}"
MASK_SIZE="${MASK_SIZE:-128}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
THRESHOLD_LABEL="${THRESHOLD_PERCENTILE//./p}"
if [[ "$GRID" == "square" ]]; then
    MASK_GRID_LABEL="square${MASK_SIZE}"
else
    MASK_GRID_LABEL="raw_depth"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/teacher_masks/da3_giant_p${THRESHOLD_LABEL}_${MASK_GRID_LABEL}_seed42}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3-GIANT-1.1}"
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
PROCESS_RES="${PROCESS_RES:-504}"

export HF_HOME
mkdir -p "$OUTPUT_DIR" "$HF_HOME"
cd "$REPO_DIR"

ARGS=(
    dataset/da3/precompute_da3_reliability_masks.py
    --data_root "$DATA_ROOT"
    --split_file "$SPLIT_FILE"
    --output_dir "$OUTPUT_DIR"
    --grid "$GRID"
    --mask_size "$MASK_SIZE"
    --threshold_percentile "$THRESHOLD_PERCENTILE"
    --model_dir "$MODEL_DIR"
    --process_res "$PROCESS_RES"
)

if [[ -n "${DA3_REPO:-}" ]]; then
    ARGS+=(--da3_repo "$DA3_REPO")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/precompute_da3_reliability_masks.log"
