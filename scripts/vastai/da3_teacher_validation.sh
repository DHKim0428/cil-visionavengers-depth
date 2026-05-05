#!/bin/bash

REPO_DIR="${REPO_DIR:-/workspace/cil-visionavengers-depth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/workspace/cil-visionavengers-depth}"
INPUT_DIR="${INPUT_DIR:-$SCRATCH_ROOT/debug/tilt_geometry_samples}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/debug/da3_teacher_validation}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3-GIANT-1.1}"
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
PROCESS_RES="${PROCESS_RES:-504}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

export HF_HOME
mkdir -p "$OUTPUT_DIR" "$HF_HOME"
cd "$REPO_DIR"

ARGS=(
    scripts/da3_teacher_validate_tilt.py
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_dir "$MODEL_DIR"
    --max_samples "$MAX_SAMPLES"
    --process_res "$PROCESS_RES"
)

if [[ -n "${DA3_REPO:-}" ]]; then
    ARGS+=(--da3_repo "$DA3_REPO")
fi

if [[ "$SAVE_VISUALIZATIONS" != "0" ]]; then
    ARGS+=(--save_visualizations)
fi

if [[ "$SAVE_PREDICTIONS" != "0" ]]; then
    ARGS+=(--save_predictions)
fi

python "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/da3_teacher_validation.log"
