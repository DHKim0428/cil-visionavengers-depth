#!/bin/bash
#SBATCH --job-name=da3-teacher-val
#SBATCH --account=cil_jobs
#SBATCH --time=02:00:00
#SBATCH --gpus 5060ti:1
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Diagnostic run: compare DA3 relative-depth predictions against geometry-tilt
# debug targets. This script does not install DA3; provide either an environment
# where depth_anything_3 is importable or DA3_REPO=/path/to/Depth-Anything-3.

source ~/.bashrc || true
module load cuda/12.6.0 || true

if [[ -n "${DA3_CONDA_ENV:-}" ]]; then
    eval "$(/cluster/courses/cil/envs/bin/conda shell.bash hook 2> /dev/null)"
    conda activate "$DA3_CONDA_ENV"
elif declare -F conda_da3 >/dev/null; then
    conda_da3
elif declare -F conda_cil >/dev/null; then
    conda_cil
else
    eval "$(/cluster/courses/cil/envs/bin/conda shell.bash hook 2> /dev/null)"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
    if [[ "$GPU_NAME" == *"5060"* ]]; then
        conda activate /cluster/courses/cil/envs/envs/monodepth-5060
    else
        conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
    fi
fi

REPO_DIR="${REPO_DIR:-$HOME/workspace/cil-visionavengers-depth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/$USER/cil-visionavengers-depth}"
INPUT_DIR="${INPUT_DIR:-$SCRATCH_ROOT/debug_tilt_geometry_samples_rgb_unmasked}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/da3_teacher_validation_${SLURM_JOB_ID:-local}}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3-GIANT-1.1}"
HF_HOME="${HF_HOME:-/work/scratch/$USER/hf_cache}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
PROCESS_RES="${PROCESS_RES:-504}"
TARGET_VARIANTS="${TARGET_VARIANTS:-geo}"
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

read -r -a TARGET_VARIANT_ARGS <<< "$TARGET_VARIANTS"
ARGS+=(--target_variants "${TARGET_VARIANT_ARGS[@]}")

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
