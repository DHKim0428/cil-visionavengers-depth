#!/bin/bash
#SBATCH --job-name=da3-mask-precompute
#SBATCH --account=cil_jobs
#SBATCH --time=04:00:00
#SBATCH --gpus 5060ti:1
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Precompute DA3-guided reliability masks for training labels. This script does
# not install DA3; provide either an environment where depth_anything_3 is
# importable or DA3_REPO=/path/to/Depth-Anything-3.

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
DATA_ROOT="${DATA_ROOT:-/cluster/courses/cil/monocular-depth-estimation/train}"
SPLIT_FILE="${SPLIT_FILE:-$SCRATCH_ROOT/splits/unet_seed42.json}"
IMG_SIZE="${IMG_SIZE:-128}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
THRESHOLD_LABEL="${THRESHOLD_PERCENTILE//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/teacher_masks/da3_giant_p${THRESHOLD_LABEL}_img${IMG_SIZE}_seed42}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3-GIANT-1.1}"
HF_HOME="${HF_HOME:-/work/scratch/$USER/hf_cache}"
PROCESS_RES="${PROCESS_RES:-504}"

export HF_HOME
mkdir -p "$OUTPUT_DIR" "$HF_HOME"
cd "$REPO_DIR"

ARGS=(
    scripts/precompute_da3_reliability_masks.py
    --data_root "$DATA_ROOT"
    --split_file "$SPLIT_FILE"
    --output_dir "$OUTPUT_DIR"
    --img_size "$IMG_SIZE"
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
