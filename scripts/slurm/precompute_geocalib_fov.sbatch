#!/bin/bash
#SBATCH --job-name=geocalib-fov
#SBATCH --account=cil_jobs
#SBATCH --time=02:00:00
#SBATCH --gpus 5060ti:1
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Precompute per-image GeoCalib horizontal FOV estimates for adaptive-FOV
# geometry tilt diagnostics. Provide GEOCALIB_REPO=/path/to/GeoCalib unless
# geocalib is already importable in the active environment.

source ~/.bashrc || true
module load cuda/12.6.0 || true

if [[ -n "${GEOCALIB_CONDA_ENV:-}" ]]; then
    eval "$(/cluster/courses/cil/envs/bin/conda shell.bash hook 2> /dev/null)"
    conda activate "$GEOCALIB_CONDA_ENV"
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
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/geocalib_fov}"
HF_HOME="${HF_HOME:-/work/scratch/$USER/hf_cache}"
CLIP_MIN_DEG="${CLIP_MIN_DEG:-35}"
CLIP_MAX_DEG="${CLIP_MAX_DEG:-95}"

export HF_HOME
mkdir -p "$OUTPUT_DIR" "$HF_HOME"
cd "$REPO_DIR"

ARGS=(
    scripts/precompute_geocalib_fov.py
    --data_root "$DATA_ROOT"
    --output_dir "$OUTPUT_DIR"
    --clip_min_deg "$CLIP_MIN_DEG"
    --clip_max_deg "$CLIP_MAX_DEG"
)

if [[ -n "${GEOCALIB_REPO:-}" ]]; then
    ARGS+=(--geocalib_repo "$GEOCALIB_REPO")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/precompute_geocalib_fov.log"
