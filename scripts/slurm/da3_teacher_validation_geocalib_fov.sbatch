#!/bin/bash
#SBATCH --job-name=da3-geocalib-fov
#SBATCH --account=cil_jobs
#SBATCH --time=02:30:00
#SBATCH --gpus 5060ti:1
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Generate fixed-FOV plus GeoCalib adaptive-FOV tilt debug samples, then compare
# them against DA3. Requires a precomputed GeoCalib FOV CSV by default at
# $SCRATCH_ROOT/geocalib_fov/geocalib_fov.csv.

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
FOV_TABLE="${FOV_TABLE:-$SCRATCH_ROOT/geocalib_fov/geocalib_fov.csv}"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-$SCRATCH_ROOT/debug_tilt_geocalib_fov_${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/da3_teacher_geocalib_fov_${SLURM_JOB_ID:-local}}"
MODEL_DIR="${MODEL_DIR:-depth-anything/DA3-GIANT-1.1}"
HF_HOME="${HF_HOME:-/work/scratch/$USER/hf_cache}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
PROCESS_RES="${PROCESS_RES:-504}"
TARGET_VARIANTS="${TARGET_VARIANTS:-naive geo_fov50 geo_fov60 geo_fov70 geo_geocalib}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

export HF_HOME
mkdir -p "$DEBUG_OUTPUT_DIR" "$OUTPUT_DIR" "$HF_HOME"
cd "$REPO_DIR"

if [[ ! -f "$FOV_TABLE" ]]; then
    FOV_OUTPUT_DIR="$(dirname "$FOV_TABLE")"
    mkdir -p "$FOV_OUTPUT_DIR"
    GEOCALIB_ARGS=(
        scripts/precompute_geocalib_fov.py
        --data_root "$DATA_ROOT"
        --output_dir "$FOV_OUTPUT_DIR"
        --max_samples "$MAX_SAMPLES"
    )
    if [[ -n "${GEOCALIB_REPO:-}" ]]; then
        GEOCALIB_ARGS+=(--geocalib_repo "$GEOCALIB_REPO")
    fi
    python "${GEOCALIB_ARGS[@]}" 2>&1 | tee "$FOV_OUTPUT_DIR/precompute_geocalib_fov.log"
fi

python scripts/debug_save_tilt_samples.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$DEBUG_OUTPUT_DIR" \
    --num_samples "$MAX_SAMPLES" \
    --save_naive \
    --fov_variants 50 60 70 \
    --fov_table "$FOV_TABLE" \
    2>&1 | tee "$DEBUG_OUTPUT_DIR/debug_save_tilt_samples.log"

ARGS=(
    scripts/da3_teacher_validate_tilt.py
    --input_dir "$DEBUG_OUTPUT_DIR"
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
