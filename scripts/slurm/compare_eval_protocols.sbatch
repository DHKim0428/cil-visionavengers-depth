#!/bin/bash
#SBATCH --job-name=depth-protocol-compare
#SBATCH --account=cil
#SBATCH --time=01:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

source ~/.bashrc || true
module load cuda/12.6.0 || true

if declare -F conda_cil >/dev/null; then
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
CONFIG="${CONFIG:-configs/experiments/da2_vits_zero_shot.yaml}"
RUN_NAME="${RUN_NAME:-protocol_compare_${SLURM_JOB_ID:-local}}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/logs}"

mkdir -p "$LOG_DIR" "$SCRATCH_ROOT/evaluations"
cd "$REPO_DIR"

ARGS=(
    --config "$CONFIG"
    --run-name "$RUN_NAME"
    --data-root "$DATA_ROOT"
)

if [[ -n "${CHECKPOINT:-}" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${IMG_SIZE:-}" ]]; then
    ARGS+=(--img-size "$IMG_SIZE")
fi
if [[ -n "${SPLIT_FILE:-}" ]]; then
    ARGS+=(--split-file "$SPLIT_FILE")
fi
if [[ -n "${VAL_FRACTION:-}" ]]; then
    ARGS+=(--val-fraction "$VAL_FRACTION")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${FRACTION:-}" ]]; then
    ARGS+=(--fraction "$FRACTION")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi

echo "[compare_eval_protocols.sbatch] repo=$REPO_DIR"
echo "[compare_eval_protocols.sbatch] config=$CONFIG"
echo "[compare_eval_protocols.sbatch] run=$RUN_NAME"
echo "[compare_eval_protocols.sbatch] checkpoint=${CHECKPOINT:-config/default}"
echo "[compare_eval_protocols.sbatch] data_root=$DATA_ROOT"
echo "[compare_eval_protocols.sbatch] command: python compare_eval_protocols.py ${ARGS[*]}"

python compare_eval_protocols.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
