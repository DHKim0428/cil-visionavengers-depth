#!/bin/bash
#SBATCH --job-name=depth-eval
#SBATCH --account=cil
#SBATCH --time=01:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Canonical unified depth-evaluation wrapper.
# Examples:
#   sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,RUN_NAME=r5_vits_zero_shot scripts/slurm/eval.sbatch
#   sbatch --export=ALL,CONFIG=configs/experiments/unet_baseline_eval.yaml,CHECKPOINT=/work/scratch/$USER/cil-visionavengers-depth/checkpoints/unet_epoch_10.pth,RUN_NAME=r5_unet_baseline scripts/slurm/eval.sbatch

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
CONFIG="${CONFIG:-configs/experiments/da2_vits_refinenets_output.yaml}"
RUN_NAME="${RUN_NAME:-depth_eval_${SLURM_JOB_ID:-local}}"
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
if [[ -n "${PROTOCOL:-}" ]]; then
    ARGS+=(--protocol "$PROTOCOL")
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
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${FRACTION:-}" ]]; then
    ARGS+=(--fraction "$FRACTION")
fi
if [[ "${NO_WANDB:-0}" == "1" ]]; then
    ARGS+=(--no-wandb)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi

echo "[eval.sbatch] repo=$REPO_DIR"
echo "[eval.sbatch] config=$CONFIG"
echo "[eval.sbatch] run=$RUN_NAME"
echo "[eval.sbatch] protocol=${PROTOCOL:-config-default}"
echo "[eval.sbatch] checkpoint=${CHECKPOINT:-config/default}"
echo "[eval.sbatch] data_root=$DATA_ROOT"
echo "[eval.sbatch] command: python eval.py ${ARGS[*]}"

python eval.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
