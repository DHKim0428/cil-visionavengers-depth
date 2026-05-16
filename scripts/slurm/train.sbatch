#!/bin/bash
#SBATCH --job-name=depth-train
#SBATCH --account=cil_jobs
#SBATCH --time=04:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=24G
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
CONFIG="${CONFIG:-configs/experiments/da2_vits_refinenets_output.yaml}"
RUN_NAME="${RUN_NAME:-depth_train_${SLURM_JOB_ID:-local}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/logs}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$SCRATCH_ROOT/splits"
cd "$REPO_DIR"

ARGS=(
    --config "$CONFIG"
    --run-name "$RUN_NAME"
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
)

if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi
if [[ -n "${EPOCHS:-}" ]]; then ARGS+=(--epochs "$EPOCHS"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "${IMG_SIZE:-}" ]]; then ARGS+=(--img-size "$IMG_SIZE"); fi
if [[ -n "${NUM_WORKERS:-}" ]]; then ARGS+=(--num-workers "$NUM_WORKERS"); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then ARGS+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "${SCHEDULER:-}" ]]; then ARGS+=(--scheduler "$SCHEDULER"); fi
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then ARGS+=(--grad-accum-steps "$GRAD_ACCUM_STEPS"); fi
if [[ -n "${LOG_EVERY:-}" ]]; then ARGS+=(--log-every "$LOG_EVERY"); fi
if [[ -n "${SAVE_POLICY:-}" ]]; then ARGS+=(--save-policy "$SAVE_POLICY"); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then ARGS+=(--dry-run); fi

echo "[train.sbatch] repo=$REPO_DIR"
echo "[train.sbatch] config=$CONFIG"
echo "[train.sbatch] run=$RUN_NAME"
echo "[train.sbatch] output_root=$OUTPUT_ROOT"
echo "[train.sbatch] data_root=$DATA_ROOT"
echo "[train.sbatch] command: python train.py ${ARGS[*]}"
python train.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
