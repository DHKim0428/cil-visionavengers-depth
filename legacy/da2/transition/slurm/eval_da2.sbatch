#!/bin/bash
#SBATCH --job-name=da2-eval
#SBATCH --account=cil
#SBATCH --time=01:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Generic canonical DA2 evaluation wrapper.
# Override behavior with --export, for example:
#   sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,PROTOCOL=raw_infer_native,MAX_SAMPLES=256,RUN_NAME=phase9a_eval scripts/slurm/eval_da2.sbatch

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
RUN_NAME="${RUN_NAME:-da2_eval_${SLURM_JOB_ID:-local}}"
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
if [[ -n "${BATCH_SIZE:-}" ]]; then
    ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
    ARGS+=(--num-workers "$NUM_WORKERS")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${FRACTION:-}" ]]; then
    ARGS+=(--fraction "$FRACTION")
fi
if [[ -n "${VIS_DIR:-}" ]]; then
    ARGS+=(--vis-dir "$VIS_DIR")
fi
if [[ -n "${NUM_VIS:-}" ]]; then
    ARGS+=(--num-vis "$NUM_VIS")
fi
if [[ "${NO_WANDB:-0}" == "1" ]]; then
    ARGS+=(--no-wandb)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi

echo "[eval_da2.sbatch] repo=$REPO_DIR"
echo "[eval_da2.sbatch] config=$CONFIG"
echo "[eval_da2.sbatch] run=$RUN_NAME"
echo "[eval_da2.sbatch] protocol=${PROTOCOL:-config-default}"
echo "[eval_da2.sbatch] data_root=$DATA_ROOT"
echo "[eval_da2.sbatch] command: python legacy/da2/transition/eval_da2.py ${ARGS[*]}"

python legacy/da2/transition/eval_da2.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
