#!/bin/bash
#SBATCH --job-name=depth-predict
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
TEST_ROOT="${TEST_ROOT:-/cluster/courses/cil/monocular-depth-estimation/test}"
CONFIG="${CONFIG:-configs/experiments/da2_vits_refinenets_output.yaml}"
RUN_NAME="${RUN_NAME:-depth_predict_${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/predictions/$RUN_NAME/preds}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$REPO_DIR"

ARGS=(
    --config "$CONFIG"
    --run-name "$RUN_NAME"
    --test-root "$TEST_ROOT"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${MODEL:-}" ]]; then ARGS+=(--model "$MODEL"); fi
if [[ -n "${SUBMISSION_CSV:-}" ]]; then ARGS+=(--submission-csv "$SUBMISSION_CSV"); fi
if [[ -n "${IMG_SIZE:-}" ]]; then ARGS+=(--img-size "$IMG_SIZE"); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then ARGS+=(--max-samples "$MAX_SAMPLES"); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then ARGS+=(--dry-run); fi

echo "[predict.sbatch] repo=$REPO_DIR"
echo "[predict.sbatch] config=$CONFIG"
echo "[predict.sbatch] run=$RUN_NAME"
echo "[predict.sbatch] checkpoint=${CHECKPOINT:-config/default}"
echo "[predict.sbatch] test_root=$TEST_ROOT"
echo "[predict.sbatch] output_dir=$OUTPUT_DIR"
echo "[predict.sbatch] submission_csv=${SUBMISSION_CSV:-none}"
echo "[predict.sbatch] command: python predict.py ${ARGS[*]}"

python predict.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
