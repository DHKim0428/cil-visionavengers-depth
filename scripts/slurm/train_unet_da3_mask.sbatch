#!/bin/bash
#SBATCH --job-name=unet-da3-mask
#SBATCH --account=cil_jobs
#SBATCH --time=02:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Baseline U-Net training with DA3-guided reliability masks applied only to the
# training loss. Validation keeps the original dataset mask.
#
# Override MASK_PERCENTILE=95, 97.5, or 99 to choose the mask directory.

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
SPLIT_FILE="${SPLIT_FILE:-$SCRATCH_ROOT/splits/unet_seed42.json}"
MASK_PERCENTILE="${MASK_PERCENTILE:-95}"
MASK_LABEL="${MASK_PERCENTILE//./p}"
TEACHER_MASK_DIR="${TEACHER_MASK_DIR:-$SCRATCH_ROOT/teacher_masks/da3_giant_p${MASK_LABEL}_img128_seed42}"
RUN_NAME="${RUN_NAME:-unet_da3_mask_p${MASK_LABEL}_${SLURM_JOB_ID:-local}}"

mkdir -p "$SCRATCH_ROOT/checkpoints/$RUN_NAME" "$SCRATCH_ROOT/logs" "$SCRATCH_ROOT/splits"
cd "$REPO_DIR"

python train.py \
    --data_root "$DATA_ROOT" \
    --save_dir "$SCRATCH_ROOT/checkpoints/$RUN_NAME" \
    --split_file "$SPLIT_FILE" \
    --split_seed 42 \
    --img_size 128 \
    --batch_size 8 \
    --num_epochs "${NUM_EPOCHS:-10}" \
    --lr 1e-3 \
    --val_split 0.20 \
    --tilt_mode none \
    --teacher_mask_dir "$TEACHER_MASK_DIR" \
    2>&1 | tee "$SCRATCH_ROOT/logs/$RUN_NAME.log"
