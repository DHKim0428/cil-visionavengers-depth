#!/bin/bash
#SBATCH --job-name=unet-basic-aug
#SBATCH --account=cil_jobs
#SBATCH --time=02:00:00
#SBATCH --gpus 2080ti:1
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Full U-Net training with only standard paired augmentations enabled. This is a
# cleaner comparison point before mixing in geometry-aware tilt augmentation.

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
RUN_NAME="${RUN_NAME:-unet_basic_aug_${SLURM_JOB_ID:-local}}"

mkdir -p "$SCRATCH_ROOT/checkpoints/$RUN_NAME" "$SCRATCH_ROOT/logs" "$SCRATCH_ROOT/splits"
cd "$REPO_DIR"

python train.py \
    --data_root "$DATA_ROOT" \
    --save_dir "$SCRATCH_ROOT/checkpoints/$RUN_NAME" \
    --split_file "$SCRATCH_ROOT/splits/unet_seed42.json" \
    --split_seed 42 \
    --img_size 128 \
    --batch_size 8 \
    --num_epochs "${NUM_EPOCHS:-10}" \
    --lr 1e-3 \
    --val_split 0.20 \
    --enable_hflip \
    --enable_rotation \
    --rotation_deg "${ROTATION_DEG:-3}" \
    --enable_crop \
    --crop_scale_min "${CROP_SCALE_MIN:-0.9}" \
    --enable_color_jitter \
    --color_jitter_prob "${COLOR_JITTER_PROB:-0.8}" \
    --brightness "${BRIGHTNESS:-0.1}" \
    --contrast "${CONTRAST:-0.1}" \
    --saturation "${SATURATION:-0.1}" \
    --tilt_mode none \
    2>&1 | tee "$SCRATCH_ROOT/logs/$RUN_NAME.log"
