#!/bin/bash
#SBATCH -A cil_jobs
#SBATCH --qos=heelee-cil_jobs
#SBATCH --gpus=2080ti:1
#SBATCH -t 04:00:00
#SBATCH --job-name=da2_unet_feats
#SBATCH -o /home/heelee/logs/da2_unet_feats_%j.out
#SBATCH -e /home/heelee/logs/da2_unet_feats_%j.err

mkdir -p /home/heelee/logs /home/heelee/checkpoints/cil_unet_feats

PYTHON=/work/courses/3dv/team24/conda_envs/video-depth-anything/bin/python
SAVE=/home/heelee/checkpoints/cil_unet_feats

RESUME_FLAG=""
if [ -f "$SAVE/latest.pth" ]; then
    RESUME_FLAG="--resume $SAVE/latest.pth"
    echo "Resuming from $SAVE/latest.pth"
fi

$PYTHON /home/heelee/train_unet_feats.py \
    --data-dir     /cluster/courses/cil/monocular-depth-estimation/train \
    --feats-dir    /work/scratch/heelee/da2_features \
    --img-size     518 \
    --epochs       10 \
    --bs           16 \
    --lr           1e-4 \
    --save-path    $SAVE \
    --val-fraction 0.05 \
    --num-workers  2 \
    $RESUME_FLAG
