#!/bin/bash
#SBATCH -A cil_jobs
#SBATCH --qos=heelee-cil_jobs
#SBATCH --gpus=2080ti:1
#SBATCH -t 04:00:00
#SBATCH --job-name=decoder_ft_vits
#SBATCH -o /home/heelee/logs/decoder_ft_vits_%j.out
#SBATCH -e /home/heelee/logs/decoder_ft_vits_%j.err

mkdir -p /home/heelee/logs /home/heelee/checkpoints/cil_decoder_finetune

PYTHON=/work/courses/3dv/team24/conda_envs/video-depth-anything/bin/python

$PYTHON /home/heelee/train_cil.py \
    --encoder      vits \
    --strategy     decoder \
    --data-dir     /cluster/courses/cil/monocular-depth-estimation/train \
    --img-size     518 \
    --epochs       10 \
    --bs           8 \
    --lr           1e-4 \
    --save-path    /home/heelee/checkpoints/cil_decoder_finetune \
    --resume       /home/heelee/checkpoints/cil_decoder_finetune/latest.pth \
    --val-fraction 0.05
