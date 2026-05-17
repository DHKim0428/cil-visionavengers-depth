#!/bin/bash
#SBATCH -A cil_jobs
#SBATCH --qos=heelee-cil_jobs
#SBATCH --gpus=2080ti:1
#SBATCH -t 02:00:00
#SBATCH --job-name=precompute_da2_feats
#SBATCH -o /home/heelee/logs/precompute_da2_feats_%j.out
#SBATCH -e /home/heelee/logs/precompute_da2_feats_%j.err

mkdir -p /home/heelee/logs /work/scratch/heelee/da2_features

PYTHON=/work/courses/3dv/team24/conda_envs/video-depth-anything/bin/python

$PYTHON /home/heelee/precompute_da2_features.py \
    --data-dir   /cluster/courses/cil/monocular-depth-estimation/train \
    --output-dir /work/scratch/heelee/da2_features \
    --checkpoint /home/heelee/checkpoints/depth_anything_v2_vits.pth \
    --img-size   518 \
    --bs         16
