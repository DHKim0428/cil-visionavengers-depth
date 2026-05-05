#!/bin/bash
#SBATCH --job-name=depth_anything
#SBATCH --account=cil
#SBATCH --time=00:30:00
#SBATCH --gpus=2080ti:1
#SBATCH --output=/home/dchileban/cil/logs/%j.out
#SBATCH --error=/home/dchileban/cil/logs/%j.err

cd /home/dchileban/cil/cil-visionavengers-depth/fine-tune

conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation


echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_depth_anything_sirmse.py \
  --epochs 5 \
  --batch-size 1 \
  --accum-steps 4 \
  --num-workers 2 \
  --lr 2e-6 \
  --input-size 392 \
  --amp \
  --output-dir /home/dchileban/cil/logs/da_v2_sirmse \
  --log-dir /home/dchileban/cil/logs/da_v2_sirmse \
  --log-images-every 1 \
  --num-log-images 4