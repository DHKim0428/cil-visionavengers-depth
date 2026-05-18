# Scripts

This directory is for shell wrappers only.

- `setup_da2.sh`: clone/download DA2 assets.
- `setup_da3.sh`: clone DA3 assets for offline teacher-mask precomputation.
- `slurm/*.sbatch`: cluster wrappers.
- `vastai/*.sh`: Vast.ai wrappers for dataset/DA3 preprocessing experiments.

Python entrypoints live at the repository root:

```bash
python train.py --config configs/experiments/da2_vits_refinenets_output.yaml
python eval.py --config configs/experiments/da2_vits_zero_shot.yaml
```

Python dataset-processing utilities live under `dataset/`.
