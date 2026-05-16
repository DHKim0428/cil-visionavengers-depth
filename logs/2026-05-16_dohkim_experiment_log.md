# Dohkim Experiment Log

Owner: dohkim

## U-Net Baselines, Shared 5% Validation Split

Validation split:
- File: `configs/splits/cil_depth_val_05pct_seed42.json`
- Train samples: 21475
- Val samples: 1130
- Metric: validation siRMSE

| Date | Run name | Command | Augmentation | Status | Best local result | Best checkpoint | Kaggle report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-16 | `unet_baseline_noaug_10ep` | `sbatch --export=ALL,CONFIG=configs/experiments/unet_baseline.yaml,RUN_NAME=unet_baseline_noaug_10ep scripts/slurm/train.sbatch` | none | running, job `75190`; last parsed epoch `8/10` | `val_siRMSE=0.4888` at epoch 8 | `/work/scratch/dohkim/cil-visionavengers-depth/checkpoints/unet_baseline_noaug_10ep/20260516_184941/best.pth` | TBD |
| TBD | `unet_baseline_aug_10ep` | TBD after adding/selecting an augmentation config | TBD | planned | TBD | TBD | TBD |

Notes:
- `logs/unet_baseline.log` is an old raw training stdout log with tqdm progress output, so it is expected to be very long.
- Keep Kaggle numbers here as the submitted leaderboard/report score, not local validation.
- Update the no-augmentation row after job `75190` completes if epochs 9 or 10 improve the best validation result.
