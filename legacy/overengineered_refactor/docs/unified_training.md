# Unified training

R10 introduced one canonical outer training entrypoint:

```bash
python scripts/train.py --config configs/experiments/<experiment>.yaml
```

The goal is one visible pipeline with model-specific behavior isolated behind a
small task interface, rather than separate monolithic scripts per model family.

## Structure

```text
scripts/train.py              # outer CLI / config overrides
training/runner.py            # shared orchestration
training/tasks/da2.py         # DA2-specific model/loss/checkpoint hooks
training/tasks/unet.py        # U-Net-specific model/loss hooks
```

The shared runner owns:

- config expansion and effective-config persistence;
- output directory creation, seeding, W&B setup;
- canonical CIL dataloader construction and teacher-mask validation;
- optimizer, scheduler, AMP, gradient accumulation;
- train/validation loop, latest/best checkpointing, summary files, image logs.

Each task owns only what genuinely differs:

- model construction;
- prediction semantics;
- prediction-to-loss / prediction-to-depth conversion;
- any family-specific model logging or checkpoint reconstruction metadata.

## Supported model families

### DA2

```yaml
model:
  family: da2_relative
```

DA2 predicts disparity and can combine base trainable scopes with LoRA adapters.
Storage-conscious trainable-only checkpoints still reconstruct from the base DA2
checkpoint plus the effective config.

### U-Net

Canonical U-Net training supports both prediction contracts:

```yaml
model:
  family: unet
  architecture: UNetBaseline
  prediction_kind: disparity   # disparity | depth
```

- `disparity`: predicts positive disparity, then uses the shared
  disparity-to-depth siRMSE path;
- `depth`: predicts positive metric depth directly, then uses the shared
  direct-depth siRMSE path.

The default canonical preset is `configs/experiments/unet_disparity.yaml`; the
sibling direct-depth preset is `configs/experiments/unet_metric_depth.yaml`.
Both use the same canonical dataloader, valid-mask policy, 5% saved split, W&B,
and full-model checkpoints.

## Transition notes

- Transition-era DA2-only wrappers now live under `legacy/da2/transition/`;
  the current outer trainer is only `scripts/train.py`.
- The original argparse-only U-Net trainer and its model definitions moved to
  `legacy/unet/` after the new path passed smoke validation.
- Old U-Net SLURM wrappers also moved under `legacy/unet/slurm/`; new jobs should
  use `scripts/slurm/train.sbatch` plus an experiment config.

## Validation completed in R10

- unified CLI dry-runs for DA2, U-Net disparity, U-Net depth, and the old DA2
  compatibility wrapper;
- synthetic 1-epoch train smoke for both U-Net prediction kinds;
- unified evaluator smoke on the resulting U-Net checkpoints, confirming that
  checkpoint metadata restores the correct prediction kind;
- unified SLURM wrapper dry-run.
