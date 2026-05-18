# DA3 Teacher-Mask Filtering

This directory contains offline utilities for DA3-guided data cleaning. DA3 is
used only before training to create binary teacher reliability masks. Training
should consume the saved masks and metadata, not import DA3.

## Setup

From the repository root:

```bash
bash scripts/setup_da3.sh
```

This prepares a local DA3 checkout under `external/Depth-Anything-3`, installs it
in editable mode, and uses a Hugging Face cache under scratch. If DA3 dependencies are missing in the active
environment, install them in the DA3/precompute environment rather than adding
DA3 imports to the training loop.

## Recommended DA2 Mask Grid

For DA2 fine-tuning, use raw-depth masks:

```bash
sbatch --export=ALL,\
GRID=raw_depth,\
SPLIT_FILE=configs/splits/cil_depth_val_05pct_seed42.json,\
DATA_ROOT=/cluster/courses/cil/monocular-depth-estimation/train \
scripts/slurm/precompute_da3_reliability_masks.sbatch
```

Raw-depth masks are aligned with the original depth files and can later follow
the same DA2 resize/crop path as RGB, depth, and the dataset-valid mask.

## Artifact Contract

The precompute output directory contains:

- `{stem}_teacher_mask.png`: `255` means keep the GT pixel for training loss;
  `0` means ignore it.
- `metadata.json`: mask grid, split file, threshold, DA3 model, process
  resolution, and count metadata.
- `summary.json`: aggregate filtering statistics.
- `per_sample.csv`: per-image filtering statistics.

Validation masks are not generated or consumed. The precompute script reads
`train_names` from the split file, and training should apply teacher masks only
to the train loader.

## Config Shape

Use a `supervision.teacher_mask` section, not `augmentation` or `model`:

```yaml
supervision:
  teacher_mask:
    enabled: true
    dir: /work/scratch/$USER/cil-visionavengers-depth/teacher_masks/da3_giant_p95_raw_depth_seed42
    grid: raw_depth
    threshold_percentile: 95.0
    model_dir: depth-anything/DA3-GIANT-1.1
    process_res: 504
```

This is supervision filtering: it narrows the valid GT pixels used by the loss.
It is not DA3 pseudo-label training and not data augmentation.
