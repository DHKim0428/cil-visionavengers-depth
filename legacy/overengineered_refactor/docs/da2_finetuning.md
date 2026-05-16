# DA2 fine-tuning

This document describes Depth Anything V2 fine-tuning through the canonical
unified training entrypoint introduced by R10.

## Entrypoint

Use the unified `scripts/train.py` entrypoint from the repository root:

```bash
module load cuda/12.6.0
conda_cil
python scripts/train.py --config configs/experiments/da2_vits_refinenets_output.yaml
```

If `--config` is omitted, the trainer defaults to:

```text
configs/experiments/da2_vits_refinenets_output.yaml
```

This default is intentionally lightweight: it uses `vits` and the
`refinenets_output` fine-tuning strategy. Transition-era DA2-only wrappers now
live under `legacy/da2/transition/`; new runs should use `scripts/train.py`.

## What the trainer consumes

The trainer reads the experiment YAML first, then applies a small number of CLI
overrides intended for paths and smoke/debug runs.  The config remains the source
of truth for experiment identity and behavior.

Currently supported config-driven choices include:

- DA2 encoder: `vits`, `vitb`, `vitl`;
- tuning mode: `base`, `lora`, `mixed`;
- base DA2 trainable scope: `frozen`, `full`, `decoder`, `refinenets_output`;
- data pipeline: `legacy_square`, `dpt_native`;
- augmentation preset: `none`, `basic`, `tilt_geometry`;
- scheduler: `constant`, `poly_decay`;
- optimizer: `adamw`;
- W&B logging destination.

## Useful CLI overrides

```bash
python scripts/train.py \
  --config configs/experiments/da2_vits_refinenets_output.yaml \
  --run-name my_debug_run \
  --data-root /cluster/courses/cil/monocular-depth-estimation/train \
  --output-root /work/scratch/$USER/cil-visionavengers-depth/checkpoints \
  --checkpoint /work/scratch/$USER/cil-visionavengers-depth/models/da2/depth_anything_v2_vits.pth
```

Debug/smoke overrides:

```bash
python scripts/train.py \
  --img-size 56 \
  --batch-size 1 \
  --num-workers 0 \
  --max-samples 2 \
  --epochs 1 \
  --run-name phase4_tiny_smoke
```

Dry-run config/path resolution only:

```bash
python scripts/train.py --dry-run
```

## W&B behavior

The canonical trainer expects W&B by default and uses the shared team location:

```text
entity: cil-visionavengers
project: cil-visionavengers-depth
```

Run this once before training:

```bash
wandb login
```

If no W&B login is detected, the trainer emits a startup warning and continues
with local logging/checkpoint saving rather than silently pretending the run is
tracked.

## Outputs

Each run writes to:

```text
$output_root/$run_name/$timestamp/
```

The directory contains:

- `effective_config.yaml`;
- `latest.pth`;
- `best.pth` when validation improves;
- `summary.json`;
- local W&B files if W&B is enabled.

For LoRA/adapter runs, checkpoint saving uses a minimal-storage policy by
default: `best.pth` and `latest.pth` store only the trainable payload needed on
top of the configured base DA2 checkpoint, rather than always writing a full DA2
model state dict.  Full-model saving is an explicit opt-in policy for
experiments that truly need it.



## Augmentation presets

DA2 training now resolves `augmentation.preset` from `configs/augmentations/`
and applies the resulting paired augmentation only to the training split.
Validation and evaluation remain unaugmented.

The current presets are:

- `none`;
- `basic`;
- `tilt_geometry`.

The shared implementation lives in `dataset/augmentations.py`, so the DA2 path
reuses the same augmentation behavior as the existing U-Net data path rather than
copying another augmentation stack.  Teacher masks remain supervision inputs;
they are combined into the training valid mask before paired train-time warps so
RGB, depth, and valid pixels stay aligned.

## Adapter / LoRA support

LoRA is configured through the `adapter` section and is independent from
`base.trainable_scope`.  See [DA2 adapters and checkpoints](da2_adapters.md) for
LoRA target modes, config examples, and the trainable-only checkpoint format.

## Related evaluation entrypoint

Use `scripts/eval.py` for canonical DA2 evaluation.  See
[DA2 evaluation](da2_evaluation.md) for the explicit `native_resolution`,
`legacy_square`, and `raw_infer_native` protocols.


## Historical entrypoints

Older DA2 training scripts were moved to `legacy/da2/` in Phase 8 after their
useful behavior was absorbed into this canonical trainer.  They are preserved for
reference only and should not be used for new experiments.
