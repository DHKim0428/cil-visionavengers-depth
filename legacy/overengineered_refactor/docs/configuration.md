# Configuration design

This document records the first configuration layer introduced during the DA2
refactor.  It describes the config contract before existing training scripts are
rewired to consume it.

## Why configs are being added

The current repository stores experiment meaning in a mixture of Python defaults
and shell wrappers.  That makes it too easy for model choice, dataset pipeline,
evaluation protocol, and logging behavior to drift apart between experiments.
The new config layer keeps those decisions visible and reviewable.

## Current scope

Phase 2 introduced declarative config files. Phase 3 added executable DA2
data-layer support. Phase 4 introduced config-driven DA2 training, and R10 moved
canonical training behind one outer `scripts/train.py` entrypoint with model
family-specific tasks for DA2 and U-Net. Legacy entrypoints remain only under
`legacy/` or as thin transition wrappers.

## Canonical experiment config sections

| Section | Purpose |
|---|---|
| `experiment` | Human-facing run name and tags |
| `model` | Model family and family-specific choices such as DA2 encoder or U-Net prediction kind |
| `data` | Dataset root, train pipeline, eval protocol, image size, split policy |
| `augmentation` | Named preset reference |
| `supervision` | Teacher-mask or other supervision filtering policy |
| `train` | Optimizer, scheduler, precision, accumulation, worker defaults |
| `logging` | W&B team/project defaults and qualitative logging cadence |
| `paths` | Source/checkpoint/output locations used by the selected family |
| `adapter` | Optional LoRA/adapter configuration |
| `base` | DA2-only original-parameter trainable scope |
| `checkpoint` | Save policy such as trainable-only vs full-model checkpoint saving |

## Canonical data views

The canonical data path no longer treats `legacy_square` and `dpt_native` as the
main code-level abstraction.  Experiment configs describe behavior through
composable view knobs:

```yaml
data:
  image_size: 518
  views:
    train:
      resize_policy: dpt_lower_bound
      output_grid: model_input
      crop_size: image_size
      normalize: imagenet
    eval:
      resize_policy: dpt_lower_bound
      output_grid: native_gt
      crop_size: null
      normalize: imagenet
```

`dataset/cil_depth.py` implements one metric-depth `CILDepthDataset`, and
`dataset/factory.py` builds loaders from these fields.  Legacy-square, DA2
native, future U-Net square, and raw-native behavior are now configurations of
one system rather than separate canonical dataset classes.  Transition DA2
entrypoints preserved under `legacy/da2/transition/` carry their own compatibility
loader wrapper; new code should use `dataset/factory.py` directly.

### Evaluation protocol

- `legacy_square`
  - compare on the resized square validation grid;
- `native_resolution`
  - keep GT on its native grid and resize prediction back to GT size;
- `raw_infer_native`
  - preserve the older zero-shot `model.infer_image(...)` path while comparing
    prediction on the native GT grid.

### Augmentation presets

- `none`
- `basic`
- `tilt_geometry`

Experiment configs refer to these through:

```yaml
augmentation:
  preset: basic
```

The trainer resolves that preset from `configs/augmentations/<preset>.yaml` and
applies it only to the training dataset.  Validation/evaluation datasets remain
unaugmented.  Teacher masks remain conceptually outside augmentation under
`supervision.teacher_mask`; they are supervision inputs that are kept spatially
aligned with RGB/depth during train-time paired transforms.

### Teacher-mask contract

Canonical teacher-mask configs must declare the artifact grid explicitly:

```yaml
supervision:
  teacher_mask:
    enabled: true
    dir: /work/scratch/$USER/cil-visionavengers-depth/teacher_masks/da3_giant_p95_raw_depth_seed42
    grid: raw_depth              # raw_depth | square
    size: null                   # null for raw_depth; integer for square
    threshold_percentile: 95.0   # optional expected artifact metadata
    model_dir: depth-anything/DA3-GIANT-1.1
    process_res: 504
```

`raw_depth` is the preferred long-term contract because masks are stored on the
native depth grid and transformed together with RGB/depth before the loss.
`square` remains supported for current U-Net-style artifacts such as
`grid: square, size: 128`, but only with a matching square/model-input train
view.  DA2-native lower-bound views reject square masks instead of guessing how
to align them.

Mask directories must contain `metadata.json`; canonical validation checks the
artifact grid/size, train-split coverage, sampled mask shapes, threshold, DA3
model, process resolution, and saved-mask count.  Training runs attach both the
artifact metadata and validation result to `effective_config.yaml` and W&B.


### U-Net prediction contract

Canonical U-Net presets select output semantics explicitly:

```yaml
model:
  family: unet
  architecture: UNetBaseline
  prediction_kind: disparity   # disparity | depth
```

`disparity` uses the same disparity-to-depth siRMSE path as DA2; `depth` predicts
positive metric depth directly and uses the shared direct-depth siRMSE path. Both
variants keep the same outer runner, dataloader, valid-mask policy, W&B logging,
and checkpoint interface.

## W&B destination

The shared canonical W&B destination is:

```text
entity: cil-visionavengers
project: cil-visionavengers-depth
```

Specifying both values is important.  If `entity` is omitted, W&B may log runs
under each user's personal default entity instead of the shared team workspace.


## Checkpoint policy for adapter work

The default for LoRA/adapter experiments is **trainable-only**
checkpoint saving rather than full-model checkpoint saving.  This is important because
student-cluster scratch storage is limited and the base DA2 checkpoints already
live under scratch.

The reconstruction contract should be:

```text
base DA2 checkpoint + effective_config.yaml + trainable checkpoint payload
```

The config shape is:

```yaml
checkpoint:
  save_policy: trainable_only
  keep_latest: true
  keep_best: true
  save_optimizer: true
```

`trainable_only` should store only LoRA adapter weights and any changed original
DA2 trainable parameters.  Full-model checkpoints should remain explicit opt-in
through a policy such as `full_model`.

## Override policy

Future canonical entrypoints should resolve values in this order:

1. config file defaults;
2. documented environment-derived path defaults;
3. explicit CLI overrides.

Every effective config should be logged to W&B so later comparisons can identify
exactly which model, tuning mode, base trainable scope, data view settings,
evaluation protocol, and runtime choices were used.

## Representative configs added in Phase 2

| Config | Purpose |
|---|---|
| `configs/experiments/da2_vits_decoder.yaml` | Structured version of the partial decoder fine-tuning path |
| `configs/experiments/da2_vits_refinenets_output.yaml` | Structured version of the narrower partial decoder strategy |
| `configs/experiments/da2_vitb_full.yaml` | Structured version of the legacy square full fine-tuning path |

