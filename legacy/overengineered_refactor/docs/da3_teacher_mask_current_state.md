# DA3 teacher-mask / filtering current state

This document records the Phase 9B audit of the current DA3 teacher-mask and
filtering-related code.  It is intentionally descriptive: the goal is to make the
existing behavior explicit before moving or rewriting scripts.

## Why this exists

The DA2 refactor introduced `supervision.teacher_mask` as a conceptual slot for
teacher-guided filtering, but the repository already contains DA3 mask
precomputation, tilt validation, GeoCalib FOV diagnostics, and U-Net training
wrappers that use teacher masks.  Before canonicalizing those paths, we need to
understand what each script actually does and where the contracts are fragile.

## Current components

| Component | Current role | Canonical status |
|---|---|---|
| `scripts/da3/precompute_da3_reliability_masks.py` | Generates binary per-sample training masks by comparing DA3 depth with CIL ground-truth depth. | Useful, but still standalone and U-Net-oriented by default. |
| `scripts/slurm/precompute_da3_reliability_masks.sbatch` | Cluster wrapper for DA3 mask precomputation. | Operational wrapper; should eventually become config-driven. |
| `legacy/unet/train.py` + `legacy/unet/simple_dataset.py` | Historical U-Net baseline path; can consume teacher masks during training only. | Legacy reference only after R10/C12 cleanup. |
| `dataset/supervision.py` | Shared helper for loading `{stem}_teacher_mask.png` and composing dataset-valid mask with teacher mask. | Good conceptual home for mask composition. |
| `dataset/factory.py` + `dataset/cil_depth.py` | Canonical CIL dataloader path; consumes `TeacherMaskSpec` and validates mask grid alignment. | Current path. |
| `scripts/da3/da3_teacher_validate_tilt.py` | Diagnostic comparison between DA3 depth and generated tilt-depth variants. | Analysis/debug script, not a training path. |
| `scripts/da3/precompute_geocalib_fov.py` | Precomputes per-image GeoCalib horizontal FOV used by tilt diagnostics. | Analysis/precompute utility. |
| `scripts/debug/debug_save_tilt_samples.py` | Saves tilt augmentation debug samples used by DA3 validation. | Analysis/debug utility. |
| `scripts/slurm/da3_teacher_validation*.sbatch` | Cluster wrappers for DA3-vs-tilt diagnostics. | Useful but experimental. |
| `legacy/da3/artifacts/da3_geocalib_fov_clip20/`, `legacy/da3/artifacts/geocalib_fov_full_clip20/` | Preserved historical diagnostic outputs moved out of the repository root in R9. | Historical context only; new artifacts should live under scratch. |

## `precompute_da3_reliability_masks.py`

### Inputs

Required / important inputs:

- `--data_root`: CIL training data directory containing `*_rgb.png` and
  corresponding `*_depth.npy` files.
- `--split_file`: JSON split file.  The script only processes `train_names`.
- `--output_dir`: output directory for masks and metadata.
- `--grid`: mask target grid, either `square` or `raw_depth`.  Default is
  `square` for current U-Net compatibility.
- `--mask_size`: square target size used when `--grid square`.  Default is
  `128`; ignored for `raw_depth`.
- `--threshold_percentile`: per-image AbsRel percentile used to decide which
  pixels remain reliable.  Default is `95`.
- `--model_dir`: DA3 model identifier, default `depth-anything/DA3-GIANT-1.1`.
- `--da3_repo` / `DA3_REPO`: optional DA3 source checkout path.
- `--process_res`: DA3 inference resolution, default `504`.

### Algorithm

For each training RGB file in the split:

1. load the CIL ground-truth depth;
2. choose the target supervision grid: native GT for `raw_depth`, or
   `mask_size x mask_size` for `square`;
3. run DA3 inference on the RGB image;
4. resize DA3 prediction to the resized GT shape using bilinear interpolation;
5. compute valid pixels where both target depth and teacher depth are finite and
   positive;
6. median-scale DA3 depth to the GT depth for that image;
7. compute AbsRel between scaled DA3 and GT;
8. keep pixels whose AbsRel is at or below the requested percentile threshold;
9. save a binary PNG mask named `{stem}_teacher_mask.png`, where `255` means
   keep for training loss and `0` means ignore.

### Outputs

The output directory contains:

- `{stem}_teacher_mask.png` for each processed training sample;
- `per_sample.csv` with per-image mask statistics;
- `metadata.json` with setup information;
- `summary.json` with aggregate statistics;
- usually a wrapper log such as `precompute_da3_reliability_masks.log`.

The output naming convention matches `dataset.supervision.load_teacher_mask(...)`.

## Current U-Net mask usage

`legacy/unet/train.py` passed `--teacher_mask_dir` into `legacy/unet/simple_dataset.py` for the training
split only.  Validation uses a separate `SimpleDepthDataset` without teacher
masks, which is the right conceptual behavior: teacher masks filter noisy
training supervision and should not alter validation.

`legacy/unet/simple_dataset.py` currently:

1. loads RGB and depth;
2. resizes both to `img_size x img_size`;
3. builds a basic valid mask from resized depth;
4. loads `{stem}_teacher_mask.png` if provided;
5. resizes that mask to `img_size x img_size`;
6. multiplies the dataset mask by the teacher mask;
7. applies paired augmentations;
8. normalizes depth by `max_depth`.

This works naturally with the current default DA3 masks because both the U-Net
training path and the precompute script default to `img_size=128`.

## Current DA2 mask hook

DA2 configs already expose:

```yaml
supervision:
  teacher_mask:
    enabled: false
```

`scripts/train.py` reads `supervision.teacher_mask.enabled` and, if true,
passes a `TeacherMaskSpec` into the canonical `dataset.factory` dataloader path.

The DA2 datasets then call `load_teacher_mask(...)` and compose the teacher mask
with the dataset-valid mask via `compose_valid_mask(...)`.

This means the hook exists, but DA2 teacher-mask training has **not** been fully
validated yet.

## Important alignment risk

There is a subtle but important contract mismatch:

- `dataset/supervision.py` describes teacher masks as being on the raw sample
  grid.
- `precompute_da3_reliability_masks.py` currently saves masks after resizing GT
  depth to `img_size x img_size`, default `128 x 128`.
- This is fine for the current U-Net path because U-Net also trains at `128 x
  128` and resizes masks to that grid.
- It is suspicious for DA2, especially `dpt_native`, because DA2 starts from the
  native image/depth grid and performs DPT-style lower-bound resize/crop.  A
  precomputed square `128 x 128` mask may not be spatially aligned with the raw
  image before DA2 transforms.

Therefore, before enabling teacher masks for DA2 experiments, Phase 9B should
choose one explicit mask-grid contract:

1. **Raw-grid masks**: precompute masks at the original depth resolution and let
   every dataset pipeline resize/transform them together with RGB/depth.
2. **Pipeline-grid masks**: declare that a mask directory is tied to a specific
   pipeline and `img_size`, and validate that the consuming dataset uses exactly
   the same preprocessing contract.

Raw-grid masks are cleaner long-term; pipeline-grid masks are easier to preserve
for current U-Net experiments.

### Agreed direction after discussion

Teacher masks are pixel-level supervision filters, not image-level keep/drop
labels.  The mask does not attach to the model input by itself; it attaches to
whatever target-depth grid is used when computing the loss.  Therefore the
important configuration is the **supervision target grid**, not merely the model
input size.

Short term, preserving the current U-Net behavior is acceptable with a
pipeline-grid default:

```yaml
supervision:
  teacher_mask:
    enabled: true
    dir: /work/scratch/$USER/cil-visionavengers-depth/teacher_masks/da3_giant_p95_square128_seed42
    grid: square
    size: 128
```

This matches the current U-Net path because both depth targets and masks are
resized to `128 x 128` before the loss.

Long term, the preferred canonical contract is raw-grid masks:

```yaml
supervision:
  teacher_mask:
    enabled: true
    grid: raw_depth
```

In that contract, DA3 reliability masks are stored on the original depth grid,
and every dataset pipeline is responsible for applying the exact same paired
spatial transforms to RGB, depth, dataset-valid mask, and teacher mask before
loss computation.  This is safer for DA2 `dpt_native`, where lower-bound resize
and crop make a precomputed square mask ambiguous unless the mask directory is
explicitly tied to that same preprocessing pipeline.

That contract is now implemented.  The precompute script exposes
`--grid raw_depth` or `--grid square --mask_size 128`, and every generated mask
artifact must provide `metadata.json` with grid, size, split, threshold, model,
process resolution, mask count, and mask semantics.

Canonical consumption now behaves differently by grid:

- `raw_depth` masks are loaded on the native GT grid and travel through the same
  paired spatial transforms as RGB/depth/GT-valid masks;
- `square` masks are loaded only after a matching square/model-input view exists;
- square masks are rejected for DA2-style DPT-native views rather than silently
  resized into an ambiguous alignment.

The canonical validator lives in `dataset/supervision.py` and is also exposed via
`scripts/da3/validate_teacher_masks.py --config ...`.  Canonical DA2 runs attach the
artifact metadata and validation result to `effective_config.yaml` and the W&B
config.  If the experiment config also pins `threshold_percentile`, `model_dir`,
or `process_res`, those expectations must match the artifact metadata exactly.

## Other risks / open questions

- **Split coupling**: DA3 masks are generated from a specific split file's
  `train_names`.  DA2 configs currently often rely on `val_fraction` and
  `split_seed`; if the split file differs, masks may be missing or may not match
  the intended training set.
- **Mask semantics**: the mask does not replace labels with DA3 pseudo-labels.
  It filters CIL GT pixels based on agreement with median-scaled DA3.  This is a
  supervision reliability policy, not pseudo-label training.
- **Dependency setup**: DA3 and GeoCalib are not part of the DA2 setup flow.
  Their environments/repos need a separate documented setup contract if they
  remain in the canonical workflow.
- **Logging**: DA3 precompute and validation scripts write CSV/JSON/log files but
  do not use W&B.  That is acceptable for one-off precompute, but final
  experiment tracking should at least log mask metadata into training runs.
- **Storage**: mask directories and diagnostic outputs can be large; they should
  live under scratch, never in the repository root.
- **Policy comparison**: percentile thresholds such as p95 are hard-coded in
  wrappers.  The final config should make threshold, DA3 model, process
  resolution, split file, and mask-grid contract explicit.

## Recommended Phase 9B path

### 9B-0 — audit and freeze current contracts

Deliverables:

- this document;
- a table of DA3 mask/diagnostic scripts and outputs;
- explicit statement that current p95/img128 masks are U-Net-oriented unless
  proven otherwise.

### 9B-1 — cleanup without behavior changes

Deliverables:

- move generated DA3/GeoCalib outputs out of the repository root or document
  them under ignored artifact paths;
- strengthen `.gitignore` for root debug artifacts and SLURM outputs if needed;
- leave running/queued DA2 Phase 9A jobs untouched.

### 9B-2 — define canonical mask contract

Deliverables:

- choose raw-grid or pipeline-grid masks;
- document the mask directory schema;
- add a small validator that checks mask coverage, shape expectations, and split
  consistency before training.

### 9B-3 — config-driven precompute/evaluate path

Deliverables:

- config section for teacher-mask precompute policy;
- config-driven SLURM wrapper or canonical script entrypoint;
- W&B or metadata attachment from training runs to the exact mask summary.

### 9B-4 — DA2 teacher-mask smoke test

Deliverables:

- a tiny DA2 training dry-run/smoke with `supervision.teacher_mask.enabled=true`;
- explicit validation that masks align with RGB/depth after DA2 preprocessing;
- no large experiment until the alignment contract is proven.

## R8 implementation note

R8 completed the first canonicalization slice without changing the legacy root
U-Net trainer yet.  Current U-Net-oriented square artifacts remain usable by
contract, while future model-agnostic DA2 experiments should precompute
`raw_depth` masks.  A synthetic smoke validated both artifact types and confirmed
that the incompatible `square` + DA2-native combination fails fast.
