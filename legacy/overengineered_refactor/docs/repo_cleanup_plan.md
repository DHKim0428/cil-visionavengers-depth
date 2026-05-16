# Repository cleanup and canonicalization plan

This document records the cleanup plan after the DA2 refactor reached a working
canonical path but before the whole repository is fully canonicalized.

The current state is normal for an active research repo: the new DA2 path works,
but old U-Net scripts, DA3 diagnostics, generated outputs, SLURM wrappers, and
legacy artifacts are still mixed together.  The goal is not to make the repo look
pretty for its own sake; the goal is to make future experiments reproducible,
reviewable, and hard to accidentally mis-run.

## Final target

The intended destination is:

- one canonical augmentation implementation configured by args/config presets;
- multiple model implementations under `models/`;
- one outer training entrypoint that reads a config and dispatches to the right
  model/training loop;
- one outer evaluation entrypoint or evaluation class that reads a config and
  dispatches to the right evaluator;
- model-specific preprocessing and prediction semantics kept modular rather than
  hidden inside monolithic scripts;
- generated artifacts stored in scratch, not in the source tree;
- historical scripts/results preserved under `legacy/` only when useful for
  reproduction.

A possible final shape:

```text
configs/
  experiments/
  augmentations/
  models/

dataset/
  raw_cil.py
  augmentations.py          # one configurable augmentation policy
  supervision.py            # teacher masks / final valid-mask policy
  pipelines/
    da2.py
    unet.py

models/
  da2.py
  unet.py
  adapters.py

training/
  train.py                  # shared train orchestration helpers
  eval.py                   # shared eval orchestration helpers
  losses.py
  metrics.py
  checkpoints.py
  splits.py
  engines/
    da2.py
    unet.py

scripts/
  train.py                  # single outer CLI: --config ...
  eval.py                   # single outer CLI: --config ...
  setup/
  analysis/
  slurm/

legacy/
  da2/
  unet/                     # only if old U-Net entrypoints are superseded
```

This is a destination map, not something to force in one giant commit.

## Dirty areas observed before cleanup

### 1. Generated artifacts in the repository root

Observed root-level generated outputs include:

```text
da3_geocalib_fov_clip20/
geocalib_fov_full_clip20/
slurm-da2-eval-*.out
slurm-da2-train-*.out
__pycache__/
```

These should eventually be removed from the source tree.  If any are worth
preserving, they should move to `legacy/` or be summarized in docs while the
actual bulky artifacts live in scratch.

### 2. Empty or superseded old directories

The old DA2 runnable files were moved to `legacy/da2/`, but old directories may
still appear locally:

```text
fine-tune/
comparison/script/
comparison/results/
```

Git will not track empty directories, but they still make the working tree feel
messy.  They can be removed locally after confirming no active job references
them.

### 3. Script sprawl before R11

Before R11, `scripts/` mixed canonical DA2 entrypoints, DA3 precompute scripts,
debug tools, analysis utilities, SLURM wrappers, and Vast.ai wrappers.  R11
split these by role and kept only `train.py` / `eval.py` as current root-level
front doors.

Examples from the pre-R11 state:

```text
scripts/train_da2.py
scripts/eval_da2.py
scripts/precompute_da3_reliability_masks.py
scripts/da3_teacher_validate_tilt.py
scripts/precompute_geocalib_fov.py
scripts/debug_save_tilt_samples.py
scripts/analyze_*.py
scripts/slurm/*.sbatch
scripts/vastai/*.sh
```

Eventually, scripts should be grouped by role or hidden behind a smaller number
of canonical front doors.

### 4. Root U-Net training path is still pre-canonical

The repository still has:

```text
train.py
model.py
```

This path works as the U-Net baseline path, but it is not yet aligned with the
DA2 config/W&B/checkpoint structure.  It has its own split code, checkpointing,
logging, CLI args, and augmentation wiring.

### 5. DA3 teacher-mask path is useful but not canonical

DA3 masks and tilt validation are meaningful research machinery, but they are
currently standalone scripts.  Their relationship to `supervision.teacher_mask`
needs to be made explicit before they become part of the main training story.

## Cleanup principles

1. **Do not move files used by running jobs.**  Finish the active Phase 9A jobs
   before renaming canonical scripts or wrappers.
2. **Preserve reproducibility before deleting.**  If a script produced a result
   we still compare against, move it to `legacy/` or document the replacement.
3. **Generated artifacts belong in scratch.**  The repo should store code,
   configs, and small documentation artifacts only.
4. **Configs should define experiment meaning.**  Shell wrappers should only
   select resources and override paths/runtime details.
5. **One front door, modular internals.**  The final user interface should be
   simple, but internals should remain model-specific where prediction semantics
   genuinely differ.

## Proposed phases

### Cleanup Phase C0 — wait for active validation jobs

Status: complete for the cleanup decision.

The user confirmed there were no active/queued jobs before R9 cleanup began, so
no runnable path was moved out from under an active SLURM job.  Historical
pre-unified Phase 9A outputs remain documented separately; new final comparisons
should use the unified evaluator instead of extending that old queue.

### Cleanup Phase C1 — no-behavior artifact cleanup

Status: complete.

Completed without changing Python behavior:

- removed repo-local `__pycache__/` directories;
- moved root `slurm-da2-*.out` logs to `legacy/da2/slurm_logs/`;
- moved root DA3/GeoCalib generated outputs to `legacy/da3/artifacts/` and added
  small README notes;
- removed empty `fine-tune/` and `comparison/` shells after their preserved
  contents had already moved to `legacy/da2/`;
- added ignore patterns for regenerated root DA3/GeoCalib outputs;
- fixed the over-broad `models/` ignore rule so source files under `models/`
  are visible to git while checkpoint extensions stay ignored.

This phase intentionally did not rename canonical scripts or change runtime
behavior.

### Cleanup Phase C2 — script taxonomy without changing semantics

Status: complete in R11.

Implemented layout:

```text
scripts/
  train.py
  eval.py
  setup/
  analysis/
  da3/
  debug/
  slurm/
  vastai/
```

The canonical front doors stay at the root; role-specific utilities moved into
subdirectories after docs and wrappers were updated.  Transition-era DA2-only
entrypoints moved to `legacy/da2/transition/` once the unified paths had replaced
them.

### Cleanup Phase C3 — config-unify U-Net

Status: complete in R10.

Implemented:

- `configs/experiments/unet_disparity.yaml` and
  `configs/experiments/unet_metric_depth.yaml`;
- U-Net full-model checkpoints through `training/checkpoints.py`;
- W&B defaults, shared splits, and the canonical `dataset/factory.py` path;
- explicit U-Net prediction contracts selected by config rather than by a
  separate monolithic script.

The old root U-Net trainer now lives under `legacy/unet/`.

### Cleanup Phase C4 — one canonical train entrypoint

Status: complete in R10.

Implemented outer command:

```bash
python scripts/train.py --config configs/experiments/<experiment>.yaml
```

The command dispatches by config:

```yaml
model:
  family: da2_relative  # or unet
```

Shared orchestration lives in `training/runner.py`; small family-specific hooks
live under `training/tasks/` so the outer pipeline is unified without erasing
real DA2/U-Net differences.

### Cleanup Phase C5 — one canonical eval entrypoint

Status: complete across R2-R5 and finalized in R11.

Current outer command:

```bash
python scripts/eval.py --config configs/experiments/<experiment>.yaml --checkpoint ...
```

DA2-specific protocols such as `raw_infer_native`, `native_resolution`, and
`legacy_square` remain explicit through config/CLI choices, while model-family
semantics are isolated in adapters instead of separate outer scripts.

### Cleanup Phase C6 — canonical data pipeline and augmentation policy

Goal: keep one configurable CIL depth data construction path, with one
augmentation implementation configured by presets.

Current progress:

- `dataset/augmentations.py` contains the shared `DepthAugmentation` path;
- `dataset/cil_depth.py` and `dataset/factory.py` now provide the canonical
  metric-depth data path;
- DA2 and canonical U-Net configs now express train/eval data behavior through
  composable `data.views.*` fields;
- the unified trainer uses the new factory directly for both model families;
- the old argparse-only U-Net path is preserved under `legacy/unet/` rather than
  remaining a second canonical path.

Deliverables:

- one dataset/dataloader factory constructs model inputs from composable config
  fields rather than hardcoded DA2/U-Net/raw-native/legacy-square branches;
- DA2, U-Net, raw-native eval, and legacy-square behavior live as preset YAMLs
  or smoke-test labels, not as separate dataset implementations;
- augmentation presets are the source of truth for both DA2 and U-Net;
- paired spatial transforms operate on RGB/depth/masks together;
- model-specific preprocessing happens after paired augmentation;
- teacher masks remain supervision policy, not augmentation.

### Cleanup Phase C7 — canonical teacher-mask / filtering policy

Status: first canonical slice implemented in R8.

Implemented:

- explicit `raw_depth` vs `square` mask-grid contract;
- grid-aware DA3 precompute arguments and metadata;
- validator for artifact metadata, split coverage, and mask shapes;
- canonical DA2 runs attach exact mask metadata/validation to config and W&B;
- canonical CIL dataloaders consume teacher masks through one
  `dataset/supervision.py` contract.

Still deferred until the train-entrypoint migration:

- root legacy U-Net training still uses its old direct mask loading path;
- a future canonical U-Net trainer should consume the same shared contract.

### Cleanup Phase C8 — final README simplification

Status: complete in R11.

README now presents a small number of current commands:

```bash
python scripts/train.py --config ...
python scripts/eval.py --config ...
sbatch scripts/slurm/train.sbatch --export=ALL,CONFIG=...
sbatch scripts/slurm/eval.sbatch --export=ALL,CONFIG=...
```

Detailed legacy/parity/debug information should live in docs, not the main quick
start.

## What not to do yet

- Transition-era `train_da2.py` / `eval_da2.py` wrappers have now moved to
  `legacy/da2/transition/` because no old jobs were active and unified paths had
  replaced them.
- Do not delete legacy DA2 scripts/results until BCD parity results are recorded.
- Do not enable DA3 teacher masks for DA2 full experiments until mask alignment
  is validated.
- Do not collapse DA2 and U-Net internals into one over-generalized training
  loop before their prediction/evaluation semantics are documented.
## Consolidated remaining roadmap

The detailed remaining roadmap now lives in
`docs/remaining_refactor_plan.md`.  That document supersedes this cleanup plan as
the ordering guide: unified siRMSE evaluation comes before final cleanup and
before restarting the main comparison matrix.



### Cleanup Phase C12 — readability cleanup

Status: C12-A/B implemented; C12-C/D deferred.

C12-A removed `tuning.mode` from current DA2 configs so config files contain only
behavior-defining knobs.  C12-B moved transition-only DA2 loader/eval/loss helper
modules into `legacy/da2/transition/`.  The next candidate cleanup is to reduce
thin wrapper functions and possibly merge overly fine-grained files, but only
after reviewing the current diff.
