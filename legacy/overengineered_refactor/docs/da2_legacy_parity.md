# DA2 legacy parity plan

This document defines Phase 9A of the DA2 refactor: preserving pre-refactor DA2
results and mapping them to the canonical config-driven trainer/evaluator.

The goal is not to force old and new numbers to match immediately.  The goal is
to make mismatches diagnosable: if the canonical path differs from a legacy
result, we should be able to tell whether the cause is split selection,
preprocessing, evaluation protocol, masking, checkpoint availability, or an
actual implementation bug.

## Scope of Phase 9A

In scope:

- move legacy DA2 result artifacts under `legacy/da2/results/`;
- document old experiment intent and recorded metrics;
- map each old run to a canonical config/command;
- define parity checks and likely causes of divergence;
- run lightweight dry-runs/sanity checks on the canonical commands.

Out of scope for this first pass:

- full re-training to match historical metrics;
- deciding whether the legacy square or DPT-native pipeline is better;
- refactoring DA3/teacher-mask generation.  That belongs to Phase 9B.

## Legacy artifact mapping

| Legacy result | Historical behavior | Canonical command skeleton | Expected comparison |
|---|---|---|---|
| `legacy/da2/results/si_rmse_results.txt` | `vitb` DA2 zero-shot raw `infer_image(...)` evaluation on native GT grid | `python scripts/eval.py --config configs/experiments/da2_vitb_full.yaml --protocol raw_infer_native --checkpoint /work/scratch/$USER/cil-visionavengers-depth/models/da2/depth_anything_v2_vitb.pth` | Compare aggregate siRMSE after matching sample fraction/seed |
| `legacy/da2/results/si_rmse_results_vits.txt` | `vits` DA2 zero-shot raw/native evaluation variant | `python scripts/eval.py --config configs/experiments/da2_vits_refinenets_output.yaml --protocol raw_infer_native --checkpoint /work/scratch/$USER/cil-visionavengers-depth/models/da2/depth_anything_v2_vits.pth` | Compare aggregate siRMSE after matching sample fraction/seed |
| `legacy/da2/results/results2.txt` | Evaluation of a `vitb` full/square fine-tuned checkpoint from `finetune_depth_anything_sirmse.py` | `python scripts/train.py --config configs/experiments/da2_vitb_full.yaml` then `python scripts/eval.py --protocol legacy_square --checkpoint <canonical best.pth>` | Compare only after reproducing or locating the old checkpoint/split |
| `legacy/da2/results/refinenets_ft_vits_run_log.txt` | `vits` partial `refinenets_output` fine-tuning from `train_cil.py` | `python scripts/train.py --config configs/experiments/da2_vits_refinenets_output.yaml` | Compare trainable parameter count first, then validation protocol/score |

## Parity checklist

Before treating a metric mismatch as a bug, check:

1. **Sample set** — old scripts used random fractions/splits; exact filenames may
   not be recoverable from summary text alone.
2. **Checkpoint** — old scripts referenced personal paths under `/home/...`; if
   the exact checkpoint is unavailable, parity is approximate.
3. **Pipeline** — `legacy_square`, `dpt_native`, and `raw_infer_native` are not
   interchangeable protocols.
4. **Masking** — old scripts used slightly different valid-key names and valid
   mask construction; canonical code should make these explicit.
5. **Prediction resize** — old scripts differed in interpolation and
   `align_corners` behavior.
6. **Training randomness** — full training parity needs the same split, seed,
   augmentation path, batch/drop-last behavior, and scheduler stepping.

## Minimum 9A acceptance checks

A first-pass 9A cleanup is acceptable when:

- legacy artifacts are preserved under `legacy/da2/results/`;
- every artifact has a canonical command skeleton;
- canonical train/eval dry-runs still work after the move;
- no old runnable DA2 entrypoint remains in the active `fine-tune/` or
  `comparison/script/` paths;
- unresolved parity risks are documented rather than silently ignored.

Status of this first pass: these acceptance checks are satisfied for artifact
movement, command skeletons, canonical dry-runs, and file layout.  Full metric
parity remains future work because the exact old splits/checkpoints are not all
available in-repository.

## Phase 9A-2: SLURM validation commands

The first Phase 9A pass created mappings and dry-ran command skeletons.  The next
step is to submit real cluster jobs and inspect W&B/log outputs.  Generic wrappers
now live at:

```text
legacy/da2/transition/slurm/train_da2.sbatch
legacy/da2/transition/slurm/eval_da2.sbatch
```

Both wrappers have been directly checked with `DRY_RUN=1`, so the commands below
should primarily validate real GPU execution, W&B logging, checkpoint writing,
and metric behavior rather than basic argument forwarding.

### 1. Refinenets fine-tuning smoke

Purpose: verify DA2 import, checkpoint loading, trainable parameter selection,
W&B logging, checkpoint writing, and scratch log creation.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,EPOCHS=1,MAX_SAMPLES=32,BATCH_SIZE=2,NUM_WORKERS=2,RUN_NAME=phase9a_refinenets_smoke \
  legacy/da2/transition/slurm/train_da2.sbatch
```

Check:

- W&B run appears under `cil-visionavengers/cil-visionavengers-depth`;
- `$SCRATCH_ROOT/logs/phase9a_refinenets_smoke.log` exists;
- `$SCRATCH_ROOT/checkpoints/phase9a_refinenets_smoke/<timestamp>/effective_config.yaml` exists;
- trainable base params are close to the legacy log value `635,233`.

### 2. VITS zero-shot raw/native eval smoke

Purpose: compare the canonical raw-infer protocol against
`legacy/da2/results/si_rmse_results_vits.txt` at small sample count first.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,PROTOCOL=raw_infer_native,MAX_SAMPLES=256,NUM_WORKERS=2,RUN_NAME=phase9a_vits_raw_smoke \
  legacy/da2/transition/slurm/eval_da2.sbatch
```

Legacy reference: mean siRMSE `0.6010`, but exact equality is not expected unless
the sampled filenames match.

### 3. VITB zero-shot raw/native eval smoke

Purpose: compare the canonical raw-infer protocol against
`legacy/da2/results/si_rmse_results.txt`.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vitb_full.yaml,PROTOCOL=raw_infer_native,MAX_SAMPLES=256,NUM_WORKERS=2,RUN_NAME=phase9a_vitb_raw_smoke \
  legacy/da2/transition/slurm/eval_da2.sbatch
```

Legacy reference: mean siRMSE `0.6158`, again approximate unless sampled
filenames match.

### 4. Longer refinenets parity run

Only run this after the smoke job looks healthy.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,EPOCHS=10,RUN_NAME=phase9a_refinenets_full \
  legacy/da2/transition/slurm/train_da2.sbatch
```

Legacy reference: best validation siRMSE `0.5135`.  If the canonical run is far
away from that scale, inspect split, protocol, mask, augmentation, and scheduler
before changing model code.


## Phase 9A-3: Real SLURM validation status

Status after the first real cluster validation round:

| Check | Run name | Status | Observed result | Legacy reference | Notes |
|---|---|---|---|---|---|
| Refinenets train smoke | `phase9a_refinenets_smoke` | passed | trainable params `635,233`; W&B/checkpoints/logs created | trainable params `635,233` | Confirms DA2 import, checkpoint loading, trainable-scope selection, W&B logging, and checkpoint writing. |
| VITS raw eval smoke | `phase9a_vits_raw_smoke` | passed | mean siRMSE `0.5132` over `256` samples | mean `0.6010` over `4521` samples | Smoke only; sample count intentionally small. |
| VITB raw eval smoke | `phase9a_vitb_raw_smoke_retry` | passed after downloading `vitb` checkpoint | mean siRMSE `0.5362` over `256` samples | mean `0.6158` over `2260` samples | First attempt failed because `depth_anything_v2_vitb.pth` was missing from scratch. |
| VITS full-count raw eval | `phase9a_vits_raw_4521_frac020` | passed | mean siRMSE `0.5235`, median `0.5029`, std `0.1719`, samples `4521` | mean `0.6010`, median `0.5363`, std `0.2734`, samples `4521` | Corrected run used `FRACTION=0.2`; earlier `phase9a_vits_raw_4521` only evaluated `1130` samples because config `val_fraction=0.05` was applied before `MAX_SAMPLES`. |
| VITB full-count raw eval | `phase9a_vitb_raw_2260` | passed | mean siRMSE `0.5477`, samples `2260` | mean `0.6158`, median `0.5468`, std `0.2887`, samples `2260` | Sample count matches the legacy result. |
| Refinenets full training | `phase9a_refinenets_full` | running at last check | train samples `21475`, val samples `1130`, trainable params `635,233` | train `21475`, val `1130`, trainable params `635,233`, best val `0.5135` | Early invariants match legacy exactly; wait for epoch-level validation curve. |
| Decoder full training | `phase9a_decoder_full` | queued at last check | pending due to `QOSMaxJobsPerUserLimit` | no preserved metric file | Coverage check for the legacy `decoder` strategy. |
| VITB full legacy-square training | `phase9a_vitb_full_legacy_square` | queued at last check | pending due to `QOSMaxJobsPerUserLimit` | `results2.txt` mean `0.4518` over `2260` samples | Needs post-training eval of the produced `best.pth` for comparison. |

Interpretation so far:

- Operationally, the canonical DA2 train/eval path is healthy: real SLURM jobs
  run, W&B syncs, logs are written to scratch, and trainable-only checkpoints are
  produced.
- The refinenets training setup matches the legacy cheap invariants exactly:
  train/val sample counts and trainable parameter count are identical.
- The zero-shot raw/native metrics are consistently better than the preserved
  legacy summaries for both `vits` and `vitb`.  This should not be silently
  treated as metric parity.  The next diagnosis should compare sample selection,
  valid-mask construction, prediction resize details, and disparity/depth
  conversion against the legacy evaluator before changing model code.
- The course valid-pixel guidance masks missing ground truth, especially zero
  depth pixels.  It does not by itself require dropping model pixels where DA2
  predicts non-positive disparity.  The legacy zero-shot evaluator used GT-valid
  pixels and converted disparity with `1 / (pred_disp + eps)`, while the current
  canonical helper also filters `pred_disp > 0`; this prediction-valid filtering
  is a likely source of the observed metric gap and should be tested explicitly.
- Agreed evaluation/loss direction: masks should encode GT availability and
  supervision policy, not whether the model produced a convenient value.  For
  DA2, `pred_disp <= 0` should be handled by explicit clamping/sanitization
  before disparity-to-depth conversion rather than by removing those pixels from
  the metric/loss mask.

## Phase 9B pointer: filtering / teacher masks

Teacher-mask filtering is only partially covered by the current canonical DA2
path.  `supervision.teacher_mask` lets the trainer consume precomputed masks,
but DA3 reliability-mask generation, validation, and policy comparison are still
separate scripts.  Phase 9B should audit and refactor that filtering/supervision
pipeline explicitly.


## Unified-eval transition note after R4/R5

Phase 9A results remain useful historical diagnostics, but new final comparisons
should use `scripts/eval.py` plus the saved canonical 5% split rather than the
older DA2-only evaluator commands above.  R5 intentionally ran only a tiny smoke
matrix to validate the new evaluator path; it did not attempt a new full result
table yet.
