# DA2 evaluation

This document describes Depth Anything V2 evaluation through the canonical
unified evaluator used after R11.

## Entrypoint

Use `scripts/eval.py` from the repository root:

```bash
module load cuda/12.6.0
conda_cil
python scripts/eval.py --config configs/experiments/da2_vits_refinenets_output.yaml
```

If `--config` is omitted, the evaluator uses the same default config as the
canonical trainer:

```text
configs/experiments/da2_vits_refinenets_output.yaml
```

## Evaluation protocols

The evaluator makes the spatial comparison contract explicit through
`--protocol`.

### `native_resolution`

This is the intended canonical fine-tuning comparison protocol:

1. use the DPT/native DA2 preprocessing path;
2. keep ground-truth depth and valid masks on the native grid for validation;
3. run the model on its transformed input grid;
4. resize predicted disparity back to the ground-truth grid;
5. compute siRMSE there.

Example:

```bash
python scripts/eval.py --protocol native_resolution
```

### `legacy_square`

This reproduces the square-resized validation contract from the older full
fine-tuning script:

1. resize RGB, depth, and valid mask to the square DA2 input grid;
2. run the model on that square grid;
3. compare prediction and ground truth on the square grid.

Example:

```bash
python scripts/eval.py --protocol legacy_square
```

### `raw_infer_native`

This preserves the older zero-shot `model.infer_image(...)` evaluation behavior:

1. load raw BGR images with OpenCV;
2. call `model.infer_image(image, input_size)`;
3. resize predicted disparity to the native ground-truth grid;
4. invert disparity to depth for visualization;
5. compute siRMSE on the native ground-truth grid.

Example:

```bash
python scripts/eval.py \
  --protocol raw_infer_native \
  --fraction 0.1 \
  --num-vis 8
```


## Valid-pixel policy

The course specification says that ground-truth depth is only available on valid
pixels and that zero-depth pixels should be masked during training and
evaluation.  For this project, the primary evaluation mask should therefore be a
**ground-truth validity mask**:

```text
0.001 <= gt_depth <= 80.0
```

This is distinct from silently masking arbitrary model failures.  A model
prediction that is non-finite, zero, or negative is not a missing ground-truth
label; it is an invalid model output.  For historical DA2 relative-disparity
evaluation, the legacy script effectively handled such cases by computing
`1 / (pred_disp + eps)` and clipping the resulting depth into the valid depth
range, rather than dropping those pixels from the sample.

During Phase 9A parity work, this distinction mattered because the canonical
metric previously also required `pred_disp > 0`, while the legacy zero-shot
script used only the GT valid mask before disparity-to-depth conversion.  That
extra prediction-valid filter could make the canonical metric look better and is
a candidate explanation for the lower zero-shot siRMSE observed in the early
canonical runs.

The agreed comparison plan is:

1. keep a legacy-compatible raw-infer evaluation mode for reproducing historical
   numbers: GT-valid mask, `input_size=518`, `cv2.INTER_LINEAR` resize, and
   legacy-style disparity-to-depth conversion;
2. define the canonical report metric explicitly after parity diagnosis, rather
   than mixing GT-valid masking and prediction-valid filtering without naming the
   protocol.

After discussion, the preferred evaluation/loss policy is that masks should
represent **ground-truth availability and supervision policy**, not model-output
success.  Therefore:

- zero-depth / out-of-range GT pixels should be masked out;
- optional teacher masks may further remove unreliable GT supervision pixels;
- non-finite or non-positive model predictions should not silently remove pixels
  from the evaluation/loss mask;
- DA2 relative-disparity outputs should instead be sanitized with an explicit
  clamp, e.g. `pred_disp = max(pred_disp, eps)`, then converted to depth and
  clipped to the report depth range.

In short: `pred_disp <= 0` is a model-output sanitization issue, not a reason to
pretend the corresponding GT-valid pixel was unavailable.  This is also closer
to the legacy zero-shot evaluator, which converted disparity with
`1 / (pred_disp + eps)` and clipped the resulting depth rather than filtering
non-positive disparity pixels out of the score.

Implementation status: `training/metrics.py` now owns the shared GT-valid mask,
DA2 disparity-to-depth sanitization, direct-depth sanitization, resize helper,
and siRMSE calculation.  `training/da2_losses.py`, `training/da2_eval.py`,
`scripts/train.py`, and `scripts/eval.py` now route DA2 loss/evaluation
through that policy.  U-Net still needs to be wired into the same helper module
before the unified evaluator is complete.

## Unified evaluator entrypoint

New comparison runs should prefer the model-family-agnostic evaluator:

```bash
python scripts/eval.py --config configs/experiments/da2_vits_refinenets_output.yaml
```

`docs/unified_evaluation.md` describes the shared evaluator core and adapters.
The prior DA2-only transition evaluator now lives under
`legacy/da2/transition/eval_da2.py`; new commands should use the unified
entrypoint.

## Checkpoints

`--checkpoint` can point to:

- an official DA2 checkpoint downloaded by `scripts/setup/setup_da2.sh`;
- trainable-only `best.pth` from `scripts/train.py`;
- trainable-only `latest.pth` from `scripts/train.py`;
- full-model checkpoints if `checkpoint.save_policy: full_model` was used.

If omitted, the evaluator resolves the official checkpoint for the configured
encoder from `paths.da2_checkpoint_dir`.

For trainable-only checkpoints, the evaluator reads the config stored in the
checkpoint, reloads the base DA2 checkpoint, applies the configured adapter, and
then loads the saved trainable payload before evaluation.

## Useful smoke command

```bash
python scripts/eval.py \
  --protocol native_resolution \
  --img-size 56 \
  --batch-size 1 \
  --num-workers 0 \
  --max-samples 2 \
  --run-name phase5_eval_native_smoke \
  --no-wandb
```

## Outputs

Each evaluation writes to:

```text
/work/scratch/$USER/cil-visionavengers-depth/evaluations/$run_name/$timestamp/
```

unless `--output-dir` is provided.

The directory contains:

- `effective_config.yaml`;
- `eval_summary.json`;
- `eval_summary.txt`;
- optional `visualizations/` images.

The JSON summary records the model, checkpoint path, train pipeline,
evaluation protocol, sample names, per-sample siRMSE values, and aggregate
statistics.

## W&B behavior

The evaluator can log aggregate metrics to W&B using the config logging
settings.  Pass `--no-wandb` for local-only evaluation smoke tests.


## Historical evaluator

The older zero-shot DA2 evaluator was moved to `legacy/da2/evaluate_depth_anything.py`
in Phase 8.  Use `scripts/eval.py --protocol raw_infer_native` for new
zero-shot/native-grid evaluations.

## Unified evaluator transition

The project has now agreed that the final comparison should use one siRMSE-only
evaluator across U-Net, DA2 zero-shot, DA2 full fine-tuning, DA2 decoder-only,
DA2 refinenets-output, and future adapter variants.  The consolidated plan lives
in `docs/remaining_refactor_plan.md`.

Historical DA2 evaluator behavior remains useful for diagnosing discrepancies,
but final model comparisons should use the unified evaluator and a saved sample
list rather than underspecified report-era scripts.

