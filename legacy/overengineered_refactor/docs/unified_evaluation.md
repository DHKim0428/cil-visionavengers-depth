# Unified depth evaluation

This document records the canonical evaluator introduced after the Phase 9A
metric-policy discussion.

## Goal

Use one siRMSE evaluator for all depth model families:

- legacy U-Net baseline direct-depth checkpoints;
- canonical U-Net disparity or metric-depth checkpoints;
- DA2 official zero-shot checkpoints;
- DA2 full fine-tuning checkpoints;
- DA2 trainable-only checkpoints, including decoder/refinenets/LoRA/mixed runs.

The evaluator intentionally separates:

```text
model adapter -> prediction kind/depth semantics -> canonical metric
```

The adapter owns model-specific preprocessing and whether the model predicts
`depth` or `disparity`.  The evaluator owns GT-valid masking, prediction
sanitization, resizing to the GT grid, siRMSE, and output writing.

## Entry point

```bash
python scripts/eval.py --config configs/experiments/<experiment>.yaml --checkpoint <path>
```

For official DA2 configs, `--checkpoint` may be omitted because the checkpoint
is resolved from `paths.da2_checkpoint_dir` and `model.encoder`.

For U-Net, pass an explicit checkpoint:

```bash
python scripts/eval.py \
  --config configs/experiments/unet_baseline_eval.yaml \
  --checkpoint /work/scratch/$USER/cil-visionavengers-depth/checkpoints/unet_epoch_10.pth
```

Useful DA2 smoke:

```bash
python scripts/eval.py \
  --config configs/experiments/da2_vits_refinenets_output.yaml \
  --max-samples 1 \
  --no-wandb \
  --output-dir /tmp/cil_unified_eval_da2_smoke
```

## Canonical metric contract

The evaluator uses `training/metrics.py`.

Evaluation mask:

```text
0.001 <= gt_depth <= 80.0
```

DA2 disparity predictions are sanitized as:

```text
pred_disp = clamp(pred_disp, eps)
pred_depth = clamp(1 / pred_disp, 0.001, 80.0)
```

Direct-depth predictions, such as U-Net outputs, are sanitized as:

```text
pred_depth = clamp(pred_depth, 0.001, 80.0)
```

All predictions are resized to the native GT depth grid before siRMSE.

## Implemented adapters

| Adapter | Config selector | Prediction kind | Notes |
|---|---|---|---|
| DA2 raw infer | `model.family: da2_relative`, `data.eval_protocol: raw_infer_native` | disparity | uses `model.infer_image(...)` |
| DA2 tensor | `model.family: da2_relative` | disparity | supports `dpt_lower_bound` and `square` preprocessing |
| U-Net legacy baseline | `model.family: unet_baseline` | depth | loads historical normalized-depth state_dict checkpoints |
| U-Net canonical | `model.family: unet`, `model.prediction_kind: disparity` | disparity | loads canonical full-model checkpoints |
| U-Net canonical | `model.family: unet`, `model.prediction_kind: depth` | depth | loads canonical full-model checkpoints |

The legacy U-Net adapter preserves the old sigmoid-normalized-depth convention
and multiplies by `data.max_depth`. Canonical U-Net checkpoints store their
`prediction_kind` in checkpoint config metadata, so the evaluator restores the
correct disparity or direct-depth semantics automatically.

## Outputs

The evaluator writes:

- `effective_config.yaml`;
- `eval_summary.json`;
- `eval_summary.txt`;
- `sample_names.txt`.

The JSON contains selected sample names, evaluated sample names, per-sample
siRMSE values, adapter metadata, checkpoint path, config, and aggregate stats.

## Canonical comparison split

Final model comparisons use one shared saved validation split:

```text
/work/scratch/$USER/cil-visionavengers-depth/splits/canonical_val_5pct_seed42.json
```

The agreed contract is `val_fraction=0.05`, `split_seed=42`, which gives `21475`
train names and `1130` validation names from the `22605` CIL training samples.
Pass the split explicitly for comparison runs so every model sees the same
filenames:

```bash
python scripts/eval.py \
  --config configs/experiments/da2_vits_refinenets_output.yaml \
  --split-file /work/scratch/$USER/cil-visionavengers-depth/splits/canonical_val_5pct_seed42.json
```

Small smoke runs may add `--max-samples`, but their outputs should still record
the selected sample names.

## SLURM wrapper

New unified-eval jobs should use:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,SPLIT_FILE=/work/scratch/$USER/cil-visionavengers-depth/splits/canonical_val_5pct_seed42.json,RUN_NAME=my_eval \
  scripts/slurm/eval.sbatch
```

`eval.sbatch` forwards config/checkpoint/protocol/split/runtime overrides into
`scripts/eval.py`.

## Transition status

`scripts/eval.py` is now the only current evaluator for new comparison runs, and
`scripts/slurm/eval.sbatch` is the current SLURM wrapper.  The earlier DA2-only
transition entrypoints moved to `legacy/da2/transition/` in R11 after the
unified path replaced them.

R5 smoke validation has passed on the shared split for U-Net, DA2 `vits`
zero-shot, DA2 `vitb` zero-shot, and one fine-tuned DA2 checkpoint.  Full matrix
runs are intentionally deferred; the smoke only validated wiring and output
contracts.

Canonical DA2 policy is now explicit: use `img_size: 518` for zero-shot
evaluation and try `518` first for future fine-tuning.  `392` remains only for
legacy full-finetuning reproduction unless a future resource-constrained run
records a lower size deliberately.  Use `da2_vits_zero_shot.yaml` and
`da2_vitb_zero_shot.yaml` for canonical zero-shot evaluation rather than
borrowing legacy full-finetuning configs.
