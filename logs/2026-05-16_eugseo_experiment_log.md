# May 16, 2026 — eugseo experiment log

Purpose: first canonical DA2 comparison after simplifying the training/eval pipeline.


## Correction: DA2 relative checkpoint output convention

The DA2 relative checkpoints used here output an inverse-depth/disparity-like
quantity. A small sanity check on CIL samples showed raw DA2 output is negatively
correlated with GT depth, while `1 / raw` is positively correlated. Therefore
DA2 evaluation and fine-tuning loss should convert raw DA2 output with
`1 / pred.clamp_min(1e-6)` before computing siRMSE against GT depth.

The earlier legacy-style baseline evals that inverted DA2 output were using the
right output direction. They may still differ from later canonical runs if the
split/config differs, but they are not invalid merely because of the inversion.

Common environment variables used for the commands below:

```bash
export SCRATCH_ROOT=/work/scratch/$USER/cil-visionavengers-depth
export DATA_ROOT=/cluster/courses/cil/monocular-depth-estimation/train
export SPLIT_FILE=$SCRATCH_ROOT/splits/canonical_val_5pct_seed42.json
```

## Submitted / running

### 1. DA2 ViT-S zero-shot baseline eval

Account: `cil` via `scripts/slurm/eval.sbatch`.

```bash
sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vits_zero_shot.yaml,\
RUN_NAME=da2_vits_zero_shot_eval,\
DATA_ROOT=$DATA_ROOT,\
SPLIT_FILE=$SPLIT_FILE,\
SAVE_IMAGES=8 \
scripts/slurm/eval.sbatch
```

### 2. DA2 ViT-B zero-shot baseline eval

Account: `cil` via `scripts/slurm/eval.sbatch`.

```bash
sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vitb_zero_shot.yaml,\
RUN_NAME=da2_vitb_zero_shot_eval,\
DATA_ROOT=$DATA_ROOT,\
SPLIT_FILE=$SPLIT_FILE,\
SAVE_IMAGES=8 \
scripts/slurm/eval.sbatch
```

### 3. DA2 ViT-B full fine-tuning

Account: `cil_jobs` via `scripts/slurm/train.sbatch`.

```bash
sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vitb_full.yaml,\
RUN_NAME=da2_vitb_full_train_b8_acc1,\
DATA_ROOT=$DATA_ROOT,\
BATCH_SIZE=8,\
GRAD_ACCUM_STEPS=1,\
LOG_EVERY=100 \
scripts/slurm/train.sbatch
```

### 4. DA2 ViT-S decoder-only fine-tuning

Account: `cil_jobs` via `scripts/slurm/train.sbatch`.

```bash
sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vits_decoder.yaml,\
RUN_NAME=da2_vits_decoder_train,\
DATA_ROOT=$DATA_ROOT \
scripts/slurm/train.sbatch
```

## Completed results

### DA2 zero-shot baselines — legacy inversion baseline

These numbers are kept only as a record of the invalid pre-correction run. Both baseline eval jobs completed on the canonical 5% validation split
(`1130/1130` samples evaluated) using DA2 upstream `infer_image(...)` inside
the unified `eval.py` loop.

| Run | Config | Output dir | siRMSE mean | siRMSE median | siRMSE std | Samples |
|---|---|---|---:|---:|---:|---:|
| DA2 ViT-S zero-shot | `configs/experiments/da2_vits_zero_shot.yaml` | `/work/scratch/eugseo/cil-visionavengers-depth/evaluations/da2_vits_zero_shot_eval/20260516_050903` | `0.5197` | `0.5027` | `0.1774` | `1130` |
| DA2 ViT-B zero-shot | `configs/experiments/da2_vitb_zero_shot.yaml` | `/work/scratch/eugseo/cil-visionavengers-depth/evaluations/da2_vitb_zero_shot_eval/20260516_051026` | `0.5492` | `0.5326` | `0.1870` | `1130` |

Correction note: these values used DA2 inversion, which is the correct direction for the relative DA2 checkpoint. Re-run if the split/config changed, but do not discard them for the inversion itself.

## Not submitted yet

### 5. DA2 ViT-S all-linear LoRA fine-tuning

This is the intended LoRA-full experiment: base DA2 is frozen, and LoRA adapters
are attached to all `nn.Linear` modules.

```bash
sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vits_lora_full.yaml,\
RUN_NAME=da2_vits_lora_full_train,\
DATA_ROOT=$DATA_ROOT \
scripts/slurm/train.sbatch
```

After training finishes, evaluate best checkpoint with:

```bash
LORA_RUN_DIR=$(ls -td $SCRATCH_ROOT/checkpoints/da2_vits_lora_full_train/* | head -n 1)

sbatch --export=ALL,\
CONFIG=configs/experiments/da2_vits_lora_full.yaml,\
RUN_NAME=da2_vits_lora_full_eval,\
DATA_ROOT=$DATA_ROOT,\
CHECKPOINT=$LORA_RUN_DIR/best.pth,\
SPLIT_FILE=$SPLIT_FILE,\
SAVE_IMAGES=8 \
scripts/slurm/eval.sbatch
```

## Notes

- Baseline eval uses DA2 upstream `infer_image(...)` inside the unified `eval.py` loop.
- All reported comparison metrics should use the same siRMSE implementation.
- Prediction images, when requested, are saved by `eval.py --save-images N`.
