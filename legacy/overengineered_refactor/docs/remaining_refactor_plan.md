# Remaining refactor plan: simple canonical code first

## Reset decision

We decided to stop extending the previous over-abstracted refactor.  The active
code should be boring and readable, even if some model choices are handled by
plain `if` statements.

Canonical active structure:

```text
dataset/
  augmentations.py      # one small DepthAugmentation class
  cil_depth.py          # CIL Dataset + split + DataLoader construction

models/
  da2.py                # DA2 import/load/freeze/optional LoRA helpers
  unet.py               # U-Net baseline

training/
  trainer.py            # one Trainer class for train loop/checkpoints/wandb
  utils.py              # config loading, overrides, wandb, seed, save helpers
  losses_metrics.py     # siRMSE only

scripts/
  train.py              # canonical train entrypoint
  eval.py               # canonical eval entrypoint

legacy/
  overengineered_refactor/  # preserved previous active refactor code
```

## Current implementation status

Implemented in the active path:

- moved the previous over-abstracted active modules to `legacy/overengineered_refactor/`;
- replaced `training/losses_metrics.py` with a single `sirmse(...)` function;
- replaced `dataset/augmentations.py` with one small augmentation class;
- replaced `dataset/cil_depth.py` with dataset, split, and loader construction in one file;
- replaced `models/da2.py` with direct DA2 load/freeze/LoRA/checkpoint helpers;
- added `training/trainer.py` as the single training loop;
- replaced `scripts/train.py` and `scripts/eval.py` with thin canonical entrypoints.

Verified:

```bash
python3 -m py_compile dataset/*.py models/*.py training/*.py scripts/train.py scripts/eval.py
source ~/.bashrc && conda_cil && python scripts/train.py --dry-run --max-samples 4 --num-workers 0
source ~/.bashrc && conda_cil && python scripts/eval.py --dry-run --max-samples 4 --no-wandb
```

## Remaining cleanup

1. Read the simplified active files once more for readability.
2. Remove any stale docs/README references to deleted active modules.
3. Run a tiny real smoke test on 1-4 samples if checkpoints/data are available.
4. Only after that, decide whether DA3/teacher-mask scripts should remain legacy or be re-added cleanly.

## Coding rule from here

Do not add a helper unless it makes the caller easier to read.  Three-line model
semantics such as disparity-to-depth conversion should stay inline in train/eval.
