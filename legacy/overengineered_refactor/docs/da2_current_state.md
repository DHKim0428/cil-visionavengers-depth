# Current state after simplification reset

The previous DA2 refactor became too abstract for this project.  The active path
has been reset to a small canonical structure and the previous implementation is
preserved under `legacy/overengineered_refactor/`.

Active training/evaluation now flows through:

```text
scripts/train.py -> training.trainer.Trainer -> dataset.cil_depth + models.da2/unet
scripts/eval.py  -> direct eval loop using the same siRMSE metric
```

Important behavior kept:

- DA2 uses relative disparity and evaluates by converting to pseudo-depth before siRMSE.
- U-Net can be trained/evaluated as disparity or direct metric depth.
- siRMSE masks only ground-truth-valid pixels: `0.001 <= gt <= 80`.
- W&B remains the default logging backend, but missing login only warns and continues locally.
- Checkpoints are storage-conscious for DA2 by default (`trainable_only`) and full-model for U-Net configs.

Important behavior intentionally not reintroduced yet:

- teacher-mask training integration;
- DA3 geometry-heavy scripts in the canonical training loop;
- complex evaluation adapter classes;
- extra metric helper wrappers.

The next step is to run a small real smoke test, then update README commands to
match the simplified active path.
