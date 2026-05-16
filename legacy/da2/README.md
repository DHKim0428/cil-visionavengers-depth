# Legacy DA2 entrypoints

These files are preserved for historical reference only.  New DA2 experiments
should use the canonical config-driven entrypoints from the repository root:

```bash
python scripts/train.py --config configs/experiments/da2_vits_refinenets_output.yaml
python scripts/eval.py --config configs/experiments/da2_vits_refinenets_output.yaml
```

The legacy scripts below were moved here in Phase 8 of the DA2 refactor because
their useful behavior has been absorbed into the canonical path while their
hard-coded personal paths make them unsafe as team-wide runners.

## Migration audit

| Legacy file | Historical role | Canonical replacement |
|---|---|---|
| `train_cil.py` | Partial DA2 fine-tuning with `decoder` / `refinenets_output`, encoder selection, poly LR, resume, TensorBoard/run-log bookkeeping | `scripts/train.py` with `base.trainable_scope`, `model.encoder`, `train.scheduler: poly_decay`, trainable-only checkpoints, W&B logging |
| `finetune_depth_anything_sirmse.py` | Square-resized full/partial DA2 fine-tuning with AMP, gradient accumulation, validation image logging, TensorBoard, best/last checkpoints | `scripts/train.py` with `configs/experiments/da2_vitb_full.yaml` or other configs, `legacy_square` pipeline, AMP/grad accumulation, validation image logging to W&B |
| `evaluate_depth_anything.py` | Zero-shot `infer_image(...)` DA2 evaluation on native ground-truth depth, optional visualizations | `scripts/eval.py --protocol raw_infer_native --num-vis ...` |
| `run_exp.sh` | Personal SLURM wrapper for `finetune_depth_anything_sirmse.py` with `/home/dchileban/...` paths | Use documented setup plus canonical trainer configs; write outputs under `/work/scratch/$USER/cil-visionavengers-depth` |
| `submit_train_cil.sh` | Personal SLURM wrapper for `train_cil.py` with `/home/heelee/...` paths | Use documented setup plus canonical trainer configs; write outputs under `/work/scratch/$USER/cil-visionavengers-depth` |

## Important differences in the canonical path

- Paths are config-driven and use the shared DA2 setup convention instead of
  personal absolute paths.
- W&B is the default experiment tracker instead of TensorBoard/text-only logs.
- Checkpoints default to a storage-conscious trainable-only format for adapter
  and partial fine-tuning runs.
- Dataset pipeline and evaluation protocol are explicit config/log fields:
  `legacy_square`, `dpt_native`, `native_resolution`, `legacy_square`, and
  `raw_infer_native`.
- LoRA/adapters are supported through `adapter`, `tuning`, and `base` config
  sections.

Do not add new features here.  If a missing behavior from these files turns out
to be important, port it into the canonical modular path instead.


## Historical result artifacts

Legacy numeric outputs produced by the old DA2 scripts are stored under
`legacy/da2/results/`.  See `legacy/da2/results/README.md` and
`docs/da2_legacy_parity.md` before using them as parity targets.


## Transition wrappers

The intermediate `train_da2.py` / `eval_da2.py` wrappers and their SLURM files
now live under `legacy/da2/transition/`.  They are kept only so old refactor notes
remain interpretable; they are not current entrypoints.
