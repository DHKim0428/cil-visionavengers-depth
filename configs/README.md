# Configs

Experiment configs live in `configs/experiments/`. Augmentation presets live in
`configs/augmentations/`.

Use them from the root entrypoints:

```bash
python train.py --config configs/experiments/da2_vits_refinenets_output.yaml
python train.py --config configs/experiments/unet_baseline.yaml
python eval.py --config configs/experiments/da2_vits_zero_shot.yaml
```

Important sections:

- `experiment`: run name
- `model`: one plain string: `da2_vits`, `da2_vitb`, `da2_vitl`, or `unet`
- `trainable`: DA2 scope: `frozen`, `full`, `decoder`, or `refinenets_output`
- `adapter`: optional PEFT LoRA settings
- `data`: dataset root, split, image size
- `augmentation`: preset name such as `none` or `basic`
  DA2 training configs currently use `basic`; zero-shot eval configs keep `none`.
- `train`: optimizer, epochs, batch size, AMP, workers, optional `early_stopping`
- `logging`: W&B entity/project
- `paths`: DA2 repo/checkpoints/output root
- `checkpoint`: `trainable_only` or `full_model`

Optional early stopping is configured under `train.early_stopping`:

```yaml
train:
  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.0
```

It monitors validation `sirmse_mean`; lower is better. If omitted, training runs for all configured epochs.

The canonical metric is siRMSE.  Ground-truth-valid pixels are
`0.001 <= depth <= 80.0`.

Predicted depths are not clipped to `[0.001, 80]` for siRMSE; only ground-truth pixels define the valid mask. Predictions are only clamped to a small positive epsilon before `log`.
