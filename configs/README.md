# Configs

Experiment configs live in `configs/experiments/`. Augmentation presets live in
`configs/augmentations/`.

Use them from the root entrypoints:

```bash
python train.py --config configs/experiments/da2_vits_refinenets_output.yaml
python eval.py --config configs/experiments/da2_vits_zero_shot.yaml
```

Important sections:

- `experiment`: run name and tags
- `model`: `da2_relative` or `unet`
- `base`: DA2 trainable scope (`frozen`, `full`, `decoder`, `refinenets_output`)
- `adapter`: optional LoRA settings
- `data`: dataset root, split, image size, train/eval view settings
- `augmentation`: preset name
- `train`: optimizer, epochs, batch size, AMP, workers
- `logging`: W&B entity/project
- `paths`: DA2 repo/checkpoints/output root
- `checkpoint`: `trainable_only` or `full_model`

The canonical metric is siRMSE.  Ground-truth-valid pixels are
`0.001 <= depth <= 80.0`.
