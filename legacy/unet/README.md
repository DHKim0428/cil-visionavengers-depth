# Legacy U-Net baseline

These files preserve the original root-level U-Net baseline path that existed
before R10 introduced the canonical unified trainer.

- `train.py` uses the old argparse-only training loop, normalized depth targets,
  and raw state_dict checkpoints.
- `model.py` contains the original model definitions used by that legacy script.

New experiments should use `python scripts/train.py --config ...` with either
`configs/experiments/unet_disparity.yaml` or
`configs/experiments/unet_metric_depth.yaml`.


The old Vast.ai wrappers that target this trainer now live under
`legacy/unet/vastai/` for the same reason.
