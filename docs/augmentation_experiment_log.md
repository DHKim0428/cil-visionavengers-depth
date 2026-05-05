# Augmentation Experiment Log

This note tracks the supervised U-Net augmentation experiments run so far on the
ETH CIL monocular depth dataset.

## Setup

- Model: `UNetBaseline`
- Training script: [train.py](/home/dohkim/workspace/cil-visionavengers-depth/train.py)
- Dataset split: `/work/scratch/$USER/cil-visionavengers-depth/splits/unet_seed42.json`
- Common training defaults:
  - image size `128`
  - batch size `8`
  - learning rate `1e-3`
  - epochs `10`
  - val split `0.20`

The goal of these runs was to test whether standard augmentations or
geometry-aware camera tilt augmentation improve validation SILog.

## Runs

### 1. Baseline

- Job: `unet-baseline` (`57113`)
- Augmentation:
  - basic augmentation disabled
  - tilt augmentation disabled (`tilt_mode=none`)
- Runtime: `00:28:20`
- Final train SILog: `0.6724`
- Final val SILog: `0.6393`
- Best val SILog: `0.6393` at epoch `10`

Validation curve:

```text
1.1451 -> 1.0146 -> 0.9250 -> 0.8571 -> 0.8400
-> 0.7917 -> 0.7764 -> 0.7181 -> 0.6582 -> 0.6393
```

### 2. Geometry Tilt, Strong

- Job: `unet-tilt-geom` (`57114`)
- Augmentation:
  - basic augmentation disabled
  - tilt augmentation enabled
  - `tilt_mode=geometry`
  - `tilt_prob=0.5`
  - `tilt_max_yaw_deg=5`
  - `tilt_max_pitch_deg=5`
  - `tilt_fov_deg=60`
- Runtime: `00:24:08`
- Final train SILog: `0.7665`
- Final val SILog: `0.8040`
- Best val SILog: `0.8040` at epoch `10`

Validation curve:

```text
1.4776 -> 1.4201 -> 1.0320 -> 0.9827 -> 0.9577
-> 1.0938 -> 0.8734 -> 0.9810 -> 0.8699 -> 0.8040
```

### 3. Geometry Tilt, Weak

- Job: `unet-tilt-weak` (`57176`)
- Augmentation:
  - basic augmentation disabled
  - tilt augmentation enabled
  - `tilt_mode=geometry`
  - `tilt_prob=0.25`
  - `tilt_max_yaw_deg=3`
  - `tilt_max_pitch_deg=3`
  - `tilt_fov_deg=60`
- Runtime: `00:40:07`
- Final train SILog: `0.7065`
- Final val SILog: `0.7289`
- Best val SILog: `0.7204` at epoch `9`

Validation curve:

```text
1.1859 -> 1.0424 -> 0.9410 -> 0.8969 -> 1.0320
-> 0.8210 -> 0.7829 -> 0.7655 -> 0.7204 -> 0.7289
```

### 4. Basic Augmentation Only

- Job: `unet-basic-aug` (`57185`)
- Augmentation:
  - horizontal flip enabled
  - small rotation enabled
  - square crop-resize enabled
  - mild color jitter enabled
  - tilt augmentation disabled (`tilt_mode=none`)
- Runtime: `00:28:32`
- Final train SILog: `0.7291`
- Final val SILog: `0.7296`
- Best val SILog: `0.7296` at epoch `10`

Validation curve:

```text
1.2583 -> 1.1521 -> 1.0153 -> 1.0003 -> 0.8938
-> 0.9147 -> 0.8606 -> 0.8013 -> 0.7651 -> 0.7296
```

## Current Interpretation

- The plain baseline is the best run so far.
- Strong geometry tilt clearly hurts validation performance.
- Weakening the tilt helps a lot compared with the strong tilt setup, but still
  does not beat the no-augmentation baseline.
- The current basic augmentation bundle also underperforms the baseline.

At the moment, the evidence suggests that:

- the current U-Net baseline may be too sensitive to aggressive augmentation;
- the current geometry tilt implementation may introduce artifacts or noisy
  supervision;
- even standard online augmentations need to be ablated more carefully instead
  of bundled together.

## Important Implementation Notes

- `basic aug` runs did **not** multiply the RGB image by the valid mask.
- `tilt` runs **did** multiply the warped RGB image by the valid mask after the
  homography-based warp.
- This means the poor `basic aug` result is not explained by the black invalid
  border issue from the tilt path.

Relevant code:

- [dataset.py](/home/dohkim/workspace/cil-visionavengers-depth/dataset.py)
- [train.py](/home/dohkim/workspace/cil-visionavengers-depth/train.py)
- [scripts/slurm](/home/dohkim/workspace/cil-visionavengers-depth/scripts/slurm)

## Recommended Next Steps

1. Test simpler standard augmentations one at a time:
   - `flip only`
   - `rotation only`
   - `crop only`
2. Improve the tilt path before further geometry experiments:
   - crop to valid bounding box after tilt
   - reduce invalid border artifacts
3. Evaluate the same augmentation ideas on a stronger pretrained model such as
   Depth Anything fine-tuning, since the current U-Net may be too weak to judge
   augmentation quality fairly.
