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

- Job: `unet-tilt-geom` (`58285`)
- Augmentation:
  - basic augmentation disabled
  - tilt augmentation enabled
  - `tilt_mode=geometry`
  - `tilt_prob=0.5`
  - `tilt_max_yaw_deg=5`
  - `tilt_max_pitch_deg=5`
  - `tilt_fov_deg=60`
- Runtime: `00:24:08`
- Final train SILog: `0.6539`
- Final val SILog: `0.8148`
- Best val SILog: `0.7027` at epoch `9`

Validation curve:

```text
1.0638 -> 1.0621 -> 0.9040 -> 0.9084 -> 0.8782
-> 0.8214 -> 0.7808 -> 0.7767 -> 0.7027 -> 0.8148
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

## DA3 Teacher Validation Diagnostic

This diagnostic was run after adding the strict FOV sweep debug exporter and the
DA3 teacher validation script. It is not a training run and should not be
interpreted as pseudo-label supervision. The goal was to check whether the
tilted depth targets are structurally plausible when compared with a strong
external relative-depth model.

- Input debug folder:
  `/workspace/cil-visionavengers-depth/debug/tilt_geometry_samples`
- Output folder:
  `/workspace/cil-visionavengers-depth/debug/da3_teacher_validation`
- Teacher model: `depth-anything/DA3-GIANT-1.1`
- Samples: `30`
- Image size: `128`
- Tilt:
  - `tilt_max_yaw_deg=5`
  - `tilt_max_pitch_deg=5`
  - primary `tilt_fov_deg=60`
- FOV sweep:
  - `geo_fov50`
  - `geo_fov60`
  - `geo_fov70`
- Baseline target:
  - `naive`, the homography-warped depth for the primary FOV60 view
- Comparison:
  - DA3 was run on each variant's own tilted RGB warp where available.
  - DA3 predictions were median-scale aligned to each target per image.
  - Metrics were computed only on valid pixels where the variant mask is valid,
    the target depth is positive, and the DA3 prediction is finite and positive.

Aggregate results:

| Target | SILog scaled | Median AbsRel | p90 AbsRel | p95 AbsRel | Spearman | Pearson log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 0.35466 | 0.03362 | 0.28936 | 0.58161 | 0.81760 | 0.40798 |
| `geo_fov50` | 0.35445 | 0.03671 | 0.28298 | 0.53790 | 0.83314 | 0.43387 |
| `geo_fov60` | 0.35456 | 0.02922 | 0.26253 | 0.57907 | 0.82363 | 0.41780 |
| `geo_fov70` | 0.35737 | 0.03697 | 0.27522 | 0.51613 | 0.80609 | 0.43730 |

Notes:

- `geo_fov60` is the cleanest comparison against `naive`, because both are tied
  to the same primary FOV60 view. It improves median AbsRel, p90 AbsRel, and
  Spearman slightly, while SILog is nearly unchanged.
- This supports the claim that geometry recomputation is at least as plausible
  as naive depth warping under a DA3 relative-depth sanity check.
- The improvement is small, so this should be framed as supporting evidence
  rather than a decisive validation of the augmentation.
- FOV selection is inconclusive:
  - `geo_fov50` has the best Spearman and nearly best SILog, suggesting strong
    relative ordering.
  - `geo_fov60` has the best median and p90 AbsRel, suggesting the most stable
    typical-pixel agreement.
  - `geo_fov70` has better p95 AbsRel and Pearson log than the others, but worse
    Spearman and SILog.
- The large weighted mean AbsRel values from the full `summary.json` should be
  treated cautiously because mean AbsRel is very sensitive to small target
  depths, depth boundaries, and warp/mask artifacts. Median AbsRel, p90/p95,
  SILog, and Spearman are more useful for this diagnostic.

Current DA3-based interpretation:

```text
Geometry-consistent tilt targets look plausible under a strong external
relative-depth teacher. The FOV60 geometry target is slightly more DA3-consistent
than naive depth warping on typical-pixel and rank-structure metrics, but the
margin is not large enough to claim a clear win. The FOV sweep does not identify
a single best intrinsic assumption across all metrics.
```

## Current Interpretation

- The plain baseline is the best run so far.
- Strong geometry tilt clearly hurts validation performance.
- Weakening the tilt helps a lot compared with the strong tilt setup, but still
  does not beat the no-augmentation baseline.
- The current basic augmentation bundle also underperforms the baseline.
- The DA3 teacher diagnostic suggests that the geometry targets themselves are
  not obviously broken, and `geo_fov60` is slightly more plausible than naive
  warping on several relative-structure metrics. This does not yet explain why
  U-Net training performance worsened.

At the moment, the evidence suggests that:

- the current U-Net baseline may be too sensitive to aggressive augmentation;
- the current geometry tilt implementation may still introduce artifacts or
  noisy supervision during training, even if the recomputed labels look
  plausible under DA3;
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
