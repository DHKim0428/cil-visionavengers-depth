# Teacher-Guided Future Experiments

This note collects two possible next experiments motivated by the DA3 teacher
validation diagnostic. These are planning notes, not completed results.

The common framing is that the CIL dataset depth scale is only meaningful within
each image, and the competition metric is scale-invariant RMSE. This makes
relative depth structure and reliable within-image supervision more important
than exact metric scale.

Related notes:

- [Project specification](../project_spec.md)
- [Augmentation experiment log](augmentation_experiment_log.md)
- [DA3 teacher validation brainstorm](da3_teacher_validation_brainstorm.md)

## Experiment 1: DA3-Guided Training-Time Label Denoising

### Motivation

Some ground-truth depth pixels may be noisy or structurally inconsistent with
the RGB image. This can happen around depth boundaries, missing-depth regions,
reflective surfaces, thin objects, sky/background areas, or dataset-specific
label artifacts.

Since DA3-GIANT-1.1 is a strong relative-depth teacher, we can use it as an
external sanity check for the training labels. The goal is not to replace the
ground truth and not to use DA3 as a pseudo-label source. The idea is only to
identify pixels where the provided target may be unreliable and exclude those
pixels from the supervised loss.

### Proposed Method

For each training image:

1. Run DA3 on the original RGB image.
2. Resize DA3 prediction to the training target resolution.
3. Compare DA3 prediction against the provided ground-truth depth only on
   existing valid pixels.
4. Median-scale align DA3 to the ground truth per image:

```text
scale = median(D_gt[valid]) / median(DA3[valid])
DA3_scaled = scale * DA3
```

5. Compute per-pixel absolute relative disagreement:

```text
err_i = abs(DA3_scaled_i - D_gt_i) / D_gt_i
```

6. Mark high-disagreement pixels as unreliable, for example:
   - remove pixels above image-wise p95 disagreement
   - remove pixels above image-wise p97.5 disagreement
   - remove pixels above image-wise p99 disagreement
7. During U-Net training, use:

```text
loss_mask = original_valid_mask & teacher_reliability_mask
```

This mask should be used only for training. Validation and Kaggle prediction
should not use DA3 or any teacher-based postprocessing.

### Baselines and Ablations

- Baseline U-Net with the original valid mask.
- U-Net with DA3 p95 training mask.
- U-Net with DA3 p97.5 training mask.
- U-Net with DA3 p99 training mask.
- Optional: random mask removing the same percentage of pixels, to test whether
  the gain comes from teacher-guided denoising rather than simply using fewer
  supervised pixels.

### Metrics to Track

- Validation SILog / scale-invariant RMSE.
- Final and best validation score.
- Average percentage of pixels removed by the teacher mask.
- Removed-pixel visualizations for a small debug subset.
- Per-image removed-pixel ratio distribution.

### Results Template

Status: completed, values to be filled from training logs and mask summaries.

#### Experimental Setup

| Field | Value |
| --- | --- |
| Dataset split | `splits/unet_seed42.json` |
| Train samples | 18084 |
| Validation samples | 4521 |
| Image size | 128 |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 1e-3 |
| Baseline augmentation | none |
| Teacher model | DA3-GIANT-1.1 |
| Mask application | training loss only |
| Validation mask | original valid-depth mask only |

Notes:

- DA3 predictions were median-scale aligned to ground truth per image before
  computing disagreement.
- Reliability masks were computed only on pixels where both ground-truth depth
  and DA3 depth were positive and finite.
- Pixels with zero ground-truth depth were not compared against DA3 and remain
  excluded by the original valid-depth mask.

#### Mask Precomputation Summary

| Mask | Samples | Weighted removed/valid | Mean removed/valid | Median removed/valid | Median threshold AbsRel | Median AbsRel median | Median AbsRel p95 | Median AbsRel p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p95 | 18084 | 0.0500 | 0.0501 | 0.0500 | 0.4378 | 0.0231 | 0.4378 | 10.8825 |
| p97.5 | 18084 | 0.0251 | 0.0251 | 0.0251 | 2.1251 | 0.0231 | 0.4378 | 10.8825 |
| p99 | 18084 | 0.0101 | 0.0101 | 0.0101 | 10.8825 | 0.0231 | 0.4378 | 10.8825 |

Mask-quality observations:

- TODO: Are removed pixels concentrated around depth discontinuities, invalid
  labels, sky/background, reflective surfaces, or thin structures?
- TODO: Do p99 removals look like extreme outlier cleanup rather than broad
  denoising?
- TODO: Any failure cases where DA3 removes visually plausible ground-truth
  supervision?

#### Training Results

| Run | Teacher mask | Best val SILog | Best epoch | Final val SILog | Final train SILog | Checkpoint / log |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | none | 0.6858 | 10 | 0.6858 | 0.6962 | checkpoint: `checkpoints/unet_baseline`  log: `logs/unet_baseline.log` |
| DA3 mask p95 | p95 | 0.6740 | 10 | 0.6740 | 0.4603 | checkpoint: `checkpoints/unet_da3_mask_p95_local` log: `logs/unet_da3_mask_p95_local.log` |
| DA3 mask p97.5 | p97.5 | 0.6515 | 10 | 0.6515 | 0.4543 | checkpoint: `checkpoints/unet_da3_mask_p97p5_local` log: `logs/unet_da3_mask_p97p5_local.log` |
| DA3 mask p99 | p99 | 0.6412 | 10 | 0.6412 | 0.4904 | checkpoint: `checkpoints/unet_da3_mask_p99_local` log:  `logs/unet_da3_mask_p99_local.log` |

#### Result Summary

Primary comparison:

- Best validation run: DA3 mask p99
- Improvement over baseline: DA3 mask p95 / p97.5 / p99
- Best threshold among p95 / p97.5 / p99: <- Not sure what this means
- Does the result support teacher-guided label denoising? **Yes**

Interpretation:

- TODO: If p95 wins, note that removing the noisiest 5% of valid supervision
  likely helps more than the lost supervision hurts.
- TODO: If p97.5 or p99 wins, note that DA3 is most useful as a conservative
  outlier filter rather than an aggressive denoising signal.
- TODO: If baseline wins, note whether high-disagreement pixels may be hard but
  useful supervision, or whether DA3 disagreement is mismatched to this dataset.

Recommended next step:

- TODO: Keep best mask setting for the final pipeline, run a random-mask control,
  combine with basic augmentation, or stop here.

### Expected Outcomes

If this helps, it supports the claim that some training labels contain noisy
supervision and that teacher-guided reliability masking improves learning under
a scale-invariant depth objective.

If it hurts, possible interpretations are:

- DA3 disagreement is not a reliable proxy for label noise.
- High-disagreement pixels are hard but useful supervision.
- The threshold is too aggressive.
- DA3 has its own systematic failures on this dataset.

### Reporting Framing

Safe claim:

```text
We use DA3 only as a training-time reliability filter. It does not provide
pseudo-labels and is not used at validation or test time.
```

This is likely the lower-risk next experiment because it directly targets label
quality while keeping the model and prediction pipeline simple.

## Experiment 2: Adaptive-FOV Geometry Tilt

### Motivation

Our geometry-consistent tilt augmentation currently assumes a fixed field of
view, usually `FOV = 60 deg`, to construct the intrinsic matrix:

```text
H = K R K^-1
```

The DA3 FOV sweep diagnostic showed that geometry recomputation is slightly more
plausible than naive depth warping, but it did not identify a single clearly
best FOV across all metrics. This makes fixed-FOV geometry tilt defensible but
still somewhat arbitrary.

An adaptive-FOV variant could estimate an image-specific or dataset-specific FOV
using an external camera/intrinsics teacher, then use that FOV when computing
the virtual tilt homography and depth correction.

### Proposed Method

1. Select a monocular camera/FOV estimation teacher.
2. Run it on the training images, or on a representative subset.
3. Convert predicted intrinsics/FOV into the same convention used by
   `DepthAugmentation`.
4. Use the predicted FOV when applying geometry tilt:

```text
K_i = intrinsics_from_fov(fov_i)
H_i = K_i R K_i^-1
D_geo_i = D_src_i * alpha(K_i, R)
```

5. Generate debug samples with adaptive FOV.
6. Run the DA3 teacher validation pipeline again:
   - naive baseline
   - fixed `geo_fov20`, `geo_fov25`, `geo_fov30`, `geo_fov35`, `geo_fov40`
   - fixed `geo_fov50`, `geo_fov60`, `geo_fov70`
   - adaptive `geo_geocalib`
7. If the diagnostic looks promising, run a supervised U-Net training ablation.

### Diagnostic Metrics

Use the same DA3 validation metrics:

- SILog scaled.
- Median AbsRel.
- p90/p95 AbsRel.
- Spearman rank correlation.
- Pearson correlation in log-depth space.

Also track:

- predicted FOV distribution
- per-image FOV variance
- examples with extreme predicted FOV
- DA3 metric correlation with predicted FOV
- valid-pixel ratio after tilt, because narrow FOVs create larger out-of-bounds
  regions and black borders for the same yaw/pitch angle

### Implemented Diagnostic Setup

Status: diagnostic completed; not selected for final training yet.

GeoCalib was selected as the camera/FOV estimator because it is designed for
single-image camera calibration and exposes a pinhole camera estimate with
horizontal and vertical FOV. The diagnostic used horizontal FOV because
`DepthAugmentation` constructs approximate intrinsics from image width:

```text
f = W / (2 tan(FOV / 2))
cx = (W - 1) / 2
cy = (H - 1) / 2
```

Implemented scripts:

- `scripts/precompute_geocalib_fov.py`
- `scripts/debug_save_tilt_samples.py` with `--fov_table`
- `scripts/da3_teacher_validate_tilt.py` with `geo_geocalib`
- `scripts/slurm/precompute_geocalib_fov.sbatch`
- `scripts/slurm/da3_teacher_validation_geocalib_fov.sbatch`

#### GeoCalib FOV Precomputation

GeoCalib was run on all training images with a clipped per-image horizontal FOV
policy:

```text
tilt_fov_deg = clip(hfov_deg_raw, 20 deg, 95 deg)
```

Full-dataset summary:

| Field | Value |
| --- | ---: |
| Images | 22,605 |
| Successful estimates | 22,605 |
| Errors | 0 |
| Clipped estimates | 1,100 |
| Clipped fraction | 4.85% |
| Raw HFOV min | 5.00 deg |
| Raw HFOV median | 38.22 deg |
| Raw HFOV mean | 38.48 deg |
| Raw HFOV max | 106.13 deg |
| Tilt FOV min after clipping | 20.00 deg |
| Tilt FOV median after clipping | 38.22 deg |
| Tilt FOV mean after clipping | 38.63 deg |
| Tilt FOV p95 after clipping | 58.47 deg |
| Tilt FOV max after clipping | 95.00 deg |

Cumulative clipped-FOV distribution:

| Threshold | Images <= threshold | Fraction |
| --- | ---: | ---: |
| 20 deg | 1,096 | 4.85% |
| 25 deg | 2,625 | 11.61% |
| 30 deg | 5,545 | 24.53% |
| 35 deg | 8,824 | 39.04% |
| 40 deg | 12,830 | 56.76% |
| 45 deg | 16,653 | 73.67% |
| 50 deg | 19,375 | 85.71% |
| 60 deg | 21,636 | 95.71% |
| 70 deg | 22,269 | 98.51% |
| 80 deg | 22,557 | 99.79% |
| 95 deg | 22,605 | 100.00% |

Interpretation:

- GeoCalib estimates a much narrower dataset-level FOV than the original fixed
  `FOV = 60 deg` assumption.
- A lower clip of 35 deg was too aggressive: in a 100-image subset, 38% of
  images were clipped to the lower bound. A lower clip of 20 deg preserved the
  raw distribution while only suppressing extreme outliers.
- Because the median FOV is near 38 deg, fixed narrow-FOV baselines such as
  `geo_fov30`, `geo_fov35`, and `geo_fov40` are necessary controls. Otherwise,
  an apparent gain from `geo_geocalib` could simply reflect a better
  dataset-level FOV prior rather than useful per-image adaptation.

#### DA3 Diagnostic Result

Diagnostic setup:

| Field | Value |
| --- | --- |
| Samples | 30 |
| Image size | 128 |
| Tilt max yaw/pitch | 5 deg / 5 deg |
| Adaptive FOV table | GeoCalib full train, clipped to 20--95 deg |
| Teacher | DA3-GIANT-1.1 |
| Target variants | `naive`, `geo_fov20`, `geo_fov25`, `geo_fov30`, `geo_fov35`, `geo_fov40`, `geo_fov50`, `geo_fov60`, `geo_fov70`, `geo_geocalib` |
| Alignment | per-image median scaling on valid pixels |

Aggregate result:

| Variant | SILog scaled ↓ | Median AbsRel ↓ | p90 AbsRel ↓ | p95 AbsRel ↓ | Spearman ↑ | Pearson log ↑ | Median valid ratio ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 0.3547 | 0.0337 | 0.2898 | 0.5820 | 0.8176 | 0.4080 | 0.5616 |
| `geo_fov20` | 0.3424 | 0.0436 | 0.2999 | 0.5450 | 0.8069 | 0.4203 | 0.4588 |
| `geo_fov25` | 0.3494 | 0.0365 | 0.2771 | 0.5491 | 0.8038 | 0.4295 | 0.5108 |
| `geo_fov30` | 0.3475 | 0.0302 | 0.2929 | 0.5030 | 0.8225 | 0.4552 | 0.5409 |
| `geo_fov35` | 0.3478 | 0.0330 | 0.2895 | 0.5413 | 0.8105 | 0.4380 | 0.5503 |
| `geo_fov40` | 0.3509 | 0.0319 | 0.2892 | 0.5494 | 0.8033 | 0.4161 | 0.5561 |
| `geo_fov50` | 0.3544 | 0.0369 | 0.2826 | 0.5375 | 0.8332 | 0.4336 | 0.5580 |
| `geo_fov60` | 0.3546 | 0.0293 | 0.2628 | 0.5793 | 0.8236 | 0.4179 | 0.5616 |
| `geo_fov70` | 0.3575 | 0.0370 | 0.2750 | 0.5160 | 0.8063 | 0.4367 | 0.5612 |
| `geo_geocalib` | 0.3505 | 0.0361 | 0.2899 | 0.5612 | 0.8190 | 0.4505 | 0.5394 |

Diagnostic interpretation:

- `geo_geocalib` did not clearly outperform the best fixed-FOV variants.
- `geo_fov30` had the strongest overall trade-off among fixed variants: good
  SILog, best p95 AbsRel, best Pearson log, and acceptable valid-pixel ratio.
- `geo_fov20` had the best SILog but a much lower valid-pixel ratio, consistent
  with the observation that narrower FOVs create larger black/invalid borders
  under the same tilt angle.
- `geo_fov60` remained strong on median and p90 AbsRel and preserved more valid
  pixels, so the original FOV60 assumption is not obviously broken under all
  metrics.
- The adaptive per-image GeoCalib FOV was useful as a diagnostic for the dataset
  FOV distribution, but the DA3 comparison does not support a strong claim that
  per-image FOV adaptation improves the tilt target.

### Important Geometry Caveat: Principal Point and Cropping

The current geometry tilt implementation assumes both an approximate focal
length and a centered principal point:

```text
K = [[f, 0, W/2],
     [0, f, H/2],
     [0, 0, 1]]
```

This assumption is central to both the image homography and the depth
correction:

```text
H = K R K^-1
D_geo = D_src * alpha(K, R, pixel)
```

If the dataset images are center-cropped from a larger image, the effective FOV
changes but the principal point may remain near the resized image center. In
that case, estimating a narrower FOV can still be a reasonable approximation.

However, if images are off-center crops, the effective principal point is no
longer the image center. Then the current approximation can introduce structured
label noise:

- source rays are back-projected with the wrong `cx, cy`
- the tilt homography samples from the wrong source locations
- the depth correction factor `alpha(K, R, pixel)` becomes spatially biased
- the resulting RGB-depth pair may look geometrically plausible but supervise
  the model with incorrect relative depth changes

This is especially concerning because Experiment 1 suggests that reducing noisy
supervision improves validation performance. Adding tilt augmentation based on
uncertain intrinsics could undo that benefit by injecting new structured label
noise.

### Risks

- Single-image FOV estimation can be unstable.
- The teacher's FOV convention may not match the resized/cropped CIL images.
- The original dataset may not have a consistent camera model.
- Unknown off-center crops may violate the centered-principal-point assumption.
- Bad per-image FOV or principal-point assumptions could introduce more label
  noise than fixed FOV60.
- Narrow FOVs produce larger out-of-bounds regions for the same tilt angle,
  increasing black borders and reducing valid supervision.
- This adds implementation complexity and another external model dependency.

### Expected Outcomes

If adaptive FOV beats fixed FOV60 under DA3 validation and improves training,
then it becomes a possible project contribution:

```text
Adaptive-intrinsics geometry tilt reduces label noise from unknown camera
intrinsics and better preserves relative scene geometry.
```

Current diagnostic evidence does **not** support this strong claim. A safer
interpretation is:

```text
GeoCalib suggests that the dataset has a narrow effective FOV distribution, but
per-image adaptive FOV did not consistently improve DA3 agreement over fixed
narrow-FOV controls. The method is sensitive to unknown intrinsics, especially
principal point shifts caused by possible off-center crops.
```

### Current Decision

Do **not** use adaptive-FOV geometry tilt in the final training pipeline unless a
later supervised ablation clearly improves validation performance.

Recommended final-project usage:

- Keep Experiment 1, DA3-guided training-time label denoising, as the primary
  low-risk method.
- Treat Experiment 2 as an exploratory diagnostic and limitation study.
- If time remains, the only relatively safe training ablation is a weak fixed
  narrow-FOV tilt, for example `geo_fov30` with `tilt_prob=0.1--0.25` and
  `yaw/pitch <= 3 deg`; even this should be compared carefully against the
  denoised baseline.

## Suggested Priority

1. Run DA3-guided training-time label denoising first.
   - Lower implementation risk.
   - Directly targets noisy supervision.
   - Easy to compare against the baseline U-Net.
2. Treat adaptive-FOV geometry tilt as an exploratory diagnostic, not a default
   final-pipeline component.
   - GeoCalib suggests a narrow effective FOV distribution, but per-image
     adaptive FOV did not clearly beat fixed narrow-FOV controls.
   - The method is high-risk because unknown off-center crops may violate the
     centered-principal-point assumption and inject structured label noise.
   - Only run further supervised ablations if there is enough time, and compare
     them directly against the denoised baseline.

## Possible Unified Project Claim

```text
Because the dataset depth scale is only meaningful within each image and the
evaluation metric is scale-invariant, reliable relative depth structure is the
central training signal. We use teacher-guided supervision filtering as the
primary way to reduce noisy labels, and analyze geometry-consistent tilt
augmentation as a high-risk exploratory method whose reliability depends on
uncertain camera intrinsics and cropping assumptions.
```
