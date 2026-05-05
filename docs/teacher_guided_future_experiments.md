# Teacher-Guided Future Experiments

This note collects two possible next experiments motivated by the DA3 teacher
validation diagnostic. These are planning notes, not completed results.

The common framing is that the CIL dataset depth scale is only meaningful within
each image, and the competition metric is scale-invariant RMSE. This makes
relative depth structure and reliable within-image supervision more important
than exact metric scale.

Related notes:

- [Project specification](project_spec.md)
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
   - fixed `geo_fov60`
   - fixed `geo_fov50`
   - fixed `geo_fov70`
   - adaptive FOV geometry
   - naive baseline
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

### Risks

- Single-image FOV estimation can be unstable.
- The teacher's FOV convention may not match the resized/cropped CIL images.
- The original dataset may not have a consistent camera model.
- Bad per-image FOV estimates could introduce more label noise than fixed FOV60.
- This adds implementation complexity and another external model dependency.

### Expected Outcomes

If adaptive FOV beats fixed FOV60 under DA3 validation and improves training,
then it becomes a strong project contribution:

```text
Adaptive-intrinsics geometry tilt reduces label noise from unknown camera
intrinsics and better preserves relative scene geometry.
```

If it only improves DA3 validation but not U-Net training, it can still be
reported as an interesting diagnostic result and a limitation of the current
training setup.

If it fails, the fixed FOV60 setup remains a simpler and more reproducible
choice.

## Suggested Priority

1. Run DA3-guided training-time label denoising first.
   - Lower implementation risk.
   - Directly targets noisy supervision.
   - Easy to compare against the baseline U-Net.
2. Run adaptive-FOV geometry tilt as an exploratory extension.
   - More creative and potentially stronger as a method.
   - Higher risk because FOV estimation may be unstable.
   - Best evaluated first as a DA3 diagnostic before full training.

## Possible Unified Project Claim

```text
Because the dataset depth scale is only meaningful within each image and the
evaluation metric is scale-invariant, reliable relative depth structure is the
central training signal. We investigate teacher-guided supervision filtering and
geometry-consistent tilt augmentation as two ways to reduce structured label
noise and preserve relative scene geometry.
```
