# DA3 Teacher Validation Brainstorm

This note sketches a possible diagnostic experiment around Depth Anything 3
(`DA3-GIANT-1.1`) and our geometry-consistent camera tilt augmentation.

This is not meant to define DA3 as the main method, and it is not a plan to use
DA3 predictions as pseudo-labels for training. The purpose is narrower: use a
strong relative-depth teacher as a sanity check for whether our augmented target
depth `D_geo` has plausible relative scene structure after virtual camera tilt.

## Project Motivation

The CIL monocular depth project is evaluated with scale-invariant RMSE. The
training depth maps are stored in meters, but the project specification notes
that depth scales between images are not meaningful. The metric therefore
emphasizes within-image relative geometry, ordering, and structure more than
absolute metric scale.

Our geometry-consistent tilt augmentation tries to preserve this structure under
small virtual pitch/yaw perturbations. Instead of applying the same 2D image warp
to both RGB and depth, it warps the image with the pure-rotation homography and
then recomputes the target depth:

```text
D_geo(p') = D_src(p') * alpha_R(p)
```

where `D_src` is the naive homography-warped depth and `alpha_R(p)` is the
pixel-dependent depth scale induced by the virtual camera rotation.

The weak point is that the true camera intrinsics are unknown. We use an
approximate intrinsic matrix based on an assumed field of view, currently
`FOV = 60 deg`. If this assumption is wrong, the augmentation may introduce
structured label noise even though the math is internally consistent.

`DA3-GIANT-1.1` is a strong relative-depth teacher. Since the project metric is
scale-invariant, this makes it a useful external reference for checking whether
our tilted targets preserve plausible relative depth structure.

## Candidate Main Claim

Geometry-consistent virtual camera tilt augmentation is a perspective-aware data
augmentation for supervised monocular depth training. It simulates small pitch
and yaw changes, warps RGB with the induced homography, and recomputes the depth
label in the rotated camera frame instead of naively warping depth in image
space.

The goal is not to recover perfect metric depth under unknown intrinsics. The
goal is to preserve relative scene geometry under augmentation in a way that is
better aligned with the scale-invariant evaluation metric.

## Hypotheses

`H1`: Geometry recomputation should produce more plausible augmented depth labels
than naive 2D depth warping, because camera pitch/yaw changes the camera-frame
z-depth of each scene point by a pixel-dependent factor.

`H2`: Geometry-consistent tilt may regularize supervised monocular depth
training by exposing the model to perspective changes while keeping depth labels
geometrically coherent. However, current U-Net results should be interpreted
carefully because the baseline model may be too weak or too sensitive to
augmentation artifacts.

`H3`: If scale-aligned DA3-GIANT-1.1 predictions on tilted RGB images agree with
our `D_geo` targets on valid pixels, then the approximate intrinsic/FOV
assumption is more defensible. If they disagree systematically, the assumed FOV,
rotation convention, or border/mask handling may be introducing label noise.

## Proposed DA3 Validation Pipeline

Use the existing tilt debug sample folder as input. The folder should contain:

```text
*_rgb_aug.png
*_depth_aug.npy
*_mask_aug.png
metadata.json
```

Here, `*_rgb_aug.png` is the tilted RGB image, `*_depth_aug.npy` is our
geometry-recomputed target `D_geo`, and `*_mask_aug.png` marks valid supervision
pixels after the tilt warp and depth checks.

Pipeline:

1. Load tilted RGB images from the debug folder.
2. Run `depth-anything/DA3-GIANT-1.1` on the tilted RGB images.
3. Resize DA3 predictions to the `D_geo` resolution if needed.
4. Compare only on pixels where:
   - `mask_aug > 0`
   - `D_geo > 0`
   - DA3 prediction is finite and positive
5. Align DA3 prediction to `D_geo` per image using median scaling:

```text
scale = median(D_geo[valid]) / median(DA3_depth[valid])
DA3_scaled = scale * DA3_depth
```

6. Compute relative-structure metrics:
   - SILog between `DA3_scaled` and `D_geo`
   - median-scaled AbsRel
   - median, p90, and p95 absolute relative error
   - Spearman rank correlation
   - optional Pearson correlation in log-depth space

Suggested outputs:

```text
summary.json
per_sample.csv
visualizations/
```

Each visualization can show:

```text
Tilt RGB | D_geo | DA3 scaled | abs relative error heatmap
```

## Interpretation Guide

High Spearman correlation and low SILog would suggest that `D_geo` preserves a
relative depth structure that is consistent with a strong external teacher.

Good aligned metrics but poor raw-scale agreement would mostly indicate a scale
mismatch, which is not necessarily a problem for this project because the main
evaluation metric is scale-invariant.

Systematic spatial errors, such as one side of the image consistently disagreeing
after yaw-like tilts, may indicate that the assumed FOV, intrinsic matrix, or
rotation convention is not well matched to the data.

DA3 agreement should be treated as supporting evidence, not ground truth. DA3 is
a learned teacher with its own priors and failure modes. The primary project
evidence should still come from validation/Kaggle performance and controlled
ablations against baseline and naive augmentation settings.

## Relation to Current Experiments

This diagnostic complements the existing augmentation runs in
`docs/augmentation_experiment_log.md`. Those runs evaluate whether the
augmentation helps a U-Net baseline under validation SILog. The DA3 comparison
would answer a different question: whether the augmented labels themselves look
structurally plausible to a strong relative-depth model.

It also complements the geometry plan in
`cil-notes/geometry_consistent_tilt_augmentation.md`, which defines the
projection geometry and explains why naive depth warping is not generally
camera-frame-depth consistent.

## Open Questions

- Should we compare only one FOV assumption first, or sweep `FOV in {50, 60, 70}`?
- Should DA3-GIANT-1.1 be run on the resized `128 x 128` tilted RGB images, or
  should we regenerate larger debug samples for a stronger teacher signal?
- Should the report include DA3 as a formal experiment, or only as a short
  sanity-check appendix/diagnostic?
- If DA3 disagrees with `D_geo`, should we tune the assumed FOV or simply report
  the limitation?
