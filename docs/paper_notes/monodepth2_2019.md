# Paper Note: Monodepth2 (Godard et al., 2019)

## Basic info
- Title: Digging Into Self-Supervised Monocular Depth Estimation
- Authors: Clément Godard, Oisin Mac Aodha, Michael Firman, Gabriel Brostow
- Venue / Year: ICCV, 2019
- Link: arXiv:1806.01260

## Problem setting
- Task: Predict dense depth from a single RGB image.
- Supervision type: Self-supervised depth estimation using monocular video, stereo pairs, or both during training.
- Relative or metric depth?: Primarily relative depth under monocular training; stereo or mixed training can recover metric scale from the known stereo baseline.
- Main benchmark(s): KITTI Eigen split; also additional results on Make3D and KITTI depth/odometry benchmarks.

## Core idea
- One-sentence summary: Monodepth2 shows that a relatively simple self-supervised depth model becomes much stronger if training is cleaned up with better handling of occlusions, static pixels, and multi-scale sampling artifacts.
- Main technical idea: Keep the standard depth-from-view-synthesis framework, but improve three failure points in the loss: use per-pixel minimum reprojection, auto-mask pixels that violate motion assumptions, and compute multi-scale losses at full input resolution.
- Why it should work: Most self-supervised monocular errors come less from missing architectural complexity and more from corrupted supervision signals caused by occlusions, camera stops, moving objects, and low-resolution warping artifacts.

## Method details
- Backbone / architecture:
  - Depth network: U-Net-style encoder-decoder with skip connections.
  - Encoder: ResNet18 by default, ImageNet pretrained.
  - Pose network: separate ResNet18-style network taking two frames as input and predicting relative 6-DoF pose.
  - Depth prediction uses a sigmoid output converted to bounded depth values.
- Losses:
  - Photometric reprojection loss using a weighted combination of SSIM and L1.
  - Edge-aware smoothness loss on normalized inverse depth.
  - Per-pixel minimum reprojection across source frames instead of averaging reprojection errors.
  - Auto-masking: only keep pixels where the warped source image explains the target better than the unwarped source image.
- Special components:
  - Minimum reprojection loss for occlusions and out-of-view pixels.
  - Auto-masking for stationary cameras and objects moving with the camera.
  - Full-resolution multi-scale sampling to reduce texture-copy and low-resolution warping artifacts.
  - Reflection padding in the decoder to reduce border artifacts.
- Training data:
  - Main experiments use KITTI 2015 with the Eigen split.
  - For monocular training they use triplets of frames.
  - Default training: 20 epochs, Adam, batch size 12, resolution 640×192.
  - Data augmentation includes horizontal flips and color jitter.

## Results
- Main claims:
  - Monodepth2 achieves state-of-the-art self-supervised monocular depth results on KITTI at the time.
  - With monocular training only, it improves over prior self-supervised monocular baselines such as Zhou et al., GeoNet, DDVO, EPC++, and Struct2Depth.
  - The full model also performs strongly in stereo-only and mixed mono+stereo settings.
- Representative numbers (KITTI Eigen split):
  - Monodepth2 (M): Abs Rel 0.115, Sq Rel 0.903, RMSE 4.863, RMSE log 0.193.
  - Monodepth2 (S): Abs Rel 0.109, Sq Rel 0.873, RMSE 4.960, RMSE log 0.209.
  - Monodepth2 (MS): Abs Rel 0.106, Sq Rel 0.818, RMSE 4.750, RMSE log 0.196.
- Ablation highlights:
  - Baseline monocular model: Abs Rel 0.140.
  - Adding minimum reprojection alone: 0.122.
  - Adding auto-masking alone: 0.124.
  - Full Monodepth2: 0.115.
  - The paper convincingly shows that these “small” design choices account for much of the gain.
- Strengths:
  - Strong ablation study: the paper clearly isolates why performance improves.
  - Very practical: improvements are simple, cheap, and easy to adopt.
  - High qualitative sharpness compared with prior self-supervised methods.
  - Good example of improving supervision quality rather than only enlarging the architecture.
- Weaknesses / limitations:
  - Still depends on brightness constancy / Lambertian assumptions.
  - Fails on reflective, saturated, distorted, or ambiguous regions.
  - Monocular training still needs median scaling at evaluation because absolute scale is not identifiable.
  - Main contribution is in self-supervised training; some ideas transfer less directly to fully supervised settings.

## Relevance to our project
- Most relevant idea: Training details and loss design matter a lot; removing bad supervision signal can help as much as changing the model.
- What seems reusable:
  - edge-aware smoothness and detail-preserving regularization ideas;
  - multi-scale prediction with careful full-resolution supervision;
  - the general lesson that qualitative artifacts often come from the loss pipeline, not only the backbone.
- What probably does not transfer:
  - the pose network and view-synthesis training setup, since our project has direct depth supervision;
  - evaluation assumptions tied to monocular self-supervision and median scaling.
- Should this influence our baseline or twist?: Yes, but mostly indirectly. It should not be our main baseline because our setting is supervised and likely better matched to modern pretrained depth models. However, it is a very useful source of training tricks and a good citation for why artifact-aware loss design matters.

## Short takeaway
- Monodepth2 is important because it shows that self-supervised monocular depth improves a lot when the reprojection loss is made more trustworthy.
- Its biggest lesson for us is not the exact architecture, but the discipline of identifying which pixels or scales produce misleading supervision.
- For our project, this paper is best used as a training-design reference and historical milestone, not as the final baseline model family.
