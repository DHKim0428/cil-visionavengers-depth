# Paper Note: Depth Anything (Yang et al., 2024)

## Basic info
- Title: Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data
- Authors: Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao
- Venue / Year: arXiv preprint, 2024
- Link: arXiv:2401.10891

## Problem setting
- Task: Build a robust foundation model for monocular depth estimation that generalizes well to unseen images and domains.
- Supervision type: Joint training with labeled depth datasets plus large-scale pseudo-labeled unlabeled images.
- Relative or metric depth?: Base model predicts relative / affine-invariant depth; downstream metric-depth models are obtained by fine-tuning.
- Main benchmark(s): Zero-shot relative-depth evaluation on KITTI, NYUv2, Sintel, DDAD, ETH3D, and DIODE; metric-depth fine-tuning on NYUv2 and KITTI.

## Core idea
- One-sentence summary: Depth Anything argues that a strong monocular-depth foundation model can be built mainly by scaling up diverse unlabeled monocular images and using them carefully in self-training.
- Main technical idea: Use a teacher trained on labeled depth data to pseudo-label 62M unlabeled images, but make the student’s task harder through strong perturbations and preserve semantic priors through feature alignment with a frozen DINOv2 encoder.
- Why it should work: The unlabeled images massively expand scene coverage, while the harder student objective forces the model to learn additional robust visual knowledge instead of merely copying the teacher.

## Method details
- Backbone / architecture:
  - DINOv2 encoder plus DPT decoder.
  - Three released scales: ViT-S (24.8M), ViT-B (97.5M), and ViT-L (335.3M).
  - Teacher for pseudo-labeling uses the best-performing ViT-L model.
- Losses:
  - Labeled loss: affine-invariant loss in normalized disparity space.
  - Unlabeled loss: affine-invariant loss against teacher pseudo labels.
  - Feature alignment loss: cosine-similarity-based alignment to frozen DINOv2 features, with a tolerance margin.
- Special components:
  - Strong perturbations on unlabeled images during student training.
  - Two perturbation types are emphasized: strong color distortion and CutMix.
  - Feature alignment is used instead of an auxiliary semantic segmentation task to preserve richer semantic priors.
  - Sky region is detected with a segmentation model and assigned farthest disparity during teacher training.
- Training data:
  - 1.5M labeled images from 6 public datasets: BlendedMVS, DIML, HRWSI, IRS, MegaDepth, and TartanAir.
  - 62M unlabeled images from 8 public datasets including SA-1B, Open Images, Places365, LSUN, ImageNet-21K, and BDD100K.
  - Teacher is trained on labeled data for 20 epochs.
  - In joint training, labeled:unlabeled ratio is 1:2 per batch.

## Results
- Main claims:
  - Depth Anything strongly improves zero-shot relative-depth estimation over MiDaS v3.1.
  - Its pretrained encoder is also much stronger for downstream metric-depth fine-tuning.
- Representative zero-shot results (Table 2):
  - MiDaS v3.1 ViT-L on KITTI: AbsRel 0.127, δ1 0.850.
  - Depth Anything ViT-L on KITTI: AbsRel 0.076, δ1 0.947.
  - MiDaS v3.1 ViT-L on DDAD: AbsRel 0.251, δ1 0.766.
  - Depth Anything ViT-L on DDAD: AbsRel 0.230, δ1 0.789.
- Representative metric fine-tuning results:
  - NYUv2: AbsRel 0.056, δ1 0.984.
  - KITTI: AbsRel 0.046, δ1 0.982.
- Ablation highlights:
  - Simply adding pseudo-labeled unlabeled images gives little or no gain.
  - Adding strong perturbations improves performance noticeably.
  - Adding feature alignment improves it further.
  - This supports the paper’s main claim that unlabeled data only helps when the student is trained in the right way.
- Strengths:
  - Very practical and scalable data-centric recipe.
  - Strong empirical evidence for zero-shot robustness.
  - Clear ablation showing why naive self-training is insufficient.
  - Produces a useful pretrained encoder for both depth and segmentation.
- Weaknesses / limitations:
  - Pseudo labels still come from the teacher, so there is a ceiling imposed by teacher quality.
  - The method is still tied to large-scale data collection and significant compute.
  - The paper improves robustness and transfer more than it targets delicate fine-detail fidelity; this is one reason V2 later revisits the labeled-data design.
  - Since the base model is affine-invariant, metric scale still requires downstream fine-tuning.

## Relevance to our project
- Most relevant idea: Large-scale pretraining on diverse data can matter enormously for monocular depth generalization.
- What seems reusable:
  - using a strong pretrained Depth Anything encoder as a baseline starting point;
  - thinking about whether data diversity, rather than only model architecture, is the main bottleneck;
  - using feature-level semantic preservation or related auxiliary regularization if fine-tuning becomes unstable.
- What probably does not transfer:
  - reproducing the 62M-image pseudo-labeling pipeline ourselves;
  - depending on internet-scale data collection in a course-project setting.
- Should this influence our baseline or twist?: Yes. It strongly supports starting from a Depth Anything family baseline. However, for our project it is more valuable as a pretrained baseline source than as the project’s novel twist.

## Short takeaway
- Depth Anything V1 is important because it shows that monocular depth can benefit enormously from internet-scale unlabeled images when self-training is done carefully.
- Its key insight is that unlabeled data alone is not enough; the student must be challenged and semantically regularized.
- For our project, this makes Depth Anything a highly relevant baseline family, even if we cannot reproduce its full training pipeline.
