# Paper Note: Eigen et al. (2014)

## Basic info
- Title: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
- Authors: David Eigen, Christian Puhrsch, Rob Fergus
- Venue / Year: arXiv preprint, 2014 (PDF dated June 9, 2014)
- Link: arXiv:1406.2283

## Problem setting
- Task: Predict a dense depth map from a single RGB image.
- Supervision type: Supervised monocular depth estimation.
- Relative or metric depth?: Relative-oriented, though the model still predicts absolute depth values; the paper explicitly emphasizes scale ambiguity and scale-invariant evaluation.
- Main benchmark(s): NYU Depth v2 and KITTI.

## Core idea
- One-sentence summary: Use a coarse global CNN to infer overall scene geometry, then a finer local CNN to refine boundaries and details, while evaluating and partly training with a scale-invariant depth objective.
- Main technical idea: Split monocular depth prediction into global layout estimation and local correction, because monocular cues require both scene-level context and edge/detail alignment.
- Why it should work: Global context helps infer room layout, perspective, and object-scale relations, while a local refinement stage sharpens transitions near walls, objects, and road boundaries.

## Method details
- Backbone / architecture:
  - Coarse network: 5 convolutional layers + 2 fully connected layers.
  - Fine network: convolution-only refinement network.
  - The fine network takes both the original image and the coarse depth prediction as input.
  - Output resolution is 1/4 of the input resolution.
  - Coarse conv layers are pretrained on ImageNet.
- Losses:
  - Predicts log depth.
  - Training loss mixes elementwise squared error in log depth with a scale-invariant term.
  - The paper uses \(\lambda = 0.5\), i.e. halfway between pure pointwise \(\ell_2\) and fully scale-invariant loss.
- Special components:
  - Scale-invariant error for evaluation:
    - compares relative depth relations rather than absolute global scale;
    - directly matches the fact that monocular depth has unavoidable scale ambiguity.
  - Missing depth values are masked out during training.
- Training data:
  - NYU Depth raw data: about 120K unique training images, reshuffled to 220K samples.
  - KITTI raw data: about 20K unique images, reshuffled to 40K samples.
  - Data augmentation includes scaling, rotation (NYU), translation/cropping, color scaling, and horizontal flips.

## Results
- Main claims:
  - Achieves state-of-the-art performance on NYU Depth and KITTI at the time.
  - On NYU, the model clearly outperforms Make3D and prior methods on the reported metrics.
  - On KITTI, it also improves over Make3D across all listed metrics.
- Strengths:
  - Foundational articulation of scale ambiguity in monocular depth.
  - Coarse-to-fine design is intuitive and still influential.
  - Strong use of large raw training data rather than only small curated splits.
  - The paper distinguishes quantitative metric gains from qualitative boundary refinement.
- Weaknesses / limitations:
  - Comparisons are not perfectly controlled; some baselines are trained on much smaller datasets.
  - The fine network improves visual sharpness more than aggregate metrics.
  - Architecture is old by modern standards: low-resolution output, heavy fully connected layers, and limited detail recovery.
  - Some gains likely come from better data usage as much as from the architecture itself.

## Relevance to our project
- Most relevant idea: The scale-invariant depth viewpoint is directly aligned with our evaluation setting, where cross-image absolute scale is not meaningful.
- What seems reusable:
  - framing the problem around relative depth structure rather than absolute metric depth;
  - using scale-aware / scale-invariant losses or analysis in experiments;
  - coarse-to-fine refinement as a conceptual design pattern.
- What probably does not transfer:
  - reproducing the exact 2014 CNN architecture;
  - relying on fully connected global heads instead of stronger modern pretrained backbones.
- Should this influence our baseline or twist?: Yes. It should influence our evaluation framing and possibly motivate a small twist around scale-invariant structure, edge/detail refinement, or relative-depth-oriented training, but it should not be our main baseline model.

## Short takeaway
- This paper is important mainly because it makes the case that monocular depth needs both global scene reasoning and local refinement.
- Its scale-invariant error is especially relevant for our project, since our benchmark also values within-image depth structure more than cross-image absolute scale.
- We should cite it as a foundational paper and reuse its problem framing, but build our actual baseline on a much stronger modern pretrained model family.
