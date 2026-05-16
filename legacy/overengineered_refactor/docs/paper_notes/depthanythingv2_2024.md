# Paper Note: Depth Anything V2 (Yang et al., 2024)

## Basic info
- Title: Depth Anything V2
- Authors: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao
- Venue / Year: NeurIPS, 2024
- Link: arXiv:2406.09414

## Problem setting
- Task: Open-world monocular depth estimation, with both relative-depth foundation models and downstream metric-depth fine-tuning.
- Supervision type: Large-scale supervised training on precise synthetic depth, plus pseudo-supervision on large-scale unlabeled real images.
- Relative or metric depth?: Base models predict affine-invariant inverse depth; separate metric models are obtained by fine-tuning.
- Main benchmark(s): Conventional zero-shot relative-depth benchmarks, the authors’ new DA-2K benchmark, and metric-depth fine-tuning on KITTI, NYU-D, Sintel, ETH3D, and DIODE.

## Core idea
- One-sentence summary: Depth Anything V2 improves V1 mainly through better data design: train a very strong teacher on clean synthetic depth, pseudo-label massive real-image corpora, then train student models on those pseudo-labeled real images.
- Main technical idea: Replace noisy real depth labels with precise synthetic labels, then use unlabeled real images as a bridge to recover robustness and scene diversity without reintroducing label noise.
- Why it should work: Real labeled depth is often noisy and coarse, while synthetic depth is precise but suffers from domain gap; the teacher–pseudo-label–student pipeline tries to combine the strengths of both.

## Method details
- Backbone / architecture:
  - Uses DPT as the depth decoder on top of DINOv2 encoders.
  - Teacher: DINOv2-Giant.
  - Student models released at four scales: ViT-S, ViT-B, ViT-L, and ViT-G.
- Losses:
  - Scale- and shift-invariant loss \(L_{ssi}\).
  - Gradient matching loss \(L_{gm}\), which the paper emphasizes is especially helpful for sharp details with synthetic labels.
  - On pseudo-labeled real images, they also use the feature-alignment loss from Depth Anything V1.
- Special components:
  - Three-stage pipeline: synthetic-only teacher -> pseudo-label real images -> train students on pseudo-labeled real images.
  - For pseudo-labeled samples, the top 10% highest-loss regions are ignored as potentially noisy.
  - The paper also introduces DA-2K, a new sparse relative-depth benchmark designed to better capture difficult real-world scenarios and reduce annotation noise.
- Training data:
  - 5 precise synthetic datasets, totaling about 595K images.
  - 8 pseudo-labeled real datasets, totaling about 62M images.
  - Training resolution: 518×518 crop after resizing the shorter side to 518.
  - Teacher training: batch size 64 for 160K iterations.
  - Student training on pseudo-labeled real images: batch size 192 for 480K iterations.

## Results
- Main claims:
  - Depth Anything V2 is much better than V1 in fine details and robustness, even when conventional zero-shot metrics do not fully show it.
  - It is more efficient and more accurate than recent Stable-Diffusion-based depth models such as Marigold and DepthFM.
  - Its pretrained encoders transfer strongly to metric-depth estimation.
- Representative results:
  - DA-2K benchmark accuracy: Marigold 86.8, Geowizard 88.1, DepthFM 85.8, Depth Anything V1 88.5, while Depth Anything V2 achieves 95.3 / 97.0 / 97.1 / 97.4 for ViT-S / B / L / G.
  - Conventional zero-shot relative-depth benchmarks: V2 is better than MiDaS and roughly comparable to V1 numerically, but the authors argue those benchmarks under-reward improvements on thin structures, transparency, reflections, and layout robustness.
  - Metric-depth fine-tuning (Table 4): ViT-L reaches KITTI AbsRel 0.074 and NYU-D AbsRel 0.045, while the teacher ViT-G reaches KITTI AbsRel 0.075 and NYU-D AbsRel 0.044.
- Strengths:
  - Strong and practically important data-centric insight: better labels matter more than fancy modeling tricks.
  - Good explanation of why synthetic-only training is not enough, and why pseudo-labeled real images help.
  - Offers a full family of model scales, which is useful in practice.
  - Introduces a benchmark motivated by real failure modes of existing test sets.
- Weaknesses / limitations:
  - The method depends on a very large teacher and very large pseudo-labeled real-image corpus, so reproducing it from scratch is unrealistic for a course project.
  - Some core claims rely on the authors’ new DA-2K benchmark rather than only on standard benchmarks.
  - The paper argues that conventional metrics miss important improvements, which is plausible but also makes cross-paper comparison trickier.
  - It is more a foundation-model/data-pipeline paper than a lightweight methodological twist.

## Relevance to our project
- Most relevant idea: Clean supervision aligned with the task may matter more than piling up more noisy data.
- What seems reusable:
  - starting from a strong Depth Anything V2 pretrained model family as a baseline;
  - thinking carefully about whether our training labels and augmentations preserve the fine structures our metric should reward;
  - considering edge/detail-sensitive losses or evaluation slices, since the paper highlights that standard metrics can miss meaningful quality gains.
- What probably does not transfer:
  - reproducing the teacher-student data pipeline ourselves;
  - collecting or pseudo-labeling tens of millions of real images.
- Should this influence our baseline or twist?: Yes. This is one of the strongest baseline families for our project. It is probably better as the starting baseline than as the source of the twist; the twist should likely be smaller and more project-specific.

## Short takeaway
- Depth Anything V2 argues that the path to a stronger monocular-depth foundation model is mostly about fixing the data pipeline, not inventing a more exotic architecture.
- Its key recipe is precise synthetic supervision for detail, plus pseudo-labeled real images for robustness and coverage.
- For our project, this makes Depth Anything V2 a top baseline candidate, while also reminding us to care about label quality and fine-detail behavior, not just aggregate benchmark scores.
