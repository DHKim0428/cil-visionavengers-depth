# Reading Plan: Monocular Depth Estimation

## Project context
- Course: ETH Computational Intelligence Lab
- Task: monocular depth estimation
- Evaluation: scale-invariant RMSE
- Important caveat: absolute depth scale across images is not meaningful
- Allowed strategy: external pretrained models are allowed
- Planned approach: fine-tune a strong pretrained model + add a small, well-motivated twist
- Practical goal: target score below 0.582 while keeping the report and experiments aligned with the rubric

## Main learning goals
By the end of this reading plan, we should be able to answer:
1. What makes monocular depth estimation difficult?
2. Why is scale ambiguity central to this problem?
3. What does scale-invariant RMSE reward?
4. Which pretrained model families are strongest and most practical?
5. What small twist is realistic for our project?

## Priority order

### Must read
1. Deep Learning-based Depth Estimation Methods from Monocular Images and Videos: A Comprehensive Survey (2024) (skim)
2. Eigen et al. (2014)
3. Monodepth2 (2019)
4. DPT (2021)
5. Depth Anything V2 (2024)
6. SSI Depth (2024)

### Should read
7. Depth Anything (2024)
8. ZoeDepth (2023)

### Nice to read
9. AdaBins (2021)
10. UniDepth / UniDepthV2

---

## Week 1

### Day 1 — Build the map
**Read**
- Deep Learning-based Depth Estimation Methods from Monocular Images and Videos: A Comprehensive Survey (2024)

**Focus**
- task taxonomy
- supervised vs self-supervised vs metric vs relative depth
- common datasets
- common metrics
- historical milestones

**Deliverable**
Write 1–2 lines each for:
- relative depth vs metric depth
- supervised vs self-supervised
- common architectures
- common losses
- common datasets
- common metrics

**Time budget**
- 2–3 hours maximum

### Day 2 — Understand the classic problem
**Read**
- Eigen et al. (2014), *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*

**Focus**
- why monocular depth is ambiguous
- coarse-to-fine prediction
- why absolute scale is unreliable
- intuition behind scale-invariant evaluation

**Deliverable**
- 3-sentence paper summary
- short note on what “scale-invariant” means
- short note on why this matters for our competition

**Time budget**
- 1.5–2 hours

### Day 3 — Learn a major training paradigm
**Read**
- Monodepth2 (2019)

**Focus**
- reprojection loss
- auto-masking
- minimum reprojection
- occlusion/artifact handling

**Deliverable**
- list 3 key training tricks
- explain what failure mode each trick addresses
- note what may transfer to our setup

**Time budget**
- 2 hours

### Day 4 — Learn the modern architecture shift
**Read**
- DPT (2021)

**Focus**
- encoder-decoder structure
- why transformers help dense prediction
- multi-scale feature fusion

**Deliverable**
- model sketch
- note on why DPT improved over older CNN-style models
- note on whether this is relevant for our baseline

**Time budget**
- 1.5–2 hours

### Day 5 — Read one strong supervised milestone
**Read**
- AdaBins (2021)

**Focus**
- adaptive binning idea
- why discretization can help depth prediction
- difference from direct regression

**Deliverable**
- 2–3 line method summary
- strengths and weaknesses
- whether it is worth treating as a report baseline/reference

**Time budget**
- 1.5 hours

### Day 6 — Read one practical modern baseline
**Read**
- Depth Anything (2024)

**Focus**
- pseudo-labeling pipeline
- role of unlabeled data
- relative vs metric depth variants
- practicality for transfer/fine-tuning

**Deliverable**
- note on why it generalizes well
- note on what supervision it uses
- note on whether we should start from this family

**Time budget**
- 1.5 hours

### Day 7 — Read the most practical strong baseline
**Read**
- Depth Anything V2 (2024)

**Focus**
- what changed from V1
- synthetic labeled data + pseudo-labeled real data
- efficiency and model scale choices
- practical fine-tuning relevance

**Deliverable**
- note on why V2 is stronger than V1
- recommended model size for our compute budget
- note on whether this should be our main baseline

**Time budget**
- 2 hours

---

## Week 2

### Day 8 — Read the paper most aligned with our metric
**Read**
- SSI Depth (2024)

**Focus**
- scale-invariant vs SSI formulation
- why high-frequency detail matters
- ordinal/detail-oriented viewpoint
- relevance to our benchmark metric

**Deliverable**
- 3-line summary of the key insight
- note on what can be reused as a small project twist
- note on whether ordinal or edge/detail loss fits our setup

**Time budget**
- 2 hours

### Day 9 — Read one bridge paper
**Read**
- ZoeDepth (2023)

**Focus**
- relative pretraining + metric heads
- multi-domain routing idea
- zero-shot generalization

**Deliverable**
- note on what problem ZoeDepth solves
- note on which ideas still help even if our benchmark is scale-invariant

**Time budget**
- 1.5–2 hours

### Day 10 — Read modern universal depth
**Read**
- UniDepth (2024)
- skim UniDepthV2 (2025)

**Focus**
- camera/intrinsics handling
- 3D point prediction viewpoint
- geometry-aware design
- edge-guided improvements in later work

**Deliverable**
- note on what makes the method “universal”
- note on which parts matter for our setup and which do not
- note on whether geometric or edge-guided ideas inspire a project twist

**Time budget**
- 2–3 hours total

---

## Recommended project direction after reading

### Baseline family
- Start from a strong pretrained model, most likely Depth Anything V2

### Strong baseline + small twist
Most realistic twist candidates:
1. scale-invariant / ordinal auxiliary loss
2. edge-aware refinement loss
3. multi-scale refinement
4. pseudo-label self-training on augmented train data
5. test-time ensembling for leaderboard improvement

### Current recommended default
- Fine-tune Depth Anything V2
- Add a scale-aware or ordinal auxiliary loss
- Optionally add edge-aware refinement if implementation budget allows

---

## Suggested output after finishing the reading plan
1. A 1-page field summary
2. A paper comparison table
3. A baseline choice
4. A shortlist of 2–3 project twists
5. A first implementation plan

---

## Sources
1. Survey: https://arxiv.org/abs/2406.19675
2. Eigen et al. (2014): https://papers.nips.cc/paper_files/paper/2014/hash/91c56ce4a249fae5419b90cba831e303-Abstract.html
3. Monodepth2: https://openaccess.thecvf.com/content_ICCV_2019/html/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.html
4. DPT: https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html
5. AdaBins: https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.html
6. ZoeDepth: https://arxiv.org/abs/2302.12288
7. Depth Anything: https://github.com/LiheYoung/Depth-Anything
8. Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
9. SSI Depth: https://yaksoy.github.io/sidepth/
10. UniDepth: https://github.com/lpiccinelli-eth/UniDepth
11. UniDepthV2: https://arxiv.org/abs/2502.20110

## Project idea notes
- [Geometry-consistent camera tilt augmentation](geometry_consistent_tilt_augmentation.md): detailed brainstorming note on RGB-depth virtual camera tilt augmentation, depth recomputation, FOV assumptions, affine-invariant loss interaction, and ablation plan.
