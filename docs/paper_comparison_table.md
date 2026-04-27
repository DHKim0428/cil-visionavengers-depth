# Paper Comparison Table

| Paper | Year | Supervision | Relative / Metric | Core Idea | Strength | Weakness | Relevance to Our Project |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deep Learning-based Depth Estimation Methods from Monocular Images and Videos: A Comprehensive Survey | 2024 | N/A | Both | Field overview and taxonomy | Good mental map | Not an implementation target | Helps scope the field |
| Eigen et al. | 2014 | Supervised | Relative-oriented | Multi-scale single-image depth prediction | Foundational treatment of scale ambiguity | Old architecture | Important for understanding scale-invariant depth |
| Monodepth2 | 2019 | Self-supervised | Relative | Reprojection-based monocular training with key masking tricks | Highly influential training ideas | Different setting from our labeled data | Useful conceptual training reference |
| DPT | 2021 | Supervised | Relative / dense prediction | Transformer-based dense prediction | Strong architecture reference | Not the newest model family | Good baseline architecture reference |
| AdaBins | 2021 | Supervised | Metric-oriented | Adaptive discretization of depth bins | Clever output parameterization | Less aligned with recent foundation-model trend | Useful report baseline/reference |
| ZoeDepth | 2023 | Mixed / transfer | Relative + metric | Combine relative depth priors with metric heads | Strong generalization insight | More metric-depth focused | Useful bridge paper |
| Depth Anything | 2024 | Large-scale pseudo-labeling | Relative / metric variants | Large-scale data curation and strong pretrained depth models | Strong practical transfer | May need careful adaptation | Good modern baseline family |
| Depth Anything V2 | 2024 | Large-scale pseudo-labeling + synthetic data | Relative / metric variants | Stronger and more practical version of Depth Anything | Likely strongest starting baseline | May still need project-specific tuning | Top candidate baseline |
| SSI Depth | 2024 | Supervised | Scale-invariant | Reformulates learning toward scale-invariant depth/detail | Highly aligned with our metric | Need to check implementation complexity | Strong source of twist ideas |
| UniDepth / UniDepthV2 | 2024 / 2025 | Supervised / universal | Metric-oriented | Geometry-aware universal depth estimation | Strong modern robustness | Less directly aligned with our metric | Good inspiration/reference |

## Notes
- Add exact benchmark numbers only after verifying the settings carefully.
- Use this table later in the report for related work and method motivation.
