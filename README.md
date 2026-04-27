# cil-visionavengers-depth

This is the **VisionAvengers** team project repository for the ETH Computational Intelligence Lab **Monocular Depth Estimation** project. It contains our code, experiments, and report material.

## Index
- [Getting started](#getting-started)
- [Documentation](#documentation)
- [Baseline Model: UNetBaseline](#baseline-model-unetbaseline)

## Getting started
After logging in to the student cluster, create a workspace directory under your home directory and clone this repository:

```bash
cd ~
mkdir -p workspace
cd workspace
git clone https://github.com/DHKim0428/cil-visionavengers-depth.git
cd cil-visionavengers-depth
```

Here, `~` refers to the current user's home directory, for example `/home/$USER`.

## Documentation
- [Official project specification](docs/project_spec.md)
- [Cluster setup](docs/cluster_setup.md) — general cluster usage, workspace layout, SLURM usage, GPU sessions
- [Environment setup](docs/environment_setup.md) — Conda initialization, environment activation, and troubleshooting conflicts
- [Dataset setup](docs/dataset_setup.md)
- [ETH student cluster documentation](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster)
- [ETH student cluster job documentation](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)

## Baseline Model: UNetBaseline

We have implemented a standard **U-Net** architecture with **Batch Normalization** to serve as a robust baseline for this project. This model is significantly more capable than the initial `TinyUNet` provided in the demo, offering better convergence and higher capacity.

### Key Features
- **Architecture**: Standard U-Net with 4 encoder/decoder levels.
- **Capacity**: Channels scale from 32 up to 256 in the bottleneck.
- **Stability**: Integrated Batch Normalization after each convolution layer.
- **Loss Function**: Scale-Invariant Log RMSE (SILog).

### Performance (Default Settings)
Below are the results achieved using the default parameters provided in `train.py`:

| Parameter | Value |
| :--- | :--- |
| Image Size | 128 x 128 |
| Batch Size | 8 |
| Learning Rate | 1e-3 |
| Max Epochs | 10 |
| Val Split | 20% |

**Results:**
- **Best Validation SILog Loss**: `0.6709` *(Achieved at the final epoch (Epoch 10), indicating that the model has not yet fully converged and training for more epochs will likely improve performance further.)*
- **Training Time**: Approx. 35 mins on RTX 5060 Ti.

---