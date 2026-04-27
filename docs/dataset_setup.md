# Dataset setup

The training data for our **Monocular Depth Estimation** project (Project 3) is provided by the course and is already available on the cluster.

## 1. Accessing Data on the Cluster (Recommended)
You can find the dataset directly on the student cluster at the following path:

```bash
/cluster/courses/cil/monocular-depth-estimation
```

You can point your dataset paths in the training scripts directly to this directory.

**⚠️ Important Note on Storage:** Please **do not** copy the dataset into your Git repository or home directory (`~/workspace/...`). 
If you need to heavily preprocess the data or experience slow I/O during training, copy the dataset to your temporary scratch space instead:

```bash
# Example: Copying data to your scratch directory
mkdir -p /work/scratch/$USER/cil-visionavengers-depth/data
cp -r /cluster/courses/cil/monocular-depth-estimation/* /work/scratch/$USER/cil-visionavengers-depth/data/
```

## 2. Accessing Data via Kaggle
Alternatively, the dataset is also hosted on Kaggle. If you want to download the data to your local machine for local debugging, please refer to the official Kaggle link provided in the course material or project specification.