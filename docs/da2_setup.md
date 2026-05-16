# DA2 setup on the student cluster

This project uses the official **Depth Anything V2 relative-depth** codebase for
DA2 experiments.  The upstream project provides relative-depth checkpoints for
`vits`, `vitb`, and `vitl`, and expects checkpoints named
`depth_anything_v2_{encoder}.pth`.  Our current refactor keeps that upstream
contract while making the cluster setup reproducible.

## Storage policy

The repository itself stays in the home workspace, and the external DA2 source
checkout is cloned inside it under a gitignored directory:

```text
~/workspace/cil-visionavengers-depth/
└── external/
    └── Depth-Anything-V2/
```

Large DA2 checkpoints live in scratch:

```text
/work/scratch/$USER/cil-visionavengers-depth/
└── models/
    └── da2/
        ├── depth_anything_v2_vits.pth
        ├── depth_anything_v2_vitb.pth
        └── ...
```

This keeps a stable repo-relative import path for DA2 source code while keeping
large pretrained weights out of the Git checkout.

## One-time / recovery setup

From the project repository, run:

```bash
cd ~/workspace/cil-visionavengers-depth
bash scripts/setup_da2.sh
```

By default this prepares the relative-depth `vits` and `vitb` checkpoints,
which are the variants already referenced by the current repository.  If you
also need the large model:

```bash
bash scripts/setup_da2.sh --encoders "vits vitb vitl"
```

The script is intended to be idempotent:

- if the upstream repo already exists, it keeps it;
- if a requested checkpoint already exists and is non-empty, it keeps it;
- if the upstream checkout already exists at `external/Depth-Anything-V2/`, it keeps it;
- if scratch was cleaned, rerunning the script recreates the missing pieces.

After setup, later configs/scripts should use:

```text
DA2_REPO=~/workspace/cil-visionavengers-depth/external/Depth-Anything-V2
DA2_CKPT_DIR=/work/scratch/$USER/cil-visionavengers-depth/models/da2
```

The refactor will remove the current user-specific hard-coded DA2 paths and use
this shared contract instead.

## Environment preparation before DA2 setup

Prepare the course environment and install the repository-owned extra
dependencies before fetching DA2 assets:

```bash
module load cuda/12.6.0
conda_cil
python -m pip install -r requirements.txt
wandb login
```

New canonical training entrypoints are intended to use **Weights & Biases
(W&B)** by default.

Future canonical training scripts should warn at startup if no W&B login is
detected, so an untracked run is not accidental.

## DA2 upstream requirements

The official DA2 upstream repository has its own `requirements.txt`, but the
refactor intentionally does **not** install it yet.  We first rely on the course
Conda environment plus this repository's `requirements.txt`.  If a later partial
fine-tuning smoke test fails because an upstream-only dependency is missing, we
will revisit that decision with evidence instead of installing an extra stack
preemptively.

## Before moving on

Before starting the next refactor phase, run the environment preparation once
and run the asset setup script end-to-end.  During Phase 1 we verified:

- `conda_cil` activates the standard course environment on a login node;
- `python -m pip install -r requirements.txt` succeeds;
- `bash scripts/setup_da2.sh --encoders "vits"` clones the upstream repo and
  downloads the `vits` checkpoint;
- rerunning the same setup command reuses the existing repo and checkpoint.
