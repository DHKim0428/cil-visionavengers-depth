# cil-visionavengers-depth

VisionAvengers CIL monocular depth-estimation repository.

## Clone

For a fresh checkout, clone with the DA2 submodule:

```bash
git clone --recurse-submodules https://github.com/DHKim0428/cil-visionavengers-depth.git
cd cil-visionavengers-depth
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Setup

```bash
module load cuda/12.6.0
conda_cil
python -m pip install -r requirements.txt
wandb login
bash scripts/setup_da2.sh
```

If `conda_cil` is not available, see `docs/environment_setup.md`.

## Main commands

Python entrypoints live at the repository root:

```bash
python train.py --config configs/experiments/da2_vits_refinenets_output.yaml
python eval.py --config configs/experiments/da2_vits_zero_shot.yaml
```

SLURM wrappers live under `scripts/slurm/`:

```bash
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_refinenets_output.yaml,RUN_NAME=da2_refinenets scripts/slurm/train.sbatch
sbatch --export=ALL,CONFIG=configs/experiments/da2_vits_zero_shot.yaml,RUN_NAME=da2_eval scripts/slurm/eval.sbatch
```

## Evaluation semantics

The canonical metric is implemented locally from `docs/project_spec.md`:
siRMSE is computed on ground-truth-valid pixels, where `0.001 <= gt <= 80`.
DA2 upstream code is used for model loading/inference preprocessing, but not for
the project metric. DA2 relative outputs are treated directly as positive
relative depth, not inverted as disparity.

## Current structure

```text
train.py                 # canonical training entrypoint
eval.py                  # canonical siRMSE evaluation entrypoint
configs/                 # experiment and augmentation YAMLs
dataset/                 # data_loader.py, data_augment.py, and dataset preprocessing utilities
models/                  # DA2 and U-Net model helpers
utils/                   # config/wandb helpers and siRMSE
scripts/                 # shell and SLURM wrappers only
legacy/                  # old or superseded code kept for reference
```

## Docs

- `docs/project_spec.md`
- `docs/environment_setup.md`
- `docs/cluster_setup.md`
- `docs/dataset_setup.md`
- `docs/da2_setup.md`
