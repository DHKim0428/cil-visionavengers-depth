# Cluster setup

## Cluster workspace and storage
The repository should be cloned under the home directory:

```bash
~/workspace/cil-visionavengers-depth
```

Large files should not be stored in the repository. Use the scratch filesystem for datasets, intermediate predictions, temporary training logs, and model checkpoints:

```bash
/work/scratch/$USER/cil-visionavengers-depth
```

Scratch storage is temporary and cleaned automatically depending on how much space is used:

| Scratch usage | Maximum age |
|---|---|
| Less than 10 GB | 7 days |
| 10 GB to 50 GB | 2 days |
| More than 50 GB | 1 day |

The cleanup job starts every day at `23:00`. Anything stored in scratch should be reproducible from code, configuration files, or external downloads.

Important checkpoints, selected logs, final predictions, and report artifacts should be copied to a persistent location under the home directory before they are needed for submission:

```bash
~/workspace/results/cil-visionavengers-depth
```

Suggested layout:

```text
~/workspace/
├── cil-visionavengers-depth/              # Git repository: code, configs, docs
└── results/
    └── cil-visionavengers-depth/          # Persistent important outputs
        ├── final_predictions/
        ├── selected_checkpoints/
        ├── selected_logs/
        └── report_artifacts/

/work/scratch/$USER/
└── cil-visionavengers-depth/              # Temporary large files
    ├── data/
    ├── checkpoints/
    ├── logs/
    └── outputs/
```

## Student cluster usage
The ETH student cluster uses SLURM. Use the CIL course tags when requesting compute resources:

| Course tag | Intended use | Runtime limit |
|---|---|---|
| `cil` | Short interactive GPU sessions | 60 minutes |
| `cil_jobs` | Longer GPU jobs or tunnel sessions | Up to 24 hours |

### SSH access from your local machine
From your local terminal, connect to the student cluster login node with your ETH username:

```bash
ssh <ETH_USERNAME>@student-cluster.inf.ethz.ch
```

### Job execution options
There are several ways to run work on the cluster:

| Method | Behavior | Typical use |
|---|---|---|
| `srun <command>` | Runs one command and keeps the terminal attached until it finishes | Quick checks and short tests |
| `srun --pty ... bash --login` | Opens an interactive shell on a compute node | Manual setup, debugging, lightweight interactive work |
| `sbatch <script>` | Submits a script to the queue and returns to the login node; the job starts when resources become available | Longer training runs and reproducible experiments |
| `cluster-tunnel` | Starts a SLURM job and lets you SSH into the allocated compute node | Remote development, VS Code, interactive GPU work |

For quick debugging, use `srun` or `cluster-tunnel`. For full training runs, prefer `sbatch` so the job can wait in the queue and run without keeping your terminal attached.

### Interactive shell with `srun`
Use `srun` for quick interactive work on a compute node. The session is tied to the terminal, so it will stop if the terminal disconnects.

```bash
srun --pty -A cil -t 60 bash --login
```

For a longer interactive allocation, use the jobs course tag:

```bash
srun --pty -A cil_jobs -t 2:00:00 bash --login
```

To explicitly request no GPU:

```bash
srun --gpus=0 --pty -A cil -t 60 bash --login
```

### GPU session via `cluster-tunnel`
For longer GPU work, use `cluster-tunnel` to allocate a GPU node and connect to it from your local machine, for example through SSH or VS Code Remote SSH.

First-time setup only: connect to the login node:

```bash
ssh <ETH_USERNAME>@student-cluster.inf.ethz.ch
```

Then run the tunnel configuration command:

```bash
cluster-tunnel config
```

This command prints two pieces of SSH configuration for your local machine:

```text
1. A host key line to add to your local ~/.ssh/known_hosts
2. A Host cluster-tunnel block to add to your local ~/.ssh/config
```

Copy them into the corresponding files on your local machine. This only needs to be done once.

Back at the student cluster login node, start a tunnel job:

```bash
cluster-tunnel start -A cil_jobs --time=2:00:00
```

The tunnel runs as a SLURM job. If your SSH connection to the login node disconnects, the allocated job can keep running until its time limit is reached or you cancel it.

After the tunnel is active, connect from your local machine:

```bash
ssh cluster-tunnel
```

This opens a shell directly on the allocated compute node, where you can run commands interactively.

### GPU type notes
By default, the cluster scheduler will choose an available GPU. Only request a specific GPU type when there is a clear reason.

The `gb10` nodes have a different architecture. They can be useful when a job needs a larger combined CPU/GPU memory budget, but they are not the default recommendation because environment compatibility and performance characteristics may differ from the regular GPU nodes.

Example request for a `gb10` GPU:

```bash
cluster-tunnel start --gpus gb10:1 -A cil_jobs --time=2:00:00
```

Resource limits (RAM, CPU cores, /tmp space, runtime cap) vary by GPU type. See the [student cluster documentation](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster) for the full table.

### Useful commands
```bash
hostname                                      # Check which node you are on
nvidia-smi                                    # Check GPU availability
scontrol show job $SLURM_JOB_ID | grep -i mem # Check memory allocated to the current SLURM job
courses                                       # Show available course tags, time limits, and compute resources
space                                         # Show writable storage locations
cluster-tunnel status                         # Check whether a tunnel job is active and find the job ID
scancel <JOBID>                               # Cancel a running SLURM job, including a tunnel job
```