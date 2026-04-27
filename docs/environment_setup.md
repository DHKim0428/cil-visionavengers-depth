# Environment Setup

On the cluster, a central installation of Conda exists for this course. 

## 1. Standard Setup
Add the following initialization block to your `~/.bashrc` file:

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cluster/courses/cil/envs/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/courses/cil/envs/etc/profile.d/conda.sh" ]; then
        . "/cluster/courses/cil/envs/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/courses/cil/envs/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

After saving the file, restart your shell (or run `source ~/.bashrc`), and then execute the following commands to start working:

```bash
module load cuda/12.6.0
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
```

---

## 2. Troubleshooting: Conda Environment Conflicts & GPU Architecture Issues

The cluster has different GPU generations. The standard environment crashes on the new RTX 5060 Ti GPUs (`cudaErrorNoKernelImageForDevice`), while the 5060-specific environment might not be stable on older GPUs. 

Also, you might have other Conda installations from different courses conflicting in your `~/.bashrc`.

**To fix this automatically and manage multiple projects, disable auto-initialization and use a smart bash function instead:**

1. Open your `~/.bashrc` file.
2. **Comment out or delete** all existing `# >>> conda initialize >>> ... # <<< conda initialize <<<` blocks.
3. Add the following function and aliases at the bottom of the file:

```bash
# ---------------------------------------------------------
# 1. Smart Conda activation for CIL Project
# ---------------------------------------------------------
conda_cil() {
    # Initialize CIL Conda quietly
    eval "$(/cluster/courses/cil/envs/bin/conda shell.bash hook 2> /dev/null)"
    
    # Check if nvidia-smi is available and get the GPU name
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        
        # Activate the matching environment
        if [[ "$GPU_NAME" == *"5060"* ]]; then
            echo "[*] Detected $GPU_NAME: Activating 5060-specific environment..."
            conda activate /cluster/courses/cil/envs/envs/monodepth-5060
        else
            echo "[*] Detected $GPU_NAME: Activating standard environment..."
            conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
        fi
    else
        # Fallback if no GPU is found (e.g., login node)
        echo "[*] No GPU detected. Activating standard environment..."
        conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
    fi
}

# ---------------------------------------------------------
# 2. Alias for your other projects (Placeholder)
# ---------------------------------------------------------
# Update the path with your actual other Conda path (e.g., Miniconda in your workspace)
alias conda_other='eval "$(/path/to/your/other/miniconda3/bin/conda shell.bash hook 2> /dev/null)"'
```

4. Save the file and apply the changes:
```bash
source ~/.bashrc
```

**How to start your work:**
From now on, Conda will not start automatically. 

- **For this CIL project:** No matter which GPU node you get allocated, just run:
  ```bash
  module load cuda/12.6.0
  conda_cil
  ```
- **For other projects:** Run `conda_other`, then activate your specific environment (e.g., `conda activate my_env`).