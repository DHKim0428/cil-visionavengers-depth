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

## 2. Troubleshooting: Conda Environment Conflicts
If you encounter errors like `import: command not found` or `syntax error near unexpected token 'sys.argv'` when trying to activate the environment, you likely have multiple Conda installations conflicting in your `~/.bashrc` (e.g., from other courses or personal projects).

**To fix this, disable auto-initialization and use aliases instead:**

1. Open your `~/.bashrc` file.
2. **Comment out or delete** all existing `# >>> conda initialize >>> ... # <<< conda initialize <<<` blocks.
3. Add the following aliases at the bottom of the file (update the second path with your actual other Conda path):

```bash
# Alias for this CIL project
alias conda_cil='eval "$(/cluster/courses/cil/envs/bin/conda shell.bash hook 2> /dev/null)"'

# Alias for your other project (Placeholder - update the path accordingly)
alias conda_other='eval "$(/path/to/your/other/miniconda3/bin/conda shell.bash hook 2> /dev/null)"'
```

4. Save the file and apply the changes by running:
```bash
source ~/.bashrc
```

**How to run the project using the alias:**
From now on, whenever you open a new terminal, Conda will not activate automatically. To work on this project, initialize the specific Conda environment first using your alias:

```bash
# 1. Initialize the CIL Conda system
conda_cil

# 2. Load the module and activate the environment
module load cuda/12.6.0
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
```