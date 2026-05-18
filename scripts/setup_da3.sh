#!/usr/bin/env bash
set -euo pipefail

# Minimal Depth Anything 3 setup for DA3 teacher-mask precomputation.
#
# Training code must not import DA3 directly. DA3 is used offline to generate
# teacher reliability masks, and training consumes only the saved mask artifact.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/${USER}/cil-visionavengers-depth}"
DA3_REPO_DIR="${DA3_REPO_DIR:-${REPO_ROOT}/external/Depth-Anything-3}"
DA3_GIT_URL="${DA3_GIT_URL:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
DA3_INSTALL_DEPS="${DA3_INSTALL_DEPS:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<USAGE
Usage: bash scripts/setup_da3.sh

Defaults:
  repo root      : ${REPO_ROOT}
  DA3 source dir : ${DA3_REPO_DIR}
  scratch root   : ${SCRATCH_ROOT}
  HF cache       : ${HF_HOME}
  install deps    : ${DA3_INSTALL_DEPS}

Environment overrides:
  SCRATCH_ROOT, DA3_REPO_DIR, DA3_GIT_URL, HF_HOME, DA3_INSTALL_DEPS
USAGE
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[!] Required command not found: git" >&2
    exit 1
fi

mkdir -p "$(dirname "$DA3_REPO_DIR")" "$HF_HOME"

echo "[*] Repo root      : $REPO_ROOT"
echo "[*] DA3 source dir : $DA3_REPO_DIR"
echo "[*] Scratch root   : $SCRATCH_ROOT"
echo "[*] HF cache       : $HF_HOME"
echo "[*] Install deps   : $DA3_INSTALL_DEPS"

if [[ -d "$DA3_REPO_DIR/.git" ]]; then
    echo "[*] DA3 repo already present: $DA3_REPO_DIR"
elif [[ -e "$DA3_REPO_DIR" ]]; then
    echo "[!] Path exists but is not a git checkout: $DA3_REPO_DIR" >&2
    echo "    Move it aside or set DA3_REPO_DIR to a different location." >&2
    exit 1
else
    git clone "$DA3_GIT_URL" "$DA3_REPO_DIR"
fi

if [[ "$DA3_INSTALL_DEPS" == "1" ]]; then
    python -m pip install -e "$DA3_REPO_DIR"
else
    echo "[*] Skipping DA3 dependency install because DA3_INSTALL_DEPS=$DA3_INSTALL_DEPS"
fi

PYTHONPATH="${DA3_REPO_DIR}/src:${DA3_REPO_DIR}:${PYTHONPATH:-}" HF_HOME="$HF_HOME" python -c "from depth_anything_3.api import DepthAnything3; print('[*] DA3 import check passed:', DepthAnything3.__name__)"

cat <<SUMMARY

[✓] Depth Anything 3 setup complete.

Use these paths when precomputing DA3 teacher masks:
  DA3_REPO=${DA3_REPO_DIR}
  HF_HOME=${HF_HOME}

For filtering details, see:
  dataset/da3/README.md
SUMMARY
