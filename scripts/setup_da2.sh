#!/usr/bin/env bash
set -euo pipefail

# Reproducible Depth Anything V2 asset setup for the ETH student cluster.
#
# The upstream source checkout lives inside this repository under external/ so
# local code can use a stable relative path.  Large pretrained checkpoints live
# in scratch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRATCH_ROOT="${SCRATCH_ROOT:-/work/scratch/${USER}/cil-visionavengers-depth}"
DA2_REPO_DIR="${DA2_REPO_DIR:-${REPO_ROOT}/external/Depth-Anything-V2}"
DA2_CKPT_DIR="${DA2_CKPT_DIR:-${SCRATCH_ROOT}/models/da2}"
DA2_GIT_URL="${DA2_GIT_URL:-https://github.com/DepthAnything/Depth-Anything-V2.git}"
DA2_ENCODERS="${DA2_ENCODERS:-vits vitb}"

usage() {
    cat <<USAGE
Usage: bash scripts/setup_da2.sh [--encoders "vits vitb [vitl]"]

Defaults:
  repo root      : ${REPO_ROOT}
  DA2 source dir : ${DA2_REPO_DIR}
  scratch root   : ${SCRATCH_ROOT}
  checkpoint dir : ${DA2_CKPT_DIR}
  encoders       : ${DA2_ENCODERS}

Examples:
  bash scripts/setup_da2.sh
  bash scripts/setup_da2.sh --encoders "vits vitb vitl"

Environment overrides:
  SCRATCH_ROOT, DA2_REPO_DIR, DA2_CKPT_DIR, DA2_GIT_URL, DA2_ENCODERS
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoders)
            DA2_ENCODERS="${2//,/ }"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[!] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[!] Required command not found: $1" >&2
        exit 1
    fi
}

checkpoint_url() {
    case "$1" in
        vits) echo "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" ;;
        vitb) echo "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" ;;
        vitl) echo "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" ;;
        *)
            echo "[!] Unsupported encoder: $1 (expected one of: vits vitb vitl)" >&2
            exit 2
            ;;
    esac
}

download_file() {
    local url="$1"
    local output="$2"

    if [[ -s "$output" ]]; then
        echo "[*] Checkpoint already present: $output"
        return
    fi

    local tmp="${output}.part"
    rm -f "$tmp"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --output "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        echo "[!] Need either curl or wget to download checkpoints." >&2
        exit 1
    fi

    mv "$tmp" "$output"
    echo "[*] Downloaded checkpoint: $output"
}

require_cmd git
mkdir -p "$(dirname "$DA2_REPO_DIR")" "$DA2_CKPT_DIR"

echo "[*] Repo root      : $REPO_ROOT"
echo "[*] DA2 source dir : $DA2_REPO_DIR"
echo "[*] Scratch root   : $SCRATCH_ROOT"
echo "[*] Checkpoints    : $DA2_CKPT_DIR"
echo "[*] Encoders       : $DA2_ENCODERS"

if [[ -d "$DA2_REPO_DIR/.git" ]]; then
    echo "[*] DA2 repo already present: $DA2_REPO_DIR"
elif [[ -e "$DA2_REPO_DIR" ]]; then
    echo "[!] Path exists but is not a git checkout: $DA2_REPO_DIR" >&2
    echo "    Move it aside or set DA2_REPO_DIR to a different location." >&2
    exit 1
else
    git clone "$DA2_GIT_URL" "$DA2_REPO_DIR"
fi

for encoder in $DA2_ENCODERS; do
    output="${DA2_CKPT_DIR}/depth_anything_v2_${encoder}.pth"
    download_file "$(checkpoint_url "$encoder")" "$output"
done

cat <<SUMMARY

[✓] Depth Anything V2 asset setup complete.

Use these paths in later configs/scripts:
  DA2_REPO=${DA2_REPO_DIR}
  DA2_CKPT_DIR=${DA2_CKPT_DIR}

Recommended one-time W&B login before training:
  wandb login
SUMMARY
