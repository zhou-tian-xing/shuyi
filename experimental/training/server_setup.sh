#!/usr/bin/env bash
# Provision and verify a fresh GPU box for Shuyi, then print the cost plan.
#
# Everything that generates positions (self-play, one-ply scan, arena) runs in
# one C++/CUDA binary, and training runs in PyTorch on the same GPU.  This
# script only checks the prerequisites, builds the binary, verifies it against
# the CPU reference, and measures throughput so the run can be sized from
# numbers instead of guesses.
#
#   bash experimental/training/server_setup.sh                # check + build + verify + bench
#   bash experimental/training/server_setup.sh --skip-apt     # no sudo/apt, build only
#
# Idempotent: safe to re-run after a reboot or a pull.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${SHUYI_BUILD_DIR:-$REPO/build}"
SKIP_APT=0
for arg in "$@"; do
    case "$arg" in
        --skip-apt) SKIP_APT=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

log "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: this machine has no usable NVIDIA driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

log "CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    else
        cat >&2 <<'EOF'
nvcc not found.  Install the CUDA toolkit (12.1+), for example:
  sudo apt install -y cuda-toolkit-12-4
  export PATH=/usr/local/cuda/bin:$PATH
EOF
        exit 1
    fi
fi
nvcc --version | tail -2

if [ "$SKIP_APT" = "0" ]; then
    log "packages"
    need=()
    command -v cmake >/dev/null 2>&1 || need+=(cmake)
    command -v ninja >/dev/null 2>&1 || need+=(ninja-build)
    command -v g++ >/dev/null 2>&1 || need+=(g++)
    if [ "${#need[@]}" -gt 0 ]; then
        sudo apt-get update -qq
        sudo apt-get install -y "${need[@]}"
    fi
fi

log "python"
python3 - <<'EOF'
import sys
try:
    import torch
except ImportError:
    print("PyTorch missing.  Install it, for example:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the GPU; fix that before training")
EOF

log "build (SHUYI_CUDA=ON)"
cmake -S "$REPO" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHUYI_CUDA=ON
cmake --build "$BUILD" --target shuyi_selfplay_gpu shuyi_tests

log "unit tests"
"$BUILD/shuyi_tests"

PROBE="$BUILD/models/probe.bin"
mkdir -p "$BUILD/models"
if [ ! -f "$PROBE" ]; then
    python3 "$REPO/experimental/training/train_resnet.py" --init-only --out "$PROBE" \
        --channels 32 --blocks 4 --input-planes 37
fi

log "cuda vs cpu parity"
"$BUILD/shuyi_selfplay_gpu" --model "$PROBE" --check \
    --dump-features "$BUILD/layout_check.bin"

log "engine planes == trainer planes"
python3 "$REPO/experimental/training/check_layout.py" "$BUILD/layout_check.bin"

log "throughput"
"$BUILD/shuyi_selfplay_gpu" --model "$PROBE" --mode bench

cat <<'EOF'

Next:
  python3 experimental/training/pipeline.py --plan --channels 32 --blocks 4   # cost table
  bash    experimental/training/run_generation.sh                             # start the run
EOF
