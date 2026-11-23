#!/usr/bin/env bash
# Prepare an Ubuntu 22.04/24.04 host with CUDA 12.4 for Shuyi training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== host =="
uname -a
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found; training needs an NVIDIA GPU" >&2
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "WARNING: nvcc not found; PyTorch wheel still works, but install CUDA 12.4 toolkit for a standard setup" >&2
fi

for tool in python3 cmake ninja g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" >&2
        echo "on Ubuntu: sudo apt-get install -y python3 python3-venv cmake ninja-build g++" >&2
        exit 1
    fi
done

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# CUDA 12.4 wheels.  If the host has a different driver/CUDA, change the
# index URL (cu124 -> cu121/cu118) and verify torch.cuda.is_available().
python -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.1"
python -m pip install "numpy>=1.26,<2" "tensorboard>=2.16"

python - <<'PY'
import sys
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: torch cannot see CUDA; check the NVIDIA driver / cu124 wheel", file=sys.stderr)
    sys.exit(1)
print("cuda runtime", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

ctest --test-dir build --output-on-failure

echo
echo "== ready =="
echo "activate: source $ROOT/.venv/bin/activate"
echo "quick check: ./build/shuyi_selfplay --help"
echo "15 generations: bash experimental/training/run_15gen.sh"
