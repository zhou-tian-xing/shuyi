#!/usr/bin/env bash
# Start a Shuyi generation run in the background, with logs that survive a
# disconnect, and a machine-readable manifest that makes --resume safe.
#
#   bash training/run_generation.sh                        # 8 generations
#   GENERATIONS=6 VISITS=3000 bash training/run_generation.sh
#   INIT_MODEL=models/b4c32-s34760-d470705.bin \
#   INIT_DATA="runs/training/b4c32-blocking-20261122/gen_00*/selfplay" bash training/run_generation.sh
#
# INIT_MODEL/INIT_DATA continue from an existing model instead of starting a
# chain from a random net: generation 0 fine-tunes INIT_MODEL on the shards in
# INIT_DATA (and is measured against INIT_MODEL in the arena).
#
# Everything runs on the GPU: positions from shuyi_selfplay_gpu (C++ tree +
# batched CUDA evaluation), training from training/train_resnet.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${SHUYI_BUILD_DIR:-$REPO/build}"
GENERATIONS="${GENERATIONS:-8}"
ONEPLY_GENS="${ONEPLY_GENS:-2}"
GAMES="${GAMES:-2000}"
VISITS="${VISITS:-2000}"
EPOCHS="${EPOCHS:-40}"
BLOCKS="${BLOCKS:-4}"
CHANNELS="${CHANNELS:-32}"
# Empty -> the pipeline picks the measured value per rule set (blocking
# +1.3, classic -0.5).  A hard-coded default here silently overrode that.
KOMI="${KOMI:-}"
# nopass is the product rule set: no pass anywhere, a side with no legal
# placement ends the game immediately.  Every run defaults to it.
RULESET="${RULESET:-nopass}"
# One directory per run: b4c32-nopass-20261124.  Architecture, rule set, date --
# the same three things the model name carries, in the same order.
RUN_NAME="${RUN_NAME:-b${BLOCKS}c${CHANNELS}-${RULESET}-$(date +%Y%m%d)}"
ARENA_GAMES="${ARENA_GAMES:-160}"
BATCH_GAMES="${BATCH_GAMES:-128}"
# A as a network input (model format v4).  Needs SYSELF4 data, which every run
# made after 2026-11-23 produces.  A_ANCHOR empty -> the pipeline's measured
# default (blocking +1.26, classic -0.5).
A_COND="${A_COND:-1}"
A_ANCHOR="${A_ANCHOR:-}"
# Continue from an existing model instead of a random chain start.  INIT_DATA
# may list several shard files, globs or directories (space separated).
INIT_MODEL="${INIT_MODEL:-}"
INIT_DATA="${INIT_DATA:-}"
# Empty -> the pipeline defaults (lr 2e-3 is a from-scratch rate: a warm start
# needs something like 5e-4 or it throws the old weights away).
LR="${LR:-}"
# Continue a run whose manifest exists (a window that ran out mid-chain, or an
# extra generation on the same box).  Without this the wrapper can only start new
# runs: the pipeline refuses an existing directory, and re-running the wrapper
# would silently be "start over" instead of "continue".
RESUME="${RESUME:-0}"
ELO_EVERY="${ELO_EVERY:-}"
ELO_VISITS="${ELO_VISITS:-}"

# The interpreter that has torch: setup_ubuntu.sh installs into .venv, so a run
# must not fall back to the system python3 (numpy/torch are not there, and the
# failure shows up as ModuleNotFoundError two minutes into generation 0).
PYTHON="${SHUYI_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then
        PYTHON="$REPO/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
"$PYTHON" -c "import torch, numpy" 2>/dev/null || {
    echo "no torch/numpy in $PYTHON: run bash training/setup_ubuntu.sh first" >&2
    exit 1
}

BIN="$BUILD/shuyi_selfplay_gpu"
[ -x "$BIN" ] || { echo "build first: bash training/server_setup.sh" >&2; exit 1; }

mkdir -p "$REPO/runs/training/$RUN_NAME"
LOG="$REPO/runs/training/$RUN_NAME/pipeline.log"

EXTRA=()
if [ "$RESUME" = "1" ]; then
    EXTRA+=(--resume)
fi
if [ "$A_COND" = "1" ]; then
    EXTRA+=(--a-cond)
    if [ -n "$A_ANCHOR" ]; then
        EXTRA+=(--a-anchor "$A_ANCHOR")
    fi
fi
if [ -n "$RULESET" ]; then
    EXTRA+=(--ruleset "$RULESET")
fi
if [ -n "$KOMI" ]; then
    EXTRA+=(--komi "$KOMI")
fi
if [ -n "$LR" ]; then
    EXTRA+=(--lr "$LR")
fi
if [ -n "$ELO_EVERY" ]; then
    EXTRA+=(--elo-every "$ELO_EVERY")
fi
if [ -n "$ELO_VISITS" ]; then
    EXTRA+=(--elo-visits "$ELO_VISITS")
fi
if [ -n "$INIT_MODEL" ]; then
    if [ -z "$INIT_DATA" ]; then
        echo "INIT_MODEL needs INIT_DATA (shards the warm-start generation trains on)" >&2
        exit 1
    fi
    EXTRA+=(--init-model "$INIT_MODEL")
    read -r -a INIT_SHARDS <<< "$INIT_DATA"
    EXTRA+=(--init-data "${INIT_SHARDS[@]}")
fi

nohup "$PYTHON" -u "$REPO/training/pipeline.py" \
    --python "$PYTHON" \
    --run-name "$RUN_NAME" \
    --generations "$GENERATIONS" \
    --oneply-gens "$ONEPLY_GENS" \
    --games-per-gen "$GAMES" \
    --visits "$VISITS" \
    --epochs "$EPOCHS" \
    --blocks "$BLOCKS" \
    --channels "$CHANNELS" \
    --arena-games "$ARENA_GAMES" \
    --batch-games "$BATCH_GAMES" \
    --selfplay-gpu "$BIN" \
    "${EXTRA[@]}" \
    >"$LOG" 2>&1 &

echo "$!" > "$REPO/runs/training/$RUN_NAME/pipeline.pid"
echo "started pid $(cat "$REPO/runs/training/$RUN_NAME/pipeline.pid")"
echo "log:  $LOG"
echo "tail: tail -f $LOG"
