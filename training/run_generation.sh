#!/usr/bin/env bash
# Start a Shuyi generation run in the background, with logs that survive a
# disconnect, and a machine-readable manifest that makes --resume safe.
#
#   bash training/run_generation.sh                        # 8 generations
#   GENERATIONS=6 VISITS=3000 bash training/run_generation.sh
#   INIT_MODEL=models/b4c32-s34760-d470705.bin \
#   INIT_DATA="runs/blocking/blk1/gen_00*/selfplay" bash training/run_generation.sh
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
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
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

BIN="$BUILD/shuyi_selfplay_gpu"
[ -x "$BIN" ] || { echo "build first: bash training/server_setup.sh" >&2; exit 1; }

mkdir -p "$REPO/runs/$RUN_NAME"
LOG="$REPO/runs/$RUN_NAME/pipeline.log"

python3 - "$REPO" "$RUN_NAME" <<'EOF'
import json, sys
repo, name = sys.argv[1], sys.argv[2]
print(json.dumps({"run": name, "started_by": "run_generation.sh"}, indent=2))
EOF

EXTRA=()
if [ "$A_COND" = "1" ]; then
    EXTRA+=(--a-cond)
    if [ -n "$A_ANCHOR" ]; then
        EXTRA+=(--a-anchor "$A_ANCHOR")
    fi
fi
if [ -n "$KOMI" ]; then
    EXTRA+=(--komi "$KOMI")
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

nohup python3 -u "$REPO/training/pipeline.py" \
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

echo "$!" > "$REPO/runs/$RUN_NAME/pipeline.pid"
echo "started pid $(cat "$REPO/runs/$RUN_NAME/pipeline.pid")"
echo "log:  $LOG"
echo "tail: tail -f $LOG"
