#!/usr/bin/env bash
# Start a Shuyi generation run in the background, with logs that survive a
# disconnect, and a machine-readable manifest that makes --resume safe.
#
#   bash experimental/training/run_generation.sh                        # 8 generations
#   GENERATIONS=6 VISITS=2000 bash experimental/training/run_generation.sh
#
# Everything runs on the GPU: positions from shuyi_selfplay_gpu (C++ tree +
# batched CUDA evaluation), training from experimental/training/train_resnet.py.
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
KOMI="${KOMI:--0.5}"
ARENA_GAMES="${ARENA_GAMES:-160}"
BATCH_GAMES="${BATCH_GAMES:-128}"

BIN="$BUILD/shuyi_selfplay_gpu"
[ -x "$BIN" ] || { echo "build first: bash experimental/training/server_setup.sh" >&2; exit 1; }

mkdir -p "$REPO/runs/$RUN_NAME"
LOG="$REPO/runs/$RUN_NAME/pipeline.log"

python3 - "$REPO" "$RUN_NAME" <<'EOF'
import json, sys
repo, name = sys.argv[1], sys.argv[2]
print(json.dumps({"run": name, "started_by": "run_generation.sh"}, indent=2))
EOF

nohup python3 -u "$REPO/experimental/training/pipeline.py" \
    --run-name "$RUN_NAME" \
    --generations "$GENERATIONS" \
    --oneply-gens "$ONEPLY_GENS" \
    --games-per-gen "$GAMES" \
    --visits "$VISITS" \
    --epochs "$EPOCHS" \
    --blocks "$BLOCKS" \
    --channels "$CHANNELS" \
    --komi "$KOMI" \
    --arena-games "$ARENA_GAMES" \
    --batch-games "$BATCH_GAMES" \
    --selfplay-gpu "$BIN" \
    >"$LOG" 2>&1 &

echo "$!" > "$REPO/runs/$RUN_NAME/pipeline.pid"
echo "started pid $(cat "$REPO/runs/$RUN_NAME/pipeline.pid")"
echo "log:  $LOG"
echo "tail: tail -f $LOG"
