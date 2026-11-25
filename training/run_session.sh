#!/usr/bin/env bash
# One command for a rented GPU window: measure the box, size the run, then run it
# to the clock.
#
#   HOURS=10 MODEL=runs/training/<run>/models/b6c96-s25920-d258451.bin \
#       bash training/run_session.sh
#
# What it does, in order:
#   1. probes self-play throughput with the real model and visit budget
#      (`SHUYI_FAKE_RATE` skips the probe, for checking the arithmetic),
#   2. turns the window into generations x games x epochs (training/budget.py),
#      so the per-generation data and optimizer steps scale with the model,
#   3. runs training/pipeline.py with `--max-hours`, which stops between
#      generations instead of overrunning the rental,
#   4. leaves the plan and the log in the run directory.
#
# Everything after the probe is unattended: the pipeline only needs the GPU and
# the clock.  Nothing here waits for a human.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOURS="${HOURS:-10}"
MODEL="${MODEL:?set MODEL to the weights to continue from}"
VISITS="${VISITS:-1024}"
BLOCKS="${BLOCKS:-6}"
CHANNELS="${CHANNELS:-96}"
RULESET="${RULESET:-nopass}"
KOMI="${KOMI:-0.5}"
LR="${LR:-2e-4}"
REPLAY_GENS="${REPLAY_GENS:-2}"
SCORE_WEIGHT="${SCORE_WEIGHT:-0.5}"
ARENA_GAMES="${ARENA_GAMES:-100}"
# The probe must use the batch the run uses: probing at 8 measured 0.25 games/s
# where the run actually does 0.45, which sizes the whole session for a box half
# as fast as the one that is running.
BATCH_GAMES="${BATCH_GAMES:-32}"
# Independent game groups running concurrently, each with its own model instance
# and CUDA stream.  One group leaves the card idle while the CPU descends and
# expands; more groups overlap those halves.  The probe below measures each
# candidate on this box and keeps the fastest.
# (Not `GROUPS`: that is a bash builtin array, so the assignment would be
# ignored and the flag would receive the caller's group ids.)
PLAY_GROUPS="${PLAY_GROUPS:-1 2}"
SEED="${SEED:-20261201}"
RUN_NAME="${RUN_NAME:-b${BLOCKS}c${CHANNELS}-${RULESET}-$(date +%Y%m%d)}"
PYTHON="${SHUYI_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="${SHUYI_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || { echo "no python: $PYTHON" >&2; exit 1; }
BIN="${SHUYI_SELFPLAY:-$REPO/build/shuyi_selfplay_gpu}"
TRAIN_RATE="${TRAIN_RATE:-20000}"
DRY="${DRY:-0}"
# Parameters from the model file when it exists (exact), from the shape otherwise:
# 37x9 stem + 6 blocks of two 3x3 convs + heads, which is 1.03 M for 6x96.
PARAMS=$("$PYTHON" - "$MODEL" "$BLOCKS" "$CHANNELS" <<'PY'
import os, sys
model, blocks, channels = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if os.path.exists(model):
    print(f"{(os.path.getsize(model) - 36) / 4 / 1e6:.2f}")   # 36 B header, FP32
else:
    params = 37 * 9 * channels + blocks * 18 * channels * channels + 2000
    print(f"{params / 1e6:.2f}")
PY
)

START=$(date +%s)
RUN_DIR="$REPO/runs/training/$RUN_NAME"
mkdir -p "$RUN_DIR/logs"

if [ -n "${SHUYI_FAKE_RATE:-}" ]; then
    RATE="$SHUYI_FAKE_RATE"
    PLAY_GROUPS=$(echo "$PLAY_GROUPS" | awk '{print $1}')
    echo "using fake probe rate $RATE games/s, $PLAY_GROUPS group(s)"
else
    # Probe every candidate group count on the real model and the real visit
    # budget, then keep the fastest.  Overlapping the CPU half of a tick with
    # another group's kernels only pays when the card is idle between calls; on a
    # box that is already saturated the extra groups cost nothing but buy
    # nothing either.  Measuring beats assuming, and it takes minutes.
    BEST_RATE=0
    BEST_GROUPS=$(echo "$PLAY_GROUPS" | awk '{print $1}')
    for g in $PLAY_GROUPS; do
        PROBE="$RUN_DIR/logs/probe_g$g.log"
        echo "probing self-play: $((BATCH_GAMES * g)) games, $VISITS visits, $g group(s)..."
        "$BIN" --mode selfplay --model "$MODEL" --games $((BATCH_GAMES * g)) \
            --batch-games "$BATCH_GAMES" --groups "$g" --stats \
            --visits "$VISITS" --ruleset "$RULESET" --komi "$KOMI" \
            --root-k 32 --top-k 16 --noise 0.25 --dirichlet 0.3 \
            --temp-moves 12 --temperature 1.0 --opening 2 --seed "$SEED" \
            --chunk 1024 --out "$RUN_DIR/logs/probe.bin" > "$PROBE" 2>&1
        r=$(grep -o 'games/s=[0-9.]*' "$PROBE" | head -1 | cut -d= -f2)
        [ -n "$r" ] || { echo "no games/s in $PROBE" >&2; exit 1; }
        echo "  groups=$g: $r games/s  $(grep -o 'gpu=[0-9]*%' "$PROBE" | head -1)"
        if awk "BEGIN{exit !($r > $BEST_RATE)}"; then
            BEST_RATE="$r"
            BEST_GROUPS="$g"
        fi
    done
    rm -f "$RUN_DIR/logs/probe.bin"
    RATE="$BEST_RATE"
    PLAY_GROUPS="$BEST_GROUPS"
    echo "fastest: $RATE games/s with $PLAY_GROUPS group(s)"
fi

"$PYTHON" "$REPO/training/budget.py" --rate "$RATE" --hours "$HOURS" --params "$PARAMS" \
    --visits "$VISITS" --arena-games "$ARENA_GAMES" --train-rate "$TRAIN_RATE" \
    --replay-gens "$REPLAY_GENS" \
    --json > "$RUN_DIR/plan.json"
"$PYTHON" "$REPO/training/budget.py" --rate "$RATE" --hours "$HOURS" --params "$PARAMS" \
    --visits "$VISITS" --arena-games "$ARENA_GAMES" --train-rate "$TRAIN_RATE" \
    --replay-gens "$REPLAY_GENS"

# Read the plan from inside the run directory: a run path may contain non-ASCII
# characters, and passing it through `python -c` breaks on Windows code pages.
GENS=$(cd "$RUN_DIR" && "$PYTHON" -c 'import json;print(json.load(open("plan.json"))["generations"])')
GAMES=$(cd "$RUN_DIR" && "$PYTHON" -c 'import json;print(json.load(open("plan.json"))["games_per_generation"])')
EPOCHS=$(cd "$RUN_DIR" && "$PYTHON" -c 'import json;print(json.load(open("plan.json"))["epochs"])')
ELAPSED=$(( $(date +%s) - START ))
RACE_HOURS=$("$PYTHON" -c "print(f'{(($HOURS * 3600) - $ELAPSED) / 3600:.3f}')")

echo "launching $GENS generations x $GAMES games x $EPOCHS epochs, deadline ${RACE_HOURS}h from now"

RESUME=""
if [ -f "$RUN_DIR/manifest.json" ]; then
    RESUME="--resume"
    echo "run directory already has a manifest: resuming it"
fi
EXTRA=""
[ "$DRY" = "1" ] && EXTRA="--dry-run"

exec "$PYTHON" "$REPO/training/pipeline.py" \
    --run-name "$RUN_NAME" \
    --blocks "$BLOCKS" --channels "$CHANNELS" \
    --generations "$GENS" \
    --games-per-gen "$GAMES" \
    --visits "$VISITS" \
    --batch-games "$BATCH_GAMES" \
    --groups "$PLAY_GROUPS" \
    --komi "$KOMI" --ruleset "$RULESET" \
    --a-cond --a-anchor "$KOMI" \
    --score-weight "$SCORE_WEIGHT" \
    --replay-gens "$REPLAY_GENS" \
    --lr "$LR" --epochs "$EPOCHS" \
    --arena-games "$ARENA_GAMES" \
    --oneply-gens 0 \
    --elo-every 0 \
    --max-hours "$RACE_HOURS" \
    --init-model "$MODEL" \
    $EXTRA $RESUME
