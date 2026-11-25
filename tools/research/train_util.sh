#!/usr/bin/env bash
# What does the training step actually do with the GPU?
#
# The self-play side of a generation is measured by `--stats`; the trainer has no
# such switch, so this samples `nvidia-smi` while a real training run goes past
# and reports the mean/median utilisation per batch size.
#
#   SHARD=runs/training/<run>/gen_003/selfplay/shard_000.bin \
#   PY=/home/vipuser/miniconda3/bin/python \
#       bash tools/research/train_util.sh 256 512 1024
#
# The batch sizes are arguments so the same number can be compared before and
# after a change; the per-epoch wall time comes from the trainer's own
# `epoch N/M ... time=` line.
set -euo pipefail

SHARD="${SHARD:?set SHARD to a self-play shard}"
PY="${PY:-python3}"
EPOCHS="${EPOCHS:-2}"
CSV="${CSV:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for B in "$@"; do
    log="$(mktemp)"
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        --loop-ms=100 > "$log" 2>/dev/null &
    sampler=$!
    echo "== batch $B =="
    /usr/bin/time -f "batch=$B wall=%e s" "$PY" training/train_resnet.py \
        --data "$SHARD" --out "/tmp/train_util_$B.bin" --summary "/tmp/train_util_$B.json" \
        ${CSV:+--log-csv "$CSV"} \
        --channels 96 --blocks 6 --input-planes 37 --ruleset nopass \
        --epochs "$EPOCHS" --batch "$B" --max-samples 600000 --lr 0.0002 \
        --value-weight 1.0 --score-weight 0.5 --val 0.06 --seed 1 --device cuda \
        --a-cond --a-anchor 0.5 --a-range 4.0 --a-anchor-fraction 0.35 --amp \
        2>&1 | grep -E "^epoch |batch=" || true
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    "$PY" - "$log" "$B" <<'PY'
import statistics, sys
values = [float(line) for line in open(sys.argv[1]) if line.strip().isdigit()]
if not values:
    print("batch=%s: no samples" % sys.argv[2])
else:
    print("batch=%s gpu mean=%.0f%% median=%.0f%% max=%.0f%% (n=%d)"
          % (sys.argv[2], statistics.mean(values), statistics.median(values),
             max(values), len(values)))
PY
    rm -f "$log"
done
