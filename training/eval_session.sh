#!/usr/bin/env bash
# 一轮补数据的评测：每个预算跑一遍梯子，每对固定局数，最弱那一档对随机。
#
#   BUDGETS="64 128" GAMES=500 bash training/eval_session.sh --plan
#   BUDGETS="64 128" GAMES=500 bash training/eval_session.sh
#
# --plan 只按实测吞吐算一遍时间，不跑任何对局：租机器前先看这一行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-build/shuyi_selfplay_gpu}"
MODELS="${MODELS:-models}"
LADDER="${LADDER:-runs/eval/ladder_order.txt}"
BUDGETS="${BUDGETS:-64 128}"
GAMES="${GAMES:-500}"
BATCH="${BATCH_GAMES:-64}"
CHUNK="${CHUNK:-256}"
KOMI="${KOMI:-0.5}"
RULESET="${RULESET:-nopass}"

if [ ! -f "$LADDER" ]; then
    echo "missing ladder order: $LADDER" >&2
    echo "regenerate: python training/leaderboard.py --write-ladder $LADDER" >&2
    exit 1
fi
RUNGS=$(grep -vc '^\(#\|$\)' "$LADDER")
MATCHES=$((RUNGS - 1))

# 实测吞吐（V100，梯子混合 b4c32/b6c96，单进程，8 核）：
#   64 visits 5.8 局/s，128 3.0，256 1.3，512 0.8
rate_for() {
    case "$1" in
        32) echo 7.0 ;;
        64) echo 5.8 ;;
        128) echo 3.0 ;;
        256) echo 1.3 ;;
        512) echo 0.8 ;;
        1024) echo 0.4 ;;
        *) echo 1.0 ;;
    esac
}

TOTAL=0
for visits in $BUDGETS; do
    games=$((MATCHES * GAMES))
    seconds=$(awk -v g="$games" -v r="$(rate_for "$visits")" 'BEGIN { printf "%d", g / r }')
    TOTAL=$((TOTAL + seconds))
    printf '%5s visits: %d 场 × %d 局 = %d 局, 约 %d 分钟\n' \
        "$visits" "$MATCHES" "$GAMES" "$games" "$((seconds / 60))"
done
printf '合计约 %d 分钟（另加 ~10 分钟编译与传输）\n' "$((TOTAL / 60))"

if [ "${1:-}" = "--plan" ]; then
    exit 0
fi

for visits in $BUDGETS; do
    out="runs/eval/ladder${visits}"
    mkdir -p "$out" logs
    echo "=== $visits visits -> $out ($(date +%H:%M:%S)) ==="
    ${PYTHON:-python3} training/roundrobin.py \
        --binary "$BIN" --models "$MODELS" --ladder "$LADDER" \
        --visits "$visits" --games "$GAMES" --batch-games "$BATCH" --chunk "$CHUNK" \
        --komi "$KOMI" --ruleset "$RULESET" --random-anchor \
        --out "$out" 2>&1 | tee "logs/ladder${visits}.log"
done
echo "=== done $(date +%H:%M:%S) ==="
for visits in $BUDGETS; do
    echo "pull:  python training/pull.py --remote /root/shuyi/runs/eval/ladder${visits} \\"
    echo "                          --local runs/eval/ladder${visits}"
done
