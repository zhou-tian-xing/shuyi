#!/usr/bin/env python3
"""Play every pair of models at one visit budget and write the pair results.

This is the data-collection half of the leaderboard: it produces one JSON per
pair (the `shuyi_selfplay_gpu --mode match` result), plus a summary CSV with the
`胜-和-负` cell for each pair.  `training/leaderboard.py` only *computes* Elo from
a table; this tool fills a table.

  python training/roundrobin.py --binary build/shuyi_selfplay_gpu \\
      --models models --visits 64 --games 20 --workers 6 --out runs/eval/v64

Pairs are unordered: (A,B) is played once with colour-alternating games, so the
cell already contains both colours.  Nothing is silently skipped: a pair that
fails to run is reported at the end and the exit status is non-zero.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def model_list(directory: Path, include: str, skip: str) -> list[Path]:
    skipped = [name for name in skip.split(",") if name.strip()]
    files = [p for p in sorted(directory.glob("*.bin"))
             if include in p.name and p.stem not in skipped]
    if not files:
        raise SystemExit(f"no *.bin under {directory}" + (f" matching {include!r}" if include else ""))
    return files


def play_pair(binary: Path, a: Path, b: Path, args, out: Path) -> dict:
    target = out / f"{a.stem}__{b.stem}.json"
    cmd = [
        str(binary), "--mode", "match",
        "--model", str(a), "--name-a", a.stem,
        "--model-b", str(b), "--name-b", b.stem,
        "--ruleset", args.ruleset, "--komi", f"{args.komi}",
        "--visits", str(args.visits), "--games", str(args.games),
        "--batch-games", str(min(args.games, args.batch_games)),
        "--chunk", str(args.chunk), "--seed", str(args.seed),
        "--csv", str(target.with_suffix(".csv")), "--json", str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"a": a.stem, "b": b.stem, "error": proc.stderr.strip()[-500:]}
    stat = json.loads(target.read_text())
    # `a_wins` counts wins for side A; the cell is from the row model's view.
    return {
        "a": a.stem,
        "b": b.stem,
        "visits": args.visits,
        "wins": stat["a_wins"],
        "draws": stat["draws"],
        "losses": stat["b_wins"],
        "games": stat["games"],
        "seconds": stat["seconds"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=Path("build/shuyi_selfplay_gpu"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--include", default="", help="only model names containing this")
    ap.add_argument("--skip", default="", help="comma separated model names to leave out")
    ap.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass")
    ap.add_argument("--komi", default="1.26")
    ap.add_argument("--visits", type=int, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--batch-games", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel matches; each process already saturates every "
                         "core, so more than one only pays off with OMP_NUM_THREADS=1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    models = model_list(args.models, args.include, args.skip)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = list(itertools.combinations(models, 2))
    print(f"{len(models)} models, {len(pairs)} pairs, {args.games} games each, "
          f"visits={args.visits}, workers={args.workers}", flush=True)

    started = time.time()
    rows: list[dict] = []
    failed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(play_pair, args.binary, a, b, args, args.out) for a, b in pairs]
        for i, future in enumerate(futures, 1):
            result = future.result()
            if "error" in result:
                failed.append(result)
                print(f"  !! {result['a']} vs {result['b']}: {result['error']}", flush=True)
            else:
                rows.append(result)
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - started
                games = sum(r["games"] for r in rows)
                print(f"  {i}/{len(futures)} pairs, {games} games, "
                      f"{games / max(elapsed, 1e-9):.2f} games/s, {elapsed:.0f}s", flush=True)

    table = args.out / "results.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "b", "visits", "wins", "draws", "losses", "games", "seconds"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["games"] for r in rows)
    print(f"\n{len(rows)} pairs, {total} games, {time.time() - started:.0f}s -> {table}")
    if failed:
        print(f"FAILED {len(failed)} pairs:", file=sys.stderr)
        for row in failed:
            print(f"  {row['a']} vs {row['b']}: {row['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
