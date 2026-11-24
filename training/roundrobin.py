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


def play_pair(binary: Path, a: Path | str, b: Path | str, args, out: Path) -> dict:
    """One match.  `b` may be the string "random", which the binary plays itself."""
    a_path, a_name = (a, Path(str(a)).stem)
    b_path, b_name = (b, "random") if str(b) == "random" else (b, Path(str(b)).stem)
    target = out / f"{a_name}__{b_name}.json"
    cmd = [
        str(binary), "--mode", "match",
        "--model", str(a_path), "--name-a", a_name,
        "--model-b", str(b_path), "--name-b", b_name,
        "--ruleset", args.ruleset, "--komi", f"{args.komi}",
        "--visits", str(args.visits), "--games", str(args.games),
        "--batch-games", str(min(args.games, args.batch_games)),
        "--chunk", str(args.chunk), "--seed", str(args.seed),
        "--csv", str(target.with_suffix(".csv")), "--json", str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"a": a_name, "b": b_name, "error": proc.stderr.strip()[-500:]}
    stat = json.loads(target.read_text())
    # `a_wins` counts wins for side A; the cell is from the row model's view.
    return {
        "a": a_name,
        "b": b_name,
        "visits": args.visits,
        "wins": stat["a_wins"],
        "draws": stat["draws"],
        "losses": stat["b_wins"],
        "games": stat["games"],
        "seconds": stat["seconds"],
    }


def tasks_for(models: list[Path], args) -> list[tuple[Path | str, Path | str]]:
    if args.random_only:
        return [(path, "random") for path in models]
    if args.ladder:
        # 只要相邻两档：实力差太远的对局只会给出 20-0 和一个没有意义的下界。
        order = [line.strip() for line in Path(args.ladder).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        by_name = {path.stem: path for path in models}
        missing = [name for name in order if name not in by_name]
        if missing:
            raise SystemExit("ladder names not found: " + ", ".join(missing))
        rungs = [by_name[name] for name in order]
        tasks = list(zip(rungs, rungs[1:]))
        if args.random_anchor:
            # 只有最弱那一档跟随机下：它给出绝对 0 分，其余各档的分差由梯子递推，
            # 不需要每个人都去打随机（那样每场都是 40-0，什么也读不出来）。
            tasks.append((rungs[0], "random"))
        return tasks
    if args.random_anchor:
        raise SystemExit("--random-anchor 要配合 --ladder：随机锚只挂在最弱那一档上")
    return list(itertools.combinations(models, 2))


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
    ap.add_argument("--random-anchor", action="store_true",
                    help="every model (or the bottom of the ladder) also plays uniform random, "
                         "which is Elo 0 at every budget")
    ap.add_argument("--random-only", action="store_true",
                    help="only play model vs random (the absolute anchor), no model pairs")
    ap.add_argument("--ladder", type=Path, default=None,
                    help="file with one model name per line, weakest first: play only the "
                         "adjacent pairs instead of the full round robin")
    args = ap.parse_args()

    models = model_list(args.models, args.include, args.skip)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = tasks_for(models, args)
    print(f"{len(models)} models, {len(pairs)} matches, {args.games} games each, "
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
