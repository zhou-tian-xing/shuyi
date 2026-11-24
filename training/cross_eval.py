#!/usr/bin/env python3
"""One rating list for every line of the engine, per rule set.

The two halves of the picture need different harnesses:

* which *line* is stronger (random / pruning / plain MCTS / MCTS+NN / GPU nets)
  needs heterogeneous players, so it goes through `training/arena.py`;
* how the *generations* compare needs many games between networks, which is only
  affordable batched, so it goes through `training/elo.py --models`.

Both write the same matrix format, `elo.py --report a --report b` merges them,
and the merged pool is refit with `random` pinned at 0.  Ratings from different
rule sets are never mixed: the models carry a ruleset and the engines refuse a
mismatched pair.

    python training/cross_eval.py --ruleset blocking \
        --models runs/training/b4c32-blocking-20261122/models/b4c32-s*-d*.bin \
        --out-dir runs/probes/arena/blocking --lines-games 8 --gens-games 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalconfig import EVAL_VISITS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)
PY = sys.executable


def run(cmd, echo=True):
    if echo:
        print("$ " + " ".join(str(c) for c in cmd), flush=True)
    # The child prints Chinese table headers; decode as UTF-8 regardless of the
    # console code page, or the merged report ends up with mojibake.
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if echo:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return proc.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="generation models in training order")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="labels for --models; default is the file stem")
    ap.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass")
    ap.add_argument("--komi", type=float, default=0.0)
    ap.add_argument("--visits", type=int, default=EVAL_VISITS,
                    help="fixed budget per move, identical for every player")
    ap.add_argument("--lines-visits", type=int, default=EVAL_VISITS,
                    help="same budget for the line arena; never give it its own value")
    ap.add_argument("--lines-games", type=int, default=8, help="games per pair in the line arena")
    ap.add_argument("--lines-time", type=int, default=400, help="ms/move for CPU engines")
    ap.add_argument("--lines-players", nargs="*", default=None,
                    help="override the line arena players")
    ap.add_argument("--gens-games", type=int, default=40, help="games per pair in the ladder")
    ap.add_argument("--gens-mode", choices=["full", "gauntlet"], default="full")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="bootstrap resamples for the 90%% interval (0 = skip)")
    ap.add_argument("--skip-lines", action="store_true")
    ap.add_argument("--skip-gens", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--engine-binary", default=str(REPO / "build" / "shuyi_engine"))
    ap.add_argument("--gpu-binary", default=str(REPO / "build" / "shuyi_selfplay_gpu"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    models = [Path(m) for m in args.models]
    for model in models:
        if not model.exists():
            raise SystemExit(f"missing model: {model}")
    labels = args.labels or [m.stem for m in models]
    if len(labels) != len(models):
        raise SystemExit("--labels must match --models")

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "runs" / "arena" / args.ruleset
    out_dir.mkdir(parents=True, exist_ok=True)
    gens_labels = [f"gpu:{name}" for name in labels]
    started = time.time()
    reports = []

    if not args.skip_lines:
        players = args.lines_players or [
            "random", "pruning", "mcts",
            f"mcts_nn:{models[-1]}",
            f"gpu:{models[0]}", f"gpu:{models[-1]}",
        ]
        lines_json = out_dir / "lines.json"
        run([PY, REPO / "training" / "arena.py",
             "--players", *players,
             "--games", args.lines_games,
             "--visits", args.lines_visits,
             "--time", args.lines_time,
             "--komi", args.komi,
             "--ruleset", args.ruleset,
             "--workers", args.workers,
             "--engine-binary", args.engine_binary,
             "--gpu-binary", args.gpu_binary,
             "--out", lines_json])
        reports.append(lines_json)
        print(f"[cross-eval] line arena done in {time.time() - started:.0f}s", flush=True)

    if not args.skip_gens and len(models) > 1:
        gens_json = out_dir / "gens.json"
        cmd = [PY, REPO / "training" / "elo.py",
               "--models", *models,
               "--labels", *gens_labels,
               "--games", args.gens_games,
               "--visits", args.visits,
               "--komi", args.komi,
               "--ruleset", args.ruleset,
               "--json", gens_json,
               "--bootstrap", 0,
               "--selfplay-gpu", args.gpu_binary]
        if args.gens_mode == "gauntlet":
            cmd += ["--gauntlet", gens_labels[0]]
        run(cmd)
        reports.append(gens_json)
        print(f"[cross-eval] model ladder done in {time.time() - started:.0f}s", flush=True)

    if not reports:
        raise SystemExit("nothing to rate")

    merged = out_dir / "merged.json"
    if len(reports) > 1:
        merge_cmd = [PY, REPO / "training" / "elo.py", "--anchor", "random",
                     "--bootstrap", args.bootstrap, "--ratings-json", out_dir / "ratings.json"]
        for report in reports:
            merge_cmd += ["--report", report]
        text = run(merge_cmd)
        # Keep a merged matrix on disk as well, so the table can be refitted later.
        import importlib.util
        spec = importlib.util.spec_from_file_location("elo", str(REPO / "training" / "elo.py"))
        elo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(elo)
        names, pairs, visits = elo.merge_reports([str(r) for r in reports])
        merged.write_text(json.dumps({
            "names": names,
            "pairs": [list(p) for p in pairs],
            "visits": visits,
            "komi": args.komi,
            "ruleset": args.ruleset,
        }, indent=2), encoding="utf-8")
    else:
        text = run([PY, REPO / "training" / "elo.py", "--report", reports[0],
                    "--anchor", "random", "--bootstrap", args.bootstrap])

    header = (f"# {args.ruleset} cross-line rating\n\n"
              f"- komi {args.komi:+.2f}, GPU visits {args.visits}, "
              f"line arena {args.lines_visits} visits / CPU {args.lines_time} ms per move\n"
              f"- `random` is pinned at 0; ratings are points above uniform-random play\n"
              f"- generated {time.strftime('%Y-%m-%d %H:%M')} in {time.time() - started:.0f}s\n\n"
              "```\n" + text.strip() + "\n```\n")
    (out_dir / "REPORT.md").write_text(header, encoding="utf-8")
    print("\n" + header)
    print(f"[cross-eval] wrote {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
