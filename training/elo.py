#!/usr/bin/env python3
"""ELO calibration over a pool of models, played on the GPU.

Why this exists
---------------
The self-play loop used to decide everything with one pairwise match against the
previous generation, and the training logs made it worse: after the 3080 run the
`val_policy` fell from 4.40 to 3.34 while the 200-game head-to-head against the
previous best stayed at 0.495.  Fitting the previous generation's targets says
nothing about strength -- only games between the models do.

So: play a round robin (or a gauntlet against one anchor) with
`shuyi_selfplay_gpu --mode match`, then fit Elo by maximum likelihood and report
a bootstrap confidence interval.  Everything is on the GPU; models may differ in
size and in input-plane layout.

    python training/elo.py --models models/a.bin models/b.bin ... --games 40
    python training/elo.py --report games.json          # refit from saved games

Model files are matched by name, so mixed b2c16 / b10c128 / 37-plane models can
share one rating list.
"""
import argparse
import itertools
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalconfig import EVAL_VISITS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)
DEFAULT_SELFPLAY = REPO / "build" / "shuyi_selfplay_gpu"


def resolve_binary(value):
    path = Path(value)
    if not path.exists() and os.name == "nt" and not path.suffix:
        candidate = Path(str(path) + ".exe")
        if candidate.exists():
            return candidate
    return path


def play_pair(binary, a, b, games, visits, komi, opening, ruleset, seed, log_dir, dry_run=False):
    """Returns (wins_a, draws, wins_b) for one pair, colours alternating."""
    json_out = log_dir / f"match_{Path(a).stem}__vs__{Path(b).stem}.json"
    cmd = [
        str(binary), "--mode", "match",
        "--model", str(a), "--model-b", str(b),
        "--name-a", Path(a).stem, "--name-b", Path(b).stem,
        "--games", str(games), "--batch-games", str(games),
        "--visits", str(visits), "--opening", str(opening),
        "--komi", str(komi), "--ruleset", ruleset, "--seed", str(seed),
        "--json", str(json_out),
    ]
    if dry_run:
        print(" ".join(cmd))
        return 0, 0, 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"match failed for {a} vs {b}:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("MATCH_RESULT"):
            r = json.loads(line[len("MATCH_RESULT"):])
            return int(r["a_wins"]), int(r["draws"]), int(r["b_wins"])
    raise RuntimeError(f"no MATCH_RESULT from {a} vs {b}")


def fit_elo(names, pairs, prior_sigma=400.0, passes=60):
    """Maximum-likelihood Elo with a Gaussian prior around the pool mean.

    Coordinate-wise Newton steps (the likelihood is logistic in rating
    differences, so each player's update is one-dimensional).  The prior keeps an
    undefeated model finite: without it a 40-0 result has no maximum likelihood
    estimate at all, and the bootstrap CI degenerates."""
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    scale = math.log(10.0) / 400.0
    rating = [0.0] * n
    precision = 1.0 / (prior_sigma * prior_sigma)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    for _ in range(passes):
        for i in range(n):
            grad = -rating[i] * precision
            hess = precision
            for a, b, wins_a, draws, wins_b, weight in pairs:
                games = wins_a + draws + wins_b
                if games == 0:
                    continue
                if a == names[i]:
                    other = index[b]
                elif b == names[i]:
                    other = index[a]
                else:
                    continue
                observed = ((wins_a + 0.5 * draws) / games) if a == names[i] else                            ((wins_b + 0.5 * draws) / games)
                p = expected(rating[i], rating[other])
                grad += weight * games * (observed - p) * scale
                hess += weight * games * p * (1.0 - p) * scale * scale
            if hess > 0:
                rating[i] += grad / hess
        mean = sum(rating) / n
        rating = [r - mean for r in rating]
    return rating


def bootstrap_ci(names, pairs, rounds=200, seed=7, passes=60):
    """95% interval from resampling the games.

    `passes` has to match the point estimate: a short Newton run stops before the
    prior has been overcome and returns ratings compressed toward the mean, which
    is how a CI ended up not containing the estimate.
    """
    rng = random.Random(seed)
    samples = [[] for _ in names]
    for _ in range(rounds):
        resampled = []
        for a, b, wins_a, draws, wins_b, weight in pairs:
            games = wins_a + draws + wins_b
            if games == 0:
                continue
            sa = sb = sd = 0
            for _ in range(int(games)):
                # 0 -> A win, 1 -> draw, 2 -> B win, with the empirical frequencies
                u = rng.random() * games
                if u < wins_a:
                    sa += 1
                elif u < wins_a + draws:
                    sd += 1
                else:
                    sb += 1
            resampled.append((a, b, sa, sd, sb, weight))
        rating = fit_elo(names, resampled, passes=passes)
        for i, r in enumerate(rating):
            samples[i].append(r)
    ci = []
    for values in samples:
        values.sort()
        lo = values[int(0.05 * len(values))]
        hi = values[int(0.95 * len(values)) - 1]
        ci.append((lo, hi))
    return ci


def merge_reports(paths):
    """Unions several matrices into one rating pool.

    Reports only record (winner-count, draw-count, loser-count) per unordered
    pair, so merging keeps the pair key unordered and adds the counts.  This is
    what lets a mixed-engine arena (which contains the random anchor) and a
    model-only ladder (which is GPU-batched and therefore cheap) share one
    rating list.
    """
    names = []
    totals = {}
    visits = None
    seen_visits = {}
    for path in paths:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        report_visits = saved.get("visits")
        seen_visits[str(path)] = report_visits
        if visits is None:
            visits = report_visits
        for name in saved["names"]:
            if name not in names:
                names.append(name)
        for row in saved["pairs"]:
            a, b, wa, dd, wb = row[0], row[1], int(row[2]), int(row[3]), int(row[4])
            key = (a, b)
            flipped = False
            if a > b:
                key = (b, a)
                flipped = True
            w, d, l = totals.get(key, (0, 0, 0))
            if flipped:
                totals[key] = (w + wb, d + dd, l + wa)
            else:
                totals[key] = (w + wa, d + dd, l + wb)
    # Merging tables with different budgets is how a rating quietly stops
    # measuring the models: a 96-visit row and a 512-visit row are not the same
    # unit of work, so refuse instead of averaging them.
    if len(set(seen_visits.values())) > 1:
        detail = ", ".join(f"{Path(k).name}: {v}" for k, v in seen_visits.items())
        raise SystemExit(
            "refusing to merge reports with different visit budgets (" + detail + "); "
            "every player in one rating must use the same fixed budget")
    pairs = [(a, b, w, d, l, 1.0) for (a, b), (w, d, l) in totals.items()]
    return names, pairs, visits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[], help="display names, same order as --models")
    ap.add_argument("--games", type=int, default=40, help="games per pair")
    ap.add_argument("--visits", type=int, default=EVAL_VISITS,
                    help="fixed budget per move, identical for every player")
    ap.add_argument("--komi", type=float, default=-0.5)
    ap.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass")
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--gauntlet", default="", help="play only against this model")
    ap.add_argument("--json", default="", help="save the game matrix here")
    ap.add_argument("--report", action="append", default=[],
                    help="refit from a saved game matrix; repeat to merge several "
                         "(e.g. a mixed-engine arena plus a model-only ladder)")
    ap.add_argument("--anchor", default="random",
                    help="player pinned at Elo 0 (the scale is relative to it)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--ratings-json", default="", help="write the fitted ratings here")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfplay-gpu", default=str(DEFAULT_SELFPLAY))
    args = ap.parse_args(argv)
    binary = resolve_binary(args.selfplay_gpu)

    if args.report:
        names, pairs, visits = merge_reports(args.report)
    else:
        if len(args.models) < 2:
            raise SystemExit("need at least two --models")
        visited = [p for p in args.models if Path(p).exists()]
        if len(visited) != len(args.models):
            missing = [p for p in args.models if not Path(p).exists()]
            raise SystemExit(f"missing model(s): {missing}")
        names = args.labels if args.labels else [Path(p).stem for p in args.models]
        if len(names) != len(args.models):
            raise SystemExit("--labels must match --models")
        visits = args.visits
        log_dir = REPO / "runs" / "elo"
        log_dir.mkdir(parents=True, exist_ok=True)

        combos = list(itertools.combinations(range(len(args.models)), 2))
        if args.gauntlet:
            anchor = names.index(args.gauntlet) if args.gauntlet in names else None
            if anchor is None:
                raise SystemExit(f"--gauntlet {args.gauntlet} is not one of the labels")
            combos = [c for c in combos if anchor in c]
        pairs = []
        for k, (i, j) in enumerate(combos):
            wins_a, draws, wins_b = play_pair(
                binary, args.models[i], args.models[j], args.games, visits, args.komi,
                args.opening, args.ruleset, args.seed + 1000 * k, log_dir, args.dry_run,
            )
            print(f"{names[i]:>28} vs {names[j]:<28} {wins_a:>3}/{draws:>2}/{wins_b:<3}", flush=True)
            pairs.append((names[i], names[j], wins_a, draws, wins_b, 1.0))
        matrix = {"names": names, "pairs": pairs, "visits": visits,
                  "komi": args.komi, "games_per_pair": args.games}
        out = Path(args.json) if args.json else log_dir / "matrix.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        print(f"saved {out}")

    if not pairs:
        raise SystemExit("no games")
    rating = fit_elo(names, pairs)
    anchor = args.anchor
    if anchor and anchor in names:
        shift = rating[names.index(anchor)]
        rating = [r - shift for r in rating]
        print(f"anchor: {anchor} = 0 (scale is relative to it)")
    elif anchor:
        print(f"warning: anchor {anchor!r} is not in the pool; ratings are relative to the mean")
    ci = bootstrap_ci(names, pairs, args.bootstrap) if args.bootstrap else [(0.0, 0.0)] * len(names)
    if anchor and anchor in names:
        # The bootstrap refits from scratch, so shift it into the anchored frame.
        shift_ci = ci[names.index(anchor)]
        centre = 0.5 * (shift_ci[0] + shift_ci[1])
        ci = [(lo - centre, hi - centre) for lo, hi in ci]
    order = sorted(range(len(names)), key=lambda i: -rating[i])
    print(f"\n{'model':>30} {'elo':>7} {'±90%':>13}  {'W/D/L':>8}")
    for i in order:
        w = d = l = 0
        for a, b, wa, dd, wb, _w in pairs:
            if a == names[i]:
                w += wa
                d += dd
                l += wb
            elif b == names[i]:
                w += wb
                d += dd
                l += wa
        print(f"{names[i]:>30} {rating[i]:>7.1f} [{ci[i][0]:>6.1f},{ci[i][1]:>6.1f}]  "
              f"{w:>3}/{d:>2}/{l:<3}")
    if args.ratings_json:
        out = Path(args.ratings_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "names": names,
            "elo": {names[i]: rating[i] for i in range(len(names))},
            "ci": {names[i]: ci[i] for i in range(len(names))},
            "visits": visits,
            "komi": args.komi,
            "games": sum(p[2] + p[3] + p[4] for p in pairs),
        }, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    spread = max(rating) - min(rating)
    print(f"\nspread {spread:.1f} elo over {len(names)} models, "
          f"{sum(p[2]+p[3]+p[4] for p in pairs)} games, {visits} visits/move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
