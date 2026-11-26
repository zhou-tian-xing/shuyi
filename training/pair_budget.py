#!/usr/bin/env python3
"""How many games each pair needs, given a budget and what is already known.

One game tells you about the rating difference through p(1-p): its Fisher
information about the Elo gap is p(1-p)/k^2 with k = 400/ln10 ~= 173.7.  So a
pair near 50% carries the most per game, and a pair already at 95% carries almost
nothing.  To reach a half-width h on a pair:

    n = (1.96 * 173.7 / h)^2 / (p (1 - p))

Two consequences, both of which decide where the games go:

* pairs whose models are close (p near 0.5) need the most games -- they are also
  the ones whose ordering actually matters, since a ladder is decided rung by
  rung;
* a lopsided pair cannot be pinned down cheaply, and usually should not be: it
  is a lower bound, not a measurement.

`--budget` turns that into an allocation: proportional to 1/(p(1-p)), with a
floor so nothing is measured on a handful of games and a cap so one pair cannot
eat the window.

    python training/pair_budget.py --table runs/leaderboard/matches_64.csv \\
        --budget 23500 --seconds-per-game 1.53 --visits 512
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaderboard import decisive, read_table  # noqa: E402
from elo import fit_elo_checked  # noqa: E402

K = 400.0 / math.log(10.0)


def expected(gap):
    return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))


def half_width(games, p):
    """95% half-width in Elo for `games` at success probability p."""
    if games <= 0 or p <= 0.0 or p >= 1.0:
        return float("inf")
    return 1.96 * K / math.sqrt(games * p * (1.0 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--budget", type=int, required=True, help="total games to spend")
    ap.add_argument("--seconds-per-game", type=float, default=1.53)
    ap.add_argument("--visits", type=int, default=0, help="label only")
    ap.add_argument("--hopeless", type=float, default=0.985,
                    help="skip a pair whose expected win rate is above this: a "
                         "one-sided cell carries no rating information")
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--extra-pairs", nargs="*", default=[], metavar="A:B",
                    help="pairs to add even though the table has no cell for them "
                         "(long jumps keep a chain from accumulating error)")
    ap.add_argument("--keep", default="",
                    help="only pairs whose two ends match this regex (random always kept)")
    ap.add_argument("--plan", type=Path, help="write the per-pair counts here")
    args = ap.parse_args()

    _, results = read_table(args.table)
    pairs = [(a, b, v[0], v[1], v[2]) for (a, b), v in sorted(results.items())
             if a < b and decisive(*v)]
    for spec in args.extra_pairs:
        a, _, b = spec.partition(":")
        if not b:
            raise SystemExit(f"--extra-pairs entries look like A:B, read {spec!r}")
        pairs.append((a, b, 0, 0, 0))
    if args.keep:
        import re
        pattern = re.compile(args.keep)

        def usable(name):
            return name == "random" or bool(pattern.search(name))

        pairs = [p for p in pairs if usable(p[0]) and usable(p[1])]
    names = sorted({n for a, b, *_ in pairs for n in (a, b)})
    fit = [(a, b, w, d, l, 1.0) for a, b, w, d, l in pairs]
    rating, moved = fit_elo_checked(names, fit, passes=40000, tolerance=0.01)
    if moved >= 0.01:
        raise SystemExit(f"the table's fit did not converge ({moved:.4g})")
    elo = dict(zip(names, rating))

    gaps = {f"{a}|{b}": elo[a] - elo[b] for a, b, *_ in pairs
            if a in elo and b in elo}
    have = {f"{a}|{b}": wins + draws + losses for a, b, wins, draws, losses in pairs}

    def counts(target):
        """Games each pair still needs to reach a half-width of `target` Elo.

        The rule is "top up what is uncertain, skip what cannot be measured":

        * a pair already known to `target` gets nothing;
        * a pair whose models are so far apart that even a large budget would
          come back 300-0 is skipped, because a one-sided cell contributes no
          rating information at all -- spending games there is the same mistake
          as reporting 20-0-0 as a result;
        * everything in between gets exactly what it takes.
        """
        out = {}
        for key, gap in gaps.items():
            p = expected(abs(gap))
            if p > args.hopeless:
                continue                      # no affordable number of games decides this
            need = (1.96 * K / target) ** 2 / (p * (1.0 - p))
            short = need - have.get(key, 0)
            if short <= args.floor:
                continue                      # already this precise, or not worth a session
            out[key] = int(min(args.cap, round(short)))
        return out

    # A larger target buys fewer games, so bisect for the tightest precision the
    # budget allows: nothing is measured finer than the decision needs.
    lo, hi = 3.0, 400.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(counts(mid).values()) > args.budget:
            lo = mid
        else:
            hi = mid
    allocation = counts(hi)
    print(f"target half-width {hi:.0f} Elo (skip if expected score > {args.hopeless:.2f})")

    rows = []
    for a, b, wins, draws, losses in pairs:
        key = f"{a}|{b}"
        if key not in gaps:
            continue
        gap = gaps[key]
        p = expected(abs(gap))
        n = allocation[key]
        total = n + wins + draws + losses
        rows.append((key, gap, n, half_width(total, p), wins + draws + losses))
    rows.sort(key=lambda r: abs(r[1]))

    print(f"{'pair':>52} {'gap':>7} {'games':>6} {'+-Elo':>6} {'have':>6}")
    for key, gap, n, h, have in rows:
        print(f"{key:>52} {gap:+7.0f} {n:>6} {h:>6.0f} {have:>6}")
    games = sum(r[2] for r in rows)
    hours = games * args.seconds_per_game / 3600.0
    print(f"\n{len(rows)} pairs, {games} games, ~{hours:.1f} h at "
          f"{args.seconds_per_game:.2f} s/game" + (f" ({args.visits} visits)" if args.visits else ""))
    tight = [r for r in rows if abs(r[1]) < 100]
    print(f"tight pairs (|gap| < 100 Elo): {len(tight)}, median half-width "
          f"{sorted(r[3] for r in tight)[len(tight) // 2]:.0f} Elo" if tight else "no tight pairs")
    if args.plan:
        args.plan.write_text(json.dumps({k: n for k, _, n, _, _ in rows}, indent=1),
                             encoding="utf-8")
        print(f"plan written to {args.plan}")


if __name__ == "__main__":
    main()
