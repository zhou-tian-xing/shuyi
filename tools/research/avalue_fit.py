#!/usr/bin/env python3
"""Fit A* from the per-komi game CSVs that training/avalue.py leaves behind.

    python tools/research/avalue_fit.py runs/eval/avalue_20261125c

Each `komi_*.csv` has `game,a_is_red,result` with the result from A's side
(1 win, 0 draw, -1 loss).  Red's score is that result when A was Red and its
negation otherwise.  A* is where Red's score crosses 0.5, found by logistic
regression of (komi -> Red win probability) plus a bootstrap interval, which is
the same model training/avalue.py uses -- this just skips its own printing so a
run that was cut short can still be read.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path


def red_scores(path: Path) -> list[float]:
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = float(row["result"])
            red = r if int(row["a_is_red"]) == 1 else -r
            out.append(0.5 * (red + 1.0))  # -1/0/1 -> 0/0.5/1
    return out


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares on the logit: logit(p) = (A - A*) / s.  Returns (A*, s)."""
    weight, sx, sy, sxx, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
    for a, p in points:
        p = min(max(p, 0.02), 0.98)
        y = math.log(p / (1 - p))
        weight += 1.0
        sx += a
        sy += y
        sxx += a * a
        sxy += a * y
    det = weight * sxx - sx * sx
    if abs(det) < 1e-9:
        return float("nan"), float("nan")
    slope = (weight * sxy - sx * sy) / det
    intercept = (sy * sxx - sx * sxy) / det
    if slope == 0:
        return float("nan"), float("nan")
    return -intercept / slope, 1.0 / slope


def main(argv: list[str]) -> int:
    directory = Path(argv[1] if len(argv) > 1 else ".")
    files = sorted(directory.glob("komi_*.csv"))
    if not files:
        print(f"no komi_*.csv under {directory}", file=sys.stderr)
        return 1
    komi: dict[float, list[float]] = {}
    for path in files:
        name = path.stem.replace("komi_p", "").replace("komi_m", "-").replace("p", ".")
        komi[float(name)] = red_scores(path)

    points = []
    for a in sorted(komi):
        scores = komi[a]
        n = len(scores)
        p = sum(scores) / n
        se = math.sqrt(max(p * (1 - p), 1e-9) / n)
        points.append((a, p))
        print(f"A={a:+.2f}  Red {p:.3f} +- {1.96 * se:.3f}  ({n} games)")

    a_star, scale = fit(points)
    rng = random.Random(7)
    boots = []
    for _ in range(2000):
        sample = []
        for a in sorted(komi):
            scores = komi[a]
            draw = [rng.choice(scores) for _ in scores]
            sample.append((a, sum(draw) / len(draw)))
        value = fit(sample)[0]
        if not math.isnan(value):
            boots.append(value)
    boots.sort()
    if boots:
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots)) - 1]
        print(f"A* = {a_star:+.2f}  (95% [{lo:+.2f}, {hi:+.2f}], scale {scale:.2f})")
    else:
        print(f"A* = {a_star:+.2f} (scale {scale:.2f}, no bootstrap)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
