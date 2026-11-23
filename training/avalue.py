#!/usr/bin/env python3
"""A-value (komi) calibration for a rule set.

`A值` in the Shuyi docs is the balance term added to Red's raw board score:
`T = P + A`.  This tool measures what A makes Red win half the time, using
models trained for that rule set -- equal models on both sides, so the colour
imbalance is the only thing left in the result.

  python training/avalue.py --model runs/blocking/blk1/models/b4c32-s34760-d470705.bin \
      --ruleset blocking --komis -2,-1,0,1 --games 300 --visits 256 --json out.json

Each komi runs `shuyi_selfplay_gpu --mode match` with the same model on both
sides and colour-alternating games, then reads the per-game CSV to get Red's
win rate.  A* is fitted by logistic regression on (komi -> Red win), so the
answer is not limited to the probed values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalconfig import EVAL_VISITS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)


def _wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def play_one(binary: Path, model: Path, model_b: Path | None, args, komi: float,
             workdir: Path) -> dict:
    csv_path = workdir / f"komi_{komi:+.2f}.csv".replace("+", "p").replace("-", "m")
    json_path = workdir / f"komi_{komi:+.2f}.json".replace("+", "p").replace("-", "m")
    cmd = [
        str(binary), "--mode", "match",
        "--model", str(model), "--name-a", model.stem,
        "--model-b", str(model_b or model), "--name-b", (model_b or model).stem,
        "--ruleset", args.ruleset, "--komi", f"{komi}", "--visits", str(args.visits),
        "--games", str(args.games), "--batch-games", str(args.batch_games),
        "--chunk", str(args.chunk), "--seed", str(args.seed),
        "--csv", str(csv_path), "--json", str(json_path),
    ]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"match failed at komi {komi} (exit {proc.returncode})")

    red_wins = 0.0
    games = 0
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            result = int(row["result"])
            a_is_red = int(row["a_is_red"]) == 1
            games += 1
            if result == 0:
                # A draw is half a win for Red, matching the Elo convention.
                red_wins += 0.5
            elif (result > 0) == a_is_red:
                red_wins += 1.0
    stat = json.loads(json_path.read_text())
    return {
        "komi": komi,
        "games": games,
        "red_score": red_wins / games if games else 0.5,
        "a_wins": stat["a_wins"],
        "b_wins": stat["b_wins"],
        "draws": stat["draws"],
        "seconds": stat["seconds"],
    }


def fit_logistic(points: list[dict]) -> tuple[float, float, float]:
    """Weighted logistic fit of Red's score on komi.

    Returns (A*, slope, se(A*)).  A* is where the fitted curve crosses 0.5;
    `slope` is the log-odds change per point of komi (positive: more komi for
    Red means a higher Red score).
    """
    rows = []
    for point in points:
        n = point["games"]
        w = point["red_score"] * n
        rows.append((point["komi"], w, n))
    total = sum(n for _, _, n in rows)

    def loglik(b0: float, b1: float) -> float:
        total_ll = 0.0
        for komi, w, n in rows:
            z = max(-30.0, min(30.0, b0 + b1 * komi))
            p = 1.0 / (1.0 + math.exp(-z))
            p = min(1 - 1e-9, max(1e-9, p))
            total_ll += w * math.log(p) + (n - w) * math.log(1 - p)
        return total_ll

    # Newton on a two-parameter logistic, with a weak prior that keeps the slope
    # negative and finite even if the probed komis are all on one side.
    b0, b1 = 0.0, 0.4
    for _ in range(200):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for komi, w, n in rows:
            z = max(-30.0, min(30.0, b0 + b1 * komi))
            p = 1.0 / (1.0 + math.exp(-z))
            r = w - n * p
            g0 += r
            g1 += r * komi
            h00 += n * p * (1 - p)
            h01 += n * p * (1 - p) * komi
            h11 += n * p * (1 - p) * komi * komi
        # Prior: b0 ~ N(0, 400), b1 ~ N(0.4, 1.0) expressed as extra curvature.
        g0 -= b0 / 400.0
        g1 -= (b1 - 0.4) / 1.0
        h00 += 1.0 / 400.0
        h11 += 1.0
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (-h01 * g0 + h00 * g1) / det
        b0 += d0
        b1 += d1
        if abs(d0) < 1e-12 and abs(d1) < 1e-12:
            break
    # komi is added to Red's raw score, so more komi must mean a higher Red
    # score: p = sigmoid(b0 + b1*K) with b1 > 0, and the crossing is -b0/b1.
    b1 = max(b1, 1e-6)
    a_star = -b0 / b1
    # Standard error of the crossing point by the delta method on the Hessian.
    det = h00 * h11 - h01 * h01
    se = math.sqrt(max(1e-12, (h11 / det) + (a_star * a_star) * (h00 / det))) / abs(b1)
    return a_star, b1, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None,
                    help="model to probe; required unless --self-test")
    ap.add_argument("--model-b", type=Path, default=None,
                    help="optional second model; default is the same model on both sides")
    ap.add_argument("--ruleset", choices=["classic", "blocking"], default="blocking")
    ap.add_argument("--komis", default="-2,-1,0,1",
                    help="comma separated A values to probe")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--visits", type=int, default=EVAL_VISITS,
                    help="fixed budget per move (see training/evalconfig.py)")
    ap.add_argument("--batch-games", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261201)
    ap.add_argument("--binary", type=Path,
                    default=REPO / "build" / "shuyi_selfplay_gpu")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="fit a synthetic curve with a known crossing and exit")
    args = ap.parse_args()

    if args.self_test:
        # Guards the sign convention: komi is added to Red's score, so the fit
        # slope must be positive and the crossing must come back negative when
        # Red is the stronger side at komi 0.
        true_b0, true_b1 = 0.1, 0.44
        synthetic = []
        for komi in (-2.0, -1.0, 0.0, 1.0):
            p = 1.0 / (1.0 + math.exp(-(true_b0 + true_b1 * komi)))
            synthetic.append({"komi": komi, "games": 240, "red_score": p})
        a_star, slope, _se = fit_logistic(synthetic)
        expected = -true_b0 / true_b1
        ok = abs(a_star - expected) < 0.01 and slope > 0
        print(f"self-test: A*={a_star:+.3f} (expected {expected:+.3f}) slope={slope:+.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.model is None:
        raise SystemExit("--model is required")
    komis = [float(x) for x in args.komis.split(",") if x.strip()]
    workdir = args.workdir or (REPO / "runs" / "avalue")
    workdir.mkdir(parents=True, exist_ok=True)

    points = []
    for komi in komis:
        points.append(play_one(args.binary, args.model, args.model_b, args, komi, workdir))

    print("\nA-value probe (%s, %s visits, %d games per point)" %
          (args.model.name, args.visits, args.games))
    print("  A      Red score   95% CI              W/D/L")
    for point in points:
        low, high = _wilson(point["red_score"] * point["games"], point["games"])
        print("  %+5.2f  %.3f       [%.3f, %.3f]      %d/%d/%d"
              % (point["komi"], point["red_score"], low, high,
                 point["a_wins"], point["draws"], point["b_wins"]))

    a_star, slope, se = fit_logistic(points)
    print("\n  fitted A* = %+.3f  (slope %.3f log-odds/point, SE %.3f)" % (a_star, slope, se))
    print("  => %s rule set, model %s: fair A = %+.2f" % (args.ruleset, args.model.name, a_star))

    record = {
        "ruleset": args.ruleset,
        "model": str(args.model),
        "model_b": str(args.model_b) if args.model_b else None,
        "visits": args.visits,
        "games_per_point": args.games,
        "points": points,
        "a_star": a_star,
        "slope": slope,
        "a_star_se": se,
        "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    out = args.json or (workdir / f"avalue-{args.ruleset}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("wrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
