#!/usr/bin/env python3
"""The model leaderboard: 2D match matrices at fixed visit budgets, absolute Elo.

Why this exists
---------------
Training logs answer "did generation N beat N-1".  That is not a rating: a chain
of 0.55s can climb while the model is still far below an older one, and a single
pairwise result says nothing about the field.  This tool keeps one table for every
model in the repository, at one or more fixed visit budgets, and fits an absolute
Eloh scale anchored at uniform-random play (the only strength unit this repo
accepts).

How it stays cheap
------------------
* `random` is in every table, so every rating is "points above uniform random".
* Pairs are chosen by strength: the first pass plays everything (there is nothing
  to sort by yet), later passes only play pairs within `--max-gap` Elo of each
  other, because a 3000-Elo difference is measured just as well by 10 games as by
  200 and nobody needs the number.
* Work is incremental: each pair's games live in their own matrix file, and a
  re-run only plays what is missing.  Deleting a file re-measures that pair.

    python training/leaderboard.py                       # 500 and 2000 visits
    python training/leaderboard.py --visits 500 --games 20
    python training/leaderboard.py --pairs all            # ignore the gap filter

Outputs (under runs/leaderboard/):
    v<visits>/<n>.json      one arena matrix per group of models
    ratings.json            Elo + bootstrap CI per budget
    leaderboard.md          the table, one section per budget
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import bootstrap_ci, fit_elo  # noqa: E402  (same fit as every other table)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "leaderboard"
ARENA = REPO / "training" / "arena.py"


def model_label(path: Path) -> str:
    """The file name without .bin: canonical and unique inside a run."""
    return path.stem


def model_ruleset(path: Path) -> int:
    """The rule-set byte, read the way the engine reads it (-1 = unreadable)."""
    try:
        head = path.read_bytes()[:36]
    except OSError:
        return -1
    if len(head) < 12:
        return -1
    magic = head[:8]
    if magic in (b"SHUYINN3", b"SHUYINN4"):
        if len(head) < 33:
            return -1
        return head[32]
    if magic == b"SHUYINN2":
        return 0  # v2 predates the byte: those weights are classic
    return -1


def compatible(ruleset_byte: int, game: str) -> bool:
    """Same rule as the engine: equality, or blocking weights under nopass."""
    codes = {"classic": 0, "blocking": 1, "nopass": 2}
    game_code = codes.get(game, 2)
    if ruleset_byte < 0:
        return False
    return ruleset_byte == game_code or (ruleset_byte == 1 and game_code == 2)


def discover(patterns) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(REPO.glob(pattern)))
    unique = []
    seen = set()
    for path in found:
        if not path.is_file() or path.suffix == ".tmp":
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def arena_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrices(visits: int) -> list[dict]:
    directory = OUT / f"v{visits}"
    if not directory.exists():
        return []
    return [arena_matrix(p) for p in sorted(directory.glob("*.json"))]


def merge_pairs(matrices) -> tuple[list[str], list[tuple]]:
    """Sum games per unordered pair so re-playing a pair adds evidence."""
    names: list[str] = []
    totals: dict[tuple[str, str], list[float]] = {}
    for matrix in matrices:
        for name in matrix.get("names", []):
            if name not in names:
                names.append(name)
        for entry in matrix.get("pairs", []):
            a, b, wins_a, draws, wins_b, weight = entry
            key = (a, b)
            if key not in totals:
                totals[key] = [0.0, 0.0, 0.0, 0.0]
            totals[key][0] += wins_a
            totals[key][1] += draws
            totals[key][2] += wins_b
            totals[key][3] += weight if weight else 1.0
    pairs = [(a, b, int(v[0]), int(v[1]), int(v[2]), v[3] / max(1.0, v[3]))
             for (a, b), v in totals.items()]
    return names, pairs


def score_matrix(names, pairs) -> dict[tuple[str, str], tuple[float, int]]:
    """(score for the row player, games) for every measured pair."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for a, b, wins_a, draws, wins_b, _weight in pairs:
        games = wins_a + draws + wins_b
        if games <= 0:
            continue
        out[(a, b)] = ((wins_a + 0.5 * draws) / games, games)
        out[(b, a)] = ((wins_b + 0.5 * draws) / games, games)
    return out


def ratings_for(names, pairs) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    ratings = fit_elo(names, pairs)
    ci = bootstrap_ci(names, pairs)
    anchored = dict(zip(names, ratings))
    if "random" in anchored:
        # Uniform random play is the zero point: without this the pool mean is 0
        # and every table would be on a different scale.
        zero = anchored["random"]
        anchored = {name: value - zero for name, value in anchored.items()}
    result_ci = {}
    for name, interval in zip(names, ci):
        if name in anchored and "random" in dict(zip(names, ratings)):
            zeros = dict(zip(names, ratings))["random"]
            result_ci[name] = (interval[0] - zeros, interval[1] - zeros)
        else:
            result_ci[name] = interval
    return anchored, result_ci


def pick_pairs(names, ratings, max_gap: float) -> list[tuple[str, str]]:
    """Every pair closer than max_gap; all pairs when there is no rating yet."""
    if not ratings:
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    chosen = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = abs(ratings.get(a, 0.0) - ratings.get(b, 0.0))
            if gap <= max_gap:
                chosen.append((a, b))
    return chosen


def groups_for(pairs, size: int) -> list[list[str]]:
    """Collect pairs into groups so one arena call covers `size` models."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    components: dict[str, set[str]] = {}
    for name in parent:
        components.setdefault(find(name), set()).add(name)
    groups = []
    for members in components.values():
        ordered = sorted(members)
        for start in range(0, len(ordered), size):
            chunk = ordered[start:start + size]
            if len(chunk) >= 2:
                groups.append(chunk)
    return groups


def have_games(paths, a, b) -> int:
    games = 0
    for path in paths:
        try:
            matrix = arena_matrix(path)
        except Exception:  # noqa: BLE001 - a truncated file is just "not measured"
            continue
        for entry in matrix.get("pairs", []):
            pa, pb, wa, d, wb = entry[0], entry[1], entry[2], entry[3], entry[4]
            if {pa, pb} == {a, b}:
                games += wa + d + wb
    return games


def run_arena(group, visits, games, extra) -> Path:
    index = 1
    while True:
        out = OUT / f"v{visits}" / f"g{index:02d}.json"
        if not out.exists():
            break
        index += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ARENA), "--players", *group, "--games", str(games),
           "--visits", str(visits), "--out", str(out), *extra]
    print("  " + " ".join(cmd[-6:]), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return out if out.exists() else Path()
    return out


def display_name(name: str) -> str:
    """Strip the arena player prefix: the table is about models, not specs."""
    return name[4:] if name.startswith("gpu:") else name


def write_markdown(budgets, ratings, cis, matrices, path: Path) -> None:
    lines = ["# 模型总表（自动生成，勿手改）", ""]
    lines += ["`python training/leaderboard.py` 生成。ELO 以 **random = 0** 为锚，",
              "带 200 次 bootstrap 的 95% 置信区间；每格是行方在对局中的得分率",
              "（和棋算半分），括号里是局数。", ""]
    for visits in budgets:
        names, pairs = merge_pairs(matrices[visits])
        if not names:
            lines += [f"## {visits} visits", "", "（还没有数据）", ""]
            continue
        scores = score_matrix(names, pairs)
        rating = ratings[visits]
        order = sorted(names, key=lambda n: -rating.get(n, -1e9))
        lines += [f"## {visits} visits", "", "| 模型 | ELO | 95% CI | " +
                  " | ".join(display_name(n) for n in order) + " |"]
        lines.append("| --- | ---: | --- | " + " | ".join("---:" for _ in order) + " |")
        for row in order:
            ci = cis[visits].get(row)
            ci_text = f"[{ci[0]:+.0f}, {ci[1]:+.0f}]" if ci else "—"
            cells = []
            for col in order:
                if col == row:
                    cells.append("—")
                    continue
                entry = scores.get((row, col))
                cells.append(f"{entry[0] * 100:.0f}% ({entry[1]})" if entry else "·")
            lines.append(f"| `{display_name(row)}` | {rating.get(row, 0.0):+.0f} | "
                         f"{ci_text} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["models/*.bin", "runs/training/*/models/*.bin"],
                    help="globs, relative to the repo root")
    ap.add_argument("--visits", default="500,2000", help="comma separated budgets")
    ap.add_argument("--games", type=int, default=10, help="games per pair per budget")
    ap.add_argument("--max-gap", type=float, default=250.0,
                    help="only play pairs within this Elo distance (needs a previous run)")
    ap.add_argument("--pairs", choices=["auto", "all"], default="auto")
    ap.add_argument("--group", type=int, default=6, help="models per arena call")
    ap.add_argument("--ruleset", default="nopass")
    ap.add_argument("--komi", default="",
                    help="points added to Red; default follows the rule set "
                         "(classic -0.5, blocking/nopass +1.26) like pipeline.py")
    ap.add_argument("--gpu-binary", default="", help="shuyi_selfplay_gpu path")
    ap.add_argument("--engine-binary", default="", help="shuyi_engine path")
    ap.add_argument("--force", action="store_true", help="ignore existing matrices")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    budgets = [int(v) for v in args.visits.split(",") if v.strip()]
    models = discover(args.models)
    if not models:
        print("no models found")
        return 1
    labels = {}
    for path in models:
        label = model_label(path)
        if label in labels.values():
            label = f"{label}-{path.parent.parent.name}"
        labels[path] = label
    # A model trained for another rule set cannot play these games: the engine
    # refuses it, and silently playing on would rate weights that never ran.
    skipped = [label for path, label in labels.items()
               if not compatible(model_ruleset(path), args.ruleset)]
    for path in [p for p, label in labels.items() if label in skipped]:
        del labels[path]
    if skipped:
        print("skipped (other rule set): " + ", ".join(sorted(skipped)), flush=True)
    print(f"{len(models)} models: " + ", ".join(sorted(labels.values())), flush=True)

    ratings_path = OUT / "ratings.json"
    previous = json.loads(ratings_path.read_text(encoding="utf-8")) if ratings_path.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)

    extra = []
    if args.ruleset:
        extra += ["--ruleset", args.ruleset]
    komi = args.komi or ("-0.5" if args.ruleset == "classic" else "1.26")
    extra += ["--komi", komi]
    if args.gpu_binary:
        extra += ["--gpu-binary", args.gpu_binary]
    if args.engine_binary:
        extra += ["--engine-binary", args.engine_binary]

    ratings: dict[int, dict[str, float]] = {}
    cis: dict[int, dict[str, tuple[float, float]]] = {}
    matrices: dict[int, list[dict]] = {}

    for visits in budgets:
        directory = OUT / f"v{visits}"
        if args.force and directory.exists():
            for path in directory.glob("*.json"):
                path.unlink()
        by_label = {label: f"gpu:{path.resolve()}" for path, label in labels.items()}
        names = sorted(by_label)
        prior = (previous.get(str(visits)) or {}).get("ratings") or {}
        pairs = ([(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
                 if args.pairs == "all" else pick_pairs(names, prior, args.max_gap))
        existing = sorted(directory.glob("*.json")) if directory.exists() else []
        todo = [pair for pair in pairs
                if args.force or have_games(existing, pair[0], pair[1]) < args.games]
        print(f"[{visits} visits] {len(pairs)} pairs selected, {len(todo)} need games",
              flush=True)
        if todo and not args.dry_run:
            for group in groups_for(todo, args.group):
                run_arena([by_label[name] for name in group], visits, args.games, extra)
        found = load_matrices(visits)
        matrices[visits] = found
        names_all, merged = merge_pairs(found)
        if merged:
            ratings[visits], cis[visits] = ratings_for(names_all, merged)
        else:
            ratings[visits], cis[visits] = {}, {}

    if args.dry_run:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {}
    for visits in budgets:
        payload[str(visits)] = {
            "ratings": ratings.get(visits, {}),
            "ci": {k: list(v) for k, v in cis.get(visits, {}).items()},
        }
    ratings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    write_markdown(budgets, ratings, cis, matrices, OUT / "leaderboard.md")
    for visits in budgets:
        best = ratings.get(visits) or {}
        if best:
            top = sorted(best.items(), key=lambda kv: -kv[1])[:3]
            print(f"{visits} visits: " +
                  ", ".join(f"{display_name(n)} {v:+.0f}" for n, v in top), flush=True)
    print(f"wrote {OUT / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
