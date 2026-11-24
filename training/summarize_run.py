#!/usr/bin/env python3
"""Turn a run's manifest.json into a markdown table for the training notes.

    python training/summarize_run.py runs/training/b4c32-blocking-20261122/manifest.json

Columns: generation, mode, self-play positions, the model it produced, the
head-to-head arena against the previous generation and the pool Elo that was
recorded for that generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    gens = data.get("generations", data if isinstance(data, list) else [])

    print("| 代 | 方式 | 自对弈局面 | 模型 | 对上一代 | 池内 ELO |")
    print("|---:|------|-----------:|------|----------|----------|")
    # The pool Elo of each generation lives in <run>/elo/ratings_gen_XXX.json.
    elo_by_gen = {}
    elo_dir = args.manifest.parent / "elo"
    if elo_dir.is_dir():
        for rating_file in sorted(elo_dir.glob("ratings_gen_*.json")):
            try:
                ratings = json.loads(rating_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for name, value in ratings.get("elo", {}).items():
                elo_by_gen[name] = value
    for entry in gens:
        gen = entry.get("gen", 0)
        mode = entry.get("mode", "?")
        samples = entry.get("selfplay_samples", entry.get("samples", 0))
        name = Path(entry.get("named_model") or entry.get("final_model") or "-").name
        arena = entry.get("arena") or {}
        if "score_a" in arena:
            a = arena.get("a_wins", 0)
            d = arena.get("draws", 0)
            b = arena.get("b_wins", 0)
            head = f"{arena['score_a']:.3f} ({a}/{d}/{b})"
        else:
            head = arena.get("note", "-")
        key = f"gen_{gen:03d}"
        elo = elo_by_gen.get(key)
        print(f"| {gen} | {mode} | {samples} | {name} | {head} | "
              f"{('%+.0f' % elo) if isinstance(elo, (int, float)) else '-'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
