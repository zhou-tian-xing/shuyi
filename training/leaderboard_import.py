#!/usr/bin/env python3
"""按规则、A 值和 visits 汇总可追溯的评测报告。

只做"填表"，不算 ELO —— 那是 `leaderboard.py` 的事。
名字要落到真实存在的模型上：`gen_003` 用该 run 的 manifest 换成累计模型名，
`gpu:`/`mcts_nn:` 前缀和旧的 `-blk`/`-cls` 后缀去掉；换不出来、仓库里也没有
这个文件的对局直接丢掉（宁愿少一行，也不要一行指不到模型的分）。

每个源文件只计一次，原路径重测时更新该源的成绩。默认扫描 runs/eval/。
缺少评测条件的历史文件留在原来的历史榜。
"""
from __future__ import annotations

import csv
import io
import json
from tempfile import NamedTemporaryFile
from pathlib import Path

from evalconfig import board_directory, match_config

REPO = Path(__file__).resolve().parents[1]
PSEUDO = ("random", "pruning", "mcts")


def canonical(name: str) -> str:
    name = str(name)
    for prefix in ("gpu:", "mcts_nn:", "mnn:", "pruning:", "mcts:"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("-blk", "-cls", "-nps"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def known_models() -> set[str]:
    labels = set()
    for pattern in ("models/*.bin", "runs/training/*/models/*.bin"):
        for path in REPO.glob(pattern):
            labels.add(path.stem)
    return labels


def gen_map(run_dir: Path) -> dict[str, str]:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    mapping = {}
    for record in data.get("generations", []):
        gen = record.get("gen")
        if gen is None:
            continue
        name = record.get("model_name") or Path(str(record.get("final_model") or "")).stem
        if name:
            mapping[f"gen_{int(gen):03d}"] = canonical(name)
    return mapping


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                            delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_table(path: Path, cells) -> None:
    order = sorted({name for pair in cells for name in pair})
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["model", *order])
    for row in order:
        line = [row]
        for col in order:
            value = cells.get((row, col))
            if row == col:
                line.append("—")
            else:
                line.append("-".join(map(str, value)) if value else "")
        writer.writerow(line)
    atomic_write(path, output.getvalue())


def add(cells, a, b, wins, draws, losses, source, config, conflicts) -> None:
    """追加。格子是累计的：同一对多下几次就多记几次，绝不覆盖。

    但两次来源打架时要说出来：同一对里一方这次 80% 赢、上次 80% 输，说明有一份
    数据是错的（名字指错模型、或者 a/b 写反了）。这种格子照样累计进去，同时记进
    `conflicts.txt`，不假装没发生。
    """
    if a == b or min(wins, draws, losses) < 0 or wins + draws + losses == 0:
        return
    prev = cells.get((a, b))
    if prev and prev[0] + prev[1] + prev[2] >= 10 and wins + draws + losses >= 10:
        old_rate = (prev[0] + 0.5 * prev[1]) / sum(prev)
        new_rate = (wins + 0.5 * draws) / (wins + draws + losses)
        if (old_rate > 0.7 and new_rate < 0.3) or (old_rate < 0.3 and new_rate > 0.7):
            conflicts.append(
                f"{config}: {a} vs {b}: 已有 {prev[0]}-{prev[1]}-{prev[2]}, "
                f"这次 {wins}-{draws}-{losses}（来自 {source}）")
    prev = prev or (0, 0, 0)
    total = (prev[0] + wins, prev[1] + draws, prev[2] + losses)
    cells[(a, b)] = total
    cells[(b, a)] = (total[2], total[1], total[0])


def source_pairs(data, path):
    if "a" in data and "b" in data and "a_wins" in data:
        pairs = [(data["a"], data["b"], data["a_wins"], data.get("draws", 0), data["b_wins"])]
    elif "names" in data and "pairs" in data and "source_reports" not in data:
        pairs = data["pairs"]
    else:
        return []
    run_dir = next((parent for parent in path.parents if (parent / "manifest.json").exists()), None)
    mapping = gen_map(run_dir) if run_dir else {}
    def model(name):
        name = canonical(name)
        return mapping.get(name, name)

    return [(model(a), model(b), wins, draws, losses) for a, b, wins, draws, losses, *_ in pairs]


def import_all(out_dir: Path, roots: list[Path] | None = None) -> dict[int, int]:
    known = known_models().union(PSEUDO)
    ledger_path = out_dir / "scoped_sources.json"
    sources = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    scopes = {match_config(source) for source in sources.values()}
    imported: dict[int, int] = {}
    skipped_untagged = 0

    output_path = out_dir.resolve()
    paths = {path.resolve() for root in roots or [REPO / "runs" / "eval"]
             for path in root.rglob("*.json")}
    for path in sorted(paths):
        if output_path in path.parents or "leaderboard" in path.parts:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                continue
            pairs = source_pairs(report, path)
            if not pairs:
                continue
            ruleset, komi, visits = match_config(report)
            rows = [[a, b, int(wins), int(draws), int(losses)]
                    for a, b, wins, draws, losses in pairs]
            rows = [row for row in rows if row[0] != row[1] and min(row[2:]) >= 0
                    and sum(row[2:]) > 0]
            if not rows:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            skipped_untagged += 1
            continue
        key = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        source = {"ruleset": ruleset, "komi": komi, "visits": visits, "pairs": rows}
        if "seed" in report:
            source["seed"] = report["seed"]
        if source != sources.get(key):
            sources[key] = source
            imported[visits] = imported.get(visits, 0) + sum(
                sum(row[2:]) for row in rows if row[0] in known and row[1] in known)

    # CSVs are derived from the source ledger. After an interruption, rerunning
    # the import recreates the same counts instead of incrementing them twice.
    atomic_write(ledger_path, json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    tables = {scope: {} for scope in scopes}
    conflicts = []
    skipped_unknown = 0
    for key, source in sorted(sources.items()):
        config = match_config(source)
        cells = tables.setdefault(config, {})
        for a, b, wins, draws, losses in source["pairs"]:
            if a not in known or b not in known:
                skipped_unknown += 1
                continue
            add(cells, a, b, wins, draws, losses, key, config, conflicts)
    for (ruleset, komi, visits), cells in tables.items():
        write_table(board_directory(out_dir, ruleset, komi) / f"matches_{visits}.csv", cells)

    atomic_write(out_dir / "scoped_conflicts.txt", "\n".join(conflicts) + "\n" if conflicts else "")
    if conflicts:
        print(f"{len(conflicts)} conflicting cells -> {out_dir / 'scoped_conflicts.txt'}")
    if skipped_untagged:
        print(f"skipped {skipped_untagged} reports with missing or incompatible settings")
    if skipped_unknown:
        print(f"skipped {skipped_unknown} pairs with an unknown model")
    if imported:
        print("observed " + ", ".join(f"{visits}: {games} games from changed sources"
                                      for visits, games in sorted(imported.items())))
    return imported


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="directory to scan (repeatable, default runs/eval)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "leaderboard")
    args = ap.parse_args()
    roots = [Path(item) for item in args.root] if args.root else [REPO / "runs" / "eval"]
    import_all(args.out, roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
