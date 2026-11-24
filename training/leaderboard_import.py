#!/usr/bin/env python3
"""把评测结果（`roundrobin.py` 的 pair JSON）填进 `matches_<visits>.csv`。

只做"填表"，不算 ELO —— 那是 `leaderboard.py` 的事。
名字要落到真实存在的模型上：`gen_003` 用该 run 的 manifest 换成累计模型名，
`gpu:`/`mcts_nn:` 前缀和旧的 `-blk`/`-cls` 后缀去掉；换不出来、仓库里也没有
这个文件的对局直接丢掉（宁愿少一行，也不要一行指不到模型的分）。

已经数过的文件记在 `runs/leaderboard/imported_sources.json`，所以重复运行不会
把同一批对局加两遍。默认只扫 `runs/eval/`（新跑的数据放这里）；历史的
runs/training/**/ 结果在 2026-11-24 已经一次性导入过。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

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


def read_table(path: Path):
    if not path.exists():
        return [], {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [cell.strip() for cell in rows[0][1:]]
    order = [row[0].strip() for row in rows[1:] if row and row[0].strip()]
    cells = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        for col, text in zip(header, row[1:]):
            text = (text or "").strip().replace("—", "-").replace("–", "-")
            if not text or text == "-":
                continue
            parts = text.split("-")
            if len(parts) != 3:
                continue
            try:
                cells[(row[0].strip(), col)] = tuple(int(part) for part in parts)
            except ValueError:
                continue
    return order, cells


def write_table(path: Path, order, cells) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"] + order)
        for row in order:
            line = [row]
            for col in order:
                if row == col:
                    line.append("—")
                    continue
                value = cells.get((row, col))
                line.append("-".join(str(part) for part in value) if value else "")
            writer.writerow(line)


def add(cells, a, b, wins, draws, losses) -> None:
    if a == b or min(wins, draws, losses) < 0 or wins + draws + losses == 0:
        return
    prev = cells.get((a, b), (0, 0, 0))
    total = (prev[0] + wins, prev[1] + draws, prev[2] + losses)
    cells[(a, b)] = total
    cells[(b, a)] = (total[2], total[1], total[0])


def import_all(out_dir: Path, roots: list[Path] | None = None) -> dict[int, int]:
    known = known_models()
    imported: dict[int, int] = {}
    skipped = 0
    seen: set[Path] = set()
    # Every file counted into the tables is remembered, so running the import
    # twice does not add the same games twice.
    ledger_path = out_dir / "imported_sources.json"
    ledger: set[str] = set()
    if ledger_path.exists():
        ledger = set(json.loads(ledger_path.read_text(encoding="utf-8")))

    def record(visits, a, b, wins, draws, losses):
        nonlocal skipped
        a, b = canonical(a), canonical(b)
        for name in (a, b):
            if name not in PSEUDO and name not in known:
                skipped += 1
                return
        path = out_dir / f"matches_{visits}.csv"
        order, cells = read_table(path)
        for name in (a, b):
            if name not in order:
                order.append(name)
        add(cells, a, b, wins, draws, losses)
        write_table(path, order, cells)
        imported[visits] = imported.get(visits, 0) + wins + draws + losses

    scanned: list[Path] = []
    for root in roots or [REPO / "runs"]:
        scanned.extend(sorted(root.rglob("*.json")))
    for path in scanned:
        if path in seen or "leaderboard" in path.parts:
            continue
        key = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        if key in ledger:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - not every json is a result file
            continue
        if not isinstance(data, dict):
            continue
        visits = int(data.get("visits") or 0)
        if visits <= 0:
            continue
        if "a" in data and "b" in data and "a_wins" in data:
            seen.add(path)
            record(visits, data["a"], data["b"], int(data["a_wins"]),
                   int(data.get("draws", 0)), int(data["b_wins"]))
            ledger.add(key)
            continue
        if "names" not in data or "pairs" not in data:
            continue
        seen.add(path)
        # gen_NNN only means something inside its own run.
        run_dir = None
        for parent in path.parents:
            if (parent / "manifest.json").exists():
                run_dir = parent
                break
        mapping = gen_map(run_dir) if run_dir else {}
        for entry in data["pairs"]:
            a = canonical(entry[0])
            b = canonical(entry[1])
            a = mapping.get(a, a)
            b = mapping.get(b, b)
            record(visits, a, b, int(entry[2]), int(entry[3]), int(entry[4]))
        ledger.add(key)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(sorted(ledger), indent=0) + "\n", encoding="utf-8")

    if skipped:
        print(f"skipped {skipped} pair(s) naming a model that is not in the repo",
              flush=True)
    if imported:
        print("filled — " + ", ".join(f"{visits}: {games} games"
                                     for visits, games in sorted(imported.items())),
              flush=True)
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
