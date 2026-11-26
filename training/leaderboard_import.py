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


def unadd(cells, a, b, wins, draws, losses) -> None:
    """`add` 的反向：一处对局如果重测了，先把上一次记进去的那份拿走再加新的。

    拿不回来就只能停：负数说明这张表被别人手工改过，猜下去就是把数据改坏。
    """
    for (x, y), cell in (((a, b), (wins, draws, losses)), ((b, a), (losses, draws, wins))):
        prev = cells.get((x, y))
        if prev is None:
            raise SystemExit(f"表里没有 {x} vs {y}，但记录说加过 —— 表被手工改过？")
        rest = (prev[0] - cell[0], prev[1] - cell[1], prev[2] - cell[2])
        if min(rest) < 0:
            raise SystemExit(f"{x} vs {y}: 表里是 {prev[0]}-{prev[1]}-{prev[2]}，"
                             f"要拿走的却是 {cell[0]}-{cell[1]}-{cell[2]}")
        if rest == (0, 0, 0):
            del cells[(x, y)]
        else:
            cells[(x, y)] = rest


def read_cells(path: Path):
    """把一张 `胜-和-负` 表读成 {(甲, 乙): (胜, 和, 负)}，两个方向都填上。"""
    from leaderboard import parse_cell  # 一个解析器，两种表：不另写一份

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise SystemExit(f"{path} 是空文件")
    header = [cell.strip() for cell in rows[0][1:]]
    cells: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        for col, text in zip(header, row[1:]):
            try:
                parsed = parse_cell(text)
            except ValueError as error:
                raise SystemExit(f"{path.name}: {name} vs {col}: {error}") from None
            if not parsed or name == col:
                continue
            mirror = cells.get((col, name))
            if mirror is not None:
                # 两边都写了同一个格子：一致就跳过，不一致就停（先手工查清哪边对）。
                if mirror != (parsed[2], parsed[1], parsed[0]):
                    raise SystemExit(f"{path.name}: {name} vs {col} 是 {parsed}，"
                                     f"镜像格却是 {mirror} —— 表里两半自相矛盾")
                continue
            cells[(name, col)] = parsed
            cells[(col, name)] = (parsed[2], parsed[1], parsed[0])
    return cells


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


def append_into(table: Path, roots: list[Path]) -> int:
    """把 `roots` 下的对局加进一张已经在用的表，而不是从零重建一张。

    老表是历史：它里面的对局分散在当年的 eval 目录里，很多没有评测条件标记，
    重建只会得到一张空表。所以这里反过来做——表是底稿，新对局往上加：

        * 一个来源文件的对局只记一次；重测同一个文件时先抽掉上一次那份再记新的，
          所以重跑是幂等的，不会悄悄翻倍。
        * 抽掉的格子和表里对不上就直接停：说明这张表被手工改过。
        * 表名决定预算（`matches_64.csv` 只能收 64 visits 的对局），
          一次只收一种规则 + A 值，混着收就等于把不同条件下的胜负加在一起。
    """
    cells = read_cells(table)
    ledger_path = table.with_name(table.name + ".sources.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    if not isinstance(ledger, dict):
        raise SystemExit(f"{ledger_path} 不是这份工具写的账本（不是对象），先手工看一遍")
    sources: dict = ledger.get("sources", {})
    known = known_models().union(PSEUDO)
    try:
        expected_visits = int(table.stem.split("_")[1])
    except (IndexError, ValueError):
        raise SystemExit(f"{table.name} 的名字里读不出预算：表要叫 matches_<visits>.csv") from None

    before_names = {name for pair in cells for name in pair}
    conflicts: list[str] = []
    scope = None
    added_games = 0
    changed: list[str] = []
    seen: set[str] = set()
    for path in sorted({p.resolve() for root in roots for p in root.rglob("*.json")}):
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
            print(f"skip {path}: 没有评测条件标记，不知道该放进哪张表")
            continue
        if visits != expected_visits:
            raise SystemExit(f"{path.name} 是 {visits} visits，但 {table.name} 是 "
                             f"{expected_visits} visits 的表")
        if scope is None:
            scope = (ruleset, komi)
        elif scope != (ruleset, komi):
            raise SystemExit(f"{path.name} 是 {ruleset}/A={komi:g}，前面的是 "
                             f"{scope[0]}/A={scope[1]:g}：一次只能往一张表加一种条件")
        unknown = sorted({name for row in rows for name in row[:2] if name not in known})
        if unknown:
            raise SystemExit("这些名字找不到对应的模型文件，加进去就是一行查不到的东西: "
                             + ", ".join(unknown))
        key = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
        seen.add(key)
        previous = sources.get(key)
        if previous is not None:
            if previous["pairs"] == rows:
                continue  # 同一个文件，同样的对局：已经记过了
            for a, b, wins, draws, losses in previous["pairs"]:
                unadd(cells, a, b, wins, draws, losses)
        for a, b, wins, draws, losses in rows:
            add(cells, a, b, wins, draws, losses, key, f"{ruleset}/A={komi:g}", conflicts)
            added_games += wins + draws + losses
        sources[key] = {"ruleset": ruleset, "komi": komi, "visits": visits, "pairs": rows}
        changed.append(key)

    for key in list(sources):
        if key not in seen:
            print(f"note: {key} 不在这次扫描的目录里，它记的对局留在表里")

    write_table(table, cells)
    atomic_write(ledger_path, json.dumps({"sources": sources}, ensure_ascii=False,
                                         indent=2) + "\n")
    new_names = sorted({name for pair in cells for name in pair} - before_names)
    print(f"{len(changed)} source(s), {added_games} games -> {table}")
    if new_names:
        print(f"new participants: {', '.join(new_names)}")
    if conflicts:
        report = table.with_name(table.name + ".conflicts.txt")
        atomic_write(report, "\n".join(conflicts) + "\n")
        print(f"{len(conflicts)} conflicting cell(s) -> {report}")
    return added_games


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="directory to scan (repeatable, default runs/eval)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "leaderboard")
    ap.add_argument("--into", type=Path, default=None,
                    help="append the scanned games into this existing table (胜-和-负 CSV) "
                         "instead of rebuilding the scoped tables")
    args = ap.parse_args()
    roots = [Path(item) for item in args.root] if args.root else [REPO / "runs" / "eval"]
    if args.into:
        append_into(args.into, roots)
    else:
        import_all(args.out, roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
