#!/usr/bin/env python3
"""模型排行榜：读二维 CSV 里的胜负结果，算 ELO。

表格就是数据
------------
`runs/leaderboard/matches_<visits>.csv`：

    model,random,b4c32-s5960-d80938,b6c96-s13520-d183077
    random,—,0-0-120,0-0-120
    b4c32-s5960-d80938,120-0-0,—,87-0-33

格子 = 行方对列方的 `胜-和-负`，空格 = 没打过，对角线是 `—`。
想改结果就改这个文件：改一个三角就够了，另一边空着会自动镜像；
两边都填但互相矛盾时会警告。程序不写表格（`--import-existing` 除外，
那是把历史评测一次性填进来的入口）。

它只做一件事
------------
读表格 → 用 `training/elo.py` 的同一个 `fit_elo`（逻辑回归 + 高斯先验）拟合，
**random 钉在 0**，附 200 次 bootstrap 的 95% 区间 → 打印排行榜 + 写
`runs/leaderboard/leaderboard.md`。不同 `visits` 各是一张表、各算各的，
从不混在一起（混预算等于比较不同搜索量）。

    python training/leaderboard.py                 # 读所有 matches_*.csv
    python training/leaderboard.py --visits 500    # 只看一个预算
    python training/leaderboard.py --import-existing   # 把历史评测填进 CSV（一次性）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import bootstrap_ci, fit_elo  # noqa: E402  (同一个拟合，别的表才能比)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "leaderboard"
PSEUDO = ("random", "pruning", "mcts")


# ---------------------------------------------------------------------------
# 表格 ↔ (名字, 对局)
# ---------------------------------------------------------------------------
def parse_cell(text: str):
    text = (text or "").strip().replace("–", "-").replace("—", "-")
    if not text or text == "-":
        return None
    parts = text.split("-")
    if len(parts) != 3:
        raise ValueError(f"格子要写成 胜-和-负，读到 {text!r}")
    wins, draws, losses = (int(part) for part in parts)
    if min(wins, draws, losses) < 0 or wins + draws + losses == 0:
        return None
    return wins, draws, losses


def read_table(path: Path):
    """(名字顺序, 对称化的 results[(a, b)] = (win, draw, loss))"""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], {}
    header = [cell.strip() for cell in rows[0][1:]]
    order: list[str] = []
    raw: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        order.append(name)
        for col, text in zip(header, row[1:]):
            try:
                parsed = parse_cell(text)
            except ValueError as error:
                raise SystemExit(f"{path.name}: {name} vs {col}: {error}")
            if parsed:
                raw[(name, col)] = parsed
    for name in header:
        if name not in order:
            order.append(name)
    results: dict[tuple[str, str], tuple[int, int, int]] = {}
    for (a, b), value in raw.items():
        results[(a, b)] = value
        mirror = (value[2], value[1], value[0])
        if (b, a) in raw:
            if raw[(b, a)] != mirror:
                print(f"warning: {path.name}: {a} vs {b} 与 {b} vs {a} 矛盾", file=sys.stderr)
        else:
            results[(b, a)] = mirror
    return order, results


# ---------------------------------------------------------------------------
# 排行榜
# ---------------------------------------------------------------------------
def leaderboard(path: Path):
    order, results = read_table(path)
    pairs = [(a, b, wins, draws, losses, 1.0)
             for (a, b), (wins, draws, losses) in results.items() if a < b]
    games = sum(wins + draws + losses for (_, _, wins, draws, losses, _) in pairs)
    if not pairs:
        return order, {}, {}, games
    names = list(order)
    ratings = dict(zip(names, fit_elo(names, pairs)))
    intervals = dict(zip(names, bootstrap_ci(names, pairs)))
    if "random" in ratings:
        # 均匀随机走子是 0 分：不换算的话每张表的零点都不同，数字没法比较。
        zero = ratings["random"]
        ratings = {name: value - zero for name, value in ratings.items()}
        intervals = {name: (low - zero, high - zero) for name, (low, high) in intervals.items()}
    return order, ratings, intervals, games


def components(order, results) -> list[list[str]]:
    """Groups that never played each other: their relative Elo is arbitrary.

    A rating is only defined relative to someone.  Two groups with no game between
    them are connected only through the prior, so the number is a guess dressed up
    as a measurement -- the table says so instead of hiding it.
    """
    seen: dict[str, str] = {}
    parent: dict[str, str] = {name: name for name in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), _value in results.items():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for name in order:
        groups.setdefault(find(name), []).append(name)
    del seen
    return sorted(groups.values(), key=lambda group: -len(group))


def render(path: Path) -> list[str]:
    order, results = read_table(path)
    _, ratings, intervals, games = leaderboard(path)
    visits = path.stem.split("_")[1]
    lines = [f"## {visits} visits — {games} 局，{len(order)} 个参赛者", ""]
    if not ratings:
        return lines + ["（还没有数据）", ""]
    if "random" not in order:
        lines += ["> 这张表没有 `random`：分数只与池内均值可比。补几局 random 即可。", ""]
    groups = components(order, results)
    if len(groups) > 1:
        described = "、".join(
            "{" + ", ".join(group[:3]) + ("…" if len(group) > 3 else "") + "}"
            for group in groups)
        lines += [f"> 这些组之间没有对局：{described}。组与组之间的分数是任意定的，",
                  "> 只有组内比较有意义；补一场跨组对局（往往只需几十局）就能连上。", ""]
    ranked = sorted(order, key=lambda name: -ratings.get(name, -1e9))
    lines.append("| 模型 | ELO | 95% CI | " + " | ".join(ranked) + " |")
    lines.append("| --- | ---: | --- | " + " | ".join("---:" for _ in ranked) + " |")
    for row in ranked:
        ci = intervals.get(row)
        ci_text = f"[{ci[0]:+.0f}, {ci[1]:+.0f}]" if ci else "—"
        cells = []
        for col in ranked:
            if col == row:
                cells.append("—")
                continue
            value = results.get((row, col))
            cells.append("-".join(str(part) for part in value) if value else "·")
        lines.append(f"| `{row}` | {ratings.get(row, 0.0):+.0f} | {ci_text} | " +
                     " | ".join(cells) + " |")
    return lines + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visits", default="",
                    help="只看一个预算（默认读 runs/leaderboard/matches_*.csv 全部）")
    ap.add_argument("--all", action="store_true",
                    help="also show tables that hold only engine kinds")
    ap.add_argument("--import-existing", action="store_true",
                    help="把 runs/ 下已有的评测结果一次性填进 CSV，然后照常算榜")
    args = ap.parse_args(argv)

    if args.import_existing:
        from leaderboard_import import import_all
        import_all(OUT)

    paths = []
    for path in sorted(OUT.glob("matches_*.csv")):
        if args.visits and path.stem != f"matches_{args.visits}":
            continue
        order, _results = read_table(path)
        # A table of engine kinds (random/pruning/mcts) is a fine measurement but
        # not a model ranking: leave it out unless it is asked for.
        if args.all or any(name not in PSEUDO for name in order):
            paths.append(path)
    if not paths:
        print(f"{OUT} 下没有 matches_*.csv：先用 --import-existing 填，或手工建一张表")
        return 1

    lines = ["# 模型排行榜", "",
             "由 `training/leaderboard.py` 从 `runs/leaderboard/matches_<visits>.csv` 算出。",
             "格子是 `胜-和-负`（从行方看），`·` = 没打过。ELO 以 **random = 0** 为锚，",
             "95% 区间是 200 次 bootstrap；不同 visits 从不混算。", ""]
    for path in paths:
        lines += render(path)
        _, ratings, _, _ = leaderboard(path)
        if ratings:
            top = sorted(ratings.items(), key=lambda kv: -kv[1])[:5]
            print(f"{path.stem.split('_')[1]:>5} visits: " +
                  ", ".join(f"{name} {value:+.0f}" for name, value in top))
    path = OUT / "leaderboard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
