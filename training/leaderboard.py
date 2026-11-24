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
def leaderboard(path: Path, anchor: str = "random"):
    order, results = read_table(path)
    pairs = [(a, b, wins, draws, losses, 1.0)
             for (a, b), (wins, draws, losses) in results.items() if a < b]
    games = sum(wins + draws + losses for (_, _, wins, draws, losses, _) in pairs)
    if not pairs:
        return order, {}, {}, games
    names = list(order)
    ratings = dict(zip(names, fit_elo(names, pairs)))
    intervals = dict(zip(names, bootstrap_ci(names, pairs)))
    if anchor in ratings:
        # 锚就是 0 分：不换算的话每张表的零点都不同，数字没法比较。
        zero = ratings[anchor]
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


def render(path: Path, anchor: str = "random") -> list[str]:
    order, results = read_table(path)
    _, ratings, intervals, games = leaderboard(path, anchor)
    visits = path.stem.split("_")[1]
    lines = [f"## {visits} visits — {games} 局，{len(order)} 个参赛者", ""]
    if not ratings:
        return lines + ["（还没有数据）", ""]
    if anchor not in order:
        lines += [f"> 这张表没有锚 `{anchor}`：分数只与池内均值可比。", ""]
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


def pool_score(results) -> dict[str, tuple[float, int]]:
    """(score, games) per model over the whole table: 胜=1 和=0.5 负=0。

    这张表里大量格子是 20-0，ELO 的数值会被拉到几千分、区间也失去意义；
    池内胜率是同一批数据的稳当读法，按它看"预算上去有没有变强"不会被饱和格子骗。
    """
    total: dict[str, list[float]] = {}
    for (a, b), (wins, draws, losses) in results.items():
        for name, won, drew, lost in ((a, wins, draws, losses), (b, losses, draws, wins)):
            row = total.setdefault(name, [0.0, 0])
            row[0] += won + 0.5 * drew
            row[1] += won + drew + lost
    return {name: (score, int(games)) for name, (score, games) in total.items()}


def compare(paths: list[Path], anchor: str) -> list[str]:
    """同一批模型在不同预算下的分数：涨了还是没涨。"""
    tables = []
    skipped = []
    sparse = []
    for path in paths:
        order, results = read_table(path)
        if anchor not in order:
            skipped.append(path.stem.split("_")[1])
            continue
        _, ratings, intervals, games = leaderboard(path, anchor)
        if ratings:
            label = path.stem.split("_")[1]
            need = len(order) * (len(order) - 1) // 2
            played = sum(1 for (a, b) in results if a < b)
            if need and played < 0.9 * need:
                sparse.append(f"{label}（{played}/{need} 对）")
            tables.append((label, ratings, intervals, games, pool_score(results)))
    if len(tables) < 2:
        return ["", "（少于两张带锚的表，没有可比的预算）", ""]
    tables.sort(key=lambda item: int(item[0]))
    names = sorted({name for _, ratings, _, _, _ in tables for name in ratings},
                   key=lambda name: -sum(r.get(name, -1e9) for _, r, _, _, _ in tables) / len(tables))
    lines = ["", "## 同一模型跨预算对比", "",
             f"锚：`{anchor}`（每张表都减掉它的分数）。括号里是相对最便宜一张表的差。", ""]
    lines.append("| 模型 | " + " | ".join(f"{label} visits" for label, _, _, _, _ in tables) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tables) + " |")
    base_label, base_ratings, _, _, base_pool = tables[0]
    for name in names:
        first = base_ratings.get(name)
        cells = []
        for label, ratings, intervals, _, pool in tables:
            value = ratings.get(name)
            if value is None:
                cells.append("·")
                continue
            delta = value - first if first is not None else 0.0
            mark = "" if label == base_label else f" ({delta:+.0f})"
            cells.append(f"{value:+.0f}{mark}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    # Same data, but as "score against the whole pool", which stays readable when
    # a cell is 20-0 and the rating differences run into the thousands.
    lines += ["", "### 同一模型跨预算：池内胜率", "",
              "格子是 `(胜+和/2)/局数`，后面的括号是相对最便宜一张表的百分点差。", ""]
    lines.append("| 模型 | " + " | ".join(f"{label} visits" for label, _, _, _, _ in tables) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tables) + " |")
    for name in names:
        first = base_pool.get(name)
        cells = []
        for label, _, _, _, pool in tables:
            entry = pool.get(name)
            if entry is None or entry[1] == 0:
                cells.append("·")
                continue
            score, games = entry
            rate = 100.0 * score / games
            delta = (rate - 100.0 * first[0] / first[1]) if first and first[1] else 0.0
            mark = "" if label == base_label else f" ({delta:+.1f})"
            cells.append(f"{rate:.1f}%{mark}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "> 括号里的差若小于 CI 宽度（通常 ±40~90 ELO，或 ±2~3 个百分点），",
              "> 就是噪声，不是变强了。", ""]
    if skipped:
        lines += ["", f"> 这些预算的表里没有 `{anchor}`，跨表比不了，已略过："
                  + ", ".join(skipped) + "。", ""]
    if sparse:
        lines += ["", "> 这些表只下了一部分对局（" + "、".join(sparse) + "），"
                  "池内胜率只按下过的对局算，和满表放一起看会偏高或偏低。", ""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visits", default="",
                    help="只看一个预算（默认读 runs/leaderboard/matches_*.csv 全部）")
    ap.add_argument("--anchor", default="random",
                    help="把它的分数定为 0（默认 random；没有 random 时可用某个模型）")
    ap.add_argument("--all", action="store_true",
                    help="also show tables that hold only engine kinds")
    ap.add_argument("--import-existing", action="store_true",
                    help="把 runs/eval 下新的评测结果填进 CSV，然后照常算榜")
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
             "格子是 `胜-和-负`（从行方看），`·` = 没打过。",
             f"ELO 以 **{args.anchor} = 0** 为锚，95% 区间是 200 次 bootstrap；",
             "不同 visits 从不混算。", ""]
    for path in paths:
        lines += render(path, args.anchor)
        _, ratings, _, _ = leaderboard(path, args.anchor)
        if ratings:
            top = sorted(ratings.items(), key=lambda kv: -kv[1])[:5]
            print(f"{path.stem.split('_')[1]:>5} visits: " +
                  ", ".join(f"{name} {value:+.0f}" for name, value in top))
    lines += compare(paths, args.anchor)
    path = OUT / "leaderboard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
