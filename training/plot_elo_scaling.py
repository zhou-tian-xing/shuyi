#!/usr/bin/env python3
"""Data-scaling chart: one dot per local weight, Elo against cumulative positions.

The y axis is the 64-visit board's own fit (`random = 0`), which is one
self-consistent scale: every number comes from the same fit over the same
match graph.  Mixing in a differently anchored rating would put two rulers on
one axis, so the separately anchored re-measurement lives in the CSV only.

    python training/plot_elo_scaling.py
"""
import argparse
import csv
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.optimize import least_squares
from scipy.special import expit

from elo import fit_elo_checked
from evalconfig import match_config
from leaderboard import REPO, components, decisive, read_table

NAME = re.compile(r"b(\d+)c(\d+)-s\d+-d(\d+)")
BASE, REF, LAST = ("b6c96-s87696-d937553", "b6c96-s179866-d1959110", "b6c96-s305983-d3226791")
CEILING_GAP = 400.0
COLORS = {"b2c16": "#7c8798", "b4c32": "#0d6fb8", "b6c64": "#a83a7c",
          "b6c96": "#c2521f", "b10c128": "#1f7a5a"}


def inventory():
    """(relative path, name, architecture, cumulative positions) per weight file."""
    for scope in (REPO / "models", REPO / "runs"):
        for path in sorted(scope.rglob("*.bin")):
            if scope.name == "runs" and path.parent.name != "models":
                continue
            with path.open("rb") as handle:
                header = handle.read(28)
            if header[:8] not in (b"SHUYINN2", b"SHUYINN3", b"SHUYINN4"):
                continue
            _, channels, blocks, _, _ = struct.unpack("<IIIII", header[8:])
            arch = f"b{blocks}c{channels}"
            match = NAME.fullmatch(path.stem)
            if match and arch != f"b{match[1]}c{match[2]}":
                raise ValueError(f"{path}: name and header disagree about the architecture")
            yield path.relative_to(REPO), path.stem, arch, int(match[3]) if match else None


def match_sets(board, visits):
    """(whole table, the subset whose sources carry a verified scope)."""
    _, results = read_table(board)
    whole = {key: value for key, value in results.items() if key[0] < key[1]}
    ledger = json.loads(board.with_suffix(board.suffix + ".sources.json").read_text(encoding="utf-8"))
    tagged = defaultdict(lambda: (0, 0, 0))
    for path, report in ledger["sources"].items():
        if match_config(report) != ("nopass", 0.5, visits):
            raise ValueError(f"unexpected scope in {path}")
        for a, b, wins, draws, losses in report["pairs"]:
            if b < a:
                a, b, wins, losses = b, a, losses, wins
            tagged[a, b] = tuple(x + y for x, y in zip(tagged[a, b], (wins, draws, losses)))
    for key, value in tagged.items():
        if any(x > y for x, y in zip(value, whole[key])):
            raise ValueError(f"tagged totals exceed the matrix cell {key}")
    return whole, tagged


def anchored_fit(cells, anchor):
    """Elo with `anchor` at 0, over the anchor's connected component only."""
    usable = {key: value for key, value in cells.items() if decisive(*value)}
    names = sorted({name for key in usable for name in key})
    try:
        group = next(g for g in components(names, usable) if anchor in g)
    except StopIteration:
        raise SystemExit(f"{anchor} has no decisive match, cannot anchor")
    pairs = [(a, b, *value, 1.0) for (a, b), value in sorted(usable.items()) if a in group]
    values, moved = fit_elo_checked(group, pairs, passes=10000, tolerance=0.0001)
    if moved >= 0.0001:
        raise SystemExit(f"Elo fit did not converge (last sweep moved {moved:.4g})")
    zero = values[group.index(anchor)]
    return dict(zip(group, (v - zero for v in values))), sum(sum(p[2:5]) for p in pairs)


def draw(rows, games, out, visits, emphasis):
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Microsoft YaHei", "DengXian", "Noto Sans CJK SC", "SimHei")
                 if n in available), None)
    if font is None:
        raise SystemExit("a Chinese plotting font is required")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "text.color": "#26313d", "axes.labelcolor": "#26313d",
                         "xtick.color": "#47556a", "ytick.color": "#47556a"})
    fig, ax = plt.subplots(figsize=(11.4, 6.5), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    palette = dict(COLORS)
    for row in rows:
        palette.setdefault(row[2], "#26313d")
    rated = [row for row in rows if row[3] and row[4] is not None]
    # `emphasis` marks the board where only part of the cells were re-measured
    # under a verified scope; a board measured end to end gets one marker style.
    newest = {row[1] for row in rows if row[5] is not None} if emphasis else set()

    ax.set_xscale("log")
    ax.set_xlim(2.6e4, 4.1e6)
    ax.xaxis.set_major_locator(FixedLocator([3e4, 1e5, 3e5, 1e6, 3e6]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e4:g}万"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", color="#e3e9f2", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b3bfd0")

    for x, y, arch, name in [(r[3], r[4], r[2], r[1]) for r in rated]:
        if not emphasis or name in newest:      # re-measured under the documented scope
            ax.scatter(x, y, s=58, color=palette[arch], edgecolor="white",
                       linewidth=.9, zorder=3)
        else:
            ax.scatter(x, y, s=34, color=palette[arch], alpha=.42,
                       edgecolor="none", zorder=2)

    # One logistic per architecture: a single curve across architectures fits
    # badly (RMSE ~120 Elo) because each one has its own data economy.
    print(f"assumed ceiling = the architecture's best point + {CEILING_GAP} Elo")
    for arch in palette:
        pts = sorted((r[3], r[4]) for r in rated if r[2] == arch)
        if len(pts) < 8:
            continue
        log_x = np.log10([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        top = y.max() + CEILING_GAP

        def curve(z, p):
            low, rate, mid = p
            return low + (top - low) * expit(rate * (z - mid))

        fit = least_squares(lambda p: curve(log_x, p) - y, [y.min() - 400, 4, 6.2],
                            bounds=([-10000, .001, 3], [y.min(), 100, 9]))
        grid = np.linspace(log_x.min(), log_x.max(), 240)
        ax.plot(10 ** grid, curve(grid, fit.x), color=palette[arch], lw=1.5, alpha=.85,
                label=f"{arch}（渐近线 +{top:.0f}）")
        rmse = np.sqrt(np.mean(fit.fun ** 2))
        print(f"  {arch}: {len(pts)} points, {pts[0][0]/1e4:.0f}万→{pts[-1][0]/1e4:.0f}万 positions, "
              f"{pts[0][1]:.0f}→{pts[-1][1]:.0f} Elo, kernel {10**fit.x[2]/1e4:.0f}万, RMSE {rmse:.0f}")

    for name, offset in [(BASE, (8, 10)), (REF, (8, -20)), (LAST, (-10, 14))]:
        x, elo = next((r[3], r[4]) for r in rated if r[1] == name)
        ax.annotate(name.split("-")[1], (x, elo), xytext=offset, textcoords="offset points",
                    fontsize=8.5, color="#8a3b12",
                    ha="right" if offset[0] < 0 else "left")

    handles = [Line2D([], [], marker="o", ls="none", ms=8, color=palette[a], label=a)
               for a in palette if any(r[2] == a for r in rated)]
    if emphasis:
        handles.append(Line2D([], [], marker="o", ls="none", ms=5.5,
                              markerfacecolor="#8894a6", markeredgecolor="none",
                              alpha=.45, label="只有旧榜证据（范围未核验）"))
        handles.append(Line2D([], [], marker="o", ls="none", ms=7.5, markerfacecolor="#8894a6",
                              markeredgecolor="white", label="2026-11-26 同条件复测"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              title="架构 / 证据" if emphasis else "架构", title_fontsize=9)

    ax.set_xlabel("累计自对弈局面数（对数刻度）", fontsize=10.5, labelpad=8)
    ax.set_ylabel(f"ELO（{visits} visits，random = 0）", fontsize=10.5, labelpad=8)
    ax.set_title(f"棋力随训练数据量增长（{visits} visits）", loc="left",
                 fontsize=19, weight="bold", pad=30)
    ax.text(0, 1.02, f"{len(rated)} 份可评分权重 · {games:,} 局能读出分差 · "
                     "曲线是每种架构各拟合一条逻辑斯蒂，渐近线按最优点 +400 假定",
            transform=ax.transAxes, fontsize=9.5, color="#4d5c70")
    fig.text(.125, .018, ("这张表的每条边都在同一批条件下重测（nopass、A=+0.5、"
                          f"{visits} visits），所有点在同一个零点上；"
                          if not emphasis else
                          "旧榜里有一部分对局来自未标记来源，其规则与 A 值未逐条核验；"
                          "淡色点没有同条件复测，低数据端的点因此不作强结论；")
                         + "折线只表示趋势，横轴点位按文件名中的累计局面数。",
             fontsize=8.5, color="#5d6b7e")
    fig.subplots_adjust(left=.085, right=.975, top=.845, bottom=.135)
    stem = "elo_vs_data" if visits == 64 else f"elo_vs_data_{visits}"
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=REPO / "runs/leaderboard/matches_64.csv")
    parser.add_argument("--visits", type=int, default=None,
                        help="board scope; default is the number in the file name")
    parser.add_argument("--out", type=Path, default=REPO / "runs/analysis/scaling")
    parser.add_argument("--figures", type=Path, default=REPO / "docs/figures",
                        help="where the PNG/SVG go; the CSV stays in --out")
    args = parser.parse_args()
    visits = args.visits or int(re.search(r"matches_(\d+)", args.board.name)[1])
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    whole, tagged = match_sets(args.board, visits)
    history, games = anchored_fit(whole, "random")
    # A board measured end to end needs no second ruler; the 64-visit table is
    # the only one where part of the cells predate the verified scope.
    emphasis = set(tagged) != set(whole)
    if emphasis:
        verified, verified_games = anchored_fit(tagged, REF)
    else:
        verified, verified_games = {}, games
        print(f"{args.board.name}: every cell carries a verified scope, one ruler")
    rows = [(p.as_posix(), n, a, d, history.get(n), verified.get(n))
            for p, n, a, d in inventory()]
    stem = "elo_vs_data" if visits == 64 else f"elo_vs_data_{visits}"
    with (args.out / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(f"# historical_elo: {visits}-visit board fit, random = 0, "
                     f"{games} decisive games\n")
        handle.write("# verified_elo: " + ("same-fit subset anchored at "
                     f"{REF} = 0, {verified_games} decisive games\n" if emphasis else
                     "identical to historical_elo, no unverified cells remain\n"))
        writer = csv.writer(handle)
        writer.writerow(("file", "model", "architecture", "cumulative_positions",
                         "historical_elo", "verified_elo"))
        writer.writerows(rows)
    draw(rows, games, args.figures, visits, emphasis)
    rated = sum(1 for r in rows if r[3] and r[4] is not None)
    print(f"{len(rows)} files, {len({r[1] for r in rows})} names, {rated} rated, "
          f"{sum(r[5] is not None for r in rows)} verified")
    print(args.figures / f"{stem}.png")


if __name__ == "__main__":
    main()
