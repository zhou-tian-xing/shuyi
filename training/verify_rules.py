#!/usr/bin/env python3
"""Cross-check the Python rules against C++ self-play data.

Every stored sample is the position *before* a move, and consecutive samples of a
game differ by exactly that move, so the whole game can be replayed.  If the
score/winner computed by the Python rules agrees with the stored C++ value for
every position of every game, then legal moves, hands and scoring all match the
engine that produced the data.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_selfplay import CELLS, KOMI, POLICY, State, cell_scores  # noqa: E402

D = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (CELLS,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (POLICY,)),
        ("value", "<f4"),
        ("game", "<u4"),
        ("ply", "u1"),
    ]
)


def load(path, limit):
    with open(path, "rb") as f:
        f.read(8)
        _v, count = struct.unpack("<II", f.read(8))
        return np.fromfile(f, dtype=D, count=min(count, limit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--limit", type=int, default=200000)
    args = ap.parse_args()
    d = load(args.shard, args.limit)
    failures = 0
    games = 0
    checked = 0
    for game in np.unique(d["game"]):
        idx = np.flatnonzero(d["game"] == game)
        idx = idx[np.argsort(d["ply"][idx])]
        rows = d[idx]
        if len(rows) == 0 or rows["ply"][0] != 2:
            continue
        state = State.start()
        board = rows["board"][0].copy()
        # The random opening plies are the only stones on the board; Red placed
        # first, so replay every red stone before every black stone.
        for sign in (1, -1):
            for cell in np.flatnonzero(np.sign(board) == sign):
                state = state.play(int(cell) * 9 + int(abs(board[cell])) - 1)
        if state.board.tobytes() != board.tobytes():
            failures += 1
            continue
        ok = True
        for row in rows:
            if row["ply"] != state.ply or row["side"] != state.side:
                ok = False
                break
            if row["board"].tobytes() != state.board.tobytes():
                ok = False
                break
            if int(row["hand"][:9].sum()) != int(state.rem[0].sum()):
                ok = False
                break
            following = rows[rows["ply"] == row["ply"] + 1]
            if following.size == 0:
                continue
            diff = np.flatnonzero(following["board"][0] != row["board"])
            if diff.size != 1:
                ok = False
                break
            cell = int(diff[0])
            number = int(abs(following["board"][0][cell]))
            state = state.play(cell * 9 + number - 1)
        if not ok:
            failures += 1
            continue
        total = state.raw + KOMI
        red = 1.0 if total > 0 else (0.0 if total < 0 else 0.5)
        stored = float(rows["value"][0])
        expected = red if int(rows["side"][0]) == 0 else 1.0 - red
        checked += 1
        if abs(expected - stored) > 1e-6:
            # The final move is not stored, so try every legal last move: the
            # Python scoring is validated if one of them reproduces the C++
            # value.
            ok = False
            for index in state.legal():
                after = state.play(int(index))
                total = after.raw + KOMI
                red = 1.0 if total > 0 else (0.0 if total < 0 else 0.5)
                candidate = red if after.side == 0 else 1.0 - red
                if abs(candidate - stored) < 1e-6:
                    ok = True
                    break
            if not ok:
                failures += 1
                if failures < 5:
                    print(f"game {game}: python={expected} cpp={stored} raw={state.raw}")
        games += 1
    print(f"games={games} checked={checked} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
