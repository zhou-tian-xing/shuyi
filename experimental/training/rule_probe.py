#!/usr/bin/env python3
"""Structural statistics of the classic rule vs the proposed blocking rule.

Classic:  any empty cell x any number still in hand.
Blocking: additionally, a cell is illegal for number n if one of its four
          orthogonal neighbours holds an *oppposing* stone of number >= n.

Both use the same score: for every cell, compare the largest red and largest
black number in the 5-cell cross; the sign of that comparison sums to the raw
score.  No captures, 27 stones per side.

Random play is a bad player but a fine instrument here: it measures the shape of
the legal-move set (branching factor), how often a side runs out of moves, and
the distribution of final scores -- all of which decide whether a search can
work on the new rule at all.
"""
import argparse
import random
from collections import Counter

SIZE = 9
CELLS = SIZE * SIZE
NUMBERS = 9
COPIES = 3
TOTAL_PLIES = 2 * NUMBERS * COPIES


def cross(cell):
    x, y = cell % SIZE, cell // SIZE
    out = [cell]
    if x > 0:
        out.append(cell - 1)
    if x < SIZE - 1:
        out.append(cell + 1)
    if y > 0:
        out.append(cell - SIZE)
    if y < SIZE - 1:
        out.append(cell + SIZE)
    return out


CROSSES = [cross(c) for c in range(CELLS)]
NEIGHBOURS = [cross(c)[1:] for c in range(CELLS)]
NEIGHBOURS8 = [
    [n for n in range(CELLS)
     if n != c and abs(n % SIZE - c % SIZE) <= 1 and abs(n // SIZE - c // SIZE) <= 1]
    for c in range(CELLS)
]


class State:
    __slots__ = ("board", "rem", "ply", "passes")

    def __init__(self):
        self.board = [0] * CELLS
        self.rem = [[COPIES] * NUMBERS for _ in range(2)]
        self.ply = 0
        self.passes = 0

    def copy(self):
        s = State()
        s.board = self.board[:]
        s.rem = [self.rem[0][:], self.rem[1][:]]
        s.ply = self.ply
        s.passes = self.passes
        return s

    @property
    def side(self):
        return self.ply & 1

    def blocked(self, cell, number, blocking, ring=NEIGHBOURS):
        """Is `cell` forbidden to the side to move for this number?"""
        if not blocking:
            return False
        sign = 1 if self.side == 0 else -1
        for n in ring[cell]:
            v = self.board[n]
            if v and (v > 0) != (sign > 0) and abs(v) >= number:
                return True
        return False

    def legal(self, blocking, ring=NEIGHBOURS):
        out = []
        side = self.side
        if self.ply >= TOTAL_PLIES:
            return out
        for number in range(1, NUMBERS + 1):
            if self.rem[side][number - 1] == 0:
                continue
            for cell in range(CELLS):
                if self.board[cell]:
                    continue
                if self.blocked(cell, number, blocking, ring):
                    continue
                out.append((cell, number))
        return out

    def play(self, move):
        cell, number = move
        self.board[cell] = number if self.side == 0 else -number
        self.rem[self.side][number - 1] -= 1
        self.ply += 1
        self.passes = 0


def raw_score(board):
    total = 0
    for cells in CROSSES:
        red = black = 0
        for c in cells:
            v = board[c]
            if v > red:
                red = v
            elif -v > black:
                black = -v
        total += (red > black) - (red < black)
    return total


def run_game(blocking, rng, max_plies=TOTAL_PLIES, ring=NEIGHBOURS):
    state = State()
    while state.ply < max_plies:
        moves = state.legal(blocking, ring)
        if not moves:
            state.passes += 1
            state.ply += 1  # a pass consumes the turn but not a stone
            if state.passes >= 2:
                break
            continue
        state.play(moves[rng.randrange(len(moves))])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="4 or 8: run only that neighbourhood")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    variants = [(False, "classic", NEIGHBOURS), (True, "blocking-4", NEIGHBOURS),
                (True, "blocking-8", NEIGHBOURS8)]
    if args.only:
        variants = [v for v in variants if args.only in v[1]]
    for blocking, label, ring in variants:
        legal_by_ply = [[] for _ in range(TOTAL_PLIES)]
        scores = []
        lengths = []
        stone_counts = []
        for _ in range(args.games):
            state = State()
            while state.ply < TOTAL_PLIES:
                moves = state.legal(blocking, ring)
                legal_by_ply[state.ply].append(len(moves))
                if not moves:
                    state.passes += 1
                    state.ply += 1
                    if state.passes >= 2:
                        break
                    continue
                state.play(moves[rng.randrange(len(moves))])
            scores.append(raw_score(state.board))
            lengths.append(state.ply)
            stone_counts.append(sum(1 for v in state.board if v))

        def at(ply):
            values = legal_by_ply[ply]
            return sum(values) / len(values) if values else 0.0

        mean_legal = sum(sum(v) for v in legal_by_ply) / max(1, sum(len(v) for v in legal_by_ply))
        mid = [at(p) for p in (0, 5, 10, 20, 30, 40, 50)]
        dist = Counter(scores)
        print(f"\n=== {label} ===")
        print(f"legal moves per ply: mean {mean_legal:.1f}   "
              f"at ply " + " ".join(f"{p}:{v:.0f}" for p, v in zip((0, 5, 10, 20, 30, 40, 50), mid)))
        print(f"game length: mean {sum(lengths)/len(lengths):.1f}  "
              f"min {min(lengths)}  max {max(lengths)}   "
              f"stones on board mean {sum(stone_counts)/len(stone_counts):.1f}")
        print(f"raw score: mean {sum(scores)/len(scores):+.2f}  "
              f"min {min(scores):+d}  max {max(scores):+d}")
        draws = sum(1 for s in scores if s == 0)
        red = sum(1 for s in scores if s > 0)
        print(f"raw==0 (would be a draw at komi 0): {draws/len(scores):.1%}   "
              f"red wins at komi 0: {red/len(scores):.1%}")
        for komi in (0.0, -0.5, -1.0, -2.0):
            d = sum(1 for s in scores if s + komi == 0)
            r = sum(1 for s in scores if s + komi > 0)
            print(f"  komi {komi:+.1f}: red {r/len(scores):6.1%}  draw {d/len(scores):6.2%}")
        top = dist.most_common(5)
        print(f"most common scores: {top}")


if __name__ == "__main__":
    main()
