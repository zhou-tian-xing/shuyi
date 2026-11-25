"""Build a legal nopass position with six plies left and dump one analysis line.

Used by hand: it writes cmds.txt, the engine reads it via stdin and its output
lands in out.txt, so no pipes are needed.
"""
import random
import sys

PLIES_LEFT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261124
random.seed(SEED)
cells = [(x, y) for y in range(9) for x in range(9)]


def idx(x, y):
    return y * 9 + x


def name(cell):
    x, y = cell
    return "abcdefghi"[x] + str(y + 1)


def neighbours(cell):
    x, y = cell
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < 9 and 0 <= ny < 9:
            out.append((nx, ny))
    return out


board = {}
hand = {0: [3] * 9, 1: [3] * 9}


def legal(side, cell):
    if cell in board:
        return []
    out = []
    for number in range(1, 10):
        if hand[side][number - 1] <= 0:
            continue
        blocked = False
        for nb in neighbours(cell):
            stone = board.get(nb)
            if stone and stone[0] != side and stone[1] >= number:
                blocked = True
                break
        if not blocked:
            out.append(number)
    return out


lines = ["protocol_version", "ruleset nopass", "clear_board", "set_engine mcts",
         "set_a 0.5"]
side = 0
for ply in range(54 - PLIES_LEFT):
    options = [(c, n) for c in cells for n in legal(side, c)]
    if not options:
        raise SystemExit(f"no legal move at ply {ply}")
    # Prefer moves that keep the opponent alive as well: a starvation ending would
    # end the game before the six-ply window this script is here to reach.
    random.shuffle(options)
    chosen = None
    for cell, number in options:
        board[cell] = (side, number)
        hand[side][number - 1] -= 1
        alive = any(legal(1 - side, c) for c in cells)
        if alive:
            chosen = (cell, number)
            break
        hand[side][number - 1] += 1
        del board[cell]
    if chosen is None:
        raise SystemExit(f"every move starves the opponent at ply {ply}")
    cell, number = chosen
    lines.append(f"play {'red' if side == 0 else 'black'} {name(cell)} {number}")
    side ^= 1

if not any(legal(side, c) for c in cells):
    raise SystemExit(f"side to move has no legal placement at {len(board)} stones (seed {SEED})")
lines.append(f"analyze {'red' if side == 0 else 'black'} 512 interval 50")
lines.append("quit")
open("cmds.txt", "w").write("\n".join(lines) + "\n")
print("plies", 54 - PLIES_LEFT, "stones", len(board), "left", PLIES_LEFT)
