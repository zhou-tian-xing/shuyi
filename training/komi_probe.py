import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "training")
import verify_rules as V  # noqa: E402
from gpu_selfplay import State  # noqa: E402

d = V.load(sys.argv[1], 20000)
cnt = Counter()
examples = []
for game in np.unique(d["game"]):
    idx = np.flatnonzero(d["game"] == game)
    idx = idx[np.argsort(d["ply"][idx])]
    rows = d[idx]
    board = rows["board"][0].copy()
    state = State.start()
    for sign in (1, -1):
        for cell in np.flatnonzero(np.sign(board) == sign):
            state = state.play(int(cell) * 9 + int(abs(board[cell])) - 1)
    for row in rows:
        fol = rows[rows["ply"] == row["ply"] + 1]
        if fol.size == 0:
            continue
        diff = np.flatnonzero(fol["board"][0] != row["board"])
        c = int(diff[0])
        state = state.play(c * 9 + int(abs(fol["board"][0][c])) - 1)
    stored = float(rows["value"][0])
    finals = [state] + [state.play(int(i)) for i in state.legal()]
    fit = None
    for k in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        for f in finals:
            t = f.raw + k
            red = 1.0 if t > 0 else (0.0 if t < 0 else 0.5)
            e = red if f.side == 0 else 1.0 - red
            if abs(e - stored) < 1e-6:
                fit = k
                break
        if fit is not None:
            break
    cnt[fit] += 1
    if fit != 1.0 and len(examples) < 6:
        examples.append((int(game), int(state.raw), stored, fit, int(state.side)))
print(cnt)
for e in examples:
    print(e)
