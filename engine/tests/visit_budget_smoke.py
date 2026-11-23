#!/usr/bin/env python3
"""Check that every engine spends exactly the visit budget it is given.

A rating is only a statement about the models if each move costs the same search
effort on every machine.  This drives the engine protocol directly and reads the
visit/node counts back out of `info ...` lines:

    python tests/visit_budget_smoke.py <path to shuyi_engine> [model.bin]

For MCTS and MCTS+NN `visits` must equal the requested budget exactly.  For the
pruning engine alpha-beta has no notion of a simulation, so one visit is one
node (one leaf evaluation); the node count must land inside the budget plus the
node-cost of the one iteration that was cut, and two runs of the same position
must agree exactly (that is what "machine independent" means).
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = 256


class Engine:
    def __init__(self, binary):
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return lines
            lines.append(line.strip())
            if predicate(line.strip()):
                return lines

    def command(self, line, payload=False):
        self.send(line)
        if payload:
            return self.read_until(lambda l: l.startswith("= ") and len(l) > 2)
        self.read_until(lambda l: l.startswith("=") or l.startswith("?"))
        return []

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def info_field(lines, key):
    # The last `info` line is the final analysis; earlier ones are the
    # time-based progress reports, which say nothing about the budget.
    for line in reversed(lines):
        if not line.startswith("info"):
            continue
        for token in line.split():
            if token.startswith(key + "="):
                return token.split("=", 1)[1]
    return None


def run(kind, binary, model, ruleset="classic", moves=1):
    engine = Engine(binary)
    engine.command(f"ruleset {ruleset}")
    engine.command(f"set_engine {kind}")
    if model:
        engine.command(f"set_model {model}")
    engine.command(f"set_visits {BUDGET}")
    engine.command("set_time 3600000")
    results = []
    for _ in range(moves):
        lines = engine.command("genmove red", payload=True)
        visits = info_field(lines, "visits")
        nodes = info_field(lines, "nodes")
        move = [l for l in lines if l.startswith("= ") and len(l) > 2]
        results.append((visits, nodes, move[0] if move else None))
    engine.close()
    return results


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1])
    model = argv[2] if len(argv) > 2 else None
    problems = []

    for kind in ("mcts", "mcts_nn"):
        if kind == "mcts_nn" and not model:
            continue
        # Two fresh processes with the same command sequence: that is the
        # property a rating needs (same position, same budget, same move).
        first = run(kind, binary, model)[0]
        second = run(kind, binary, model)[0]
        visits = int(first[0]) if first[0] else -1
        if visits != BUDGET:
            problems.append(f"{kind}: ran {visits} visits, budget was {BUDGET}")
        if first[2] != second[2]:
            problems.append(f"{kind}: not deterministic ({first[2]} vs {second[2]})")
        print(f"  {kind:<8} visits={visits:<5} move={first[2]}  (deterministic: "
              f"{first[2] == second[2]})")

    first = run("pruning", binary, model)[0]
    second = run("pruning", binary, model)[0]
    nodes = int(first[1]) if first[1] else -1
    # Alpha-beta finishes the iteration it is inside, so it can overshoot; the
    # exact overshoot is deterministic, which is the property that matters.
    if not (BUDGET <= nodes <= BUDGET * 4):
        problems.append(f"pruning: {nodes} nodes for a {BUDGET}-node budget")
    if first[1] != second[1] or first[2] != second[2]:
        problems.append(f"pruning: not deterministic ({first[1]}/{first[2]} vs "
                        f"{second[1]}/{second[2]})")
    print(f"  pruning  nodes={nodes:<5} move={first[2]}  (deterministic: "
          f"{first[1] == second[1] and first[2] == second[2]})")

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print(f"ok: every engine honours the {BUDGET}-visit budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
