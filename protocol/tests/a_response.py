#!/usr/bin/env python3
"""Manual probe: does the loaded network respond to the A value?

Not a gated test, because it needs a *trained* v4 model (the repository does not
ship one) and a random-init model has no reason to be monotone in A.  Run it
after training:

    python protocol/tests/a_response.py build/shuyi_engine.exe runs/x/gen_005/candidate.bin
    python protocol/tests/a_response.py build/shuyi_engine.exe old_v3_model.bin

A v4 model must move win rate monotonically with A; a v3 model must not move at
all (`a_input=0`).  Both are correct answers -- the wrong outcome is a v3 file
that pretends to respond, or a v4 file that does not.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_client import EngineProcess  # noqa: E402

A_VALUES = [-6.0, -2.0, 0.0, 1.26, 4.0, 8.0]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    exe, model = argv

    engine = EngineProcess(exe)
    for command in ["boardsize 9", "ruleset blocking", f"set_model {model}"]:
        reply = engine.command(command)
        if not reply.ok:
            print(f"{command}: {reply.text}", file=sys.stderr)
            return 1
    # A midgame position: build it by letting the engine play itself and telling
    # the position the moves, exactly as a front end does.
    for ply in range(12):
        side = "red" if ply % 2 == 0 else "black"
        reply = engine.command(f"genmove {side} 64")
        if not reply.ok:
            print(f"genmove {side}: {reply.text}", file=sys.stderr)
            return 1
        cell, number = reply.text.split()
        if not engine.command(f"play {side} {cell} {number}").ok:
            return 1

    winrates = []
    for a in A_VALUES:
        if not engine.command(f"set_a {a}").ok:
            return 1
        reply = engine.command("analyze red 512")
        if not reply.ok:
            print(f"analyze: {reply.text}", file=sys.stderr)
            return 1
        last = reply.infos[-1] if reply.infos else {}
        winrates.append(last.get("winrate", 0.0))
        print(f"A={a:+.2f} winrate={last.get('winrate')} score={last.get('score')} "
              f"visits={last.get('visits')} a_input={last.get('a_input')} "
              f"best={last.get('best')}")
    engine.command("quit")

    a_input = last.get("a_input", 0) if winrates else 0
    constant = max(winrates) - min(winrates) < 1e-4
    if a_input == 1 and constant:
        print("FAIL: model reports a_input=1 but the win rate ignores A", file=sys.stderr)
        return 1
    if a_input == 0 and not constant:
        print("FAIL: model reports a_input=0 but the win rate moved with A", file=sys.stderr)
        return 1
    print("OK: " + ("A is an input" if a_input == 1 else "A-blind (v3), as reported"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
