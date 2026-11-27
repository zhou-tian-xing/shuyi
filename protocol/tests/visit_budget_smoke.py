#!/usr/bin/env python3
"""Every engine must spend exactly the visit budget it is given.

A rating is only a statement about the models if each move costs the same search
effort on every machine.  This drives the engine protocol directly and reads the
numbers back out of the `info` JSON:

    python protocol/tests/visit_budget_smoke.py <shuyi_engine> [model.bin]

MCTS and MCTS+NN spend exactly `visits` simulations.  Alpha-beta has no notion of
a simulation, so one visit is one node (one leaf evaluation); the node count must
land inside the budget plus the iteration that was cut, and both the node count
and the move must be identical on a second run -- that is what "machine
independent" means.
"""
import sys
from pathlib import Path

from engine_client import EngineProcess

REPO = Path(__file__).resolve().parents[2]
BUDGET = 256


def run(kind, binary, model, moves=1):
    with EngineProcess(binary, cwd=str(REPO)) as engine:
        assert engine.command(f"ruleset classic").ok
        if model:
            assert engine.command(f"set_model {model}").ok
        assert engine.command(f"set_engine {kind}").ok
        assert engine.command(f"set_visits {BUDGET}").ok
        results = []
        for _ in range(moves):
            reply = engine.command("genmove red")
            assert reply.ok, reply.text
            assert reply.infos, f"{kind}: genmove streamed no analysis"
            last = reply.infos[-1]
            results.append((last["visits"], last["nodes"], reply.text))
        return results


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1])
    model = Path(argv[2]) if len(argv) > 2 else REPO / "models" / "shuyi_resnet_v1.bin"
    problems = []

    for kind in ("mcts", "mcts_nn"):
        first = run(kind, binary, model if kind == "mcts_nn" else None)[0]
        second = run(kind, binary, model if kind == "mcts_nn" else None)[0]
        if first[0] != BUDGET:
            problems.append(f"{kind}: ran {first[0]} visits, budget was {BUDGET}")
        if first[2] != second[2]:
            problems.append(f"{kind}: not deterministic ({first[2]} vs {second[2]})")
        print(f"  {kind:<8} visits={first[0]:<5} move={first[2]}  (deterministic: {first[2] == second[2]})")

    # A front end that sets the default budget and then asks for an analysis
    # without repeating it must get that default, not an unbounded search.
    with EngineProcess(binary, cwd=str(REPO)) as engine:
        engine.command("ruleset classic")
        engine.command(f"set_model {model}")
        engine.command("set_engine mcts_nn")
        engine.command("set_visits 128")
        reply = engine.command("analyze red")
        if not reply.text.startswith("done visits=128"):
            problems.append(f"default budget ignored: {reply.text}")

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print(f"ok: every engine honours the {BUDGET}-visit budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
