#!/usr/bin/env python3
"""Conformance test for protocol/ENGINE_PROTOCOL.md.

    python protocol/tests/protocol_smoke.py <shuyi_engine> [model.bin]

It only uses the wire protocol, so it also validates an engine written by
someone else.  A failure here means a front end cannot rely on the contract.
"""
import sys
import time
from pathlib import Path

from engine_client import EngineProcess

REPO = Path(__file__).resolve().parents[2]
# A classic-rule network: the classic/blocking mismatch is part of the contract.
MODEL = REPO / "models" / "shuyi_resnet_v1.bin"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1])
    # argv[2] is accepted for symmetry with the other test but a *classic* model
    # is required here, so the mismatch check is meaningful.
    model = MODEL
    if len(argv) > 2:
        print(f"note: using {model.name} (classic) instead of {argv[2]}")

    with EngineProcess(binary, cwd=str(REPO)) as engine:
        reply = engine.command("protocol_version")
        check(reply.ok and reply.text == "2", f"protocol_version: {reply}")

        reply = engine.command("capabilities")
        for key in ("engine=shuyi_engine", "proto=2", "cuda=", "visits=1", "stream=1", "json=1"):
            check(key in reply.text, f"capabilities missing {key}: {reply.text}")

        # A wrong board size is an error, never a silent 9x9.
        reply = engine.command("boardsize 19")
        check(not reply.ok, "boardsize 19 must fail")
        check(engine.command("boardsize 9").ok, "boardsize 9")

        # Commands that cannot be executed must say so.
        check(not engine.command("frobnicate").ok, "unknown command must fail")
        check(not engine.command("set_backend quantum").ok, "bad backend must fail")

        # A model trained for the other rule set is refused before it is used.
        engine.command("ruleset blocking")
        reply = engine.command(f"set_model {model}")
        check(not reply.ok and "ruleset" in reply.text, f"classic model on blocking: {reply}")
        engine.command("ruleset classic")
        check(engine.command(f"set_model {model}").ok, "set_model classic")
        check(engine.command("set_engine mcts_nn").ok, "set_engine mcts_nn")

        # Streaming analysis: info lines, then exactly one final reply.
        reply = engine.command("analyze red 128 interval 20")
        check(reply.ok, f"analyze: {reply}")
        check(reply.text.startswith("done visits=128"), f"budget not spent: {reply.text}")
        check(len(reply.infos) >= 2, f"streaming produced {len(reply.infos)} info lines")
        last = reply.infos[-1]
        for key in ("visits", "nodes", "winrate", "best", "cells", "moves", "engine", "turn"):
            check(key in last, f"info missing {key}: {last}")
        check(len(last["cells"]) > 1, "analysis should mark more than one cell")
        check(0.0 <= last["winrate"] <= 1.0, last["winrate"])

        # The tree is inherited: a second analysis of the same position starts
        # from the visits the first one made.
        first_visits = last["visits"]
        reply = engine.command("analyze red 64")
        check(reply.ok and reply.infos, "second analyze")
        check(reply.infos[-1]["visits"] >= first_visits + 50,
              f"tree not inherited: {reply.infos[-1]['visits']} after {first_visits}")

        # Unlimited analysis runs until `stop` and answers with `stopped`.
        # `analyze` is fire-and-forget: its final reply only arrives when the
        # search ends, so it must not be awaited before sending `stop`.
        running = engine.send("analyze red 0 interval 20")
        time.sleep(0.3)
        reply = engine.command("stop")
        check(reply.ok, f"stop: {reply}")
        check(engine.wait_reply(running).text.startswith("stopped"), "stop must end the stream")

        # genmove returns a legal move for the side to move.
        reply = engine.command("genmove red 64")
        check(reply.ok and len(reply.text.split()) == 2, f"genmove: {reply.text}")
        cell, number = reply.text.split()
        check(cell[0] in "abcdefghi" and 1 <= int(number) <= 9, reply.text)
        check(engine.command(f"play red {cell} {number}").ok, "play the engine move")
        check(not engine.command("play red a1 1").ok, "wrong side to move must fail")

        # bench is part of the engine product, not of a separate tool.
        reply = engine.command("bench 64")
        check(reply.ok and "visits_per_second=" in reply.text, f"bench: {reply.text}")

        print("protocol smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
