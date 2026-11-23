#!/usr/bin/env python3
"""Smoke test for the asynchronous GTP-like engine process."""
import os
import queue
import subprocess
import sys
import threading
import time


def main():
    if len(sys.argv) < 2:
        print("usage: protocol_smoke.py <shuyi_engine.exe>")
        return 2
    engine = os.path.abspath(sys.argv[1])
    if not os.path.exists(engine):
        print(f"missing engine: {engine}")
        return 2

    process = subprocess.Popen(
        [engine],
        cwd=os.path.dirname(engine),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    lines = queue.Queue()

    def reader():
        for line in process.stdout:
            lines.put(line.rstrip("\n"))

    threading.Thread(target=reader, daemon=True).start()

    def send(command):
        process.stdin.write(command + "\n")
        process.stdin.flush()

    def wait_for(prefix, timeout):
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            try:
                line = lines.get(timeout=min(0.1, deadline - time.time()))
            except queue.Empty:
                continue
            seen.append(line)
            if line.startswith(prefix):
                return line, seen
        raise TimeoutError(f"did not see {prefix!r}; saw {seen!r}")

    try:
        send("protocol_version")
        line, _ = wait_for("= ", 2.0)
        assert line == "= 1", line

        send("name")
        line, _ = wait_for("= ", 2.0)
        assert line == "= ShuyiEngine", line

        send("set_time 60")
        wait_for("=", 2.0)
        send("set_engine mcts")
        wait_for("= mcts", 2.0)
        send("clear_board")
        wait_for("=", 2.0)
        send("genmove red")
        # The first response accepts the asynchronous command; the final
        # response is a move, not the empty acknowledgement.
        wait_for("=", 2.0)
        final = None
        deadline = time.time() + 4.0
        while time.time() < deadline:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line.startswith("= ") and len(line.split()) == 3:
                final = line
                break
        assert final is not None, "no final genmove response"
        parts = final.split()
        assert len(parts) == 3 and parts[1][0] in "abcdefghi", final
        assert 1 <= int(parts[2]) <= 9, final
        send("quit")
        process.wait(timeout=3)
        print(f"protocol smoke passed: {final}")
        return 0
    except Exception as error:  # noqa: BLE001 - smoke test should print context
        print(f"protocol smoke failed: {error}")
        return 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


if __name__ == "__main__":
    sys.exit(main())
