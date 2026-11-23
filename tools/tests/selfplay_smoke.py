#!/usr/bin/env python3
"""Validate the v2 self-play shard format and basic policy target invariants."""
import os
import struct
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) < 2:
        print("usage: selfplay_smoke.py <shuyi_selfplay>")
        return 2
    exe = os.path.abspath(sys.argv[1])
    if not os.path.exists(exe):
        print(f"missing selfplay: {exe}")
        return 2
    # Keep the shard inside the working directory: ctest runs tests in the build
    # tree, and a system temp directory may sit outside what the harness allows.
    output = os.path.abspath("shuyi_selfplay_smoke.bin")
    try:
        subprocess.run(
            [
                exe,
                "--games",
                "1",
                "--time",
                "1",
                "--engine",
                "mcts",
                "--opening",
                "2",
                "--seed",
                "7",
                "--out",
                output,
            ],
            check=True,
            cwd=os.path.dirname(exe),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        with open(output, "rb") as f:
            magic = f.read(8)
            version, count = struct.unpack("<II", f.read(8))
            assert magic == b"SYSELF2\x00", magic
            assert version == 2, version
            assert 0 < count <= 54, count
            size = 1 + 81 + 18 + 729 * 4 + 4 + 4 + 1
            payload = f.read()
            assert len(payload) == count * size, (len(payload), count * size)
            first = payload[:size]
            side = first[0]
            policy = struct.unpack_from("<729f", first, 1 + 81 + 18)
            value = struct.unpack_from("<f", first, 1 + 81 + 18 + 729 * 4)[0]
            game, ply = struct.unpack_from("<IB", first, 1 + 81 + 18 + 729 * 4 + 4)
            assert side in (0, 1), side
            assert abs(sum(policy) - 1.0) < 1e-3, sum(policy)
            assert 0.0 <= value <= 1.0, value
            assert game == 0, game
            assert 0 <= ply <= 54, ply
        print(f"selfplay smoke passed: {count} samples")
        return 0
    finally:
        if os.path.exists(output):
            os.remove(output)


if __name__ == "__main__":
    sys.exit(main())
