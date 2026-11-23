#!/usr/bin/env python3
"""Verify that the C++ feature planes match the trainer's feature planes.

`shuyi_selfplay_gpu --check --dump-features build/layout_check.bin` writes the
planes the engine actually feeds the network.  This script rebuilds the same
planes with `training/train_resnet.py::make_inputs` (the code that produced the
training tensors) and compares them.

The layout bug this exists for: the side-to-move plane used to be written at
index 36, on top of Black's "9 stones still in hand" plane, while index 18 sat
unused.  Both the engine and the trainer had the same bug, so nothing failed -
the plane count stayed at 37 and the model just silently lost one input
feature.  A dump-based comparison is the only way to catch that class of bug.
"""
import argparse
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

_spec = importlib.util.spec_from_file_location("train_resnet", str(REPO / "experimental" / "training" / "train_resnet.py"))
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


def load_dump(path):
    with open(path, "rb") as f:
        count, planes = struct.unpack("<II", f.read(8))
        out = []
        for _ in range(count):
            ply = struct.unpack("<B", f.read(1))[0]
            board = np.frombuffer(f.read(81), dtype=np.int8).copy()
            hand = np.frombuffer(f.read(18), dtype=np.uint8).copy()
            features = np.frombuffer(f.read(planes * 81 * 4), dtype="<f4").copy()
            out.append((ply, board, hand, features.reshape(planes, 9, 9)))
    return planes, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default=str(REPO / "build" / "layout_check.bin"))
    ap.add_argument("--planes", type=int, default=0, help="expected plane count (0 = from dump)")
    args = ap.parse_args()

    planes, samples = load_dump(args.dump)
    if args.planes and args.planes != planes:
        raise SystemExit(f"dump has {planes} planes, expected {args.planes}")
    if planes not in (_tr.INPUT_PLANES, _tr.INPUT_PLANES_V3):
        raise SystemExit(f"unexpected plane count {planes}")

    failures = 0
    for index, (ply, board, hand, features) in enumerate(samples):
        data = np.zeros(1, dtype=_tr.SAMPLE_DTYPE) if hasattr(_tr, "SAMPLE_DTYPE") else None
        if data is None:
            data = np.zeros(
                1,
                dtype=np.dtype(
                    [
                        ("side", "u1"),
                        ("board", "i1", (81,)),
                        ("hand", "u1", (18,)),
                        ("policy", "<f4", (729,)),
                        ("value", "<f4"),
                        ("game", "<u4"),
                        ("ply", "u1"),
                    ]
                ),
            )
        data["side"] = ply & 1
        data["board"] = board
        data["hand"] = hand
        data["ply"] = ply
        x, _g = _tr.make_inputs(data, planes)
        delta = np.abs(x[0] - features).max()
        if delta > 1e-6:
            failures += 1
            print(f"position {index}: layout mismatch, |max| = {delta}")
            diff = np.abs(x[0] - features).reshape(planes, -1).max(axis=1)
            print("  planes that differ:", [i for i, d in enumerate(diff) if d > 1e-6])
    if failures:
        raise SystemExit(f"{failures}/{len(samples)} positions differ")
    print(f"layout OK: {len(samples)} positions, {planes} planes, engine == trainer")


if __name__ == "__main__":
    sys.exit(main())
