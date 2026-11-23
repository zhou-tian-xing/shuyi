#!/usr/bin/env python3
"""Check whether the stored MCTS target is position-dependent or an artefact.

Hypothesis: with ~12 simulations the root visits are determined by the noisy
prior, so the target is spread uniformly over an arbitrary set of (cell,number)
pairs (e.g. the first empty cells in scan order) and carries no information.
"""
import argparse
import struct

import numpy as np

CELLS = 81
POLICY = 729
D = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (CELLS,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (POLICY,)),
        ("value", "<f4"),
        ("game", "<u4"),
        ("ply", "u1"),
    ]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()
    with open(args.shard, "rb") as f:
        f.read(8)
        _v, count = struct.unpack("<II", f.read(8))
        d = np.fromfile(f, dtype=D, count=min(count, args.limit))
    n = len(d)
    boards = d["board"].astype(np.int16).reshape(n, 81)
    hands = d["hand"].reshape(n, 2, 9)
    sides = d["side"].astype(np.int64)
    remaining = hands[np.arange(n), sides]
    p = d["policy"].astype(np.float64).reshape(n, 81, 9)
    nz = p.sum(axis=2) > 0  # cells that received visits

    # Where do the visited cells sit in scan order?
    ranks = []
    for i in range(n):
        empty = np.flatnonzero(boards[i] == 0)
        idx = {c: k for k, c in enumerate(empty)}
        visited = np.flatnonzero(nz[i])
        ranks.extend(idx.get(c, -1) for c in visited)
    ranks = np.array(ranks)
    first12 = (ranks >= 0) & (ranks < 12)
    print(f"visited cells total        {len(ranks)}")
    print(f"within first 12 empty      {first12.mean():.3f}")
    print(f"mean scan rank of visited  {ranks.mean():.1f}  (random would be {79/2:.1f})")

    # How many distinct numbers per position, and is the mass one number?
    nums_per_pos = (p.sum(axis=1) > 0).sum(axis=1)
    mass_per_num = p.sum(axis=1) / np.maximum(p.sum(axis=(1, 2), keepdims=True), 1e-12)
    top_num_mass = mass_per_num.max(axis=1)
    print(f"distinct numbers visited   {nums_per_pos.mean():.2f}")
    print(f"mass on single number      {top_num_mass.mean():.3f}")

    # How many simulations per position (sum of integer-ish counts * total?)
    nz_count = nz.sum(axis=1)
    print(f"visited cells per position {nz_count.mean():.1f} (min {nz_count.min()} max {nz_count.max()})")

    # Are the visited cells the empties with the lowest index *within their number*?
    print(f"ply of these positions     {d['ply'].mean():.1f}")


if __name__ == "__main__":
    main()
