#!/usr/bin/env python3
"""Diagnose the information content of a self-play shard.

Reports, per position: number of legal moves, policy-target entropy (nats),
max target probability, top-5 mass, and the value-target distribution.  A target
distribution with entropy close to ln(#legal) means the search added nothing
over a uniform prior -- that data cannot teach a policy.
"""
import argparse
import math
import struct

import numpy as np

CELLS = 81
POLICY = 729
V2_DTYPE = np.dtype(
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


V4_DTYPE = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (CELLS,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (POLICY,)),
        ("value", "<f4"),
        ("game", "<u4"),
        ("ply", "u1"),
        ("score", "i1"),
    ]
)


def load(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        version, count = struct.unpack("<II", f.read(8))
        ruleset = 0
        dtype = V2_DTYPE
        if magic in (b"SYSELF3\x00", b"SYSELF4\x00"):
            ruleset = struct.unpack("<B", f.read(1))[0]
            f.read(3)
        if magic == b"SYSELF4\x00":
            dtype = V4_DTYPE
        data = np.fromfile(f, dtype=dtype, count=count)
    return data, ruleset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--limit", type=int, default=200000)
    args = ap.parse_args()
    data, ruleset = load(args.shard)
    d = data[: args.limit]
    n = len(d)
    boards = d["board"].astype(np.int16).reshape(n, 81)
    hands = d["hand"].reshape(n, 2, 9).astype(np.int16)
    sides = d["side"].astype(np.int64)
    remaining = hands[np.arange(n), sides]
    legal = (boards == 0)[:, :, None] & (remaining[:, None, :] > 0)
    if ruleset == 1:
        # Blocking rule: number n is illegal on a cell with an adjacent enemy
        # stone of number >= n.  Same four-neighbour reach the engine uses.
        board3 = boards.reshape(n, 9, 9)
        enemy_max = np.zeros((n, 81), dtype=np.int16)
        for side, sign in ((0, 1), (1, -1)):
            layer = np.where(board3 * sign > 0, np.abs(board3), 0)
            padded = np.pad(layer, ((0, 0), (1, 1), (1, 1)), mode="constant")
            reach = np.maximum(
                np.maximum(padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]),
                np.maximum(padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]),
            ).reshape(n, 81)
            enemy_max = np.where(sides[:, None] == side, np.maximum(enemy_max, reach), enemy_max)
        numbers = np.arange(1, 10)[None, None, :]
        legal = legal & (numbers > enemy_max[:, :, None])
    legal_flat = legal.reshape(n, 729)
    nmoves = legal_flat.sum(1)

    p = d["policy"].astype(np.float64)
    p = np.clip(p, 0, None)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    nz = (p > 0).sum(1)
    ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0.0)).sum(1)
    maxp = p.max(1)
    top5 = -np.sort(-p, axis=1)[:, :5].sum(1)
    cover = legal_flat.sum(0)  # how often each (cell,number) is legal
    # visit mass concentrated on the top-k moves only
    top8 = -np.sort(-p, axis=1)[:, :8].sum(1)

    v = d["value"]
    print(f"ruleset              {ruleset} (0 classic, 1 blocking)")
    print(f"positions            {n}")
    print(f"games                {len(np.unique(d['game']))}")
    print(f"legal moves mean     {nmoves.mean():.1f}  min {nmoves.min()} max {nmoves.max()}")
    safe_moves = np.maximum(nmoves, 1)
    print(f"uniform entropy      {np.log(safe_moves).mean():.3f} nats")
    print(f"target entropy       {ent.mean():.3f} nats  (p10 {np.percentile(ent,10):.2f} p90 {np.percentile(ent,90):.2f})")
    print(f"target max prob      {maxp.mean():.4f}  (uniform would be {1.0/nmoves.mean():.4f})")
    print(f"target top5 mass     {top5.mean():.3f}")
    print(f"target top8 mass     {top8.mean():.3f}")
    print(f"nonzero entries      {nz.mean():.1f} of {nmoves.mean():.1f} legal")
    print(f"nonzero / legal      {(nz/np.maximum(nmoves,1)).mean():.3f}")
    print(f"value mean/std       {v.mean():.4f} / {v.std():.4f}")
    print(f"value==0.5 fraction  {(v==0.5).mean():.4f}")
    # Effective number of moves the target distinguishes
    print(f"effective moves      {np.exp(ent).mean():.1f}")
    print(f"ply range            {d['ply'].min()} .. {d['ply'].max()}")


if __name__ == "__main__":
    main()
