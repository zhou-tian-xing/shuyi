#!/usr/bin/env python3
"""One-ply value-scan policy targets.
DEPRECATED: PyTorch reference implementation of the one-ply value scan.
The pipeline now uses `shuyi_selfplay_gpu --mode oneply`. Kept for
algorithm cross-checks only.

Why this exists
---------------
MCTS with 64 simulations over ~360 legal moves cannot produce a learnable
policy target: whichever way the candidate set is chosen (noisy prior, or the
prior's top-k when the prior itself is flat), the *support* of the target is
not a smooth function of the position.  A convolutional net then learns the
"which number" part and nothing about "which cell".

This script isolates that claim.  Instead of MCTS it evaluates every legal move
with the value net (one ply) and turns the values into a policy target,

    p(a) ~ exp((Q(a) - max Q) / T),   Q(a) = 1 - V(position after a)

which is a deterministic, position-dependent function -- exactly what a policy
head can learn.  Only used to demonstrate that the training pipeline and the
37-plane features are fine; it is not the final training recipe.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_selfplay as G  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_resnet", str(HERE.parent / "tools" / "train_resnet.py"))
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


@torch.inference_mode()
def batch_values(net, states, device):
    out = np.zeros(len(states), dtype=np.float32)
    for start in range(0, len(states), 1024):
        chunk = states[start : start + 1024]
        x, g = G.build_inputs(chunk, net.planes, device)
        _logits, value = net.model(x, g)
        out[start : start + 1024] = value.float().cpu().numpy().reshape(-1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--komi", type=float, default=-5.0)
    ap.add_argument("--out", default="data/oneply.bin")
    args = ap.parse_args(argv)

    G.KOMI = float(args.komi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = G.Net(args.model, device)
    rng = np.random.default_rng(args.seed)

    states = []
    for _ in range(args.games):
        state = G.State.start()
        for _ in range(args.opening):
            legal = state.legal()
            state = state.play(int(legal[rng.integers(legal.size)]))
        states.append(state)

    samples = []
    started = time.time()
    for move_no in range(G.TOTAL_MOVES - args.opening):
        # Enumerate every legal move of every game.
        flat = []
        owner = []
        for g, state in enumerate(states):
            for move in state.legal():
                flat.append(state.play(int(move)))
                owner.append((g, int(move)))
        values = batch_values(net, flat, device)
        offsets = []
        cursor = 0
        for state in states:
            n = int(state.legal().size)
            offsets.append((cursor, cursor + n))
            cursor += n
        for g, state in enumerate(states):
            begin, end = offsets[g]
            q = 1.0 - values[begin:end]  # value at the child is the opponent's win prob
            p = np.exp((q - q.max()) / max(1e-4, args.temperature))
            p /= p.sum()
            target = np.zeros(G.POLICY, dtype=np.float32)
            legal = state.legal()
            target[legal] = p
            samples.append(
                (state.side, state.board.copy(), state.rem.copy(), target, np.float32(0.0), state.ply, g)
            )
            pick = int(legal[rng.choice(len(p), p=p)]) if move_no < 12 else int(legal[int(np.argmax(p))])
            states[g] = state.play(pick)
        if states[0].ply >= G.TOTAL_MOVES:
            break
        if (move_no + 1) % 10 == 0:
            print(f"ply {states[0].ply} {time.time() - started:.0f}s", flush=True)

    results = np.zeros(args.games, dtype=np.int8)
    for g, state in enumerate(states):
        total = state.raw + G.KOMI
        results[g] = 1 if total > 0 else (-1 if total < 0 else 0)
    fixed = []
    for side, board, rem, target, _v, ply, g in samples:
        red = 1.0 if results[g] > 0 else (0.0 if results[g] < 0 else 0.5)
        value = red if side == 0 else 1.0 - red
        fixed.append((side, board, rem, target, np.float32(value), ply, g))
    count = G.write_shard(args.out, fixed)
    print(f"wrote {count} samples to {args.out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
