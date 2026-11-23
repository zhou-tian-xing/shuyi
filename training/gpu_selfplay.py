#!/usr/bin/env python3
"""GPU-batched MCTS self-play and arena for Shuyi.
DEPRECATED: PyTorch reference implementation of the batched MCTS.
It is no longer part of any pipeline; the same search now lives in
tools/selfplay/selfplay_gpu.cpp (C++ tree + batched CUDA evaluation). Kept for
algorithm cross-checks only.

Why this exists
---------------
The C++ self-play engine calls the network once per MCTS simulation, scalar and
on the CPU.  At a 10 ms budget that is about 11 simulations per move -- fewer
than the number of legal moves (up to 729) -- so the stored "policy target"
degenerates into the noisy root prior: a near-uniform spread over an arbitrary
handful of cells of a single number.  Training on that data converges to a
uniform policy and a base-rate value, which is exactly what the b10c128 run did
(policy loss 3.75 = ln(42), top-1 2.4%, value MSE 0.2595 = the variance of a
coin flip).

Here every simulation still evaluates one tree node with the network, but the
leaf evaluations of many independent games are batched into one CUDA call, so a
real search budget (hundreds of simulations per move) becomes affordable.  That
is also why "CPU inference is the bottleneck" was the wrong reading: the search
was simply never given a budget, not a machine.

Modes
-----
  selfplay:  --model M --games N --sims S --out shard.bin
  arena:     --arena-a A --arena-b B --games N --sims S
"""
import argparse
import importlib.util
import os
import struct
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("train_resnet", os.path.join(REPO, "tools", "train_resnet.py"))
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

CELLS = 81
POLICY = 729
TOTAL_MOVES = 54
KOMI = 1.0  # awarded to Red, matching EngineOptions::aValue = 1.0

_NEIGH = np.zeros((CELLS, 5), dtype=np.int32)
for _c in range(CELLS):
    _x, _y = _c % 9, _c // 9
    _pts = [(_x, _y)]
    for _dx, _dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _nx, _ny = _x + _dx, _y + _dy
        if 0 <= _nx < 9 and 0 <= _ny < 9:
            _pts.append((_nx, _ny))
    _NEIGH[_c, : len(_pts)] = [_ny * 9 + _nx for _nx, _ny in _pts]
    # Pad border cells with the cell itself, not with cell 0, so the phantom
    # entries cannot smuggle a corner stone into an unrelated cross.
    _NEIGH[_c, len(_pts):] = _c

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


def cell_scores(board):
    """sign(max red number - max black number) over each cell's cross."""
    v = board[_NEIGH]
    red = np.maximum(v, 0).max(axis=1)
    black = np.maximum(-v, 0).max(axis=1)
    return (red > black).astype(np.int16) - (red < black).astype(np.int16)


def position_score(board):
    return int(cell_scores(board).sum())


class State:
    """One position.  Pieces are never captured, so the hands are implied."""

    __slots__ = ("board", "ply", "side", "rem", "raw")

    def __init__(self, board, ply, rem, raw):
        self.board = board
        self.ply = ply
        self.side = ply & 1
        self.rem = rem
        self.raw = raw

    @staticmethod
    def start():
        return State(np.zeros(CELLS, dtype=np.int8), 0, np.full((2, 9), 3, dtype=np.int8), 0)

    def legal(self):
        empty = np.flatnonzero(self.board == 0)
        nums = np.flatnonzero(self.rem[self.side] > 0) + 1
        if empty.size == 0 or nums.size == 0:
            return np.zeros(0, dtype=np.int64)
        cells = np.repeat(empty, nums.size)
        numbers = np.tile(nums, empty.size)
        return (cells * 9 + (numbers - 1)).astype(np.int64)

    def play(self, index):
        cell = int(index) // 9
        number = int(index) % 9 + 1
        board = self.board.copy()
        rem = self.rem.copy()
        board[cell] = number if self.side == 0 else -number
        rem[self.side, number - 1] -= 1
        # Only the five cells whose cross contains `cell` can change score.
        cross = _NEIGH[_NEIGH[cell]]  # [5,5]
        before = self.board[cross]
        after = board[cross]
        red = np.maximum(after, 0).max(axis=1)
        black = np.maximum(-after, 0).max(axis=1)
        delta = int((red > black).astype(np.int16).sum()) - int((red < black).astype(np.int16).sum())
        red = np.maximum(before, 0).max(axis=1)
        black = np.maximum(-before, 0).max(axis=1)
        delta -= int((red > black).astype(np.int16).sum()) - int((red < black).astype(np.int16).sum())
        return State(board, self.ply + 1, rem, self.raw + delta)

    def final_value(self):
        """1 win / 0 loss / 0.5 draw, for the side to move."""
        total = self.raw + KOMI
        if total > 0:
            return 1.0 if self.side == 0 else 0.0
        if total < 0:
            return 1.0 if self.side == 1 else 0.0
        return 0.5


class Tree:
    __slots__ = ("states", "P", "N", "W", "child", "legal", "Nsum", "ply")

    def __init__(self, state):
        self.states = [state]
        self.P = [None]
        self.N = [None]
        self.W = [None]
        self.child = [None]
        self.legal = [None]
        self.Nsum = [0]
        self.ply = [state.ply]

    def new_node(self, parent, edge):
        state = self.states[parent].play(int(self.legal[parent][edge]))
        node = len(self.states)
        self.states.append(state)
        self.P.append(None)
        self.N.append(None)
        self.W.append(None)
        self.child.append(None)
        self.legal.append(None)
        self.Nsum.append(0)
        self.ply.append(state.ply)
        return node


def build_inputs(states, planes, device):
    n = len(states)
    boards = torch.from_numpy(np.stack([s.board for s in states])).to(device, non_blocking=True)
    rems = torch.from_numpy(np.stack([s.rem for s in states])).to(device, non_blocking=True)
    sides = torch.tensor([s.side for s in states], dtype=torch.float32, device=device).unsqueeze(1)
    raws = torch.tensor([s.raw for s in states], dtype=torch.float32, device=device).unsqueeze(1)
    plys = torch.tensor([s.ply for s in states], dtype=torch.float32, device=device).unsqueeze(1)

    b = boards.to(torch.int64)
    # class 0..8 red 1..9, 9..17 black 1..9, 18 empty; one_hot is a single kernel.
    idx = torch.where(b > 0, b - 1, torch.where(b < 0, 9 - b - 1, torch.full_like(b, 18)))
    stone = torch.nn.functional.one_hot(idx, num_classes=19)[:, :, :18]
    stone = stone.permute(0, 2, 1).to(torch.float32)
    hand = (rems.to(torch.float32).reshape(n, 18, 1) / 3.0).expand(n, 18, CELLS)
    side_plane = sides.unsqueeze(2).expand(n, 1, CELLS)
    if planes == _tr.INPUT_PLANES_V3:
        # Must match training/train_resnet.py::make_inputs exactly: stones 0..17,
        # side to move at 18, hand counts at 19..36.
        pad = torch.zeros(n, 1, CELLS, dtype=torch.float32, device=device)
        x = torch.cat([stone, pad, hand], dim=1)
        x[:, 18:19] = side_plane
    else:
        x = torch.cat([stone, side_plane], dim=1)

    g = torch.cat(
        [
            (rems.to(torch.float32) / 3.0).reshape(n, 18),
            sides,
            raws / 81.0,
            plys / 54.0,
            (b == 0).sum(dim=1, keepdim=True).to(torch.float32) / 81.0,
        ],
        dim=1,
    )
    return x.reshape(n, planes, 9, 9), g


class Net:
    def __init__(self, path, device):
        with open(path, "rb") as handle:
            handle.read(8)
            version, channels, blocks, planes, globals_ = struct.unpack("<IIIII", handle.read(20))
        if version != 2 or globals_ != _tr.GLOBAL:
            raise ValueError(f"unsupported model {path}")
        self.planes = planes
        self.channels = channels
        self.blocks = blocks
        self.path = path
        self.device = device
        self.model = _tr.ResNet(channels, blocks, planes).to(device)
        _tr.load_bin(self.model, path)
        self.model.eval()

    @torch.inference_mode()
    def evaluate(self, states):
        x, g = build_inputs(states, self.planes, self.device)
        logits, value = self.model(x, g)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()


def descend(tree, c_puct):
    node = 0
    path = []
    while True:
        if tree.ply[node] >= TOTAL_MOVES:
            return node, path
        priors = tree.P[node]
        if priors is None or priors.size == 0:
            return node, path
        visits = tree.N[node]
        q = tree.W[node] / np.maximum(visits, 1.0)
        u = c_puct * priors * np.sqrt(tree.Nsum[node]) / (1.0 + visits)
        score = np.where(visits > 0, q + u, 1e6 + priors)
        edge = int(np.argmax(score))
        path.append((node, edge))
        child = int(tree.child[node][edge])
        if child < 0:
            return tree.new_node(node, edge), path
        node = child


def expand(tree, node, logits, top_k, noise, rng):
    state = tree.states[node]
    legal = state.legal()
    if legal.size == 0:
        tree.P[node] = np.zeros(0, dtype=np.float32)
        tree.N[node] = np.zeros(0, dtype=np.float32)
        tree.W[node] = np.zeros(0, dtype=np.float32)
        tree.child[node] = np.zeros(0, dtype=np.int32)
        tree.legal[node] = legal
        return
    p = np.exp(logits[legal].astype(np.float64) - logits[legal].max())
    p /= p.sum()
    if noise > 0.0 and p.size > 1:
        # Noise before truncation: otherwise the kept candidates are just the
        # first cells in scan order, because the prior is flat inside a number.
        gamma = rng.gamma(0.3, 1.0, size=p.size)
        gamma /= gamma.sum()
        p = (1.0 - noise) * p + noise * gamma
    if top_k < legal.size:
        keep = np.zeros(legal.size, dtype=bool)
        keep[np.argsort(-p)[:top_k]] = True
        p = np.where(keep, p, 0.0)
        p /= p.sum()
        legal = legal[keep]
        p = p[keep]
    tree.P[node] = p.astype(np.float32)
    tree.N[node] = np.zeros(p.size, dtype=np.float32)
    tree.W[node] = np.zeros(p.size, dtype=np.float32)
    tree.child[node] = np.full(p.size, -1, dtype=np.int32)
    tree.legal[node] = legal


def backup(tree, path, value):
    v = float(value)
    visited = set()
    for node, edge in reversed(path):
        v = 1.0 - v
        tree.N[node][edge] += 1.0
        tree.W[node][edge] += v
        visited.add(node)
    for node in visited:
        tree.Nsum[node] += 1


def run_move(nets, trees, sims, c_puct, top_k, root_k, noise, rngs):
    for _ in range(sims):
        leaves = [descend(tree, c_puct) for tree in trees]
        groups = {}
        for g, (node, _path) in enumerate(leaves):
            if trees[g].ply[node] < TOTAL_MOVES:
                groups.setdefault(id(nets[g]), []).append(g)
        values = [None] * len(trees)
        for game_ids in groups.values():
            net = nets[game_ids[0]]
            logits, value = net.evaluate([trees[g].states[leaves[g][0]] for g in game_ids])
            for k, g in enumerate(game_ids):
                values[g] = (logits[k], float(value[k]))
        for g, tree in enumerate(trees):
            node, path = leaves[g]
            if tree.ply[node] >= TOTAL_MOVES:
                backup(tree, path, tree.states[node].final_value())
            else:
                logits, value = values[g]
                expand(tree, node, logits, root_k if node == 0 else top_k,
                        noise if node == 0 else 0.0, rngs[g])
                backup(tree, path, value)


def policy_target(tree):
    target = np.zeros(POLICY, dtype=np.float32)
    visits = tree.N[0]
    if visits is None or visits.size == 0:
        return target
    total = visits.sum()
    if total <= 0:
        target[tree.legal[0]] = 1.0 / tree.legal[0].size
    else:
        target[tree.legal[0]] = visits / total
    return target


def pick_move(tree, temperature, greedy, rng):
    visits = tree.N[0]
    legal = tree.legal[0]
    if visits is None or visits.size == 0:
        return None
    if greedy or temperature <= 1e-3:
        return int(legal[int(np.argmax(visits))])
    weights = np.power(visits, 1.0 / max(temperature, 1e-3))
    total = weights.sum()
    if total <= 0:
        return int(legal[int(np.argmax(visits))])
    return int(legal[int(rng.choice(visits.size, p=weights / total))])


def play_batch(nets, n_games, sims, c_puct, top_k, root_k, noise,
               temp_moves, temperature, opening, seed, game_offset):
    rng = np.random.default_rng(seed)
    rngs = [np.random.default_rng(seed + 7919 * (i + 1)) for i in range(n_games)]
    trees = []
    for _ in range(n_games):
        state = State.start()
        for _ in range(opening):
            legal = state.legal()
            if legal.size == 0:
                break
            state = state.play(int(legal[rng.integers(legal.size)]))
        trees.append(Tree(state))

    records = []  # (game_index, side, board, rem, target, ply)
    for move_no in range(TOTAL_MOVES - opening):
        run_move(nets, trees, sims, c_puct, top_k, root_k, noise, rngs)
        for g, tree in enumerate(trees):
            state = tree.states[0]
            records.append(
                (g, state.side, state.board.copy(), state.rem.copy(), policy_target(tree), state.ply)
            )
            index = pick_move(tree, temperature, move_no >= temp_moves, rngs[g])
            if index is None:
                index = int(state.legal()[0]) if state.legal().size else 0
            trees[g] = Tree(state.play(index))
        if trees[0].states[0].ply >= TOTAL_MOVES:
            break

    results = np.zeros(n_games, dtype=np.int8)
    raws = np.zeros(n_games, dtype=np.int16)
    for g, tree in enumerate(trees):
        total = tree.states[0].raw + KOMI
        results[g] = 1 if total > 0 else (-1 if total < 0 else 0)
        raws[g] = tree.states[0].raw

    samples = []
    for g, side, board, rem, target, ply in records:
        # results is +1 red win, -1 black win, 0 draw; the stored value is the
        # probability that the player to move wins.
        red = 1.0 if results[g] > 0 else (0.0 if results[g] < 0 else 0.5)
        value = red if side == 0 else 1.0 - red
        samples.append((side, board, rem, target, np.float32(value), ply, game_offset + g))
    return samples, results, raws


def write_shard(path, samples):
    data = np.zeros(len(samples), dtype=V2_DTYPE)
    for i, (side, board, rem, target, value, ply, game) in enumerate(samples):
        data["side"][i] = side
        data["board"][i] = board
        data["hand"][i] = rem.reshape(18)
        data["policy"][i] = target
        data["value"][i] = value
        data["ply"][i] = ply
        data["game"][i] = game
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"SYSELF2\x00")
        handle.write(struct.pack("<II", 2, len(data)))
        data.tofile(handle)
    return len(data)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--arena-a", default="")
    ap.add_argument("--arena-b", default="")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--root-k", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--opening", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--out", default="data/selfplay.bin")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--komi", type=float, default=1.0, help="points awarded to Red")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    global KOMI
    KOMI = float(args.komi)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    if args.arena_a:
        net_a = Net(args.arena_a, device)
        net_b = Net(args.arena_b, device)
        wins = draws = losses = 0
        started = time.time()
        done = 0
        while done < args.games:
            batch = min(args.games - done, 64)
            nets = [net_a if (done + i) % 2 == 0 else net_b for i in range(batch)]
            _samples, results, _raws = play_batch(
                nets, batch, args.sims, args.c_puct, args.top_k, args.root_k,
                args.noise, args.temp_moves, 1.0, args.opening, args.seed + done, done,
            )
            for i in range(batch):
                a_red = (done + i) % 2 == 0
                red_score = int(results[i])
                if red_score == 0:
                    draws += 1
                elif (red_score == 1) == a_red:
                    wins += 1
                else:
                    losses += 1
            done += batch
            print(
                f"arena {done}/{args.games} +{wins} ={draws} -{losses} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
        score = (wins + 0.5 * draws) / max(1, wins + draws + losses)
        print(f"ARENA A={args.arena_a} B={args.arena_b} W/D/L={wins}/{draws}/{losses} score={score:.4f}")
        return 0

    if not args.model:
        ap.error("--model is required for self-play")
    net = Net(args.model, device)
    started = time.time()
    all_samples = []
    raw_chunks = []
    wins = draws = losses = 0
    done = 0
    while done < args.games:
        batch = min(args.games - done, args.batch)
        nets = [net] * batch
        samples, results, raws = play_batch(
            nets, batch, args.sims, args.c_puct, args.top_k, args.root_k,
            args.noise, args.temp_moves, args.temperature, args.opening,
            args.seed + done * 17, args.game_offset + done,
        )
        all_samples.extend(samples)
        raw_chunks.append(raws)
        wins += int((results == 1).sum())
        draws += int((results == 0).sum())
        losses += int((results == -1).sum())
        done += batch
        elapsed = max(1e-6, time.time() - started)
        rate = done / elapsed
        print(
            f"selfplay {done}/{args.games} positions={len(all_samples)} "
            f"R/B/D={wins}/{losses}/{draws} {rate:.2f} games/s "
            f"eta={(args.games - done) / rate:.0f}s",
            flush=True,
        )
    flat = np.concatenate(raw_chunks) if raw_chunks else np.zeros(0, np.int16)
    if flat.size:
        print(
            f"raw (red-black) median={np.median(flat):.1f} mean={flat.mean():.2f} "
            f"p10={np.percentile(flat, 10):.0f} p90={np.percentile(flat, 90):.0f}",
            flush=True,
        )
        print(f"red win rate at komi={KOMI:g}: {wins / max(1, wins + draws + losses):.3f}", flush=True)
    count = write_shard(args.out, all_samples)
    print(f"wrote {count} samples to {args.out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
