#!/usr/bin/env python3
"""Train the Shuyi ResNet policy/value model on MCTS self-play.

This is the GPU/server trainer.  It keeps the C++ binary weight layout exactly
(see src/nn.cpp) while also supporting a larger, shape-tagged model such as
32 channels x 4 residual blocks.

Dataset v2 (written by shuyi_selfplay):
  magic "SYSELF2\\0", uint32 version, uint32 samples
  per sample: uint8 side, int8 board[81], uint8 hand[18],
              float32 policy[729], float32 value,
              uint32 game, uint8 ply
Dataset v1 is still accepted for the original baseline.

Key differences from the first baseline trainer:
  * 8-fold dihedral augmentation (policy + board transformed together)
  * legal-move masking in the policy softmax
  * game-level train/validation split (v2 metadata)
  * AMP, cosine LR with warmup, grad clipping
  * warm start from a .bin model, checkpoint/resume
  * multi-shard datasets and a capped replay window
"""
import argparse
import csv
import glob
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = 81
INPUT_PLANES = 19
INPUT_PLANES_V3 = 37
# Global feature width.  v4 (this trainer) appends A and |A| at 22..23 so the
# value head can answer "who wins given this A" instead of only learning the
# outcome under one fixed komi.  GLOBAL_V3 is the old 22-wide layout, still
# readable so an existing model can warm start a v4 one.
GLOBAL = 24
GLOBAL_V3 = 22
A_SCALE = 10.0
POLICY = 729
VALUE_HIDDEN = 32
MAGIC1 = b"SYSELF1\x00"
MAGIC2 = b"SYSELF2\x00"
MAGIC3 = b"SYSELF3\x00"
MAGIC4 = b"SYSELF4\x00"
RULESET_NAMES = {0: "classic", 1: "blocking", 2: "nopass"}
RULESET_CODES = {name: code for code, name in RULESET_NAMES.items()}

BASE_DTYPE = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (CELLS,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (POLICY,)),
        ("value", "<f4"),
    ],
    align=False,
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

V2_DTYPE = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (CELLS,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (POLICY,)),
        ("value", "<f4"),
        ("game", "<u4"),
        ("ply", "u1"),
    ],
    align=False,
)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(self, channels=16, blocks=2, input_planes=INPUT_PLANES, ruleset=0,
                 globals=GLOBAL):
        super().__init__()
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.input_planes = int(input_planes)
        self.globals = int(globals)
        self.ruleset = int(ruleset)
        if self.input_planes not in (INPUT_PLANES, INPUT_PLANES_V3):
            raise ValueError(f"unsupported input planes: {self.input_planes}")
        if self.globals not in (GLOBAL, GLOBAL_V3):
            raise ValueError(f"unsupported global features: {self.globals}")
        self.stem = nn.Conv2d(self.input_planes, channels, 3, padding=1, bias=True)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 9, 1, padding=0, bias=True)
        self.hand_bias = nn.Linear(self.globals, 9, bias=True)
        # Optional score head: predicts the final raw score, tanh-squashed.
        self.use_score_head = False
        self.score_conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.score_bias = nn.Linear(1 + self.globals, VALUE_HIDDEN, bias=True)
        self.score_out = nn.Linear(VALUE_HIDDEN, 1, bias=True)
        self.value_conv = nn.Conv2d(channels, 1, 1, padding=0, bias=True)
        self.value_fc1 = nn.Linear(1 + self.globals, VALUE_HIDDEN, bias=True)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1, bias=True)

    def forward(self, x, global_features):
        h = F.relu(self.stem(x))
        for block in self.res_blocks:
            h = block(h)
        # [B,9,9,9] -> [B,9,81] -> [B,81,9] -> [B,729]
        p = self.policy(h).permute(0, 2, 3, 1)
        hand = self.hand_bias(global_features).unsqueeze(1).unsqueeze(1)
        p = p + hand
        policy_logits = p.reshape(p.shape[0], -1)
        v = self.value_conv(h).mean(dim=(2, 3))
        v = torch.cat([v, global_features], dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.sigmoid(self.value_fc2(v)).squeeze(1)
        if self.use_score_head:
            s2 = self.score_conv(h).mean(dim=(2, 3))
            s2 = torch.cat([s2, global_features], dim=1)
            s2 = F.relu(self.score_bias(s2))
            s2 = torch.tanh(self.score_out(s2)).squeeze(1)
            return policy_logits, v, s2
        return policy_logits, v


def _read_shard(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic == MAGIC4:
            version, count = struct.unpack("<II", f.read(8))
            ruleset = struct.unpack("<B", f.read(1))[0]
            f.read(3)
            dtype = V4_DTYPE
            data = np.fromfile(f, dtype=dtype, count=count)
            if len(data) != count:
                raise ValueError(f"truncated dataset {path}: {len(data)}/{count}")
            return data, version, ruleset
        if magic == MAGIC3:
            version, count = struct.unpack("<II", f.read(8))
            ruleset = struct.unpack("<B", f.read(1))[0]
            f.read(3)  # reserved
            dtype = V2_DTYPE
        elif magic in (MAGIC1, MAGIC2):
            version, count = struct.unpack("<II", f.read(8))
            ruleset = 0  # v1/v2 predate the rule sets
            dtype = V2_DTYPE if version == 2 else BASE_DTYPE
        else:
            raise ValueError(f"bad dataset magic {magic!r}: {path}")
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported dataset version {version}: {path}")
        data = np.fromfile(f, dtype=dtype, count=count)
        if len(data) != count:
            raise ValueError(f"truncated dataset {path}: {len(data)}/{count}")
    return data, version, ruleset


def load_shards(paths, max_samples=0, seed=1, ruleset=0):
    """Read v1/v2/v3 shards, verify the rule set, concatenate, cap by whole games."""
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("no dataset shards")
    chunks = []
    total = 0
    blocking_for_nopass = 0
    for path in paths:
        data, version, shard_ruleset = _read_shard(path)
        # A nopass game is a blocking game up to its earlier end, so blocking
        # shards can seed a nopass run (warm start from the previous model's
        # data).  The reverse is not true and stays an error.
        if shard_ruleset == 1 and ruleset == 2:
            blocking_for_nopass += 1
        elif shard_ruleset != ruleset:
            raise ValueError(
                f"{path}: data is for the {RULESET_NAMES.get(shard_ruleset, shard_ruleset)} "
                f"rule set but --ruleset is {RULESET_NAMES.get(ruleset, ruleset)}"
            )
        if version == 1:
            # Treat every v1 sample as its own game so the game-level split
            # still reduces leakage as much as the old format allows.
            data = data.copy()
            # Rebuild with v2 fields.  `astype` would drop fields, so assemble.
            upgraded = np.zeros(len(data), dtype=V2_DTYPE)
            for name in ("side", "board", "hand", "policy", "value"):
                upgraded[name] = data[name]
            upgraded["game"] = np.arange(len(data), dtype=np.uint32) + total
            upgraded["ply"] = 0
            data = upgraded
        chunks.append(data)
        total += len(data)
    # Every shard numbers its games from 0, so concatenating them makes two
    # different games share an id.  The split is by game id, so unrelated games
    # were grouped together and `--max-samples` could keep "half a game"; the
    # logged `games=` count was also halved.  Namespace by shard.
    offset = 0
    for index, chunk in enumerate(chunks):
        if not len(chunk):
            continue
        counted = chunk["game"].astype(np.int64) + offset
        offset = int(counted.max()) + 1
        chunk = chunk.copy()
        chunk["game"] = counted.astype(np.uint32)
        chunks[index] = chunk
    data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if blocking_for_nopass:
        print(f"note: {blocking_for_nopass} blocking shard(s) used for a nopass run "
              f"(a nopass game is a blocking game up to its earlier end)")
    if max_samples and len(data) > max_samples:
        games = np.unique(data["game"])
        rng = np.random.default_rng(seed)
        rng.shuffle(games)
        selected = []
        count = 0
        for game in games:
            mask = data["game"] == game
            selected.append(data[mask])
            count += int(mask.sum())
            if count >= max_samples:
                break
        data = np.concatenate(selected)
    return data


def _max_neighbor5(planes):
    """Max over the cell and its four orthogonal neighbours, zero outside."""
    padded = np.pad(planes, ((0, 0), (1, 1), (1, 1)), mode="constant")
    out = padded[:, 1:-1, 1:-1].copy()
    out = np.maximum(out, padded[:, :-2, 1:-1])
    out = np.maximum(out, padded[:, 2:, 1:-1])
    out = np.maximum(out, padded[:, 1:-1, :-2])
    out = np.maximum(out, padded[:, 1:-1, 2:])
    return out


def make_inputs(data, input_planes=INPUT_PLANES, a_values=None, globals_width=GLOBAL):
    """Vectorized feature construction matching src/nn.cpp."""
    if input_planes not in (INPUT_PLANES, INPUT_PLANES_V3):
        raise ValueError(f"unsupported input planes: {input_planes}")
    if globals_width not in (GLOBAL, GLOBAL_V3):
        raise ValueError(f"unsupported global features: {globals_width}")
    if globals_width == GLOBAL and a_values is None:
        raise ValueError("a 24-wide global vector needs a-values")
    n = len(data)
    boards = data["board"].astype(np.int16)
    hands = data["hand"].astype(np.float32)
    sides = data["side"].astype(np.float32)
    x = np.zeros((n, input_planes, 9, 9), dtype=np.float32)
    for side, sign in ((0, 1), (1, -1)):
        for number in range(1, 10):
            plane = side * 9 + (number - 1)
            x[:, plane] = (boards == sign * number).reshape(n, 9, 9)
    # Fixed layout: 0..17 stones, 18 side to move, 19..36 remaining hand counts.
    if input_planes == INPUT_PLANES_V3:
        for side in range(2):
            for number in range(9):
                plane = 19 + side * 9 + number
                x[:, plane, :, :] = (hands[:, side * 9 + number] / 3.0)[:, None, None]
    x[:, 18, :, :] = sides[:, None, None]

    g = np.zeros((n, globals_width), dtype=np.float32)
    g[:, :18] = hands / 3.0
    g[:, 18] = sides
    board3 = boards.reshape(n, 9, 9)
    red = np.maximum(board3, 0).astype(np.float32)
    black = np.maximum(-board3, 0).astype(np.float32)
    rmax = _max_neighbor5(red)
    bmax = _max_neighbor5(black)
    raw = ((rmax > bmax).astype(np.float32) - (rmax < bmax).astype(np.float32)).sum(
        axis=(1, 2)
    )
    g[:, 19] = raw / 81.0
    g[:, 20] = data["ply"].astype(np.float32) / 54.0 if "ply" in data.dtype.names else 0.0
    g[:, 21] = (boards.reshape(n, CELLS) == 0).sum(axis=1) / 81.0
    if globals_width == GLOBAL:
        apply_a_features(g, a_values)
    return torch.from_numpy(x), torch.from_numpy(g), raw


def apply_a_features(g, a_values):
    """Writes A/10 and |A|/10 into columns 22..23, matching buildGlobals()."""
    scaled = np.asarray(a_values, dtype=np.float32) / A_SCALE
    g[:, 22] = np.clip(scaled, -2.0, 2.0)
    g[:, 23] = np.clip(np.abs(scaled), 0.0, 2.0)
    return g


def value_from_score(data, a_values):
    """P(mover wins) given A: sign(final raw score + A, mover's perspective).

    `score` is stored raw (no komi) from the mover's point of view, so the A
    term flips sign with the side: A is added to Red's score, and Black wants
    Red's total to be negative.  This is the exact game result, which makes it a
    better value target than the search value it replaces (that one is a
    bootstrapped estimate of the same thing under a single A).
    """
    if "score" not in data.dtype.names:
        raise ValueError("A-conditioned value targets need SYSELF4 data with a score field")
    score = data["score"].astype(np.float32)
    side_sign = np.where(data["side"].astype(np.int64) == 0, 1.0, -1.0).astype(np.float32)
    total = score + side_sign * np.asarray(a_values, dtype=np.float32)
    return (total > 0).astype(np.float32), (total == 0).astype(np.float32)


def make_legal_mask(data, ruleset=0):
    """Actions a player may legally take, as a [n, 729] bool mask.

    The blocking rule is a *placement* rule, and nopass keeps it (nopass only
    changes what happens once a side has no placement): number n may not go on a
    cell orthogonally adjacent to an enemy stone of number >= n.  Classic is the
    only rule set without it.  Leaving it out kept ~5% of the policy
    normalisation on moves the engine will never play, and made the target
    statistics' legal denominator wrong for nopass.
    """
    n = len(data)
    board3 = data["board"].reshape(n, 9, 9).astype(np.int16)
    boards = board3.reshape(n, CELLS)
    hands = data["hand"].reshape(n, 2, 9)
    sides = data["side"].astype(np.int64)
    remaining = hands[np.arange(n), sides]  # [n,9]
    mask = (boards == 0)[:, :, None] & (remaining[:, None, :] > 0)  # [n,81,9]
    if ruleset in (1, 2):
        # A cell rejects number n when a neighbour holds an *opposing* stone of
        # number >= n, so a red 9 denies all nine numbers of its cross.
        mover = np.where(sides == 0, 1, -1).astype(np.int16)[:, None, None]
        enemy = np.where(board3 * mover < 0, np.abs(board3), 0).astype(np.float32)
        enemy_max = _max_neighbor5(enemy).astype(np.int16).reshape(n, CELLS)
        numbers = np.arange(1, 10)[None, None, :]
        mask = mask & (numbers > enemy_max[:, :, None])
    return torch.from_numpy(mask.reshape(n, POLICY))


def _read_float_tensor(f, shape):
    count = int(np.prod(shape))
    values = np.fromfile(f, dtype="<f4", count=count)
    if len(values) != count:
        raise ValueError("truncated model weights")
    return torch.from_numpy(values.reshape(shape).copy())


def load_bin(model, path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"SHUYINN2", b"SHUYINN3", b"SHUYINN4"):
            raise ValueError(f"bad model magic {magic!r}")
        version, channels, blocks, inputs, globals_ = struct.unpack("<IIIII", f.read(20))
        file_ruleset = 0
        if magic in (b"SHUYINN3", b"SHUYINN4"):
            flags, file_ruleset, feature_version, _reserved = struct.unpack("<IBBH", f.read(8))
            if feature_version != 1:
                raise ValueError(f"unsupported feature layout {feature_version}")
            has_score = bool(flags & 1)
            expected = 4 if magic == b"SHUYINN4" else 3
            if version != expected:
                raise ValueError("unsupported model format")
        elif version != 2:
            raise ValueError("unsupported model format")
        if globals_ not in (GLOBAL, GLOBAL_V3):
            raise ValueError(
                f"model has {globals_} global features, expected {GLOBAL} (v4) or "
                f"{GLOBAL_V3} (v3)"
            )
        model_ruleset = getattr(model, "ruleset", 0)
        # A blocking model may warm start a NoPass run: every NoPass game is a
        # blocking game up to its earlier end, so the weights mean the same
        # thing there.  The reverse is not true and is refused.
        compatible = file_ruleset == model_ruleset or (file_ruleset == 1 and model_ruleset == 2)
        if not compatible:
            raise ValueError(
                f"model was trained for the {RULESET_NAMES.get(file_ruleset, file_ruleset)} "
                f"rule set, cannot load it into a {RULESET_NAMES.get(model_ruleset)} model"
            )
        # A 19-plane checkpoint can warm start a 37-plane model: the 19 planes
        # are a prefix of the 37-plane layout, so the extra hand planes start at
        # zero weight.  This is a warm start, not distillation.
        expand = inputs != model.input_planes
        if expand and not (inputs == INPUT_PLANES and model.input_planes == INPUT_PLANES_V3):
            raise ValueError(f"cannot warm start {model.input_planes} planes from {inputs}")
        # A 22-wide global vector (v3) can warm start a 24-wide one (v4): the new
        # A columns start at zero, which makes the old value head exactly
        # preserved until fine-tuning gives A real weight.
        grow_globals = globals_ != model.globals
        if grow_globals and not (globals_ == GLOBAL_V3 and model.globals == GLOBAL):
            raise ValueError(f"cannot warm start {model.globals} globals from {globals_}")
        if channels != model.channels or blocks != model.blocks:
            raise ValueError(
                f"model shape mismatch: file {channels}x{blocks}, "
                f"expected {model.channels}x{model.blocks}"
            )
        state = {}
        stem = _read_float_tensor(f, (channels, inputs, 3, 3))
        if expand:
            grown = torch.zeros(channels, model.input_planes, 3, 3)
            grown[:, :inputs] = stem
            stem = grown
        state["stem.weight"] = stem
        state["stem.bias"] = _read_float_tensor(f, (channels,))
        for i, block in enumerate(model.res_blocks):
            state[f"res_blocks.{i}.conv1.weight"] = _read_float_tensor(
                f, (channels, channels, 3, 3)
            )
            state[f"res_blocks.{i}.conv1.bias"] = _read_float_tensor(f, (channels,))
            state[f"res_blocks.{i}.conv2.weight"] = _read_float_tensor(
                f, (channels, channels, 3, 3)
            )
            state[f"res_blocks.{i}.conv2.bias"] = _read_float_tensor(f, (channels,))
        state["policy.weight"] = _read_float_tensor(f, (9, channels, 1, 1))
        state["policy.bias"] = _read_float_tensor(f, (9,))

        def widen(tensor, src_width, dst_width):
            """Zero-pad a [out, width] tensor on the right.

            The global vector is a trailing block, so the first `src_width`
            columns keep their meaning; for value_fc1 that block starts with the
            spatial value, hence 1 + globals rather than globals.
            """
            if not grow_globals:
                return tensor
            grown = torch.zeros(tensor.shape[0], dst_width)
            grown[:, :src_width] = tensor
            return grown

        state["hand_bias.weight"] = widen(
            _read_float_tensor(f, (9, globals_)), globals_, model.globals
        )
        state["hand_bias.bias"] = _read_float_tensor(f, (9,))
        state["value_conv.weight"] = _read_float_tensor(f, (1, channels, 1, 1))
        state["value_conv.bias"] = _read_float_tensor(f, (1,))
        state["value_fc1.weight"] = widen(
            _read_float_tensor(f, (VALUE_HIDDEN, 1 + globals_)), 1 + globals_,
            1 + model.globals
        )
        state["value_fc1.bias"] = _read_float_tensor(f, (VALUE_HIDDEN,))
        state["value_fc2.weight"] = _read_float_tensor(f, (1, VALUE_HIDDEN))
        state["value_fc2.bias"] = _read_float_tensor(f, (1,))
        if has_score:
            # save_bin writes these whenever the score head is enabled, so a
            # continued run has to read them back.  Dropping them here silently
            # reset the head to its random initialisation at every generation,
            # while its loss still flowed into the shared trunk.
            state["score_conv.weight"] = _read_float_tensor(f, (1, channels, 1, 1))
            state["score_conv.bias"] = _read_float_tensor(f, (1,))
            state["score_bias.weight"] = _read_float_tensor(f, (VALUE_HIDDEN, 1 + globals_))
            state["score_bias.bias"] = _read_float_tensor(f, (VALUE_HIDDEN,))
            state["score_out.weight"] = _read_float_tensor(f, (1, VALUE_HIDDEN))
            state["score_out.bias"] = _read_float_tensor(f, (1,))
    # Non-strict: a file without a score head leaves that head at its
    # initialisation; everything else must match exactly.
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("score_")]
    if bad_missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={bad_missing} unexpected={unexpected}")
    return model


def save_bin(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    globals_width = getattr(model, "globals", GLOBAL)
    # A 24-wide global vector means the model reads A, so the file must say v4:
    # an older engine refuses a v4 file instead of reading 24-wide weights as 22.
    magic, version = (b"SHUYINN4", 4) if globals_width == GLOBAL else (b"SHUYINN3", 3)
    with open(tmp, "wb") as f:
        f.write(magic)
        f.write(
            struct.pack(
                "<IIIII", version, model.channels, model.blocks, model.input_planes,
                globals_width
            )
        )
        flags = 1 if getattr(model, "use_score_head", False) else 0
        f.write(struct.pack("<IBBH", flags, model.ruleset, 1, 0))

        def write(name):
            tensor = model.state_dict()[name].detach().cpu().contiguous().numpy()
            f.write(tensor.astype("<f4").tobytes())

        write("stem.weight")
        write("stem.bias")
        for i in range(model.blocks):
            write(f"res_blocks.{i}.conv1.weight")
            write(f"res_blocks.{i}.conv1.bias")
            write(f"res_blocks.{i}.conv2.weight")
            write(f"res_blocks.{i}.conv2.bias")
        write("policy.weight")
        write("policy.bias")
        write("hand_bias.weight")
        write("hand_bias.bias")
        write("value_conv.weight")
        write("value_conv.bias")
        write("value_fc1.weight")
        write("value_fc1.bias")
        write("value_fc2.weight")
        write("value_fc2.bias")
        if getattr(model, "use_score_head", False):
            # Same tail shape as the value head; nn.cpp reads it only when the
            # flag is set.
            write("score_conv.weight")
            write("score_conv.bias")
            write("score_bias.weight")
            write("score_bias.bias")
            write("score_out.weight")
            write("score_out.bias")
    os.replace(tmp, path)


def augment_batch(x, policy, mask, generator=None):
    """Apply one random dihedral transform to a whole batch (on GPU)."""
    k = int(torch.randint(0, 4, (1,), device=x.device, generator=generator).item())
    flip = bool(torch.randint(0, 2, (1,), device=x.device, generator=generator).item())
    x = torch.rot90(x, k, dims=(-2, -1))
    p = policy.reshape(-1, 9, 9, 9)
    m = mask.reshape(-1, 9, 9, 9)
    p = torch.rot90(p, k, dims=(1, 2))
    m = torch.rot90(m, k, dims=(1, 2))
    if flip:
        x = torch.flip(x, dims=(-1,))
        p = torch.flip(p, dims=(2,))
        m = torch.flip(m, dims=(2,))
    return x, p.reshape(-1, POLICY), m.reshape(-1, POLICY)


def split_by_game(data, val_fraction, seed):
    games = np.unique(data["game"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    val_games = max(1, int(round(len(games) * val_fraction)))
    val_set = set(games[:val_games].tolist())
    is_val = np.fromiter((int(g) in val_set for g in data["game"]), bool, len(data))
    return ~is_val, is_val


def lr_at(step, total_steps, base_lr, min_lr, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, x, g, policy, value, mask, indices, device, batch, value_weight):
    model.eval()
    total = 0.0
    val_policy = 0.0
    val_value = 0.0
    top1 = 0
    top5 = 0
    mae = 0.0
    for start in range(0, len(indices), batch):
        idx = indices[start : start + batch]
        bx = x[idx].to(device)
        bg = g[idx].to(device)
        bp = policy[idx].to(device)
        bv = value[idx].to(device)
        bm = mask[idx].to(device)
        outputs = model(bx, bg)
        logits, pred_value = outputs[0], outputs[1]
        masked = logits.masked_fill(~bm, -1e4)
        logp = F.log_softmax(masked, dim=1)
        ploss = -(bp * logp).sum(dim=1).mean()
        vloss = F.mse_loss(pred_value, bv)
        total += float(ploss + value_weight * vloss) * len(idx)
        val_policy += float(ploss) * len(idx)
        val_value += float(vloss) * len(idx)
        target = bp.argmax(dim=1)
        top = masked.topk(5, dim=1).indices
        top1 += int((top[:, 0] == target).sum())
        top5 += int((top == target[:, None]).any(dim=1).sum())
        mae += float((pred_value - bv).abs().sum())
    n = max(1, len(indices))
    return {
        "loss": total / n,
        "policy": val_policy / n,
        "value": val_value / n,
        "top1": top1 / n,
        "top5": top5 / n,
        "mae": mae / n,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=[], help="dataset shards")
    parser.add_argument("--out", required=True, help="output .bin model")
    parser.add_argument("--init-only", action="store_true",
                        help="write a freshly initialised model and exit (no training)")
    parser.add_argument("--checkpoint", default="", help="last checkpoint .pt")
    parser.add_argument("--summary", default="", help="final metrics .json")
    parser.add_argument("--log-csv", default="", help="per-epoch metrics .csv")
    parser.add_argument("--init-model", default="", help="warm start from .bin")
    parser.add_argument("--resume", default="", help="resume from .pt")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--input-planes", type=int, default=INPUT_PLANES,
                        choices=[INPUT_PLANES, INPUT_PLANES_V3])
    parser.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass",
                        help="rule set the data and the model belong to")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--score-weight", type=float, default=0.0,
                        help=">0 enables the score head (needs SYSELF4 data with a score field)")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    # --- A conditioning (v4) -------------------------------------------------
    parser.add_argument("--a-cond", action="store_true",
                        help="train with A as an input (v4 model): the global vector grows "
                             "to 24 and the value target is sign(final raw score + A)")
    parser.add_argument("--a-range", type=float, default=4.0,
                        help="legacy: A is drawn uniformly from [-range, range].  Only "
                             "used when --a-values is empty; A is equivalent inside an "
                             "interval between two integers, so a continuous draw "
                             "teaches the same thing many times over")
    parser.add_argument("--a-values", default="",
                        help="comma-separated A values to draw from instead (one "
                             "representative per equivalence class).  A only enters the "
                             "label as sign(raw + A) with raw an integer, so every A in "
                             "(k, k+1) gives the identical target; half-integers are the "
                             "natural representatives and they cannot draw (raw + k.5 is "
                             "never 0).  Integers are their own class and can draw")
    parser.add_argument("--a-anchor", type=float, default=1.26,
                        help="the A the engine actually plays with (avalue.py measured "
                             "+1.26 at 192 visits); part of every epoch uses it verbatim")
    parser.add_argument("--a-anchor-fraction", type=float, default=0.35,
                        help="fraction of training samples drawn at the anchor A; the "
                             "rest explore the range (0 = never, 1 = anchor only)")
    parser.add_argument("--val", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=400000)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="for the first N steps, print how long the host-side "
                             "data path (gather + upload) takes versus the GPU step")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    if not 0.0 <= args.a_anchor_fraction <= 1.0:
        raise SystemExit("--a-anchor-fraction must be in [0, 1]")
    if args.a_range < 0.0:
        raise SystemExit("--a-range must be >= 0")

    a_choices = []
    if args.a_values.strip():
        try:
            a_choices = [float(item) for item in args.a_values.replace(" ", "").split(",") if item]
        except ValueError:
            raise SystemExit(f"--a-values must be comma-separated numbers: {args.a_values!r}")
        if not a_choices:
            raise SystemExit("--a-values was given but holds no value")
        # Two A values teach the same target when no integer lies between them, so
        # listing both is a config mistake, not extra coverage.
        seen = {}
        for value in a_choices:
            key = ("int", value) if float(value).is_integer() else ("open", math.floor(value))
            if key in seen:
                raise SystemExit(
                    f"--a-values has two equivalents: {seen[key]:+g} and {value:+g} "
                    f"(A only matters between two integers)")
            seen[key] = value

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.init_only:
        code = RULESET_CODES[args.ruleset]
        save_bin(
            ResNet(args.channels, args.blocks, args.input_planes, code,
                   GLOBAL if args.a_cond else GLOBAL_V3),
            args.out,
        )
        print(f"wrote initial model {args.out}", flush=True)
        return 0
    if not args.data:
        raise SystemExit("--data is required unless --init-only is given")

    data = load_shards(args.data, args.max_samples, args.seed,
                       RULESET_CODES[args.ruleset])
    train_mask, val_mask = split_by_game(data, args.val, args.seed)
    if args.a_cond and "score" not in data.dtype.names:
        raise SystemExit(
            "--a-cond needs the final score of every game (SYSELF4 data); the older "
            "shards cannot relabel the value target for a random A"
        )
    # Without --a-cond the model keeps the old 22-wide global vector: an A-blind
    # model stays a deliberate choice rather than an accident.
    globals_width = GLOBAL if args.a_cond else GLOBAL_V3
    a_values = np.full(len(data), args.a_anchor, dtype=np.float32)
    x, g, _raw = make_inputs(data, args.input_planes, a_values, globals_width)
    policy = torch.from_numpy(data["policy"].astype(np.float32))
    if args.a_cond:
        win, tie = value_from_score(data, a_values)
        value = torch.from_numpy((win * (1.0 - 0.5 * tie) + 0.5 * tie).astype(np.float32))
    else:
        value = torch.from_numpy(data["value"].astype(np.float32))
    score = None
    if args.score_weight > 0:
        if "score" not in data.dtype.names:
            raise SystemExit("--score-weight needs SYSELF4 data (with a score field)")
        score = torch.from_numpy(data["score"].astype(np.float32) / 30.0)
    legal = make_legal_mask(data, RULESET_CODES[args.ruleset])
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    val_idx = torch.from_numpy(np.flatnonzero(val_mask))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(
        f"samples={len(data)} games={len(np.unique(data['game']))} "
        f"train={len(train_idx)} val={len(val_idx)} device={device}",
        flush=True,
    )

    ruleset_code = RULESET_CODES[args.ruleset]
    model = ResNet(args.channels, args.blocks, args.input_planes, ruleset_code,
                   globals_width).to(device)
    model.use_score_head = args.score_weight > 0
    if args.init_model:
        load_bin(model, args.init_model)
        print(f"warm start: {args.init_model}", flush=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    history = []
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint["channels"] != args.channels or checkpoint["blocks"] != args.blocks
                or checkpoint.get("input_planes", INPUT_PLANES) != args.input_planes
                or checkpoint.get("globals", globals_width) != globals_width):
            raise ValueError(
                "checkpoint shape does not match --channels/--blocks/--input-planes/globals"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", [])
        print(f"resumed {args.resume} at epoch {start_epoch}", flush=True)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if checkpoint is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    steps_per_epoch = max(1, math.ceil(len(train_idx) / args.batch))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup)
    best_score = float("inf")
    if history:
        best_score = min(float(row.get("val_loss", 1e9)) for row in history)
    global_step = start_epoch * steps_per_epoch

    # A-conditioned training: every epoch redraws A per sample, so one position
    # teaches the head "wins under A=-6, loses under A=+6" instead of memorising
    # a single komi.  Validation always sits at the anchor A, which keeps
    # val_loss comparable between epochs and equal to the operating point.
    g_np = g.numpy()
    value_np = value.numpy()

    def draw_a(epoch):
        if not args.a_cond:
            return
        # Plain Python ints: np.uint64 * int upcasts to float64 on numpy 1.x,
        # and default_rng refuses a float seed.
        rng = np.random.default_rng(int(args.seed) * 7919 + int(epoch))
        if a_choices:
            # One representative per equivalence class: A and A' teach the same
            # target whenever no integer lies between them.
            pick = rng.integers(0, len(a_choices), len(data))
            draw = np.asarray(a_choices, dtype=np.float32)[pick]
        else:
            draw = rng.uniform(-args.a_range, args.a_range, len(data)).astype(np.float32)
        if args.a_anchor_fraction > 0.0:
            at_anchor = rng.random(len(data)) < args.a_anchor_fraction
            draw[at_anchor] = args.a_anchor
        draw[~train_mask] = args.a_anchor
        apply_a_features(g_np, draw)
        win, tie = value_from_score(data, draw)
        value_np[:] = win * (1.0 - 0.5 * tie) + 0.5 * tie

    if args.log_csv and not Path(args.log_csv).exists():
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "val_loss",
                    "val_policy",
                    "val_value",
                    "top1",
                    "top5",
                    "value_mae",
                    "seconds",
                ]
            )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        draw_a(epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(args.seed + epoch))]
        total_loss = 0.0
        seen = 0
        started = time.time()
        for start in range(0, len(order), args.batch):
            idx = order[start : start + args.batch]
            profiling = args.profile_steps > 0 and global_step < args.profile_steps
            if profiling:
                t_data0 = time.perf_counter()
            bx = x[idx].to(device, non_blocking=True)
            bg = g[idx].to(device, non_blocking=True)
            bp = policy[idx].to(device, non_blocking=True)
            bv = value[idx].to(device, non_blocking=True)
            bm = legal[idx].to(device, non_blocking=True)
            if args.augment:
                bx, bp, bm = augment_batch(bx, bp, bm)
            if args.label_smoothing > 0:
                # Uniform smoothing only over legal moves.
                count = bm.sum(dim=1, keepdim=True).clamp(min=1)
                eps = args.label_smoothing
                bp = (1.0 - eps) * bp + eps * bm.float() / count

            lr = lr_at(global_step, total_steps, args.lr, args.min_lr, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if profiling:
                torch.cuda.synchronize()
                t_data1 = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(bx, bg)
                logits, pred_value = outputs[0], outputs[1]
                masked = logits.masked_fill(~bm, -1e4)
                logp = F.log_softmax(masked, dim=1)
                policy_loss = -(bp * logp).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_value, bv)
                loss = policy_loss + args.value_weight * value_loss
                if score is not None and len(outputs) == 3:
                    bs = score[idx].to(device, non_blocking=True)
                    score_loss = F.smooth_l1_loss(outputs[2], bs)
                    loss = loss + args.score_weight * score_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)
            global_step += 1
            if profiling:
                torch.cuda.synchronize()
                t_data2 = time.perf_counter()
                print(f"PROFILE step={global_step} data={1000 * (t_data1 - t_data0):.2f}ms "
                      f"compute={1000 * (t_data2 - t_data1):.2f}ms", flush=True)
            if args.log_every and (start // args.batch) % args.log_every == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {start // args.batch}/"
                    f"{steps_per_epoch} lr={lr:.2e} loss={float(loss.item()):.4f}",
                    flush=True,
                )

        metrics = evaluate(
            model,
            x,
            g,
            policy,
            value,
            legal,
            val_idx,
            device,
            args.batch,
            args.value_weight,
        )
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": total_loss / max(1, seen),
            "val_loss": metrics["loss"],
            "val_policy": metrics["policy"],
            "val_value": metrics["value"],
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "value_mae": metrics["mae"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(
            f"epoch {epoch + 1}/{args.epochs} train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} policy={row['val_policy']:.4f} "
            f"value={row['val_value']:.4f} top1={row['top1']:.3f} "
            f"top5={row['top5']:.3f} mae={row['value_mae']:.4f} "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )
        if args.log_csv:
            with open(args.log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

        if args.checkpoint:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "channels": args.channels,
                "blocks": args.blocks,
                "input_planes": args.input_planes,
                "globals": globals_width,
                "history": history,
                "args": vars(args),
            }
            path = Path(args.checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path.with_suffix(".tmp"))
            os.replace(path.with_suffix(".tmp"), path)
            if metrics["loss"] < best_score:
                best_score = metrics["loss"]
                best_path = Path(str(path) + ".best")
                torch.save(checkpoint, best_path)

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_bin(model, args.out)

    save_bin(model, args.out)
    if not math.isfinite(best_score) and history:
        best_score = min(float(row.get("val_loss", 1e9)) for row in history)
    summary = {
        "samples": int(len(data)),
        "games": int(len(np.unique(data["game"]))),
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "epochs": int(args.epochs),
        "channels": int(args.channels),
        "blocks": int(args.blocks),
        "input_planes": int(args.input_planes),
        "globals": int(globals_width),
        "a_cond": bool(args.a_cond),
        "a_anchor": float(args.a_anchor),
        "a_range": float(args.a_range),
        "a_values": a_choices,
        "best_val_loss": float(best_score),
        "last": history[-1] if history else {},
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
