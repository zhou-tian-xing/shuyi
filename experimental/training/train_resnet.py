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
GLOBAL = 22
POLICY = 729
VALUE_HIDDEN = 32
MAGIC1 = b"SYSELF1\x00"
MAGIC2 = b"SYSELF2\x00"
MAGIC3 = b"SYSELF3\x00"
MAGIC4 = b"SYSELF4\x00"
RULESET_NAMES = {0: "classic", 1: "blocking"}

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
    def __init__(self, channels=16, blocks=2, input_planes=INPUT_PLANES, ruleset=0):
        super().__init__()
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.input_planes = int(input_planes)
        self.ruleset = int(ruleset)
        if self.input_planes not in (INPUT_PLANES, INPUT_PLANES_V3):
            raise ValueError(f"unsupported input planes: {self.input_planes}")
        self.stem = nn.Conv2d(self.input_planes, channels, 3, padding=1, bias=True)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 9, 1, padding=0, bias=True)
        self.hand_bias = nn.Linear(GLOBAL, 9, bias=True)
        # Optional score head: predicts the final raw score, tanh-squashed.
        self.use_score_head = False
        self.score_conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.score_bias = nn.Linear(1 + GLOBAL, VALUE_HIDDEN, bias=True)
        self.score_out = nn.Linear(VALUE_HIDDEN, 1, bias=True)
        self.value_conv = nn.Conv2d(channels, 1, 1, padding=0, bias=True)
        self.value_fc1 = nn.Linear(1 + GLOBAL, VALUE_HIDDEN, bias=True)
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
    for path in paths:
        data, version, shard_ruleset = _read_shard(path)
        if shard_ruleset != ruleset:
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
    data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
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


def make_inputs(data, input_planes=INPUT_PLANES):
    """Vectorized feature construction matching src/nn.cpp."""
    if input_planes not in (INPUT_PLANES, INPUT_PLANES_V3):
        raise ValueError(f"unsupported input planes: {input_planes}")
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

    g = np.zeros((n, GLOBAL), dtype=np.float32)
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
    return torch.from_numpy(x), torch.from_numpy(g)


def make_legal_mask(data):
    n = len(data)
    boards = data["board"].reshape(n, CELLS)
    hands = data["hand"].reshape(n, 2, 9)
    sides = data["side"].astype(np.int64)
    remaining = hands[np.arange(n), sides]  # [n,9]
    mask = (boards == 0)[:, :, None] & (remaining[:, None, :] > 0)  # [n,81,9]
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
        if magic not in (b"SHUYINN2", b"SHUYINN3"):
            raise ValueError(f"bad model magic {magic!r}")
        version, channels, blocks, inputs, globals_ = struct.unpack("<IIIII", f.read(20))
        file_ruleset = 0
        if magic == b"SHUYINN3":
            flags, file_ruleset, feature_version, _reserved = struct.unpack("<IBBH", f.read(8))
            if feature_version != 1:
                raise ValueError(f"unsupported feature layout {feature_version}")
            has_score = bool(flags & 1)
            if version != 3:
                raise ValueError("unsupported model format")
        elif version != 2:
            raise ValueError("unsupported model format")
        if globals_ != GLOBAL:
            raise ValueError("unsupported model format")
        if file_ruleset != getattr(model, "ruleset", 0):
            raise ValueError(
                f"model was trained for the {RULESET_NAMES.get(file_ruleset, file_ruleset)} "
                f"rule set, cannot load it into a {RULESET_NAMES.get(getattr(model, 'ruleset', 0))} model"
            )
        # A 19-plane checkpoint can warm start a 37-plane model: the 19 planes
        # are a prefix of the 37-plane layout, so the extra hand planes start at
        # zero weight.  This is a warm start, not distillation.
        expand = inputs != model.input_planes
        if expand and not (inputs == INPUT_PLANES and model.input_planes == INPUT_PLANES_V3):
            raise ValueError(f"cannot warm start {model.input_planes} planes from {inputs}")
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
        state["hand_bias.weight"] = _read_float_tensor(f, (9, GLOBAL))
        state["hand_bias.bias"] = _read_float_tensor(f, (9,))
        state["value_conv.weight"] = _read_float_tensor(f, (1, channels, 1, 1))
        state["value_conv.bias"] = _read_float_tensor(f, (1,))
        state["value_fc1.weight"] = _read_float_tensor(f, (VALUE_HIDDEN, 1 + GLOBAL))
        state["value_fc1.bias"] = _read_float_tensor(f, (VALUE_HIDDEN,))
        state["value_fc2.weight"] = _read_float_tensor(f, (1, VALUE_HIDDEN))
        state["value_fc2.bias"] = _read_float_tensor(f, (1,))
    # Non-strict: a file without a score head simply leaves that head at its
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
    with open(tmp, "wb") as f:
        f.write(b"SHUYINN3")
        f.write(
            struct.pack(
                "<IIIII", 3, model.channels, model.blocks, model.input_planes, GLOBAL
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
    parser.add_argument("--ruleset", choices=["classic", "blocking"], default="classic",
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
    parser.add_argument("--val", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=400000)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.init_only:
        code = 0 if args.ruleset == "classic" else 1
        save_bin(ResNet(args.channels, args.blocks, args.input_planes, code), args.out)
        print(f"wrote initial model {args.out}", flush=True)
        return 0
    if not args.data:
        raise SystemExit("--data is required unless --init-only is given")

    data = load_shards(args.data, args.max_samples, args.seed,
                      0 if args.ruleset == "classic" else 1)
    train_mask, val_mask = split_by_game(data, args.val, args.seed)
    x, g = make_inputs(data, args.input_planes)
    policy = torch.from_numpy(data["policy"].astype(np.float32))
    value = torch.from_numpy(data["value"].astype(np.float32))
    score = None
    if args.score_weight > 0:
        if "score" not in data.dtype.names:
            raise SystemExit("--score-weight needs SYSELF4 data (with a score field)")
        score = torch.from_numpy(data["score"].astype(np.float32) / 30.0)
    legal = make_legal_mask(data)
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

    ruleset_code = 0 if args.ruleset == "classic" else 1
    model = ResNet(args.channels, args.blocks, args.input_planes, ruleset_code).to(device)
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
                or checkpoint.get("input_planes", INPUT_PLANES) != args.input_planes):
            raise ValueError("checkpoint shape does not match --channels/--blocks/--input-planes")
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
        order = train_idx[torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(args.seed + epoch))]
        total_loss = 0.0
        seen = 0
        started = time.time()
        for start in range(0, len(order), args.batch):
            idx = order[start : start + args.batch]
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
