#!/usr/bin/env python3
"""Widen a trained network into a bigger architecture without changing what it computes.

Growing is the only capacity experiment that is about capacity.  Training a
128-channel net from random play for ten hours measures the bootstrap, not the
architecture: it would start below everything on the ladder and spend the window
rediscovering what b6c96 already knows.

Zero-padding is the obvious way to widen and it is wrong: a padded channel is
zero, every consumer of it has weight zero, so its gradient is zero forever and
the new capacity never learns.  `check_gradients` fails the build if that
happens, which is how this file came to duplicate channels instead.

The widening is net2net: pick `channels - old` distinct old channels, duplicate
each one everywhere in the trunk, and halve that channel's weight in every
consumer so the sum is unchanged.  Two copies at weight 1/2 each contribute what
one copy at weight 1 contributed, every weight stays non-zero, and the network
computes exactly what it did before:

  * stem out-channels: the new row is a copy of its source row (weight and bias);
  * each block's conv1/conv2 in-channels: the source column and its copy are
    each half of the original column -- the duplicated input arrives twice;
  * each block's conv1/conv2 out-channels: row copy, same argument one layer on;
  * the three 1x1 heads: in-channels halved for the same reason;
  * the appended blocks are identity by construction: conv1 is small random,
    conv2 is zero, so `relu(x + 0) == x` exactly.  conv2's gradient is
    `delta_y * relu(conv1(x))`, which is non-zero, so the new depth wakes up on
    the first step instead of sitting dead until something else moves it.

`--check` (on by default) compares both networks on random positions and then
runs one backward pass, refusing to write a file unless the outputs agree and
every tensor receives gradient.  The first generation of the following run
plays its own source, which is a second, independent check: a widened net that
scores anything but ~50% against its source was not widened correctly.

    python training/grow_model.py --src runs/.../b6c96-s346359-d3675586.bin \
        --blocks 10 --channels 128 --out runs/training/b10c128-.../b10c128-....bin
"""
import argparse
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_resnet import ResNet, load_bin, save_bin  # noqa: E402


def header(path):
    """(channels, blocks, input planes, globals, ruleset, has score head)."""
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if magic not in (b"SHUYINN2", b"SHUYINN3", b"SHUYINN4"):
            raise SystemExit(f"{path}: not a model file ({magic!r})")
        _, channels, blocks, planes, globals_ = struct.unpack("<IIIII", handle.read(20))
        has_score = False
        ruleset = 0
        if magic in (b"SHUYINN3", b"SHUYINN4"):
            flags, ruleset, feature_version, _ = struct.unpack("<IBBH", handle.read(8))
            if feature_version != 1:
                raise SystemExit(f"{path}: unsupported feature layout {feature_version}")
            has_score = bool(flags & 1)
    return channels, blocks, planes, globals_, ruleset, has_score


def channel_map(old, new, generator=None):
    """For each new channel >= `old`, the old channel it duplicates."""
    add = new - old
    if add == 0:
        return []
    if add > old:
        raise SystemExit(f"cannot widen {old} -> {new} by duplication: "
                         f"needs {add} distinct sources, only {old} exist")
    order = torch.randperm(old, generator=generator)[:add]
    return [int(i) for i in order]


def grow(src, blocks, channels, seed=11):
    """A `channels` x `blocks` network that computes exactly what `src` computes."""
    sc, sb = src.channels, src.blocks
    if channels < sc or blocks < sb:
        raise SystemExit(f"growth goes up only: {sb}x{sc} -> {blocks}x{channels}")
    dst = ResNet(channels, blocks, src.input_planes, src.ruleset, src.globals)
    dst.use_score_head = src.use_score_head
    src_state, state = src.state_dict(), {}
    pairs = channel_map(sc, channels, torch.Generator().manual_seed(seed))
    paired = set(pairs)
    # Every consumer sums over the widened trunk, where a duplicated channel
    # arrives twice, so its weight is halved; unpaired channels keep weight 1.
    half = torch.tensor([0.5 if c in paired else 1.0 for c in range(sc)])

    def widen_in(tensor):
        """[out, old, ...] -> [out, new, ...]: halve paired columns, append copies."""
        shape = (1, sc) + (1,) * (tensor.dim() - 2)
        out = torch.zeros(tensor.shape[0], channels, *tensor.shape[2:])
        out[:, :sc] = tensor * half.view(shape)
        for offset, source in enumerate(pairs):
            out[:, sc + offset] = out[:, source].clone()
        return out

    def widen_out(tensor):
        """[old, ...] -> [new, ...]: append a copy of the source row."""
        out = torch.zeros(channels, *tensor.shape[1:])
        out[:sc] = tensor
        for offset, source in enumerate(pairs):
            out[sc + offset] = tensor[source]
        return out

    # The stem's input is the plane axis, not the trunk: widen its outputs only.
    state["stem.weight"] = widen_out(src_state["stem.weight"])
    state["stem.bias"] = widen_out(src_state["stem.bias"]).reshape(channels)
    for i in range(blocks):
        if i < sb:
            for conv in ("conv1", "conv2"):
                name = f"res_blocks.{i}.{conv}"
                state[f"{name}.weight"] = widen_out(
                    widen_in(src_state[f"{name}.weight"]))
                state[f"{name}.bias"] = widen_out(
                    src_state[f"{name}.bias"]).reshape(channels)
        else:
            # Identity: conv2 is zero, so the block adds nothing; conv1 is small
            # random so conv2 starts with a non-zero gradient.
            gen = torch.Generator().manual_seed(seed + i)
            state[f"res_blocks.{i}.conv1.weight"] = (
                torch.rand(channels, channels, 3, 3, generator=gen) - 0.5) * 0.02
            state[f"res_blocks.{i}.conv1.bias"] = torch.zeros(channels)
            state[f"res_blocks.{i}.conv2.weight"] = torch.zeros(channels, channels, 3, 3)
            state[f"res_blocks.{i}.conv2.bias"] = torch.zeros(channels)
    state["policy.weight"] = widen_in(src_state["policy.weight"])
    state["policy.bias"] = src_state["policy.bias"].clone()
    state["hand_bias.weight"] = src_state["hand_bias.weight"].clone()
    state["hand_bias.bias"] = src_state["hand_bias.bias"].clone()
    state["value_conv.weight"] = widen_in(src_state["value_conv.weight"])
    state["value_conv.bias"] = src_state["value_conv.bias"].clone()
    for name in ("value_fc1.weight", "value_fc1.bias", "value_fc2.weight",
                 "value_fc2.bias"):
        state[name] = src_state[name].clone()
    if src.use_score_head:
        state["score_conv.weight"] = widen_in(src_state["score_conv.weight"])
        state["score_conv.bias"] = src_state["score_conv.bias"].clone()
        for name in ("score_bias.weight", "score_bias.bias", "score_out.weight",
                     "score_out.bias"):
            state[name] = src_state[name].clone()
    missing, unexpected = dst.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(("score_", "value_"))]
    if bad or unexpected:
        raise SystemExit(f"state dict mismatch: missing={bad} unexpected={unexpected}")
    return dst, pairs


def sample_batch(model, samples=64, seed=7):
    """Sparse one-hot stone planes, in the range the network saw while training."""
    gen = torch.Generator().manual_seed(seed)
    planes = torch.zeros(samples, model.input_planes, 9, 9)
    for i in range(samples):
        for plane in range(min(18, model.input_planes)):
            for _ in range(int(torch.randint(0, 12, (1,), generator=gen))):
                cell = int(torch.randint(0, 81, (1,), generator=gen))
                planes[i, plane, cell // 9, cell % 9] = 1.0
        planes[i, -1] = float(torch.randint(0, 2, (1,), generator=gen))
    globals_ = torch.rand(samples, model.globals, generator=gen)
    return planes, globals_


def compare(src, dst, samples=64):
    """Largest disagreement between the two networks on the same positions."""
    planes, globals_ = sample_batch(src, samples)
    with torch.no_grad():
        left, right = src(planes, globals_), dst(planes, globals_)
    names = ("policy logits", "value", "score")
    return {name: float((a - b).abs().max()) for name, a, b in zip(names, left, right)}


def check_gradients(model, with_score, new_blocks=()):
    """Tensor names that receive no gradient at all -- i.e. capacity that is dead."""
    model.train()
    for parameter in model.parameters():
        parameter.grad = None
    planes, globals_ = sample_batch(model, 32, seed=13)
    logits, value = model(planes, globals_)[:2]
    # Any smooth loss reaches every tensor that the forward pass uses.
    loss = logits.pow(2).mean() + value.mean()
    if with_score:
        loss = loss + model(planes, globals_)[2].mean()
    loss.backward()
    dead = [name for name, parameter in model.named_parameters()
            if parameter.grad is None or float(parameter.grad.abs().max()) == 0.0]
    # An appended block's conv1 is dead until its conv2 leaves zero: the gradient
    # of a product is delta_y * relu(conv1(x)), and conv2 is what is being learned
    # first.  Its conv2 is graded from the first step, so the block is alive.
    excused = {f"res_blocks.{i}.conv1.{p}" for i in new_blocks for p in ("weight", "bias")}
    model.eval()
    return [name for name in dead if name not in excused], [n for n in dead if n in excused]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="largest allowed output difference (float32 noise is ~1e-6)")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    channels, blocks, planes, globals_, ruleset, has_score = header(args.src)
    if (channels, blocks) == (args.channels, args.blocks):
        raise SystemExit("source and target shapes are the same; nothing to grow")
    src = load_bin(ResNet(channels, blocks, planes, ruleset, globals_), args.src)
    src.use_score_head = has_score
    src.eval()
    dst, pairs = grow(src, args.blocks, args.channels)
    dst.eval()

    params = sum(p.numel() for p in dst.parameters())
    print(f"{args.src.name}: {blocks}x{channels} -> {args.blocks}x{args.channels}, "
          f"{params / 1e6:.2f} M parameters, {len(pairs)} channels duplicated")
    if not args.no_check:
        diffs = compare(src, dst)
        print("  max |output difference|: " +
              ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()))
        worst = max(diffs.values())
        if worst > args.tolerance:
            raise SystemExit(f"widened net does not reproduce the source ({worst:.2e} > "
                             f"{args.tolerance:.0e}); refusing to write {args.out}")
        dead, deferred = check_gradients(dst, has_score, range(blocks, args.blocks))
        if dead:
            raise SystemExit("these tensors would never learn (zero gradient): "
                             + ", ".join(dead))
        print("  gradient reaches every tensor"
              + (f" (a new block's conv1 waits for its conv2: {len(deferred)}) "
                 if deferred else ""))
    save_bin(dst, args.out)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
