#!/usr/bin/env python3
"""How far is each checkpoint from its own search?  (No training, no server.)

A self-play shard stores, per position, the policy target the search produced and
the value target the game produced.  Running older checkpoints over those same
positions answers a question the loss curves cannot: has the current network
already absorbed the signal its teacher produces, or is there still room?

    python training/probe_teacher_gap.py --shard runs/analysis/teacher_gap/shard1024.bin

Output, per checkpoint: policy cross entropy against the search target, top-1
agreement, KL(target || model), and value MAE.  A gap that keeps shrinking toward
the end means more of the same data still buys something; a gap that has gone
flat while play strength stalls means the teacher, not the student, is the limit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_resnet as T

DEFAULT_SEQUENCE = [
    "b6c96-s2800-d37761",
    "b6c96-s8160-d110349",
    "b6c96-s31536-d371618",
    "b6c96-s87696-d937553",
    "b6c96-s179866-d1959110",
    "b6c96-s253603-d2698576",
    "b6c96-s305983-d3226791",
]


REPO = Path(__file__).resolve().parents[1]
ANCHOR = 0.5


def load(path, channels, blocks, planes, ruleset, globals_width):
    model = T.ResNet(channels, blocks, planes, ruleset, globals_width)
    model.use_score_head = True
    T.load_bin(model, path)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--models", nargs="*", help="paths or stems; default is a ladder")
    p.add_argument("--root", type=Path, default=REPO / "runs/training")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    data, version, ruleset = T._read_shard(args.shard)
    print(f"{args.shard.name}: {len(data)} positions, version {version}, ruleset {ruleset}, "
          f"{len(np.unique(data['game']))} games")

    paths = []
    if args.models:
        for item in args.models:
            candidate = Path(item)
            if candidate.is_file():
                paths.append(candidate)
            else:
                hits = sorted(args.root.rglob(f"{item}.bin"))
                if not hits:
                    raise SystemExit(f"model not found: {item}")
                paths.append(hits[0])
    else:
        for stem in DEFAULT_SEQUENCE:
            hits = sorted(args.root.rglob(f"{stem}.bin"))
            paths.append(hits[0] if hits else None)

    a_values = np.full(len(data), ANCHOR, dtype=np.float32)
    first = next(p for p in paths if p is not None)
    with first.open("rb") as handle:
        head = handle.read(28)
    _, channels, blocks, planes, globals_width = __import__("struct").unpack("<IIIII", head[8:])
    x, g, _ = T.make_inputs(data, planes, a_values, globals_width)
    target = torch.from_numpy(data["policy"].astype(np.float32))
    target = target / target.sum(1, keepdim=True).clamp_min(1e-9)
    # Under --a-cond the trainer rebuilds the value label from the final score and
    # the sampled A; the shard's own `value` field is ignored.  Compare against the
    # same target the model was trained on, or the number is meaningless.
    if "score" in data.dtype.names and globals_width == T.GLOBAL:
        win, tie = T.value_from_score(data, a_values)
        value = torch.from_numpy((win * (1.0 - 0.5 * tie) + 0.5 * tie).astype(np.float32))
        value_note = f"value target = sign(score + {ANCHOR}), {float(tie.mean()):.1%} ties"
    else:
        value = torch.from_numpy(data["value"].astype(np.float32))
        value_note = "value target = the shard's own field"
    print(value_note)
    legal = T.make_legal_mask(data, ruleset)
    device = torch.device(args.device)

    print(f"{'checkpoint':>28} {'CE':>7} {'KL':>7} {'top1':>6} {'vMAE':>6} {'ent':>5}")
    for path in paths:
        if path is None:
            continue
        with path.open("rb") as handle:
            head = handle.read(28)
        _, c, b, pl, gw = __import__("struct").unpack("<IIIII", head[8:])
        if (pl, gw) != (planes, globals_width):
            raise SystemExit(f"{path.name}: {pl} planes / {gw} globals, "
                             f"sample was built for {planes} / {globals_width}")
        model = load(path, c, b, pl, ruleset, gw).to(device)
        ce = kl = top1 = mae = 0.0
        ent = 0.0
        with torch.no_grad():
            for start in range(0, len(data), args.batch):
                idx = slice(start, start + args.batch)
                logits, pred, *_ = model(x[idx].to(device), g[idx].to(device))
                logits = logits.masked_fill(~legal[idx].to(device), -1e9)
                logp = F.log_softmax(logits, dim=1)
                p = logp.exp()
                bt = target[idx].to(device)
                ce += float(-(bt * logp).sum(1).sum())
                kl += float((bt * (torch.log(bt.clamp_min(1e-12)) - logp)).sum(1).sum())
                top1 += int((logp.argmax(1) == bt.argmax(1)).sum())
                mae += float((pred - value[idx].to(device)).abs().sum())
                ent += float(-(p * logp).sum(1).sum())
        n = len(data)
        print(f"{path.stem:>28} {ce/n:7.3f} {kl/n:7.3f} {top1/n:6.1%} {mae/n:6.3f} {ent/n:5.2f}")


if __name__ == "__main__":
    main()
