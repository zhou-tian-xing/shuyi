#!/usr/bin/env python3
"""Diagnose where the policy net wants to play in the opening.

Example:
  python training/diagnose_openings.py --model models/shuyi_resnet_v2.bin \
      --data runs/run_xxx/gen_*/selfplay/shard_*.bin --ply 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.train_resnet import (  # noqa: E402
    ResNet,
    load_bin,
    load_shards,
    make_inputs,
    make_legal_mask,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--ply", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--input-planes", type=int, default=19, choices=[19, 37])
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--json", default="", help="optional metrics json")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(args.channels, args.blocks, args.input_planes)
    load_bin(model, args.model)
    model.to(device)
    model.eval()

    data = load_shards(args.data, args.max_samples, 1)
    data = data[data["ply"] <= args.ply]
    if len(data) == 0:
        raise SystemExit("no opening samples")
    x, g = make_inputs(data, args.input_planes)
    mask = make_legal_mask(data)
    x = x.to(device)
    g = g.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        logits, value = model(x, g)
        masked = logits.masked_fill(~mask, -1e4)
        pred = masked.argmax(dim=1).cpu()
        target = torch.from_numpy(np.ascontiguousarray(data["policy"])).argmax(dim=1)
    cells = pred.numpy() // 9
    xs, ys = cells % 9, cells // 9
    corner = ((xs == 0) | (xs == 8)) & ((ys == 0) | (ys == 8))
    edge = (((xs == 0) | (xs == 8)) | ((ys == 0) | (ys == 8))) & ~corner
    center = (xs >= 2) & (xs <= 6) & (ys >= 2) & (ys <= 6)
    inner = ~(corner | edge | center)
    n = len(data)
    print(f"model={args.model} samples={n} ply<={args.ply} channels={args.channels} blocks={args.blocks} planes={args.input_planes}")
    print(f"corner={corner.mean():.3f} edge={edge.mean():.3f} inner={inner.mean():.3f} center={center.mean():.3f}")
    print(f"top1_agreement={(pred == target).float().mean():.3f}")
    counts = np.bincount(cells, minlength=81)
    top = np.argsort(counts)[-10:][::-1]
    print("top cells:", ", ".join(f"{chr(97 + c % 9)}{c // 9 + 1}:{counts[c]}" for c in top))
    print(f"mean_value={float(value.mean()):.3f}")
    metrics = {
        "model": args.model,
        "samples": int(n),
        "max_ply": int(args.ply),
        "corner": float(corner.mean()),
        "edge": float(edge.mean()),
        "inner": float(inner.mean()),
        "center": float(center.mean()),
        "top1_agreement": float((pred == target).float().mean()),
        "mean_value": float(value.mean()),
        "top_cells": [
            {"cell": int(c), "name": f"{chr(97 + c % 9)}{c // 9 + 1}", "count": int(counts[c])}
            for c in top
        ],
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
