#!/usr/bin/env python3
"""Is the teacher at 1024 visits leaving anything on the table?

Two corpora from the same model, the same openings and the same rules, differing
only in how hard the search looked (1024 vs 4096 visits).  This reports, per
corpus: what the teacher's targets look like, how far the model is from them,
and — the question that decides where to spend GPU hours — whether the 4x teacher
asks for something the policy does not already know.

Value numbers come with a per-game bootstrap: positions inside one game share a
single outcome, so treating them as independent would invent certainty.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_resnet as T

REPO = Path(__file__).resolve().parents[1]
ANCHOR = 0.5
BUCKETS = [(0, 12, "opening 0-11"), (12, 26, "middlegame 12-25"),
           (26, 38, "late 26-37"), (38, 99, "endgame 38+")]


def targets(data, planes, globals_width):
    a = np.full(len(data), ANCHOR, dtype=np.float32)
    x, g, _ = T.make_inputs(data, planes, a, globals_width)
    policy = data["policy"].astype(np.float32)
    policy /= np.maximum(policy.sum(1, keepdims=True), 1e-9)
    if "score" in data.dtype.names:
        win, tie = T.value_from_score(data, a)
        value = win * (1.0 - 0.5 * tie) + 0.5 * tie
    else:
        value = data["value"].astype(np.float32)
    return x, g, torch.from_numpy(policy), torch.from_numpy(value.astype(np.float32))


def run(model, x, g, legal, policy, value, batch, device):
    """Per-position policy CE, KL, top-1 and squared value error."""
    n = len(policy)
    ce = np.zeros(n, dtype=np.float64)
    kl = np.zeros(n, dtype=np.float64)
    hit = np.zeros(n, dtype=np.float64)
    sq = np.zeros(n, dtype=np.float64)
    ent = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = slice(start, min(start + batch, n))
            logits, pred, *_ = model(x[idx].to(device), g[idx].to(device))
            logits = logits.masked_fill(~legal[idx].to(device), -1e9)
            logp = F.log_softmax(logits, dim=1)
            t = policy[idx].to(device)
            ce[idx] = -(t * logp).sum(1).cpu().numpy()
            kl[idx] = (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum(1).cpu().numpy()
            hit[idx] = (logp.argmax(1) == t.argmax(1)).cpu().numpy()
            ent[idx] = -(logp.exp() * logp).sum(1).cpu().numpy()
            sq[idx] = ((pred - value[idx].to(device)) ** 2).cpu().numpy()
    return ce, kl, hit, sq, ent


def per_game_bootstrap(sq, value, games, rounds=2000, seed=1):
    """95% interval for value MSE, resampling whole games."""
    order = np.argsort(games, kind="stable")
    uniq, starts = np.unique(games[order], return_index=True)
    bounds = np.append(starts, len(order))
    sorted_sq = sq[order]
    per_game_sq = np.array([sorted_sq[bounds[i]:bounds[i + 1]].mean() for i in range(len(uniq))])
    per_game_n = np.diff(bounds)
    per_game_v = np.array([value[order][bounds[i]:bounds[i + 1]].mean() for i in range(len(uniq))])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(rounds, len(uniq)))
    mse = (per_game_sq[picks] * per_game_n[picks]).sum(1) / per_game_n[picks].sum(1)
    # baseline: always predict the resampled mean outcome
    mean_v = (per_game_v[picks] * per_game_n[picks]).sum(1) / per_game_n[picks].sum(1)
    return float(sq.mean()), np.percentile(mse, [2.5, 97.5]), float(value.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpora", nargs="+", required=True)
    p.add_argument("--models", nargs="*")
    p.add_argument("--root", type=Path, default=REPO / "runs/training")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    corpora = {}
    for path in args.corpora:
        data, _, ruleset = T._read_shard(path)
        corpora[Path(path).stem] = (data, ruleset)
        print(f"{Path(path).name}: {len(data)} positions, "
              f"{len(np.unique(data['game']))} games, ruleset {ruleset}")

    print(f"\n{'corpus':>18} {'pos':>6} {'policy entropy':>14} {'support':>8} "
          f"{'max p':>6} {'value sd':>8}")
    for name, (data, _) in corpora.items():
        pol = data["policy"].astype(np.float64)
        pol /= np.maximum(pol.sum(1, keepdims=True), 1e-12)
        ent = -(np.where(pol > 0, pol * np.log(np.maximum(pol, 1e-30)), 0.0)).sum(1)
        support = (pol > 0).sum(1)
        print(f"{name:>18} {len(data):>6} {ent.mean():>14.3f} {support.mean():>8.1f} "
              f"{pol.max(1).mean():>6.3f} {data['value'].std():>8.3f}")

    paths = []
    for item in (args.models or ["b6c96-s305983-d3226791", "b6c96-s179866-d1959110"]):
        hit = sorted(args.root.rglob(f"{item}.bin"))
        if not hit:
            raise SystemExit(f"model not found: {item}")
        paths.append(hit[0])

    device = torch.device(args.device)
    for path in paths:
        with path.open("rb") as handle:
            head = handle.read(28)
        _, channels, blocks, planes, globals_width = __import__("struct").unpack("<IIIII", head[8:])
        print(f"\n=== {path.stem} ===")
        print(f"{'corpus':>18} {'CE':>7} {'KL':>7} {'top1':>6} {'vMSE':>7} {'vMSE 95%':>16}")
        model = T.ResNet(channels, blocks, planes, 2, globals_width)
        model.use_score_head = True
        T.load_bin(model, path)
        model.eval().to(device)
        for name, (data, ruleset) in corpora.items():
            x, g, policy, value = targets(data, planes, globals_width)
            legal = T.make_legal_mask(data, ruleset)
            ce, kl, hit, sq, ent = run(model, x, g, legal, policy, value, args.batch, device)
            mse, ci, base = per_game_bootstrap(sq, value.numpy(), data["game"])
            print(f"{name:>18} {ce.mean():7.3f} {kl.mean():7.3f} {hit.mean():6.1%} "
                  f"{mse:7.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]")
            if name.endswith("1024"):
                print(f"{'  by phase':>18}")
                for low, high, label in BUCKETS:
                    sel = (data["ply"] >= low) & (data["ply"] < high)
                    if not sel.any():
                        continue
                    print(f"    {label:<18} n={int(sel.sum()):>6} CE {ce[sel].mean():6.3f} "
                          f"top1 {hit[sel].mean():5.1%} target entropy "
                          f"{-(policy.numpy()[sel] * np.log(np.maximum(policy.numpy()[sel], 1e-12))).sum(1).mean():5.3f}")


if __name__ == "__main__":
    main()
