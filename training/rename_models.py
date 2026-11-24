#!/usr/bin/env python3
"""Rewrite model file names so `s` and `d` are cumulative over the whole chain.

`b4c32-s34760-d470705.bin` means "34,760 optimizer steps on 470,705 positions".
A run that continues that model must keep counting: its generation 0 is
34,760 + N steps, not N.  A name that restarts at zero describes the same weights
with a smaller number, and two numbers that cannot be compared are worse than no
number at all.

This scans every run manifest, recomputes the cumulative counters the way
pipeline.py now does, and renames the files that disagree.  Dry run by default:
it prints the plan and changes nothing.

    python training/rename_models.py            # plan only
    python training/rename_models.py --apply

Files are only renamed when the target does not already exist; when the target
exists and the bytes match, the duplicate is removed instead (same weights, one
name).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
from pipeline import lineage_start  # noqa: E402  (one definition of the counters)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def model_name(blocks, channels, steps, data) -> str:
    return f"b{blocks}c{channels}-s{steps}-d{data}"


def plan_for(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    config = manifest.get("config") or {}
    blocks, channels = int(config.get("blocks", 0)), int(config.get("channels", 0))
    if blocks <= 0 or channels <= 0:
        return []
    base_steps, base_data = lineage_start(config.get("init_model") or "")
    run_steps = run_data = 0
    actions = []
    for record in manifest.get("generations", []):
        run_steps += int(record.get("optimizer_steps", 0))
        if record.get("mode") != "warm":
            run_data += int(record.get("selfplay_samples", 0))
        inherited_steps = run_steps if config.get("init_model") else run_steps
        inherited_data = run_data if config.get("init_model") else run_data
        target = model_name(blocks, channels, base_steps + inherited_steps,
                            base_data + inherited_data)
        # The file as it exists now: the run-local name, whatever it was.
        local = model_name(blocks, channels, run_steps,
                           run_data if config.get("init_model") else run_data)
        source = None
        models_dir = run_dir / "models"
        if models_dir.is_dir():
            for candidate in sorted(models_dir.glob("*.bin")):
                stem = candidate.stem
                if stem == target:
                    source = candidate
                    break
                if re.search(rf"-s{run_steps}(-|$)", stem):
                    source = candidate
                    break
                if re.search(rf"-s{run_steps}\b", stem):
                    source = candidate
                    break
        if source is None:
            continue
        if source.stem == target:
            continue
        actions.append((source, models_dir / f"{target}.bin", record.get("gen"),
                        local, target))
    return actions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-smoke", action="store_true",
                    help="also rename runs/smoke (pipeline tests, not a lineage)")
    args = ap.parse_args(argv)

    plans = []
    for manifest in sorted((REPO / "runs").rglob("manifest.json")):
        if not args.include_smoke and "smoke" in manifest.parts:
            continue
        try:
            actions = plan_for(manifest)
        except Exception as exc:  # noqa: BLE001 - one broken run must not stop the scan
            print(f"skip {manifest}: {exc}")
            continue
        if actions:
            plans.append((manifest, actions))

    if not plans:
        print("every model name already counts cumulatively")
        return 0

    for manifest, actions in plans:
        print(f"\n{manifest.parent.relative_to(REPO)}")
        for source, target, gen, local, wanted in actions:
            print(f"  gen {gen}: {source.name}  ->  {target.name}")
            if target.exists():
                same = md5(source) == md5(target)
                print(f"         target exists, {'same bytes' if same else 'DIFFERENT bytes'}")
            if args.apply:
                if target.exists():
                    if md5(source) == md5(target):
                        source.unlink()
                    else:
                        print("         refusing to overwrite different weights")
                        continue
                else:
                    source.rename(target)
                if args.apply:
                    text = manifest.read_text(encoding="utf-8")
                    patched = text.replace(source.stem, target.stem)
                    if patched != text:
                        manifest.write_text(patched, encoding="utf-8")
    if not args.apply:
        print("\n(dry run: pass --apply to rename)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
