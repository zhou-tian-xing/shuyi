#!/usr/bin/env python3
"""Rewrite a v2 model header as v3 (rule set + feature layout), weights untouched.

v2 headers only describe the shape, so a 37-plane file from before the plane fix
(side to move written at index 36) and one from after are byte-indistinguishable.
v3 records the rule set and the feature layout, and the C++ loader refuses v2
files with 37 planes because it cannot know which layout they used.  This script
is the explicit migration: it stamps the layout the caller states.

    python training/model_migrate.py docs/reports/srv1_run/models/*.bin --ruleset classic
"""
import argparse
import struct
import sys
from pathlib import Path

V2_MAGIC = b"SHUYINN2"
V3_MAGIC = b"SHUYINN3"
V2_HEADER = 8 + 20          # magic + version/channels/blocks/inputs/globals
V3_HEADER = V2_HEADER + 8   # + flags + ruleset + featureVersion + reserved


def migrate(path, ruleset, dry_run=False):
    data = Path(path).read_bytes()
    if len(data) < V2_HEADER:
        raise SystemExit(f"{path}: truncated file")
    if data[:8] == V3_MAGIC:
        print(f"{path}: already v3, skipped")
        return False
    if data[:8] != V2_MAGIC:
        raise SystemExit(f"{path}: unknown magic {data[:8]!r}")
    version, channels, blocks, inputs, globals_ = struct.unpack("<IIIII", data[8:V2_HEADER])
    if version != 2:
        raise SystemExit(f"{path}: unexpected version {version}")
    flags = 0
    header = V3_MAGIC + struct.pack(
        "<IIIII", 3, channels, blocks, inputs, globals_
    ) + struct.pack("<IBBH", flags, ruleset, 1, 0)
    assert len(header) == V3_HEADER, len(header)
    if not dry_run:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            f.write(header)
            f.write(data[V2_HEADER:])
        tmp.replace(path)
    print(f"{path}: v2 -> v3 (ruleset={ruleset}, planes={inputs}, b{blocks}c{channels})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ruleset", choices=["classic", "blocking", "nopass"], default="nopass")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    code = {"classic": 0, "blocking": 1, "nopass": 2}[args.ruleset]
    changed = 0
    for model in args.models:
        if migrate(model, code, args.dry_run):
            changed += 1
    print(f"{changed} file(s) migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
