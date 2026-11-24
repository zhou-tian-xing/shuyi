#!/usr/bin/env python3
"""Mirror a run directory from the training server, resumably.

The rented box can be released the moment the hour runs out, so every artifact
has to exist locally: models, self-play shards, arena JSON and the log.  This
walks the remote tree and copies anything missing or with a different size --
run it repeatedly while training is still going and the final sync is nearly
free.

    python training/pull.py --remote /root/blk/runs/training/b4c32-nopass-20261124 \
        --local runs/training/b4c32-nopass-20261124
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssh  # noqa: E402  (same directory, shares host/port/password defaults)


def walk(sftp, root, prefix=""):
    out = []
    for entry in sftp.listdir_attr(root):
        remote = root.rstrip("/") + "/" + entry.filename
        if stat.S_ISDIR(entry.st_mode):
            out.extend(walk(sftp, remote, prefix))
        else:
            out.append((remote, int(entry.st_size)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--include", action="append", default=None,
                    help="only paths containing this substring (repeatable)")
    ap.add_argument("--max-bytes", type=int, default=0, help="stop after this many bytes")
    args = ap.parse_args()

    client = ssh.connect()
    sftp = client.open_sftp()
    include = [item for group in (args.include or []) for item in group.split(",") if item]
    files = sorted(walk(sftp, args.remote))
    total = sum(size for _r, size in files)
    print(f"{len(files)} remote files, {total / 1e6:.1f} MB", flush=True)

    copied = skipped = failed = 0
    sent = 0
    started = time.time()
    for remote, size in files:
        if include and not any(key in remote for key in include):
            continue
        local = Path(args.local) / Path(remote).relative_to(args.remote)
        if local.exists() and local.stat().st_size == size:
            skipped += 1
            continue
        if args.max_bytes and sent + size > args.max_bytes:
            print("byte budget reached", flush=True)
            break
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_suffix(local.suffix + ".part")
        try:
            sftp.get(remote, str(part))
            if part.stat().st_size != size:
                raise OSError(f"size mismatch {part.stat().st_size} != {size}")
            os.replace(part, local)
            copied += 1
            sent += size
            rate = sent / 1e6 / max(1e-9, time.time() - started)
            print(f"  {size / 1e6:7.1f} MB  {rate:5.2f} MB/s  {local}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a flaky link must not lose the run
            failed += 1
            print(f"  !! {remote}: {exc}", flush=True)
            if part.exists():
                part.unlink()
    sftp.close()
    client.close()
    print(f"copied={copied} skipped={skipped} failed={failed} "
          f"{sent / 1e6:.1f} MB in {time.time() - started:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
