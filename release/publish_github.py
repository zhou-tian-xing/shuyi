#!/usr/bin/env python3
"""Create the GitHub release for a tag and upload its assets.

    python release/publish_github.py --tag v1.0 --notes release/NOTES-1.0.md \\
        --asset dist/1.0/*.zip models/b6c96-s305983-d3226791.bin

The token comes from the credential helper git already uses (`git credential
fill`), so nothing is read from a file or an environment variable and nothing is
printed.  Re-running is safe: an existing release is reused and assets whose name
and size already match are skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"


def token() -> str:
    proc = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, text=True, timeout=60)
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("git has no credential for github.com; run `git push` once first")


def call(method: str, url: str, auth: str, payload=None, raw: bytes | None = None,
         content_type: str = "application/json"):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload else None)
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {auth}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "shuyi-release")
    if data is not None:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read()
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v1.0")
    parser.add_argument("--name", default="数弈 1.0")
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--asset", type=Path, nargs="+", required=True)
    parser.add_argument("--repo", default="zhou-tian-xing/shuyi")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()

    auth = token()
    status, me = call("GET", f"{API}/user", auth)
    if status != 200:
        raise SystemExit(f"token rejected ({status}): {me.get('message')}")
    print(f"authenticated as {me['login']}")

    body = args.notes.read_text(encoding="utf-8")
    status, release = call("POST", f"{API}/repos/{args.repo}/releases", auth, payload={
        "tag_name": args.tag,
        "name": args.name,
        "body": body,
        "draft": args.draft,
        "prerelease": False,
    })
    if status == 422:  # already there: reuse it
        status, release = call("GET", f"{API}/repos/{args.repo}/releases/tags/{args.tag}", auth)
    if status not in (200, 201):
        raise SystemExit(f"release failed ({status}): {release.get('message')}")
    print(f"release {release['html_url']}")

    status, existing = call("GET", f"{API}/repos/{args.repo}/releases/{release['id']}/assets", auth)
    have = {a["name"]: a["size"] for a in existing} if status == 200 else {}

    for path in args.asset:
        if not path.is_file():
            raise SystemExit(f"missing asset: {path}")
        size = path.stat().st_size
        if have.get(path.name) == size:
            print(f"  {path.name:34s} already uploaded")
            continue
        query = urllib.parse.urlencode({"name": path.name})
        status, result = call("POST",
                              f"https://uploads.github.com/repos/{args.repo}/releases/{release['id']}/assets?{query}",
                              auth, raw=path.read_bytes(), content_type="application/octet-stream")
        if status not in (200, 201):
            raise SystemExit(f"upload failed for {path.name} ({status}): {result.get('message')}")
        print(f"  {path.name:34s} {size / 1024 / 1024:6.1f} MB uploaded")

    status, release = call("GET", f"{API}/repos/{args.repo}/releases/tags/{args.tag}", auth)
    print(f"\n{release['html_url']}")
    for asset in release.get("assets", []):
        print(f"  {asset['name']:34s} {asset['size'] / 1024 / 1024:6.1f} MB  {asset['browser_download_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
