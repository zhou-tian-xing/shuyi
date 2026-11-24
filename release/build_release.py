#!/usr/bin/env python3
"""One command: build both products, run the tests, stage the package.

The package is the two binaries plus the documents, and **no network**: weights
are data, they change far more often than the binaries, and a model file is the
one thing a user is expected to download.  `--model` ships one anyway.

    python release/build_release.py                     # CPU build, no model
    python release/build_release.py --cuda-build build-cuda
    python release/build_release.py --cuda-build build-cuda --model models/b4c32-s42012-d595499.bin

What it refuses to do:
* ship a UI without an engine next to it (the pair is the product);
* skip the tests (a release that fails `ctest` is not a release);
* fall back to a silent CPU-only build when a CUDA build was asked for.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd, cwd=None):
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.run([str(part) for part in cmd], cwd=str(cwd or REPO))
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd[0]}")


def configure_and_build(build_dir: Path, cuda: bool, tools: bool) -> None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        cmd = ["cmake", "-S", str(REPO), "-B", str(build_dir),
               "-DCMAKE_BUILD_TYPE=Release"]
        if cuda:
            cmd.append("-DSHUYI_CUDA=ON")
        if tools:
            cmd.append("-DSHUYI_BUILD_TOOLS=ON")
        run(cmd)
    run(["cmake", "--build", str(build_dir), "--config", "Release"])


def binary_optional(build_dir: Path, name: str) -> Path | None:
    for candidate in (build_dir / f"{name}.exe", build_dir / name):
        if candidate.exists():
            return candidate
    return None


def binary(build_dir: Path, name: str) -> Path:
    for candidate in (build_dir / f"{name}.exe", build_dir / name):
        if candidate.exists():
            return candidate
    raise SystemExit(f"missing {name} in {build_dir} (build failed?)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build", help="CPU/tooling build directory")
    parser.add_argument("--cuda-build", default="",
                        help="CUDA build directory whose binaries are shipped (optional)")
    parser.add_argument("--build-cuda", action="store_true",
                        help="also build that directory.  Off by default because a CUDA "
                             "build on Windows needs vcvars64 first (see "
                             "training/setup_ubuntu.sh for Linux); passing it from a plain "
                             "shell fails with 'cannot open cstdint'")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)

    build = (REPO / args.build).resolve() if not Path(args.build).is_absolute() else Path(args.build)
    cuda_build = None
    if args.cuda_build:
        cuda_build = (REPO / args.cuda_build).resolve() \
            if not Path(args.cuda_build).is_absolute() else Path(args.cuda_build)
        if not cuda_build.exists():
            raise SystemExit(f"--cuda-build does not exist: {cuda_build}")

    if not args.skip_build:
        configure_and_build(build, cuda=False, tools=True)
        if cuda_build is not None and args.build_cuda:
            configure_and_build(cuda_build, cuda=True, tools=True)

    if not args.skip_tests:
        run(["ctest", "--test-dir", str(build), "--output-on-failure"])

    # The shipped engine is the CUDA one when it was built: same code, faster
    # forward, and the package is what a player runs.
    engine_source = cuda_build if cuda_build is not None else build
    engine = binary(engine_source, "shuyi_engine")
    # One toolchain per package: the CUDA tree builds the UI too on Windows, and a
    # UI from the same compiler as the engine means one runtime, not two.
    ui_source = build
    if cuda_build is not None and binary_optional(cuda_build, "shuyi"):
        ui_source = cuda_build
    ui = binary(ui_source, "shuyi")
    if not engine.exists() or not ui.exists():
        raise SystemExit("both products are required")

    stage_cmd = [sys.executable, str(REPO / "release" / "make_alpha.py"),
                 "--engine", str(engine), "--ui", str(ui), "--out", str(args.out)]
    if args.model:
        stage_cmd += ["--model", str(args.model)]
    run(stage_cmd)

    archive = args.out / "shuyi-alpha.zip"
    print(f"\npackage: {args.out / 'shuyi-alpha'}")
    print(f"archive: {archive}  ({archive.stat().st_size / 1024:.0f} KB)")
    if not args.model:
        print("no model shipped: drop a b<blocks>c<channels>-s<steps>-d<data>.bin "
              "into models/ next to shuyi.exe (see README.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
