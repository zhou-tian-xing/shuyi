#!/usr/bin/env python3
"""One command: build both products, run the tests, stage the package.

The package contains the CPU UI and engine, documentation, and optionally
an additional CUDA engine and a model file.

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
    if not cache.exists() or not cuda:
        source = REPO
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                    source = Path(line.partition("=")[2])
                    break
        cmd = ["cmake", "-S", str(source), "-B", str(build_dir),
               "-DCMAKE_BUILD_TYPE=Release"]
        cmd.append(f"-DSHUYI_CUDA={'ON' if cuda else 'OFF'}")
        if sys.platform == "win32" and not cuda:
            cmd.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>")
        if tools:
            cmd.append("-DSHUYI_BUILD_TOOLS=ON")
        run(cmd)
    run(["cmake", "--build", str(build_dir), "--config", "Release"])


def binary(build_dir: Path, name: str) -> Path:
    for directory in (build_dir / "Release", build_dir):
        for filename in (f"{name}.exe", name):
            candidate = directory / filename
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
        cuda_build = REPO / args.cuda_build
        if cuda_build.resolve() == build.resolve():
            parser.error("--build and --cuda-build require distinct build directories")
        if not cuda_build.exists() and (not args.build_cuda or args.skip_build):
            raise SystemExit(f"--cuda-build does not exist: {cuda_build}")

    if not args.skip_build:
        configure_and_build(build, cuda=False, tools=True)
        if cuda_build is not None and args.build_cuda:
            configure_and_build(cuda_build, cuda=True, tools=True)

    if not args.skip_tests:
        for test_build in filter(None, (build, cuda_build)):
            run(["ctest", "--test-dir", str(test_build), "-C", "Release", "--output-on-failure"])

    engine = binary(build, "shuyi_engine")
    ui = binary(build, "shuyi")
    stage_cmd = [sys.executable, str(REPO / "release" / "make_alpha.py"),
                 "--engine", str(engine), "--ui", str(ui), "--out", str(args.out)]
    if cuda_build:
        stage_cmd += ["--cuda-engine", str(binary(cuda_build, "shuyi_engine"))]
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
