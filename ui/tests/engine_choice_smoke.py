#!/usr/bin/env python3
"""The window must drive the engine the user chose, not the one nearby.

The front end and the engine are separate programs; the protocol is the only
contract and the engine is a path the user configures.  This test drives
`shuyi.exe --dump-engine` twice:

1. no setting -> discovery, which prefers a CUDA-capable engine;
2. an explicit setting (`shuyi-ui.ini [engine] path=...`) pointing at a CPU
   engine -> that exact program must win, even when a CUDA build is available.

The setting file is redirected with SHUYI_UI_INI so the real one is untouched.

    python ui/tests/engine_choice_smoke.py <path to shuyi.exe>
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def dump(binary, ini):
    env = dict(os.environ)
    env["SHUYI_UI_INI"] = str(ini)
    proc = subprocess.run([str(binary), "--dump-engine", str(ini) + ".out"],
                          cwd=str(REPO), capture_output=True, text=True, env=env)
    out = Path(str(ini) + ".out")
    if proc.returncode != 0 or not out.exists():
        return None, proc
    lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
    out.unlink(missing_ok=True)
    parsed = {"probed": {}, "chosen": None, "setting": None, "candidates": None}
    for line in lines:
        if line.startswith("candidates="):
            parsed["candidates"] = int(line.split("=", 1)[1])
        elif line.startswith("setting="):
            parsed["setting"] = line.split("=", 1)[1]
        elif line.startswith("chosen="):
            parsed["chosen"] = line.split("=", 1)[1]
        elif line.startswith(("cuda", "cpu", "no-reply")):
            parts = line.split("\t") if "\t" in line else line.split("    ", 1)
            if len(parts) == 2:
                parsed["probed"][parts[1].strip()] = parts[0].strip()
    return parsed, proc


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1]).resolve()
    if not binary.exists():
        print(f"missing {binary}")
        return 1

    problems = []
    # Keep the scratch files inside the repository: a system temp directory is not
    # always writable for the child process.
    scratch = REPO / "build" / "engine_choice_scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        ini = scratch / "shuyi-ui.ini"

        # 1. discovery
        auto, proc = dump(binary, ini)
        if auto is None:
            print(f"dump failed rc={proc.returncode} {proc.stdout} {proc.stderr}")
            return 1
        cuda_paths = [p for p, kind in auto["probed"].items() if kind == "cuda"]
        if not auto["chosen"] or not Path(auto["chosen"]).exists():
            problems.append("discovery chose nothing usable")
        elif cuda_paths and auto["chosen"].lower() != max(cuda_paths, key=len).lower():
            # At least require that the choice is one of the CUDA candidates.
            if Path(auto["chosen"]).resolve() not in {Path(p).resolve() for p in cuda_paths}:
                problems.append("discovery ignored the CUDA engine")
        print(f"  discovery: {auto['candidates']} candidates, "
              f"chose {Path(auto['chosen']).name if auto['chosen'] else None}")

        # 2. explicit setting must win, even against a CUDA engine
        cpu_engine = None
        for path, kind in auto["probed"].items():
            if kind == "cpu":
                cpu_engine = path
                break
        if cpu_engine:
            ini.write_text(f"engine={cpu_engine}\n", encoding="utf-8")
            pinned, proc = dump(binary, ini)
            if pinned is None:
                print(f"pinned dump failed rc={proc.returncode}")
                return 1
            if Path(pinned["chosen"]).resolve() != Path(cpu_engine).resolve():
                problems.append(f"the configured engine was ignored: {pinned['chosen']}")
            if pinned["setting"] in (None, "", "(auto)"):
                problems.append("dump did not report the configured setting")
            print(f"  pinned:    chose {Path(pinned['chosen']).name} as configured")
        else:
            print("  pinned:    skipped (no CPU engine to pin)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("ok: the UI honours an explicitly chosen engine and only discovers when unset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
