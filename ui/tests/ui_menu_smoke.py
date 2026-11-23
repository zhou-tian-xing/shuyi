#!/usr/bin/env python3
"""Regression check for the UI's "MCTS + network" model submenu.

`shuyi.exe --dump-models FILE` builds the submenu exactly as the window does and
writes what it produced, so the menu can be checked without a GUI session:

    python ui/tests/ui_menu_smoke.py <path to shuyi.exe>

What is asserted:

* the dump is non-empty and its header count matches the listed entries;
* ids are sequential from ID_MODEL_BASE, and the rescan entry is present;
* every entry carries a rule-set tag (read from the model header, not guessed
  from the file name);
* both rule sets show up, because the repository ships classic and blocking
  models -- this is the regression that made the menu look empty;
* every *.bin under models/ is offered, so the window cannot silently hide
  hand-made fixtures.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ID_MODEL_BASE = 6100
ID_MODEL_LIMIT = 6400
ID_MODEL_RESCAN = 6090
TAGS = ("[现行]", "[阻挡]", "[未知]")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1])
    if not binary.exists():
        print(f"missing {binary}")
        return 1

    # Absolute path with a Chinese component on purpose: the UI has to decode it.
    out = (REPO / "build" / "ui_models_dump.txt").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(binary), "--dump-models", str(out)],
                          cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        print(f"dump failed rc={proc.returncode} {proc.stdout} {proc.stderr}")
        return 1
    lines = out.read_text(encoding="utf-8").splitlines()
    out.unlink(missing_ok=True)

    problems = []
    header = re.fullmatch(r"models=(\d+)", lines[0])
    if not header:
        print(f"bad header: {lines[0]!r}")
        return 1
    count = int(header.group(1))
    entries = [line.split("\t", 1) for line in lines[1:]]
    model_entries = [(int(i), label) for i, label in entries if label and int(i) >= ID_MODEL_BASE]

    if count != len(model_entries):
        problems.append(f"header says {count} models, listed {len(model_entries)}")
    if count == 0:
        problems.append("the submenu is empty")
    for index, (ident, label) in enumerate(model_entries):
        if ident != ID_MODEL_BASE + index:
            problems.append(f"entry {index} has id {ident}, expected {ID_MODEL_BASE + index}")
        if ident >= ID_MODEL_LIMIT:
            problems.append(f"id {ident} collides with the model limit")
        if not label.rstrip().endswith(TAGS):
            problems.append(f"entry {label!r} has no rule-set tag")
    if not any(int(ident) == ID_MODEL_RESCAN for ident, _ in entries if ident.isdigit()):
        problems.append("missing the rescan entry")

    labels = [label for _, label in model_entries]
    if not any(label.rstrip().endswith("[阻挡]") for label in labels):
        problems.append("no blocking model in the menu (the header byte should tag it)")
    if not any(label.rstrip().endswith("[现行]") for label in labels):
        problems.append("no classic model in the menu")

    offered = {label.split("  [")[0].split("/")[-1] for label in labels}
    for fixture in sorted((REPO / "models").glob("*.bin")):
        if fixture.name not in offered:
            problems.append(f"models/{fixture.name} is not offered in the menu")

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print(f"ok: {count} models listed from every scanned root, tags and ids consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
