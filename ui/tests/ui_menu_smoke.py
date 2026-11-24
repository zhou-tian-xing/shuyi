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
  hand-made fixtures;
* no two entries carry the same label: the build tree holds a copy of the
  fixtures, and listing it produced the same model twice;
* a model whose header says `nopass` (2) is listed as such.  `readModelRuleset`
  used to collapse every non-blocking byte onto classic, which hid a nopass model
  under the nopass rule set -- the same way the v4 header bug emptied the menu.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ID_MODEL_BASE = 6100
ID_MODEL_LIMIT = 6400
ID_MODEL_RESCAN = 6090
TAGS = ("[现行]", "[阻挡]", "[无过手]", "[未知]")


def dump(binary, out):
    proc = subprocess.run([str(binary), "--dump-models", str(out)],
                          cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"dump failed rc={proc.returncode} {proc.stdout} {proc.stderr}")
    lines = out.read_text(encoding="utf-8").splitlines()
    out.unlink()
    return lines


def entries_of(lines):
    header = re.fullmatch(r"models=(\d+)", lines[0])
    if not header:
        raise RuntimeError(f"bad header: {lines[0]!r}")
    entries = [line.split("\t", 1) for line in lines[1:] if "\t" in line]
    models = [(int(i), label) for i, label in entries if label and int(i) >= ID_MODEL_BASE]
    return int(header.group(1)), models


def check_first_dump(binary, problems):
    out = (REPO / "build" / "ui_models_dump.txt").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = dump(binary, out)
    count, model_entries = entries_of(lines)

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
    entries = [line.split("\t", 1) for line in lines[1:] if "\t" in line]
    if not any(ident.isdigit() and int(ident) == ID_MODEL_RESCAN for ident, _ in entries):
        problems.append("missing the rescan entry")

    labels = [label for _, label in model_entries]
    if not any(label.rstrip().endswith("[阻挡]") for label in labels):
        problems.append("no blocking model in the menu (the header byte should tag it)")
    if not any(label.rstrip().endswith("[现行]") for label in labels):
        problems.append("no classic model in the menu")
    if len(labels) != len(set(labels)):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        problems.append(f"the same model is listed more than once: {duplicates}")

    offered = {label.split("  [")[0].split("/")[-1] for label in labels}
    for fixture in sorted((REPO / "models").glob("*.bin")):
        if fixture.name not in offered:
            problems.append(f"models/{fixture.name} is not offered in the menu")


def check_nopass_tag(binary, problems):
    """One temporary copy of a blocking model with ruleset byte 2 must be tagged."""
    sources = sorted((REPO / "models").glob("b*.bin"))
    if not sources:
        problems.append("no blocking model to build the nopass probe from")
        return
    probe = REPO / "models" / "zz-nopass-probe.bin"
    data = bytearray(sources[0].read_bytes())
    if len(data) < 36:
        problems.append(f"{sources[0].name} is too short to be a model")
        return
    data[32] = 2
    try:
        probe.write_bytes(bytes(data))
        out = (REPO / "build" / "ui_models_nopass.txt").resolve()
        _, model_entries = entries_of(dump(binary, out))
    finally:
        probe.unlink(missing_ok=True)
    matches = [label for _, label in model_entries if label.rstrip().endswith("[无过手]")]
    if not matches:
        problems.append("a model with ruleset byte 2 is not tagged [无过手]")
    elif not any("zz-nopass-probe.bin" in label for label in matches):
        problems.append(f"the nopass probe is not offered: {matches}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    binary = Path(argv[1])
    if not binary.exists():
        print(f"missing {binary}")
        return 1

    problems = []
    try:
        check_first_dump(binary, problems)
        check_nopass_tag(binary, problems)
    except RuntimeError as error:
        print(f"FAIL {error}")
        return 1

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("ok: models listed once from every scanned root, tags and ids consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
