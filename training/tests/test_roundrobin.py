"""Tests for the round-robin pair selection: which games a table run will play.

Adding new models to an existing table means naming the pairs explicitly, and a
mistyped name would otherwise cost an hour of GPU on the wrong pair -- or, worse,
put a result in a row nobody reads.  The selection is pure arithmetic over names,
so it is tested here instead of by watching a match run.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roundrobin  # noqa: E402

failures = 0


def check(condition, what):
    global failures
    if condition:
        return
    print(f"FAIL {what}")
    failures += 1


def make_files(directory: str, names, size=4):
    for name in names:
        with open(os.path.join(directory, f"{name}.bin"), "wb") as handle:
            handle.write(b"\0" * size)


def args(**overrides):
    base = dict(pair=[], ladder=None, random_only=False, random_anchor=False,
                plan=None, games=20)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_duplicate_names_stop_the_run():
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        make_files(one, ["a", "b"])
        make_files(two, ["a"], size=8)  # same name, different file
        try:
            roundrobin.model_list([roundrobin.Path(one), roundrobin.Path(two)], "", "")
        except SystemExit as stop:
            check("both named a" in str(stop), f"message should name the clash, got {stop}")
        else:
            check(False, "two different models with one name must stop, not merge rows")
        # Same name, same size: a copy.  It must still resolve to one entry.
        make_files(two, [], size=4)
        make_files(two, ["a"], size=4)
        index = roundrobin.model_list([roundrobin.Path(one), roundrobin.Path(two)], "", "")
        check(len(index) == 2, f"a copied model is one row, got {len(index)}")


def test_pair_specs_are_resolved_and_ordered():
    with tempfile.TemporaryDirectory() as one:
        make_files(one, ["strong", "weak"])
        index = roundrobin.model_list([roundrobin.Path(one)], "", "")
        tasks = roundrobin.tasks_for(
            index, args(pair=["weak:strong", "strong:random"]))
        check([os.path.basename(str(a)) if a != "random" else a for a, _, _n in tasks]
              == ["weak.bin", "strong.bin"], f"pairs keep the order given, got {tasks}")
        check(tasks[1][1] == "random", "random is passed through to the binary")
        check(roundrobin.tasks_for(index, args(random_only=True))[0][1] == "random",
              "--random-only still plays every model against random")


def test_bad_pair_specs_are_rejected():
    with tempfile.TemporaryDirectory() as one:
        make_files(one, ["a", "b"])
        index = roundrobin.model_list([roundrobin.Path(one)], "", "")
        for spec, why in (("a-b", "separator must be a colon"),
                          ("a:", "the second name is missing"),
                          (":b", "the first name is missing"),
                          ("a:nope", "an unknown name must not be skipped")):
            try:
                roundrobin.tasks_for(index, args(pair=[spec]))
            except SystemExit:
                pass
            else:
                check(False, f"--pair {spec!r} must stop: {why}")
        try:
            roundrobin.tasks_for(index, args(pair=["a:b"], ladder="ladder.txt"))
        except SystemExit:
            pass
        else:
            check(False, "--pair together with --ladder is ambiguous and must stop")


def test_ladder_plays_adjacent_rungs_only():
    with tempfile.TemporaryDirectory() as one:
        make_files(one, ["w", "m", "s"])
        index = roundrobin.model_list([roundrobin.Path(one)], "", "")
        ladder = os.path.join(one, "ladder.txt")
        with open(ladder, "w", encoding="utf-8") as handle:
            handle.write("# weakest first\nw\nm\ns\n")
        tasks = roundrobin.tasks_for(index, args(ladder=ladder))
        got = [(os.path.basename(str(a)), os.path.basename(str(b))) for a, b, _n in tasks]
        check(got == [("w.bin", "m.bin"), ("m.bin", "s.bin")],
              f"a ladder is a chain, not a round robin, got {got}")
        anchored = roundrobin.tasks_for(index, args(ladder=ladder, random_anchor=True))
        check(len(anchored) == 3 and anchored[-1][1] == "random",
              "the random anchor hangs off the weakest rung only")


def main():
    test_duplicate_names_stop_the_run()
    test_pair_specs_are_resolved_and_ordered()
    test_bad_pair_specs_are_rejected()
    test_ladder_plays_adjacent_rungs_only()
    if failures:
        print(f"{failures} failure(s)")
        return 1
    print("roundrobin: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
