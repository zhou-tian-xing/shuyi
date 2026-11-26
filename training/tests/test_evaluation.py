"""Regressions for rating isolation and fixed-reference plateau detection."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elo import elo_from_score, fit_elo, fit_elo_checked, merge_reports
from evalconfig import board_directory
from leaderboard_import import append_into, import_all
from leaderboard import read_table
from pipeline import plateau_report


class EvaluationTest(unittest.TestCase):
    def report(self, root, name, **changes):
        data = {
            "a": "pruning", "b": "random", "a_wins": 8, "b_wins": 2,
            "ruleset": "nopass", "komi": 0.5, "visits": 64, "visits_b": 64,
        }
        data.update(changes)
        path = root / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_import_keeps_settings_separate_and_replaces_same_source(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "reports"
            board = Path(temp) / "boards"
            runs.mkdir()
            self.report(runs, "nopass")
            self.report(runs, "blocking", ruleset="blocking", komi=1.26, a_wins=4, b_wins=6)
            self.report(runs, "other_komi", komi=1.5)
            self.report(runs, "other_visits", visits=128, visits_b=128)
            self.report(runs, "budget_mismatch", visits_b=128)
            self.report(runs, "unknown", ruleset=None)
            self.report(runs, "no_budget_b", visits_b=None)
            import_all(board, [runs])
            self.assertEqual(import_all(board, [runs]), {})
            self.report(runs, "nopass", a_wins=9, b_wins=3)
            import_all(board, [runs])
            for ruleset, komi, visits, expected in [
                ("nopass", 0.5, 64, (9, 0, 3)),
                ("nopass", 0.5, 128, (8, 0, 2)),
                ("nopass", 1.5, 64, (8, 0, 2)),
                ("blocking", 1.26, 64, (4, 0, 6)),
            ]:
                path = board_directory(board, ruleset, komi) / f"matches_{visits}.csv"
                _, cells = read_table(path)
                self.assertEqual(cells["pruning", "random"], expected)
            self.assertFalse((board / "matches_64.csv").exists())

    def test_elo_merge_validates_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = {"names": ["pruning", "random"], "pairs": [["pruning", "random", 8, 0, 2]]}
            original = self.report(root, "original", **matrix)
            compatible = self.report(root, "compatible", **matrix)
            legacy = self.report(root, "legacy", **matrix, ruleset=None)
            self.assertEqual(merge_reports([legacy])[2], (None, 0.5, 64))
            _, pairs, config = merge_reports([original, compatible])
            self.assertEqual(pairs[0][2:5], (16, 0, 4))
            self.assertEqual(config, ("nopass", 0.5, 64))
            for mismatch in ({"ruleset": "blocking"}, {"komi": 1.5}, {"visits_b": 128},
                             {"visits": 128, "visits_b": 128}, {"ruleset": None}):
                with self.subTest(mismatch=mismatch), self.assertRaises(SystemExit):
                    merge_reports([original, self.report(root, "wrong", **matrix, **mismatch)])

    def test_plateau_counts_only_comparable_results_and_resets_on_progress(self):
        args = SimpleNamespace(plateau_generations=10, plateau_margin=20,
                               capacity_ladder="b6c96,b8c128", blocks=6, channels=96)

        def record(gen, elo, ref="fixed"):
            return {"gen": gen, "arena": {"strength_reference": {"sha256": ref},
                                          "strength_elo": elo}}

        records = [record(0, 80)] + [record(gen, 99) for gen in range(1, 11)]
        self.assertFalse(plateau_report(records[:-1], args)["review"])
        self.assertTrue(plateau_report(records, args)["review"])
        self.assertFalse(plateau_report([*records, record(11, 100)], args)["review"])
        self.assertFalse(plateau_report([*records, record(11, 30, "new")], args)["review"])
        self.assertIsNone(plateau_report([*records, {"gen": 11, "elo": {"gen_011": 500}}], args))

        regressions = [record(gen, -50) for gen in range(10)]
        for item in regressions:
            item["arena"]["strength_reference"]["start_elo"] = 0
        self.assertTrue(plateau_report(regressions, args)["review"])


class AppendIntoTableTest(unittest.TestCase):
    """Adding new models to a table that already holds history.

    The table is the only record of games whose source files never carried
    settings tags, so it cannot be rebuilt -- it has to be extended in place,
    exactly once per source file.
    """

    def report(self, root, name, **changes):
        data = {
            "a": "pruning", "b": "mcts", "a_wins": 30, "b_wins": 20, "draws": 0,
            "ruleset": "nopass", "komi": 0.5, "visits": 64, "visits_b": 64,
        }
        data.update(changes)
        path = root / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def table(self, temp):
        path = Path(temp) / "matches_64.csv"
        path.write_text("model,random,pruning\n"
                        "random,—,10-0-40\n"
                        "pruning,40-0-10,—\n", encoding="utf-8")
        return path

    def test_adds_once_and_replaces_a_remeasured_source(self):
        with tempfile.TemporaryDirectory() as temp:
            table = self.table(temp)
            runs = Path(temp) / "reports"
            runs.mkdir()
            self.report(runs, "one")
            append_into(table, [runs])
            self.assertEqual(read_table(table)[1][("pruning", "mcts")], (30, 0, 20))
            append_into(table, [runs])  # same file again: not a second measurement
            self.assertEqual(read_table(table)[1][("pruning", "mcts")], (30, 0, 20))
            # A re-run of the same path replaces its own games instead of stacking them.
            self.report(runs, "one", a_wins=31, b_wins=19)
            append_into(table, [runs])
            self.assertEqual(read_table(table)[1][("pruning", "mcts")], (31, 0, 19))
            # History that came from neither source is left alone.
            self.assertEqual(read_table(table)[1][("random", "pruning")], (10, 0, 40))

    def test_a_source_for_another_budget_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            table = self.table(temp)
            runs = Path(temp) / "reports"
            runs.mkdir()
            self.report(runs, "big", visits=128, visits_b=128)
            with self.assertRaises(SystemExit):
                append_into(table, [runs])
            self.assertNotIn(("pruning", "mcts"), read_table(table)[1])

    def test_a_hand_edited_table_stops_instead_of_going_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            table = self.table(temp)
            runs = Path(temp) / "reports"
            runs.mkdir()
            self.report(runs, "one")
            append_into(table, [runs])
            text = table.read_text(encoding="utf-8").replace("30-0-20", "3-0-2")
            table.write_text(text, encoding="utf-8")
            self.report(runs, "one", a_wins=40, b_wins=10)
            with self.assertRaises(SystemExit):
                append_into(table, [runs])


class FitEloConvergenceTest(unittest.TestCase):
    """The fit has to finish, and it must not shrink what it measured.

    Both failures are silent: an unfinished Newton run and an over-strong prior
    both return a complete, plausible, too-small table.  A long chain whose every
    link is a known 60% is enough to catch both -- and the length matters: at
    twelve models a 60-sweep run is within 1% of the answer, at forty it is half.
    """

    def chain(self, links=40, games=300, wins=180):
        names = [f"m{i:02d}" for i in range(links + 1)]
        pairs = [(names[i + 1], names[i], wins, 0, games - wins, 1.0)
                 for i in range(links)]
        return names, pairs, links * elo_from_score(0.6)

    def test_a_long_chain_is_not_left_compressed(self):
        names, pairs, truth = self.chain()
        rating, moved = fit_elo_checked(names, pairs)
        self.assertLess(moved, 0.02, "fit reported an unfinished iteration")
        self.assertAlmostEqual(max(rating) - min(rating), truth, delta=0.05 * truth)
        # The old behaviour: a fixed 60 sweeps on the same data reads far smaller.
        stub = fit_elo(names, pairs, passes=60, tolerance=0.0)
        self.assertLess(max(stub) - min(stub), 0.7 * truth,
                        "a 60-sweep run is expected to be visibly compressed; if this "
                        "fails the convergence check above is not testing anything")

    def test_the_prior_does_not_shrink_measured_differences(self):
        names, pairs, truth = self.chain()

        def spread(**kwargs):
            rating = fit_elo(names, pairs, **kwargs)
            return max(rating) - min(rating)

        weak = spread(prior_sigma=2000.0)
        strong = spread(prior_sigma=400.0)
        self.assertAlmostEqual(weak, truth, delta=0.05 * truth)
        self.assertLess(strong, 0.9 * weak, "a 400-point prior must compress this pool, "
                                           "which is why the default is not 400")


if __name__ == "__main__":
    unittest.main()
