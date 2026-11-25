"""Regressions for rating isolation and fixed-reference plateau detection."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elo import merge_reports
from evalconfig import board_directory
from leaderboard_import import import_all
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


if __name__ == "__main__":
    unittest.main()
