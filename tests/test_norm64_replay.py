"""Regression tests for the archived six runs and their blinded clock readings."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from pipeline.clock_check import check
from pipeline.clock_normalize import normalize_clock
from pipeline.score_norm64 import load_cases, replay, validate


RESULTS = Path(__file__).resolve().parents[1] / "results"


class Norm64ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_cases(RESULTS / "norm64_cases.jsonl")
        cls.report = replay(cls.rows, check, normalize_clock)

    def test_run_manifest_and_outcomes(self):
        manifest = json.loads((RESULTS / "norm64_run.json").read_text(encoding="utf-8"))
        case_hash = hashlib.sha256((RESULTS / manifest["case_file"]).read_bytes()).hexdigest()
        self.assertEqual(case_hash, manifest["case_file_sha256"])
        self.assertEqual(self.report["normalization_verified"], 64)
        # Exact messages and complete A/B scenarios in the primary and transfer panels.
        expected = {
            "nllb_base_spoken": (0, 0, 0, 0),
            "r4_spoken": (10, 0, 1, 0),
            "nllb_base": (21, 8, 10, 4),
            "r4_normalized": (46, 22, 16, 8),
            "opus": (5, 1, 3, 0),
            "gemma_1b": (39, 18, 15, 7),
        }
        self.assertEqual(set(manifest["systems"]), set(expected))
        for system, totals in expected.items():
            with self.subTest(system=system):
                primary = self.report["counts"]["primary_pl_hr_ru"][system]
                uk = self.report["counts"]["uk_separate"][system]
                self.assertEqual((primary["variants"], primary["pairs"], uk["variants"], uk["pairs"]),
                                 (48, 24, 16, 8))
                self.assertEqual((primary["exact_correct"], primary["full_pairs"],
                                  uk["exact_correct"], uk["full_pairs"]), totals)

    def test_guard_disagreements(self):
        counts = self.report["counts"]["primary_pl_hr_ru"]
        self.assertEqual(counts["r4_normalized"]["guard"], {"match": 46, "mismatch": 2})
        self.assertEqual(counts["gemma_1b"]["guard_match_blind_fail_ids"], ["R4H-PL-07-B"])
        self.assertEqual(counts["nllb_base"]["guard_flag_blind_pass_ids"], ["R4H-PL-08-B"])

    def test_corrupt_inputs_fail(self):
        rows = deepcopy(self.rows)
        rows[0]["expected_24h"] = "23:30" if rows[0]["expected_24h"] != "23:30" else "01:30"
        with self.assertRaises(ValueError):
            replay(rows, check, normalize_clock)
        with self.assertRaisesRegex(ValueError, "Duplicate case ID"):
            validate(self.rows[1:] + [self.rows[-1]])


if __name__ == "__main__":
    unittest.main()
