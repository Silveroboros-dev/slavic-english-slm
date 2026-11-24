"""Validate the bindings in the offline clock dashboard."""

import json
from pathlib import Path
import tempfile
import unittest

from pipeline.build_norm64_dashboard import DATA_ELEMENT, OPEN_TAG, build_payload, embed, load_metrics
from pipeline.score_norm64 import load_cases


RESULTS = Path(__file__).resolve().parents[1] / "results"


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_cases(RESULTS / "norm64_cases.jsonl")
        cls.data = build_payload(RESULTS)

    def test_case_identity_and_independent_guard(self):
        archive = {row["id"]: row for row in self.rows}
        self.assertEqual(len(self.data["cases"]), len(archive))
        for case in self.data["cases"]:
            row = archive[case["id"]]
            for field in ("spoken_source", "normalized_input", "reference", "expected_24h", "pair_id", "variant"):
                self.assertEqual(case[field], row[field])
            for arm, output in case["outputs"].items():
                self.assertEqual(output["text"], row["outputs"][arm]["text"])
                self.assertEqual(output["label"], row["outputs"][arm]["label"])
        pl = next(case for case in self.data["cases"] if case["id"] == "R4H-PL-07-B")
        self.assertEqual(pl["outputs"]["gemma_1b"]["guard"], "match")
        self.assertFalse(pl["outputs"]["gemma_1b"]["correct"])
        self.assertIsNone(pl["outputs"]["r4_spoken"]["guard"])

    def test_panel_units_and_missing_metric(self):
        groups = self.data["groups"]
        self.assertEqual(groups["primary"]["arms"]["r4_normalized"], {
            "pairs": 24, "full_pairs": 22, "variants": 48, "exact_correct": 46,
        })
        self.assertEqual(groups["uk"]["arms"]["r4_normalized"]["pairs"], 8)
        self.assertEqual(len(self.data["metrics"]), 60)
        self.assertTrue(all(m["mean_comet"] is None for m in self.data["metrics"]
                            if m["prediction_notation"] == "single_clock_HH:MM"))

    def test_rejects_wrong_manifest_and_incomplete_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "norm64_run.json").write_bytes((RESULTS / "norm64_run.json").read_bytes())
            (directory / "norm64_cases.jsonl").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                build_payload(directory)
            metrics = directory / "norm64_text_metrics.csv"
            original = (RESULTS / metrics.name).read_text(encoding="utf-8").splitlines()
            metrics.write_text("\n".join(original[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing metric rows"):
                load_metrics(metrics, self.rows)
            metrics.write_text("\n".join([*original, original[-1]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_metrics(metrics, self.rows)

    def test_template_integrity_and_round_trip(self):
        element = OPEN_TAG + "{}</script>"
        template = "before\r\n" + element + "\nafter"
        payload = {"translation": "Dosłownie </script><b>пів</b>"}
        updated = embed(template, payload)
        self.assertEqual(updated.count("</script>"), 1)
        self.assertEqual(json.loads(DATA_ELEMENT.search(updated).group(1)), payload)
        self.assertEqual(embed(updated, payload), updated)
        self.assertTrue(updated.startswith("before\r\n" + OPEN_TAG))
        self.assertTrue(updated.endswith("</script>\nafter"))
        for malformed in ("<html></html>", element + element, OPEN_TAG + "{}<script></script>"):
            with self.subTest(template=malformed), self.assertRaises(ValueError):
                embed(malformed, payload)

    def test_shipped_dashboard_uses_current_records(self):
        dashboard = (RESULTS.parent / "docs" / "index.html").read_text(encoding="utf-8")
        self.assertEqual(embed(dashboard, self.data), dashboard)
