"""Embed the archived Norm64 results in the offline dashboard.

Run from the repository root with python3 pipeline/build_norm64_dashboard.py.
Pass --check to verify the embedded payload without writing the file.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

if __package__:
    from .clock_check import check
    from .clock_normalize import normalize_clock
    from .score_norm64 import ARMS, GROUPS, NORMALIZED_ARMS, correct, replay
else:
    from clock_check import check
    from clock_normalize import normalize_clock
    from score_norm64 import ARMS, GROUPS, NORMALIZED_ARMS, correct, replay


ROOT = Path(__file__).resolve().parents[1]
GROUP_INFO = {
    "primary": ("primary_pl_hr_ru", "Polish, Croatian and Russian"),
    "pl": ("pl", "Polish"),
    "hr": ("hr", "Croatian"),
    "ru": ("ru", "Russian"),
    "uk": ("uk_separate", "Ukrainian"),
}
OPEN_TAG = '<script id="norm64-data" type="application/json">'
DATA_ELEMENT = re.compile(re.escape(OPEN_TAG) + r"([^<]*)</script>")
NOTATIONS = {"original", "single_clock_HH:MM"}


def load_metrics(path, rows):
    """Require one row per cohort, system and notation with the correct denominator."""
    expected = {(group, arm, notation) for group in GROUP_INFO for arm in ARMS for notation in NOTATIONS}
    cohorts = {
        group: [row for row in rows if row["lang"] in GROUPS[scorer_group]]
        for group, (scorer_group, _) in GROUP_INFO.items()
    }
    metrics = []
    with path.open(encoding="utf-8", newline="") as file:
        for record in csv.DictReader(file):
            group, arm, notation = key = (record["group"], record["system"], record["prediction_notation"])
            if key not in expected:
                raise ValueError(f"Duplicate or unexpected metric key: {key}")
            expected.remove(key)
            if int(record["messages"]) != len(cohorts[group]):
                raise ValueError(f"Wrong number of messages for {key}")
            scores = {name: float(record[name]) for name in ("bleu", "chrf_pp")}
            if not all(math.isfinite(value) and 0 <= value <= 100 for value in scores.values()):
                raise ValueError(f"Invalid corpus score for {key}")
            comet = record["mean_comet"]
            if notation == "original" and arm in ("r4_spoken", "r4_normalized"):
                saved = mean(row["outputs"][arm]["comet"] for row in cohorts[group])
                if comet != f"{saved:.8f}":
                    raise ValueError(f"COMET mean differs from archived readings for {key}")
            elif comet != "":
                raise ValueError(f"Unexpected COMET score for {key}")
            metrics.append({
                "group": group, "system": arm, "prediction_notation": notation,
                **scores, "mean_comet": float(comet) if comet else None,
            })
    if expected:
        raise ValueError(f"Missing metric rows: {sorted(expected)}")
    return metrics


def build_payload(results=ROOT / "results"):
    """Bind output text, labels, numeric checks and scores to the same dataset release."""
    manifest = json.loads((results / "norm64_run.json").read_bytes())
    path = results / "norm64_cases.jsonl"
    raw = path.read_bytes()
    if (manifest["panel"] != "Norm64" or manifest["case_file"] != path.name
            or hashlib.sha256(raw).hexdigest() != manifest["case_file_sha256"]):
        raise ValueError("Dashboard case file differs from the Norm64 run manifest")
    checker = Path(__file__).with_name("clock_check.py").read_bytes()
    if hashlib.sha256(checker).hexdigest() != manifest["clock_check_sha256"]:
        raise ValueError("Clock checker differs from the frozen run")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    counts = replay(rows, check, normalize_clock)["counts"]
    if (set(manifest["systems"]) != set(ARMS) or len(rows) != manifest["cases"]
            or len({row["pair_id"] for row in rows}) != manifest["scenarios"]):
        raise ValueError("System or cohort identity differs from the run manifest")

    groups = {}
    for group, (scorer_group, title) in GROUP_INFO.items():
        groups[group] = {"title": title, "arms": {
            arm: {key: total[key] for key in ("pairs", "full_pairs", "variants", "exact_correct")}
            for arm, total in counts[scorer_group].items()
        }}

    cases = []
    for row in rows:
        case = {key: row[key] for key in (
            "id", "pair_id", "lang", "variant", "spoken_source", "normalized_input",
            "reference", "expected_24h",
        )}
        outputs = {}
        for arm, output in row["outputs"].items():
            guard = None
            if arm in NORMALIZED_ARMS:
                guard = check(row["normalized_input"], output["text"])["status"]
            outputs[arm] = {
                "text": output["text"], "label": output["label"],
                "correct": correct(row, arm), "guard": guard,
            }
        cases.append({**case, "outputs": outputs})
    return {"groups": groups, "cases": cases,
            "metrics": load_metrics(results / "norm64_text_metrics.csv", rows)}


def embed(html, payload):
    """Replace exactly one inert JSON element while preserving the surrounding HTML."""
    matches = list(DATA_ELEMENT.finditer(html))
    if html.count(OPEN_TAG) != 1 or len(matches) != 1:
        raise ValueError("Expected exactly one complete Norm64 JSON script element")
    # Escaping < prevents a translation from closing the HTML script element.
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    start, end = matches[0].span(1)
    return html[:start] + data.replace("<", r"\u003c") + html[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    args = parser.parse_args()
    target = ROOT / "docs" / "index.html"
    try:
        current = target.read_bytes().decode("utf-8")
        updated = embed(current, build_payload())
        if current == updated:
            print("Dashboard data matches the frozen Norm64 results.")
        elif args.check:
            parser.exit(1, "Dashboard data is stale. Run this script without --check.\n")
        else:
            target.write_bytes(updated.encode("utf-8"))
            print("Updated the embedded Norm64 results in docs/index.html.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Dashboard build failed: {error}\n")


if __name__ == "__main__":
    main()
