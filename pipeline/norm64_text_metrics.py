"""Replay the archived Norm64 text metrics with sacrebleu==2.5.1."""

import argparse
import csv
import re
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from sacrebleu.metrics import BLEU, CHRF

if __package__:
    from .score_norm64 import load_cases, validate
else:
    from score_norm64 import load_cases, validate


CASES = Path(__file__).resolve().parents[1] / "results" / "norm64_cases.jsonl"
SYSTEMS = ("r4_spoken", "r4_normalized", "nllb_base_spoken", "nllb_base", "opus", "gemma_1b")
AM_PM = re.compile(
    r"(?<![\w:.])(?P<h>1[0-2]|[1-9]):(?P<m>[0-5]\d)[ \t]*"
    r"(?P<ap>[ap])\.?m(?P<dot>\.)?(?![\w.])",
    re.IGNORECASE,
)
CLOCK = re.compile(r"(?<!\d)\d{1,2}:[0-5]\d(?!\d)")


def canonicalize_prediction(text):
    """Use the archived notation diagnostic on a single explicit AM/PM clock."""
    matches = list(AM_PM.finditer(text))
    if len(matches) != 1 or len(CLOCK.findall(text)) != 1:
        return text
    match = matches[0]
    hour = int(match["h"]) % 12 + (12 if match["ap"].lower() == "p" else 0)
    suffix = text[match.end():]
    following = suffix.lstrip()
    # An abbreviation's final period can also end the sentence.
    period = "." if match["dot"] and (not following or following[0].isupper()) else ""
    return text[:match.start()] + f"{hour:02d}:{match['m']}" + period + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    args = parser.parse_args()
    if version("sacrebleu") != "2.5.1":
        parser.error("replay requires sacrebleu==2.5.1")
    cases = load_cases(args.cases)
    validate(cases)
    cases.sort(key=lambda row: row["id"])
    bleu = BLEU(tokenize="13a")
    chrf = CHRF(word_order=2)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("group", "messages", "system", "prediction_notation", "bleu", "chrf_pp", "mean_comet"))
    groups = {"primary": ("pl", "hr", "ru"), **{lang: (lang,) for lang in ("pl", "hr", "ru", "uk")}}
    for group, languages in groups.items():
        rows = [row for row in cases if row["lang"] in languages]
        references = [[row["reference"] for row in rows]]
        for system in SYSTEMS:
            outputs = [row["outputs"][system] for row in rows]
            translations = [out["text"] for out in outputs]
            conditions = {
                "original": translations,
                "single_clock_HH:MM": [canonicalize_prediction(t) for t in translations],
            }
            for notation, texts in conditions.items():
                # The recorded COMET scores are for the original model outputs.
                comet = ""
                if notation == "original" and system in ("r4_spoken", "r4_normalized"):
                    comet = f"{mean(out['comet'] for out in outputs):.8f}"
                writer.writerow((
                    group, len(rows), system, notation,
                    f"{bleu.corpus_score(texts, references).score:.8f}",
                    f"{chrf.corpus_score(texts, references).score:.8f}", comet,
                ))


if __name__ == "__main__":
    main()
