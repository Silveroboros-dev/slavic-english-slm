#!/usr/bin/env python3
"""Conservative, reference-free numeric clock preservation audit.

The source must already be normalized: exactly one zero-padded HH:MM in
24-hour notation. This module does not normalize spoken Slavic times. English
accepts H[:MM] a.m./p.m. (also am/pm), or exactly HH:MM as 24-hour notation.
Bare 00:xx through 09:xx and 13:xx through 23:xx are unambiguous; 10:xx
through 12:xx need an explicit 24-hour context. Bare 9:00 and 9 abstain.
The check concerns local clock values, not timezone conversion,
dates, scheduling intent, or translation quality. Missing, invalid, multiple,
and unsupported clock expressions return review, never mismatch.

Run: python clock_check.py input.json [--output audit.json]
Input is a JSON array of objects with id, lang, source, translation. IDs and
language are copied for traceability only; neither influences the decision.
No references, expected answers, labels, or external services are used.
"""

import argparse
import json
import re
import sys


# Capture entire numeric candidates before validating them, so malformed values
# such as 123:45 or 09:000 cannot yield a valid substring. Boundaries also reject
# clock fragments embedded in words. Colon-separated seconds remain one token.
_CANDIDATE = re.compile(
    r"(?<![\w:])\d+(?::\d+)+(?:\s*[ap](?:\.?m\.?))?(?![\w:])"
    r"|(?<![\w:])\d+\s*[ap](?:\.?m\.?)(?![\w:])",
    re.IGNORECASE,
)
_SOURCE = re.compile(r"([0-9]{2}):([0-9]{2})", re.ASCII)
_ENGLISH = re.compile(
    r"([0-9]{1,2})(?::([0-9]{2}))?\s*([ap])\.?m\.?", re.IGNORECASE | re.ASCII
)
_NON_NUMERIC = re.compile(r"\b(?:noon|midnight|o['’]clock)\b", re.IGNORECASE)
_EXPLICIT_24H = re.compile(r"\b24[ -]?(?:hour|hr|h)\b", re.IGNORECASE)
_ZONE = re.compile(
    r"\b(?:UTC|GMT|CET|CEST|EST|EDT|PST|PDT|timezone|time\s+zone)\b|[+-]\d{2}:\d{2}",
    re.IGNORECASE,
)


def extract(text, source=False):
    """Return normalized clock plus literal evidence, or an abstention reason."""
    if not isinstance(text, str):
        return {"time": None, "evidence": [], "reason": "text_required"}
    matches = list(_CANDIDATE.finditer(text))
    evidence = [{"text": m.group(), "start": m.start(), "end": m.end()} for m in matches]
    result = {"time": None, "evidence": evidence, "reason": None}
    if _ZONE.search(text):
        result["reason"] = "timezone_context_requires_review"
        return result
    if not matches:
        result["reason"] = "missing_numeric_clock"
        return result
    if len(matches) != 1:
        result["reason"] = "multiple_numeric_clocks"
        return result
    before, after = text[:matches[0].start()], text[matches[0].end():]
    if re.search(r"\d\s*[-–—/]\s*$", before) or re.match(r"\s*[-–—/]\s*\d", after):
        result["reason"] = "numeric_range_requires_review"
        return result
    # Numeric plus verbal alternatives are outside this bounded checker.
    if not source and _NON_NUMERIC.search(text):
        result["reason"] = "verbal_time_context_requires_review"
        return result
    token = matches[0].group()
    if re.match(r"\s*[ap]\.?(?:m|\.)", text[matches[0].end():], re.IGNORECASE):
        result["reason"] = "malformed_or_unparsed_meridiem"
        return result
    marker = None if source else _ENGLISH.fullmatch(token)
    if marker:
        if re.match(r"\.?\w", text[matches[0].end():]):
            result["reason"] = "malformed_or_unparsed_meridiem"
            return result
        hour, minute = int(marker[1]), int(marker[2] or "0")
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            result["reason"] = "invalid_12_hour_clock"
            return result
        hour = hour % 12 + (12 if marker[3].lower() == "p" else 0)
    else:
        clock = _SOURCE.fullmatch(token)
        if not clock:
            result["reason"] = "unsupported_or_ambiguous_numeric_clock"
            return result
        hour, minute = int(clock[1]), int(clock[2])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            result["reason"] = "invalid_24_hour_clock"
            return result
        if not source and 10 <= hour <= 12 and not _EXPLICIT_24H.search(text):
            result["reason"] = "unmarked_12_hour_ambiguity"
            return result
    if not source:
        # Only an immediately attached daypart qualifies the parsed clock.
        # A different sentence discussing the morning is unrelated evidence.
        local = re.match(r"\s+(?:in\s+the\s+)?(morning|afternoon|evening|night)\b", text[matches[0].end():], re.IGNORECASE)
        if local:
            part = local[1].lower()
            valid = hour < 12 if part == "morning" else hour >= 12
            if not valid:
                result["reason"] = "contradictory_local_daypart"
                return result
    result["time"] = f"{hour:02d}:{minute:02d}"
    return result


def check(source, translation):
    """Compare two independently parsed values, abstaining on either failure."""
    left, right = extract(source, source=True), extract(translation)
    if left["time"] is None or right["time"] is None:
        status = "review"
    else:
        status = "match" if left["time"] == right["time"] else "mismatch"
    return {"status": status, "source": left, "translation": right}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON input array; - reads stdin")
    parser.add_argument("--output", help="Write JSON here instead of stdout")
    args = parser.parse_args()
    try:
        if args.input == "-":
            records = json.load(sys.stdin)
        else:
            with open(args.input, encoding="utf-8") as handle:
                records = json.load(handle)
        if not isinstance(records, list) or any(
            not isinstance(row, dict) or not all(key in row for key in ("id", "lang", "source", "translation"))
            for row in records
        ):
            raise ValueError("input must be a JSON array of objects with id, lang, source, translation")
        audited = [dict(id=row["id"], lang=row["lang"], **check(row["source"], row["translation"])) for row in records]
        rendered = json.dumps(audited, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
