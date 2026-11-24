#!/usr/bin/env python3
"""Bounded source-only half-hour normalizer, Python 3.9+, stdlib only.

Lexicon and hour arithmetic come from the norm64 protocol's data/generate.py.
No target, reference or gold is read. The extracted rules reproduce all 64
archived source replacements. Added boundaries and abstentions protect against
partial matches and detected multiple clocks. Half-to-twelve with an evening
daypart abstains, rather than preserving the legacy 11:30 error outside that set.
This is a bounded lexical experiment, not a general Slavic time parser.
Unsupported language/forms and detected multiple clocks are left unchanged.

API: normalize_clock(lang, text) returns source text with one replacement.
CLI: python -m pipeline.clock_normalize input.json [--output normalized.json]
Input: array of objects with string lang and source. Other fields are preserved.
Use pipeline.clock_check.check for match/mismatch/review, parsed times,
literal evidence offsets, and abstention reasons (see its module docstring).
"""
import argparse
import json
import re
import sys

# Ordinal/upcoming-hour words in the original experiment's order.
_HOUR_WORDS = {
    'pl': (
        'pierwszej drugiej trzeciej czwartej piątej szóstej '
        'siódmej ósmej dziewiątej dziesiątej jedenastej dwunastej'
    ),
    'hr': (
        'jedan dva tri četiri pet šest sedam osam devet deset jedanaest dvanaest'
    ),
    'ru': (
        'первого второго третьего четвёртого пятого шестого '
        'седьмого восьмого девятого десятого одиннадцатого двенадцатого'
    ),
    'uk': (
        'першу другу третю четверту п’яту шосту сьому восьму '
        'дев’яту десяту одинадцяту дванадцяту'
    ),
}
WORDS = {
    lang: dict(zip(words.split(), range(1, 13)))
    for lang, words in _HOUR_WORDS.items()
}
PATTERNS = {
    'pl': r'(?<!\w)o wpół do (\S+) (rano|po południu|wieczorem|w nocy)(?!\w)',
    'hr': r'(?<!\w)u pola (\S+) (noću|ujutro|prijepodne|poslijepodne|navečer)(?!\w)',
    'ru': r'(?<!\w)в половине (\S+) (ночи|утра|дня|вечера)(?!\w)',
    'uk': r'(?<!\w)о пів на (\S+) (ранку|ночі|дня|вечора)(?!\w)',
}
PM = {
    'po południu', 'wieczorem', 'poslijepodne', 'navečer',
    'дня', 'вечера', 'вечора',
}
NIGHT = {'w nocy', 'noću', 'ночи', 'ночі'}
EVENING = {'wieczorem', 'navečer', 'вечера', 'вечора'}
PREPOSITIONS = {
    'pl': 'o',
    'hr': 'u',
    'ru': 'в',
    'uk': 'о',
}
# Count incomplete expressions too: a valid phrase must not hide another clock.
_STARTS = {
    'pl': r'\bo wpół do\b',
    'hr': r'\bu pola\b',
    'ru': r'\bв половине\b',
    'uk': r'\bо пів на\b',
}
_NUMERIC = re.compile(
    r'(?<![\w:])\d+(?::\d+)+(?![\w:])'
    r'|(?<!\w)\d+\s*[ap]\.?m\.?\b',
    re.IGNORECASE,
)


def normalize_clock(lang: str, text: str) -> str:
    """Preserve outside characters; unchanged on unsupported/ambiguous input.

    Non-string arguments raise TypeError. Multiple supported phrases, incomplete
    phrases with the same grammar prefix, and coexisting numeric clocks abstain.
    Arbitrary unsupported verbal clocks cannot all be detected by this grammar.
    """
    if not isinstance(lang, str) or not isinstance(text, str):
        raise TypeError('lang and text must be strings')
    if lang not in PATTERNS or _NUMERIC.search(text):
        return text
    if len(list(re.finditer(_STARTS[lang], text))) != 1:
        return text
    matches = list(re.finditer(PATTERNS[lang], text))
    if len(matches) != 1:
        return text
    match = matches[0]
    word, daypart = match.groups()
    upcoming = WORDS[lang].get(word)
    if upcoming is None:
        return text
    if upcoming == 12 and daypart in EVENING:
        return text
    hour = upcoming - 1
    if daypart in PM and hour < 12 and hour != 11:
        hour += 12
    if daypart in NIGHT and hour >= 6:
        hour += 12
    prep = PREPOSITIONS[lang]
    if lang == 'uk' and hour == 11:
        prep = 'об'
    return text[:match.start()] + f'{prep} {hour:02d}:30' + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON array; - reads stdin')
    parser.add_argument('--output', help='Write JSON instead of stdout')
    args = parser.parse_args()
    try:
        if args.input == '-':
            rows = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8') as handle:
                rows = json.load(handle)
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get('lang'), str)
            or not isinstance(row.get('source'), str) for row in rows
        ):
            raise ValueError('input must be array of objects with string lang and source')
        result = [dict(row, source=normalize_clock(row['lang'], row['source'])) for row in rows]
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as handle:
                handle.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
