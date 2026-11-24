#!/usr/bin/env python3
"""Replay frozen English clock labels, normalization and guard audit offline.

Run: python pipeline/score_norm64.py [--data results/norm64_cases.jsonl]
Blind labels are observations; the engineering guard never assigns these labels.
Clock preservation is not a measure of translation quality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_ARMS = ('r4_normalized', 'nllb_base', 'opus', 'gemma_1b')
ARMS = (*NORMALIZED_ARMS, 'r4_spoken', 'nllb_base_spoken')
GROUPS = {'primary_pl_hr_ru': {'pl', 'hr', 'ru'}, 'pl': {'pl'},
          'hr': {'hr'}, 'ru': {'ru'}, 'uk_separate': {'uk'}}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def correct(row, arm):
    label = row['outputs'][arm]['label']
    return (label['status'] == 'exact'
            and label['times_24h'] == [row['expected_24h']]
            and label.get('output_class', 'english') == 'english')


def transition(before, after):
    if before == after:
        return 'unchanged_pass' if before else 'unchanged_fail'
    return 'fix' if after else 'loss'


def require(condition, message="Invalid frozen case data"):
    if not condition:
        raise ValueError(message)


def group_pairs(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    return pairs


def _validate(rows):
    """Validate shape and frozen text bindings before computing any counts."""
    require(len(rows) == 64, 'Expected exactly 64 cases')
    require(len({r['id'] for r in rows}) == 64, 'Duplicate case ID')
    require(Counter((r['lang'] for r in rows)) == dict.fromkeys(('pl', 'hr', 'ru', 'uk'), 16))
    for field in ('spoken_source', 'normalized_input'):
        require(len({r[field] for r in rows}) == 64, f'Duplicate {field}')
    pairs = group_pairs(rows)
    for row in rows:
        require(row['id'] == row['pair_id'] + '-' + row['variant'])
        require(re.fullmatch('R4H-' + row['lang'].upper() + '-0[1-8]-[AB]', row['id']))
        require(re.fullmatch('(?:[01][0-9]|2[0-3]):30', row['expected_24h']))
        require(digest(row['normalized_input']) == row['normalized_input_sha256'])
        require(set(row['outputs']) == set(ARMS))
        for arm, output in row['outputs'].items():
            if arm in ('r4_spoken', 'r4_normalized'):
                require(isinstance(output.get('comet'), float), 'Missing COMET score')
                require(math.isfinite(output['comet']), 'Non-finite COMET score')
            else:
                require('comet' not in output, 'Unexpected COMET score')
            require(isinstance(output['text'], str))
            require(digest(output['text']) == output['text_sha256'])
            label = output['label']
            require(isinstance(label['evidence'], str), 'Missing observation evidence')
            require(label['status'] in ('exact', 'ambiguous', 'multiple', 'no_time', 'invalid'))
            require(label.get('output_class', 'english') in ('english', 'non_english', 'silent', 'commentary_contradiction'))
            for field, limit in (('times_24h', 24), ('clock12', 12)):
                values = label[field]
                require(len(values) == len(set(values)))
                for value in values:
                    require(re.fullmatch('[0-9]{2}:[0-9]{2}', value))
                    require(int(value[:2]) < limit and int(value[3:]) < 60)
            require(label['status'] != 'exact' or len(label['times_24h']) == 1)
    require(len(pairs) == 32)
    for pair in pairs.values():
        require(sorted((row['variant'] for row in pair)) == ['A', 'B'])
        require(len({row['lang'] for row in pair}) == 1)
        require(len({row['expected_24h'] for row in pair}) == 2,
                'Pair members must have different expected clocks')
    return pairs


def validate(rows):
    """Expose malformed records as data errors, including missing fields."""
    try:
        return _validate(rows)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed frozen case data: {exc}") from exc


def load_cases(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def summarize(rows, arm, guard):
    pairs = group_pairs(rows)
    results = [guard(row['normalized_input'], row['outputs'][arm]['text'])
               for row in rows] if arm in NORMALIZED_ARMS else []
    false_matches = [row['id'] for row, result in zip(rows, results)
                     if result['status'] == 'match' and not correct(row, arm)]
    flagged_correct = [row['id'] for row, result in zip(rows, results)
                       if result['status'] != 'match' and correct(row, arm)]
    return {
        'variants': len(rows),
        'exact_correct': sum(correct(row, arm) for row in rows),
        'pairs': len(pairs),
        'full_pairs': sum(all(correct(row, arm) for row in pair) for pair in pairs.values()),
        'statuses': dict(Counter(row['outputs'][arm]['label']['status'] for row in rows)),
        'output_classes': dict(Counter(row['outputs'][arm]['label'].get('output_class', 'english') for row in rows)),
        'guard': dict(Counter(result['status'] for result in results)),
        'guard_match_blind_fail_ids': false_matches,
        'guard_flag_blind_pass_ids': flagged_correct,
    }


def replay(rows, check, normalize_clock):
    validate(rows)
    for row in rows:
        actual = normalize_clock(row['lang'], row['spoken_source'])
        require(actual == row['normalized_input'], f"Normalization mismatch: {row['id']}")
        clocks = re.findall(r'(?<!\w)(?:[01]\d|2[0-3]):[0-5]\d(?!\w)', actual)
        require(clocks == [row['expected_24h']], f"Expected clock mismatch: {row['id']}")
    counts = {}
    ablation = {}
    for group, langs in GROUPS.items():
        subset = [row for row in rows if row['lang'] in langs]
        counts[group] = {arm: summarize(subset, arm, check) for arm in ARMS}
        variant_transitions = Counter()
        pair_transitions = Counter()
        for row in subset:
            before = correct(row, 'r4_spoken')
            after = correct(row, 'r4_normalized')
            variant_transitions[transition(before, after)] += 1
        for pair in group_pairs(subset).values():
            before = all(correct(row, 'r4_spoken') for row in pair)
            after = all(correct(row, 'r4_normalized') for row in pair)
            pair_transitions[transition(before, after)] += 1
        ablation[group] = {
            'before_arm': 'r4_spoken',
            'after_arm': 'r4_normalized',
            'variant_transitions': dict(variant_transitions),
            'pair_transitions': dict(pair_transitions),
        }
    special = next(r for r in rows if r['id'] == 'R4H-PL-07-B')
    output = special['outputs']['gemma_1b']
    return {'normalization_verified': len(rows), 'counts': counts,
            'normalization_ablation': ablation,
            'gemma_pl_07_b_guard_audit': {'id': special['id'], 'text': output['text'],
                'blind_label': output['label'], 'blind_correct': correct(special, 'gemma_1b'),
                'guard': check(special['normalized_input'], output['text'])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'results' / 'norm64_cases.jsonl')
    parser.add_argument('--output', type=Path, help='Write replay JSON instead of stdout')
    args = parser.parse_args()
    if __package__:
        from .clock_check import check
        from .clock_normalize import normalize_clock
    else:
        from clock_check import check
        from clock_normalize import normalize_clock
    rows = load_cases(args.data)
    rendered = json.dumps(replay(rows, check, normalize_clock), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
