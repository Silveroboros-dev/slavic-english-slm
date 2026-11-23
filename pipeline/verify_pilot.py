#!/usr/bin/env python3
"""Independently verify exported pilot JSONL files. Uses only the standard library."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import zlib


QUOTAS = {"train": 100_000, "validation": 1_000, "test": 1_000}
CORPUS_LANG = {"europarl": "pl", "jrc": "pl", "dgt": "hr", "unpc": "ru"}
FIELDS = ("lang", "corpus", "doc_id", "src", "tgt")
EXAMPLE_LIMIT = 8
NEAR_MIN_WORDS = 12
SHINGLE_WORDS = 5
NEAR_WORDS = re.compile(r"\w+")
TEMPLATE_WORDS = re.compile(r"[^\W_]+")
DIGITS = re.compile(r"\d+")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def archive_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Findings:
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, reason, **context):
        self.counts[reason] += 1
        if len(self.examples[reason]) < EXAMPLE_LIMIT:
            self.examples[reason].append(context)


def numeric_template(text):
    words = TEMPLATE_WORDS.findall(unicodedata.normalize("NFKC", text).casefold())
    return " ".join(DIGITS.sub("#", word) for word in words)


def shingle_set(text):
    words = NEAR_WORDS.findall(unicodedata.normalize("NFKC", text).casefold())
    if len(words) < NEAR_MIN_WORDS:
        return set()
    return {tuple(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}


class NearIndex:
    """Exact threshold search on held-out targets; result IDs are zero-based input positions."""

    def __init__(self, target_texts):
        self.shingles = [shingle_set(text) for text in target_texts]
        frequencies = Counter()
        for shingles in self.shingles:
            frequencies.update(shingles)
        self.postings = defaultdict(list)
        self.indexed_count = 0
        for idx, shingles in enumerate(self.shingles):
            if not shingles:
                continue
            self.indexed_count += 1
            # Jaccard >= 4/5 implies at most floor(n/5) missing eval shingles.
            # Index one more; every qualifying candidate must hit at least one.
            anchors = sorted(shingles, key=lambda sh: (frequencies[sh], sh))
            for anchor in anchors[:len(shingles) // 5 + 1]:
                self.postings[anchor].append(idx)

    def _find(self, probe):
        candidate_ids = set()
        for shingle in probe:
            posting = self.postings.get(shingle)
            if posting:
                candidate_ids.update(posting)
        for idx in sorted(candidate_ids):
            ref = self.shingles[idx]
            if 5 * min(len(probe), len(ref)) < 4 * max(len(probe), len(ref)):
                continue
            shared = len(probe.intersection(ref))
            union = len(probe) + len(ref) - shared
            if 5 * shared >= 4 * union:
                yield idx

    def matches(self, text):
        return next(self._find(shingle_set(text)), None) is not None

    def match_ids(self, text):
        return list(self._find(shingle_set(text)))


class NearDuplicateAudit:
    """Audit a single language side across all splits, indexing only evaluation sentences."""

    def __init__(self, side):
        self.side = side
        self.heldout_texts = []
        self.heldout_rows = []
        self.index = None
        self.eligible = Counter()
        self.short = Counter()
        self.pair_counts = Counter()
        self.flagged_rows = Counter()
        self.examples = []

    def begin(self, split):
        if split in ("test", "train"):
            self.index = NearIndex(self.heldout_texts)

    def observe(self, split, row, text):
        shingles = shingle_set(text)
        if shingles:
            self.eligible[split] += 1
        else:
            self.short[split] += 1
        flagged = False
        if self.index is not None and shingles:
            for candidate in self.index._find(shingles):
                reference_split, reference_row = self.heldout_rows[candidate]
                self.pair_counts[f"{split}_vs_{reference_split}"] += 1
                flagged = True
                if len(self.examples) < EXAMPLE_LIMIT:
                    shared = len(shingles.intersection(self.index.shingles[candidate]))
                    union = len(shingles) + len(self.index.shingles[candidate]) - shared
                    self.examples.append({"split": split, "row": row, "compared_split": reference_split,
                                          "compared_row": reference_row, "jaccard": round(shared / union, 6)})
        if flagged:
            self.flagged_rows[split] += 1
        if split != "train":
            self.heldout_texts.append(text)
            self.heldout_rows.append((split, row))

    def report(self):
        return {
            "blocking": True,
            "scope": f"{self.side}, across all languages and every pair of different splits.",
            "method": (
                "NFKC and Unicode casefold, Python Unicode word regex \\w+, at least 12 words on "
                "both sides. Distinct contiguous 5-word shingles. For n reference shingles, "
                "select floor(n/5)+1 anchors by rarity (ties lexicographic); probe all candidate "
                "shingles; complete set Jaccard >= 4/5 exactly, with no truncation or candidate cap."
            ),
            "limitations": (
                "Excludes short texts and comparisons across the source/target fields. Paraphrases and heavily "
                "reordered, edited or lexically different sentences can score below threshold. "
                "No inference about pretraining or semantic independence."
            ),
            "eligible_rows": dict(self.eligible), "excluded_short_rows": dict(self.short),
            "flagged_pairs": {k: self.pair_counts[k] for k in
                              ("train_vs_validation", "train_vs_test", "test_vs_validation")},
            "flagged_probe_rows": dict(self.flagged_rows), "examples": self.examples,
        }


class SliceStats:
    def __init__(self):
        self.n = 0
        self.lengths = {}

    def observe(self, src, tgt, n_src, n_tgt):
        self.n += 1
        for key, value in (("src_words", n_src), ("tgt_words", n_tgt),
                           ("src_chars", len(src)), ("tgt_chars", len(tgt))):
            if key not in self.lengths:
                self.lengths[key] = [value, value]
            else:
                bounds = self.lengths[key]
                bounds[0] = min(value, bounds[0])
                bounds[1] = max(value, bounds[1])

    def report(self):
        return {"rows": self.n, "min_max": self.lengths}


class SplitStats:
    def __init__(self):
        self.rows = 0
        self.lang = Counter()
        self.corpora = defaultdict(SliceStats)
        self.doc_ids = set()
        self.corpus_docs = defaultdict(set)
        self.overall = SliceStats()
        self.runs = Counter()
        self.last_lang = None
        self.current_run = 0
        self.longest_run = 0

    def observe(self, lang, corpus, doc, src, tgt, n_src, n_tgt):
        self.lang[lang] += 1
        self.doc_ids.add(doc)
        self.corpus_docs[corpus].add(doc)
        self.overall.observe(src, tgt, n_src, n_tgt)
        self.corpora[corpus].observe(src, tgt, n_src, n_tgt)
        if lang != self.last_lang:
            self.runs[lang] += 1
            self.current_run = 0
            self.last_lang = lang
        self.current_run += 1
        self.longest_run = max(self.longest_run, self.current_run)

    def report(self):
        return {"physical_rows": self.rows, "eligible_rows": self.overall.n,
                "unique_documents": len(self.doc_ids), "by_language": dict(self.lang),
                "by_corpus": {name: {**stats.report(), "unique_documents": len(self.corpus_docs[name])}
                              for name, stats in self.corpora.items()},
                "min_max": self.overall.lengths, "language_runs": dict(self.runs),
                "longest_language_run": self.longest_run}


def register_split_key(index, key, split, row, label, findings):
    previous = index.setdefault(key, split)
    if previous != split:
        findings.add(label, split=split, row=row, previous_split=previous,
                     key_sha256=key.hex())


def check_file(split, path, exact_index, target_audit, source_audit, findings):
    stats = SplitStats()
    report = {"filename": path.name}
    pairs = set()
    try:
        report.update(bytes=path.stat().st_size, sha256=archive_hash(path))
        with gzip.open(path, "rb") as stream:
            for number, raw in enumerate(stream, 1):
                stats.rows += 1
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except (ValueError, RecursionError):
                    findings.add("invalid_utf8_or_json", split=split, row=number)
                    continue
                if not isinstance(rec, dict) or not all(
                    isinstance(rec.get(field), str) and rec[field].strip() for field in FIELDS
                ):
                    findings.add("missing_or_invalid_field", split=split, row=number)
                    continue
                lang, corpus, doc, src, tgt = (rec[f] for f in FIELDS)
                if CORPUS_LANG.get(corpus) != lang:
                    findings.add("invalid_language_or_corpus", split=split, row=number)
                    continue
                namespace = "eu" if corpus in ("jrc", "dgt") else corpus
                prefix, _, identifier = doc.partition(":")
                if prefix != namespace or not identifier.strip() or doc != doc.strip():
                    findings.add("invalid_doc_namespace", split=split, row=number)
                raw_source, raw_target = src, tgt
                src, tgt = normalize(src), normalize(tgt)
                if not src or not tgt:
                    findings.add("empty_normalized_text", split=split, row=number)
                    continue
                src_key, tgt_key, doc_key = fingerprint(src), fingerprint(tgt), fingerprint(doc)
                for label, key in (("document_overlap", doc_key), ("source_overlap", src_key),
                                   ("english_target_overlap", tgt_key)):
                    register_split_key(exact_index[label], key, split, number, label, findings)
                for label, key, opposite in (("source_matches_other_target", src_key, "english_target_overlap"),
                                              ("target_matches_other_source", tgt_key, "source_overlap")):
                    previous = exact_index[opposite].get(key)
                    if previous and previous != split:
                        findings.add(label, split=split, row=number, previous_split=previous,
                                     key_sha256=key.hex())
                template = numeric_template(raw_target)
                if template:
                    register_split_key(exact_index["english_template_overlap"], fingerprint(template),
                                       split, number, "english_template_overlap", findings)
                pair = src_key + tgt_key
                if pair in pairs:
                    findings.add("duplicate_pair_in_split", split=split, row=number,
                                 pair_sha256=hashlib.sha256(pair).hexdigest())
                pairs.add(pair)
                tgt_words = tgt.split()
                stats.observe(lang, corpus, doc, src, tgt, len(src.split()), len(tgt_words))
                target_audit.observe(split, number, raw_target)
                source_audit.observe(split, number, raw_source)
    except (OSError, EOFError, zlib.error) as exc:
        findings.add("archive_read_failure", split=split, error_type=type(exc).__name__)
    for lang in sorted(set(CORPUS_LANG.values())):
        if stats.lang[lang] != QUOTAS[split]:
            findings.add("quota", split=split, lang=lang, actual=stats.lang[lang],
                         expected=QUOTAS[split])
    for corpus in CORPUS_LANG:
        if corpus not in stats.corpora:
            findings.add("missing_corpus", split=split, corpus=corpus)
    if len(stats.runs) > 1 and all(run == 1 for run in stats.runs.values()):
        findings.add("concatenated_language_blocks", split=split)
    return {**report, **stats.report()}


def verify(directory):
    findings = Findings()
    target_audit = NearDuplicateAudit("English targets")
    source_audit = NearDuplicateAudit("Source texts")
    index = defaultdict(dict)
    files = {}
    for split in ("validation", "test", "train"):
        target_audit.begin(split)
        source_audit.begin(split)
        path = directory / (split + ".jsonl.gz")
        files[split] = check_file(split, path, index, target_audit, source_audit, findings)
    for label, audit in (("english", target_audit), ("source", source_audit)):
        for split_pair, count in audit.pair_counts.items():
            if count:
                findings.counts[f"near_{label}_overlap:{split_pair}"] = count
    return {
        "passed": not bool(findings.counts), "quotas_per_language": QUOTAS, "splits": files,
        "exact_checks": {
            "normalization": "NFKC -> Unicode casefold -> collapse/trim Unicode whitespace",
            "normalized_source_scope": "global across languages",
            "english_target_scope": "global across languages",
            "cross_field_scope": "normalized source on either side vs English target in another split",
            "english_template_scope": "NFKC, casefold, Unicode alphanumeric tokens without underscores; "
                                      "replace each run of decimal digits with #; skip empty keys",
            "doc_id_policy": "exact, nonempty ID; namespaces europarl:, unpc:, or eu: shared by jrc/dgt",
            "pair_policy": "unique normalized source-target tuple within each split",
            "mixed_order_policy": "reject one contiguous block per language; no randomness test",
            "length_units": "characters and whitespace tokens in normalized text; not model tokens",
            "overlap_count_unit": "rows conflicting with the first split for a key",
            "violations": dict(findings.counts), "examples": dict(findings.examples),
            "limitations": "Archive integrity and text overlap checks do not validate the upstream "
                           "document mapping, correct language, semantic fidelity, or model pretraining.",
        },
        "near_duplicate_audit": target_audit.report(),
        "source_near_duplicate_audit": source_audit.report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Folder containing the three split files")
    parser.add_argument("--report", help="Report path, or - for stdout (default verification.json)")
    args = parser.parse_args()
    result = verify(args.directory)
    summary = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report == "-":
        sys.stdout.write(summary)
    else:
        path = Path(args.report) if args.report else args.directory / "verification.json"
        path.write_text(summary, encoding="utf-8")
        print(f"Verification {'passed' if result['passed'] else 'failed'}; report: {path}",
              file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
