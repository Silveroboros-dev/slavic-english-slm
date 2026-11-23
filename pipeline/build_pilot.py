"""Build a balanced document-disjoint Slavic to English pilot using Python 3.10+."""
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import re
import tempfile
import unicodedata
import zipfile

from hr_source import SOURCE_INFO, iter_hr
from readers import iter_pairs, verify_raw
from verify_pilot import NearIndex

LANG = {"europarl": "pl", "jrc": "pl", "dgt": "hr", "unpc": "ru"}
# Independent of the eventual evaluation outcomes. Polish stays at 28% Europarl / 72% JRC.
QUOTAS = {
    "test": {"europarl": 280, "jrc": 720, "dgt": 1000, "unpc": 1000},
    "validation": {"europarl": 280, "jrc": 720, "dgt": 1000, "unpc": 1000},
    "train": {"europarl": 28000, "jrc": 72000, "dgt": 100000, "unpc": 100000},
}
DEFAULT_SEED = 42

def rng_for(seed, label):
    return random.Random(f"slavic-pilot-v1|{seed}|{label}")

@lru_cache(maxsize=16384)
def doc_split(seed, doc_id):
    key = f"slavic-pilot-v1|{seed}|doc|{doc_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"

def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())

def template(s):
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold())
    return " ".join(re.sub(r"\d+", "#", word) for word in tokens)

def clean_text(s):
    # OPUS Moses exports can contain HTML entities. Keep one record per JSON line.
    return " ".join(html.unescape(s).split())

def eligible(a, b):
    if not a or not b or any(marker in text for text in (a, b) for marker in ('\ufffd', '愦灭;')):
        return False
    x, y = len(a.split()), len(b.split())
    if max(x, y) > 120 or max(x, y) > 3.5 * min(x, y):
        return False
    if not any(ch.isalpha() for ch in a) or not any(ch.isalpha() for ch in b):
        return False
    return norm(a) != norm(b)

class Reservoir:
    def __init__(self, cap, rng):
        self.cap = cap
        self.rng = rng
        self.rows = []
        self.seen = 0

    def add(self, doc, src, tgt, i):
        self.seen += 1
        if self.seen <= self.cap:
            self.rows.append((doc, src, tgt, i))
        else:
            j = self.rng.randrange(self.seen)
            if j < self.cap:
                self.rows[j] = (doc, src, tgt, i)

def candidates(data_dir, cache_dir, seed):
    pooled = {}
    counts = {}
    for corpus in LANG:
        reservoirs = {split: Reservoir(3*q[corpus] + 1500, rng_for(seed, f"pool/{corpus}/{split}"))
                      for split, q in QUOTAS.items()}
        records = iter_hr(data_dir, cache_dir) if corpus == "dgt" else iter_pairs(data_dir, corpus)
        for row, (doc, src, tgt) in enumerate(records, 1):
            reservoirs[doc_split(seed, doc)].add(doc, src, tgt, row)
        counts[corpus] = {s: {"rows_scanned": r.seen, "sampled_candidates": len(r.rows)}
                          for s, r in reservoirs.items()}
        for split, r in reservoirs.items():
            rng_for(seed, f"draw/{corpus}/{split}").shuffle(r.rows)
            pooled[corpus, split] = r.rows
        print(f"Scanned {corpus}: {sum(r.seen for r in reservoirs.values()):,} rows", flush=True)
    return pooled, counts

def select(pooled, seed):
    outputs = {}
    stats = {}
    prior_texts, prior_templates = set(), set()
    previous_sources, previous_targets = [], []
    for split, quotas in QUOTAS.items():
        near_target = NearIndex(previous_targets)
        near_source = NearIndex(previous_sources)
        records, split_texts, split_templates, pairs = [], set(), set(), set()
        stats[split] = {}
        for corpus, requested in quotas.items():
            tally = Counter()
            lang = LANG[corpus]
            for doc, raw_src, raw_tgt, row in pooled.pop((corpus, split)):
                src, tgt = clean_text(raw_src), clean_text(raw_tgt)
                if not eligible(src, tgt):
                    tally["quality"] += 1
                    continue
                a, b = norm(src), norm(tgt)
                if a in prior_texts or b in prior_texts:
                    tally["exact_text_in_prior_split"] += 1
                    continue
                fingerprint = template(tgt)
                if fingerprint in prior_templates:
                    tally["english_template_in_prior_split"] += 1
                    continue
                pair = (a, b)
                if pair in pairs:
                    tally["duplicate_pair_within_split"] += 1
                    continue
                if near_source.matches(src):
                    tally["source_near_duplicate_in_prior_split"] += 1
                    continue
                if near_target.matches(tgt):
                    tally["english_near_duplicate_in_prior_split"] += 1
                    continue
                pairs.add(pair)
                if split != "train":
                    split_texts.update((a, b))
                    split_templates.add(fingerprint)
                records.append(dict(lang=lang, corpus=corpus, doc_id=doc, row_id=row, src=src, tgt=tgt))
                tally["kept"] += 1
                if tally["kept"] == requested:
                    break
            if tally["kept"] != requested:
                raise ValueError(f"{corpus}/{split}: found {tally['kept']} of {requested} required rows")
            stats[split][corpus] = dict(tally)
        rng_for(seed, f"order/{split}").shuffle(records)
        outputs[split] = records
        if split != "train":
            prior_texts.update(split_texts)
            prior_templates.update(split_templates)
            previous_sources.extend(r["src"] for r in records)
            previous_targets.extend(r["tgt"] for r in records)
        print(f"Selected {split}: {len(records):,} examples", flush=True)
    return outputs, stats

def write_records(path, rows):
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            out.write(line.encode("utf-8"))

def write_notices(data_dir, out):
    texts = [
        "Slavic-English NMT pilot v1. Dataset source notices. Research snapshot, 20 November 2026.",
        "Europarl v10. European Parliament proceedings. Cite Philipp Koehn (2005). The compiler requests contact for research use. Terms: https://www.statmt.org/europarl/ . File: https://www.statmt.org/europarl/v10/training/europarl-v10.pl-en.tsv.gz",
        "JRC-Acquis v3.0. Credit the European Communities and the Language Technology team at the European Commission Joint Research Centre. Cite Steinberger et al. (2006) and Tiedemann (2012) for OPUS. Research-use conditions and prescribed notices: https://wt-public.emm4u.eu/Acquis/JRC-Acquis.3.0/doc/licence.html . Only European Community legislation printed in the paper edition of the Official Journal of the European Union is deemed authentic.",
        "DGT Translation Memory. Source: https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-translation-memory_en . The European Commission retains ownership. OPUS v2021 package dated 18 July 2026 per its README. The exact upstream update date for this bilingual extraction is unverified. Cite Steinberger et al. (2012) and Tiedemann (2012).",
        "UN Parallel Corpus v1.0. Source: United Nations. Cite Ziemski, Junczys-Dowmunt and Pouliquen (2016), https://aclanthology.org/L16-1561/ .",
    ]
    for filename in ("JRC-Acquis-v3.0.en-pl.txt.zip", "DGT-v2021.en-hr.txt.zip", "UNPC-v1.0.en-ru.txt.zip"):
        with zipfile.ZipFile(data_dir / filename) as z:
            texts.append(filename + "\n" + z.read("LICENSE").decode("utf-8"))
    (out / "SOURCE_NOTICES.txt").write_text("\n\n".join(texts) + "\n", encoding="utf-8")

def build(data_dir, cache_dir, out, seed):
    if out.exists():
        raise FileExistsError(f"Choose an unused output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pinned = verify_raw(data_dir)
    pooled, counts = candidates(data_dir, cache_dir, seed)
    outputs, stats = select(pooled, seed)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".pilot-build-") as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        for split, rows in outputs.items():
            write_records(stage / f"{split}.jsonl.gz", rows)
        write_notices(data_dir, stage)
        manifest = dict(version=1, seed=seed, target_language="en", source_archives=pinned,
                        croatian_document_index=SOURCE_INFO, quotas=QUOTAS, input_counts=counts,
                        sample_decisions=stats, counts_note="Rejection counts cover candidates examined until each quota filled.",
                        source_row_note="row_id is the reader's one-based record ordinal before length/quality filtering; see the reader for Europarl repair and empty-row handling.",
                        design="Documents assigned before sampling by stable hash (90% train, 5% validation, 5% test buckets). Uniform random candidate reservoir within each corpus/bucket, then seeded selection without replacement. Test has precedence, then validation, then train. Canonical EU document keys are shared across JRC and DGT.",
                        coverage="Institutional Slavic-to-English translation. No bilingual adequacy assessment or model pretraining overlap audit.")
        (stage / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).with_name("verify_pilot.py")), str(stage)], check=True)
        stage.rename(out)
    print(f"Verified pilot written to {out}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the four raw project archives")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory to publish only after verification")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Local cache for the pinned Croatian document index")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    build(args.data_dir, args.cache_dir, args.output_dir, args.seed)
