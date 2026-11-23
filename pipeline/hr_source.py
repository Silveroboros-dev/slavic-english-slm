"""Restore document IDs for the frozen OPUS DGT Croatian–English Moses archive."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

_BASE = "https://object.pouta.csc.fi/OPUS-DGT/v2021"
_BITEXT = {
    "filename": "DGT-v2021.en-hr.txt.zip",
    "url": f"{_BASE}/moses/en-hr.txt.zip",
    "bytes": 81_466_113,
    "sha256": "7f6c14973220f2890819b4742c1f64711ed01c9526c8ae036338d4502b654210",
}
_ALIGNMENT = {
    "filename": "DGT-v2021.en-hr.xml.gz",
    "url": f"{_BASE}/xml/en-hr.xml.gz",
    "bytes": 9_448_896,
    "sha256": "8f26ba44df964029ef859f77d1f4164599c67ddab98cf3bbeb9ff35ca12adff5",
}
SOURCE_INFO = {
    "corpus": "OPUS DGT",
    "release": "v2021",
    "pair": "hr-en",
    "pairs": 1_416_461,
    "documents": 8_147,
    "bitext": _BITEXT,
    "alignment": _ALIGNMENT,
    "document_key": "eu:<CELEX ID>; restore parentheses around numeric suffix",
    "order_verification": {
        "paired_rows_checked": 1_416_461,
        "paired_rows_matching_both_languages": 1_416_461,
        "basis": "Reconstructed both strings of every XCES link from OPUS raw XML "
                 "sentence IDs, joined multiple sentences with a single space, "
                 "and compared both strings to the corresponding Moses lines.",
        "verification_only_inputs": [
            {
                "url": f"{_BASE}/raw/en.zip",
                "bytes": 247_736_448,
                "sha256": "fbef7e8a1df28636b1f470e18d5322534a218503af9bd0325e261c8a42fa9ecd",
            },
            {
                "url": f"{_BASE}/raw/hr.zip",
                "bytes": 56_292_060,
                "sha256": "d363eec97d9202896ce96b4c2d94a4f4c7ebe3416ca76a51dfc42b70e831c98e",
            },
        ],
    },
}


def _verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Unexpected size for {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")


def _source_file(preferred_dir, cache_dir, spec):
    preferred = Path(preferred_dir) / spec["filename"]
    cache = Path(cache_dir) / spec["filename"]
    for path in (preferred, cache):
        if path.exists():
            _verify(path, spec)
            return path
    cache.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache.parent, prefix=cache.name + ".", delete=False) as out:
        temp = Path(out.name)
    try:
        request = Request(spec["url"], headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=90) as response, temp.open("wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                if out.tell() + len(block) > spec["bytes"]:
                    raise ValueError(f"Download larger than pinned size for {cache.name}")
                out.write(block)
        _verify(temp, spec)
        temp.replace(cache)
    finally:
        temp.unlink(missing_ok=True)
    return cache


def _document_groups(alignment):
    seen = set()
    count = 0
    with gzip.open(alignment, "rb") as stream:
        for _, group in ElementTree.iterparse(stream, events=("end",)):
            if group.tag != "linkGrp":
                continue
            doc = re.fullmatch(r"en/([A-Z0-9]+)(?:_(\d+)_)?\.xml\.gz", group.attrib["fromDoc"])
            if doc is None:
                raise ValueError(f"Unexpected DGT document path {group.attrib['fromDoc']!r}")
            if group.attrib["toDoc"] != "hr/" + group.attrib["fromDoc"][3:]:
                raise ValueError("English and Croatian document IDs differ")
            key = "eu:" + doc[1] + (f"({doc[2]})" if doc[2] else "")
            if key in seen:
                raise ValueError(f"Repeated DGT document {key}")
            seen.add(key)
            count += len(group)
            yield key, group
            group.clear()
    if (count, len(seen)) != (SOURCE_INFO["pairs"], SOURCE_INFO["documents"]):
        raise ValueError(f"Unexpected DGT alignment counts: {count} links, {len(seen)} documents")


def _document_keys(alignment):
    for key, group in _document_groups(alignment):
        for link in group:
            sides = link.get("xtargets", "").split(";")
            if link.tag != "link" or len(sides) != 2 or not all(side.strip() for side in sides):
                raise ValueError(f"Invalid XCES link in {key}")
            yield key


def iter_hr(data_dir, cache_dir):
    """Yield (canonical EU document key, Croatian source, English target) for every link."""
    bitext = _source_file(data_dir, cache_dir, _BITEXT)
    alignment = _source_file(cache_dir, cache_dir, _ALIGNMENT)
    with ZipFile(bitext) as corpus:
        with corpus.open("DGT.en-hr.hr") as hrs, corpus.open("DGT.en-hr.en") as ens:
            for key, hr, en in zip_longest(_document_keys(alignment), hrs, ens):
                if key is None or hr is None or en is None:
                    raise ValueError("DGT XCES and Moses files end on different rows")
                yield key, hr.decode("utf-8").rstrip("\r\n"), en.decode("utf-8").rstrip("\r\n")


def _raw_sentences(archive, document):
    member = "DGT/raw/" + document.removesuffix(".gz")
    with archive.open(member) as stream:
        root = ElementTree.parse(stream).getroot()
    sentences = {}
    for sentence in root.iter("s"):
        index = sentence.attrib["id"]
        if index in sentences:
            raise ValueError(f"Duplicate sentence ID in {document}")
        sentences[index] = "".join(sentence.itertext())
    return sentences


def check_document_mapping(data_dir, cache_dir):
    """Check each source and target against the raw XML sentences named in its XCES link."""
    alignment = _source_file(cache_dir, cache_dir, _ALIGNMENT)
    row_count = document_count = 0
    with ExitStack() as stack:
        raw = {}
        for spec in SOURCE_INFO["order_verification"]["verification_only_inputs"]:
            lang = Path(spec["url"]).stem
            named_spec = dict(spec, filename=f"DGT-v2021.raw.{lang}.zip")
            path = _source_file(cache_dir, cache_dir, named_spec)
            raw[lang] = stack.enter_context(ZipFile(path))
        rows = iter_hr(data_dir, cache_dir)
        stack.callback(rows.close)
        for document, group in _document_groups(alignment):
            document_count += 1
            en = _raw_sentences(raw["en"], group.attrib["fromDoc"])
            hr = _raw_sentences(raw["hr"], group.attrib["toDoc"])
            for link in group:
                row_count += 1
                en_ids, hr_ids = link.attrib["xtargets"].split(";")
                expected = (
                    document,
                    " ".join(hr[i] for i in hr_ids.split()),
                    " ".join(en[i] for i in en_ids.split()),
                )
                if next(rows, None) != expected:
                    raise ValueError(f"Moses pair differs from raw XML at row {row_count} in {document}")
        if next(rows, None) is not None:
            raise ValueError("Moses contains extra pairs after the final XCES link")
    return row_count, document_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-document-mapping", action="store_true", required=True,
                        help="compare every Moses row and document ID with the raw aligned source")
    parser.add_argument("--data-dir", required=True, type=Path, help="directory with the DGT Moses ZIP")
    parser.add_argument("--cache-dir", required=True, type=Path, help="directory for pinned OPUS sources")
    args = parser.parse_args()
    rows, documents = check_document_mapping(args.data_dir, args.cache_dir)
    print(f"Verified {rows:,} Croatian–English pairs from {documents:,} documents against raw XML.")
