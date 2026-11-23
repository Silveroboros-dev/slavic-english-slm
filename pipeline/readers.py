"""Read the pinned Europarl and OPUS archives with document-level provenance."""
import gzip
import hashlib
import io
import re
from pathlib import Path
from xml.parsers import expat
from zipfile import ZipFile


RAW_FILES = {
    "europarl": ("Europarl-v10.pl-en.tsv.gz", "10968af7f3b203d2c3b7f2c305f2a1baad417858c7a51444a25632f3a58f8f67"),
    "jrc": ("JRC-Acquis-v3.0.en-pl.txt.zip", "7307d157478a125aefa85717c5e5156ba18eff19cf8e7e7b7e3eb15b5fba5dcb"),
    "unpc": ("UNPC-v1.0.en-ru.txt.zip", "6b97f111ecfa67f2534cb699441c91b4eb77e6e9aee31f2bdac40e993213683b"),
    "dgt": ("DGT-v2021.en-hr.txt.zip", "7f6c14973220f2890819b4742c1f64711ed01c9526c8ae036338d4502b654210"),
}
_EXPECTED_ROWS = {"europarl": 632600, "jrc": 1610962, "unpc": 25173398}
_BLOCK = 1024 * 1024
_EP_FILE = re.compile(r"ep-\d{2}-\d{2}-\d{2}-[a-zA-Z0-9-]+\.txt")
# Five prefixed tabs, and two numeric labels separated from Polish text by a tab.
_EP_REPAIRS = {32236: "", 44621: "", 52494: "", 109377: "", 365823: "",
               462440: "13.", 614854: "12."}


def verify_raw(data_dir):
    """Verify all four original compressed archives, returning filename -> SHA-256."""
    verified = {}
    for filename, expected in RAW_FILES.values():
        digest = hashlib.sha256()
        with (Path(data_dir) / filename).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * _BLOCK), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {filename}: {digest.hexdigest()}")
        verified[filename] = expected
    return verified


def eu_doc_key(celex):
    """Canonical key shared by the JRC and DGT representations of an EU act."""
    identifier = re.sub(r"_(\d+)_?", r"(\1)", celex.upper())
    if not re.fullmatch(r"[0-9A-Z]+(?:\(\d+\)[0-9A-Z]*)*", identifier):
        raise ValueError(f"Unrecognized CELEX identifier: {celex!r}")
    return "eu:" + identifier


def _europarl(path):
    rows = 0
    with gzip.open(path, "rb") as stream:
        for rows, line in enumerate(stream, 1):
            fields = line.rstrip(b"\r\n").decode("utf-8").split("\t")
            if rows in _EP_REPAIRS:
                if len(fields) != 9 or fields[0].strip() != _EP_REPAIRS[rows]:
                    raise ValueError(f"Unexpected Europarl repair layout at row {rows}")
                fields[:2] = [" ".join(s.strip() for s in fields[:2] if s.strip())]
            if len(fields) != 8 or not _EP_FILE.fullmatch(fields[2]):
                raise ValueError(f"Invalid Europarl layout or file ID at row {rows}")
            yield "europarl:" + fields[2], fields[0], fields[1]
    if rows != _EXPECTED_ROWS["europarl"]:
        raise ValueError(f"Unexpected Europarl row count: {rows}")


def _opus_doc_key(corpus, en_path, src_path):
    lang = "pl" if corpus == "jrc" else "ru"
    if not en_path.startswith("en/") or not src_path.startswith(lang + "/"):
        raise ValueError(f"Unexpected {corpus} alignment paths: {en_path}, {src_path}")
    if corpus == "unpc":
        if en_path[3:] != src_path[3:]:
            raise ValueError(f"UN document paths disagree: {en_path}, {src_path}")
        return "unpc:" + en_path[3:]
    en = re.fullmatch(r"en/(\d{4})/jrc([^/]+)-en\.xml\.gz", en_path)
    pl = re.fullmatch(r"pl/(\d{4})/jrc([^/]+)-pl\.xml\.gz", src_path)
    if not en or not pl or en.groups() != pl.groups():
        raise ValueError(f"JRC document paths disagree: {en_path}, {src_path}")
    return eu_doc_key(en[2])


def _opus_groups(stream, corpus):
    """Stream (document key, number of nonempty Moses alignments) from XCES."""
    key = None
    count = 0
    ready = []
    parser = expat.ParserCreate()

    def start(name, attrs):
        nonlocal key, count
        if name == "link":
            if key is None:
                raise ValueError(f"{corpus} link outside linkGrp")
            left, right = attrs["xtargets"].split(";")
            if left.strip() and right.strip():
                count += 1
        elif name == "linkGrp":
            if key is not None:
                raise ValueError(f"Nested {corpus} linkGrp")
            key = _opus_doc_key(corpus, attrs["fromDoc"], attrs["toDoc"])
            count = 0
        elif name != "cesAlign":
            raise ValueError(f"Unexpected {corpus} XCES element: {name}")

    def end(name):
        nonlocal key
        if name == "linkGrp":
            ready.append((key, count))
            key = None

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    while block := stream.read(_BLOCK):
        parser.Parse(block, False)
        yield from ready
        ready.clear()
    parser.Parse(b"", True)


def iter_pairs(data_dir, corpus):
    """Yield (canonical document key, Polish/Russian text, English text).

    Full consumption checks XML versus Moses cardinality, archive CRCs and row
    count. Call verify_raw once before iterating to establish input identity.
    """
    if corpus not in _EXPECTED_ROWS:
        raise ValueError(f"Unsupported corpus: {corpus!r}")
    path = Path(data_dir) / RAW_FILES[corpus][0]
    if corpus == "europarl":
        yield from _europarl(path)
        return

    base, lang = ("JRC-Acquis.en-pl", "pl") if corpus == "jrc" else ("UNPC.en-ru", "ru")
    rows = 0
    with ZipFile(path) as archive, archive.open(base + ".xml") as xml:
        with io.BufferedReader(archive.open(base + "." + lang), _BLOCK) as sources, \
             io.BufferedReader(archive.open(base + ".en"), _BLOCK) as targets:
            for key, count in _opus_groups(xml, corpus):
                for _ in range(count):
                    source, target = sources.readline(), targets.readline()
                    if not source or not target:
                        raise ValueError(f"{corpus}: text exhausted at XML alignment {rows + 1}")
                    rows += 1
                    yield key, source.rstrip(b"\r\n").decode("utf-8"), target.rstrip(b"\r\n").decode("utf-8")
            if sources.readline() or targets.readline():
                raise ValueError(f"{corpus}: additional text after {rows} XML alignments")
    if rows != _EXPECTED_ROWS[corpus]:
        raise ValueError(f"{corpus}: expected {_EXPECTED_ROWS[corpus]} pairs, parsed {rows}")
