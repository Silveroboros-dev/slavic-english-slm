"""Fetch the pinned NLLB checkpoint and tokenizer from the original Meta repository."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

MODEL_ID = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
FILES = {
    "pytorch_model.bin": (2460457927, "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42"),
    "sentencepiece.bpe.model": (4852054, "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"),
    "tokenizer.json": (17331176, "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"),
    "config.json": (846, "f9b4081d3d108e9d06532ea8d92b932c04ed3a15ff2f87e6982f8e14db51fbc5"),
    "generation_config.json": (189, "0bb604cdb8392649176935a447ce994128f170a6908c135a17d0e3dfb8113cb2"),
    "special_tokens_map.json": (3548, "992bd4ed610d644d6823081937bcc91bb8878dd556cea4ae5327f2480361330e"),
    "tokenizer_config.json": (564, "d1aa8c3697d3e35674f97b5b7e9c99d22b010f528e80140257d97316be90d044"),
}

def verify_file(path, spec):
    if path.stat().st_size != spec[0]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == spec[1]

def prepare_model(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def fetch(name):
        dest = directory / name
        spec = FILES[name]
        if dest.exists():
            if not verify_file(dest, spec):
                raise ValueError(f"Invalid existing {dest}. Move it aside before retrying.")
            return
        tmp = directory / (name + ".partial")
        if tmp.exists() and verify_file(tmp, spec):
            tmp.replace(dest)
            return
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true"
        req = Request(url, headers={"Accept-Encoding": "identity"})
        with urlopen(req, timeout=90) as source, tmp.open("wb") as f:
            shutil.copyfileobj(source, f, 1 << 20)
        if not verify_file(tmp, spec):
            raise ValueError(f"Downloaded bytes differ from pinned {name}")
        tmp.replace(dest)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, FILES))
    return directory

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    args = ap.parse_args()
    print(f"Verified {MODEL_ID}@{REVISION} at {prepare_model(args.directory)}")
