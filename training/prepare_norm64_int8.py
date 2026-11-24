"""Build two local CTranslate2 INT8 packages from the pinned Norm64 checkpoints."""

import argparse
import gc
import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path


NLLB = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
GEMMA = "google/gemma-3/transformers/gemma-3-1b-it/1"
CHAT_SHA256 = "5437d0ad30bb8db69f45cd62c6e02d4269ee617ec9637c9650f8f04051bd9dbc"
QUANTIZATION = "int8_float32"
RECEIPT = "preparation.json"
NLLB_HASHES = {
    "pytorch_model.bin": "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42",
    "sentencepiece.bpe.model": "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a",
    "tokenizer.json": "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665",
    "config.json": "f9b4081d3d108e9d06532ea8d92b932c04ed3a15ff2f87e6982f8e14db51fbc5",
    "generation_config.json": "0bb604cdb8392649176935a447ce994128f170a6908c135a17d0e3dfb8113cb2",
    "special_tokens_map.json": "992bd4ed610d644d6823081937bcc91bb8878dd556cea4ae5327f2480361330e",
    "tokenizer_config.json": "d1aa8c3697d3e35674f97b5b7e9c99d22b010f528e80140257d97316be90d044",
}
ADAPTER_HASHES = {
    "adapter_config.json": "16a60aa410763b49f9331f5ef41b7c3c038177b4144e5d82c0728669738acabd",
    "adapter_model.safetensors": "529053d6b875e72483d64dab8b3d8b0e37a7900ebf4683a4e860966f7493b4c0",
}
GEMMA_HASHES = {
    "model.safetensors": "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6",
    "config.json": "4dd5218cebc4fdea9004b944d34672a2d47f9ef930e3bb16a3fc4bbbe0fcee81",
    "tokenizer.json": "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "generation_config.json": "5aebcd66c28ab91bc09f1a5c7d1646e4d015c9eae11bf58e5eff92e97ef59bea",
    "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": "3aeb8e9ca0876029b5b4c70499c1ecc5c8848d9ef721d1512e067f6242c59fa1",
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(directory, digests):
    for name, expected in digests.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch for {name}.")


def merge_nllb(stage, adapter):
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    verify(adapter, ADAPTER_HASHES)
    print("Verified the saved R4 digits adapter. Fetching the pinned NLLB release.", flush=True)
    base = Path(
        snapshot_download(NLLB, revision=REVISION, allow_patterns=list(NLLB_HASHES), token=False)
    )
    verify(base, NLLB_HASHES)
    model = AutoModelForSeq2SeqLM.from_pretrained(base, local_files_only=True, dtype=torch.float32)
    merged = PeftModel.from_pretrained(model, adapter, is_trainable=False).merge_and_unload(
        safe_merge=True
    )
    destination = stage / "merged"
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, use_fast=True, legacy_behaviour=False, tgt_lang="eng_Latn"
    )
    tokenizer.save_pretrained(destination)
    return destination


def gemma_checkpoint(local_directory):
    if local_directory is None:
        import kagglehub

        local_directory = Path(kagglehub.model_download(GEMMA))
    matches = [
        p.parent
        for p in local_directory.glob("**/model.safetensors")
        if (p.parent / "config.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one original Gemma 3 1B-IT checkpoint in the supplied directory."
        )
    verify(matches[0], GEMMA_HASHES)
    return matches[0]


def package_digests(directory):
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != RECEIPT
    }


def prepare(system, args):
    import ctranslate2
    from transformers import AutoTokenizer

    if ctranslate2.__version__ != "4.9.1" or version("transformers") != "4.57.1":
        raise RuntimeError("This benchmark requires CTranslate2 4.9.1 and Transformers 4.57.1.")
    contract = {
        "system": system,
        "source": f"{NLLB}@{REVISION}" if system == "nllb_r4" else GEMMA,
        "source_sha256": NLLB_HASHES if system == "nllb_r4" else GEMMA_HASHES,
        "adapter_sha256": ADAPTER_HASHES if system == "nllb_r4" else None,
        "conversion": QUANTIZATION,
        "preparation_script_sha256": sha256(Path(__file__)),
        "versions": {
            name: version(name)
            for name in ("ctranslate2", "transformers", "peft", "torch", "tokenizers")
        },
    }
    target = args.work_dir / system
    if target.exists():
        previous = json.loads((target / RECEIPT).read_text())
        if (
            previous["contract"] != contract
            or package_digests(target) != previous["package_sha256"]
        ):
            raise ValueError(f"Existing {system} package differs. Use a fresh work directory.")
        print(f"{system}: reusing verified INT8 package.", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix=system + "-", dir=args.work_dir) as tmp:
        stage = Path(tmp)
        if system == "nllb_r4":
            checkpoint = merge_nllb(stage, args.adapter_dir)
        else:
            checkpoint = gemma_checkpoint(args.gemma_dir)
        gc.collect()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
        if system == "gemma_1b":
            chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
            if chat_hash != CHAT_SHA256 or tokenizer.convert_tokens_to_ids("<end_of_turn>") != 106:
                raise ValueError("Chat rendering does not match the recorded Gemma run.")
        package = stage / "package"
        print(f"{system}: converting verified weights to {QUANTIZATION}.", flush=True)
        converter = ctranslate2.converters.TransformersConverter(
            str(checkpoint), low_cpu_mem_usage=True
        )
        converter.convert(str(package), quantization=QUANTIZATION)
        tokenizer.save_pretrained(package / "tokenizer")
        digests = package_digests(package)
        if "model.bin" not in digests or "config.json" not in digests:
            raise ValueError("Converted model is incomplete.")
        receipt = {"contract": contract, "package_sha256": digests}
        (package / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        package.rename(target)
        print(f"{system}: verified runtime package ready at {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--gemma-dir", type=Path, help="Existing byte-identical original Kaggle checkpoint."
    )
    parser.add_argument("--system", choices=("nllb_r4", "gemma_1b"))
    args = parser.parse_args()
    if args.system != "gemma_1b" and not args.adapter_dir:
        parser.error(
            "--adapter-dir containing the original R4 digits adapter is required to prepare NLLB."
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for system in [args.system] if args.system else ["nllb_r4", "gemma_1b"]:
        prepare(system, args)


if __name__ == "__main__":
    main()
