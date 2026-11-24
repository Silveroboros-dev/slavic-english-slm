"""Benchmark one prepared CTranslate2 INT8 model in a fresh Colab CPU process."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


CASE_SHA256 = "8c597d695a71ee16261d048d11ae3bdeb68a44828ed2141de136b04df85232c0"
CHAT_TEMPLATE_SHA256 = "5437d0ad30bb8db69f45cd62c6e02d4269ee617ec9637c9650f8f04051bd9dbc"
LANG_NAMES = {"pl": "Polish", "hr": "Croatian", "ru": "Russian", "uk": "Ukrainian"}
LANG_CODES = {"pl": "pol_Latn", "hr": "hrv_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl"}
PROMPT = "Translate the following {language} message into English. Return only the translation."
MAX_NEW_TOKENS = 256


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_cases(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA256:
        raise ValueError("Unexpected Norm64 case file. Use the frozen repository JSONL.")
    cases = []
    for line in raw.splitlines():
        record = json.loads(line)
        fields = ("id", "lang", "pair_id", "variant", "normalized_input", "normalized_input_sha256")
        case = {field: record[field] for field in fields}
        if sha(case["normalized_input"]) != case["normalized_input_sha256"]:
            raise ValueError(f"Input text digest differs for {case['id']}")
        cases.append(case)
    if Counter(c["lang"] for c in cases) != dict.fromkeys(LANG_CODES, 16):
        raise ValueError("Expected 16 messages per language.")
    return cases


def model_files(directory):
    """List logical deployment bytes. Hash the large binary after timing."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = path.stat().st_size
    if "model.bin" not in files or "tokenizer/tokenizer_config.json" not in files:
        raise ValueError("Expected a CT2 deployment folder with a tokenizer/ subdirectory.")
    return files


def weight_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_description():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.machine()


def build_runner(system, directory, threads):
    import ctranslate2
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    create_model = ctranslate2.Translator if system == "nllb_r4" else ctranslate2.Generator
    engine = create_model(
        str(directory), device="cpu", compute_type="int8", inter_threads=1, intra_threads=threads
    )
    if engine.device != "cpu" or engine.compute_type != "int8_float32":
        raise RuntimeError(f"Expected CPU int8_float32, got {engine.device}/{engine.compute_type}")

    if system == "nllb_r4":

        def translate(lang, source):
            tokenizer.src_lang = LANG_CODES[lang]
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(source))
            if tokens[0] != LANG_CODES[lang]:
                raise ValueError(f"Missing NLLB source language token for {lang}")
            result = engine.translate_batch(
                [tokens],
                target_prefix=[["eng_Latn"]],
                beam_size=1,
                max_decoding_length=MAX_NEW_TOKENS,
                min_decoding_length=0,
                max_input_length=0,
            )[0]
            hypothesis = result.hypotheses[0]
            if hypothesis[:1] != ["eng_Latn"]:
                raise ValueError("Missing NLLB target language prefix.")
            ids = tokenizer.convert_tokens_to_ids(hypothesis[1:])
            return tokenizer.decode(ids, skip_special_tokens=True), len(ids)
    else:
        if sha(tokenizer.chat_template) != CHAT_TEMPLATE_SHA256:
            raise ValueError("Gemma chat template differs from the frozen baseline.")
        stop_tokens = tokenizer.convert_ids_to_tokens(sorted({tokenizer.eos_token_id, 106}))
        if "<end_of_turn>" not in stop_tokens:
            raise ValueError("Gemma token 106 must match the recorded turn terminator.")

        def translate(lang, source):
            content = PROMPT.format(language=LANG_NAMES[lang]) + "\n\n" + source
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
            )
            tokens = tokenizer.convert_ids_to_tokens(ids)
            result = engine.generate_batch(
                [tokens],
                beam_size=1,
                sampling_topk=1,
                include_prompt_in_result=False,
                max_length=MAX_NEW_TOKENS,
                min_length=0,
                end_token=stop_tokens,
            )[0]
            generated = result.sequences_ids[0]
            return tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ), len(generated)

    return translate, engine.compute_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=("nllb_r4", "gemma_1b"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Requires the Linux Colab runtime for consistent RSS units.")
    available = sorted(os.sched_getaffinity(0))
    if not 1 <= args.threads <= len(available):
        parser.error(f"Choose 1 to {len(available)} CPU threads.")
    os.sched_setaffinity(0, available[: args.threads])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    cases = read_cases(args.cases)
    files = model_files(args.model_dir)
    started = time.perf_counter()
    translate, actual_compute_type = build_runner(args.system, args.model_dir, args.threads)
    load_seconds = time.perf_counter() - started

    # One identical, off-panel warm-up. All 64 scored calls use batch size one.
    warmup = time.perf_counter()
    translate("pl", "Biblioteka jest dziś otwarta.")
    warmup_seconds = time.perf_counter() - warmup
    predictions = []
    for case in cases:
        started = time.perf_counter()
        translation, generated_tokens = translate(case["lang"], case["normalized_input"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        row = {
            field: case[field]
            for field in ("id", "pair_id", "variant", "lang", "normalized_input_sha256")
        }
        row.update(
            translation=translation, latency_ms=elapsed_ms, generated_tokens=generated_tokens
        )
        predictions.append(row)
        if len(predictions) % 8 == 0:
            print(f"{args.system}: {len(predictions)}/{len(cases)}", flush=True)

    # Linux ru_maxrss is peak resident size for this process in KiB. Take it
    # before hashing artifacts or serializing the final report.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    times = sorted(r["latency_ms"] for r in predictions)
    receipt = {
        "system": args.system,
        "complete": True,
        "case_file_sha256": CASE_SHA256,
        "cpu": cpu_description(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "vm_boot_id_sha256": sha(Path("/proc/sys/kernel/random/boot_id").read_text().strip()),
        "intra_threads": args.threads,
        "inter_threads": 1,
        "device": "cpu",
        "actual_compute_type": actual_compute_type,
        "max_new_tokens": MAX_NEW_TOKENS,
        "beam_size": 1,
        "batch_size": 1,
        "versions": {
            k: importlib.metadata.version(k) for k in ("ctranslate2", "transformers", "tokenizers")
        },
        "load_seconds_including_imports": load_seconds,
        "off_panel_warmup_seconds": warmup_seconds,
        "timing_scope": "One timed call per message, including tokenizer, generation and detokenization. Print and input loading excluded.",
        "total_warm_inference_seconds": sum(times) / 1000,
        "mean_latency_ms": statistics.mean(times),
        "median_latency_ms": statistics.median(times),
        "p95_latency_ms_nearest_rank": times[math.ceil(0.95 * len(times)) - 1],
        "peak_process_rss_bytes": peak_rss,
        "package_bytes": sum(files.values()),
        "package_file_sizes": files,
        "ct2_weight_sha256": weight_sha256(args.model_dir / "model.bin"),
        "predictions": predictions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.system}.json"
    pending = destination.with_suffix(".json.tmp")
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(destination)
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
