"""GPU-only R4 format intervention; original R3 recipe is imported unchanged."""

import argparse, gc, json, os, platform, shutil, time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import r3_recipe as recipe
from r3_common import EPOCHS, epoch_plan, put, sha, validate_rows
from model_snapshot import MODEL_ID, REVISION, FILES, prepare_model, verify_file

ARMS = ("digits", "words")
MODELS = ("base", "r3", "r4_digits", "r4_words")


def validate(root):
    # The public notebook never needs private training inputs. Historical replay
    # requires the exact independently pinned input manifest, before any model load.
    if (
        sha(root / "INPUTS.json")
        != "6182ed7ec8cc5d4f3bcd8ecacb5d01bdb54026aa026042365ecdd305c2fc128c"
    ):
        raise ValueError("Historical private input manifest digest differs")
    manifest = json.loads((root / "INPUTS.json").read_text())
    if manifest["format"] != "NLLB-r4-formats-v1":
        raise ValueError("Wrong payload format")
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Input checksum differs: {name}")
    original = json.loads((root / "original_r3/train_rows.json").read_text())
    old_plan = json.loads((root / "original_r3/epoch_plan.json").read_text())
    old_receipt = json.loads((root / "original_r3/training_receipt.json").read_text())
    if (
        len(original) != 2316
        or old_receipt["optimizer_updates"] != 450
        or recipe.TRAIN != old_receipt["contract"]["training"]
    ):
        raise ValueError("Original R3 recipe/size/updates differ")
    if sha(root / "r3_recipe.py") != old_receipt["contract"]["trainer_sha256"]:
        raise ValueError("Original R3 trainer changed")
    if sha(root / "r3_common.py") != old_receipt["contract"]["common_sha256"]:
        raise ValueError("Original R3 scheduling/alignment changed")
    rows_by_arm, plans, summaries = {}, None, None
    for arm in ARMS:
        rows = json.loads((root / arm / "train_rows.json").read_text())
        # validate_rows derives wrong_tgt from pair partner; check the supplied
        # values before and after so corruption cannot be silently repaired.
        before = json.loads(json.dumps(rows))
        validate_rows(rows)
        if rows != before or len(rows) != len(original):
            raise ValueError(f"{arm}: malformed matched targets or row count")
        for old, new in zip(original, rows, strict=True):
            if old["origin"] != "clock":
                if new != old:
                    raise ValueError(f"{arm}: nonclock row changed")
            elif {k: v for k, v in old.items() if k not in ("tgt", "wrong_tgt")} != {
                k: v for k, v in new.items() if k not in ("tgt", "wrong_tgt")
            }:
                raise ValueError(f"{arm}: clock field other than tgt/wrong_tgt changed")
        current_plans, current_summaries = [], []
        for epoch in range(1, EPOCHS + 1):
            batches, summary = epoch_plan(rows, epoch)
            current_plans.extend(batches)
            current_summaries.append(summary)
        if (
            current_plans != old_plan["batches"]
            or current_summaries != old_plan["epochs"]
            or len(current_plans) != 450
        ):
            raise ValueError(f"{arm}: exposure plan differs from actual R3 run")
        rows_by_arm[arm] = rows
        plans, summaries = current_plans, current_summaries
    holdout = json.loads((root / "holdout_inputs.json").read_text())
    cases = json.loads((root / "evaluation_inputs.json").read_text())
    metadata = json.loads((root / "evaluation_metadata.json").read_text())
    for records in (holdout, cases):
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Duplicate evaluation ID")
        for r in records:
            if (
                set(r) != {"id", "lang", "source"}
                or r["lang"] not in recipe.LANG
                or not all(isinstance(r[k], str) and r[k].strip() for k in r)
            ):
                raise ValueError("Evaluation inputs must be source-only")
    if set(metadata) != {r["id"] for r in cases}:
        raise ValueError("Evaluation metadata differs")
    if cases[: len(holdout)] != holdout:
        raise ValueError("New holdout altered during assembly")
    expected_train = [
        {"id": "trainclock::" + r["id"], "lang": r["lang"], "source": r["src"]}
        for r in original
        if r["origin"] == "clock"
    ]
    if cases[len(holdout) :] != expected_train or len(expected_train) != 108:
        raise ValueError("Training-clock diagnostics differ from the original 108 rows")
    for arm, rows in rows_by_arm.items():
        train_sources = {(r["lang"], " ".join(r["src"].split()).casefold()) for r in rows}
        if any(
            (r["lang"], " ".join(r["source"].split()).casefold()) in train_sources for r in holdout
        ):
            raise ValueError(f"{arm}: new holdout overlaps training sources")
    if any(
        metadata[r["id"]]["role"] not in ("new_holdout", "historical_c3_singleton") for r in holdout
    ) or any(metadata[r["id"]]["role"] != "training_clock_diagnostic" for r in expected_train):
        raise ValueError("Holdout/diagnostic role labels differ")
    return manifest, rows_by_arm, cases, metadata, plans, summaries, old_receipt


def validate_completed_arm(out, arm, expected_initial):
    directory = out / arm
    receipt_path = directory / "training_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["optimizer_updates"] != 450
        or receipt["initial_parameter_checks"]["fingerprint"] != expected_initial
    ):
        raise ValueError(f"{arm}: training receipt differs")
    if any(
        sha(directory / "final_adapter" / name) != digest
        for name, digest in receipt["adapter_files"].items()
    ):
        raise ValueError(f"{arm}: completed adapter checksum differs")
    return receipt


def infer(name, adapter, cases, out, model_dir, tok, root):
    """Greedy float32, one common source-only cohort, atomic batch recovery."""
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from peft import PeftModel

    identity = "base:" + REVISION if adapter is None else sha(adapter / "adapter_model.safetensors")
    config = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": 256,
        "forced_target": "eng_Latn",
        "batch_size": 2,
        "dtype": "float32",
        "source_truncation": False,
        "use_cache": True,
    }
    contract = {
        "model": name,
        "model_identity": identity,
        "evaluation_inputs_sha256": sha(root / "evaluation_inputs.json"),
        "decoding": config,
        "script_sha256": sha(Path(__file__)),
    }
    path = out / f"{name}_greedy_predictions.json"
    receipt_path = out / f"{name}_greedy_prediction_receipt.json"
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        values = json.loads(path.read_text())
        if (
            receipt["contract"] != contract
            or receipt["prediction_sha256"] != sha(path)
            or [r["id"] for r in values] != [r["id"] for r in cases]
        ):
            raise ValueError(f"{name}: invalid completed inference cache")
        print(f"{name}: verified complete greedy translations; skipped model load.", flush=True)
        return receipt
    partial = out / f"{name}_greedy_partial.json"
    pred = {}
    elapsed_prior = 0
    if partial.exists():
        cache = json.loads(partial.read_text())
        if cache["contract"] != contract or not set(cache["predictions"]).issubset(
            {r["id"] for r in cases}
        ):
            raise ValueError(f"{name}: partial inference identity differs")
        pred = cache["predictions"]
        elapsed_prior = cache.get("elapsed_seconds", 0)
        if any(not isinstance(t, str) or not t.strip() for t in pred.values()):
            raise ValueError("Invalid cached translation")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=False, dtype=torch.float32
    )
    model = (
        base
        if adapter is None
        else PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    )
    model = model.cuda().eval()
    recipe.runtime_seed()
    english = tok.convert_tokens_to_ids("eng_Latn")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for lang, tag in recipe.LANG.items():
                entries = [r for r in cases if r["lang"] == lang and r["id"] not in pred]
                tok.src_lang = tag
                tok.tgt_lang = "eng_Latn"
                for offset in range(0, len(entries), 2):
                    batch = entries[offset : offset + 2]
                    enc = tok(
                        [r["source"] for r in batch],
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    if enc.input_ids.shape[1] > 512 or not torch.all(
                        enc.input_ids[:, 0] == tok.convert_tokens_to_ids(tag)
                    ):
                        raise ValueError("Invalid source tokenization; truncation forbidden")
                    seq = model.generate(
                        **enc.to("cuda"),
                        do_sample=False,
                        num_beams=1,
                        num_return_sequences=1,
                        forced_bos_token_id=english,
                        max_new_tokens=256,
                        use_cache=True,
                    )
                    if seq.shape[1] < 2 or not torch.all(seq[:, 1] == english):
                        raise ValueError("Wrong generated target language")
                    text = tok.batch_decode(seq, skip_special_tokens=True)
                    for row, translation in zip(batch, text, strict=True):
                        if not translation.strip():
                            raise ValueError("Empty generated translation")
                        pred[row["id"]] = translation
                    put(
                        partial,
                        {
                            "contract": contract,
                            "predictions": pred,
                            "elapsed_seconds": elapsed_prior + time.monotonic() - start,
                        },
                    )
                    print(
                        f"{name} greedy: {len(pred)}/{len(cases)} raw translations (new holdout + training diagnostics)",
                        flush=True,
                    )
        if set(pred) != {r["id"] for r in cases}:
            raise ValueError("Missing/extra prediction")
        values = [{"id": r["id"], "lang": r["lang"], "translation": pred[r["id"]]} for r in cases]
        put(path, values)
        receipt = {
            "contract": contract,
            "rows": len(values),
            "prediction_sha256": sha(path),
            "empty_translations": 0,
            "elapsed_inference_seconds": elapsed_prior + time.monotonic() - start,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        put(receipt_path, receipt)
        partial.unlink(missing_ok=True)
        return receipt
    finally:
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


def run(root, out):
    import torch, transformers, peft
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Choose exactly one CUDA GPU runtime in Colab.")
    if (
        transformers.__version__ != "4.57.1"
        or peft.__version__ != "0.17.1"
        or version("tokenizers") != "0.22.2"
        or version("sentencepiece") != "0.2.1"
        or tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6)
    ):
        raise RuntimeError("Dependency lock differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic CUDA environment missing")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    memory = torch.cuda.get_device_properties(0).total_memory
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "tokenizers": version("tokenizers"),
        "sentencepiece": version("sentencepiece"),
        "safetensors": version("safetensors"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": memory,
        "amp_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "micro_batch": 4,
        "accumulation": 8,
        "deterministic_algorithms": True,
        "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    manifest, rows, cases, metadata, plans, epochs, old_receipt = validate(root)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"format": "NLLB-r4-formats-v1", "inputs_manifest_sha256": sha(root / "INPUTS.json")}
    if (out / "RUN_IDENTITY.json").exists() and json.loads(
        (out / "RUN_IDENTITY.json").read_text()
    ) != identity:
        raise ValueError("Recovery belongs to another payload")
    put(out / "RUN_IDENTITY.json", identity)
    put(out / "runtime.json", runtime)
    put(out / "evaluation_inputs.json", cases)
    put(out / "evaluation_metadata.json", metadata)
    put(out / "epoch_plan.json", {"epochs": epochs, "batches": plans})
    print(json.dumps(runtime, indent=2), flush=True)
    print(
        "R4 controlled run: DIGITS and WORDS each fresh base + identical seeded LoRA; 450 updates each, same original R3 exposure plan. Only 108 clock targets/wrong targets differ.",
        flush=True,
    )
    print(
        f"Greedy evaluation: {sum(v['role'] == 'new_holdout' for v in metadata.values())} NEW holdout + {sum(v['role'] == 'historical_c3_singleton' for v in metadata.values())} historical C3 singletons +108 TRAINING-CLOCK diagnostics ×4 models. No redundant historical cohort calls.",
        flush=True,
    )
    model_dir = prepare_model(root / "model")
    if any(not verify_file(model_dir / n, spec) for n, spec in FILES.items()):
        raise ValueError("Pinned base checksum differs")
    tok = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, legacy_behaviour=False, local_files_only=True, tgt_lang="eng_Latn"
    )
    expected_initial = old_receipt["initial_parameter_checks"]["fingerprint"]
    original_inspect = recipe.inspect

    def controlled_inspect(model, initial=None):
        checks = original_inspect(model, initial)
        if initial is None and checks["fingerprint"] != expected_initial:
            raise ValueError(
                "Fresh initialization differs from saved R3 initial fingerprint; stopped before updates"
            )
        return checks

    recipe.inspect = controlled_inspect
    receipts = {}
    started = time.monotonic()
    # Do not call recipe.run(): its historical multi-beam evaluation is excluded.
    for arm in ARMS:
        directory = out / arm
        directory.mkdir(exist_ok=True)
        print(f"R4_ARM_START {arm}: 450 updates; fresh original R3 initialization.", flush=True)
        receipts[arm] = recipe.train(
            rows[arm], plans, epochs, root / arm, directory, model_dir, tok, runtime
        )
        validate_completed_arm(out, arm, expected_initial)
        put(
            out / "PROGRESS.json",
            {
                "status": "training",
                "finished_arms": list(receipts),
                "total_training_updates": sum(r["optimizer_updates"] for r in receipts.values()),
                "payload_identity": identity,
            },
        )
        print("R4_ARM_COMPLETE " + arm, flush=True)
    if (
        receipts["digits"]["initial_parameter_checks"]["fingerprint"]
        != receipts["words"]["initial_parameter_checks"]["fingerprint"]
    ):
        raise ValueError("R4 arm initializations differ")
    adapters = {
        "base": None,
        "r3": root / "r3_adapter",
        "r4_digits": out / "digits/final_adapter",
        "r4_words": out / "words/final_adapter",
    }
    prediction_receipts = []
    for name in MODELS:
        prediction_receipts.append(infer(name, adapters[name], cases, out, model_dir, tok, root))
        print("R4_INFERENCE_COMPLETE " + name, flush=True)
    weights = {
        "r3": {
            n: sha(root / "r3_adapter" / n)
            for n in ("adapter_config.json", "adapter_model.safetensors")
        }
    }
    weights.update({"r4_" + arm: receipts[arm]["adapter_files"] for arm in ARMS})
    summary = {
        "status": "complete",
        "format": "NLLB-r4-formats-v1",
        "base_model": MODEL_ID,
        "base_revision": REVISION,
        "base_weight_sha256": FILES["pytorch_model.bin"][1],
        "training": recipe.TRAIN,
        "optimizer_updates_per_arm": 450,
        "total_optimizer_updates": 900,
        "training_unique_rows_per_arm": 2316,
        "clock_training_rows_per_arm": 108,
        "training_receipts": {arm: sha(out / arm / "training_receipt.json") for arm in ARMS},
        "initial_fingerprint": expected_initial,
        "same_initialization_across_arms": True,
        "same_r3_epoch_plan": True,
        "inputs_manifest_sha256": sha(root / "INPUTS.json"),
        "new_holdout_inputs_sha256": sha(root / "holdout_inputs.json"),
        "runtime": runtime,
        "model_weight_checksums": weights,
        "prediction_runs": len(prediction_receipts),
        "predictions": sum(r["rows"] for r in prediction_receipts),
        "prediction_receipts": prediction_receipts,
        "evaluation_role_counts": dict(Counter(v["role"] for v in metadata.values())),
        "elapsed_execution_seconds_this_invocation": time.monotonic() - started,
        "training_initialization": "Fresh pinned frozen NLLB plus identical seed42 LoRA, never the R3 adapter",
        "validation_limit": "Quality is unknown until returned raw translations are scored outside this training payload. No scoring keys are present. Different GPU stacks may produce different numerical results.",
    }
    put(out / "SUMMARY.json", summary)
    put(
        out / "PROGRESS.json",
        {"status": "complete", "finished_arms": list(ARMS), "payload_identity": identity},
    )
    print(
        f"R4_COMPLETE: 900 optimizer updates; {summary['predictions']} raw greedy translations across four models.",
        flush=True,
    )


def public_demo(data, out):
    """Two new optimizer updates on public B01: a mechanics demo, not R4 replay.

    Reuses the historical tokenization, objective, LoRA and optimizer code.
    All public B01 records used here become training data, not a held-out test.
    """
    import torch, transformers, peft
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Demo requires exactly one CUDA GPU")
    if (
        transformers.__version__ != "4.57.1"
        or peft.__version__ != "0.17.1"
        or version("tokenizers") != "0.22.2"
        or version("sentencepiece") != "0.2.1"
        or tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6)
    ):
        raise RuntimeError(
            "Use torch >=2.6, transformers 4.57.1, peft 0.17.1, tokenizers 0.22.2, sentencepiece 0.2.1"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rows = []
    for line in data.read_text().splitlines():
        r = json.loads(line)
        if r["lang"] not in ("pl", "hr", "ru"):
            continue
        for variant, other in (("A", "B"), ("B", "A")):
            v = r["variants"][variant]
            rows.append(
                {
                    "id": r["scenario_id"] + "-" + variant,
                    "lang": r["lang"],
                    "src": v["source"],
                    "tgt": v["reference_en"],
                    "wrong_tgt": r["variants"][other]["reference_en"],
                    "origin": "B01",
                    "pair_id": r["scenario_id"],
                    "variant": variant,
                }
            )
        if len(rows) >= 30:
            break
    if len(rows) != 30:
        raise ValueError("Demo requires at least 15 public contrast pairs")
    # Two plain-CE controls reuse public text; no invented historical conversations.
    rows += [
        dict(
            rows[i], id="demo-plain-" + str(i), origin="conversation", wrong_tgt=None, pair_id=None
        )
        for i in (0, 1)
    ]
    out.mkdir(parents=True, exist_ok=True)
    root = out / "public_demo_inputs"
    root.mkdir(exist_ok=True)
    put(root / "train_rows.json", rows)
    put(
        root / "INPUTS.json",
        {
            "mode": "public-mechanics-demo",
            "public_data_sha256": sha(data),
            "updates": 2,
            "not_historical_replay": True,
        },
    )
    for name in ("conversation_provenance.json", "data_receipt.json", "clock.receipt.json"):
        put(
            root / name,
            {
                "scope": "public B01 mechanics demonstration; no private conversations or clock training",
                "public_data_sha256": sha(data),
            },
        )
    shutil.copyfile(Path(__file__).with_name("r3_common.py"), root / "r3_common.py")
    plans = [
        [{"id": r["id"], "exposure": "demo_repeat" if step else "demo_first"} for r in rows]
        for step in range(2)
    ]
    epochs = [
        {
            "epoch": 1,
            "steps": 2,
            "exposures": 64,
            "scope": "public demonstration; not historical epoch policy",
        }
    ]
    recipe.TRAIN = dict(
        recipe.TRAIN,
        epochs=1,
        checkpoint_every_updates=2,
        epoch_policy="Public demonstration: fixed 32 rows repeated for two updates; no historical balance claim",
    )
    runtime = {
        "amp_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "micro_batch": 4,
        "accumulation": 8,
        "mode": "public-mechanics-demo",
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "tokenizers": version("tokenizers"),
        "sentencepiece": version("sentencepiece"),
        "safetensors": version("safetensors"),
        "gpu": torch.cuda.get_device_name(0),
    }
    model_dir = prepare_model(out / "model")
    tok = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, legacy_behaviour=False, local_files_only=True, tgt_lang="eng_Latn"
    )
    return recipe.train(rows, plans, epochs, root, out, model_dir, tok, runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Historical R4 replay or public two-update GPU mechanics demo"
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--public-demo", type=Path, help="Public data/b01_contrasts.jsonl; not historical replay"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.public_demo):
        parser.error("Choose exactly one of --inputs or --public-demo")
    if args.public_demo:
        public_demo(args.public_demo, args.output)
    else:
        run(args.inputs, args.output)
