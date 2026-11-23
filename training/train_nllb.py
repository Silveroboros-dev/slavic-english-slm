"""One non-commercial research run on the frozen PL/RU/HR -> EN train split."""
import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from uuid import uuid4

import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from model_snapshot import FILES as MODEL_FILES, MODEL_ID as MODEL, REVISION as REV, verify_file
TRAIN_SHA = "13f1882f113120dfd09d7e5ef7f0c7c834751856266918330d1190741013cf27"
LANG = {"pl": "pol_Latn", "ru": "rus_Cyrl", "hr": "hrv_Latn"}
TARGET = "eng_Latn"
SEED, MICRO, ACCUM, MAX_TOKENS = 42, 4, 8, 256
EFFECTIVE = MICRO * ACCUM
ROWS = 300000
LR, WEIGHT_DECAY, RANK, ALPHA, LORA_DROPOUT = 1e-4, 0.01, 8, 16, 0.05
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
FINAL_MARKER = "TRAINING_COMPLETE.json"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path):
    for name, expected in MODEL_FILES.items():
        if not verify_file(path / name, expected):
            raise ValueError(f"Model file differs from revision {REV}: {name}")
    # Prevent a local adapter or added-token file overriding the verified snapshot.
    extras = {"adapter_config.json", "added_tokens.json", "config_sentence_transformers.json"}
    if any((path / name).exists() for name in extras):
        raise ValueError("Unexpected model override file in the pinned snapshot")


def read_train(data_dir):
    path = data_dir / "train.jsonl.gz"
    if file_hash(path) != TRAIN_SHA:
        raise ValueError("train.jsonl.gz differs from the frozen pilot")
    rows = []
    counts = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            lang = record["lang"]
            rows.append((LANG[lang], record["src"], record["tgt"]))
            counts[lang] += 1
    if counts != {"pl": 100000, "ru": 100000, "hr": 100000}:
        raise ValueError(f"Unexpected train language counts: {counts}")
    return rows


def encode_update(tokenizer, rows, max_tokens=MAX_TOKENS):
    """Batch tokenization by source tag, then restore the global shuffled order."""
    groups = {}
    for i, (lang, source, target) in enumerate(rows):
        groups.setdefault(lang, []).append((i, source, target))
    encoded = [None] * len(rows)
    trunc = {"source": 0, "target": 0}
    for lang, group in groups.items():
        tokenizer.src_lang, tokenizer.tgt_lang = lang, TARGET
        batch = tokenizer([a for _, a, _ in group], text_target=[b for _, _, b in group],
                          truncation=False, padding=False, return_attention_mask=False)
        for (i, _, _), src, tgt in zip(group, batch["input_ids"], batch["labels"], strict=True):
            for name, ids, tag in (("source", src, lang), ("target", tgt, TARGET)):
                tag_id = tokenizer.convert_tokens_to_ids(tag)
                if tag_id == tokenizer.unk_token_id or ids[0] != tag_id or ids[-1] != tokenizer.eos_token_id:
                    raise ValueError(f"Unexpected native NLLB token sequence for {tag}")
                if len(ids) > max_tokens:
                    trunc[name] += 1
                    ids[max_tokens - 1:] = [tokenizer.eos_token_id]
            encoded[i] = (src, tgt)
    return encoded, trunc


def tensors(records, pad, device):
    n = len(records)
    width = max(len(s) for s, _ in records)
    length = max(len(t) for _, t in records)
    ids = torch.full((n, width), pad, dtype=torch.long)
    labels = torch.full((n, length), -100, dtype=torch.long)
    for i, (s, t) in enumerate(records):
        ids[i, :len(s)] = torch.tensor(s)
        labels[i, :len(t)] = torch.tensor(t)
    return {"input_ids": ids.to(device), "attention_mask": ids.ne(pad).long().to(device),
            "labels": labels.to(device)}


def optimizer_update(model, records, pad, device, dtype, optimizer, scheduler, scaler, micro=MICRO):
    # Weight each microbatch by valid target tokens to match one full update.
    tokens = sum(len(target) for _, target in records)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for _attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for start in range(0, len(records), micro):
            chunk = records[start:start + micro]
            frac = sum(len(t) for _, t in chunk) / tokens
            amp = torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else nullcontext()
            with amp:
                loss = model(**tensors(chunk, pad, device), use_cache=False).loss * frac
            loss_sum += loss.detach()
            scaler.scale(loss).backward()
            del loss
        if not torch.isfinite(loss_sum).item():
            raise FloatingPointError("Nonfinite training loss")
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not scaler.is_enabled() and not torch.isfinite(norm).item():
            raise FloatingPointError("Nonfinite unscaled gradient")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            return loss_sum.item()
    raise FloatingPointError("Repeated FP16 overflows while retrying the same examples")


def adapter_model(base):
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Every encoder self-attention and both decoder attention blocks have q_proj/v_proj.
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=RANK,
                           lora_alpha=ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                           target_modules=["q_proj", "v_proj"]))
    cfg = model.peft_config["default"]
    cfg.base_model_name_or_path, cfg.revision = MODEL, REV
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for _, p in trainable):
        raise ValueError("LoRA trainable parameters must be float32 for AMP updates")
    weights = [name for name, _ in trainable]
    target_count = 2 * (base.config.encoder_layers + 2 * base.config.decoder_layers)
    if len(weights) != 2 * target_count or any(".lora_" not in n for n in weights):
        raise ValueError("LoRA target modules differ from expected M2M100 attention projections")
    model.train()
    return model


def verified_marker(directory, marker, contract):
    meta = json.loads((directory / marker).read_text())
    if meta["contract"] != contract:
        raise RuntimeError(f"Incompatible training protocol in {directory}")
    expected = set(ADAPTER_FILES)
    if marker != FINAL_MARKER:
        expected.add("state.pt")
    if set(meta["files"]) != expected:
        raise ValueError(f"Incomplete file manifest in {directory}")
    for name, value in meta["files"].items():
        if file_hash(directory / name) != value:
            raise ValueError(f"Checksum mismatch in {directory.name}/{name}")
    step = meta["optimizer_steps"]
    if not isinstance(step, int) or step < 1 or step > math.ceil(contract["rows"] / contract["effective_batch"]):
        raise ValueError(f"Invalid checkpoint step in {directory}")
    if meta["rows_seen"] != min(step * contract["effective_batch"], contract["rows"]):
        raise ValueError(f"Invalid checkpoint cursor in {directory}")
    if marker == FINAL_MARKER and meta["rows_seen"] != contract["rows"]:
        raise ValueError("Incomplete final adapter")
    return meta


def find_checkpoint(directory, contract):
    found = sorted((p for p in directory.iterdir() if re.fullmatch(r"step-\d{6}", p.name)), reverse=True)
    for path in found:
        try:
            info = verified_marker(path, "CHECKPOINT.json", contract)
            if int(path.name[5:]) != info["optimizer_steps"]:
                raise ValueError("Checkpoint directory and cursor differ")
            return path, info
        except (OSError, ValueError, KeyError) as exc:
            print(f"Ignoring incomplete checkpoint {path.name}: {exc}", flush=True)
    if found:
        raise RuntimeError("No valid saved checkpoint remains. Prior files were preserved.")
    return None, None


def persist(path, model, contract, step, seen, trunc, state=None):
    marker = FINAL_MARKER if state is None else "CHECKPOINT.json"
    if path.exists():
        try:
            existing = verified_marker(path, marker, contract)
        except (OSError, ValueError, KeyError):
            # Keep damaged snapshots available for investigation while replaying from the prior good one.
            path.rename(path.with_name(f".damaged-{path.name}-{uuid4().hex}"))
        else:
            if existing["optimizer_steps"] != step:
                raise ValueError("Existing checkpoint step differs")
            return
    temp = Path(tempfile.mkdtemp(prefix=".writing-", dir=path.parent))
    try:
        model.save_pretrained(temp, safe_serialization=True, save_embedding_layers=False)
        names = list(ADAPTER_FILES)
        if state is not None:
            torch.save(state, temp / "state.pt")
            names.append("state.pt")
        meta = {"base_model": MODEL, "base_revision": REV, "train_sha256": TRAIN_SHA,
                "contract": contract, "optimizer_steps": step, "rows_seen": seen,
                "truncated_examples": dict(trunc), "files": {n: file_hash(temp / n) for n in names}}
        (temp / marker).write_text(json.dumps(meta, indent=2) + "\n")
        temp.rename(path)
        verified_marker(path, marker, contract)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def training_loop(model, tokenizer, rows, root, contract, device, dtype, time_limit_seconds,
                  checkpoint_seconds=600, checkpoint_every=200):
    """Resume exactly at an optimizer boundary. Returns 75 when the timebox expires."""
    final = root / "final_adapter"
    if final.exists():
        try:
            verified_marker(final, FINAL_MARKER, contract)
            return 0
        except (ValueError, KeyError, OSError):
            print("Final adapter did not pass readback. Recovering from a training checkpoint.", flush=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    ckpt, info = find_checkpoint(checkpoints, contract)
    effective, micro = contract["effective_batch"], contract["micro_batch"]
    total = math.ceil(len(rows) / effective)
    order = list(range(len(rows)))
    random.Random(contract["seed"]).shuffle(order)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total * 0.03), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    step, seen = 0, 0
    trunc = Counter()
    if ckpt:
        set_peft_model_state_dict(model, load_file(str(ckpt / ADAPTER_FILES[0])))
        state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        step, seen = info["optimizer_steps"], info["rows_seen"]
        trunc.update(info["truncated_examples"])
        print(f"Resumed step {step}/{total}, {seen}/{len(rows)} examples", flush=True)
        del state
    start_time = last_save = time.monotonic()
    start_seen = seen
    while seen < len(rows):
        group = [rows[i] for i in order[seen:seen + effective]]
        encoded, clipped = encode_update(tokenizer, group, contract["max_tokens"])
        loss = optimizer_update(model, encoded, tokenizer.pad_token_id, device, dtype,
                                optimizer, scheduler, scaler, micro)
        seen += len(group)
        step += 1
        trunc.update(clipped)
        now = time.monotonic()
        if step == 1 or step % 25 == 0:
            rate = (seen - start_seen) / max(now - start_time, 1e-6)
            print(f"Step {step}/{total}, rows {seen}, loss {loss:.4f}, {rate:.1f} rows/s", flush=True)
        expired = time_limit_seconds > 0 and now - start_time >= time_limit_seconds
        if step % checkpoint_every == 0 or now - last_save >= checkpoint_seconds or expired or seen == len(rows):
            state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "python_rng": random.getstate(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            persist(checkpoints / f"step-{step:06d}", model, contract, step, seen, trunc, state)
            # Only remove checkpoint directories produced by this run, after readback.
            if ckpt is not None:
                # Keep the latest two known-good snapshots; preserve unexpected directories.
                for p in checkpoints.iterdir():
                    if re.fullmatch(r"step-\d{6}", p.name) and int(p.name[5:]) < int(ckpt.name[5:]):
                        try:
                            verified_marker(p, "CHECKPOINT.json", contract)
                        except (KeyError, OSError, ValueError):
                            continue
                        shutil.rmtree(p)
            ckpt = checkpoints / f"step-{step:06d}"
            last_save = time.monotonic()
            print(f"Saved step {step}, {seen} examples", flush=True)
            if expired and seen < len(rows):
                print("Training paused at a verified checkpoint. Repeat the same command to resume.", flush=True)
                return 75
    persist(final, model, contract, step, seen, trunc)
    return 0


def run(args):
    if transformers.__version__ != "4.57.1" or peft.__version__ != "0.17.1":
        raise RuntimeError("This run requires transformers==4.57.1 and peft==0.17.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A Colab runtime with one CUDA GPU is required")
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 6):
        raise RuntimeError("The pinned .bin checkpoint requires PyTorch 2.6 or newer")
    if args.time_limit_minutes < 0:
        raise ValueError("time-limit-minutes must be >= 0")
    path = args.run_dir / "training"
    device = torch.device("cuda:0")
    # Prefer bf16 on supporting GPUs. Preserve a previous run's precision across sessions.
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    precision = "bf16" if native_bf16 else "fp16"
    old_markers = [path / "final_adapter" / FINAL_MARKER] + sorted(
        path.glob("checkpoints/step-*/CHECKPOINT.json"), reverse=True)
    for marker in old_markers:
        if marker.is_file():
            try:
                precision = json.loads(marker.read_text())["contract"]["precision"]
                break
            except (OSError, ValueError, KeyError):
                continue
    if precision not in ("fp16", "bf16") or (precision == "bf16" and not native_bf16):
        raise RuntimeError(f"Saved precision {precision} is unavailable on this GPU")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    contract = {"protocol": 1, "base": MODEL, "revision": REV, "train_sha256": TRAIN_SHA,
                "seed": SEED, "rows": ROWS, "epochs": 1, "micro_batch": MICRO, "effective_batch": EFFECTIVE,
                "max_tokens": MAX_TOKENS, "rank": RANK, "alpha": ALPHA, "lora_dropout": LORA_DROPOUT,
                "targets": ["q_proj", "v_proj"], "lr": LR, "weight_decay": WEIGHT_DECAY,
                "schedule": "linear; warm up first ceil(3% of updates)", "precision": precision,
                "torch": str(torch.__version__), "transformers": transformers.__version__, "peft": peft.__version__}
    if (path / "final_adapter").exists():
        try:
            verified_marker(path / "final_adapter", FINAL_MARKER, contract)
            print("Verified completed adapter", flush=True)
            return 0
        except (ValueError, KeyError, OSError):
            if not (path / "checkpoints").is_dir():
                raise RuntimeError("Damaged final adapter without saved checkpoints")

    verify_model(args.model_dir)
    rows = read_train(args.data_dir)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, legacy_behaviour=False,
                                        local_files_only=True, tgt_lang=TARGET)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, dtype=dtype, use_safetensors=False,
                                                 local_files_only=True, attn_implementation="sdpa")
    model = adapter_model(base).to(device)
    print(f"{torch.cuda.get_device_name(device)}, {precision}, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters", flush=True)
    return training_loop(model, tok, rows, path, contract, device, dtype, args.time_limit_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--time-limit-minutes", type=float, default=180)
    raise SystemExit(run(parser.parse_args()))
