"""R4 training core retained from R3; unrelated R3 launch/evaluation removed.

Tokenization, target shift, loss, optimizer and adapter checks retain the actual
recipe. Historical input hashes bind original payload code, while new output
receipts identify this public core. No private examples or weights are included.
"""

import gc, json, math, random, time
from pathlib import Path
from collections import Counter
from model_snapshot import MODEL_ID, REVISION
from r3_common import SEED, sha, put, focus_alignment

LANG = {"pl": "pol_Latn", "hr": "hrv_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl"}
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
TRAIN = {
    "epochs": 5,
    "effective_batch": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "clip_grad_norm": 1.0,
    "seed": 42,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": TARGETS,
    "focus_weight": 5.0,
    "margin_nats": 0.5,
    "margin_weight": 0.1,
    "max_train_tokens": 256,
    "checkpoint_every_updates": 16,
    "epoch_policy": "Every conversation+clock row once, reviewed pair cycling at exactly 8/32 (25%), explicit conversation padding to equal language exposure. See epoch_plan.json.",
}


def runtime_seed():
    import torch

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def fingerprint(state):
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode(tok, rows):
    """Build labels AND wrong-label branches in target-tokenizer mode only."""
    result = []
    special = set(tok.all_special_ids)
    anchors = Counter()
    stats = Counter()
    for r in rows:
        tok.src_lang = LANG[r["lang"]]
        tok.tgt_lang = "eng_Latn"
        source = tok(r["src"], padding=False, truncation=False)["input_ids"]
        target = tok(text_target=r["tgt"], padding=False, truncation=False)["input_ids"]
        if (
            source[0] != tok.convert_tokens_to_ids(LANG[r["lang"]])
            or target[0] != tok.convert_tokens_to_ids("eng_Latn")
            or source[-1] != tok.eos_token_id
            or target[-1] != tok.eos_token_id
        ):
            raise ValueError("Unexpected NLLB token layout")
        if max(len(source), len(target)) > TRAIN["max_train_tokens"]:
            raise ValueError(f"Training truncation forbidden: {r['id']}")
        wrong = None
        focus = [False] * len(target)
        wrong_focus = None
        score_focus = list(focus)
        wrong_score_focus = None
        if r.get("wrong_tgt"):
            wrong = tok(text_target=r["wrong_tgt"], padding=False, truncation=False)["input_ids"]
            if len(wrong) > TRAIN["max_train_tokens"]:
                raise ValueError("Wrong target too long")
            focus, wrong_focus, anchor = focus_alignment(target, wrong, special)
            anchors.update(k for k, v in anchor.items() if v is not None)
            score_focus = list(focus)
            wrong_score_focus = list(wrong_focus)
            if anchor["correct_index"] is not None:
                score_focus[anchor["correct_index"]] = True
            if anchor["wrong_index"] is not None:
                wrong_score_focus[anchor["wrong_index"]] = True
            stats.update(
                {
                    "paired_rows": 1,
                    "correct_focus_tokens": sum(focus),
                    "wrong_focus_tokens": sum(wrong_focus),
                }
            )
        result.append(
            {
                "id": r["id"],
                "src": source,
                "target": target,
                "focus": focus,
                "wrong": wrong,
                "wrong_focus": wrong_focus,
                "score_focus": score_focus,
                "wrong_score_focus": wrong_score_focus,
                "origin": r["origin"],
            }
        )
    return result, dict(stats), dict(anchors)


def batch_tensors(records, tok, wrong=False, score_mask=False):
    import torch

    source = [r["src"] for r in records]
    target = [r["wrong"] if wrong else r["target"] for r in records]
    ids = torch.full((len(records), max(map(len, source))), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(records), max(map(len, target))), -100, dtype=torch.long)
    focus = torch.zeros(labels.shape, dtype=torch.bool)
    ordinary = torch.zeros(labels.shape, dtype=torch.bool)
    special = set(tok.all_special_ids)
    for i, (r, x, y) in enumerate(zip(records, source, target, strict=True)):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask_key = (
            ("wrong_score_focus" if wrong else "score_focus")
            if score_mask
            else ("wrong_focus" if wrong else "focus")
        )
        focus[i, : len(y)] = torch.tensor(r[mask_key])
        ordinary[i, : len(y)] = torch.tensor([v not in special for v in y])
    return (
        {
            "input_ids": ids.cuda(),
            "attention_mask": ids.ne(tok.pad_token_id).long().cuda(),
            "labels": labels.cuda(),
        },
        focus.cuda(),
        ordinary.cuda(),
    )


def forward_targets(model, tensors):
    # Shift only decoder targets, then request logits without the model's redundant
    # built-in loss. Custom CE below is the sole objective and ignores padding.
    # M2M100 v4.57.1 has no prepare_decoder_input_ids_from_labels method.
    # Match its own forward(labels=...) shift: EOS start, target prefix, PAD for -100.
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

    decoder = shift_tokens_right(
        tensors["labels"], model.config.pad_token_id, model.config.decoder_start_token_id
    )
    return model(
        input_ids=tensors["input_ids"],
        attention_mask=tensors["attention_mask"],
        decoder_input_ids=decoder,
        use_cache=False,
    )


def objective(model, records, tok, amp_dtype, denominator, pair_denominator):
    """Teacher-forced target-only CE + source-conditioned focal sequence hinge.

    Each pair direction is a distinct source. Wrong labels are scored under THAT
    source and their own complete decoder prefix. No source tokens enter CE, and
    there is no global penalty for words that are valid in another source context.
    """
    import torch
    import torch.nn.functional as F

    tensors, focus, ordinary = batch_tensors(records, tok)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = forward_targets(model, tensors)
        # float32 cross_entropy is numerically stable under autocast. Ignore padding;
        # language/EOS tokens retain ordinary CE but never receive focal upweighting.
        logits = output.logits
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tensors["labels"].reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(tensors["labels"].shape)
        weights = 1 + focus.to(ce.dtype) * (TRAIN["focus_weight"] - 1)
        weighted_sum = (ce * weights).sum()
        plain_sum = ce.sum()
        loss = weighted_sum / denominator
        pair_positions = [i for i, r in enumerate(records) if r["wrong"] is not None]
        metric = {
            "weighted_ce_sum": float(weighted_sum.detach()),
            "plain_ce_sum": float(plain_sum.detach()),
            "valid_tokens": int(tensors["labels"].ne(-100).sum()),
            "focal_tokens": int(focus.sum()),
            "paired_rows": len(pair_positions),
            "hinge_sum": 0.0,
            "violations": 0,
            "correct_focal_logp_sum": 0.0,
            "wrong_focal_logp_sum": 0.0,
            "gap_sum": 0.0,
        }
        del logits, output
        if pair_positions:
            pairs = [records[i] for i in pair_positions]
            score_focus = torch.zeros_like(focus[pair_positions])
            for j, r in enumerate(pairs):
                score_focus[j, : len(r["score_focus"])] = torch.tensor(
                    r["score_focus"], device="cuda"
                )
            correct_score = -(ce[pair_positions] * score_focus).sum(1) / score_focus.sum(1)
            wrong_tensors, wrong_focus, _ = batch_tensors(pairs, tok, wrong=True, score_mask=True)
            negative = forward_targets(model, wrong_tensors)
            # Negative branch needs only its focal conditional scores. Select positions
            # before fp32 CE to avoid a second full target×vocabulary fp32 allocation.
            focal_ce = F.cross_entropy(
                negative.logits[wrong_focus].float(),
                wrong_tensors["labels"][wrong_focus],
                reduction="none",
            )
            row_indices = wrong_focus.nonzero(as_tuple=False)[:, 0]
            wrong_sums = torch.zeros(len(pairs), dtype=focal_ce.dtype, device="cuda").scatter_add(
                0, row_indices, focal_ce
            )
            wrong_score = -wrong_sums / wrong_focus.sum(1)
            gap = correct_score - wrong_score
            hinges = torch.relu(TRAIN["margin_nats"] - gap)
            loss = loss + TRAIN["margin_weight"] * hinges.sum() / pair_denominator
            metric.update(
                {
                    "hinge_sum": float(hinges.detach().sum()),
                    "violations": int((gap.detach() < TRAIN["margin_nats"]).sum()),
                    "correct_focal_logp_sum": float(correct_score.detach().sum()),
                    "wrong_focal_logp_sum": float(wrong_score.detach().sum()),
                    "gap_sum": float(gap.detach().sum()),
                }
            )
    return loss, metric


def startup_smoke(model, records, tok, amp_dtype, out):
    """Real-runtime shift + paired objective forward/backward; never optimizer.step."""
    import torch
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

    if (
        model.config.pad_token_id,
        model.config.decoder_start_token_id,
        model.config.eos_token_id,
    ) != (1, 2, 2):
        raise ValueError("Pinned NLLB special tokens differ")
    english = tok.convert_tokens_to_ids("eng_Latn")
    labels = torch.tensor(
        [[english, 17, 2, -100, -100], [english, 19, 23, 2, -100]], dtype=torch.long, device="cuda"
    )
    expected = torch.tensor(
        [[2, english, 17, 2, 1], [2, english, 19, 23, 2]], dtype=torch.long, device="cuda"
    )
    untouched = labels.clone()
    shifted = shift_tokens_right(
        labels, model.config.pad_token_id, model.config.decoder_start_token_id
    )
    if not torch.equal(shifted, expected) or not torch.equal(labels, untouched):
        raise ValueError("M2M100 decoder shift smoke failed")
    paired = next(r for r in records if r["wrong"] is not None)
    plain = next(r for r in records if r["wrong"] is None)
    probe = [paired, plain]
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    model.zero_grad(set_to_none=True)
    try:
        loss, metrics = objective(
            model, probe, tok, amp_dtype, sum(len(r["target"]) for r in probe), 1
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Startup objective is nonfinite")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Startup backward gradient is absent/nonfinite")
        torch.cuda.synchronize()
        result = {
            "status": "passed",
            "scope": "Real torch/CUDA shift + correct/wrong paired objective and ordinary CE forward/backward; no optimizer/scheduler update. RNG restored and gradients cleared.",
            "pad_token_id": 1,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "english_token_id": english,
            "shift_labels": labels.tolist(),
            "shift_expected": expected.tolist(),
            "paired_rows": metrics["paired_rows"],
            "ordinary_rows": 1,
            "finite_objective": float(loss.detach()),
            "gradient_tensors_checked": len(grads),
            "optimizer_updates": 0,
        }
    finally:
        model.zero_grad(set_to_none=True)
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    put(out / "startup_smoke.json", result)
    print(
        "R3 startup smoke PASSED: exact M2M100 target shift; paired+ordinary forward/backward finite; zero optimizer updates.",
        flush=True,
    )


def inspect(model, initial=None):
    import torch
    from peft import get_peft_model_state_dict

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any(".lora_A." not in n and ".lora_B." not in n for n, p in named):
        raise ValueError("Base parameters trainable or LoRA absent")
    if any(p.dtype != torch.float32 for n, p in named):
        raise ValueError("All trainable LoRA must be fp32")
    families = Counter()
    for name, param in named:
        family = next((target for target in TARGETS if f".{target}." in name), None)
        if family is None:
            raise ValueError("Unexpected LoRA target module")
        families[family] += 1
        if (".lora_A." in name and param.shape[0] != 16) or (
            ".lora_B." in name and param.shape[1] != 16
        ):
            raise ValueError("Unexpected LoRA rank")
    if set(families) != set(TARGETS):
        raise ValueError("Missing requested LoRA target family")
    state = get_peft_model_state_dict(model)
    checks = {
        "trainable_tensors_by_target_family": dict(families),
        "trainable_parameters": sum(p.numel() for n, p in named),
        "trainable_tensors": len(named),
        "base_trainable_parameters": 0,
        "trainable_dtype": "float32",
        "fingerprint": fingerprint(state),
        "all_lora_B_tensors_zero": all(
            torch.count_nonzero(v) == 0 for k, v in state.items() if ".lora_B." in k
        ),
    }
    if initial is not None:
        deltas = {}
        for k, v in state.items():
            new = v.detach().cpu().float()
            old = initial[k].float()
            delta = new - old
            if not torch.isfinite(new).all() or not torch.count_nonzero(delta):
                raise ValueError(f"Nonfinite/unchanged LoRA tensor {k}")
            deltas[k] = {
                "delta_l2": float(torch.linalg.vector_norm(delta)),
                "changed_elements": int(torch.count_nonzero(delta)),
            }
        checks["tensor_deltas"] = deltas
        checks["all_lora_A_and_B_changed"] = True
    return checks


def save_adapter(model, path):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    state = get_peft_model_state_dict(model)
    saved = load_file(str(path / "adapter_model.safetensors"))
    if set(state) != set(saved) or any(
        not torch.equal(state[k].detach().cpu(), saved[k]) for k in state
    ):
        raise ValueError("Adapter readback mismatch")
    return {name: sha(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")}


def train(rows, plans, epochs, root, out, model_dir, tok, runtime):
    import torch
    from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    total = len(plans)
    warmup = max(1, round(total * 0.05))
    micro = runtime["micro_batch"]
    amp_dtype = torch.bfloat16 if runtime["amp_dtype"] == "bfloat16" else torch.float16
    contract = {
        "training": TRAIN,
        "total_optimizer_updates": total,
        "warmup_updates": warmup,
        "train_rows_sha256": sha(root / "train_rows.json"),
        "inputs_manifest_sha256": sha(root / "INPUTS.json"),
        "trainer_sha256": sha(Path(__file__)),
        "common_sha256": sha(root / "r3_common.py"),
        "base_model": MODEL_ID,
        "base_revision": REVISION,
        "runtime": runtime,
    }
    final = out / "final_adapter"
    receipt_path = out / "training_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["contract"] != contract
            or receipt["optimizer_updates"] != total
            or any(sha(final / n) != h for n, h in receipt["adapter_files"].items())
        ):
            raise ValueError("Invalid completed training cache")
        print("R3: verified completed training; continuing evaluation.", flush=True)
        return receipt
    runtime_seed()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=False,
        dtype=amp_dtype,
        attn_implementation="sdpa",
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGETS,
        bias="none",
        init_lora_weights=True,
        base_model_name_or_path=MODEL_ID,
        revision=REVISION,
    )
    model = get_peft_model(base, cfg).cuda()
    model.peft_config["default"].base_model_name_or_path = MODEL_ID
    model.peft_config["default"].revision = REVISION
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    initial_checks = inspect(model)
    initial_checks["lora_B_zero_at_initialization"] = initial_checks["all_lora_B_tensors_zero"]
    if not initial_checks["lora_B_zero_at_initialization"]:
        raise ValueError("Fresh LoRA B must start zero")
    put(out / "initial_parameter_checks.json", initial_checks)
    print(
        "Fresh LoRA verified:",
        initial_checks["trainable_parameters"],
        "trainable fp32 parameters; base trainable=0; all initial B tensors zero.",
        flush=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    by_id = {r["id"]: r for r in rows}
    encoded, token_stats, anchors = encode(tok, rows)
    trace = [
        {
            "id": r["id"],
            "origin": r["origin"],
            "correct_target_ids": r["target"],
            "wrong_target_ids": r["wrong"],
            "correct_diff_positions": [i for i, v in enumerate(r["focus"]) if v],
            "wrong_diff_positions": [i for i, v in enumerate(r["wrong_focus"]) if v],
            "correct_margin_only_boundary_positions": [
                i for i, v in enumerate(r["score_focus"]) if v and not r["focus"][i]
            ],
            "wrong_margin_only_boundary_positions": [
                i for i, v in enumerate(r["wrong_score_focus"]) if v and not r["wrong_focus"][i]
            ],
        }
        for r in encoded
        if r["wrong"] is not None
    ]
    put(out / "target_alignment_trace.json", trace)
    encoded = {r["id"]: r for r in encoded}
    put(
        out / "tokenization_receipt.json",
        {
            "rows": len(rows),
            "target_alignment_trace_sha256": sha(out / "target_alignment_trace.json"),
            "contrastive_stats": token_stats,
            "insertion_deletion_anchor_counts": anchors,
            "truncated_rows": 0,
            "alignment": "SequenceMatcher ordered target-token opcodes, autojunk=False; specials never focused; actual diff-only CE mask and separate margin-only boundary anchors for pure insertion/deletion",
            "target_only": True,
        },
    )
    startup_smoke(model, list(encoded.values()), tok, amp_dtype, out)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    done = 0
    history = []
    skips = 0
    elapsed_prior = 0
    saved = sorted(checkpoints.glob("step-*/CHECKPOINT.json"))
    if saved:
        marker = saved[-1]
        meta = json.loads(marker.read_text())
        cp = marker.parent
        if meta["contract"] != contract or any(sha(cp / n) != h for n, h in meta["files"].items()):
            raise ValueError("Invalid resume checkpoint")
        set_peft_model_state_dict(model, load_file(str(cp / "adapter_model.safetensors")))
        state = torch.load(cp / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        done = meta["optimizer_updates"]
        history = meta["history"]
        skips = meta["amp_skipped_attempts"]
        elapsed_prior = meta["elapsed_training_seconds"]
        print(f"R3: resumed verified update {done}/{total}", flush=True)
    start = time.monotonic()
    epoch_steps = epochs[0]["steps"]
    for step in range(done, total):
        records = [encoded[x["id"]] for x in plans[step]]
        denominator = sum(len(r["target"]) for r in records)
        pair_count = sum(r["wrong"] is not None for r in records)
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            metrics = Counter()
            loss_total = 0.0
            for offset in range(0, 32, micro):
                loss, m = objective(
                    model,
                    records[offset : offset + micro],
                    tok,
                    amp_dtype,
                    denominator,
                    max(pair_count, 1),
                )
                loss_total += float(loss.detach())
                metrics.update(m)
                scaler.scale(loss).backward()
                del loss
            if not math.isfinite(loss_total):
                raise FloatingPointError("Nonfinite objective")
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, TRAIN["clip_grad_norm"])
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= before:
                if not torch.isfinite(norm):
                    raise FloatingPointError("Accepted nonfinite gradient")
                scheduler.step()
                break
            skips += 1
            print(
                f"R3: FP16 overflow at update {step + 1}; retry {attempt + 1} scale={scaler.get_scale()}",
                flush=True,
            )
        else:
            raise FloatingPointError("12 FP16 retries failed")
        n = max(metrics["paired_rows"], 1)
        entry = {
            "step": step + 1,
            "epoch": step // epoch_steps + 1,
            "objective": loss_total,
            "plain_ce": metrics["plain_ce_sum"] / denominator,
            "weighted_ce": metrics["weighted_ce_sum"] / denominator,
            "margin_hinge_mean": metrics["hinge_sum"] / n,
            "margin_violation_fraction": metrics["violations"] / n,
            "mean_correct_focal_logp": metrics["correct_focal_logp_sum"] / n,
            "mean_wrong_focal_logp": metrics["wrong_focal_logp_sum"] / n,
            "mean_focal_logp_gap": metrics["gap_sum"] / n,
            "paired_rows": metrics["paired_rows"],
            "focal_tokens": metrics["focal_tokens"],
            "valid_tokens": denominator,
            "lr_after_update": scheduler.get_last_lr()[0],
            "grad_norm": float(norm),
            "amp_scale": scaler.get_scale(),
        }
        history.append(entry)
        elapsed = elapsed_prior + time.monotonic() - start
        if step == done or (step + 1) % 4 == 0:
            print(
                f"Epoch {entry['epoch']}/{len(epochs)} update {step + 1}/{total} | loss {loss_total:.4f} CE {entry['weighted_ce']:.4f} margin {entry['margin_hinge_mean']:.3f} violations {entry['margin_violation_fraction']:.1%} | {elapsed / 60:.1f}min ETA {elapsed / (step + 1) * (total - step - 1) / 60:.1f}min",
                flush=True,
            )
        if (
            (step + 1) % TRAIN["checkpoint_every_updates"] == 0
            or (step + 1) % epoch_steps == 0
            or step + 1 == total
        ):
            cp = checkpoints / f"step-{step + 1:06d}"
            files = save_adapter(model, cp)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                cp / "state.pt",
            )
            files["state.pt"] = sha(cp / "state.pt")
            meta = {
                "contract": contract,
                "optimizer_updates": step + 1,
                "files": files,
                "history": history,
                "amp_skipped_attempts": skips,
                "elapsed_training_seconds": elapsed,
            }
            put(cp / "CHECKPOINT.json", meta)
            if any(sha(cp / n) != h for n, h in files.items()):
                raise ValueError("Checkpoint hash readback failed")
            put(out / "loss_history.json", history)
            print(f"Saved checkpoint {step + 1}/{total}", flush=True)
    final_checks = inspect(model, initial)
    files = save_adapter(model, final)
    receipt = {
        "contract": contract,
        "optimizer_updates": total,
        "examples_seen": total * 32,
        "epoch_exposure_summaries": epochs,
        "initial_parameter_checks": initial_checks,
        "final_parameter_checks": final_checks,
        "adapter_files": files,
        "loss_history": history,
        "amp_skipped_attempts": skips,
        "elapsed_training_seconds": time.monotonic() - start + elapsed_prior,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "base_policy": "Fresh pinned frozen NLLB; new random LoRA A and zero B; no previous adapter loaded for training",
        "loss_definition": "Sum target CE weights (5 on actual changed target spans, 1 elsewhere) / valid target tokens, plus 0.1 × mean relu(0.5 − (correct−wrong) focal normalized conditional logP). Each wrong branch uses its own decoder prefix under the same source. Both pair directions trained. Conversations use plain CE. Pure insertion/deletion uses an adjacent common token only for the missing margin-score side, never for CE5x; anchor counts are recorded separately.",
        "base_trainable_parameters": 0,
        "ukrainian_training_rows": 0,
        "official_training_rows": 0,
        "data_provenance_files": {
            n: sha(root / n)
            for n in ("conversation_provenance.json", "data_receipt.json", "clock.receipt.json")
        },
    }
    put(receipt_path, receipt)
    del model, base, optimizer, scheduler, scaler, trainable, initial
    gc.collect()
    torch.cuda.empty_cache()
    return receipt
