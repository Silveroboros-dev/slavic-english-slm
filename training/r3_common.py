"""Pure-Python R3 contracts, target sequence alignment and epoch scheduling."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib, json, math, random
from pathlib import Path

LANGS = ("pl", "hr", "ru")
SEED = 42
EPOCHS = 5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def focus_alignment(correct, wrong, special_ids=()):
    """Ordered diff-only masks and separate optional margin boundary anchors.

    SequenceMatcher preserves repeated-token positions and insertions/deletions.
    The returned CE masks NEVER mark a common boundary token as changed. A missing
    changed side can use one adjacent ordinary token only for margin scoring, with
    its position explicitly reported to the caller and in the receipt.
    """
    special = set(special_ids)
    left = [False] * len(correct)
    right = [False] * len(wrong)
    opcodes = SequenceMatcher(a=correct, b=wrong, autojunk=False).get_opcodes()
    for op, a, b, c, d in opcodes:
        if op == "equal":
            continue
        for i in range(a, b):
            left[i] = correct[i] not in special
        for i in range(c, d):
            right[i] = wrong[i] not in special
    anchors = {"correct_index": None, "wrong_index": None}
    for name, seq, mask, side in [("correct", correct, left, 0), ("wrong", wrong, right, 1)]:
        if any(mask):
            continue
        candidates = [i for i, t in enumerate(seq) if t not in special]
        changes = [(a if side == 0 else c) for op, a, b, c, d in opcodes if op != "equal"]
        if candidates and changes:
            boundary = changes[0]
            anchors[name + "_index"] = min(candidates, key=lambda i: (abs(i - boundary), i))
        else:
            raise ValueError("No lexical contrast between paired targets")
    return left, right, anchors


def paired(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get("pair_id"):
            groups[(r["lang"], r["pair_id"])].append(r)
    for key, rs in groups.items():
        if len(rs) != 2 or {r["variant"].upper() for r in rs} != {"A", "B"}:
            raise ValueError(f"Bad pair {key}")
        if rs[0]["tgt"] == rs[1]["tgt"] or rs[0]["src"] == rs[1]["src"]:
            raise ValueError(f"Degenerate pair {key}")
        for i, r in enumerate(rs):
            r["wrong_tgt"] = rs[1 - i]["tgt"]
            r["matched_id"] = rs[1 - i]["id"]
    return groups


def validate_rows(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate training IDs")
    for r in rows:
        if r["lang"] not in LANGS or r["origin"] not in ("conversation", "B01", "B02", "clock"):
            raise ValueError("Forbidden training language/origin")
        if not all(isinstance(r[k], str) and r[k].strip() for k in ("id", "src", "tgt")):
            raise ValueError("Empty training field")
        if r["origin"] != "conversation" and not r.get("pair_id"):
            raise ValueError("Contrastive row has no pair")
    if Counter(r["lang"] for r in rows if r["origin"] in ("B01", "B02")) != dict.fromkeys(
        LANGS, 64
    ):
        raise ValueError("Expected 192 B01/B02 rows")
    paired(rows)


def epoch_plan(rows, epoch):
    """Cover all conversation+clock once, scatter clock pairs, count all repeats.

    Exactly8 B01/B02 rows form4 intact reviewed pairs/update. Across3 updates each
    language has32 rows (8 reviewed+24 other). Clock pairs occupy random available
    slots throughout the epoch; shuffled conversation rows fill remaining slots.
    Existing conversation rows pad any unused slots. No true row is dropped.
    """
    validate_rows(rows)
    rng = random.Random(f"{SEED}:epoch:{epoch}")
    by_id = {r["id"]: r for r in rows}
    reviewed = defaultdict(list)
    clock_pairs = defaultdict(list)
    conversations = defaultdict(list)
    for (lang, _), rs in sorted(paired(rows).items()):
        if rs[0]["origin"] in ("B01", "B02"):
            reviewed[lang].append(sorted(rs, key=lambda r: r["variant"]))
        elif rs[0]["origin"] == "clock":
            clock_pairs[lang].append(sorted(rs, key=lambda r: r["variant"]))
    for r in rows:
        if r["origin"] == "conversation":
            conversations[r["lang"]].append(r)
    for lang in LANGS:
        if not conversations[lang]:
            raise ValueError("Each language needs conversational padding rows")
        rng.shuffle(reviewed[lang])
        rng.shuffle(clock_pairs[lang])
        rng.shuffle(conversations[lang])
    n = {l: len(conversations[l]) + 2 * len(clock_pairs[l]) for l in LANGS}
    steps = max(24, max(math.ceil(n[l] / 24) * 3 for l in LANGS))
    plan = [[] for _ in range(steps)]
    seen = Counter()
    for j, lang in enumerate(LANGS):
        capacities = [6 if j == step % 3 else 9 for step in range(steps)]
        for pair in clock_pairs[lang]:
            options = [i for i, v in enumerate(capacities) if v >= 2]
            if not options:
                raise ValueError("Insufficient clock pair slots")
            step = rng.choice(options)
            capacities[step] -= 2
            for r in pair:
                seen[r["id"]] += 1
                plan[step].append({"id": r["id"], "exposure": "once"})
        # Shuffle remaining slots, so both unique conversations and padding are spread
        # across the epoch rather than making padding a separate trailing phase.
        slots = [step for step, capacity in enumerate(capacities) for _ in range(capacity)]
        rng.shuffle(slots)
        for k, step in enumerate(slots):
            r = conversations[lang][k % len(conversations[lang])]
            seen[r["id"]] += 1
            plan[step].append(
                {
                    "id": r["id"],
                    "exposure": "once" if k < len(conversations[lang]) else "padding_repeat",
                }
            )
        cursor = 0
        for step in range(steps):
            for _ in range(2 if j == step % 3 else 1):
                pair = reviewed[lang][cursor % len(reviewed[lang])]
                cursor += 1
                for r in pair:
                    seen[r["id"]] += 1
                    plan[step].append(
                        {
                            "id": r["id"],
                            "exposure": "reviewed_first"
                            if seen[r["id"]] == 1
                            else "reviewed_repeat",
                        }
                    )
    for step, batch in enumerate(plan):
        rng.shuffle(batch)
        if len(batch) != 32 or sum(by_id[x["id"]]["origin"] in ("B01", "B02") for x in batch) != 8:
            raise ValueError("Invalid batch quota")
        for key, count in Counter(
            by_id[x["id"]]["pair_id"] for x in batch if by_id[x["id"]].get("pair_id")
        ).items():
            if count % 2:
                raise ValueError("Split matched pair")
    if set(seen) != {r["id"] for r in rows}:
        raise ValueError("Epoch missed true rows")
    summary = {
        "epoch": epoch,
        "steps": steps,
        "exposures": steps * 32,
        "unique_rows": len(rows),
        "exposure_classes": dict(Counter(x["exposure"] for b in plan for x in b)),
        "per_row_exposures": dict(sorted(seen.items())),
        "language_exposures": dict.fromkeys(LANGS, steps * 32 // 3),
        "reviewed_rows_per_update": 8,
        "reviewed_fraction": 0.25,
        "clock_unique_rows": sum(r["origin"] == "clock" for r in rows),
        "conversation_unique_rows": sum(r["origin"] == "conversation" for r in rows),
        "clock_rows_per_batch": [sum(by_id[x["id"]]["origin"] == "clock" for x in b) for b in plan],
        "definition": "Each conversation and clock row once per epoch; clock pairs scattered into deterministic random batch slots; reviewed pairs cycle to exactly8/32 slots; existing conversations pad balanced full updates. Padding/review repeats counted separately.",
    }
    return plan, summary
