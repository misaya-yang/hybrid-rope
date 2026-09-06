#!/usr/bin/env python3
"""Track B Phase-B SFT view assembly: exactly 512 views.

Composition (plan Section 7.4, adjusted to the frozen round-11 inventory):
  - Natural single-evidence QA: 64 instances x 2 content worlds at EACH of
    {8K, 16K} = 128 views per length = 256 views. Rows are COPIED VERBATIM
    (streaming) from the frozen round-11 transport_views file (family
    single_evidence, train cells near=8192 / far=16384). First-appearance
    instance order; worlds 0/1. Nothing is re-tokenized or edited.
    (Inventory 2026-09-06: 8K has exactly 64 unique instances; 16K has 160,
    of which the first 64 are used.)
  - Synthetic binding: 128 instances x 2 content worlds = 256 views, built here
    deterministically (frozen seed). Each instance defines one query key whose
    bound value DIFFERS across the two worlds; every world is internally legal
    text and the gold answer is synced to its world.

View row schema (matches round-11 transport views):
  semantic_id, world, family, layout, length_cap, prompt_ids, target_ids,
  generation_budget, accepted_full_answers, stratum
Loss mask for training: prompt positions -> -100; target_ids trained in full
(they end with EOS).

Streaming I/O throughout (server no-card RAM cap = 2GiB).

Usage:
  python data_prep_sft.py \
      --tokenizer .../OLMo-2-0425-1B-Instruct \
      --round11-views /root/autodl-tmp/claude_round11_olmo_20260905/olmo_tasks/transport_views.jsonl \
      --out .../data/sft/views.jsonl --n-binding 128 --seed 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

BINDING_LEN = 2048
QA_LENGTHS = (8192, 16384)
N_QA_INSTANCES_PER_LEN = 64      # x 2 worlds = 128 views per length
N_QA_VIEWS_PER_LEN = 128

SYLLABLES = ["ba", "ren", "kol", "ith", "mar", "dun", "ves", "ora", "tik",
             "sula", "fen", "gor", "lya", "peth", "nave", "quo", "rish",
             "tarn", "ul", "vex", "wren", "ytha", "zor", "amble"]
LEDGERS = ["the harbor ledger", "the northern registry", "the guild archive",
           "the mountain census"]

FILLER = [
    "Ferries crossed the strait twice each morning regardless of weather.",
    "The mapmaker recorded new roads only after two merchants confirmed them.",
    "Lanterns along the causeway were trimmed at midday to save oil.",
    "Grain shipments were logged in triplicate before the gates opened.",
    "The bell tower kept a separate schedule during the fishing season.",
    "Couriers swapped pouches at the halfway post without stopping.",
    "The survey office repainted its boundary stones every fourth spring.",
    "Barges waited their turn beneath the low bridge in strict order.",
]


def stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def stream_qa_views(out_fh, round11_path: Path) -> int:
    """Copy round-11 single_evidence QA views at 8K/16K verbatim, streaming.
    Returns number of views written."""
    order: dict[int, list[str]] = {lc: [] for lc in QA_LENGTHS}
    with Path(round11_path).open() as fh:
        for line in fh:
            r = json.loads(line)
            lc = int(r["length_cap"])
            if (r.get("family") != "single_evidence" or lc not in order
                    or r.get("split") != "train"):
                continue
            sid = r["semantic_id"]
            if sid not in order[lc] and len(order[lc]) < N_QA_INSTANCES_PER_LEN:
                order[lc].append(sid)
    keep = {lc: set(ids) for lc, ids in order.items()}
    counts = {lc: 0 for lc in QA_LENGTHS}
    with Path(round11_path).open() as fh:
        for line in fh:
            r = json.loads(line)
            lc = int(r["length_cap"])
            if not (r.get("family") == "single_evidence" and lc in keep
                    and r.get("split") == "train"
                    and r["semantic_id"] in keep[lc] and str(r["world"]) in ("0", "1")):
                continue
            v = {k: r[k] for k in ("semantic_id", "world", "family", "layout",
                                   "length_cap", "prompt_ids", "target_ids",
                                   "generation_budget", "accepted_full_answers")}
            v["stratum"] = f"qa_{lc}"
            out_fh.write(json.dumps(v) + "\n")
            counts[lc] += 1
    assert counts == {lc: N_QA_VIEWS_PER_LEN for lc in QA_LENGTHS}, \
        f"QA view counts off: {counts}"
    return sum(counts.values())


def _name(rng: random.Random) -> str:
    return "".join(rng.sample(SYLLABLES, rng.randint(2, 3))).capitalize()


def _key(rng: random.Random) -> str:
    return f"RC-{rng.randint(1000, 9999)}"


def _value(rng: random.Random) -> str:
    return f"{rng.randint(10, 99)}-{rng.choice(['amber', 'slate', 'verdant', 'coral', 'umber'])}-{rng.randint(1, 9)}"


def build_binding_views(out_fh, tokenizer, n_instances: int, seed: int) -> int:
    """Two-world binding views at BINDING_LEN tokens, streamed to out_fh.

    Both worlds share the distractor bindings and filler; only the queried
    binding sentence (and hence the gold value) differs per world, keeping each
    world internally coherent. Golds are synced to the world by construction.
    """
    rng = random.Random(seed)
    eos = tokenizer.eos_token_id
    n = 0
    for i in range(n_instances):
        qkey = _key(rng)
        values = [_value(rng), _value(rng)]
        while values[1] == values[0]:
            values[1] = _value(rng)
        ledgers = rng.sample(LEDGERS, 2)
        owner = _name(rng)
        distractors = [f"The registry code {_key(rng)} is assigned to {_value(rng)} "
                       f"under {owner}'s account. " for _ in range(4)]
        filler = ""
        while len(filler) < BINDING_LEN * 6:  # chars budget; trimmed by tokens
            filler += rng.choice(FILLER) + " "
        for w, (ledger, val) in enumerate(zip(ledgers, values)):
            sid = stable_id("round12_binding", seed, i)
            evidence = (f"According to {ledger}, the registry code {qkey} is assigned "
                        f"to {val}. ")
            question = (f"Question: According to {ledger}, what is the registry code "
                        f"{qkey} assigned to? Answer:")
            prompt_text = "".join(distractors) + evidence + filler + question
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)[:BINDING_LEN]
            target_ids = tokenizer.encode(" " + val, add_special_tokens=False) + [eos]
            out_fh.write(json.dumps({
                "semantic_id": sid, "world": str(w), "family": "binding",
                "layout": "binding_two_world", "length_cap": BINDING_LEN,
                "prompt_ids": ids, "target_ids": target_ids,
                "generation_budget": 24,
                "accepted_full_answers": [val],
                "stratum": "binding",
            }) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--round11-views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-binding", type=int, default=128)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists; preserved, not overwritten")

    with out.open("w") as fh:
        n_qa = stream_qa_views(fh, Path(args.round11_views))
        n_bind = build_binding_views(fh, tok, args.n_binding, args.seed)
    assert n_qa + n_bind == 512, f"expected 512 views, got {n_qa + n_bind}"

    manifest = {
        "status": "SFT_VIEWS_FROZEN_V1",
        "n_views": n_qa + n_bind,
        "strata": {"qa_8192": N_QA_VIEWS_PER_LEN, "qa_16384": N_QA_VIEWS_PER_LEN,
                   "binding": n_bind},
        "qa_instances_per_len": N_QA_INSTANCES_PER_LEN,
        "batch_recipe": "64 updates x 8 views = 2 qa_8192 + 2 qa_16384 + 4 binding per batch",
        "loss_mask": "prompt -> -100; target_ids trained incl. trailing EOS",
        "qa_source": "round-11 transport_views single_evidence, split=train only "
                     "(verbatim streaming copy, worlds 0/1, first-appearance order)",
        "binding_seed": args.seed,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / f"{out.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
