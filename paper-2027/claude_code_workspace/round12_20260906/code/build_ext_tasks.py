#!/usr/bin/env python3
"""Round-12 EXT task builder: ruler_single_key rows at the external-control
models' OWN extrapolation lengths.

Qwen2.5 is native 32K; its own 2x/4x are 65536/131072 tokens (user ruling
2026-09-06: comparisons at matched extrapolation factor, not matched absolute
length). Rows are built with the Qwen tokenizer (prompt_ids are native Qwen
token counts, EXACT), reusing the frozen round-12 ruler machinery
(deterministic filler + needle, stratified positions 0.25/0.5/0.75, hex
keys/values, distractor keys). World 0 only; generation_budget 32; scorers
identical to Track A.

Exact-length note: task_rows._fill_around estimates token counts additively
(+1 per filler sentence for the separating space), which BPE tokenizers do
not actually pay; at 64K+ that under-fills by ~10%. This builder therefore
re-encodes and tops up tail filler until the row is exactly `length` tokens.
The tail is pure filler (question sits before it), so top-up/truncation never
moves the needle or the question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import task_rows


def build_single_key_rows_exact(tok, length: int, count: int, seed: int):
    """Same row design as task_rows.build_ruler_rows(family=ruler_single_key),
    but with prompt_ids guaranteed exactly `length` tokens."""
    rng = random.Random(seed * 1000 + length)
    filler_lens = task_rows._filler_token_lens(tok)
    n_sent = len(task_rows.FILLER_SENTENCES)
    rows = []
    for i in range(count):
        key = f"KEY{rng.randint(10**5, 10**6 - 1)}"
        value = f"{rng.getrandbits(32):08x}"
        distractors = [f"KEY{rng.randint(10**5, 10**6 - 1)}:{rng.getrandbits(32):08x}"
                       for _ in range(7)]
        needle = f"The special key {key} corresponds to value {value}. "
        question = (f"What is the value of the special key {key}? "
                    "Answer with only the value(s).")
        frac = [0.25, 0.5, 0.75][i % 3]
        prefix = task_rows.RULER_KEY_HINT + "Records: " + \
            " ".join(distractors[:len(distractors) // 2]) + " "
        suffix = " ".join(distractors[len(distractors) // 2:]) + "\n\n" + question
        prompt = task_rows._fill_around(tok, rng, prefix, needle, suffix,
                                        length, int(length * frac), filler_lens)
        ids = tok.encode(prompt, add_special_tokens=False)
        guard = 0
        while len(ids) != length and guard < 40:
            if len(ids) > length:
                ids = ids[:length]          # tail is pure filler: safe cut
            else:
                shortfall = length - len(ids)
                n_add = max(1, shortfall // 6)   # ~10-13 tok/sentence
                prompt += "".join(
                    task_rows.FILLER_SENTENCES[rng.randrange(n_sent)] + " "
                    for _ in range(n_add))
                ids = tok.encode(prompt, add_special_tokens=False)
            guard += 1
        assert len(ids) == length, f"could not hit {length} exactly (got {len(ids)})"
        gid = task_rows._stable_id("round12ext", "ruler_single_key", length, i)
        rows.append({
            "row_id": task_rows._stable_id(gid, "w0"), "group_id": gid, "world": "0",
            "family": "ruler_single_key", "layout": "synthetic_ruler",
            "length_cap": length,
            "prompt_ids": ids, "generation_budget": 32,
            "accepted_full_answers": [value],
            "expected_ruler_answers": [value],
            "evidence_position_frac": frac,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Qwen tokenizer dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--seed", type=int, default=120)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists; preserved, not overwritten")

    n_rows = 0
    with out.open("w") as fh:
        for length in args.lengths:
            rows = build_single_key_rows_exact(tok, length, args.groups, args.seed)
            assert len(rows) == args.groups
            for r in rows:
                assert len(r["prompt_ids"]) == length
                fh.write(json.dumps(r) + "\n")
                n_rows += 1
            print(f"EXT length {length}: {len(rows)} rows written", flush=True)

    manifest = {
        "status": "ROUND12_EXT_TASKS_FROZEN_V1",
        "n_rows": n_rows,
        "families": ["ruler_single_key"],
        "lengths": args.lengths,
        "groups_per_length": args.groups,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "note": ("prompt_ids are NATIVE target-tokenizer token counts, exact "
                 "per row (additive filler estimate corrected by measured "
                 "re-encode + tail top-up); same row design as frozen round-12 "
                 "ruler_single_key; world 0 only, budget 32, needle fracs "
                 "0.25/0.5/0.75 stratified"),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / f"{out.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
