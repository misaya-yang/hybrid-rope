#!/usr/bin/env python3
"""Round 12 task-row builders for Track A (zero-training) evaluation.

Builds, with one frozen seed, the fixed evaluation rows:
  - RULER-style synthetic retrieval: single-key (niah_single) and multi-key,
    32 instances each at 4K/8K/16K, evidence positions stratified 0.25/0.5/0.75.
  - Round-11 natural single_evidence QA rows COPIED VERBATIM (streaming) from
    the frozen olmo_tasks transport_views file, capped per length:
        2048: 32 instances x 2 worlds = 64 rows   (7B compact eligibility)
        8192: 64 instances x 2 worlds = 128 rows
       16384: 64 instances x 2 worlds = 128 rows
    Nothing is re-tokenized or edited; rows are selected by first appearance.

All rows are pre-tokenized with the deployed tokenizer and store:
  row_id, group_id, world, family, layout, length_cap, prompt_ids,
  generation_budget, accepted_full_answers, expected_ruler_answers.

Memory note: rows are written incrementally; no full-file accumulation
(server no-card mode has a 2GiB RAM cap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

RULER_KEY_HINT = (
    "Below is a list of records. Each record contains a special key and its value. "
    "Answer the question at the end using the records only.\n\n"
)
FILLER_SENTENCES = [
    "The grass was green near the old stone wall.",
    "A small boat drifted across the quiet lake at dawn.",
    "The librarian shelved the returned books without a word.",
    "Rain tapped against the window while the kettle boiled.",
    "The courier checked the address twice before knocking.",
    "Orchards lined the road all the way to the harbor.",
    "The clock in the hall struck shortly after noon.",
    "Snow gathered on the rails while the station waited.",
]
# Track A selection caps on round-11 single_evidence (instances per length).
R11_INSTANCE_CAPS = {2048: 32, 8192: 64, 16384: 64}


def _stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _filler_token_lens(tokenizer) -> list[int]:
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in FILLER_SENTENCES]


def _fill_around(tokenizer, rng: random.Random, prefix: str, needle: str,
                 question: str, length: int, needle_start: int,
                 filler_lens: list[int]) -> str:
    """Deterministically fill so the needle starts at ~needle_start tokens and
    the total prompt is ~length tokens. Token counts are estimated from cached
    filler-sentence lengths (one encode each), never by re-encoding the growing
    text; the caller's final encode truncates to exactly `length`."""
    n = len(FILLER_SENTENCES)

    def pad_to(text: str, cur_tokens: int, want: int) -> tuple[str, int]:
        i = 0
        while cur_tokens < want and i < 200000:
            j = rng.randrange(n)
            text += FILLER_SENTENCES[j] + " "
            cur_tokens += filler_lens[j] + 1
            i += 1
        return text, cur_tokens

    pre, pre_tok = pad_to(prefix,
                          len(tokenizer.encode(prefix, add_special_tokens=False)),
                          needle_start)
    needle_tok = len(tokenizer.encode(needle, add_special_tokens=False))
    q_tok = len(tokenizer.encode(question, add_special_tokens=False))
    post = question
    want_post = length - (pre_tok + needle_tok + q_tok)
    if want_post > 0:
        filler, _ = pad_to("", 0, want_post)
        post = question + " " + filler
    return pre + needle + post


def build_ruler_rows(tokenizer, length: int, n_single: int = 32, n_multi: int = 32,
                     seed: int = 12) -> list[dict]:
    """RULER-style single-key and multi-key retrieval rows at one target length.

    Keys/values are random hex strings regenerated per instance; distractor keys
    surround the evidence; needle position is stratified across the sequence.
    """
    rng = random.Random(seed * 1000 + length)
    filler_lens = _filler_token_lens(tokenizer)
    rows = []
    for family, n_keys, n_distract, budget in (
        ("ruler_single_key", 1, 7, 32),
        ("ruler_multi_key", 3, 9, 64),
    ):
        count = n_single if family == "ruler_single_key" else n_multi
        for i in range(count):
            keys = [f"KEY{rng.randint(10**5, 10**6 - 1)}" for _ in range(n_keys)]
            values = [f"{rng.getrandbits(32):08x}" for _ in range(n_keys)]
            distractors = [f"KEY{rng.randint(10**5, 10**6 - 1)}:{rng.getrandbits(32):08x}"
                           for _ in range(n_distract)]
            needle = "".join(f"The special key {k} corresponds to value {v}. "
                             for k, v in zip(keys, values))
            question = ("What is the value of the special key " +
                        " and the special key ".join(keys) + "? Answer with only the value(s).")
            frac = [0.25, 0.5, 0.75][i % 3]
            prefix = RULER_KEY_HINT + "Records: " + \
                " ".join(distractors[:len(distractors) // 2]) + " "
            suffix = " ".join(distractors[len(distractors) // 2:]) + "\n\n" + question
            prompt = _fill_around(tokenizer, rng, prefix, needle, suffix,
                                  length, int(length * frac), filler_lens)
            ids = tokenizer.encode(prompt, add_special_tokens=False)[:length]
            gid = _stable_id("round12", family, length, i)
            rows.append({
                "row_id": _stable_id(gid, "w0"), "group_id": gid, "world": "0",
                "family": family, "layout": "synthetic_ruler", "length_cap": length,
                "prompt_ids": ids, "generation_budget": budget,
                "accepted_full_answers": list(values),
                "expected_ruler_answers": list(values),
                "evidence_position_frac": frac,
            })
    return rows


def stream_round11_single_evidence(tasks_jsonl: Path):
    """Yield round-11 single_evidence rows verbatim (streaming), capped per
    length by R11_INSTANCE_CAPS, worlds 0/1 only, first appearance order."""
    chosen: dict[int, list[str]] = {lc: [] for lc in R11_INSTANCE_CAPS}
    with Path(tasks_jsonl).open() as fh:
        for line in fh:
            r = json.loads(line)
            lc = int(r["length_cap"])
            if (r.get("family") != "single_evidence" or lc not in chosen
                    or r.get("split") != "train"):
                continue
            sid = r["semantic_id"]
            if sid not in chosen[lc] and len(chosen[lc]) < R11_INSTANCE_CAPS[lc]:
                chosen[lc].append(sid)
    keep = {lc: set(ids) for lc, ids in chosen.items()}
    counts = {lc: 0 for lc in keep}
    with Path(tasks_jsonl).open() as fh:
        for line in fh:
            r = json.loads(line)
            lc = int(r["length_cap"])
            if (r.get("family") == "single_evidence" and lc in keep
                    and r.get("split") == "train"
                    and r["semantic_id"] in keep[lc] and str(r["world"]) in ("0", "1")):
                counts[lc] += 1
                yield {
                    "row_id": _stable_id(r["semantic_id"], str(r["world"])),
                    "group_id": r["semantic_id"], "world": str(r["world"]),
                    "family": "single_evidence", "layout": r["layout"],
                    "length_cap": lc,
                    "prompt_ids": r["prompt_ids"],
                    "generation_budget": int(r["generation_budget"]),
                    "accepted_full_answers": r["accepted_full_answers"],
                    "expected_ruler_answers": r["accepted_full_answers"],
                }
    assert counts == {2048: 64, 8192: 128, 16384: 128}, \
        f"round-11 single_evidence copy counts off: {counts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to OLMo tokenizer dir")
    ap.add_argument("--round11-tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists; preserved, not overwritten")

    n_rows, families = 0, set()
    with out.open("w") as fh:
        for length in args.lengths:
            for r in build_ruler_rows(tok, length, seed=args.seed):
                fh.write(json.dumps(r) + "\n")
                n_rows += 1
                families.add(r["family"])
        if args.round11_tasks and Path(args.round11_tasks).exists():
            for r in stream_round11_single_evidence(Path(args.round11_tasks)):
                fh.write(json.dumps(r) + "\n")
                n_rows += 1
                families.add(r["family"])

    manifest = {
        "status": "ROUND12_TRACKA_TASKS_FROZEN_V1",
        "n_rows": n_rows,
        "families": sorted(families),
        "lengths": args.lengths,
        "seed": args.seed,
        "r11_instance_caps": R11_INSTANCE_CAPS,
        "note": "ruler rows regenerated deterministically from seed; single_evidence "
                "rows copied verbatim (split=train, worlds 0/1, first-appearance) "
                "from round-11 frozen transport_views",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / f"{out.stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
