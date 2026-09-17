"""Prepare counterfactual native-window probes using an existing local tokenizer.

No checkpoint weights or model outputs are loaded. These are synthetic mechanism
probes, not RULER or independent evidence of broad task improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import string

CONTRACT = "native-counterfactual-mechanism-v1"
SEED = 20261217
LENGTHS = (1024, 2048, 4096)
BUDGET = 32
WORLDS = 12


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def identifiers(rng: random.Random, count: int) -> list[str]:
    seen = set()
    values = []
    while len(values) < count:
        value = "".join(rng.choices(string.ascii_uppercase, k=5))
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def follow(edges: list[tuple[str, str]], start: str, hops: int) -> str:
    mapping = dict(edges)
    if len(mapping) != len(edges):
        raise ValueError("ambiguous source node")
    for _ in range(hops):
        start = mapping[start]
    return start


def make_world(task: str, seed: int) -> dict:
    rng = random.Random(seed)
    labels = identifiers(rng, 2200)
    if task == "native_binding":
        targets = [(labels[0], labels[1]), (labels[2], labels[3])]
        return {"task": task, "targets": targets, "queries": [labels[0], labels[2]],
                "answers": [labels[1], labels[3]],
                "distractors": list(zip(labels[8::2], labels[9::2]))}
    if task != "native_chain":
        raise ValueError("unknown task")
    a, b = labels[:4], labels[4:8]
    base = list(zip(a[:-1], a[1:])) + list(zip(b[:-1], b[1:]))
    rewired = list(base)
    rewired[1] = (a[1], b[2])
    rewired[4] = (b[1], a[2])
    order = list(range(6))
    rng.shuffle(order)
    return {"task": task, "base": [base[i] for i in order],
            "rewired": [rewired[i] for i in order], "queries": [a[0], b[0]],
            "distractors": list(zip(labels[8::2], labels[9::2]))}


def render(world: dict, count: int, condition: str, query: int) -> dict:
    distractors = world["distractors"][:count]
    task = world["task"]
    if task == "native_binding":
        target = world["targets"]
        records = target + distractors if condition == "far" else distractors + target
        instruction = "Each record assigns one VALUE to one KEY. Find the value of the queried key. Reply with only its five-letter value.\n"
        line = lambda edge: f"KEY {edge[0]} VALUE {edge[1]}\n"
        question = f"\nQuery KEY {world['queries'][query]}. Its VALUE is:"
        answer = world["answers"][query]
    else:
        target = world[condition]
        records = list(distractors)
        # Same line slots for both graphs; only two destinations are exchanged.
        for i, edge in enumerate(target):
            records.insert(round(i * count / 5) + i, edge)
        instruction = "Each record is a directed NEXT relation. Starting at the queried node, follow exactly three NEXT relations. Reply with only the five-letter node reached after the third step.\n"
        line = lambda edge: f"NODE {edge[0]} NEXT {edge[1]}\n"
        question = f"\nStart NODE {world['queries'][query]}. After exactly three steps:"
        answer = follow(records, world["queries"][query], 3)
    context = instruction + "".join(line(edge) for edge in records)
    spans = []
    cursor = len(instruction)
    for edge in records:
        record = line(edge)
        if edge in target:
            source_offset = record.index(edge[0])
            destination_offset = record.rindex(edge[1])
            spans.append({"source": edge[0], "destination": edge[1],
                          "char_start": cursor, "char_end": cursor + len(record),
                          "source_char_start": cursor + source_offset,
                          "source_char_end": cursor + source_offset + len(edge[0]),
                          "destination_char_start": cursor + destination_offset,
                          "destination_char_end": cursor + destination_offset + len(edge[1])})
        cursor += len(record)
    return {"text": context + question, "context": context, "answer": answer,
            "records": records, "evidence_char_spans": spans,
            "query_char_start": len(context), "query_char_end": len(context + question),
            "distractor_records": count}


def variants(world: dict, count: int) -> list[tuple[str, int, dict]]:
    conditions = ("near", "far") if world["task"] == "native_binding" else ("base", "rewired")
    return [(condition, query, render(world, count, condition, query))
            for condition in conditions for query in (0, 1)]


def prepare_group(tokenizer, task: str, length: int, world_index: int) -> list[dict]:
    seed = SEED + LENGTHS.index(length) * 10000 + (task == "native_chain") * 1000 + world_index
    world = make_world(task, seed)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    low, high = 0, len(world["distractors"])
    while low < high:
        mid = (low + high + 1) // 2
        fits = all(len(encode(v[2]["text"])) + BUDGET <= length for v in variants(world, mid))
        if fits:
            low = mid
        else:
            high = mid - 1
    generated = variants(world, low)
    group = f"{task}:{length}:{world_index:03d}"
    rows = []
    for condition, query, item in generated:
        encoded = tokenizer(item["text"], add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if not 0.90 * length <= len(ids) or len(ids) + BUDGET > length:
            raise ValueError("mechanism prompt does not fill its declared length condition")
        positions = []
        for span in item["evidence_char_spans"]:
            tokens = [i for i, (a, b) in enumerate(offsets)
                      if b > span["char_start"] and a < span["char_end"]]
            source_tokens = [i for i, (a, b) in enumerate(offsets)
                             if b > span["source_char_start"] and a < span["source_char_end"]]
            destination_tokens = [i for i, (a, b) in enumerate(offsets)
                                  if b > span["destination_char_start"] and a < span["destination_char_end"]]
            if not tokens or not source_tokens or not destination_tokens:
                raise ValueError("tokenizer offsets do not cover a declared evidence field")
            positions.append({**span, "token_start": min(tokens), "token_end": max(tokens) + 1,
                              "source_token_start": min(source_tokens),
                              "source_token_end": max(source_tokens) + 1,
                              "destination_token_start": min(destination_tokens),
                              "destination_token_end": max(destination_tokens) + 1,
                              "distance_to_last_prompt_token": len(ids) - 1 - max(tokens)})
        question_tokens = [i for i, (a, b) in enumerate(offsets)
                           if b > item["query_char_start"] and a < item["query_char_end"]]
        if not question_tokens or max(question_tokens) != len(ids) - 1:
            raise ValueError("final question must include the last prompt token")
        intervention = {"layout" if task == "native_binding" else "graph": condition,
                        "query": "ab"[query]}
        rows.append({"row_id": f"{group}:{condition}:{query}", "group_id": group,
                     "document_cluster_id": group, "task": task,
                     "family": "native_counterfactual_mechanism", "length_cap": length,
                     "prompt_text": item["text"], "prompt_ids": ids,
                     "prompt_sha256": digest(ids), "input_tokens": len(ids),
                     "references": [item["answer"]], "max_new_tokens": BUDGET,
                     "intervention": intervention, "context_id": digest(item["context"]),
                     "evidence_positions": positions, "source_seed": seed,
                     "final_query_token_start": min(question_tokens),
                     "final_query_token_end": max(question_tokens) + 1,
                     "source_records": item["records"], "query_node": world["queries"][query],
                     "distractor_records": low, "selection_uses_model_outputs": False})
    audit_group(rows)
    return rows


def audit_group(rows: list[dict]) -> None:
    if len(rows) != 4 or len({r["group_id"] for r in rows}) != 1:
        raise ValueError("incomplete counterfactual group")
    task = rows[0]["task"]
    dimension = "layout" if task == "native_binding" else "graph"
    conditions = ("near", "far") if task == "native_binding" else ("base", "rewired")
    mapping = {(r["intervention"][dimension], r["intervention"]["query"]): r for r in rows}
    if set(mapping) != {(c, q) for c in conditions for q in "ab"}:
        raise ValueError("counterfactual cells differ")
    for condition in conditions:
        a, b = mapping[(condition, "a")], mapping[(condition, "b")]
        if a["context_id"] != b["context_id"] or a["references"] == b["references"]:
            raise ValueError("query swap must preserve context and change answer")
    first, second = (mapping[(c, "a")] for c in conditions)
    if task == "native_binding":
        if sorted(first["source_records"]) != sorted(second["source_records"]):
            raise ValueError("layout intervention changed content")
        if first["references"] != second["references"]:
            raise ValueError("layout intervention changed the answer")
    else:
        if Counter(v for edge in first["source_records"] for v in edge) != Counter(v for edge in second["source_records"] for v in edge):
            raise ValueError("rewiring changed the node multiset")
        if sum(a != b for a, b in zip(first["source_records"], second["source_records"])) != 2:
            raise ValueError("rewiring must change exactly two edge destinations")
        for row in rows:
            if follow(row["source_records"], row["query_node"], 3) != row["references"][0]:
                raise ValueError("reference disagrees with graph traversal")
        if first["references"] == second["references"]:
            raise ValueError("rewiring must change the answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a new output directory; prepared panels are frozen")
    # Transformers is imported only here, after CPU-only environment selection.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    rows = []
    for task in ("native_binding", "native_chain"):
        for length in LENGTHS:
            for index in range(WORLDS):
                rows.extend(prepare_group(tokenizer, task, length, index))
    args.out.mkdir(parents=True)
    inputs_path = args.out / "inputs.jsonl"
    with inputs_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = {"status": "CPU_PREPARED", "contract": CONTRACT, "rows": len(rows),
                "groups": len(rows) // 4, "worlds_per_task_length": WORLDS,
                "lengths": list(LENGTHS), "seed": SEED, "model": str(args.model),
                "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
                "tokenizer_class": type(tokenizer).__name__, "model_weights_loaded": False,
                "gpu_execution": False, "model_output_selection": False,
                "synthetic_distractor_load": True, "content_truncation": False,
                "minimum_input_tokens_by_length": {str(n): min(r["input_tokens"] for r in rows if r["length_cap"] == n) for n in LENGTHS},
                "maximum_input_tokens_by_length": {str(n): max(r["input_tokens"] for r in rows if r["length_cap"] == n) for n in LENGTHS},
                "interpretation": "New synthetic mechanism panel; output-blind construction, not broad benchmark confirmation.",
                "primary_endpoints": ["per-length task-equal complete-answer exact accuracy", "all-four-correct world rate"],
                "secondary_endpoints": ["query-pair both-correct", "binding far-minus-near and its between-arm interaction", "rewired-minus-base"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
