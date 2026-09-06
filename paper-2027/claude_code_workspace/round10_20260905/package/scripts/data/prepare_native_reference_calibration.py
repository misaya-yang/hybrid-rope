#!/usr/bin/env python3
"""Build fresh Native-only reference calibration data; never load/evaluate a model.

Requires local checkpoint tokenizer files and a parquet text column. Natural
documents are selected in stream order, without scores. Capability fillers are
fixed neutral procedural sentences, never source documents or extra records.
Only manifest.json with READY status certifies both completed split row files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import string
import struct
from typing import Any, Iterable

GRID = (1024, 2048, 4096, 8192)
SEEDS = {"calibration": 202609011, "confirmation": 202609012}
NATURAL_COUNTS = {"calibration": 32, "confirmation": 64}
CAPABILITY_COUNTS = {"calibration": 64, "confirmation": 128}
TARGET_PAIRS = ((0, 2), (1, 3), (2, 5), (4, 6))
TARGET_TOKENS = 256
GENERATION_BUDGET = 48
MAX_PROMPT_UNDERAGE = 16
ASSISTANT_PREFIX = "The two codes are: "
FILLER = (
    "This passage describes a routine review of an ordinary process. "
    "The procedure can be read carefully before the next step begins. "
    "A clear workspace makes the general process easier to follow. "
    "The surrounding discussion provides no additional record information. "
    "Each paragraph continues the same neutral procedural description."
).split()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict]]:
    """Read only source identity fields; never inspect or select using scores."""
    excluded: set[str] = set()
    receipts = []
    for path in paths:
        hashes: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line).get("source_text_sha256")
                if value is None:  # Mixed files can also contain synthetic rows.
                    continue
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("invalid source_text_sha256 in exclusion rows")
                hashes.add(value)
        excluded.update(hashes)
        receipts.append({"name": path.name, "file_sha256": sha256_file(path),
                         "source_documents": len(hashes),
                         "source_set_sha256": canonical_hash(sorted(hashes))})
    return excluded, receipts


def iter_source_texts(source: Path, start_row: int) -> Iterable[tuple[int, str]]:
    import pyarrow.parquet as pq

    cursor = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64, columns=["text"]):
        end = cursor + batch.num_rows
        if end > start_row:
            for offset, text in enumerate(batch.column(0).to_pylist()):
                if cursor + offset >= start_row and isinstance(text, str):
                    yield cursor + offset, text
        cursor = end


def select_documents(rows: Iterable[tuple[int, str]], tokenizer: Any,
                     excluded: set[str], start_row: int) -> dict[str, list[dict]]:
    needed = sum(NATURAL_COUNTS.values())
    selected, seen = [], set(excluded)
    for source_row, text in rows:
        if source_row < start_row:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                               max_length=max(GRID))
        if len(ids) < max(GRID):
            continue
        seen.add(text_hash)
        selected.append({"source_row": source_row, "source_text_sha256": text_hash,
                         "document_ids": [int(value) for value in ids[:max(GRID)]]})
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise ValueError(f"need {needed} unique fresh >=8192-token documents; found {len(selected)}")
    result, cursor = {}, 0
    for split, count in NATURAL_COUNTS.items():
        result[split] = selected[cursor:cursor + count]
        cursor += count
    return result


def natural_rows(document: dict, split: str, index: int, bos_id: int) -> Iterable[dict]:
    # No EOS is added. The terminal target suffix is identical at every length.
    full = [bos_id] + document["document_ids"][:max(GRID) - 1]
    for length in GRID:
        ids = [bos_id] + full[-(length - 1):]
        yield {
            "family": "natural", "split": split, "sample_id": f"natural-{split}-{index:03d}",
            "variant": "natural", "length": length, "input_ids": ids,
            "target_start": length - TARGET_TOKENS, "target_tokens": TARGET_TOKENS,
            "source_row": document["source_row"],
            "source_text_sha256": document["source_text_sha256"],
            "prompt_ids_sha256": ids_hash(ids), "target_ids_sha256": ids_hash(ids[-TARGET_TOKENS:]),
        }


def make_blueprints(split: str) -> list[dict]:
    rng = random.Random(SEEDS[split])
    blueprints = []
    for index in range(CAPABILITY_COUNTS[split]):
        names: list[str] = []
        while len(names) < 8:
            key = "".join(rng.choices(string.ascii_lowercase, k=8))
            if key not in names:
                names.append(key)
        values = rng.sample(range(100000, 1000000), 8)
        facts = [{"index": i, "key": key, "code": str(value)}
                 for i, (key, value) in enumerate(zip(names, values))]
        stratum = index % len(TARGET_PAIRS)
        reverse = bool((index // len(TARGET_PAIRS)) % 2)
        query_indices = list(TARGET_PAIRS[stratum])
        if reverse:
            query_indices.reverse()
        identity = {"facts": facts, "query_indices": query_indices}
        blueprints.append({**identity, "blueprint_sha256": canonical_hash(identity),
                           "sample_id": f"capability-{split}-{index:03d}",
                           "depth_stratum": stratum, "query_reversed": reverse})
    return blueprints


def expected_text(blueprint: dict) -> str:
    a, b = [blueprint["facts"][i]["code"] for i in blueprint["query_indices"]]
    return f"{a}, {b}."


def render_user(blueprint: dict, filler_words: int, filler_tail: str = "") -> str:
    base, remainder = divmod(filler_words, 9)
    gaps = [" ".join(FILLER[j % len(FILLER)] for j in range(base + (i < remainder)))
            for i in range(9)]
    gaps[-1] += filler_tail
    parts = ["Read the eight records below. Only the explicitly labeled records contain codes.", gaps[0]]
    for i, fact in enumerate(blueprint["facts"]):
        parts.extend([f"Record {i + 1}: {fact['key']} has code {fact['code']}.", gaps[i + 1]])
    keys = [blueprint["facts"][i]["key"] for i in blueprint["query_indices"]]
    parts.append(
        f"Return the code for {keys[0]} followed by the code for {keys[1]}, in that order. "
        "Continue the assistant prefix with exactly two six-digit codes, separated by a comma "
        "and one space, and followed by one period. Output no other text, explanation, or newline."
    )
    return "\n\n".join(part for part in parts if part)


def chat_ids(tokenizer: Any, blueprint: dict, filler_words: int, filler_tail: str = "") -> list[int]:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_user(blueprint, filler_words, filler_tail)}],
        tokenize=False, add_generation_prompt=True,
    )
    # Tokenize the entire chat and assistant prefix together; no post-prefix padding.
    return [int(v) for v in tokenizer.encode(chat + ASSISTANT_PREFIX, add_special_tokens=False)]


def fit_prompt(tokenizer: Any, blueprint: dict, budget: int) -> tuple[list[int], int, str]:
    compact = chat_ids(tokenizer, blueprint, 0)
    if len(compact) > budget:
        raise ValueError("facts/query/chat template exceed prompt budget without filler")
    best_ids, best_words = compact, 0
    low, high = 0, budget * 2
    if len(chat_ids(tokenizer, blueprint, high)) <= budget:
        raise ValueError("filler tokenization cannot bracket requested prompt length")
    while low <= high:
        middle = (low + high) // 2
        candidate = chat_ids(tokenizer, blueprint, middle)
        if len(candidate) <= budget:
            if len(candidate) >= len(best_ids):
                best_ids, best_words = candidate, middle
            low = middle + 1
        else:
            high = middle - 1
    if budget - len(best_ids) > MAX_PROMPT_UNDERAGE:
        raise ValueError("filler-only fitting left excessive prompt underage")
    # Refine inside the final filler gap, never after the assistant prefix. A
    # bounded character-prefix scan often closes subword-tokenizer underage.
    best_tail = ""
    tail = " The routine review continues." * 2
    for chars in range(1, len(tail) + 1):
        if len(best_ids) == budget:
            break
        candidate = chat_ids(tokenizer, blueprint, best_words, tail[:chars])
        if len(best_ids) < len(candidate) <= budget:
            best_ids, best_tail = candidate, tail[:chars]
    return best_ids, best_words, best_tail


def capability_rows(blueprint: dict, split: str, tokenizer: Any) -> Iterable[dict]:
    expected = expected_text(blueprint)
    expected_ids = tokenizer.encode(expected, add_special_tokens=False)
    if tokenizer.decode(expected_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != expected:
        raise ValueError("expected answer does not round-trip exactly through tokenizer")
    if len(expected_ids) + 1 > GENERATION_BUDGET:
        raise ValueError("expected answer plus terminal EOS exceeds generation budget")
    for length in (0, *GRID):
        if length == 0:
            ids, words, tail = chat_ids(tokenizer, blueprint, 0), 0, ""
        else:
            ids, words, tail = fit_prompt(tokenizer, blueprint, length - GENERATION_BUDGET)
        yield {
            "family": "capability", "split": split, "sample_id": blueprint["sample_id"],
            "variant": "compact" if length == 0 else "distributed", "length": length,
            "input_ids": ids, "prompt_tokens": len(ids),
            "prompt_token_budget": len(ids) if length == 0 else length - GENERATION_BUDGET,
            "prompt_underage": 0 if length == 0 else length - GENERATION_BUDGET - len(ids),
            "prompt_ids_sha256": ids_hash(ids), "filler_words": words, "filler_tail": tail,
            "assistant_prefix": ASSISTANT_PREFIX, "expected_text": expected,
            "terminal_ids": [int(tokenizer.eos_token_id)], "generation_budget": GENERATION_BUDGET,
            "facts": blueprint["facts"], "query_indices": blueprint["query_indices"],
            "target_pair_indices": list(TARGET_PAIRS[blueprint["depth_stratum"]]),
            "target_keys": [blueprint["facts"][i]["key"] for i in blueprint["query_indices"]],
            "depth_stratum": blueprint["depth_stratum"], "query_reversed": blueprint["query_reversed"],
            "blueprint_sha256": blueprint["blueprint_sha256"],
        }


def scorer_fixtures(tokenizer: Any) -> dict:
    expected = "123456, 654321."
    encode = lambda text: [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
    eos = int(tokenizer.eos_token_id)
    correct = encode(expected)
    if tokenizer.decode(correct, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != expected:
        raise ValueError("scorer fixture tokenizer round-trip failed")
    cases = [
        ("exact_with_terminal_eos", correct + [eos], True),
        ("missing_terminal_eos", correct, False),
        ("missing_period", encode(expected[:-1]) + [eos], False),
        ("reversed_order", encode("654321, 123456.") + [eos], False),
        ("extra_prose", encode(expected + " Done.") + [eos], False),
        ("outer_newline_normalized", encode(expected + "\n") + [eos], True),
        ("outer_space_normalized", encode(" " + expected) + [eos], True),
        ("internal_spacing_changed", encode("123456,654321.") + [eos], False),
        ("tokens_after_eos", correct + [eos] + encode("x"), False),
        ("early_eos", encode("123456") + [eos], False),
    ]
    return {
        "expected_text": expected, "terminal_ids": [eos],
        "contract": "Generated continuation only: first terminal must be final raw token; decode preceding tokens with skip_special_tokens=False and clean_up_tokenization_spaces=False; normalize outer whitespace with strip, then exact string equality. Internal spacing/punctuation/order unchanged. No substring or first-code proxy.",
        "cases": [{"name": name, "generated_ids": ids, "expected_pass": passes}
                  for name, ids, passes in cases],
    }


def tokenizer_file_receipts(checkpoint: Path) -> list[dict]:
    patterns = ("tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.*",
                "merges.txt", "*.model", "chat_template*", "chat_templates/*.jinja")
    paths = {path for pattern in patterns for path in checkpoint.glob(pattern) if path.is_file()}
    if not paths:
        raise ValueError("checkpoint contains no tokenizer files")
    return [{"name": str(path.relative_to(checkpoint)), "sha256": sha256_file(path)}
            for path in sorted(paths)]


def load_tokenizer(checkpoint: Path) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("output directory must be fresh")
    if args.start_row < 0:
        raise ValueError("start_row must be nonnegative")
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if not str(config.get("model_type", "")).startswith("gemma"):
        raise ValueError("Native reference calibration requires a Gemma checkpoint")
    file_receipts = tokenizer_file_receipts(args.checkpoint)
    source_hash = sha256_file(args.source)
    excluded, exclusions = load_exclusions(args.exclude_rows_jsonl)
    tokenizer = load_tokenizer(args.checkpoint)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("checkpoint must provide BOS, EOS, and its actual chat template")
    documents = select_documents(iter_source_texts(args.source, args.start_row), tokenizer,
                                 excluded, args.start_row)
    blueprints = {split: make_blueprints(split) for split in SEEDS}
    document_sets = {split: {row["source_text_sha256"] for row in rows}
                     for split, rows in documents.items()}
    blueprint_sets = {split: {row["blueprint_sha256"] for row in rows}
                      for split, rows in blueprints.items()}
    if document_sets["calibration"] & document_sets["confirmation"] or any(
        values & excluded for values in document_sets.values()
    ) or blueprint_sets["calibration"] & blueprint_sets["confirmation"]:
        raise ValueError("calibration/confirmation/exclusion disjointness failed")
    fixtures = scorer_fixtures(tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    split_receipts = {}
    for split in SEEDS:
        path = args.output / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.incomplete")
        count = 0
        underages = []
        with temporary.open("x", encoding="utf-8") as handle:
            for index, document in enumerate(documents[split]):
                for row in natural_rows(document, split, index, int(tokenizer.bos_token_id)):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
            for blueprint in blueprints[split]:
                for row in capability_rows(blueprint, split, tokenizer):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    underages.append(row["prompt_underage"])
                    count += 1
        temporary.replace(path)
        split_receipts[split] = {
            "path": path.name, "sha256": sha256_file(path), "rows": count,
            "natural_documents": NATURAL_COUNTS[split],
            "natural_rows": NATURAL_COUNTS[split] * len(GRID),
            "capability_blueprints": CAPABILITY_COUNTS[split],
            "capability_rows": CAPABILITY_COUNTS[split] * (len(GRID) + 1),
            "source_document_set_sha256": canonical_hash(sorted(document_sets[split])),
            "blueprint_set_sha256": canonical_hash(sorted(blueprint_sets[split])),
            "max_prompt_underage": max(underages),
        }
    manifest = {
        "status": "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1", "schema_version": 1,
        "model_evaluation_status": "NOT_RUN", "L_ref_selected": False,
        "model_type": config["model_type"], "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "tokenizer_files": file_receipts, "tokenizer_files_sha256": canonical_hash(file_receipts),
        "tokenizer_class": type(tokenizer).__name__, "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "chat_template_sha256": canonical_hash(tokenizer.chat_template),
        "source": {"name": args.source.name, "file_sha256": source_hash,
                   "start_row_inclusive_zero_based": args.start_row},
        "grid": list(GRID), "seeds": SEEDS, "files": split_receipts,
        "selection": "First unique nonexcluded >=8192-token documents in source order; first 32 cal, next 64 confirmation; no loss/score filtering. Seeds govern independent capability blueprint RNGs only.",
        "natural_contract": "Take first 8191 document tokens plus BOS; each shorter row is BOS + final L-1 tokens; score only identical final 256 targets; never append EOS.",
        "capability_contract": {"records": 8, "queried_facts": 2, "distractors": 6,
                                "target_pairs_zero_based": TARGET_PAIRS,
                                "query_order_counterbalanced_within_stratum": True,
                                "generation_budget": GENERATION_BUDGET,
                                "prompt_budget": "L-48", "max_underage": MAX_PROMPT_UNDERAGE,
                                "filler": "fixed neutral procedural sentences, nine gaps; truncate filler only",
                                "compact_length": 0, "near_query": "OMITTED"},
        "disjointness": {"cal_confirmation_source_documents": True,
                         "cal_confirmation_blueprints": True,
                         "selected_documents_vs_supplied_exclusions": True,
                         "capability_uses_source_documents": False,
                         "historical_disjointness_scope": "supplied exclusion JSONL identities only; not unprovided prior sources"},
        "exclusion_receipts": exclusions, "excluded_source_documents": len(excluded),
        "prompt_ids_sha256_encoding": "contiguous little-endian signed int64 raw token bytes",
        "full_output_scorer_fixtures": fixtures,
        "ready_contract": "Both calibration.jsonl and confirmation.jsonl completed and hashed before this manifest; evaluator must verify hashes before use.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-rows-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--start-row", type=int, default=20000)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
