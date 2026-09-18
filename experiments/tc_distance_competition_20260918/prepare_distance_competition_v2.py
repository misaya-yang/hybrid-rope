#!/usr/bin/env python3
"""Prepare fresh position-gap confirmation and separate parity panels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer


CONFIRM_SEED = 2026091802
PARITY_SEED = 2026091801
CONFIRM_BASE_SAMPLES = 64
PARITY_BASE_SAMPLES = 4
DISTANCE_CONDITIONS = ("near", "far")
COMPETITION_CONDITIONS = ("neutral", "structured_kv")
NEAR_DEPENDENCY_DISTANCE = 8463
FAR_DEPENDENCY_DISTANCE = 30595
POSITION_GAP = FAR_DEPENDENCY_DISTANCE - NEAR_DEPENDENCY_DISTANCE
STRUCTURED_DISTRACTORS = 512
PARITY_COMMON_OFFSET = 17
MAX_POSITION_ID = 32767


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_value(rng: random.Random, used: set[str]) -> str:
    while True:
        value = str(rng.randrange(1_000_000, 10_000_000))
        if value not in used:
            used.add(value)
            return value


def find_unique_subsequence(haystack: list[int], needle: list[int]) -> int:
    matches = [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index:index + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one target-value token anchor, found {len(matches)}")
    return matches[0]


def build_panel(tokenizer, *, seed: int, base_samples: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    used_values: set[str] = set()
    filler_token_ids = tokenizer.encode(" archive", add_special_tokens=False)
    if len(filler_token_ids) != 1:
        raise ValueError("neutral filler must be exactly one token")
    filler_token = filler_token_ids[0]
    header = tokenizer.encode(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    instruction = tokenizer.encode(
        "The archive contains compact KEY=VALUE records. Return only the "
        "seven-digit VALUE bound to the requested KEY.\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    fixed = header + instruction
    target_start = max(256, len(fixed) + 64)
    rows = []
    for base_index in range(base_samples):
        base_id = f"{prefix}_base_{base_index:03d}"
        target_key = f"T{base_index:02x}"
        target_value = unique_value(rng, used_values)
        distractors = [
            (f"D{base_index:02x}{index:03x}", unique_value(rng, used_values))
            for index in range(STRUCTURED_DISTRACTORS)
        ]
        target_chunk = tokenizer.encode(
            f"\n{target_key}={target_value}", add_special_tokens=False,
        )
        structured_block = []
        for key, value in distractors:
            structured_block.extend(tokenizer.encode(
                f"\n{key}={value}", add_special_tokens=False,
            ))
        neutral_block = [filler_token] * len(structured_block)
        query = tokenizer.encode(
            f"\nWhat is the seven-digit value for key {target_key}?",
            add_special_tokens=False,
        )
        prompt_prefix = fixed + [filler_token] * (target_start - len(fixed)) + target_chunk
        target_value_ids = tokenizer.encode(target_value, add_special_tokens=False)
        target_value_index = find_unique_subsequence(prompt_prefix, target_value_ids)
        query_index = target_value_index + NEAR_DEPENDENCY_DISTANCE
        condition_prompts = {}
        for competition, competition_block in (
            ("neutral", neutral_block),
            ("structured_kv", structured_block),
        ):
            before_query = prompt_prefix + competition_block
            if len(before_query) > query_index:
                raise ValueError(
                    f"competition block exceeds the fixed near distance: {len(before_query)}>{query_index}"
                )
            prompt_ids = (
                before_query
                + [filler_token] * (query_index - len(before_query))
                + query
                + suffix
            )
            gap_boundary = len(prompt_prefix)
            near_positions = list(range(len(prompt_ids)))
            far_positions = [
                position if index < gap_boundary else position + POSITION_GAP
                for index, position in enumerate(near_positions)
            ]
            positions = {"near": near_positions, "far": far_positions}
            for distance in DISTANCE_CONDITIONS:
                position_ids = positions[distance]
                actual_distance = position_ids[query_index] - position_ids[target_value_index]
                expected_distance = (
                    NEAR_DEPENDENCY_DISTANCE
                    if distance == "near"
                    else FAR_DEPENDENCY_DISTANCE
                )
                if actual_distance != expected_distance:
                    raise AssertionError("dependency-distance construction drift")
                if min(position_ids) < 0 or max(position_ids) + PARITY_COMMON_OFFSET > MAX_POSITION_ID:
                    raise ValueError("position IDs leave the declared 32K range or parity reserve")
                row_id = f"{prefix}_{base_index:03d}_{distance}_{competition}"
                row = {
                    "row_id": row_id,
                    "task": "niah_multikey_2",
                    "family": "retrieval",
                    "prompt_ids": prompt_ids,
                    "position_ids": position_ids,
                    "prompt_sha256": sha256_json(prompt_ids),
                    "position_ids_sha256": sha256_json(position_ids),
                    "references": [target_value],
                    "length_cap": MAX_POSITION_ID + 1,
                    "actual_length": len(prompt_ids),
                    "input_tokens": len(prompt_ids),
                    "max_new_tokens": 32,
                    "selection_mode": "fresh_position_gap_distance_competition_v2",
                    "selection_uses_model_outputs": False,
                    "document_cluster_id": base_id,
                    "source_document_id": base_id,
                    "semantic_group_id": base_id,
                    "group_id": base_id,
                    "generator_seed": seed,
                    "base_sample_id": base_id,
                    "distance_condition": distance,
                    "competition_condition": competition,
                    "competition_label": (
                        "matched_length_neutral_context"
                        if competition == "neutral"
                        else "structured_key_value_competition"
                    ),
                    "target_key": target_key,
                    "target_value": target_value,
                    "target_value_token_index": target_value_index,
                    "query_token_index": query_index,
                    "gap_boundary_token_index": gap_boundary,
                    "dependency_distance": actual_distance,
                    "position_gap": 0 if distance == "near" else POSITION_GAP,
                    "candidate_values": (
                        [target_value]
                        if competition == "neutral"
                        else [target_value] + [value for _, value in distractors]
                    ),
                    "candidate_count": (
                        1 if competition == "neutral" else STRUCTURED_DISTRACTORS + 1
                    ),
                    "scorer_revision": "ruler_contains_v1",
                }
                rows.append(row)
                condition_prompts[(competition, distance)] = (
                    row["prompt_sha256"], row["position_ids_sha256"]
                )
        for competition in COMPETITION_CONDITIONS:
            if condition_prompts[(competition, "near")][0] != condition_prompts[(competition, "far")][0]:
                raise AssertionError("near/far token IDs differ")
            if condition_prompts[(competition, "near")][1] == condition_prompts[(competition, "far")][1]:
                raise AssertionError("near/far position IDs unexpectedly match")
    return rows


def write_panel(path: Path, rows: list[dict]) -> dict:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    cells = defaultdict(list)
    for row in rows:
        cell = f"{row['distance_condition']}_{row['competition_condition']}"
        cells[cell].append((
            row["row_id"], row["prompt_sha256"], row["position_ids_sha256"]
        ))
    return {
        "rows": len(rows),
        "input_file_sha256": sha256_file(path),
        "row_identity_sha256": sha256_json(sorted(
            (row["row_id"], row["prompt_sha256"], row["position_ids_sha256"])
            for row in rows
        )),
        "cell_receipt_sha256": {
            cell: sha256_json(sorted(receipts))
            for cell, receipts in sorted(cells.items())
        },
        "base_sample_ids": sorted({row["base_sample_id"] for row in rows}),
        "physical_lengths": sorted({row["input_tokens"] for row in rows}),
        "position_maxima": sorted({max(row["position_ids"]) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    confirm_rows = build_panel(
        tokenizer,
        seed=CONFIRM_SEED,
        base_samples=CONFIRM_BASE_SAMPLES,
        prefix="tc_dc_confirm2",
    )
    parity_rows = build_panel(
        tokenizer,
        seed=PARITY_SEED,
        base_samples=PARITY_BASE_SAMPLES,
        prefix="tc_dc_parity2",
    )
    confirm_receipt = write_panel(args.out_root / "inputs_confirm_v2.jsonl", confirm_rows)
    parity_receipt = write_panel(args.out_root / "inputs_parity_v2.jsonl", parity_rows)
    manifest = {
        "status": "TC_DISTANCE_COMPETITION_CONFIRM_PANEL_FROZEN_V2",
        "protocol_revision": "fresh_position_gap_distance_competition_v2",
        "confirmation_seed": CONFIRM_SEED,
        "parity_seed": PARITY_SEED,
        "confirmation_base_samples": CONFIRM_BASE_SAMPLES,
        "parity_base_samples": PARITY_BASE_SAMPLES,
        "near_dependency_distance": NEAR_DEPENDENCY_DISTANCE,
        "far_dependency_distance": FAR_DEPENDENCY_DISTANCE,
        "position_gap": POSITION_GAP,
        "structured_distractors": STRUCTURED_DISTRACTORS,
        "parity_common_offset": PARITY_COMMON_OFFSET,
        "max_position_id": MAX_POSITION_ID,
        "distance_contract": (
            "near/far share exact token IDs; only a suffix position-ID offset changes"
        ),
        "competition_contract": (
            "structured key-value competition versus token-count-matched neutral context"
        ),
        "confirmation": confirm_receipt,
        "parity": parity_receipt,
    }
    manifest_path = args.out_root / "manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "confirmation_rows": confirm_receipt["rows"],
        "confirmation_input_file_sha256": confirm_receipt["input_file_sha256"],
        "confirmation_row_identity_sha256": confirm_receipt["row_identity_sha256"],
        "physical_lengths": confirm_receipt["physical_lengths"],
        "position_maxima": confirm_receipt["position_maxima"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
