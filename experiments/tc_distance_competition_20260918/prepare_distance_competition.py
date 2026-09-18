#!/usr/bin/env python3
"""Prepare a frozen 2x2 distance-by-competition panel for T versus C."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer


SEED = 20260918
BASE_SAMPLES = 64
INPUT_TOKENS = 32600
STRONG_DISTRACTORS = 512
DISTANCE_CONDITIONS = ("near", "far")
COMPETITION_CONDITIONS = ("weak", "strong")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned(value: int, period: int) -> int:
    return value - value % period


def replace(body: list[int], start: int, chunk: list[int], occupied: list[tuple[int, int]]) -> None:
    stop = start + len(chunk)
    if start < 0 or stop > len(body):
        raise ValueError("chunk outside body")
    if any(not (stop <= left or start >= right) for left, right in occupied):
        raise ValueError("body chunks overlap")
    body[start:stop] = chunk
    occupied.append((start, stop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    header = tokenizer.encode(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    instruction = tokenizer.encode(
        "A target record is hidden in an archive. Ignore archive filler. "
        "Return only the seven-digit number bound to the requested key.\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    filler_phrase = tokenizer.encode(" archive", add_special_tokens=False)
    if not filler_phrase:
        raise ValueError("empty filler tokenization")

    rng = random.Random(SEED)
    used_values = set()
    rows = []
    base_receipts = []
    for base_index in range(BASE_SAMPLES):
        target_key = f"target-record-{base_index:03d}"
        target_value = str(rng.randrange(1_000_000, 10_000_000))
        while target_value in used_values:
            target_value = str(rng.randrange(1_000_000, 10_000_000))
        used_values.add(target_value)
        distractors = []
        for distractor_index in range(STRONG_DISTRACTORS):
            value = str(rng.randrange(1_000_000, 10_000_000))
            while value in used_values:
                value = str(rng.randrange(1_000_000, 10_000_000))
            used_values.add(value)
            distractors.append((f"distractor-{base_index:03d}-{distractor_index:03d}", value))

        target_chunk = tokenizer.encode(
            f"\nOne of the special magic numbers for {target_key} is: {target_value}.\n",
            add_special_tokens=False,
        )
        distractor_chunks = [
            tokenizer.encode(
                f"\nOne of the special magic numbers for {key} is: {value}.\n",
                add_special_tokens=False,
            )
            for key, value in distractors
        ]
        query = tokenizer.encode(
            f"\nWhat is the special magic number for {target_key} mentioned in the provided text?",
            add_special_tokens=False,
        )
        fixed = header + instruction
        tail = query + suffix
        body_length = INPUT_TOKENS - len(fixed) - len(tail)
        if body_length < 20000:
            raise ValueError("body unexpectedly short")
        filler = (filler_phrase * (body_length // len(filler_phrase) + 1))[:body_length]
        period = len(filler_phrase)
        target_positions = {
            "far": aligned(int(body_length * 0.06), period),
            "near": aligned(int(body_length * 0.74), period),
        }
        stride = max(len(chunk) for chunk in distractor_chunks) + period
        distractor_start = aligned(int(body_length * 0.14), period)
        distractor_positions = [distractor_start + stride * index for index in range(STRONG_DISTRACTORS)]
        if distractor_positions[-1] + len(distractor_chunks[-1]) >= int(body_length * 0.70):
            raise ValueError("strong competition records do not fit the frozen body")

        condition_ids = {}
        for competition in COMPETITION_CONDITIONS:
            for distance in DISTANCE_CONDITIONS:
                body = list(filler)
                occupied = []
                if competition == "strong":
                    for position, chunk in zip(distractor_positions, distractor_chunks):
                        replace(body, position, chunk, occupied)
                target_position = target_positions[distance]
                replace(body, target_position, target_chunk, occupied)
                prompt_ids = fixed + body + tail
                if len(prompt_ids) != INPUT_TOKENS:
                    raise AssertionError("input length drift")
                prompt_sha = sha256_bytes(json.dumps(prompt_ids, separators=(",", ":")).encode())
                row_id = f"tc_dc_{base_index:03d}_{distance}_{competition}"
                row = {
                    "row_id": row_id,
                    "task": "niah_multikey_2",
                    "family": "retrieval",
                    "prompt_ids": prompt_ids,
                    "prompt_sha256": prompt_sha,
                    "references": [target_value],
                    "length_cap": 32768,
                    "actual_length": INPUT_TOKENS,
                    "input_tokens": INPUT_TOKENS,
                    "max_new_tokens": 32,
                    "selection_mode": "frozen_distance_competition_factorial_v1",
                    "selection_uses_model_outputs": False,
                    "document_cluster_id": f"tc_dc_base_{base_index:03d}",
                    "source_document_id": f"tc_dc_base_{base_index:03d}",
                    "source_order_index": base_index,
                    "semantic_group_id": f"tc_dc_base_{base_index:03d}",
                    "group_id": f"tc_dc_base_{base_index:03d}",
                    "generator_seed": SEED,
                    "base_sample_id": f"base_{base_index:03d}",
                    "distance_condition": distance,
                    "competition_condition": competition,
                    "target_key": target_key,
                    "target_value": target_value,
                    "target_token_index": len(fixed) + target_position,
                    "query_token_index": len(fixed) + body_length,
                    "dependency_distance": len(fixed) + body_length - (len(fixed) + target_position),
                    "candidate_values": [target_value] + ([value for _, value in distractors] if competition == "strong" else []),
                    "candidate_count": STRONG_DISTRACTORS + 1 if competition == "strong" else 1,
                    "scorer_revision": "ruler_contains_v1",
                }
                rows.append(row)
                condition_ids[(distance, competition)] = prompt_ids

        for competition in COMPETITION_CONDITIONS:
            near = condition_ids[("near", competition)]
            far = condition_ids[("far", competition)]
            if Counter(near) != Counter(far):
                raise AssertionError("near/far token multisets differ")
        base_receipts.append({
            "base_sample_id": f"base_{base_index:03d}",
            "target_value": target_value,
            "near_dependency_distance": next(row["dependency_distance"] for row in rows if row["base_sample_id"] == f"base_{base_index:03d}" and row["distance_condition"] == "near"),
            "far_dependency_distance": next(row["dependency_distance"] for row in rows if row["base_sample_id"] == f"base_{base_index:03d}" and row["distance_condition"] == "far"),
        })

    args.out_root.mkdir(parents=True, exist_ok=True)
    inputs = args.out_root / "inputs.jsonl"
    with inputs.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "status": "TC_DISTANCE_COMPETITION_PANEL_FROZEN_V1",
        "seed": SEED,
        "base_samples": BASE_SAMPLES,
        "rows": len(rows),
        "input_tokens_each": INPUT_TOKENS,
        "cells": [f"{distance}_{competition}" for competition in COMPETITION_CONDITIONS for distance in DISTANCE_CONDITIONS],
        "rows_per_cell": BASE_SAMPLES,
        "inputs_sha256": sha256_file(inputs),
        "near_far_token_multiset_exact": True,
        "competition_candidate_counts": {"weak": 1, "strong": STRONG_DISTRACTORS + 1},
        "base_receipts": base_receipts,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "rows", "rows_per_cell", "input_tokens_each", "inputs_sha256")}, indent=2))


if __name__ == "__main__":
    main()
