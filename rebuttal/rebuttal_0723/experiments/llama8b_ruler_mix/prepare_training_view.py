#!/usr/bin/env python3
"""Build the fixed physical-8K answer-only RULER mixture with natural replay."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .common import (
    TASKS,
    TRAIN_LENGTH,
    TRAIN_SOURCE_STATUS,
    VIEW_STATUS,
    atomic_json,
    canonical_target,
    load_jsonl,
    row_sha256,
    sha256_file,
)


@dataclass
class EncodedNatural:
    source_index: int
    input_ids: list[int]
    target_start: int
    instruction_sha256: str
    output_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--longalpaca-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=96)
    parser.add_argument("--validation-per-task", type=int, default=4)
    parser.add_argument("--natural-rows", type=int, default=128)
    parser.add_argument("--natural-min-length", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=20_420_726)
    return parser.parse_args()


def text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_source_manifest(root: Path) -> dict[str, Any]:
    path = root / "train_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != TRAIN_SOURCE_STATUS:
        raise RuntimeError("RULER source status drift")
    if int(manifest["protocol"]["physical_training_length"]) != TRAIN_LENGTH:
        raise RuntimeError("RULER source training length drift")
    if tuple(manifest["protocol"]["tasks"]) != TASKS:
        raise RuntimeError("RULER source task set drift")
    for record in manifest["files"].values():
        source = root / record["path"]
        if (
            not source.is_file()
            or source.stat().st_size != int(record["size_bytes"])
            or sha256_file(source) != record["sha256"]
        ):
            raise RuntimeError(f"RULER source file drift: {source}")
    return manifest


def encode_ruler_row(
    tokenizer: Any,
    task: str,
    row: dict[str, Any],
) -> tuple[list[int], int, str]:
    prompt = tokenizer(
        str(row["input"]),
        add_special_tokens=False,
    ).input_ids
    prefix = tokenizer(
        str(row.get("answer_prefix", "")),
        add_special_tokens=False,
    ).input_ids
    target_text = canonical_target(task, row["outputs"])
    target = tokenizer(
        target_text,
        add_special_tokens=False,
    ).input_ids + [int(tokenizer.eos_token_id)]
    combined = list(prompt) + list(prefix) + list(target)
    if len(combined) > TRAIN_LENGTH:
        raise RuntimeError(
            f"{task} encoded row exceeds physical 8K: {len(combined)}"
        )
    return combined, len(prompt) + len(prefix), target_text


def natural_reservoir(
    *,
    tokenizer: Any,
    path: Path,
    rows: int,
    minimum_length: int,
    seed: int,
) -> tuple[list[EncodedNatural], dict[str, int]]:
    import ijson

    rng = random.Random(seed)
    reservoir: list[EncodedNatural] = []
    eligible = 0
    visited = 0
    batch_size = 64
    pending: list[tuple[int, str, str]] = []

    def consume_batch() -> None:
        nonlocal eligible
        if not pending:
            return
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, instruction, _ in pending
        ]
        prompt_batches = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
        ).input_ids
        target_batches = tokenizer(
            [output for _, _, output in pending],
            add_special_tokens=False,
            padding=False,
        ).input_ids
        for (
            (source_index, instruction, output),
            prompt,
            target_without_eos,
        ) in zip(
            pending,
            prompt_batches,
            target_batches,
            strict=True,
        ):
            target = list(target_without_eos) + [
                int(tokenizer.eos_token_id)
            ]
            combined = list(prompt) + target
            if not minimum_length <= len(combined) <= TRAIN_LENGTH:
                continue
            eligible += 1
            record = EncodedNatural(
                source_index=source_index,
                input_ids=combined,
                target_start=len(prompt),
                instruction_sha256=text_sha256(instruction),
                output_sha256=text_sha256(output),
            )
            if len(reservoir) < rows:
                reservoir.append(record)
            else:
                replacement = rng.randrange(eligible)
                if replacement < rows:
                    reservoir[replacement] = record
        pending.clear()

    with path.open("rb") as handle:
        for source_index, item in enumerate(ijson.items(handle, "item")):
            visited += 1
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            if not instruction or not output:
                continue
            pending.append((source_index, instruction, output))
            if len(pending) == batch_size:
                consume_batch()
    consume_batch()
    if len(reservoir) != rows:
        raise RuntimeError(
            f"only {len(reservoir)} eligible natural rows, required {rows}"
        )
    reservoir.sort(key=lambda value: value.source_index)
    return reservoir, {"visited": visited, "eligible": eligible}


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer.resolve()
    sources = args.sources.resolve()
    longalpaca = args.longalpaca_json.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    if (
        int(args.train_per_task) + int(args.validation_per_task)
        != 100
    ):
        raise ValueError("the official source has exactly 100 rows per task")
    if int(args.natural_rows) != 128:
        raise ValueError("registered protocol requires 128 replay rows")
    if int(args.natural_min_length) != 4_096:
        raise ValueError("registered replay minimum is 4096 tokens")
    source_manifest = verify_source_manifest(sources)
    if not longalpaca.is_file():
        raise FileNotFoundError(longalpaca)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else int(tokenizer.eos_token_id)
    )

    ruler_rows: list[tuple[str, dict[str, Any], int]] = []
    seen_hashes: set[str] = set()
    for task in TASKS:
        path = sources / "train" / f"L{TRAIN_LENGTH}" / task / "test.jsonl"
        rows = load_jsonl(path)
        if len(rows) != 100:
            raise RuntimeError(f"{task} training source row-count drift")
        for index, row in enumerate(rows):
            digest = row_sha256(row)
            if digest in seen_hashes:
                raise RuntimeError("duplicate row across RULER task sources")
            seen_hashes.add(digest)
            split = 0 if index < int(args.train_per_task) else 1
            ruler_rows.append((task, row, split))

    natural, natural_stats = natural_reservoir(
        tokenizer=tokenizer,
        path=longalpaca,
        rows=int(args.natural_rows),
        minimum_length=int(args.natural_min_length),
        seed=int(args.seed) + 91_003,
    )

    total = len(ruler_rows) + len(natural)
    expected_total = 13 * 100 + 128
    if total != expected_total:
        raise RuntimeError(f"training-view row drift: {total} != {expected_total}")
    input_ids = np.full(
        (total, TRAIN_LENGTH),
        pad_token_id,
        dtype=np.uint32,
    )
    assistant_mask = np.zeros((total, TRAIN_LENGTH), dtype=np.uint8)
    lengths = np.zeros(total, dtype=np.int32)
    split = np.zeros(total, dtype=np.uint8)
    metadata: list[dict[str, Any]] = []

    cursor = 0
    for task, row, row_split in ruler_rows:
        encoded, target_start, target_text = encode_ruler_row(
            tokenizer,
            task,
            row,
        )
        length = len(encoded)
        input_ids[cursor, :length] = np.asarray(encoded, dtype=np.uint32)
        assistant_mask[cursor, target_start:length] = 1
        lengths[cursor] = length
        split[cursor] = row_split
        metadata.append(
            {
                "row": cursor,
                "source": "official_ruler",
                "task": task,
                "source_index": int(row["index"]),
                "source_row_sha256": row_sha256(row),
                "split": "train" if row_split == 0 else "validation",
                "encoded_length": length,
                "supervised_tokens": length - target_start,
                "canonical_target": target_text,
            }
        )
        cursor += 1

    for record in natural:
        length = len(record.input_ids)
        input_ids[cursor, :length] = np.asarray(
            record.input_ids,
            dtype=np.uint32,
        )
        assistant_mask[cursor, record.target_start:length] = 1
        lengths[cursor] = length
        split[cursor] = 0
        metadata.append(
            {
                "row": cursor,
                "source": "longalpaca_instruction_replay",
                "source_index": record.source_index,
                "instruction_sha256": record.instruction_sha256,
                "output_sha256": record.output_sha256,
                "split": "train",
                "encoded_length": length,
                "supervised_tokens": length - record.target_start,
            }
        )
        cursor += 1
    if cursor != total:
        raise RuntimeError("training-view cursor drift")

    train_rows = int((split == 0).sum())
    validation_rows = int((split == 1).sum())
    if (train_rows, validation_rows) != (1_376, 52):
        raise RuntimeError(
            f"split drift: train={train_rows}, validation={validation_rows}"
        )
    if train_rows % 8:
        raise RuntimeError("training split must divide exactly by global batch 8")
    if not np.all(assistant_mask.sum(axis=1) > 0):
        raise RuntimeError("one or more rows have no supervised answer tokens")

    incomplete.mkdir(parents=True)
    np.save(incomplete / "input_ids.npy", input_ids, allow_pickle=False)
    np.save(
        incomplete / "assistant_mask.npy",
        assistant_mask,
        allow_pickle=False,
    )
    np.save(incomplete / "lengths.npy", lengths, allow_pickle=False)
    np.save(incomplete / "split.npy", split, allow_pickle=False)
    with (incomplete / "metadata.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in metadata:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    file_receipts = {}
    for name in (
        "input_ids.npy",
        "assistant_mask.npy",
        "lengths.npy",
        "split.npy",
        "metadata.jsonl",
    ):
        path = incomplete / name
        file_receipts[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "status": VIEW_STATUS,
        "protocol": {
            "physical_sequence_length": TRAIN_LENGTH,
            "tasks": list(TASKS),
            "ruler_train_rows_per_task": int(args.train_per_task),
            "ruler_validation_rows_per_task": int(
                args.validation_per_task
            ),
            "natural_replay_rows": int(args.natural_rows),
            "natural_replay_min_tokens": int(args.natural_min_length),
            "objective": "answer_only_including_eos",
            "padding": "right_eos_padding_with_no_padding_loss",
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "registered_global_batch_size": 8,
            "epochs": 3,
            "processed_input_tokens": train_rows * TRAIN_LENGTH * 3,
            "seed": int(args.seed),
        },
        "sources": {
            "ruler_manifest_sha256": sha256_file(
                sources / "train_manifest.json"
            ),
            "ruler_status": source_manifest["status"],
            "longalpaca_basename": longalpaca.name,
            "longalpaca_sha256": sha256_file(longalpaca),
            "natural_reservoir": natural_stats,
        },
        "tokenizer": {
            "basename": tokenizer_path.name,
            "tokenizer_json_sha256": sha256_file(
                tokenizer_path / "tokenizer.json"
            ),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": pad_token_id,
        },
        "storage": {
            "input_ids_dtype": "uint32",
            "assistant_mask_dtype": "uint8",
            "shape": [total, TRAIN_LENGTH],
            "files": file_receipts,
        },
    }
    atomic_json(incomplete / "manifest.json", manifest)
    incomplete.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
