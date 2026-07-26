#!/usr/bin/env python3
"""Build disjoint 16K natural-text backgrounds for causal routing probes.

The source text comes only from non-assistant LongAlign messages.  Rows are
split and selected by stable hashes, then cropped deterministically.  The
result is compatible with the existing OLMo-2 ``load_documents`` helper.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    LONGALIGN_ID,
    LONGALIGN_REVISION,
    MODEL_ID,
    atomic_json,
    iter_jsonl,
    sha256_file,
)


DEFAULT_LENGTH = 16_384


def stable_score(seed: int, purpose: str, source_id: str) -> int:
    payload = f"{seed}\0{purpose}\0{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def source_text(value: Mapping[str, Any]) -> str | None:
    messages = value.get("messages")
    if not isinstance(messages, list):
        return None
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            return None
        if role != "assistant" and isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts) if parts else None


def assign_split(seed: int, source_id: str) -> str:
    # An independent hash keeps selection order from influencing the split.
    return (
        "test"
        if stable_score(seed, "split", source_id) % 5 == 0
        else "validation"
    )


def candidate_rows(
    *,
    path: Path,
    seed: int,
    training_rows: int,
    evaluation_rows: int,
    oversample: int,
    minimum_length_hint: int,
    excluded_source_ids: set[str],
) -> tuple[dict[int, tuple[str, int, str]], dict[str, int]]:
    targets = {
        "validation": int(training_rows) * int(oversample),
        "test": int(evaluation_rows) * int(oversample),
    }
    heaps: dict[str, list[tuple[int, int, str]]] = {
        split: [] for split in targets
    }
    stats = {
        "source_rows": 0,
        "below_length_hint": 0,
        "unsupported_rows": 0,
        "excluded_rows": 0,
        "duplicate_source_ids": 0,
    }
    seen_source_ids: set[str] = set()
    for source_row, value in iter_jsonl(path):
        stats["source_rows"] += 1
        length_hint = value.get("length")
        if not isinstance(length_hint, int):
            stats["unsupported_rows"] += 1
            continue
        if int(length_hint) < int(minimum_length_hint):
            stats["below_length_hint"] += 1
            continue
        identifier = str(value.get("id", source_row))
        if identifier in excluded_source_ids:
            stats["excluded_rows"] += 1
            continue
        if identifier in seen_source_ids:
            stats["duplicate_source_ids"] += 1
            continue
        seen_source_ids.add(identifier)
        split = assign_split(seed, identifier)
        score = stable_score(seed, f"select:{split}", identifier)
        # Negative values make heap[0] the worst retained candidate.
        item = (-score, -int(source_row), identifier)
        heap = heaps[split]
        if len(heap) < targets[split]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected: dict[int, tuple[str, int, str]] = {}
    for split, heap in heaps.items():
        if len(heap) != targets[split]:
            raise RuntimeError(
                f"only {len(heap)} {split} candidates; need {targets[split]}"
            )
        for negative_score, negative_row, identifier in heap:
            source_row = -negative_row
            selected[source_row] = (
                split,
                -negative_score,
                identifier,
            )
        stats[f"{split}_candidates"] = len(heap)
    return selected, stats


def tokenize_candidates(
    *,
    path: Path,
    tokenizer: Any,
    selected: Mapping[int, tuple[str, int, str]],
    seed: int,
    length: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    stats = {
        "tokenized_candidates": 0,
        "unsupported_messages": 0,
        "too_short_after_tokenization": 0,
    }
    batch: list[
        tuple[int, Mapping[str, Any], tuple[str, int, str], str]
    ] = []

    def consume(
        values: list[
            tuple[int, Mapping[str, Any], tuple[str, int, str], str]
        ],
    ) -> None:
        encoded = tokenizer(
            [text for _, _, _, text in values],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != len(values):
            raise RuntimeError("background tokenizer row-count drift")
        for (
            source_row,
            _,
            candidate,
            _,
        ), token_ids in zip(values, encoded):
            split, score, identifier = candidate
            stats["tokenized_candidates"] += 1
            if len(token_ids) < int(length):
                stats["too_short_after_tokenization"] += 1
                continue
            crop_limit = len(token_ids) - int(length)
            crop_score = stable_score(seed, "crop", identifier)
            crop_start = (
                0 if crop_limit == 0 else crop_score % (crop_limit + 1)
            )
            crop = np.asarray(
                token_ids[crop_start : crop_start + int(length)],
                dtype=np.uint32,
            )
            if crop.shape != (int(length),):
                raise RuntimeError("background crop shape drift")
            rows[split].append(
                {
                    "selection_score": int(score),
                    "source_row": int(source_row),
                    "source_id": identifier,
                    "original_tokens": len(token_ids),
                    "crop_start": int(crop_start),
                    "tokens": crop,
                }
            )

    for source_row, value in iter_jsonl(path):
        candidate = selected.get(source_row)
        if candidate is None:
            continue
        split, score, identifier = candidate
        text = source_text(value)
        if text is None:
            stats["unsupported_messages"] += 1
            continue
        batch.append((source_row, value, candidate, text))
        if len(batch) == 32:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)
    for values in rows.values():
        values.sort(
            key=lambda row: (
                int(row["selection_score"]),
                int(row["source_row"]),
            )
        )
    return rows, stats


def write_background(
    *,
    output_dir: Path,
    rows: Mapping[str, list[dict[str, Any]]],
    training_rows: int,
    evaluation_rows: int,
    length: int,
    seed: int,
    source_path: Path,
    exclusion_path: Path,
    checkpoint: Path,
    selection_stats: Mapping[str, int],
    tokenization_stats: Mapping[str, int],
) -> dict[str, Any]:
    requested = {
        "validation": int(training_rows),
        "test": int(evaluation_rows),
    }
    for split, count in requested.items():
        if len(rows[split]) < count:
            raise RuntimeError(
                f"only {len(rows[split])} tokenized {split} rows; need {count}"
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    array_path = output_dir / "documents_L16384.npy"
    metadata_path = output_dir / "documents_L16384.metadata.json"
    temporary = array_path.with_name(array_path.name + ".incomplete")
    total = sum(requested.values())
    documents = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint32,
        shape=(total, int(length)),
    )
    metadata: list[dict[str, Any]] = []
    output_row = 0
    for split in ("validation", "test"):
        for value in rows[split][: requested[split]]:
            documents[output_row] = value["tokens"]
            metadata.append(
                {
                    "row": output_row,
                    "source": (
                        f"longalign/{split}/{value['source_id']}"
                    ),
                    "source_id": value["source_id"],
                    "source_row": int(value["source_row"]),
                    "split": split,
                    "original_tokens": int(value["original_tokens"]),
                    "crop_start": int(value["crop_start"]),
                }
            )
            output_row += 1
    documents.flush()
    del documents
    os.replace(temporary, array_path)
    atomic_json(metadata_path, metadata)
    manifest = {
        "format_version": 1,
        "status": "OLMO2_LORA_PROBE_BACKGROUND_PREPARED",
        "model_id": MODEL_ID,
        "tokenizer_checkpoint": checkpoint.name,
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "source": {
            "id": LONGALIGN_ID,
            "revision": LONGALIGN_REVISION,
            "path_name": source_path.name,
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        },
        "exclusion": {
            "path_name": exclusion_path.name,
            "bytes": exclusion_path.stat().st_size,
            "sha256": sha256_file(exclusion_path),
        },
        "seed": int(seed),
        "shape": [total, int(length)],
        "dtype": "uint32",
        "splits": requested,
        "selection_statistics": dict(selection_stats),
        "tokenization_statistics": dict(tokenization_stats),
        "files": {
            array_path.name: {
                "bytes": array_path.stat().st_size,
                "sha256": sha256_file(array_path),
            },
            metadata_path.name: {
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return verify_background(output_dir)


def verify_background(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, receipt in manifest["files"].items():
        path = output_dir / name
        if path.stat().st_size != int(receipt["bytes"]):
            raise RuntimeError(f"size mismatch: {path}")
        if sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"hash mismatch: {path}")
    array = np.load(
        output_dir / "documents_L16384.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected = tuple(int(value) for value in manifest["shape"])
    if array.dtype != np.uint32 or tuple(array.shape) != expected:
        raise RuntimeError("background array shape/dtype mismatch")
    metadata = json.loads(
        (output_dir / "documents_L16384.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if len(metadata) != expected[0]:
        raise RuntimeError("background metadata row-count mismatch")
    seen_sources: set[str] = set()
    split_counts = {"validation": 0, "test": 0}
    for index, row in enumerate(metadata):
        if int(row["row"]) != index:
            raise RuntimeError("background metadata ordering drift")
        source = str(row["source"])
        if source in seen_sources:
            raise RuntimeError("background source leakage/duplication")
        seen_sources.add(source)
        split = str(row["split"])
        split_counts[split] += 1
        if f"/{split}/" not in source:
            raise RuntimeError("background split/source mismatch")
    if split_counts != {
        key: int(value) for key, value in manifest["splits"].items()
    }:
        raise RuntimeError("background split-count mismatch")
    return {
        "status": "OLMO2_LORA_PROBE_BACKGROUND_VERIFIED",
        "manifest_sha256": sha256_file(manifest_path),
        "shape": list(expected),
        "split_counts": split_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--longalign-jsonl", type=Path)
    parser.add_argument("--exclude-rows-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--training-rows", type=int, default=512)
    parser.add_argument("--evaluation-rows", type=int, default=256)
    parser.add_argument("--oversample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify_only:
        print(json.dumps(verify_background(output_dir), indent=2))
        return
    if (
        args.checkpoint is None
        or args.longalign_jsonl is None
        or args.exclude_rows_jsonl is None
    ):
        raise ValueError(
            "checkpoint, LongAlign JSONL, and exclusion rows are required"
        )
    if int(args.length) != DEFAULT_LENGTH:
        raise ValueError(f"formal background length must be {DEFAULT_LENGTH}")
    if int(args.oversample) < 2:
        raise ValueError("oversample must be at least two")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    from transformers import AutoTokenizer

    checkpoint = args.checkpoint.resolve()
    source_path = args.longalign_jsonl.resolve()
    exclusion_path = args.exclude_rows_jsonl.resolve()
    excluded_source_ids: set[str] = set()
    with exclusion_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                excluded_source_ids.add(
                    str(json.loads(line)["source_id"])
                )
    if not excluded_source_ids:
        raise RuntimeError("LongAlign exclusion set is empty")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.model_max_length = 1_000_000_000
    selected, selection_stats = candidate_rows(
        path=source_path,
        seed=int(args.seed),
        training_rows=int(args.training_rows),
        evaluation_rows=int(args.evaluation_rows),
        oversample=int(args.oversample),
        minimum_length_hint=int(args.length),
        excluded_source_ids=excluded_source_ids,
    )
    rows, tokenization_stats = tokenize_candidates(
        path=source_path,
        tokenizer=tokenizer,
        selected=selected,
        seed=int(args.seed),
        length=int(args.length),
    )
    result = write_background(
        output_dir=output_dir,
        rows=rows,
        training_rows=int(args.training_rows),
        evaluation_rows=int(args.evaluation_rows),
        length=int(args.length),
        seed=int(args.seed),
        source_path=source_path,
        exclusion_path=exclusion_path,
        checkpoint=checkpoint,
        selection_stats=selection_stats,
        tokenization_stats=tokenization_stats,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
