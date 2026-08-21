#!/usr/bin/env python3
"""Prepare physical continuous-8K natural-span retrieval rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .continuous_8k_adaptation import DATA_STATUS, LENGTH
from .prepare_4k_natural_span_retrieval import occurrences, tokenizer_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--validation-rows", type=int, default=64)
    parser.add_argument("--anchor-tokens", type=int, default=8)
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--position-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_821)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = args.source_view.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if int(args.rows) <= int(args.validation_rows):
        raise ValueError("rows must exceed validation rows")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    if (
        int(tokenizer.bos_token_id) != 100_257
        or int(tokenizer.eos_token_id) != 100_257
        or int(tokenizer.pad_token_id) != 100_277
    ):
        raise RuntimeError("OLMo tokenizer special-token contract drift")
    loaded = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
        raise RuntimeError("source view must contain one rank-two tensor")
    source_ids = loaded.detach().cpu().numpy()
    if source_ids.shape[1] != 4_096:
        raise RuntimeError("continuous-8K source requires packed 4K rows")
    if source_ids.shape[0] < 2 * int(args.rows):
        raise RuntimeError("continuous-8K source has too few rows")

    prefix = tokenizer.encode(
        (
            f"{tokenizer.bos_token}<|user|>\n"
            "Read the passage. Locate the quoted anchor and copy exactly "
            "the text immediately following it.\n\nPassage:\n"
        ),
        add_special_tokens=False,
    )
    query_prefix = tokenizer.encode("\n\nAnchor:\n", add_special_tokens=False)
    query_suffix = tokenizer.encode(
        (
            "\n\nReturn only the text immediately following that anchor."
            "\n<|assistant|>\n"
        ),
        add_special_tokens=False,
    )
    overhead = (
        len(prefix)
        + len(query_prefix)
        + int(args.anchor_tokens)
        + len(query_suffix)
        + int(args.answer_tokens)
        + 1
    )
    passage_tokens = LENGTH - overhead
    if passage_tokens < 8_000:
        raise RuntimeError("continuous-8K prompt leaves too little passage")

    rng = np.random.default_rng(int(args.seed))
    chosen = rng.permutation(source_ids.shape[0])[: 2 * int(args.rows)]
    inputs = np.full(
        (int(args.rows), LENGTH),
        int(tokenizer.pad_token_id),
        dtype=np.uint32,
    )
    masks = np.zeros((int(args.rows), LENGTH), dtype=np.uint8)
    labels = np.full((int(args.rows), LENGTH), -100, dtype=np.int32)
    query_starts = np.zeros(int(args.rows), dtype=np.int32)
    active_lengths = np.full(int(args.rows), LENGTH, dtype=np.int32)
    lengths = np.full(int(args.rows), LENGTH, dtype=np.int32)
    metadata: list[dict[str, Any]] = []
    forbidden = {
        int(tokenizer.bos_token_id),
        int(tokenizer.eos_token_id),
        int(tokenizer.pad_token_id),
    }
    usable_end = min(
        3_900,
        passage_tokens
        - int(args.anchor_tokens)
        - int(args.answer_tokens)
        - 8,
    )

    for row_index in range(int(args.rows)):
        source_rows = chosen[2 * row_index : 2 * row_index + 2]
        joined = np.concatenate(
            [source_ids[int(value)] for value in source_rows]
        ).astype(np.int64, copy=False)
        built = None
        target_bin = row_index % int(args.position_bins)
        for _ in range(128):
            start = int(rng.integers(0, len(joined) - passage_tokens + 1))
            passage = np.asarray(
                joined[start : start + passage_tokens], dtype=np.int64
            )
            local = 16 + int(
                round(
                    (usable_end - 16)
                    * target_bin
                    / max(int(args.position_bins) - 1, 1)
                )
            )
            local += int(rng.integers(-8, 9))
            local = max(8, min(local, usable_end))
            anchor = passage[local : local + int(args.anchor_tokens)]
            answer = passage[
                local
                + int(args.anchor_tokens) : local
                + int(args.anchor_tokens)
                + int(args.answer_tokens)
            ]
            if (
                any(int(token) in forbidden for token in anchor)
                or any(int(token) in forbidden for token in answer)
                or occurrences(passage, anchor) != 1
            ):
                target_bin = int(rng.integers(0, int(args.position_bins)))
                continue
            sequence = np.asarray(
                [
                    *prefix,
                    *passage.tolist(),
                    *query_prefix,
                    *anchor.tolist(),
                    *query_suffix,
                    *answer.tolist(),
                    int(tokenizer.eos_token_id),
                ],
                dtype=np.int64,
            )
            if len(sequence) != LENGTH:
                raise RuntimeError("continuous-8K sequence length drift")
            built = (sequence, start, local, anchor, answer)
            break
        if built is None:
            raise RuntimeError(f"failed to build row {row_index}")
        sequence, start, local, anchor, answer = built
        inputs[row_index] = sequence.astype(np.uint32)
        query_start = len(prefix) + passage_tokens
        query_starts[row_index] = query_start
        masks[
            row_index, LENGTH - int(args.answer_tokens) - 1 :
        ] = 1
        labels[row_index, masks[row_index] == 1] = inputs[
            row_index, masks[row_index] == 1
        ]
        metadata.append(
            {
                "row": row_index,
                "source_rows": [int(value) for value in source_rows],
                "source_window_start": int(start),
                "anchor_position_in_passage": int(local),
                "position_bin": int(target_bin),
                "query_start": int(query_start),
                "anchor_sha256": hashlib.sha256(
                    anchor.astype(np.int32).tobytes()
                ).hexdigest(),
                "answer_sha256": hashlib.sha256(
                    answer.astype(np.int32).tobytes()
                ).hexdigest(),
            }
        )

    split = np.zeros(int(args.rows), dtype=np.uint8)
    validation = rng.permutation(int(args.rows))[: int(args.validation_rows)]
    split[validation] = 1
    output.mkdir(parents=True)
    np.save(output / "input_ids.npy", inputs, allow_pickle=False)
    np.save(output / "assistant_mask.npy", masks, allow_pickle=False)
    np.save(output / "labels.npy", labels, allow_pickle=False)
    np.save(output / "query_starts.npy", query_starts, allow_pickle=False)
    np.save(output / "active_lengths.npy", active_lengths, allow_pickle=False)
    np.save(output / "lengths.npy", lengths, allow_pickle=False)
    np.save(output / "split.npy", split, allow_pickle=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    files = {}
    for name in (
        "input_ids.npy",
        "assistant_mask.npy",
        "labels.npy",
        "query_starts.npy",
        "active_lengths.npy",
        "lengths.npy",
        "split.npy",
        "rows.jsonl",
    ):
        path = output / name
        files[name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    manifest = {
        "status": DATA_STATUS,
        "format_version": 1,
        "view": "continuous_natural_span_retrieval_L8192",
        "shape": [int(args.rows), LENGTH],
        "physical_storage_length": LENGTH,
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "position_ids": "contiguous_zero_based",
        "labels_only_cover_answer_and_final_eos": True,
        "pad_token_id": int(tokenizer.pad_token_id),
        "training_rows": int(args.rows) - int(args.validation_rows),
        "validation_rows": int(args.validation_rows),
        "supervision": {
            "objective": "copy_natural_span_after_unique_anchor",
            "assistant_tokens_per_row": int(args.answer_tokens) + 1,
            "anchor_tokens": int(args.anchor_tokens),
            "answer_tokens": int(args.answer_tokens),
            "anchor_support": "first_physical_4k",
            "position_bins": int(args.position_bins),
        },
        "source": {
            "kind": "paired_pretokenized_fineweb_edu_4k_tensor",
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "tokenizer": tokenizer_digest(checkpoint),
        "ruler_or_niah_rows": 0,
        "ruler_generator_used": False,
        "training_family": "natural_document_unique_anchor_copy",
        "evaluation_family_overlap": "none_by_generator_or_rows",
        "seed": int(args.seed),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
