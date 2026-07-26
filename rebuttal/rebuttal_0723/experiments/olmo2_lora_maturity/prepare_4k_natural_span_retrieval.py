#!/usr/bin/env python3
"""Prepare a 4K natural-span retrieval curriculum independent of RULER."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)


LENGTH = 4_096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_920)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--anchor-tokens", type=int, default=8)
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--position-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_730)
    return parser.parse_args()


def tokenizer_digest(checkpoint: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(files[name]["sha256"]))
    return {"files": files, "composite_sha256": digest.hexdigest()}


def occurrences(sequence: np.ndarray, needle: np.ndarray) -> int:
    if len(needle) == 0 or len(needle) > len(sequence):
        return 0
    first = np.flatnonzero(sequence[: len(sequence) - len(needle) + 1] == needle[0])
    return sum(
        bool(np.array_equal(sequence[index : index + len(needle)], needle))
        for index in first
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = args.source_view.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if int(args.rows) <= int(args.validation_rows):
        raise ValueError("rows must exceed validation rows")
    if int(args.anchor_tokens) < 4 or int(args.answer_tokens) < 1:
        raise ValueError("invalid anchor/answer width")
    if int(args.position_bins) < 2:
        raise ValueError("position bins must be at least two")

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

    source_ids = np.load(
        source / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    source_lengths = np.load(
        source / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    source_split = np.load(
        source / "split.npy", mmap_mode="r", allow_pickle=False
    )
    source_manifest = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    if source_ids.ndim != 2 or source_ids.shape[1] < 8_192:
        raise RuntimeError("source view is not a long fixed-token view")
    candidates = np.flatnonzero(source_split == 0)
    if len(candidates) < int(args.rows):
        raise RuntimeError(
            f"requested {args.rows} rows from {len(candidates)} source rows"
        )

    prefix = tokenizer.encode(
        (
            f"{tokenizer.bos_token}<|user|>\n"
            "Read the passage. Locate the quoted anchor and copy exactly "
            "the text immediately following it.\n\nPassage:\n"
        ),
        add_special_tokens=False,
    )
    query_prefix = tokenizer.encode(
        "\n\nAnchor:\n", add_special_tokens=False
    )
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
    if passage_tokens < 3_000:
        raise RuntimeError("retrieval prompt leaves too little passage")

    rng = np.random.default_rng(int(args.seed))
    chosen = rng.permutation(candidates)[: int(args.rows)]
    inputs = np.full(
        (int(args.rows), LENGTH),
        int(tokenizer.pad_token_id),
        dtype=np.int32,
    )
    masks = np.zeros((int(args.rows), LENGTH), dtype=np.uint8)
    lengths = np.full(int(args.rows), LENGTH, dtype=np.int32)
    metadata = []
    forbidden = {
        int(tokenizer.bos_token_id),
        int(tokenizer.eos_token_id),
        int(tokenizer.pad_token_id),
    }

    for row_index, source_row in enumerate(chosen):
        available = int(source_lengths[int(source_row)])
        if available < passage_tokens + 64:
            raise RuntimeError(f"source row too short: {source_row}")
        built = None
        target_bin = row_index % int(args.position_bins)
        for attempt in range(128):
            start = int(
                rng.integers(0, available - passage_tokens + 1)
            )
            passage = np.asarray(
                source_ids[
                    int(source_row), start : start + passage_tokens
                ],
                dtype=np.int64,
            )
            usable = (
                passage_tokens
                - int(args.anchor_tokens)
                - int(args.answer_tokens)
                - 32
            )
            local = 16 + int(
                round(
                    usable
                    * target_bin
                    / max(int(args.position_bins) - 1, 1)
                )
            )
            local += int(rng.integers(-8, 9))
            local = max(8, min(local, usable + 16))
            anchor = passage[
                local : local + int(args.anchor_tokens)
            ]
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
                target_bin = int(
                    rng.integers(0, int(args.position_bins))
                )
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
                raise RuntimeError("retrieval sequence length drift")
            built = (sequence, start, local, anchor, answer)
            break
        if built is None:
            raise RuntimeError(
                f"failed to construct unique natural anchor: {source_row}"
            )
        sequence, start, local, anchor, answer = built
        inputs[row_index] = sequence.astype(np.int32)
        masks[
            row_index,
            LENGTH - int(args.answer_tokens) - 1 :,
        ] = 1
        metadata.append(
            {
                "row": row_index,
                "source_row": int(source_row),
                "source_window_start": int(start),
                "anchor_position_in_passage": int(local),
                "position_bin": int(target_bin),
                "anchor_sha256": hashlib.sha256(
                    anchor.astype(np.int32).tobytes()
                ).hexdigest(),
                "answer_sha256": hashlib.sha256(
                    answer.astype(np.int32).tobytes()
                ).hexdigest(),
            }
        )

    split = np.zeros(int(args.rows), dtype=np.uint8)
    validation = rng.permutation(int(args.rows))[
        : int(args.validation_rows)
    ]
    split[validation] = 1
    output.mkdir(parents=True)
    np.save(output / "input_ids.npy", inputs, allow_pickle=False)
    np.save(output / "assistant_mask.npy", masks, allow_pickle=False)
    np.save(output / "lengths.npy", lengths, allow_pickle=False)
    np.save(output / "split.npy", split, allow_pickle=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    files = {}
    for name in (
        "input_ids.npy",
        "assistant_mask.npy",
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
        "format_version": 1,
        "view": "natural_span_retrieval_L4096",
        "shape": [int(args.rows), LENGTH],
        "pad_token_id": int(tokenizer.pad_token_id),
        "training_rows": int(args.rows) - int(args.validation_rows),
        "validation_rows": int(args.validation_rows),
        "length_statistics": {
            "minimum": LENGTH,
            "median": float(LENGTH),
            "maximum": LENGTH,
        },
        "supervision": {
            "objective": "copy_natural_span_after_unique_anchor",
            "assistant_tokens_per_row": int(args.answer_tokens) + 1,
            "anchor_tokens": int(args.anchor_tokens),
            "answer_tokens": int(args.answer_tokens),
            "position_bins": int(args.position_bins),
        },
        "source": {
            "kind": "frozen_longalign_train_split_token_view",
            "path": str(source),
            "manifest_sha256": sha256_file(source / "manifest.json"),
            "input_ids_sha256": sha256_file(source / "input_ids.npy"),
            "source_manifest": source_manifest.get("source"),
        },
        "tokenizer": tokenizer_digest(checkpoint),
        "ruler_or_niah_rows": 0,
        "ruler_generator_used": False,
        "seed": int(args.seed),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
