#!/usr/bin/env python3
"""Audit exact downstream prompt overlap in general-data Q/K training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .train_screen import FixedView, load_fixed_view


STATUS = "OLMO2_GENERAL_QK_DOWNSTREAM_OVERLAP_AUDIT_V1"
ROW_SEPARATOR = np.asarray([0xFFFFFFFF], dtype="<u4").tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--two-wiki-data", type=Path, required=True)
    parser.add_argument("--ruler-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty JSONL: {path}")
    return rows


def training_blob(view: FixedView) -> tuple[bytes, dict[str, Any]]:
    pieces: list[bytes] = []
    active_tokens = 0
    for index in view.training_rows:
        length = int(view.lengths[int(index)])
        tokens = np.asarray(
            view.input_ids[int(index), :length], dtype="<u4"
        )
        pieces.append(tokens.tobytes(order="C"))
        pieces.append(ROW_SEPARATOR)
        active_tokens += length
    return b"".join(pieces), {
        "path": str(view.path),
        "manifest_sha256": sha256_file(view.path / "manifest.json"),
        "training_rows": int(len(view.training_rows)),
        "active_tokens": int(active_tokens),
    }


def token_bytes(token_ids: Iterable[int]) -> bytes:
    values = np.asarray(list(token_ids), dtype="<u4")
    if values.size == 0:
        raise RuntimeError("overlap needle has no tokens")
    if np.any(values == 0xFFFFFFFF):
        raise RuntimeError("overlap needle collides with row separator")
    return values.tobytes(order="C")


def aligned_contains(haystack: bytes, needle: bytes) -> bool:
    offset = haystack.find(needle)
    while offset >= 0:
        if offset % 4 == 0:
            return True
        offset = haystack.find(needle, offset + 1)
    return False


def appears_in_training(
    *,
    blobs: dict[str, bytes],
    variants: list[list[int]],
) -> list[str]:
    matched: list[str] = []
    needles = [token_bytes(value) for value in variants]
    for source, blob in blobs.items():
        if any(aligned_contains(blob, needle) for needle in needles):
            matched.append(source)
    return matched


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    checkpoint = args.checkpoint.resolve()
    prepared = args.prepared_data.resolve()
    two_wiki_root = args.two_wiki_data.resolve()
    ruler_root = args.ruler_data.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )

    views = {
        "longalign": load_fixed_view(
            prepared / "longalign_paired_L4096"
        ),
        "tulu": load_fixed_view(prepared / "tulu3_replay_L4096"),
    }
    blobs: dict[str, bytes] = {}
    view_receipts: dict[str, Any] = {}
    for name, view in views.items():
        blob, receipt = training_blob(view)
        blobs[name] = blob
        view_receipts[name] = receipt

    two_wiki_path = two_wiki_root / "evaluation_rows.jsonl"
    two_wiki_rows = load_jsonl(two_wiki_path)
    two_wiki_matches: list[dict[str, Any]] = []
    for row in two_wiki_rows:
        question = str(row["question"])
        sources = appears_in_training(
            blobs=blobs,
            variants=[
                list(
                    tokenizer(
                        question, add_special_tokens=False
                    ).input_ids
                )
            ],
        )
        if sources:
            two_wiki_matches.append(
                {
                    "index": int(row["index"]),
                    "source_id": str(row["source_id"]),
                    "matched_training_sources": sources,
                }
            )

    ruler_paths = sorted((ruler_root / "cells").rglob("test.jsonl"))
    if not ruler_paths:
        raise RuntimeError("RULER audit found no test cells")
    ruler_rows = 0
    ruler_matches: list[dict[str, Any]] = []
    for path in ruler_paths:
        task = path.parent.parent.name
        nominal_length = int(path.parent.name.removeprefix("L"))
        for local_index, row in enumerate(load_jsonl(path)):
            ruler_rows += 1
            prompt = str(row["input"])
            raw_ids = list(
                tokenizer(
                    prompt, add_special_tokens=False
                ).input_ids
            )
            chat_ids = list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            sources = appears_in_training(
                blobs=blobs,
                variants=[raw_ids, chat_ids],
            )
            if sources:
                ruler_matches.append(
                    {
                        "task": task,
                        "nominal_length": nominal_length,
                        "local_index": local_index,
                        "matched_training_sources": sources,
                    }
                )

    result = {
        "status": STATUS,
        "boundary": (
            "This exact-token audit rules out verbatim 2Wiki test-question "
            "and full RULER test-prompt occurrences in the prepared general "
            "training views. It does not establish document- or topic-level "
            "non-overlap with the upstream source corpora."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "training_views": view_receipts,
        "two_wiki": {
            "rows": len(two_wiki_rows),
            "evaluation_rows_sha256": sha256_file(two_wiki_path),
            "exact_question_matches": len(two_wiki_matches),
            "matches": two_wiki_matches,
        },
        "ruler": {
            "rows": ruler_rows,
            "manifest_sha256": sha256_file(
                ruler_root / "manifest.json"
            ),
            "exact_raw_or_chat_prompt_matches": len(ruler_matches),
            "matches": ruler_matches,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
