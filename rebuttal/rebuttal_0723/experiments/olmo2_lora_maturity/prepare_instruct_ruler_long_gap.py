#!/usr/bin/env python3
"""Prepare an official-RULER 8K set entirely beyond the 4K training gap."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .freeze_conversion_evidence import (
    QUERY_PATTERNS,
    SOURCE_PAIR,
    extract_key_values,
)
from .prepare_data import atomic_json, sha256_file


RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
STATUS = "OLMO2_INSTRUCT_RULER_LONG_GAP_SCREEN_PREPARED"
TASK = "niah_single_1"
LENGTH = 8_192
TOKENS_TO_GENERATE = 128
TASK_CONFIG = {
    "type_haystack": "noise",
    "type_needle_k": "words",
    "type_needle_v": "numbers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument(
        "--forbidden-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def parse_row_identities(row: dict[str, Any]) -> dict[str, set[str]]:
    source_pairs = SOURCE_PAIR.findall(str(row["input"]))
    if not source_pairs:
        raise RuntimeError("candidate row has no source pair")
    query = None
    for pattern in QUERY_PATTERNS:
        match = pattern.search(str(row["input"]))
        if match is not None:
            query = match.group(1).strip()
            break
    if query is None:
        raise RuntimeError("candidate row has no parseable query")
    outputs = {str(value) for value in row["outputs"]}
    return {
        "source_keys": {key.strip() for key, _ in source_pairs},
        "source_values": {str(value) for _, value in source_pairs},
        "queries": {query},
        "answers": outputs,
    }


def training_gap_support(path: Path) -> dict[str, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("routing training rows are empty")
    gaps = [
        int(row["answer_start"])
        - int(row["source_token_position_answer"])
        for row in rows
    ]
    return {
        "rows": len(rows),
        "minimum_tokens": min(gaps),
        "maximum_tokens": max(gaps),
    }


def merge_forbidden(paths: list[Path]) -> tuple[
    dict[str, set[str]],
    list[dict[str, Any]],
]:
    merged = {
        "source_keys": set(),
        "source_values": set(),
        "queries": set(),
        "answers": set(),
    }
    receipts: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        values = extract_key_values(resolved)
        for field in merged:
            merged[field].update(values[field])
        receipts.append(
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "rows": int(values["rows"]),
            }
        )
    return merged, receipts


def main() -> None:
    args = parse_args()
    root = args.ruler_root.resolve()
    checkpoint = args.checkpoint.resolve()
    training_rows = args.training_rows.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if int(args.samples) <= 0:
        raise ValueError("samples must be positive")
    if int(args.candidate_count) < int(args.samples):
        raise ValueError("candidate-count must be at least samples")
    output.mkdir(parents=True)

    actual_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != RULER_COMMIT:
        raise RuntimeError(
            f"RULER commit drift: {actual_commit} != {RULER_COMMIT}"
        )

    support = training_gap_support(training_rows)
    if support != {
        "rows": 1024,
        "minimum_tokens": 62,
        "maximum_tokens": 3933,
    }:
        raise RuntimeError(f"routing training-gap support drift: {support}")
    threshold = int(support["maximum_tokens"])
    forbidden, forbidden_receipts = merge_forbidden(
        [path.resolve() for path in args.forbidden_jsonl]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    empty_raw = len(
        tokenizer("", add_special_tokens=False).input_ids
    )
    empty_chat = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
        )
    )
    chat_overhead = empty_chat - empty_raw
    if not 8 <= chat_overhead <= 16:
        raise RuntimeError(
            f"unexpected OLMo chat overhead: {chat_overhead}"
        )

    synthetic = root / "scripts" / "data" / "synthetic"
    sys.path.insert(0, str(synthetic))
    from constants import TASKS  # type: ignore[import-not-found]

    template = (
        TASKS["niah"]["template"]
        + TASKS["niah"]["answer_prefix"]
    )
    candidate_root = output / "candidates"
    command = [
        sys.executable,
        str(synthetic / "niah.py"),
        "--save_dir",
        str(candidate_root),
        "--save_name",
        TASK,
        "--subset",
        "test",
        "--tokenizer_path",
        str(checkpoint),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(LENGTH - chat_overhead),
        "--tokens_to_generate",
        str(TOKENS_TO_GENERATE),
        "--num_samples",
        str(int(args.candidate_count)),
        "--random_seed",
        str(int(args.seed)),
        "--template",
        template,
        "--num_needle_k",
        "1",
        "--num_needle_v",
        "1",
        "--num_needle_q",
        "1",
        "--type_haystack",
        TASK_CONFIG["type_haystack"],
        "--type_needle_k",
        TASK_CONFIG["type_needle_k"],
        "--type_needle_v",
        TASK_CONFIG["type_needle_v"],
    ]
    subprocess.run(command, check=True)
    candidate_path = candidate_root / TASK / "test.jsonl"
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(candidates) != int(args.candidate_count):
        raise RuntimeError("candidate row-count drift")

    selected: list[dict[str, Any]] = []
    selected_identities = {
        "source_keys": set(),
        "source_values": set(),
        "queries": set(),
        "answers": set(),
    }
    rejection_counts = {
        "not_beyond_training_gap": 0,
        "forbidden_overlap": 0,
        "selected_identity_duplicate": 0,
        "prompt_too_long": 0,
    }
    selected_gaps: list[int] = []
    selected_input_tokens: list[int] = []
    for row in candidates:
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["input"]}],
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            row.get("answer_prefix", ""),
            add_special_tokens=False,
        ).input_ids
        input_tokens = len(chat_ids) + len(prefix_ids)
        if input_tokens + TOKENS_TO_GENERATE > LENGTH:
            rejection_counts["prompt_too_long"] += 1
            continue
        gap = input_tokens - int(row["token_position_answer"])
        if gap <= threshold:
            rejection_counts["not_beyond_training_gap"] += 1
            continue
        identities = parse_row_identities(row)
        if any(
            identities[field] & forbidden[field]
            for field in forbidden
        ):
            rejection_counts["forbidden_overlap"] += 1
            continue
        if any(
            identities[field] & selected_identities[field]
            for field in selected_identities
        ):
            rejection_counts["selected_identity_duplicate"] += 1
            continue
        selected.append(row)
        selected_gaps.append(gap)
        selected_input_tokens.append(input_tokens)
        for field in selected_identities:
            selected_identities[field].update(identities[field])
        if len(selected) == int(args.samples):
            break
    if len(selected) != int(args.samples):
        raise RuntimeError(
            f"only selected {len(selected)} of {int(args.samples)} rows; "
            f"rejections={rejection_counts}"
        )

    final_path = output / f"L{LENGTH}" / TASK / "test.jsonl"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    final_values = extract_key_values(final_path)
    overlap_receipt = {
        field: {
            "count": len(set(final_values[field]) & forbidden[field]),
            "values": sorted(
                set(final_values[field]) & forbidden[field]
            ),
        }
        for field in forbidden
    }
    if any(item["count"] for item in overlap_receipt.values()):
        raise RuntimeError("selected rows overlap forbidden identities")
    if min(selected_gaps) <= threshold:
        raise RuntimeError("selected set contains an in-support gap")

    manifest = {
        "format_version": 1,
        "status": STATUS,
        "task": TASK,
        "task_config": TASK_CONFIG,
        "ruler_commit": actual_commit,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "chat_overhead_tokens": chat_overhead,
        "lengths": [LENGTH],
        "samples_per_length": int(args.samples),
        "seed": int(args.seed),
        "candidate_count": int(args.candidate_count),
        "candidate_file": {
            "relative_path": str(candidate_path.relative_to(output)),
            "sha256": sha256_file(candidate_path),
            "rows": len(candidates),
        },
        "training_gap_support": {
            **support,
            "rows_sha256": sha256_file(training_rows),
        },
        "selection_rule": (
            "input_tokens - token_position_answer > "
            "maximum frozen routing-training gap"
        ),
        "rejection_counts_before_completion": rejection_counts,
        "forbidden_sources": forbidden_receipts,
        "selected_overlap_with_forbidden": overlap_receipt,
        "files": {
            str(LENGTH): {
                "relative_path": str(final_path.relative_to(output)),
                "sha256": sha256_file(final_path),
                "rows": len(selected),
                "minimum_prompt_tokens": min(selected_input_tokens),
                "maximum_prompt_tokens": max(selected_input_tokens),
                "minimum_generation_boundary_gap_tokens": min(
                    selected_gaps
                ),
                "maximum_generation_boundary_gap_tokens": max(
                    selected_gaps
                ),
                "generation_tokens": TOKENS_TO_GENERATE,
            }
        },
    }
    atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output),
                "selected": len(selected),
                "minimum_gap": min(selected_gaps),
                "maximum_gap": max(selected_gaps),
                "rejections": rejection_counts,
                "data_sha256": sha256_file(final_path),
                "manifest_sha256": sha256_file(
                    output / "manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
