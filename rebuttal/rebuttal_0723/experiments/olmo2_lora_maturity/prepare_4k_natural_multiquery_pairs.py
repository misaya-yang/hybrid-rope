#!/usr/bin/env python3
"""Prepare paired 4K natural multi-query counterfactual data.

The generator uses only a frozen LongAlign token view.  Each pair keeps the
query block and label geometry fixed while deranging the natural eight-token
answer spans that immediately follow sixteen unique anchors in the passage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)


LENGTH = 4_096
QUERIES = 16
ANCHOR_TOKENS = 8
ANSWER_TOKENS = 8
SUPERVISED_TOKENS_PER_VARIANT = QUERIES * ANSWER_TOKENS + 1
MINIMUM_SOURCE_UNIQUE_TOKENS = 300
MAX_SOURCE_WINDOW_ATTEMPTS = 16
SET_STATUS = "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_SET_PREPARED"
ROOT_STATUS = "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=1_024)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_731)
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


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def occurrences(sequence: np.ndarray, needle: np.ndarray) -> int:
    if len(needle) == 0 or len(needle) > len(sequence):
        return 0
    starts = np.flatnonzero(
        sequence[: len(sequence) - len(needle) + 1] == needle[0]
    )
    return sum(
        bool(np.array_equal(sequence[start : start + len(needle)], needle))
        for start in starts
    )


def encoded(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False)
    ]


def prompt_parts(tokenizer: Any) -> dict[str, Any]:
    prefix = encoded(
        tokenizer,
        (
            f"{tokenizer.bos_token}<|user|>\n"
            "Read the natural passage. Sixteen quoted anchors occur exactly "
            "once each. For every anchor, copy exactly the eight tokens that "
            "immediately follow it.\n\nPassage:\n"
        ),
    )
    query_intro = encoded(
        tokenizer,
        (
            "\n\nQueries:\n"
            "Return the eight-token continuation for each anchor in order."
        ),
    )
    query_markers = [
        encoded(tokenizer, f"\nAnchor {index + 1:02d}: ")
        for index in range(QUERIES)
    ]
    assistant_prefix = encoded(tokenizer, "\n<|assistant|>\n")
    answer_markers = [
        encoded(tokenizer, f"Answer {index + 1:02d}: ")
        for index in range(QUERIES)
    ]
    answer_suffix = encoded(tokenizer, "\n")
    overhead = (
        len(prefix)
        + len(query_intro)
        + sum(len(marker) + ANCHOR_TOKENS for marker in query_markers)
        + len(assistant_prefix)
        + sum(
            len(marker) + ANSWER_TOKENS + len(answer_suffix)
            for marker in answer_markers
        )
        + 1
    )
    passage_tokens = LENGTH - overhead
    if passage_tokens < 3_000:
        raise RuntimeError(
            f"multi-query prompt leaves only {passage_tokens} passage tokens"
        )
    return {
        "prefix": prefix,
        "query_intro": query_intro,
        "query_markers": query_markers,
        "assistant_prefix": assistant_prefix,
        "answer_markers": answer_markers,
        "answer_suffix": answer_suffix,
        "overhead_tokens": overhead,
        "passage_tokens": passage_tokens,
    }


def select_anchor_answer_spans(
    *,
    passage: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[int],
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]] | None:
    positions: list[int] = []
    anchors: list[np.ndarray] = []
    answers: list[np.ndarray] = []
    anchor_digests: set[bytes] = set()
    answer_digests: set[bytes] = set()
    passage_tokens = len(passage)
    for bin_index in range(QUERIES):
        left = (bin_index * passage_tokens) // QUERIES
        right = ((bin_index + 1) * passage_tokens) // QUERIES
        minimum = left + 4
        maximum = right - ANCHOR_TOKENS - ANSWER_TOKENS - 4
        if maximum < minimum:
            return None
        candidates = rng.permutation(
            np.arange(minimum, maximum + 1, dtype=np.int64)
        )
        selected = None
        for raw_position in candidates:
            position = int(raw_position)
            anchor = np.asarray(
                passage[position : position + ANCHOR_TOKENS],
                dtype=np.int64,
            )
            answer = np.asarray(
                passage[
                    position
                    + ANCHOR_TOKENS : position
                    + ANCHOR_TOKENS
                    + ANSWER_TOKENS
                ],
                dtype=np.int64,
            )
            if any(
                int(token) in forbidden
                for token in np.concatenate((anchor, answer))
            ):
                continue
            anchor_digest = anchor.tobytes()
            answer_digest = answer.tobytes()
            if (
                anchor_digest in anchor_digests
                or answer_digest in answer_digests
                or occurrences(passage, anchor) != 1
            ):
                continue
            selected = (position, anchor, answer)
            break
        if selected is None:
            return None
        position, anchor, answer = selected
        positions.append(position)
        anchors.append(anchor)
        answers.append(answer)
        anchor_digests.add(anchor.tobytes())
        answer_digests.add(answer.tobytes())
    return positions, anchors, answers


def deterministic_derangement(
    *,
    seed: int,
    source_row: int,
    row_index: int,
) -> tuple[int, list[int]]:
    shift = 1 + (
        (int(seed) + int(source_row) + int(row_index)) % (QUERIES - 1)
    )
    mapping = [
        int((query_index + shift) % QUERIES)
        for query_index in range(QUERIES)
    ]
    if any(index == source for index, source in enumerate(mapping)):
        raise RuntimeError("answer-span permutation has a fixed point")
    return shift, mapping


def swap_passage_answers(
    *,
    passage: np.ndarray,
    positions: Sequence[int],
    answers: Sequence[np.ndarray],
    mapping: Sequence[int],
) -> np.ndarray:
    swapped = passage.copy()
    mutable = np.zeros(len(passage), dtype=bool)
    for query_index, position in enumerate(positions):
        start = int(position) + ANCHOR_TOKENS
        stop = start + ANSWER_TOKENS
        swapped[start:stop] = answers[int(mapping[query_index])]
        mutable[start:stop] = True
    if not np.array_equal(passage[~mutable], swapped[~mutable]):
        raise RuntimeError("counterfactual changed tokens outside answer spans")
    if np.array_equal(passage, swapped):
        raise RuntimeError("counterfactual passage did not change")
    return swapped


def build_sequence(
    *,
    parts: dict[str, Any],
    passage: np.ndarray,
    anchors: Sequence[np.ndarray],
    answers: Sequence[np.ndarray],
    eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    tokens: list[int] = list(parts["prefix"])
    tokens.extend(int(value) for value in passage)
    tokens.extend(parts["query_intro"])
    for marker, anchor in zip(parts["query_markers"], anchors):
        tokens.extend(marker)
        tokens.extend(int(value) for value in anchor)
    tokens.extend(parts["assistant_prefix"])
    labels = [-100] * len(tokens)
    answer_starts: list[int] = []
    for marker, answer in zip(parts["answer_markers"], answers):
        tokens.extend(marker)
        labels.extend([-100] * len(marker))
        answer_starts.append(len(tokens))
        tokens.extend(int(value) for value in answer)
        labels.extend(int(value) for value in answer)
        tokens.extend(parts["answer_suffix"])
        labels.extend([-100] * len(parts["answer_suffix"]))
    tokens.append(int(eos_token_id))
    labels.append(int(eos_token_id))
    if len(tokens) != LENGTH or len(labels) != LENGTH:
        raise RuntimeError(
            f"multi-query sequence length drift: {len(tokens)}"
        )
    if sum(value != -100 for value in labels) != (
        SUPERVISED_TOKENS_PER_VARIANT
    ):
        raise RuntimeError("multi-query supervised-token count drift")
    return (
        np.asarray(tokens, dtype=np.uint32),
        np.asarray(labels, dtype=np.int32),
        answer_starts,
    )


def source_receipt(source: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name in (
        "manifest.json",
        "input_ids.npy",
        "lengths.npy",
        "split.npy",
    ):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = file_entry(path)
    return {
        "kind": "frozen_longalign_train_split_token_view",
        "path": str(source),
        "files": files,
        "upstream_source": manifest.get("source"),
    }


def build_set(
    *,
    output: Path,
    purpose: str,
    target_rows: int,
    source_rows: Sequence[int],
    source_ids: np.ndarray,
    source_lengths: np.ndarray,
    tokenizer: Any,
    tokenizer_info: dict[str, Any],
    source_info: dict[str, Any],
    parts: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[int]]:
    output.mkdir(parents=True, exist_ok=False)
    rows = int(target_rows)
    input_ids = np.lib.format.open_memmap(
        output / "input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(rows, 2, LENGTH),
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy",
        mode="w+",
        dtype=np.int32,
        shape=(rows, 2, LENGTH),
    )
    lengths = np.lib.format.open_memmap(
        output / "lengths.npy",
        mode="w+",
        dtype=np.int32,
        shape=(rows, 2),
    )
    input_ids[:] = int(tokenizer.pad_token_id)
    labels[:] = -100
    lengths[:] = LENGTH
    forbidden = {
        int(tokenizer.bos_token_id),
        int(tokenizer.eos_token_id),
        int(tokenizer.pad_token_id),
    }
    rng = np.random.default_rng(int(seed))
    passage_tokens = int(parts["passage_tokens"])
    metadata_path = output / "rows.jsonl"
    used_source_rows: list[int] = []
    skipped_source_rows: list[int] = []
    with metadata_path.open("w", encoding="utf-8") as handle:
        for raw_source_row in source_rows:
            if len(used_source_rows) == rows:
                break
            row_index = len(used_source_rows)
            source_row = int(raw_source_row)
            available = int(source_lengths[source_row])
            if available < passage_tokens:
                raise RuntimeError(f"source row too short: {source_row}")
            built = None
            for _ in range(MAX_SOURCE_WINDOW_ATTEMPTS):
                start = int(
                    rng.integers(0, available - passage_tokens + 1)
                )
                passage = np.asarray(
                    source_ids[
                        source_row, start : start + passage_tokens
                    ],
                    dtype=np.int64,
                )
                selected = select_anchor_answer_spans(
                    passage=passage,
                    rng=rng,
                    forbidden=forbidden,
                )
                if selected is None:
                    continue
                positions, anchors, answers = selected
                shift, mapping = deterministic_derangement(
                    seed=int(seed),
                    source_row=source_row,
                    row_index=row_index,
                )
                swapped_passage = swap_passage_answers(
                    passage=passage,
                    positions=positions,
                    answers=answers,
                    mapping=mapping,
                )
                if any(
                    occurrences(swapped_passage, anchor) != 1
                    for anchor in anchors
                ):
                    continue
                swapped_answers = [
                    answers[int(mapping[index])] for index in range(QUERIES)
                ]
                original_sequence, original_labels, original_starts = (
                    build_sequence(
                        parts=parts,
                        passage=passage,
                        anchors=anchors,
                        answers=answers,
                        eos_token_id=int(tokenizer.eos_token_id),
                    )
                )
                swapped_sequence, swapped_labels, swapped_starts = (
                    build_sequence(
                        parts=parts,
                        passage=swapped_passage,
                        anchors=anchors,
                        answers=swapped_answers,
                        eos_token_id=int(tokenizer.eos_token_id),
                    )
                )
                if original_starts != swapped_starts:
                    raise RuntimeError("paired answer-label geometry drift")
                if not np.array_equal(
                    original_labels != -100, swapped_labels != -100
                ):
                    raise RuntimeError("paired supervision-mask drift")
                built = (
                    start,
                    positions,
                    anchors,
                    answers,
                    shift,
                    mapping,
                    original_sequence,
                    original_labels,
                    swapped_sequence,
                    swapped_labels,
                    original_starts,
                )
                break
            if built is None:
                skipped_source_rows.append(source_row)
                continue
            (
                start,
                positions,
                anchors,
                answers,
                shift,
                mapping,
                original_sequence,
                original_labels,
                swapped_sequence,
                swapped_labels,
                answer_starts,
            ) = built
            input_ids[row_index, 0] = original_sequence
            input_ids[row_index, 1] = swapped_sequence
            labels[row_index, 0] = original_labels
            labels[row_index, 1] = swapped_labels
            handle.write(
                json.dumps(
                    {
                        "row": int(row_index),
                        "purpose": purpose,
                        "source_row": source_row,
                        "source_window_start": int(start),
                        "source_window_tokens": passage_tokens,
                        "position_bins": list(range(QUERIES)),
                        "anchor_positions_in_passage": [
                            int(value) for value in positions
                        ],
                        "answer_label_starts": [
                            int(value) for value in answer_starts
                        ],
                        "variant1_derangement_shift": int(shift),
                        "variant1_answer_source_indices": [
                            int(value) for value in mapping
                        ],
                        "anchor_sha256": [
                            hashlib.sha256(
                                anchor.astype(np.int32).tobytes()
                            ).hexdigest()
                            for anchor in anchors
                        ],
                        "original_answer_sha256": [
                            hashlib.sha256(
                                answer.astype(np.int32).tobytes()
                            ).hexdigest()
                            for answer in answers
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            used_source_rows.append(source_row)
    if len(used_source_rows) != rows:
        raise RuntimeError(
            f"built only {len(used_source_rows)} of {rows} requested rows "
            f"after testing {len(source_rows)} disjoint source candidates"
        )
    input_ids.flush()
    labels.flush()
    lengths.flush()
    expected_labels = rows * 2 * SUPERVISED_TOKENS_PER_VARIANT
    actual_labels = int(np.count_nonzero(labels != -100))
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"set label count drift: {actual_labels} != {expected_labels}"
        )
    output_files = {
        name: file_entry(output / name)
        for name in (
            "input_ids.npy",
            "labels.npy",
            "lengths.npy",
            "rows.jsonl",
        )
    }
    manifest = {
        "format_version": 1,
        "status": SET_STATUS,
        "purpose": purpose,
        "shape": [rows, 2, LENGTH],
        "variant_names": [
            "original_natural_passage",
            "answer_spans_deranged",
        ],
        "maximum_training_length": LENGTH,
        "maximum_training_position_id": LENGTH - 1,
        "queries_per_sequence": QUERIES,
        "source_position_bins": QUERIES,
        "anchors_per_sequence": QUERIES,
        "anchor_tokens": ANCHOR_TOKENS,
        "answer_tokens_per_query": ANSWER_TOKENS,
        "answer_tokens_per_variant": QUERIES * ANSWER_TOKENS,
        "final_eos_supervised": True,
        "supervised_tokens_per_variant": SUPERVISED_TOKENS_PER_VARIANT,
        "labels_only_cover_answers_and_final_eos": True,
        "paired_answer_span_derangement": True,
        "query_anchors_identical_across_variants": True,
        "answer_label_geometry_identical_across_variants": True,
        "every_variant_exactly_4096_tokens": True,
        "benchmark_independent_generator": True,
        "ruler_generator_used": False,
        "ruler_import_used": False,
        "ruler_rows": 0,
        "ruler_or_niah_rows": 0,
        "source": source_info,
        "tokenizer": tokenizer_info,
        "seed": int(seed),
        "source_rows_sha256": hashlib.sha256(
            np.asarray(used_source_rows, dtype=np.int64).tobytes()
        ).hexdigest(),
        "source_rows_used": len(used_source_rows),
        "source_rows_skipped": len(skipped_source_rows),
        "minimum_source_unique_tokens": MINIMUM_SOURCE_UNIQUE_TOKENS,
        "maximum_source_window_attempts": MAX_SOURCE_WINDOW_ATTEMPTS,
        "files": {
            name: entry["sha256"] for name, entry in output_files.items()
        },
        "file_receipts": output_files,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest, used_source_rows


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = args.source_view.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if int(args.train_rows) <= 0 or int(args.calibration_rows) <= 0:
        raise ValueError("train and calibration row counts must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if (
        int(tokenizer.bos_token_id) != 100_257
        or int(tokenizer.eos_token_id) != 100_257
        or int(tokenizer.pad_token_id) != 100_277
    ):
        raise RuntimeError("OLMo tokenizer special-token contract drift")
    tokenizer_info = tokenizer_digest(checkpoint)
    parts = prompt_parts(tokenizer)

    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_ids = np.load(
        source / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    source_lengths = np.load(
        source / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    source_split = np.load(
        source / "split.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        source_ids.ndim != 2
        or source_lengths.shape != (source_ids.shape[0],)
        or source_split.shape != (source_ids.shape[0],)
    ):
        raise RuntimeError("frozen LongAlign source-view shape drift")
    if source_ids.shape[1] < int(parts["passage_tokens"]):
        raise RuntimeError("source view is too short for the natural passage")
    length_eligible = np.flatnonzero(
        (source_split == 0)
        & (source_lengths >= int(parts["passage_tokens"]))
    )
    eligible = np.asarray(
        [
            int(source_row)
            for source_row in length_eligible
            if np.unique(
                source_ids[
                    int(source_row), : int(source_lengths[int(source_row)])
                ]
            ).size
            >= MINIMUM_SOURCE_UNIQUE_TOKENS
        ],
        dtype=np.int64,
    )
    requested = int(args.train_rows) + int(args.calibration_rows)
    if len(eligible) < requested:
        raise RuntimeError(
            f"requested {requested} disjoint source rows from "
            f"{len(eligible)} eligible LongAlign train rows"
        )
    chooser = np.random.default_rng(int(args.seed))
    chosen = chooser.permutation(eligible)

    output.mkdir(parents=True, exist_ok=False)
    source_info = source_receipt(source, source_manifest)
    train_manifest, train_source_rows = build_set(
        output=output / "train",
        purpose="counterfactual_natural_multiquery_training",
        target_rows=int(args.train_rows),
        source_rows=chosen,
        source_ids=source_ids,
        source_lengths=source_lengths,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        source_info=source_info,
        parts=parts,
        seed=int(args.seed) + 11_000,
    )
    train_source_row_set = set(train_source_rows)
    calibration_candidates = [
        int(source_row)
        for source_row in chosen
        if int(source_row) not in train_source_row_set
    ]
    calibration_manifest, calibration_source_rows = build_set(
        output=output / "calibration",
        purpose="counterfactual_natural_multiquery_calibration",
        target_rows=int(args.calibration_rows),
        source_rows=calibration_candidates,
        source_ids=source_ids,
        source_lengths=source_lengths,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
        source_info=source_info,
        parts=parts,
        seed=int(args.seed) + 22_000,
    )
    overlap = len(
        set(train_source_rows) & set(calibration_source_rows)
    )
    if overlap:
        raise RuntimeError("train/calibration source-row overlap")
    root_manifest = {
        "format_version": 1,
        "status": ROOT_STATUS,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": tokenizer_info["files"]["tokenizer.json"][
            "sha256"
        ],
        "tokenizer": tokenizer_info,
        "source": source_info,
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "train_rows": int(args.train_rows),
        "calibration_rows": int(args.calibration_rows),
        "train_calibration_source_rows_disjoint": True,
        "train_calibration_source_row_overlap": overlap,
        "queries_per_sequence": QUERIES,
        "minimum_source_unique_tokens": MINIMUM_SOURCE_UNIQUE_TOKENS,
        "maximum_source_window_attempts": MAX_SOURCE_WINDOW_ATTEMPTS,
        "benchmark_independent_generator": True,
        "ruler_generator_used": False,
        "ruler_import_used": False,
        "ruler_rows": 0,
        "ruler_or_niah_rows": 0,
        "seed": int(args.seed),
        "sets": {
            "train": {
                "path": "train",
                "manifest_sha256": sha256_file(
                    output / "train" / "manifest.json"
                ),
                "manifest": train_manifest,
            },
            "calibration": {
                "path": "calibration",
                "manifest_sha256": sha256_file(
                    output / "calibration" / "manifest.json"
                ),
                "manifest": calibration_manifest,
            },
        },
    }
    atomic_json(output / "manifest.json", root_manifest)
    print(
        json.dumps(
            {
                "status": root_manifest["status"],
                "output": str(output),
                "train_shape": train_manifest["shape"],
                "calibration_shape": calibration_manifest["shape"],
                "queries_per_sequence": QUERIES,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
