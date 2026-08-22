#!/usr/bin/env python3
"""Build identifiable, length-matched natural multi-query data.

Each semantic group owns one compact 4K task with sixteen unique anchors in a
natural passage. The 8K and 16K views keep that passage, query, answers, and
variant identity fixed; they only insert promptless natural distractors between
the passage and query. Every query therefore points to one input-visible gold
owner at every length.

The three method-selection splits use disjoint source rows and different query
contracts: train uses 16 numbered teacher-forced queries, while component_gate
and final_validation use distinct single-query strict-generation templates.
A fourth training-only strict template reuses the first 32 train semantic
groups for immediate-EOS repair. Final validation is an evaluation asset,
never a component-selection asset.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_4k_natural_multiquery_pairs import (
    ANCHOR_TOKENS,
    ANSWER_TOKENS,
    occurrences,
)


SOURCE_LENGTH = 4_096
ANCHORS_PER_PASSAGE = 16
PAIR_LENGTHS = (4_096, 8_192, 16_384)
VARIANTS = ("correct", "answer_spans_deranged")
MAX_BUILD_ATTEMPTS = 32
PAIR_STATUS = "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2"
WARMUP_STATUS = "OLMO2_PHASE_ADAROPE_CONTINUOUS_CLM_WARMUP_V1"
RETENTION_STATUS = "OLMO2_PHASE_ADAROPE_RAW_RETENTION_V1"
ROOT_STATUS = "OLMO2_PHASE_ADAROPE_DATA_PREPARED_V3"


@dataclass(frozen=True)
class SplitContract:
    name: str
    query_count: int
    template_id: str
    query_intro: str
    marker_format: str
    query_index_offset: int
    query_index_stride: int
    method_selection_allowed: bool
    strict_autoregressive: bool

    def queried_indices(self, row_index: int) -> tuple[int, ...]:
        if self.query_count == ANCHORS_PER_PASSAGE:
            return tuple(range(ANCHORS_PER_PASSAGE))
        return tuple(
            (
                int(self.query_index_offset)
                + int(self.query_index_stride) * int(row_index)
                + slot
            )
            % ANCHORS_PER_PASSAGE
            for slot in range(self.query_count)
        )

    def validate(self) -> None:
        if (
            self.name
            not in {
                "train",
                "train_eos",
                "component_gate",
                "final_validation",
            }
            or not 1 <= self.query_count <= ANCHORS_PER_PASSAGE
            or len(set(self.queried_indices(0))) != self.query_count
            or int(self.query_index_stride) <= 0
        ):
            raise ValueError("invalid split/query contract")
        if self.name == "final_validation" and self.method_selection_allowed:
            raise ValueError("final validation cannot select a method")


SPLITS = {
    "train": SplitContract(
        name="train",
        query_count=16,
        template_id="numbered_anchor_copy_train_v1",
        query_intro=(
            "\n\nQueries:\nReturn the eight-token continuation for each "
            "numbered anchor in order."
        ),
        marker_format="\nAnchor {number:02d}: ",
        query_index_offset=0,
        query_index_stride=1,
        method_selection_allowed=True,
        strict_autoregressive=False,
    ),
    "component_gate": SplitContract(
        name="component_gate",
        query_count=1,
        template_id="single_cue_component_gate_strict_v1",
        query_intro=(
            "\n\nCapability check:\nContinue the quoted cue with exactly "
            "the next eight passage tokens. Return no explanation."
        ),
        marker_format="\nCue {number:02d}: ",
        query_index_offset=0,
        query_index_stride=1,
        method_selection_allowed=True,
        strict_autoregressive=True,
    ),
    "final_validation": SplitContract(
        name="final_validation",
        query_count=1,
        template_id="single_quote_final_validation_strict_v1",
        query_intro=(
            "\n\nHeld-out continuation:\nCopy only the eight tokens that "
            "immediately follow this quoted span in the passage."
        ),
        marker_format="\nQuote {number:02d}: ",
        query_index_offset=3,
        query_index_stride=5,
        method_selection_allowed=False,
        strict_autoregressive=True,
    ),
}
TRAIN_EOS = SplitContract(
    name="train_eos",
    query_count=1,
    template_id="single_locator_train_eos_strict_v1",
    query_intro=(
        "\n\nFocused retrieval practice:\nLocate the span below and emit "
        "its next eight passage tokens only."
    ),
    marker_format="\nLocator {number:02d}: ",
    query_index_offset=7,
    query_index_stride=3,
    method_selection_allowed=False,
    strict_autoregressive=True,
)
VIEW_CONTRACTS = {**SPLITS, "train_eos": TRAIN_EOS}
for _contract in VIEW_CONTRACTS.values():
    _contract.validate()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def encoded(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False)
    ]


def prompt_layout(
    tokenizer: Any,
    contract: SplitContract,
    queried_indices: Sequence[int],
) -> dict[str, Any]:
    contract.validate()
    prefix = encoded(
        tokenizer,
        (
            f"{tokenizer.bos_token}<|user|>\n"
            "Read the natural passage. Sixteen distinct quoted anchors occur "
            "exactly once each. Use only the passage to answer the query."
            "\n\nPassage:\n"
        ),
    )
    query_intro = encoded(tokenizer, contract.query_intro)
    query_markers = [
        encoded(
            tokenizer,
            contract.marker_format.format(number=index + 1),
        )
        for index in queried_indices
    ]
    assistant_prefix = encoded(tokenizer, "\n<|assistant|>\n")
    answer_markers = (
        [[] for _ in range(contract.query_count)]
        if contract.strict_autoregressive
        else [
            encoded(tokenizer, f"Answer {ordinal + 1:02d}: ")
            for ordinal in range(contract.query_count)
        ]
    )
    answer_suffix = (
        [] if contract.strict_autoregressive else encoded(tokenizer, "\n")
    )
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
    semantic_passage_tokens = SOURCE_LENGTH - overhead
    if semantic_passage_tokens < 3_000:
        raise RuntimeError(
            f"{contract.name} prompt leaves only "
            f"{semantic_passage_tokens} semantic tokens"
        )
    return {
        "contract": contract,
        "queried_indices": tuple(int(value) for value in queried_indices),
        "prefix": prefix,
        "query_intro": query_intro,
        "query_markers": query_markers,
        "assistant_prefix": assistant_prefix,
        "answer_markers": answer_markers,
        "answer_suffix": answer_suffix,
        "overhead_tokens": overhead,
        "semantic_passage_tokens": semantic_passage_tokens,
    }


def _normalize_source_rows(
    receipt: Mapping[str, Any],
    count: int,
    *,
    validation_source_rows: int,
) -> list[dict[str, Any]]:
    rows = receipt.get("rows", receipt.get("row_provenance"))
    if not isinstance(rows, list):
        documents = receipt.get("documents")
        if not isinstance(documents, list) or len(documents) != count:
            raise ValueError("source receipt needs rows or documents")
        validation_start = count - int(validation_source_rows)
        rows = [
            {
                "row": index,
                "split": (
                    "validation" if index >= validation_start else "train"
                ),
                "document_id": str(
                    item.get("document_id", item.get("text_sha256", index))
                ),
            }
            for index, item in enumerate(documents)
            if isinstance(item, Mapping)
        ]
    if len(rows) != count:
        raise ValueError("source receipt row count differs from tensor")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("source receipt row is not an object")
        split = str(raw.get("split", ""))
        if split == "val":
            split = "validation"
        document_id = raw.get("document_id", raw.get("doc_id"))
        if (
            int(raw.get("row", index)) != index
            or split not in {"train", "validation"}
            or document_id is None
        ):
            raise ValueError("source row identity/split drift")
        normalized.append(
            {
                **dict(raw),
                "row": index,
                "split": split,
                "document_id": str(document_id),
            }
        )
    document_ids = [row["document_id"] for row in normalized]
    if len(set(document_ids)) != count:
        raise ValueError("source documents must be distinct")
    train_docs = {
        row["document_id"] for row in normalized if row["split"] == "train"
    }
    validation_docs = {
        row["document_id"]
        for row in normalized
        if row["split"] == "validation"
    }
    if not train_docs or not validation_docs or train_docs & validation_docs:
        raise ValueError("source train/validation documents are invalid")
    return normalized


def load_source_tensor(
    tensor_path: Path,
    receipt_path: Path,
    *,
    validation_source_rows: int = 128,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    import torch

    value = torch.load(
        tensor_path.resolve(), map_location="cpu", weights_only=True
    )
    if isinstance(value, Mapping):
        value = value.get("input_ids", value.get("tokens"))
    raw = np.asarray(value) if value is not None else np.asarray([])
    if (
        raw.ndim != 2
        or raw.shape[1] != SOURCE_LENGTH
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw < 0)
        or np.any(raw > np.iinfo(np.uint32).max)
    ):
        raise ValueError("source tensor must be non-negative [rows,4096] ints")
    tokens = np.asarray(raw, dtype=np.uint32)
    receipt = json.loads(receipt_path.resolve().read_text(encoding="utf-8"))
    rows = _normalize_source_rows(
        receipt,
        len(tokens),
        validation_source_rows=int(validation_source_rows),
    )
    file_sha = sha256_file(tensor_path)
    array_sha = sha256_array(tokens)
    expected = receipt.get("tensor_sha256")
    if expected is not None and str(expected) not in {file_sha, array_sha}:
        raise ValueError("source tensor hash differs from receipt")
    return tokens, rows, {
        "tensor_path": str(tensor_path.resolve()),
        "tensor_file_sha256": file_sha,
        "tensor_array_sha256_uint32": array_sha,
        "token_receipt_path": str(receipt_path.resolve()),
        "token_receipt_sha256": sha256_file(receipt_path),
        "rows": len(tokens),
        "tokens_per_row": SOURCE_LENGTH,
    }


def tokenizer_receipt(checkpoint: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = checkpoint / name
        if path.is_file():
            files[name] = file_receipt(path)
    if "tokenizer.json" not in files:
        raise FileNotFoundError(checkpoint / "tokenizer.json")
    digest = hashlib.sha256()
    for name, receipt in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(receipt["sha256"]))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "files": files,
        "composite_sha256": digest.hexdigest(),
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
    for bin_index in range(ANCHORS_PER_PASSAGE):
        left = (bin_index * len(passage)) // ANCHORS_PER_PASSAGE
        right = ((bin_index + 1) * len(passage)) // ANCHORS_PER_PASSAGE
        minimum = left + 4
        maximum = right - ANCHOR_TOKENS - ANSWER_TOKENS - 4
        if maximum < minimum:
            return None
        selected = None
        for raw_position in rng.permutation(
            np.arange(minimum, maximum + 1, dtype=np.int64)
        ):
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
            if (
                any(
                    int(token) in forbidden
                    for token in np.concatenate((anchor, answer))
                )
                or anchor.tobytes() in anchor_digests
                or answer.tobytes() in answer_digests
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
    *, seed: int, semantic_source_row: int, row_index: int
) -> tuple[int, list[int]]:
    shift = 1 + (
        (int(seed) + int(semantic_source_row) + int(row_index))
        % (ANCHORS_PER_PASSAGE - 1)
    )
    mapping = [
        (index + shift) % ANCHORS_PER_PASSAGE
        for index in range(ANCHORS_PER_PASSAGE)
    ]
    if any(index == source for index, source in enumerate(mapping)):
        raise RuntimeError("answer derangement has a fixed point")
    return shift, mapping


def derange_passage_answers(
    *,
    passage: np.ndarray,
    positions: Sequence[int],
    answers: Sequence[np.ndarray],
    mapping: Sequence[int],
) -> np.ndarray:
    deranged = passage.copy()
    mutable = np.zeros(len(passage), dtype=bool)
    for index, position in enumerate(positions):
        start = int(position) + ANCHOR_TOKENS
        stop = start + ANSWER_TOKENS
        deranged[start:stop] = answers[int(mapping[index])]
        mutable[start:stop] = True
    if (
        not np.array_equal(passage[~mutable], deranged[~mutable])
        or np.array_equal(passage, deranged)
    ):
        raise RuntimeError("derangement escaped answer spans or did nothing")
    return deranged


def build_sequence(
    *,
    layout: Mapping[str, Any],
    semantic_passage: np.ndarray,
    distractors: np.ndarray,
    anchors: Sequence[np.ndarray],
    answers: Sequence[np.ndarray],
    eos_token_id: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    contract: SplitContract = layout["contract"]
    tokens: list[int] = list(layout["prefix"])
    passage_start = len(tokens)
    tokens.extend(int(value) for value in semantic_passage)
    semantic_passage_stop = len(tokens)
    extension_start = len(tokens)
    tokens.extend(int(value) for value in distractors)
    extension_stop = len(tokens)
    query_start = len(tokens)
    tokens.extend(layout["query_intro"])
    query_anchor_spans: list[tuple[int, int]] = []
    for marker, anchor_index in zip(
        layout["query_markers"], layout["queried_indices"]
    ):
        tokens.extend(marker)
        start = len(tokens)
        tokens.extend(int(value) for value in anchors[anchor_index])
        query_anchor_spans.append((start, len(tokens)))
    query_stop = len(tokens)
    tokens.extend(layout["assistant_prefix"])
    labels = [-100] * len(tokens)
    output_answer_spans: list[tuple[int, int]] = []
    for marker, anchor_index in zip(
        layout["answer_markers"], layout["queried_indices"]
    ):
        tokens.extend(marker)
        labels.extend([-100] * len(marker))
        start = len(tokens)
        answer = answers[anchor_index]
        tokens.extend(int(value) for value in answer)
        labels.extend(int(value) for value in answer)
        output_answer_spans.append((start, len(tokens)))
        tokens.extend(layout["answer_suffix"])
        labels.extend([-100] * len(layout["answer_suffix"]))
    eos_position = len(tokens)
    tokens.append(int(eos_token_id))
    labels.append(int(eos_token_id))
    if len(tokens) != int(length) or len(labels) != int(length):
        raise RuntimeError(
            f"{contract.name} L{length} sequence length drift: {len(tokens)}"
        )
    expected_supervision = contract.query_count * ANSWER_TOKENS + 1
    if sum(value != -100 for value in labels) != expected_supervision:
        raise RuntimeError("supervised-token count drift")
    target_positions = [
        position
        for start, stop in output_answer_spans
        for position in range(start, stop)
    ] + [eos_position]
    return (
        np.asarray(tokens, dtype=np.uint32),
        np.asarray(labels, dtype=np.int32),
        {
            "semantic_passage_span": np.asarray(
                [passage_start, semantic_passage_stop], dtype=np.int32
            ),
            "extension_span": np.asarray(
                [extension_start, extension_stop], dtype=np.int32
            ),
            "query_span": np.asarray(
                [query_start, query_stop], dtype=np.int32
            ),
            "query_anchor_spans": np.asarray(
                query_anchor_spans, dtype=np.int32
            ),
            "output_answer_spans": np.asarray(
                output_answer_spans, dtype=np.int32
            ),
            "target_positions": np.asarray(
                target_positions, dtype=np.int32
            ),
            "generation_prompt_stop": np.asarray(
                output_answer_spans[0][0]
                if contract.strict_autoregressive
                else -1,
                dtype=np.int32,
            ),
        },
    )


def _span_mask(length: int, spans: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    for start, stop in np.asarray(spans, dtype=np.int64):
        mask[int(start) : int(stop)] = True
    return mask


def validate_identifiable_pair(
    *,
    input_ids: np.ndarray,
    labels: np.ndarray,
    source_answer_spans: np.ndarray,
    queried_indices: np.ndarray,
    query_anchor_spans: np.ndarray,
    output_answer_spans: np.ndarray,
    target_positions: np.ndarray,
    eos_token_id: int,
) -> None:
    if input_ids.shape[0] != 2 or labels.shape != input_ids.shape:
        raise RuntimeError("identifiable pair shape drift")
    length = input_ids.shape[1]
    for variant in range(2):
        for query_slot, owner_index in enumerate(queried_indices.tolist()):
            source_start, source_stop = source_answer_spans[owner_index]
            query_start, query_stop = query_anchor_spans[query_slot]
            output_start, output_stop = output_answer_spans[query_slot]
            anchor = input_ids[
                variant,
                int(source_start) - ANCHOR_TOKENS : int(source_start),
            ]
            if not np.array_equal(
                anchor,
                input_ids[variant, int(query_start) : int(query_stop)],
            ):
                raise RuntimeError("query anchor does not identify source")
            source_answer = input_ids[
                variant, int(source_start) : int(source_stop)
            ]
            output_answer = input_ids[
                variant, int(output_start) : int(output_stop)
            ]
            if not np.array_equal(source_answer, output_answer):
                raise RuntimeError("gold owner is not recoverable")
            if not np.array_equal(
                labels[variant, int(output_start) : int(output_stop)],
                output_answer.astype(np.int32),
            ):
                raise RuntimeError("labels differ from source-owned answer")
        supervised = np.flatnonzero(labels[variant] != -100)
        if not np.array_equal(supervised, target_positions):
            raise RuntimeError("materialized target positions drift")
        if (
            int(input_ids[variant, -1]) != int(eos_token_id)
            or int(labels[variant, -1]) != int(eos_token_id)
        ):
            raise RuntimeError("terminal EOS contract drift")
    mutable = _span_mask(length, source_answer_spans)
    mutable |= _span_mask(length, output_answer_spans)
    if not np.array_equal(input_ids[0, ~mutable], input_ids[1, ~mutable]):
        raise RuntimeError("variants differ outside answer spans")
    if np.array_equal(input_ids[0], input_ids[1]):
        raise RuntimeError("deranged variant is identical to correct")


def build_semantic_group(
    *,
    source_tokens: np.ndarray,
    source_group_rows: Sequence[int],
    tokenizer: Any,
    contract: SplitContract,
    seed: int,
    row_index: int,
) -> dict[str, Any]:
    if len(source_group_rows) != 4:
        raise ValueError("semantic group requires one source and three distractors")
    queried_indices = contract.queried_indices(row_index)
    layout = prompt_layout(tokenizer, contract, queried_indices)
    semantic_source_row = int(source_group_rows[0])
    source = np.asarray(source_tokens[semantic_source_row], dtype=np.uint32)
    passage_tokens = int(layout["semantic_passage_tokens"])
    forbidden = {
        int(tokenizer.bos_token_id),
        int(tokenizer.eos_token_id),
        int(tokenizer.pad_token_id),
    }
    rng = np.random.default_rng(int(seed) + int(row_index) * 1_000_003)
    for _ in range(MAX_BUILD_ATTEMPTS):
        window_start = int(
            rng.integers(0, SOURCE_LENGTH - passage_tokens + 1)
        )
        passage = np.asarray(
            source[window_start : window_start + passage_tokens],
            dtype=np.int64,
        )
        selected = select_anchor_answer_spans(
            passage=passage, rng=rng, forbidden=forbidden
        )
        if selected is None:
            continue
        positions, anchors, answers = selected
        unused_semantic_source = np.concatenate(
            (
                source[:window_start],
                source[window_start + passage_tokens :],
            )
        )
        long_distractors = np.concatenate(
            (
                np.asarray(
                    source_tokens[
                        np.asarray(source_group_rows[1:], dtype=np.int32)
                    ],
                    dtype=np.uint32,
                ).reshape(-1),
                unused_semantic_source,
            )
        )
        if any(
            occurrences(long_distractors, anchor) != 0
            or occurrences(
                np.asarray(layout["prefix"], dtype=np.uint32), anchor
            )
            != 0
            for anchor in anchors
        ):
            continue
        shift, mapping = deterministic_derangement(
            seed=int(seed),
            semantic_source_row=semantic_source_row,
            row_index=int(row_index),
        )
        deranged = derange_passage_answers(
            passage=passage,
            positions=positions,
            answers=answers,
            mapping=mapping,
        )
        if any(occurrences(deranged, anchor) != 1 for anchor in anchors):
            continue
        owner_offsets = np.asarray(
            [
                window_start + int(position) + ANCHOR_TOKENS
                for position in positions
            ],
            dtype=np.int32,
        )
        return {
            "layout": layout,
            "source_group_rows": np.asarray(
                source_group_rows, dtype=np.int32
            ),
            "semantic_source_row": semantic_source_row,
            "semantic_window_start": window_start,
            "correct_passage": passage,
            "deranged_passage": deranged,
            "anchors": anchors,
            "correct_answers": answers,
            "deranged_answers": [
                answers[int(mapping[index])]
                for index in range(ANCHORS_PER_PASSAGE)
            ],
            "anchor_positions_in_passage": np.asarray(
                positions, dtype=np.int32
            ),
            "source_owner_rows": np.full(
                ANCHORS_PER_PASSAGE,
                semantic_source_row,
                dtype=np.int32,
            ),
            "source_owner_offsets": owner_offsets,
            "answer_origin_indices": np.asarray(
                [list(range(ANCHORS_PER_PASSAGE)), mapping],
                dtype=np.int16,
            ),
            "derangement_shift": shift,
            "semantic_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.stack((passage, deranged)).astype("<u4")
                ).tobytes()
            ).hexdigest(),
        }
    raise RuntimeError(
        f"could not build {contract.name} semantic row {row_index}"
    )


def render_semantic_group(
    *,
    semantic: Mapping[str, Any],
    source_tokens: np.ndarray,
    tokenizer: Any,
    contract: SplitContract,
    length: int,
    row_index: int = 0,
) -> dict[str, np.ndarray]:
    if int(length) not in PAIR_LENGTHS:
        raise ValueError("render length must be 4K/8K/16K")
    queried_indices = contract.queried_indices(row_index)
    layout = prompt_layout(tokenizer, contract, queried_indices)
    extension_tokens = (
        int(length)
        - len(semantic["correct_passage"])
        - int(layout["overhead_tokens"])
    )
    if extension_tokens < 0:
        raise RuntimeError("render contract does not fit the requested length")
    source_group_rows = np.asarray(
        semantic["source_group_rows"], dtype=np.int32
    )
    semantic_source = np.asarray(
        source_tokens[source_group_rows[0]], dtype=np.uint32
    )
    window_start = int(semantic["semantic_window_start"])
    passage_stop = window_start + len(semantic["correct_passage"])
    extension_pool = np.concatenate(
        (
            np.asarray(
                source_tokens[source_group_rows[1:]], dtype=np.uint32
            ).reshape(-1),
            semantic_source[:window_start],
            semantic_source[passage_stop:],
        )
    )
    distractors = extension_pool[:extension_tokens]
    if len(distractors) != extension_tokens:
        raise RuntimeError("semantic group lacks long-view distractors")
    correct, correct_labels, correct_positions = build_sequence(
        layout=layout,
        semantic_passage=semantic["correct_passage"],
        distractors=distractors,
        anchors=semantic["anchors"],
        answers=semantic["correct_answers"],
        eos_token_id=int(tokenizer.eos_token_id),
        length=int(length),
    )
    deranged, deranged_labels, deranged_positions = build_sequence(
        layout=layout,
        semantic_passage=semantic["deranged_passage"],
        distractors=distractors,
        anchors=semantic["anchors"],
        answers=semantic["deranged_answers"],
        eos_token_id=int(tokenizer.eos_token_id),
        length=int(length),
    )
    if any(
        not np.array_equal(correct_positions[name], deranged_positions[name])
        for name in correct_positions
    ):
        raise RuntimeError("correct/deranged position geometry drift")
    semantic_start = int(correct_positions["semantic_passage_span"][0])
    source_answer_spans = np.asarray(
        [
            [
                semantic_start + int(position) + ANCHOR_TOKENS,
                semantic_start
                + int(position)
                + ANCHOR_TOKENS
                + ANSWER_TOKENS,
            ]
            for position in semantic["anchor_positions_in_passage"]
        ],
        dtype=np.int32,
    )
    pair_ids = np.stack((correct, deranged))
    pair_labels = np.stack((correct_labels, deranged_labels))
    target_positions = correct_positions["target_positions"]
    validate_identifiable_pair(
        input_ids=pair_ids,
        labels=pair_labels,
        source_answer_spans=source_answer_spans,
        queried_indices=np.asarray(
            queried_indices, dtype=np.int16
        ),
        query_anchor_spans=correct_positions["query_anchor_spans"],
        output_answer_spans=correct_positions["output_answer_spans"],
        target_positions=target_positions,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    target_token_ids = np.stack(
        [labels[target_positions] for labels in pair_labels]
    ).astype(np.uint32)
    return {
        "input_ids": pair_ids,
        "labels": pair_labels,
        "source_answer_spans": source_answer_spans,
        "query_anchor_spans": correct_positions["query_anchor_spans"],
        "output_answer_spans": correct_positions["output_answer_spans"],
        "query_span": correct_positions["query_span"],
        "extension_span": correct_positions["extension_span"],
        "target_positions": target_positions,
        "target_token_ids": target_token_ids,
        "generation_prompt_stop": correct_positions[
            "generation_prompt_stop"
        ],
        "queried_indices": np.asarray(
            queried_indices, dtype=np.int16
        ),
        "source_owner_rows": semantic["source_owner_rows"],
        "source_owner_offsets": semantic["source_owner_offsets"],
        "answer_origin_indices": semantic["answer_origin_indices"],
        "source_group_rows": semantic["source_group_rows"],
        "semantic_window_start": np.asarray(
            semantic["semantic_window_start"], dtype=np.int32
        ),
    }


def _write_pair_view(
    *,
    output: Path,
    contract: SplitContract,
    length: int,
    semantics: Sequence[Mapping[str, Any]],
    source_tokens: np.ndarray,
    tokenizer: Any,
    tokenizer_info: Mapping[str, Any],
    source_info: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows = len(semantics)
    queries = contract.query_count
    targets = queries * ANSWER_TOKENS + 1
    specs = {
        "input_ids.npy": (np.uint32, (rows, 2, int(length))),
        "labels.npy": (np.int32, (rows, 2, int(length))),
        "source_answer_spans.npy": (
            np.int32,
            (rows, ANCHORS_PER_PASSAGE, 2),
        ),
        "query_anchor_spans.npy": (np.int32, (rows, queries, 2)),
        "output_answer_spans.npy": (np.int32, (rows, queries, 2)),
        "query_spans.npy": (np.int32, (rows, 2)),
        "extension_spans.npy": (np.int32, (rows, 2)),
        "target_positions.npy": (np.int32, (rows, targets)),
        "target_token_ids.npy": (np.uint32, (rows, 2, targets)),
        "queried_indices.npy": (np.int16, (rows, queries)),
        "source_owner_rows.npy": (
            np.int32,
            (rows, ANCHORS_PER_PASSAGE),
        ),
        "source_owner_offsets.npy": (
            np.int32,
            (rows, ANCHORS_PER_PASSAGE),
        ),
        "answer_origin_indices.npy": (
            np.int16,
            (rows, 2, ANCHORS_PER_PASSAGE),
        ),
        "source_group_rows.npy": (np.int32, (rows, 4)),
        "semantic_window_starts.npy": (np.int32, (rows,)),
        "generation_prompt_stops.npy": (np.int32, (rows,)),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            output / name, mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specs.items()
    }
    rows_path = output / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row_index, semantic in enumerate(semantics):
            result = render_semantic_group(
                semantic=semantic,
                source_tokens=source_tokens,
                tokenizer=tokenizer,
                contract=contract,
                length=int(length),
                row_index=row_index,
            )
            for name, value in result.items():
                filename = {
                    "query_span": "query_spans.npy",
                    "extension_span": "extension_spans.npy",
                    "semantic_window_start": "semantic_window_starts.npy",
                    "generation_prompt_stop": "generation_prompt_stops.npy",
                }.get(name, f"{name}.npy")
                arrays[filename][row_index] = value
            handle.write(
                json.dumps(
                    {
                        "row": row_index,
                        "split": contract.name,
                        "template_id": contract.template_id,
                        "length": int(length),
                        "source_group_rows": [
                            int(value)
                            for value in semantic["source_group_rows"]
                        ],
                        "semantic_sha256": semantic["semantic_sha256"],
                        "document_id": semantic["semantic_sha256"],
                        "input_pair_sha256": sha256_array(
                            result["input_ids"]
                        ),
                        "target_token_ids_sha256": sha256_array(
                            result["target_token_ids"]
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    for value in arrays.values():
        value.flush()
    files = {
        path.name: file_receipt(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    semantic_hashes = [
        str(semantic["semantic_sha256"]) for semantic in semantics
    ]
    manifest = {
        "schema_version": 3,
        "status": PAIR_STATUS,
        "split": contract.name,
        "method_selection_allowed": contract.method_selection_allowed,
        "training_only": contract.name in {"train", "train_eos"},
        "reuses_semantic_source_split": (
            "train" if contract.name == "train_eos" else None
        ),
        "strict_autoregressive_capability": contract.strict_autoregressive,
        "template_id": contract.template_id,
        "length": int(length),
        "shape": [rows, 2, int(length)],
        "variant_names": list(VARIANTS),
        "anchors_per_passage": ANCHORS_PER_PASSAGE,
        "query_count": contract.query_count,
        "query_index_policy": {
            "offset": contract.query_index_offset,
            "stride_by_row": contract.query_index_stride,
            "modulus": ANCHORS_PER_PASSAGE,
        },
        "anchor_tokens": ANCHOR_TOKENS,
        "answer_tokens_per_query": ANSWER_TOKENS,
        "supervised_tokens_per_variant": targets,
        "terminal_eos_supervised": True,
        "strict_generation_contract": (
            "prompt_stops_at_unique_answer_start_then_8_tokens_plus_terminal_eos"
            if contract.strict_autoregressive
            else "teacher_forced_dense_multiquery"
        ),
        "query_explicitly_lists_anchors": True,
        "anchors_unique_in_semantic_passage": True,
        "variants_differ_only_in_source_and_output_answer_spans": True,
        "gold_owner_recoverable_from_query_anchor": True,
        "semantic_group_matches_4k_8k_16k": True,
        "long_view_change_from_4k": (
            "none"
            if int(length) == SOURCE_LENGTH
            else "natural_distractors_between_passage_and_query"
        ),
        "query_shift_from_compact_4k": int(length) - SOURCE_LENGTH,
        "semantic_hashes_sha256": hashlib.sha256(
            json.dumps(semantic_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
        "source": dict(source_info),
        "tokenizer": dict(tokenizer_info),
        "benchmark_generator_used": False,
        "seed": int(seed),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _write_warmup_view(
    *, output: Path, source_tokens: np.ndarray, source_rows: np.ndarray
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows = np.asarray(source_rows, dtype=np.int32)
    inputs = np.asarray(source_tokens[rows], dtype=np.uint32)
    labels = inputs.astype(np.int32)
    labels[:, 0] = -100
    np.save(output / "input_ids.npy", inputs, allow_pickle=False)
    np.save(output / "labels.npy", labels, allow_pickle=False)
    np.save(output / "source_rows.npy", rows, allow_pickle=False)
    files = {
        name: file_receipt(output / name)
        for name in ("input_ids.npy", "labels.npy", "source_rows.npy")
    }
    manifest = {
        "schema_version": 1,
        "status": WARMUP_STATUS,
        "shape": list(inputs.shape),
        "prompt_tokens": 0,
        "continuous_within_document": True,
        "prediction_shift": "token_p_is_predicted_from_p_minus_1",
        "source_rows_sha256": sha256_array(rows.astype("<i4")),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _write_retention_view(
    *, output: Path, source_tokens: np.ndarray, source_rows: np.ndarray
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows = np.asarray(source_rows, dtype=np.int32)
    inputs = np.asarray(source_tokens[rows], dtype=np.uint32)
    np.save(output / "input_ids.npy", inputs, allow_pickle=False)
    np.save(output / "source_rows.npy", rows, allow_pickle=False)
    files = {
        name: file_receipt(output / name)
        for name in ("input_ids.npy", "source_rows.npy")
    }
    manifest = {
        "schema_version": 1,
        "status": RETENTION_STATUS,
        "shape": list(inputs.shape),
        "raw_promptless_rows": True,
        "labels_materialized": False,
        "source_rows_sha256": sha256_array(rows.astype("<i4")),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _source_code_has_benchmark_import(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    marker = "r" + "uler"
    return any(marker in name.lower() for name in imports)


def build_dataset(
    *,
    source_tokens: np.ndarray,
    source_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_info: Mapping[str, Any],
    source_info: Mapping[str, Any],
    output: Path,
    train_rows: int,
    train_eos_rows: int,
    component_gate_rows: int,
    final_validation_rows: int,
    warmup_rows: int,
    retention_rows: int,
    seed: int,
) -> dict[str, Any]:
    output = output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    counts = {
        "train": int(train_rows),
        "component_gate": int(component_gate_rows),
        "final_validation": int(final_validation_rows),
        "warmup": int(warmup_rows),
        "retention": int(retention_rows),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all output row counts must be positive")
    if not 1 <= int(train_eos_rows) <= counts["train"]:
        raise ValueError("train_eos_rows must be within the train row count")
    train_candidates = np.asarray(
        [row["row"] for row in source_rows if row["split"] == "train"],
        dtype=np.int32,
    )
    validation_candidates = np.asarray(
        [
            row["row"]
            for row in source_rows
            if row["split"] == "validation"
        ],
        dtype=np.int32,
    )
    rng = np.random.default_rng(int(seed))
    train_candidates = rng.permutation(train_candidates)
    validation_candidates = rng.permutation(validation_candidates)
    group_width = 4
    train_end = counts["train"] * group_width
    gate_end = train_end + counts["component_gate"] * group_width
    warmup_end = gate_end + counts["warmup"]
    final_end = counts["final_validation"] * group_width
    retention_end = final_end + counts["retention"]
    if len(train_candidates) < warmup_end:
        raise ValueError("too few train documents for train/gate/warmup")
    if len(validation_candidates) < retention_end:
        raise ValueError("too few validation documents for final/retention")
    groups = {
        "train": train_candidates[:train_end].reshape(
            counts["train"], group_width
        ),
        "component_gate": train_candidates[train_end:gate_end].reshape(
            counts["component_gate"], group_width
        ),
        "final_validation": validation_candidates[:final_end].reshape(
            counts["final_validation"], group_width
        ),
    }
    warmup_source_rows = train_candidates[gate_end:warmup_end]
    retention_source_rows = validation_candidates[final_end:retention_end]
    role_rows = {
        **{
            name: sorted(value.reshape(-1).tolist())
            for name, value in groups.items()
        },
        "warmup": sorted(warmup_source_rows.tolist()),
        "retention": sorted(retention_source_rows.tolist()),
    }
    role_sets = {name: set(values) for name, values in role_rows.items()}
    names = list(role_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if role_sets[first] & role_sets[second]:
                raise RuntimeError(f"source roles overlap: {first}/{second}")
    if _source_code_has_benchmark_import(Path(__file__).resolve()):
        raise RuntimeError("data builder imports a benchmark generator")
    incomplete.mkdir(parents=True)
    manifests: dict[str, Any] = {}
    semantic_hashes: dict[str, list[str]] = {}
    for split_index, (split_name, split_groups) in enumerate(groups.items()):
        contract = SPLITS[split_name]
        semantics = [
            build_semantic_group(
                source_tokens=source_tokens,
                source_group_rows=group.tolist(),
                tokenizer=tokenizer,
                contract=contract,
                seed=int(seed) + 100_000 * split_index,
                row_index=row_index,
            )
            for row_index, group in enumerate(split_groups)
        ]
        semantic_hashes[split_name] = [
            str(value["semantic_sha256"]) for value in semantics
        ]
        for length in PAIR_LENGTHS:
            view_name = f"{split_name}{length // 1024}k"
            manifests[view_name] = _write_pair_view(
                output=incomplete / view_name,
                contract=contract,
                length=length,
                semantics=semantics,
                source_tokens=source_tokens,
                tokenizer=tokenizer,
                tokenizer_info=tokenizer_info,
                source_info=source_info,
                seed=int(seed),
            )
        if split_name == "train":
            eos_semantics = semantics[: int(train_eos_rows)]
            semantic_hashes["train_eos"] = [
                str(value["semantic_sha256"]) for value in eos_semantics
            ]
            for length in PAIR_LENGTHS:
                view_name = f"train_eos{length // 1024}k"
                manifests[view_name] = _write_pair_view(
                    output=incomplete / view_name,
                    contract=TRAIN_EOS,
                    length=length,
                    semantics=eos_semantics,
                    source_tokens=source_tokens,
                    tokenizer=tokenizer,
                    tokenizer_info=tokenizer_info,
                    source_info=source_info,
                    seed=int(seed),
                )
    warmup_manifest = _write_warmup_view(
        output=incomplete / "warmup4k_clm",
        source_tokens=source_tokens,
        source_rows=warmup_source_rows,
    )
    retention_manifest = _write_retention_view(
        output=incomplete / "retention4k_raw",
        source_tokens=source_tokens,
        source_rows=retention_source_rows,
    )
    root = {
        "schema_version": 3,
        "status": ROOT_STATUS,
        "pair_views": {
            name: {
                "relative_path": name,
                "manifest_sha256": sha256_file(
                    incomplete / name / "manifest.json"
                ),
                "split": manifest["split"],
                "length": manifest["length"],
                "rows": manifest["shape"][0],
                "template_id": manifest["template_id"],
                "query_count": manifest["query_count"],
                "method_selection_allowed": manifest[
                    "method_selection_allowed"
                ],
                "strict_autoregressive_capability": manifest[
                    "strict_autoregressive_capability"
                ],
                "training_only": manifest["training_only"],
                "reuses_semantic_source_split": manifest[
                    "reuses_semantic_source_split"
                ],
            }
            for name, manifest in manifests.items()
        },
        "split_contracts": {
            name: {
                "query_count": contract.query_count,
                "template_id": contract.template_id,
                "query_index_policy": {
                    "offset": contract.query_index_offset,
                    "stride_by_row": contract.query_index_stride,
                    "modulus": ANCHORS_PER_PASSAGE,
                },
                "method_selection_allowed": contract.method_selection_allowed,
                "strict_autoregressive_capability": contract.strict_autoregressive,
                "semantic_hashes": semantic_hashes[name],
            }
            for name, contract in VIEW_CONTRACTS.items()
        },
        "warmup_view": {
            "relative_path": "warmup4k_clm",
            "manifest_sha256": sha256_file(
                incomplete / "warmup4k_clm" / "manifest.json"
            ),
            "manifest": warmup_manifest,
        },
        "retention_view": {
            "relative_path": "retention4k_raw",
            "manifest_sha256": sha256_file(
                incomplete / "retention4k_raw" / "manifest.json"
            ),
            "manifest": retention_manifest,
        },
        "source": dict(source_info),
        "tokenizer": dict(tokenizer_info),
        "source_role_rows": role_rows,
        "source_role_rows_sha256": {
            name: hashlib.sha256(
                np.asarray(values, dtype="<i4").tobytes()
            ).hexdigest()
            for name, values in role_rows.items()
        },
        "all_source_roles_disjoint": True,
        "train_eos_rows": int(train_eos_rows),
        "train_eos_reuses_train_semantics": True,
        "train_eos_adds_source_rows": False,
        "compact_and_long_semantics_identical": True,
        "long_views_only_insert_natural_distractors": True,
        "final_validation_for_method_selection": False,
        "benchmark_generator_used": False,
        "seed": int(seed),
        "code_sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "multiquery_contract": sha256_file(
                Path(
                    __import__(
                        "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_4k_natural_multiquery_pairs",
                        fromlist=["__file__"],
                    ).__file__
                ).resolve()
            ),
        },
    }
    atomic_json(incomplete / "manifest.json", root)
    incomplete.replace(output)
    return root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-tensor", type=Path, required=True)
    parser.add_argument("--source-token-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--train-eos-rows", type=int, default=32)
    parser.add_argument("--component-gate-rows", type=int, default=32)
    parser.add_argument("--final-validation-rows", type=int, default=16)
    parser.add_argument("--warmup-rows", type=int, default=32)
    parser.add_argument("--retention-rows", type=int, default=32)
    parser.add_argument("--validation-source-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_822)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if any(
        value is None
        for value in (
            tokenizer.bos_token,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
    ):
        raise RuntimeError("tokenizer lacks required special tokens")
    source_tokens, rows, source_info = load_source_tensor(
        args.source_tensor.resolve(),
        args.source_token_receipt.resolve(),
        validation_source_rows=int(args.validation_source_rows),
    )
    result = build_dataset(
        source_tokens=source_tokens,
        source_rows=rows,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_receipt(checkpoint),
        source_info=source_info,
        output=args.output.resolve(),
        train_rows=int(args.train_rows),
        train_eos_rows=int(args.train_eos_rows),
        component_gate_rows=int(args.component_gate_rows),
        final_validation_rows=int(args.final_validation_rows),
        warmup_rows=int(args.warmup_rows),
        retention_rows=int(args.retention_rows),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "pair_views": sorted(result["pair_views"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
