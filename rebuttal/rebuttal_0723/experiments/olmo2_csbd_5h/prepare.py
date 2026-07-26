#!/usr/bin/env python3
"""Freeze paired short/long counterfactual binding data for CSBD.

Every A/B pair has identical length, keys, value multiset, filler, template,
absolute positions, and query positions.  The only change is a fixed-point-free
permutation of which one-token value is bound to each key.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    broad_vocabulary_train_values,
    encode,
    random_key,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    place_phrase,
)

from .protocol import (
    EVAL_VALUE_COUNT,
    EXTRAPOLATION_DISTANCE_BANDS,
    EXTRAPOLATION_LENGTHS,
    EXTRAPOLATION_ROWS,
    FACTS,
    FINAL_TEST_ROWS,
    HELDOUT_DISTANCE_BANDS,
    LONG_LENGTH,
    QUERIES,
    SEED,
    SHORT_LENGTH,
    TRAIN_DISTANCE_BANDS,
    TRAIN_ROWS_PER_BAND,
    TRAIN_VALUE_COUNT,
    VALIDATION_ROWS,
    atomic_json,
    file_manifest,
    sha256_array,
    sha256_file,
    verify_checkpoint,
)


TRAIN_TEMPLATES = (
    (
        "\nRecord {key} stores",
        ".\n",
        "\nReturn the value stored by record {key}:",
        "record_return_v1",
    ),
    (
        "\nLedger key [{key}] contains",
        ".\n",
        "\nResolve ledger key [{key}]:",
        "ledger_resolve_v2",
    ),
    (
        "\nLookup({key}) maps to",
        ".\n",
        "\nWhat does Lookup({key}) map to?",
        "lookup_map_v3",
    ),
)
HELDOUT_TEMPLATE = (
    "\nArchive identifier <{key}> is assigned",
    ".\n",
    "\nWhich token is assigned to archive identifier <{key}>?",
    "archive_assigned_heldout_v1",
)


@dataclass(frozen=True)
class SourceSpec:
    original: np.ndarray
    swapped: np.ndarray
    value_offset: int
    template_id: str


def make_filler(
    documents: np.ndarray,
    document_rows: Sequence[int],
    length: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[int]]:
    pieces: list[np.ndarray] = []
    used: list[int] = []
    remaining = int(length)
    while remaining > 0:
        row = int(document_rows[rng.randrange(len(document_rows))])
        used.append(row)
        document = np.asarray(documents[row], dtype=np.uint32)
        if remaining >= len(document):
            pieces.append(document)
            remaining -= len(document)
        else:
            maximum = len(document) - remaining
            start = rng.randrange(maximum + 1)
            pieces.append(document[start : start + remaining])
            remaining = 0
    output = np.concatenate(pieces).astype(np.uint32, copy=True)
    if output.shape != (int(length),):
        raise RuntimeError("filler construction length drift")
    return output, used


def phrase_tokens(
    tokenizer: Any,
    *,
    key: str,
    value_id: int,
    template: tuple[str, str, str, str],
) -> tuple[np.ndarray, int, np.ndarray]:
    prefix_text, suffix_text, query_text, _ = template
    prefix = encode(tokenizer, prefix_text.format(key=key))
    suffix = encode(tokenizer, suffix_text)
    source = np.concatenate(
        (
            prefix,
            np.asarray([int(value_id)], dtype=np.int64),
            suffix,
        )
    )
    query = encode(tokenizer, query_text.format(key=key))
    return source, len(prefix), query


def fixed_point_free_permutation(
    values: Sequence[int], rng: random.Random
) -> tuple[int, ...]:
    if len(values) < 2:
        raise ValueError("derangement needs at least two values")
    shift = rng.randint(1, len(values) - 1)
    result = tuple(int(value) for value in values[shift:]) + tuple(
        int(value) for value in values[:shift]
    )
    if any(int(a) == int(b) for a, b in zip(values, result)):
        raise RuntimeError("cyclic value permutation has a fixed point")
    return result


def build_query_block(
    *,
    tokenizer: Any,
    query_fact_indices: Sequence[int],
    queries: Sequence[np.ndarray],
    original_values: Sequence[int],
    swapped_values: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    newline = encode(tokenizer, "\n")
    original: list[int] = []
    swapped: list[int] = []
    answer_offsets: list[int] = []
    for fact_index in query_fact_indices:
        query = queries[int(fact_index)]
        original.extend(int(value) for value in query)
        swapped.extend(int(value) for value in query)
        answer_offsets.append(len(original))
        original.append(int(original_values[int(fact_index)]))
        swapped.append(int(swapped_values[int(fact_index)]))
        original.extend(int(value) for value in newline)
        swapped.extend(int(value) for value in newline)
    if len(original) != len(swapped):
        raise RuntimeError("A/B query-block geometry drift")
    return (
        np.asarray(original, dtype=np.uint32),
        np.asarray(swapped, dtype=np.uint32),
        answer_offsets,
    )


def _place_sources(
    *,
    originals: np.ndarray,
    swapped: np.ndarray,
    occupied: np.ndarray,
    source_specs: Sequence[SourceSpec],
    query_fact_indices: Sequence[int],
    answer_positions: Sequence[int],
    requested_distances: Sequence[int],
    rng: random.Random,
    prefix_limit: int,
) -> tuple[list[int], list[int], list[int]]:
    query_lookup = {
        int(fact): (int(answer), int(distance))
        for fact, answer, distance in zip(
            query_fact_indices, answer_positions, requested_distances
        )
    }
    ordering = list(range(len(source_specs)))
    ordering.sort(
        key=lambda index: (
            0 if index in query_lookup else 1,
            -query_lookup.get(index, (0, 0))[1],
        )
    )
    starts = [-1] * len(source_specs)
    value_positions = [-1] * len(source_specs)
    actual_query_distances: dict[int, int] = {}
    for fact_index in ordering:
        spec = source_specs[fact_index]
        if fact_index in query_lookup:
            answer, distance = query_lookup[fact_index]
            preferred = answer - distance - int(spec.value_offset)
        else:
            maximum = max(0, int(prefix_limit) - len(spec.original))
            preferred = rng.randint(8, max(8, maximum))
        start = place_phrase(
            originals,
            occupied,
            spec.original.astype(np.uint32, copy=False),
            preferred,
        )
        if start + len(spec.swapped) > prefix_limit:
            raise RuntimeError("source phrase escaped the source prefix")
        swapped[start : start + len(spec.swapped)] = spec.swapped
        starts[fact_index] = int(start)
        value_position = int(start + spec.value_offset)
        value_positions[fact_index] = value_position
        if fact_index in query_lookup:
            answer, requested = query_lookup[fact_index]
            actual = int(answer - value_position)
            if abs(actual - requested) > 96:
                raise RuntimeError(
                    "source collision displaced a queried value too far"
                )
            actual_query_distances[fact_index] = actual
    return (
        starts,
        value_positions,
        [actual_query_distances[int(index)] for index in query_fact_indices],
    )


def build_view(
    *,
    tokenizer: Any,
    neutral: np.ndarray,
    keys: Sequence[str],
    original_values: Sequence[int],
    swapped_values: Sequence[int],
    templates: Sequence[tuple[str, str, str, str]],
    query_fact_indices: Sequence[int],
    requested_distances: Sequence[int] | None,
    rng: random.Random,
    include_deleted: bool,
    query_window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    length = len(neutral)
    source_specs: list[SourceSpec] = []
    queries: list[np.ndarray] = []
    for key, value_a, value_b, template in zip(
        keys, original_values, swapped_values, templates
    ):
        source_a, offset_a, query_a = phrase_tokens(
            tokenizer,
            key=key,
            value_id=int(value_a),
            template=template,
        )
        source_b, offset_b, query_b = phrase_tokens(
            tokenizer,
            key=key,
            value_id=int(value_b),
            template=template,
        )
        if (
            len(source_a) != len(source_b)
            or int(offset_a) != int(offset_b)
            or not np.array_equal(query_a, query_b)
        ):
            raise RuntimeError("value permutation changed source/query geometry")
        source_specs.append(
            SourceSpec(
                original=source_a,
                swapped=source_b,
                value_offset=int(offset_a),
                template_id=template[3],
            )
        )
        queries.append(query_a)

    query_a, query_b, offsets = build_query_block(
        tokenizer=tokenizer,
        query_fact_indices=query_fact_indices,
        queries=queries,
        original_values=original_values,
        swapped_values=swapped_values,
    )
    if len(query_a) > 1_024:
        raise RuntimeError("query block exceeds the frozen tail budget")
    lower = max(int(query_window[0]), 1_024)
    upper = min(int(query_window[1]), length - len(query_a) - 8)
    if upper < lower:
        raise RuntimeError("invalid query-position window")
    query_start = rng.randint(lower, upper)
    answer_positions = [query_start + int(offset) for offset in offsets]

    sourced = neutral.astype(np.uint32, copy=True)
    permuted = neutral.astype(np.uint32, copy=True)
    deleted = neutral.astype(np.uint32, copy=True)
    sourced[query_start : query_start + len(query_a)] = query_a
    permuted[query_start : query_start + len(query_b)] = query_b
    deleted[query_start : query_start + len(query_a)] = query_a
    occupied = np.zeros(length, dtype=np.bool_)
    occupied[0] = True
    occupied[query_start : query_start + len(query_a)] = True

    if requested_distances is None:
        short_upper = min(query_start - 32, SHORT_LENGTH - 256)
        if short_upper <= 256:
            raise RuntimeError("short source prefix is too small")
        requested_distances = [
            rng.randint(256, max(257, short_upper - 64))
            for _ in query_fact_indices
        ]
    if len(requested_distances) != len(query_fact_indices):
        raise RuntimeError("distance/query count drift")
    starts, value_positions, actual_distances = _place_sources(
        originals=sourced,
        swapped=permuted,
        occupied=occupied,
        source_specs=source_specs,
        query_fact_indices=query_fact_indices,
        answer_positions=answer_positions,
        requested_distances=requested_distances,
        rng=rng,
        prefix_limit=query_start,
    )

    variants = [sourced, permuted]
    target_rows = [
        [int(original_values[index]) for index in query_fact_indices],
        [int(swapped_values[index]) for index in query_fact_indices],
    ]
    names = ["association_a", "association_b"]
    if include_deleted:
        variants.append(deleted)
        target_rows.append(
            [int(original_values[index]) for index in query_fact_indices]
        )
        names.append("source_deleted_a")
    return (
        np.stack(variants).astype(np.uint32, copy=False),
        np.asarray(answer_positions, dtype=np.int32),
        np.asarray(target_rows, dtype=np.int32),
        {
            "variant_names": names,
            "query_start": int(query_start),
            "query_fact_indices": [int(value) for value in query_fact_indices],
            "answer_positions": [int(value) for value in answer_positions],
            "source_starts": starts,
            "source_value_positions": value_positions,
            "queried_value_positions": [
                int(value_positions[index]) for index in query_fact_indices
            ],
            "actual_value_to_answer_distances": actual_distances,
            "template_ids": [spec.template_id for spec in source_specs],
        },
    )


def build_row(
    *,
    tokenizer: Any,
    documents: np.ndarray,
    document_rows: Sequence[int],
    values: Sequence[tuple[str, int]],
    length: int,
    distance_bands: Sequence[tuple[int, int]],
    rng: random.Random,
    namespace: str,
    heldout_template: bool,
    include_deleted: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    selected = rng.sample(list(values), FACTS)
    original_values = [int(token_id) for _, token_id in selected]
    swapped_values = list(
        fixed_point_free_permutation(original_values, rng)
    )
    keys = [random_key(rng, namespace) for _ in range(FACTS)]
    templates = [
        HELDOUT_TEMPLATE
        if heldout_template
        else TRAIN_TEMPLATES[rng.randrange(len(TRAIN_TEMPLATES))]
        for _ in range(FACTS)
    ]
    query_indices = rng.sample(range(FACTS), QUERIES)
    rng.shuffle(query_indices)
    requested = [
        rng.randint(
            int(distance_bands[index % len(distance_bands)][0]),
            int(distance_bands[index % len(distance_bands)][1]),
        )
        for index in range(QUERIES)
    ]

    long_neutral, long_documents = make_filler(
        documents, document_rows, int(length), rng
    )
    long_view = build_view(
        tokenizer=tokenizer,
        neutral=long_neutral,
        keys=keys,
        original_values=original_values,
        swapped_values=swapped_values,
        templates=templates,
        query_fact_indices=query_indices,
        requested_distances=requested,
        rng=rng,
        include_deleted=include_deleted,
        query_window=(
            max(2_048, int(length) - 1_600),
            int(length) - 256,
        ),
    )

    short_neutral, short_documents = make_filler(
        documents, document_rows, SHORT_LENGTH, rng
    )
    short_view = build_view(
        tokenizer=tokenizer,
        neutral=short_neutral,
        keys=keys,
        original_values=original_values,
        swapped_values=swapped_values,
        templates=templates,
        query_fact_indices=query_indices,
        requested_distances=None,
        rng=rng,
        include_deleted=False,
        query_window=(3_200, 3_800),
    )
    long_ids, long_positions, long_targets, long_meta = long_view
    short_ids, short_positions, short_targets, short_meta = short_view
    if not np.array_equal(long_targets[:2], short_targets):
        raise RuntimeError("short/long target identity drift")
    return (
        long_ids,
        long_positions,
        long_targets,
        short_ids,
        short_positions,
        short_targets,
        {
            "keys": keys,
            "original_value_token_ids": original_values,
            "permuted_value_token_ids": swapped_values,
            "long_document_rows": long_documents,
            "short_document_rows": short_documents,
            "long": long_meta,
            "short": short_meta,
        },
    )


def prepare_set(
    *,
    output: Path,
    tokenizer: Any,
    documents: np.ndarray,
    document_rows: Sequence[int],
    values: Sequence[tuple[str, int]],
    rows: int,
    length: int,
    distance_bands: Sequence[tuple[int, int]],
    seed: int,
    namespace: str,
    heldout_template: bool,
    include_deleted: bool,
    purpose: str,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    long_variants = 3 if include_deleted else 2
    long_ids = np.lib.format.open_memmap(
        output / "long_input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(int(rows), long_variants, int(length)),
    )
    long_positions = np.lib.format.open_memmap(
        output / "long_answer_positions.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(rows), QUERIES),
    )
    long_targets = np.lib.format.open_memmap(
        output / "long_target_ids.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(rows), long_variants, QUERIES),
    )
    short_ids = np.lib.format.open_memmap(
        output / "short_input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(int(rows), 2, SHORT_LENGTH),
    )
    short_positions = np.lib.format.open_memmap(
        output / "short_answer_positions.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(rows), QUERIES),
    )
    short_targets = np.lib.format.open_memmap(
        output / "short_target_ids.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(rows), 2, QUERIES),
    )
    rng = random.Random(int(seed))
    rows_path = output / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row_index in range(int(rows)):
            row = build_row(
                tokenizer=tokenizer,
                documents=documents,
                document_rows=document_rows,
                values=values,
                length=int(length),
                distance_bands=distance_bands,
                rng=rng,
                namespace=f"{namespace}{row_index:05d}-",
                heldout_template=heldout_template,
                include_deleted=include_deleted,
            )
            (
                row_long_ids,
                row_long_positions,
                row_long_targets,
                row_short_ids,
                row_short_positions,
                row_short_targets,
                metadata,
            ) = row
            long_ids[row_index] = row_long_ids
            long_positions[row_index] = row_long_positions
            long_targets[row_index] = row_long_targets
            short_ids[row_index] = row_short_ids
            short_positions[row_index] = row_short_positions
            short_targets[row_index] = row_short_targets
            metadata["row"] = row_index
            handle.write(
                json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    for array in (
        long_ids,
        long_positions,
        long_targets,
        short_ids,
        short_positions,
        short_targets,
    ):
        array.flush()
    paths = [
        output / "long_input_ids.npy",
        output / "long_answer_positions.npy",
        output / "long_target_ids.npy",
        output / "short_input_ids.npy",
        output / "short_answer_positions.npy",
        output / "short_target_ids.npy",
        rows_path,
    ]
    manifest = {
        "format_version": 1,
        "status": "OLMO2_CSBD_SET_PREPARED",
        "purpose": purpose,
        "rows": int(rows),
        "long_length": int(length),
        "short_length": SHORT_LENGTH,
        "long_variants": (
            ["association_a", "association_b", "source_deleted_a"]
            if include_deleted
            else ["association_a", "association_b"]
        ),
        "short_variants": ["association_a", "association_b"],
        "facts_per_sequence": FACTS,
        "queries_per_sequence": QUERIES,
        "heldout_template": bool(heldout_template),
        "distance_bands": [list(value) for value in distance_bands],
        "document_rows": [int(value) for value in document_rows],
        "seed": int(seed),
        "files": file_manifest(paths, output),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    checkpoint_receipt = verify_checkpoint(checkpoint)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    values = broad_vocabulary_train_values(
        tokenizer,
        excluded_token_ids=set(),
        count=TRAIN_VALUE_COUNT + EVAL_VALUE_COUNT,
        seed=int(args.seed) + 701,
    )
    train_values = values[:TRAIN_VALUE_COUNT]
    eval_values = values[TRAIN_VALUE_COUNT:]
    if {
        int(token_id) for _, token_id in train_values
    } & {
        int(token_id) for _, token_id in eval_values
    }:
        raise RuntimeError("train/evaluation value pools overlap")

    background_path = args.background.resolve()
    documents = np.load(background_path, mmap_mode="r", allow_pickle=False)
    if tuple(documents.shape) != (128, 16_384):
        raise RuntimeError(
            f"expected 128x16384 background, got {documents.shape}"
        )
    partitions = {
        "train": list(range(0, 80)),
        "validation": list(range(80, 96)),
        "final_test": list(range(96, 112)),
        "extrapolation": list(range(112, 128)),
    }
    flat = [row for values_ in partitions.values() for row in values_]
    if len(flat) != len(set(flat)):
        raise RuntimeError("background document partitions overlap")

    scale = float(args.row_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("row-scale must be in (0, 1]")
    specs: list[dict[str, Any]] = []
    for index, (name, band) in enumerate(TRAIN_DISTANCE_BANDS.items()):
        specs.append(
            {
                "name": name,
                "rows": max(1, round(TRAIN_ROWS_PER_BAND * scale)),
                "length": LONG_LENGTH,
                "bands": (band,),
                "documents": partitions["train"],
                "values": train_values,
                "heldout": False,
                "deleted": False,
                "purpose": f"student_training_{name}",
                "seed": int(args.seed) + 10_000 * (index + 1),
            }
        )
    specs.extend(
        [
            {
                "name": "validation_16k",
                "rows": max(1, round(VALIDATION_ROWS * scale)),
                "length": LONG_LENGTH,
                "bands": HELDOUT_DISTANCE_BANDS,
                "documents": partitions["validation"],
                "values": eval_values,
                "heldout": True,
                "deleted": True,
                "purpose": "student_calibration_heldout_values_template_distance",
                "seed": int(args.seed) + 40_000,
            },
            {
                "name": "final_test_16k",
                "rows": max(1, round(FINAL_TEST_ROWS * scale)),
                "length": LONG_LENGTH,
                "bands": HELDOUT_DISTANCE_BANDS,
                "documents": partitions["final_test"],
                "values": eval_values,
                "heldout": True,
                "deleted": True,
                "purpose": "final_test_do_not_monitor",
                "seed": int(args.seed) + 50_000,
            },
        ]
    )
    for index, length in enumerate(EXTRAPOLATION_LENGTHS):
        specs.append(
            {
                "name": f"extrapolation_{length // 1024}k",
                "rows": max(1, round(EXTRAPOLATION_ROWS * scale)),
                "length": int(length),
                "bands": EXTRAPOLATION_DISTANCE_BANDS[int(length)],
                "documents": partitions["extrapolation"],
                "values": eval_values,
                "heldout": True,
                "deleted": True,
                "purpose": f"extrapolation_{length}_heldout",
                "seed": int(args.seed) + 60_000 + index * 10_000,
            }
        )

    entries = []
    for spec in specs:
        manifest = prepare_set(
            output=output / spec["name"],
            tokenizer=tokenizer,
            documents=documents,
            document_rows=spec["documents"],
            values=spec["values"],
            rows=spec["rows"],
            length=spec["length"],
            distance_bands=spec["bands"],
            seed=spec["seed"],
            namespace=f"{spec['name']}-",
            heldout_template=spec["heldout"],
            include_deleted=spec["deleted"],
            purpose=spec["purpose"],
        )
        entries.append(
            {
                "name": spec["name"],
                "purpose": spec["purpose"],
                "relative_path": spec["name"],
                "manifest_sha256": sha256_file(
                    output / spec["name"] / "manifest.json"
                ),
                "rows": manifest["rows"],
                "long_length": manifest["long_length"],
            }
        )

    collection = {
        "format_version": 1,
        "status": "OLMO2_CSBD_COLLECTION_PREPARED",
        "checkpoint": checkpoint_receipt,
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "background": {
            "path_name": background_path.name,
            "shape": list(documents.shape),
            "dtype": str(documents.dtype),
            "sha256": sha256_file(background_path),
            "partition_index_sha256": {
                name: sha256_array(np.asarray(rows, dtype=np.int32))
                for name, rows in partitions.items()
            },
        },
        "facts_per_sequence": FACTS,
        "queries_per_sequence": QUERIES,
        "train_value_token_ids": [
            int(token_id) for _, token_id in train_values
        ],
        "eval_value_token_ids": [
            int(token_id) for _, token_id in eval_values
        ],
        "value_pools_disjoint": True,
        "sets": entries,
        "seed": int(args.seed),
        "row_scale": scale,
    }
    atomic_json(output / "collection_manifest.json", collection)
    return collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--row-scale",
        type=float,
        default=1.0,
        help="Use <1 only for CPU smoke preparation.",
    )
    return parser.parse_args()


def main() -> None:
    result = prepare(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "sets": result["sets"],
                "collection_manifest": "collection_manifest.json",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

