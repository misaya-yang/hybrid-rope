#!/usr/bin/env python3
"""Prepare dense multi-binding data whose every sequence is at most 4K."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

from .prepare_data import atomic_json, sha256_file


LENGTH = 4_096
SLOTS = 8
TRAIN_VALUE_COUNT = 512
EVAL_VALUE_COUNT = 128
TRAIN_DISTANCE_RANGES = (
    (256, 1_099),
    (1_301, 2_699),
    (2_901, 3_299),
    (3_601, 3_880),
)
HELDOUT_DISTANCE_RANGES = (
    (1_100, 1_300),
    (2_700, 2_900),
    (3_300, 3_600),
)
B1_DISTANCE_RANGE = (256, 2_000)

TRAIN_TEMPLATES = (
    (
        "\nRecord {key} stores",
        ".\n",
        "\nValue for {key}:",
        "record_value",
    ),
    (
        "\nEntry [{key}] contains",
        ".\n",
        "\nReturn the value in [{key}]:",
        "entry_return",
    ),
    (
        "\nLookup({key}) gives",
        ".\n",
        "\nResolve Lookup({key}):",
        "lookup_resolve",
    ),
)
EVAL_TEMPLATE = (
    "\nArchive identifier <{key}> maps to",
    ".\n",
    "\nWhich value belongs to archive identifier <{key}>?",
    "archive_which",
)


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    for offset in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[offset : offset + 8 * 1024 * 1024])
    return digest.hexdigest()


def load_background(
    background_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    documents = np.load(
        background_dir / "documents_L16384.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = json.loads(
        (background_dir / "documents_L16384.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if documents.ndim != 2 or documents.shape[1] != 16_384:
        raise RuntimeError("16K background shape drift")
    if len(documents) != len(metadata):
        raise RuntimeError("background metadata length drift")
    return documents, metadata


def document_partitions(
    metadata: Sequence[dict[str, Any]],
) -> dict[str, list[int]]:
    validation = [
        int(row["row"]) for row in metadata
        if row["split"] == "validation"
    ]
    test = [
        int(row["row"]) for row in metadata
        if row["split"] == "test"
    ]
    if len(validation) < 16 or len(test) < 16:
        raise RuntimeError("insufficient background rows")
    validation_cut = max(1, int(len(validation) * 0.75))
    test_cut = max(1, len(test) // 2)
    partitions = {
        "train": validation[:validation_cut],
        "calibration": validation[validation_cut:],
        "validation": test[:test_cut],
        "final_test": test[test_cut:],
    }
    if any(not rows for rows in partitions.values()):
        raise RuntimeError("empty document partition")
    flat = [row for rows in partitions.values() for row in rows]
    if len(flat) != len(set(flat)):
        raise RuntimeError("document partitions overlap")
    return partitions


def choose_distance(
    rng: random.Random,
    *,
    profile: str,
) -> int:
    if profile == "b1":
        lower, upper = B1_DISTANCE_RANGE
    elif profile == "b2":
        lower, upper = rng.choice(TRAIN_DISTANCE_RANGES)
    elif profile == "heldout":
        lower, upper = rng.choice(HELDOUT_DISTANCE_RANGES)
    else:
        raise ValueError(f"unknown distance profile {profile!r}")
    margin = min(48, max(0, (int(upper) - int(lower)) // 4))
    return rng.randint(int(lower) + margin, int(upper) - margin)


def distance_matches_profile(distance: int, profile: str) -> bool:
    if profile == "b1":
        ranges = (B1_DISTANCE_RANGE,)
    elif profile == "b2":
        ranges = TRAIN_DISTANCE_RANGES
    elif profile == "heldout":
        ranges = HELDOUT_DISTANCE_RANGES
    else:
        raise ValueError(f"unknown distance profile {profile!r}")
    return any(
        int(lower) <= int(distance) <= int(upper)
        for lower, upper in ranges
    )


def phrase_tokens(
    tokenizer: Any,
    *,
    key: str,
    value_id: int,
    template: tuple[str, str, str, str],
) -> tuple[np.ndarray, int, np.ndarray]:
    source_prefix, source_suffix, query_prefix, _ = template
    prefix = encode(tokenizer, source_prefix.format(key=key))
    suffix = encode(tokenizer, source_suffix)
    source = np.concatenate(
        (prefix, np.asarray([value_id], dtype=np.int64), suffix)
    )
    query = encode(tokenizer, query_prefix.format(key=key))
    return source, len(prefix), query


def build_pair(
    *,
    tokenizer: Any,
    document: np.ndarray,
    crop_start: int,
    value_ids: Sequence[int],
    rng: random.Random,
    template_mode: str,
    distance_profile: str,
    namespace: str,
    include_deleted: bool,
    answer_placeholder_token_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(value_ids) != SLOTS or len(set(value_ids)) != SLOTS:
        raise ValueError("binding row requires unique slot values")
    neutral = np.asarray(
        document[crop_start : crop_start + LENGTH],
        dtype=np.uint32,
    ).copy()
    if neutral.shape != (LENGTH,):
        raise RuntimeError("background crop is shorter than 4K")

    keys = [random_key(rng, namespace) for _ in range(SLOTS)]
    templates = []
    for slot in range(SLOTS):
        if template_mode == "anchor":
            templates.append(TRAIN_TEMPLATES[0])
        elif template_mode == "train_mixed":
            templates.append(
                TRAIN_TEMPLATES[(slot + rng.randrange(3)) % 3]
            )
        elif template_mode == "eval_unseen":
            templates.append(EVAL_TEMPLATE)
        else:
            raise ValueError(f"unknown template mode {template_mode!r}")

    swapped_ids = tuple(value_ids[1:]) + (value_ids[0],)
    source_specs = []
    query_original: list[int] = []
    query_swapped: list[int] = []
    answer_offsets: list[int] = []
    for slot, (key, template) in enumerate(zip(keys, templates)):
        source, value_offset, query = phrase_tokens(
            tokenizer=tokenizer,
            key=key,
            value_id=int(value_ids[slot]),
            template=template,
        )
        swapped_source, swapped_offset, swapped_query = phrase_tokens(
            tokenizer=tokenizer,
            key=key,
            value_id=int(swapped_ids[slot]),
            template=template,
        )
        if (
            len(source) != len(swapped_source)
            or value_offset != swapped_offset
            or not np.array_equal(query, swapped_query)
        ):
            raise RuntimeError("value swap changed token geometry")
        query_original.extend(int(value) for value in query)
        answer_offsets.append(len(query_original))
        query_original.append(int(answer_placeholder_token_id))
        query_original.extend(
            int(value) for value in encode(tokenizer, "\n")
        )
        query_swapped.extend(int(value) for value in query)
        query_swapped.append(int(answer_placeholder_token_id))
        query_swapped.extend(
            int(value) for value in encode(tokenizer, "\n")
        )
        source_specs.append(
            (source, swapped_source, int(value_offset), template[3])
        )

    if len(query_original) != len(query_swapped):
        raise RuntimeError("swapped query block length drift")
    tail_length = len(query_original)
    if tail_length >= 768:
        raise RuntimeError("query block unexpectedly exceeds tail budget")
    tail_start = LENGTH - tail_length
    original = neutral.copy()
    swapped = neutral.copy()
    deleted = neutral.copy()
    original[tail_start:] = np.asarray(
        query_original, dtype=np.uint32
    )
    swapped[tail_start:] = np.asarray(
        query_swapped, dtype=np.uint32
    )
    deleted[tail_start:] = np.asarray(
        query_original, dtype=np.uint32
    )
    occupied = np.zeros(LENGTH, dtype=np.bool_)
    occupied[tail_start:] = True

    source_starts: list[int] = []
    source_value_positions: list[int] = []
    answer_positions = [
        tail_start + offset for offset in answer_offsets
    ]
    actual_distances: list[int] = []
    template_ids: list[str] = []
    for slot, (source, swapped_source, value_offset, template_id) in (
        enumerate(source_specs)
    ):
        requested_distance = choose_distance(
            rng, profile=distance_profile
        )
        preferred = (
            int(answer_positions[slot])
            - requested_distance
            - int(value_offset)
        )
        source_start = place_phrase(
            original,
            occupied,
            source.astype(np.uint32, copy=False),
            preferred,
        )
        swapped[
            source_start : source_start + len(swapped_source)
        ] = swapped_source.astype(np.uint32, copy=False)
        value_position = source_start + int(value_offset)
        source_starts.append(source_start)
        source_value_positions.append(value_position)
        actual_distances.append(
            int(answer_positions[slot] - value_position)
        )
        if not distance_matches_profile(
            actual_distances[-1], distance_profile
        ):
            raise RuntimeError(
                "actual source-to-answer distance escaped its profile"
            )
        template_ids.append(template_id)

    variants = [original, swapped]
    variant_names = ["sourced", "swapped"]
    if include_deleted:
        variants.insert(1, deleted)
        variant_names.insert(1, "deleted")
    inputs = np.stack(variants).astype(np.uint32, copy=False)
    labels = np.full(inputs.shape, -100, dtype=np.int32)
    for variant_index, name in enumerate(variant_names):
        targets = (
            swapped_ids if name == "swapped" else tuple(value_ids)
        )
        for answer_position, token_id in zip(
            answer_positions, targets
        ):
            labels[variant_index, answer_position] = int(token_id)
    return inputs, labels, {
        "keys": keys,
        "value_token_ids": [int(value) for value in value_ids],
        "swapped_value_token_ids": [
            int(value) for value in swapped_ids
        ],
        "source_starts": source_starts,
        "source_value_positions": source_value_positions,
        "answer_positions": answer_positions,
        "actual_value_to_answer_distances": actual_distances,
        "template_ids": template_ids,
        "crop_start": int(crop_start),
        "variant_names": variant_names,
        "answer_placeholder_token_id": int(answer_placeholder_token_id),
    }


def prepare_set(
    *,
    output: Path,
    tokenizer: Any,
    documents: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    document_rows: Sequence[int],
    values: Sequence[tuple[str, int]],
    count: int,
    seed: int,
    template_mode: str,
    distance_profile: str,
    namespace: str,
    include_deleted: bool,
    purpose: str,
    answer_placeholder_token_id: int,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    variants = 3 if include_deleted else 2
    inputs = np.lib.format.open_memmap(
        output / "input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(int(count), variants, LENGTH),
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(count), variants, LENGTH),
    )
    rng = random.Random(int(seed))
    rows_path = output / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row_index in range(int(count)):
            document_row = int(
                document_rows[rng.randrange(len(document_rows))]
            )
            crop_limit = documents.shape[1] - LENGTH
            crop_start = rng.randrange(crop_limit + 1)
            chosen = rng.sample(list(values), SLOTS)
            value_ids = [int(token_id) for _, token_id in chosen]
            row_inputs, row_labels, row_metadata = build_pair(
                tokenizer=tokenizer,
                document=documents[document_row],
                crop_start=crop_start,
                value_ids=value_ids,
                rng=rng,
                template_mode=template_mode,
                distance_profile=distance_profile,
                namespace=namespace,
                include_deleted=include_deleted,
                answer_placeholder_token_id=answer_placeholder_token_id,
            )
            inputs[row_index] = row_inputs
            labels[row_index] = row_labels
            row_metadata.update(
                {
                    "row": row_index,
                    "document_row": document_row,
                    "document_source": str(
                        metadata[document_row]["source"]
                    ),
                }
            )
            handle.write(
                json.dumps(
                    row_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    inputs.flush()
    labels.flush()
    manifest = {
        "format_version": 1,
        "status": "OLMO2_4K_DENSE_BINDING_SET_PREPARED",
        "purpose": purpose,
        "shape": [int(count), variants, LENGTH],
        "variant_names": (
            ["sourced", "deleted", "swapped"]
            if include_deleted
            else ["sourced", "swapped"]
        ),
        "slots_per_sequence": SLOTS,
        "answer_labels_per_sequence": SLOTS,
        "answer_placeholder_token_id": int(answer_placeholder_token_id),
        "maximum_training_length": LENGTH,
        "template_mode": template_mode,
        "distance_profile": distance_profile,
        "value_pool_size": len(values),
        "document_partition_rows": len(document_rows),
        "seed": int(seed),
        "files": {
            "input_ids.npy": sha256_file(output / "input_ids.npy"),
            "labels.npy": sha256_file(output / "labels.npy"),
            "rows.jsonl": sha256_file(rows_path),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    background = args.background_dir.resolve()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    answer_placeholder = encode(tokenizer, "\n")
    if len(answer_placeholder) != 1:
        raise RuntimeError("binding answer placeholder must be one token")
    answer_placeholder_token_id = int(answer_placeholder[0])
    all_values = broad_vocabulary_train_values(
        tokenizer,
        excluded_token_ids={answer_placeholder_token_id},
        count=TRAIN_VALUE_COUNT + EVAL_VALUE_COUNT,
        seed=int(args.seed) + 7_001,
    )
    train_values = all_values[:TRAIN_VALUE_COUNT]
    eval_values = all_values[TRAIN_VALUE_COUNT:]
    if {
        token_id for _, token_id in train_values
    } & {
        token_id for _, token_id in eval_values
    }:
        raise RuntimeError("train/eval value pools overlap")

    documents, metadata = load_background(background)
    partitions = document_partitions(metadata)
    specs = (
        (
            "train_anchor",
            partitions["train"],
            train_values[:128],
            int(args.train_anchor_rows),
            "anchor",
            "b1",
            False,
            "stage_b1_training",
        ),
        (
            "train_broad",
            partitions["train"],
            train_values,
            int(args.train_broad_rows),
            "train_mixed",
            "b2",
            False,
            "stage_b2_training",
        ),
        (
            "calibration_anchor",
            partitions["calibration"],
            train_values[:128],
            int(args.calibration_rows),
            "anchor",
            "b1",
            True,
            "stage_b1_calibration",
        ),
        (
            "validation_ood",
            partitions["validation"],
            eval_values,
            int(args.validation_rows),
            "eval_unseen",
            "heldout",
            True,
            "stage_b2_validation",
        ),
        (
            "final_test",
            partitions["final_test"],
            eval_values,
            int(args.final_test_rows),
            "eval_unseen",
            "heldout",
            True,
            "final_test_do_not_monitor",
        ),
    )
    entries = []
    for index, (
        name,
        rows,
        values,
        count,
        template_mode,
        distance_profile,
        include_deleted,
        purpose,
    ) in enumerate(specs):
        manifest = prepare_set(
            output=output / name,
            tokenizer=tokenizer,
            documents=documents,
            metadata=metadata,
            document_rows=rows,
            values=values,
            count=count,
            seed=int(args.seed) + 10_000 * (index + 1),
            template_mode=template_mode,
            distance_profile=distance_profile,
            namespace=f"B{index}-",
            include_deleted=include_deleted,
            purpose=purpose,
            answer_placeholder_token_id=answer_placeholder_token_id,
        )
        entries.append(
            {
                "name": name,
                "purpose": purpose,
                "relative_path": name,
                "manifest_sha256": sha256_file(
                    output / name / "manifest.json"
                ),
                "shape": manifest["shape"],
            }
        )
    collection = {
        "format_version": 1,
        "status": "OLMO2_4K_DENSE_BINDING_COLLECTION_PREPARED",
        "hard_maximum_training_length": LENGTH,
        "slots_per_sequence": SLOTS,
        "answer_placeholder_token_id": answer_placeholder_token_id,
        "train_value_token_ids": [
            int(token_id) for _, token_id in train_values
        ],
        "eval_value_token_ids": [
            int(token_id) for _, token_id in eval_values
        ],
        "value_pools_disjoint": True,
        "document_partitions": {
            name: {
                "count": len(rows),
                "sha256": sha256_array(
                    np.asarray(rows, dtype=np.int32)
                ),
            }
            for name, rows in partitions.items()
        },
        "train_distance_ranges": [
            list(value) for value in TRAIN_DISTANCE_RANGES
        ],
        "heldout_distance_ranges": [
            list(value) for value in HELDOUT_DISTANCE_RANGES
        ],
        "background_manifest_sha256": sha256_file(
            background / "manifest.json"
        ),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "sets": entries,
    }
    atomic_json(output / "collection_manifest.json", collection)
    return collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-anchor-rows", type=int, default=1_024)
    parser.add_argument("--train-broad-rows", type=int, default=4_096)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--final-test-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    collection = prepare(parse_args())
    print(
        json.dumps(
            {
                "status": collection["status"],
                "hard_maximum_training_length": collection[
                    "hard_maximum_training_length"
                ],
                "sets": collection["sets"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
