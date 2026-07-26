#!/usr/bin/env python3
"""Freeze continuous-position training and factorial causal evaluation sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    one_token_values,
    select_document_rows,
)

from .causal_data import save_probe_set, verify_probe_set
from .prepare_data import atomic_json, sha256_file
from .protocol import build_source_probe_set, position_coverage


TRAIN_POSITION_ANCHORS = (0.2, 0.5, 0.8)
EVAL_POSITION_ANCHORS = (0.1, 0.35, 0.65, 0.9)
DENSE_POSITION_ANCHORS = tuple(
    0.05 + 0.1 * index for index in range(10)
)


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
    if (
        documents.dtype != np.uint32
        or documents.ndim != 2
        or documents.shape[1] != 16_384
    ):
        raise RuntimeError("probe background shape/dtype drift")
    if len(metadata) != len(documents):
        raise RuntimeError("probe background metadata row-count drift")
    return documents, metadata


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    checkpoint = args.checkpoint.resolve()
    background_dir = args.background_dir.resolve()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    train_values, eval_values = one_token_values(tokenizer)
    documents, metadata = load_background(background_dir)
    train_document_rows = select_document_rows(
        metadata, split="validation"
    )
    eval_document_rows = select_document_rows(metadata, split="test")
    output_root.mkdir(parents=True)
    entries: list[dict[str, Any]] = []

    train_data = build_source_probe_set(
        tokenizer=tokenizer,
        documents=documents,
        document_metadata=metadata,
        document_rows=train_document_rows,
        values=train_values,
        length=16_384,
        count=int(args.training_examples),
        seed=int(args.seed) + 1,
        source_fractions=None,
        distractor_counts=(0, 4, 8),
        template_mode="train_compositional",
        namespace="TRC-",
    )
    train_relative = Path("train_continuous_compositional")
    train_manifest = save_probe_set(
        output_root / train_relative,
        train_data,
        receipt={
            "purpose": "training",
            "position_policy": "continuous_low_discrepancy_0.05_0.95",
            "template_policy": "compositional_8x8",
            "value_pool": "registered_train",
            "document_split": "validation",
            "distractor_counts": [0, 4, 8],
            "position_coverage": position_coverage(train_data),
        },
    )
    entries.append(
        {
            "name": "train_continuous_compositional",
            "relative_path": str(train_relative),
            "purpose": "training",
            "dataset_sha256": train_manifest["dataset_sha256"],
        }
    )

    canary_data = build_source_probe_set(
        tokenizer=tokenizer,
        documents=documents,
        document_metadata=metadata,
        document_rows=eval_document_rows,
        values=eval_values,
        length=16_384,
        count=int(args.canary_examples),
        seed=int(args.seed) + 2,
        source_fractions=EVAL_POSITION_ANCHORS,
        distractor_counts=(0, 8),
        template_mode="eval_unseen",
        namespace="EVC-",
    )
    canary_relative = Path("canary_full_heldout")
    canary_manifest = save_probe_set(
        output_root / canary_relative,
        canary_data,
        receipt={
            "purpose": "training_canary",
            "position_policy": list(EVAL_POSITION_ANCHORS),
            "template_policy": "eval_unseen",
            "value_pool": "registered_eval",
            "document_split": "test",
            "distractor_counts": [0, 8],
        },
    )
    entries.append(
        {
            "name": "canary_full_heldout",
            "relative_path": str(canary_relative),
            "purpose": "training_canary",
            "dataset_sha256": canary_manifest["dataset_sha256"],
        }
    )

    cell_specs = (
        (
            "train_position_train_template",
            TRAIN_POSITION_ANCHORS,
            "train_anchor",
        ),
        (
            "train_position_eval_template",
            TRAIN_POSITION_ANCHORS,
            "eval_unseen",
        ),
        (
            "eval_position_train_template",
            EVAL_POSITION_ANCHORS,
            "train_anchor",
        ),
        (
            "eval_position_eval_template",
            EVAL_POSITION_ANCHORS,
            "eval_unseen",
        ),
    )
    for cell_index, (name, positions, template_mode) in enumerate(cell_specs):
        data = build_source_probe_set(
            tokenizer=tokenizer,
            documents=documents,
            document_metadata=metadata,
            document_rows=eval_document_rows,
            values=eval_values,
            length=16_384,
            count=int(args.examples_per_cell),
            seed=int(args.seed) + 10_000 + cell_index,
            source_fractions=positions,
            distractor_counts=(0, 8),
            template_mode=template_mode,
            namespace="EVF-",
        )
        relative = Path("eval_context_cells") / name
        manifest = save_probe_set(
            output_root / relative,
            data,
            receipt={
                "purpose": "context_factorial_evaluation",
                "cell": name,
                "position_policy": list(positions),
                "template_policy": template_mode,
                "value_pool": "registered_eval",
                "document_split": "test",
                "distractor_counts": [0, 8],
                "position_coverage": position_coverage(data),
            },
        )
        entries.append(
            {
                "name": name,
                "relative_path": str(relative),
                "purpose": "context_factorial_evaluation",
                "dataset_sha256": manifest["dataset_sha256"],
            }
        )

    dense_count = (
        len(DENSE_POSITION_ANCHORS)
        * 2
        * int(args.dense_examples_per_cell)
    )
    dense_data = build_source_probe_set(
        tokenizer=tokenizer,
        documents=documents,
        document_metadata=metadata,
        document_rows=eval_document_rows,
        values=eval_values,
        length=16_384,
        count=dense_count,
        seed=int(args.seed) + 20_000,
        source_fractions=DENSE_POSITION_ANCHORS,
        distractor_counts=(0, 8),
        template_mode="eval_unseen",
        namespace="EVD-",
    )
    dense_relative = Path("eval_dense_positions")
    dense_manifest = save_probe_set(
        output_root / dense_relative,
        dense_data,
        receipt={
            "purpose": "dense_position_evaluation",
            "position_policy": list(DENSE_POSITION_ANCHORS),
            "template_policy": "eval_unseen",
            "value_pool": "registered_eval",
            "document_split": "test",
            "distractor_counts": [0, 8],
            "examples_per_position_distractor_cell": int(
                args.dense_examples_per_cell
            ),
            "position_coverage": position_coverage(dense_data),
        },
    )
    entries.append(
        {
            "name": "eval_dense_positions",
            "relative_path": str(dense_relative),
            "purpose": "dense_position_evaluation",
            "dataset_sha256": dense_manifest["dataset_sha256"],
        }
    )

    collection = {
        "format_version": 1,
        "status": "OLMO2_LORA_CAUSAL_DATA_PREPARED",
        "seed": int(args.seed),
        "checkpoint": checkpoint.name,
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "background_manifest_sha256": sha256_file(
            background_dir / "manifest.json"
        ),
        "training_examples": int(args.training_examples),
        "canary_examples": int(args.canary_examples),
        "examples_per_context_cell": int(args.examples_per_cell),
        "dense_examples_per_cell": int(args.dense_examples_per_cell),
        "registered_train_value_token_ids": [
            int(token_id) for _, token_id in train_values
        ],
        "registered_eval_value_token_ids": [
            int(token_id) for _, token_id in eval_values
        ],
        "value_token_ids_disjoint": not bool(
            {token_id for _, token_id in train_values}
            & {token_id for _, token_id in eval_values}
        ),
        "sets": entries,
    }
    atomic_json(output_root / "collection_manifest.json", collection)
    return verify_collection(output_root)


def verify_collection(output_root: Path) -> dict[str, Any]:
    collection_path = output_root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    verified = []
    seen_digests: set[str] = set()
    for entry in collection["sets"]:
        result = verify_probe_set(output_root / entry["relative_path"])
        if result["dataset_sha256"] != entry["dataset_sha256"]:
            raise RuntimeError("causal collection digest mismatch")
        if result["dataset_sha256"] in seen_digests:
            raise RuntimeError("duplicate causal dataset digest")
        seen_digests.add(result["dataset_sha256"])
        verified.append(
            {
                "name": entry["name"],
                "count": result["count"],
                "dataset_sha256": result["dataset_sha256"],
            }
        )
    if not bool(collection["value_token_ids_disjoint"]):
        raise RuntimeError("training/evaluation value pools overlap")
    return {
        "status": "OLMO2_LORA_CAUSAL_DATA_VERIFIED",
        "collection_sha256": sha256_file(collection_path),
        "sets": verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--background-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-examples", type=int, default=2_048)
    parser.add_argument("--canary-examples", type=int, default=24)
    parser.add_argument("--examples-per-cell", type=int, default=72)
    parser.add_argument(
        "--dense-examples-per-cell", type=int, default=8
    )
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_collection(args.output_root.resolve()),
                indent=2,
            )
        )
        return
    if args.checkpoint is None or args.background_dir is None:
        raise ValueError("checkpoint and background directory are required")
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
