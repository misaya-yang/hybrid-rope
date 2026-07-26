#!/usr/bin/env python3
"""Build one fixed 4K VT/CWE/FWE/QA training view with replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    row_sha256,
)

from .prepare_data import atomic_json, sha256_file


LENGTH = 4_096
TASKS = ("vt", "cwe", "fwe", "qa_1", "qa_2")
STATUS = "OLMO2_4K_RULER_SYNTHETIC_MIX_PREPARED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--routing-train", type=Path, required=True)
    parser.add_argument("--natural-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-train-per-task", type=int, default=96)
    parser.add_argument("--synthetic-validation-per-task", type=int, default=4)
    parser.add_argument("--routing-pairs", type=int, default=64)
    parser.add_argument("--natural-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_420_726)
    return parser.parse_args()


def canonical_target(task: str, outputs: list[Any]) -> str:
    values = [str(value) for value in outputs]
    if not values:
        raise RuntimeError(f"{task} row has no outputs")
    if task == "cwe":
        return " " + " ".join(
            f"{index + 1}. {value}"
            for index, value in enumerate(values)
        )
    if task in {"vt", "fwe"}:
        return " " + ", ".join(values)
    if task in {"qa_1", "qa_2"}:
        return " " + values[0]
    raise ValueError(task)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    synthetic_root = args.synthetic_root.resolve()
    eval_root = args.eval_root.resolve()
    routing_train = args.routing_train.resolve()
    natural_view = args.natural_view.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if (
        int(args.synthetic_train_per_task)
        + int(args.synthetic_validation_per_task)
        != 100
    ):
        raise ValueError("the prepared source has exactly 100 rows per task")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires EOS and PAD ids")

    eval_hashes: set[str] = set()
    for path in eval_root.rglob("*.jsonl"):
        for row in load_jsonl(path):
            eval_hashes.add(row_sha256(row))

    synthetic_rows: list[tuple[str, dict[str, Any], int]] = []
    train_hashes: set[str] = set()
    for task in TASKS:
        path = synthetic_root / "L4096" / task / "test.jsonl"
        rows = load_jsonl(path)
        if len(rows) != 100:
            raise RuntimeError(f"{task} source row-count drift")
        for index, row in enumerate(rows):
            digest = row_sha256(row)
            if digest in eval_hashes:
                raise RuntimeError(
                    f"synthetic training/evaluation row overlap: {task}"
                )
            if digest in train_hashes:
                raise RuntimeError(f"duplicate synthetic row: {task}")
            train_hashes.add(digest)
            split = (
                0
                if index < int(args.synthetic_train_per_task)
                else 1
            )
            synthetic_rows.append((task, row, split))

    routing_ids = np.load(
        routing_train / "input_ids.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    routing_labels = np.load(
        routing_train / "labels.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        routing_ids.shape != routing_labels.shape
        or routing_ids.ndim != 3
        or routing_ids.shape[1:] != (2, LENGTH)
    ):
        raise RuntimeError("routing-pair storage-shape drift")

    natural_ids = np.load(
        natural_view / "input_ids.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    natural_mask = np.load(
        natural_view / "assistant_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    natural_lengths = np.load(
        natural_view / "lengths.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    natural_split = np.load(
        natural_view / "split.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        natural_ids.shape != natural_mask.shape
        or natural_ids.shape[1] != LENGTH
    ):
        raise RuntimeError("natural replay storage-shape drift")

    generator = np.random.default_rng(int(args.seed))
    routing_pair_indices = generator.choice(
        routing_ids.shape[0],
        size=int(args.routing_pairs),
        replace=False,
    )
    natural_candidates = np.flatnonzero(natural_split == 0)
    natural_indices = generator.choice(
        natural_candidates,
        size=int(args.natural_rows),
        replace=False,
    )

    synthetic_count = len(synthetic_rows)
    routing_count = 2 * len(routing_pair_indices)
    natural_count = len(natural_indices)
    total = synthetic_count + routing_count + natural_count
    input_ids = np.full(
        (total, LENGTH),
        int(tokenizer.pad_token_id),
        dtype=np.uint32,
    )
    assistant_mask = np.zeros((total, LENGTH), dtype=np.uint8)
    lengths = np.zeros(total, dtype=np.int32)
    split = np.zeros(total, dtype=np.uint8)
    metadata: list[dict[str, Any]] = []

    cursor = 0
    for task, row, row_split in synthetic_rows:
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["input"])}],
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            str(row.get("answer_prefix", "")),
            add_special_tokens=False,
        ).input_ids
        target = canonical_target(task, list(row["outputs"]))
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
        ).input_ids + [int(tokenizer.eos_token_id)]
        combined = list(chat_ids) + list(prefix_ids) + target_ids
        if len(combined) > LENGTH:
            raise RuntimeError(
                f"{task} canonical answer exceeds 4K: {len(combined)}"
            )
        input_ids[cursor, : len(combined)] = np.asarray(
            combined, dtype=np.uint32
        )
        assistant_mask[cursor, len(chat_ids) + len(prefix_ids) : len(combined)] = 1
        lengths[cursor] = len(combined)
        split[cursor] = int(row_split)
        metadata.append(
            {
                "row": cursor,
                "source": "official_ruler_synthetic",
                "task": task,
                "source_row_index": int(row["index"]),
                "source_row_sha256": row_sha256(row),
                "split": "train" if row_split == 0 else "validation",
                "canonical_target": target,
            }
        )
        cursor += 1

    for pair_index in routing_pair_indices:
        for variant in range(2):
            row_ids = np.asarray(
                routing_ids[int(pair_index), variant],
                dtype=np.uint32,
            )
            row_labels = np.asarray(
                routing_labels[int(pair_index), variant],
                dtype=np.int32,
            )
            input_ids[cursor] = row_ids
            assistant_mask[cursor] = (row_labels != -100).astype(
                np.uint8
            )
            lengths[cursor] = LENGTH
            metadata.append(
                {
                    "row": cursor,
                    "source": "counterfactual_niah_replay",
                    "source_pair_index": int(pair_index),
                    "variant": int(variant),
                    "split": "train",
                }
            )
            cursor += 1

    for source_index in natural_indices:
        source_index = int(source_index)
        input_ids[cursor] = np.asarray(
            natural_ids[source_index], dtype=np.uint32
        )
        assistant_mask[cursor] = np.asarray(
            natural_mask[source_index], dtype=np.uint8
        )
        lengths[cursor] = int(natural_lengths[source_index])
        metadata.append(
            {
                "row": cursor,
                "source": "longalign_assistant_replay",
                "source_row_index": source_index,
                "split": "train",
            }
        )
        cursor += 1
    if cursor != total:
        raise RuntimeError("prepared-row cursor drift")
    if int((split == 0).sum()) % 8 != 0:
        raise RuntimeError("training split must divide global batch 8")
    if np.any(assistant_mask.sum(axis=1) == 0):
        raise RuntimeError("prepared row without a supervised token")

    output.mkdir(parents=True)
    np.save(output / "input_ids.npy", input_ids, allow_pickle=False)
    np.save(
        output / "assistant_mask.npy",
        assistant_mask,
        allow_pickle=False,
    )
    np.save(output / "lengths.npy", lengths, allow_pickle=False)
    np.save(output / "split.npy", split, allow_pickle=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "status": STATUS,
        "format_version": 1,
        "shape": [total, LENGTH],
        "files": files,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "pad_token_id": int(tokenizer.pad_token_id),
        "seed": int(args.seed),
        "training_rows": int((split == 0).sum()),
        "validation_rows": int((split == 1).sum()),
        "mixture": {
            "synthetic_train": int(
                len(TASKS) * args.synthetic_train_per_task
            ),
            "synthetic_validation": int(
                len(TASKS) * args.synthetic_validation_per_task
            ),
            "counterfactual_niah_replay": routing_count,
            "longalign_assistant_replay": natural_count,
        },
        "synthetic_tasks": list(TASKS),
        "synthetic_source_manifest_sha256": sha256_file(
            synthetic_root / "manifest.json"
        ),
        "evaluation_manifest_sha256": sha256_file(
            eval_root / "manifest.json"
        ),
        "exact_train_eval_row_overlap": 0,
        "metric_boundary": (
            "Official RULER generator-family task supervision at physical "
            "4K; held-out evaluation uses distinct generated rows. This is "
            "benchmark-family-matched adaptation, not unseen-task transfer."
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
