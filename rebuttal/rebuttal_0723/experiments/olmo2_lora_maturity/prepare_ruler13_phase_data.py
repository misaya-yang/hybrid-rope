#!/usr/bin/env python3
"""Prepare disjoint physical-4K phase supervision for all 13 RULER tasks."""

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

from .phase_adaptation import FORMAT_VERSION, LENGTH
from .prepare_data import atomic_json, sha256_file
from .evaluate_instruct_ruler_transfer import official_task_score
from .prepare_instruct_ruler_transfer import (
    DEFAULT_TASKS,
    TASK_CONFIGS,
)


STATUS = "OLMO2_4K_RULER13_PHASE_DATA_PREPARED_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-per-task", type=int, default=96)
    parser.add_argument("--validation-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_729)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_target(task: str, outputs: list[Any]) -> str:
    values = [str(value) for value in outputs]
    if not values:
        raise RuntimeError(f"{task} row has no outputs")
    if task == "cwe":
        return " " + " ".join(
            f"{index + 1}. {value}"
            for index, value in enumerate(values)
        )
    if task in {"qa_1", "qa_2"}:
        return " " + values[0]
    return " " + ", ".join(values)


def chat_tokens_and_query_start(
    tokenizer: Any,
    prompt: str,
) -> tuple[list[int], int]:
    final_line_start = prompt.rfind("\n") + 1
    if final_line_start <= 0 or final_line_start >= len(prompt):
        raise RuntimeError("RULER final query boundary is unavailable")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    content_start = rendered.rfind(prompt)
    if content_start < 0:
        raise RuntimeError("RULER prompt is absent from rendered chat")
    query_character = content_start + final_line_start
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    chat_ids = list(encoded.input_ids)
    direct_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    )
    if chat_ids != direct_ids:
        raise RuntimeError("RULER rendered-chat tokenization drift")
    candidates = [
        index
        for index, (start, end) in enumerate(encoded.offset_mapping)
        if int(end) > query_character and int(start) < len(rendered)
    ]
    if not candidates:
        raise RuntimeError("RULER query token boundary is unavailable")
    return chat_ids, int(candidates[0])


def evaluation_hashes(root: Path) -> tuple[set[str], str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: set[str] = set()
    for task in DEFAULT_TASKS:
        entry = manifest["cells"][task][str(LENGTH)]
        path = root / entry["relative_path"]
        rows = load_jsonl(path)
        if len(rows) != int(entry["rows"]):
            raise RuntimeError(f"evaluation row-count drift for {task}")
        hashes.update(row_sha256(row) for row in rows)
    return hashes, sha256_file(manifest_path)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = args.synthetic_root.resolve()
    evaluation = args.evaluation_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    per_task = int(args.training_per_task) + int(
        args.validation_per_task
    )
    if (
        int(args.training_per_task) <= 0
        or int(args.validation_per_task) <= 0
        or per_task != 100
    ):
        raise ValueError("RULER13 source requires a 96/4 split of 100 rows")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires EOS and PAD ids")

    eval_hashes, eval_manifest_sha256 = evaluation_hashes(evaluation)
    rows: list[dict[str, Any]] = []
    train_hashes: set[str] = set()
    task_counts: dict[str, dict[str, int]] = {}
    for task in DEFAULT_TASKS:
        path = source / f"L{LENGTH}" / task / "test.jsonl"
        task_rows = load_jsonl(path)
        if len(task_rows) != per_task:
            raise RuntimeError(f"{task} source row-count drift")
        task_counts[task] = {
            "train": int(args.training_per_task),
            "validation": int(args.validation_per_task),
        }
        for local_index, row in enumerate(task_rows):
            digest = row_sha256(row)
            if digest in eval_hashes:
                raise RuntimeError(
                    f"RULER13 training/evaluation overlap: {task}"
                )
            if digest in train_hashes:
                raise RuntimeError(f"duplicate RULER13 source row: {task}")
            train_hashes.add(digest)
            rows.append(
                {
                    "task": task,
                    "local_index": local_index,
                    "source_path": str(path),
                    "source_row": row,
                    "source_row_sha256": digest,
                    "split": (
                        0
                        if local_index < int(args.training_per_task)
                        else 1
                    ),
                }
            )

    total = len(rows)
    input_ids = np.full(
        (total, LENGTH),
        int(tokenizer.pad_token_id),
        dtype=np.uint32,
    )
    labels = np.full((total, LENGTH), -100, dtype=np.int32)
    query_starts = np.zeros(total, dtype=np.int32)
    active_lengths = np.zeros(total, dtype=np.int32)
    split = np.zeros(total, dtype=np.uint8)
    metadata: list[dict[str, Any]] = []
    for index, entry in enumerate(rows):
        task = str(entry["task"])
        row = entry["source_row"]
        prompt = str(row["input"])
        chat_ids, query_start = chat_tokens_and_query_start(
            tokenizer,
            prompt,
        )
        prefix_ids = list(
            tokenizer(
                str(row.get("answer_prefix", "")),
                add_special_tokens=False,
            ).input_ids
        )
        target = canonical_target(task, list(row["outputs"]))
        self_score = official_task_score(
            target,
            [str(value) for value in row["outputs"]],
            str(TASK_CONFIGS[task]["official_metric"]),
        )
        if self_score != 1.0:
            raise RuntimeError(
                f"{task} canonical target does not satisfy its scorer"
            )
        target_ids = list(
            tokenizer(target, add_special_tokens=False).input_ids
        ) + [int(tokenizer.eos_token_id)]
        answer_start = len(chat_ids) + len(prefix_ids)
        combined = chat_ids + prefix_ids + target_ids
        if len(combined) > LENGTH:
            raise RuntimeError(
                f"{task} canonical answer exceeds 4K: {len(combined)}"
            )
        input_ids[index, : len(combined)] = np.asarray(
            combined, dtype=np.uint32
        )
        labels[index, answer_start : len(combined)] = np.asarray(
            target_ids, dtype=np.int32
        )
        query_starts[index] = query_start
        active_lengths[index] = len(combined)
        split[index] = int(entry["split"])
        metadata.append(
            {
                "row": index,
                "task": task,
                "split": (
                    "train" if int(entry["split"]) == 0 else "validation"
                ),
                "source_row_index": int(row["index"]),
                "source_row_sha256": entry["source_row_sha256"],
                "query_start": query_start,
                "answer_start": answer_start,
                "answer_tokens": len(target_ids) - 1,
                "eos_position": len(combined) - 1,
                "active_length": len(combined),
                "canonical_target": target,
                "canonical_self_score": self_score,
            }
        )

    output.mkdir(parents=True)
    np.save(output / "input_ids.npy", input_ids, allow_pickle=False)
    np.save(output / "labels.npy", labels, allow_pickle=False)
    np.save(
        output / "query_starts.npy",
        query_starts,
        allow_pickle=False,
    )
    np.save(
        output / "active_lengths.npy",
        active_lengths,
        allow_pickle=False,
    )
    np.save(output / "split.npy", split, allow_pickle=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    files: dict[str, dict[str, Any]] = {}
    for name in (
        "input_ids.npy",
        "labels.npy",
        "query_starts.npy",
        "active_lengths.npy",
        "split.npy",
        "rows.jsonl",
    ):
        path = output / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    source_manifest = source / "manifest.json"
    manifest = {
        "status": STATUS,
        "format_version": FORMAT_VERSION,
        "shape": [total, LENGTH],
        "physical_storage_length": LENGTH,
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "labels_only_cover_answer_and_final_eos": True,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "seed": int(args.seed),
        "training_rows": int((split == 0).sum()),
        "validation_rows": int((split == 1).sum()),
        "tasks": list(DEFAULT_TASKS),
        "task_counts": task_counts,
        "source_manifest_sha256": sha256_file(source_manifest),
        "evaluation_manifest_sha256": eval_manifest_sha256,
        "exact_train_eval_row_overlap": 0,
        "phase_curriculum": {
            "physical_sequences": "<=4096",
            "semantic_block": "final_query_and_answer",
            "micro_batch_bucket_ratio": {
                "contiguous_4k": 1,
                "phase_to_8k": 1,
                "phase_to_16k": 2,
            },
        },
        "metric_boundary": (
            "All 13 official RULER generator families are supervised. "
            "Held-out evaluation uses different generated rows. This is "
            "task-family adaptation, not unseen-task transfer."
        ),
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
