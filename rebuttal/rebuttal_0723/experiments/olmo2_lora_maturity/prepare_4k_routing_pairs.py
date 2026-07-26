#!/usr/bin/env python3
"""Prepare paired 4K-only RULER-style counterfactual routing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_screen import RULER_COMMIT


LENGTH = 4_096
GENERATION_RESERVE = 128
NUM_KEYS = 8
TRAIN_TEMPLATE = (
    "An access-code registry is embedded below. Memorize which number "
    "belongs to each handle.\n{context}\nWhich access code belongs to "
    "{query}? The special magic {type_needle_v} for {query} mentioned "
    "in the provided text are"
)
CALIBRATION_TEMPLATE = (
    "Study the following ledger of named identifiers.\n{context}\nReturn "
    "the identifier assigned to {query}. The special magic "
    "{type_needle_v} for {query} mentioned in the provided text are"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=1_024)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def generate_rows(
    *,
    ruler_root: Path,
    checkpoint: Path,
    output: Path,
    name: str,
    template: str,
    count: int,
    seed: int,
    chat_overhead: int,
) -> tuple[Path, list[dict[str, Any]]]:
    generator = (
        ruler_root / "scripts" / "data" / "synthetic" / "niah.py"
    )
    command = [
        sys.executable,
        str(generator),
        "--save_dir",
        str(output),
        "--save_name",
        name,
        "--subset",
        "test",
        "--tokenizer_path",
        str(checkpoint),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(LENGTH),
        "--model_template_token",
        str(chat_overhead),
        "--tokens_to_generate",
        str(GENERATION_RESERVE),
        "--num_samples",
        str(int(count)),
        "--random_seed",
        str(int(seed)),
        "--template",
        template,
        "--num_needle_k",
        str(NUM_KEYS),
        "--num_needle_v",
        "1",
        "--num_needle_q",
        "1",
        "--type_haystack",
        "noise",
        "--type_needle_k",
        "words",
        "--type_needle_v",
        "numbers",
    ]
    subprocess.run(command, check=True)
    path = output / name / "test.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(count):
        raise RuntimeError(f"generated row count drift: {len(rows)}")
    return path, rows


def prompt_ids(
    tokenizer: Any,
    row: dict[str, Any],
) -> list[int]:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["input"]}],
        add_generation_prompt=True,
    )
    prefix = tokenizer(
        row["answer_prefix"], add_special_tokens=False
    ).input_ids
    return [int(value) for value in chat + prefix]


def choose_alternate(
    *,
    tokenizer: Any,
    row: dict[str, Any],
    gold: str,
    gold_ids: list[int],
    original_prompt_ids: list[int],
    used_values: set[str],
    rng: random.Random,
) -> tuple[str, dict[str, Any], list[int], list[int]]:
    if row["input"].count(gold) != 1:
        raise RuntimeError("gold value is not unique in the source prompt")
    for _ in range(10_000):
        alternate = str(rng.randint(1_000_000, 9_999_999))
        if alternate == gold or alternate in used_values:
            continue
        alternate_ids = [
            int(value)
            for value in tokenizer(
                alternate, add_special_tokens=False
            ).input_ids
        ]
        if len(alternate_ids) != len(gold_ids):
            continue
        swapped = dict(row)
        swapped["input"] = row["input"].replace(gold, alternate, 1)
        swapped["outputs"] = [alternate]
        swapped_prompt_ids = prompt_ids(tokenizer, swapped)
        if len(swapped_prompt_ids) != len(original_prompt_ids):
            continue
        return alternate, swapped, swapped_prompt_ids, alternate_ids
    raise RuntimeError("could not find geometry-matched alternate value")


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def build_pair_set(
    *,
    output: Path,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    seed: int,
    purpose: str,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    input_ids = np.lib.format.open_memmap(
        output / "input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(len(rows), 2, LENGTH),
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(rows), 2, LENGTH),
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad nor EOS token")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    input_ids[:] = int(pad_id)
    labels[:] = -100

    rng = random.Random(int(seed))
    used_values = {
        str(row["outputs"][0])
        for row in rows
    }
    answer_lengths: list[int] = []
    prompt_lengths: list[int] = []
    metadata_path = output / "rows.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if len(row["outputs"]) != 1:
                raise RuntimeError("routing pair requires one answer")
            gold = str(row["outputs"][0])
            gold_ids = [
                int(value)
                for value in tokenizer(
                    gold, add_special_tokens=False
                ).input_ids
            ]
            if not gold_ids:
                raise RuntimeError("empty gold answer tokenization")
            original_prompt_ids = prompt_ids(tokenizer, row)
            (
                alternate,
                swapped,
                swapped_prompt_ids,
                alternate_ids,
            ) = choose_alternate(
                tokenizer=tokenizer,
                row=row,
                gold=gold,
                gold_ids=gold_ids,
                original_prompt_ids=original_prompt_ids,
                used_values=used_values,
                rng=rng,
            )
            used_values.add(alternate)
            variants = (
                (row, original_prompt_ids, gold_ids, gold, alternate),
                (
                    swapped,
                    swapped_prompt_ids,
                    alternate_ids,
                    alternate,
                    gold,
                ),
            )
            starts = []
            for variant_index, (
                variant,
                local_prompt,
                answer_ids,
                answer,
                counterfactual,
            ) in enumerate(variants):
                del variant, answer, counterfactual
                sequence = local_prompt + answer_ids + [int(eos_id)]
                if len(sequence) > LENGTH:
                    raise RuntimeError(
                        f"routing sequence exceeds 4K: {len(sequence)}"
                    )
                input_ids[index, variant_index, : len(sequence)] = (
                    np.asarray(sequence, dtype=np.uint32)
                )
                answer_start = len(local_prompt)
                labels[
                    index,
                    variant_index,
                    answer_start : answer_start + len(answer_ids),
                ] = np.asarray(answer_ids, dtype=np.int32)
                starts.append(answer_start)
            if starts[0] != starts[1]:
                raise RuntimeError("counterfactual answer geometry drift")
            answer_lengths.append(len(gold_ids))
            prompt_lengths.append(starts[0])
            handle.write(
                json.dumps(
                    {
                        "row": index,
                        "source_row_sha256": row_digest(row),
                        "gold_value": gold,
                        "alternate_value": alternate,
                        "gold_token_ids": gold_ids,
                        "alternate_token_ids": alternate_ids,
                        "answer_start": starts[0],
                        "answer_tokens": len(gold_ids),
                        "source_token_position_answer": int(
                            row["token_position_answer"]
                        ),
                        "source_length": int(row["length"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    input_ids.flush()
    labels.flush()
    expected_labels = 2 * sum(answer_lengths)
    actual_labels = int((labels != -100).sum())
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"routing label count drift: {actual_labels}"
        )
    manifest = {
        "format_version": 1,
        "status": "OLMO2_4K_COUNTERFACTUAL_ROUTING_SET_PREPARED",
        "purpose": purpose,
        "shape": [len(rows), 2, LENGTH],
        "variant_names": ["sourced", "value_swapped"],
        "maximum_training_position_id": LENGTH - 1,
        "maximum_training_length": LENGTH,
        "queries_per_sequence": 1,
        "key_value_blocks_per_sequence": NUM_KEYS,
        "answer_only": True,
        "counterfactual_value_swap": True,
        "minimum_answer_tokens": min(answer_lengths),
        "maximum_answer_tokens": max(answer_lengths),
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "supervised_answer_tokens": actual_labels,
        "seed": int(seed),
        "files": {
            "input_ids.npy": sha256_file(output / "input_ids.npy"),
            "labels.npy": sha256_file(output / "labels.npy"),
            "rows.jsonl": sha256_file(metadata_path),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    ruler_root = args.ruler_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    commit = subprocess.run(
        ["git", "-C", str(ruler_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != RULER_COMMIT:
        raise RuntimeError(f"RULER commit drift: {commit}")
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
    raw_train_path, train_rows = generate_rows(
        ruler_root=ruler_root,
        checkpoint=checkpoint,
        output=output / "raw_train",
        name="routing_train",
        template=TRAIN_TEMPLATE,
        count=int(args.train_rows),
        seed=int(args.seed) + 11_000,
        chat_overhead=chat_overhead,
    )
    raw_calibration_path, calibration_rows = generate_rows(
        ruler_root=ruler_root,
        checkpoint=checkpoint,
        output=output / "raw_calibration",
        name="routing_calibration",
        template=CALIBRATION_TEMPLATE,
        count=int(args.calibration_rows),
        seed=int(args.seed) + 22_000,
        chat_overhead=chat_overhead,
    )
    train_values = {str(row["outputs"][0]) for row in train_rows}
    calibration_values = {
        str(row["outputs"][0]) for row in calibration_rows
    }
    if train_values & calibration_values:
        raise RuntimeError("train/calibration answer values overlap")
    train_manifest = build_pair_set(
        output=output / "train",
        tokenizer=tokenizer,
        rows=train_rows,
        seed=int(args.seed) + 33_000,
        purpose="counterfactual_routing_training",
    )
    calibration_manifest = build_pair_set(
        output=output / "calibration",
        tokenizer=tokenizer,
        rows=calibration_rows,
        seed=int(args.seed) + 44_000,
        purpose="counterfactual_routing_calibration",
    )
    receipt = {
        "format_version": 1,
        "status": "OLMO2_4K_COUNTERFACTUAL_ROUTING_DATA_PREPARED",
        "ruler_commit": commit,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "train_eval_values_disjoint": True,
        "official_eval_template_used_for_training": False,
        "templates": {
            "train": TRAIN_TEMPLATE,
            "calibration": CALIBRATION_TEMPLATE,
        },
        "raw": {
            "train": {
                "path": str(raw_train_path.relative_to(output)),
                "sha256": sha256_file(raw_train_path),
            },
            "calibration": {
                "path": str(
                    raw_calibration_path.relative_to(output)
                ),
                "sha256": sha256_file(raw_calibration_path),
            },
        },
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
    atomic_json(output / "manifest.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "train_shape": train_manifest["shape"],
                "calibration_shape": calibration_manifest["shape"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
