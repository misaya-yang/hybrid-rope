#!/usr/bin/env python3
"""Prepare held-out official-RULER transfer tasks for instruct checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .prepare_data import atomic_json, sha256_file


RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
DATA_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_PREPARED"
GENERIC_DATA_STATUS = "INSTRUCT_RULER_TRANSFER_PREPARED"
DEFAULT_LENGTHS = (4_096, 8_192, 16_384)
SUPPORTED_LENGTHS = DEFAULT_LENGTHS + (32_768, 65_536, 131_072)


def _niah(
    *,
    haystack: str,
    key: str,
    value: str,
    num_keys: int,
    num_values: int,
    num_queries: int,
    role: str,
) -> dict[str, Any]:
    return {
        "base_task": "niah",
        "tokens_to_generate": 128,
        "official_metric": "string_match_all",
        "args": {
            "type_haystack": haystack,
            "type_needle_k": key,
            "type_needle_v": value,
            "num_needle_k": num_keys,
            "num_needle_v": num_values,
            "num_needle_q": num_queries,
        },
        "role": role,
    }


TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "niah_single_1": _niah(
        haystack="noise",
        key="words",
        value="numbers",
        num_keys=1,
        num_values=1,
        num_queries=1,
        role="single_needle_noise_numeric",
    ),
    "niah_single_2": _niah(
        haystack="essay",
        key="words",
        value="numbers",
        num_keys=1,
        num_values=1,
        num_queries=1,
        role="single_needle_essay_numeric",
    ),
    "niah_single_3": _niah(
        haystack="essay",
        key="words",
        value="uuids",
        num_keys=1,
        num_values=1,
        num_queries=1,
        role="single_needle_essay_uuid",
    ),
    "niah_multikey_1": _niah(
        haystack="essay",
        key="words",
        value="numbers",
        num_keys=4,
        num_values=1,
        num_queries=1,
        role="multi_key_essay_numeric",
    ),
    "niah_multikey_2": _niah(
        haystack="needle",
        key="words",
        value="numbers",
        num_keys=1,
        num_values=1,
        num_queries=1,
        role="key_value_distractors_numeric",
    ),
    "niah_multikey_3": {
        **_niah(
            haystack="needle",
            key="uuids",
            value="uuids",
            num_keys=1,
            num_values=1,
            num_queries=1,
            role="key_value_distractors_uuid",
        )
    },
    "niah_multivalue": _niah(
        haystack="essay",
        key="words",
        value="numbers",
        num_keys=1,
        num_values=4,
        num_queries=1,
        role="multi_value_essay_numeric",
    ),
    "niah_multiquery": _niah(
        haystack="essay",
        key="words",
        value="numbers",
        num_keys=1,
        num_values=1,
        num_queries=4,
        role="multi_query_essay_numeric",
    ),
    "vt": {
        "base_task": "variable_tracking",
        "tokens_to_generate": 30,
        "official_metric": "string_match_all",
        "args": {
            "type_haystack": "noise",
            "num_chains": 1,
            "num_hops": 4,
        },
        "role": "multi_hop_variable_tracking",
    },
    "cwe": {
        "base_task": "common_words_extraction",
        "tokens_to_generate": 120,
        "official_metric": "string_match_all",
        "args": {
            "freq_cw": 30,
            "freq_ucw": 3,
            "num_cw": 10,
        },
        "role": "aggregation_common_words_extraction",
    },
    "fwe": {
        "base_task": "freq_words_extraction",
        "tokens_to_generate": 50,
        "official_metric": "string_match_all",
        "args": {"alpha": 2.0},
        "role": "frequency_words_extraction",
    },
    "qa_1": {
        "base_task": "qa",
        "tokens_to_generate": 32,
        "official_metric": "string_match_part",
        "args": {"dataset": "squad"},
        "role": "squad_document_qa",
    },
    "qa_2": {
        "base_task": "qa",
        "tokens_to_generate": 32,
        "official_metric": "string_match_part",
        "args": {"dataset": "hotpotqa"},
        "role": "hotpotqa_document_qa",
    },
}
DEFAULT_TASKS = tuple(TASK_CONFIGS)


def chat_input_ids(value: Any) -> Any:
    """Normalize Transformers 4.x tensors/lists and 5.x BatchEncoding."""

    if isinstance(value, dict):
        return value["input_ids"]
    input_ids = getattr(value, "input_ids", None)
    return value if input_ids is None else input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_CONFIGS),
        default=list(DEFAULT_TASKS),
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
    )
    parser.add_argument("--samples-per-cell", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent task-length generator subprocesses; default preserves serial execution.",
    )
    parser.add_argument(
        "--data-status",
        choices=(DATA_STATUS, GENERIC_DATA_STATUS),
        default=DATA_STATUS,
    )
    return parser.parse_args()


def _generator_command(
    *,
    python: str,
    generator: Path,
    save_dir: Path,
    task: str,
    checkpoint: Path,
    nominal_length: int,
    chat_overhead: int,
    samples: int,
    seed: int,
    template: str,
    config: dict[str, Any],
) -> list[str]:
    command = [
        python,
        str(generator),
        "--save_dir",
        str(save_dir),
        "--save_name",
        task,
        "--subset",
        "test",
        "--tokenizer_path",
        str(checkpoint),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(nominal_length - chat_overhead),
        "--tokens_to_generate",
        str(int(config["tokens_to_generate"])),
        "--num_samples",
        str(samples),
        "--random_seed",
        str(seed),
        "--template",
        template,
    ]
    for name, value in config["args"].items():
        command.extend((f"--{name}", str(value)))
    return command


def verify_external_assets(
    synthetic: Path,
    tasks: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    json_root = synthetic / "json"
    required = {
        "qa_1": json_root / "squad.json",
        "qa_2": json_root / "hotpotqa.json",
    }
    for task, path in required.items():
        if task not in tasks:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        assets[task] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if any(
        TASK_CONFIGS[task]["args"].get("type_haystack") == "essay"
        for task in tasks
    ):
        essay_corpus = json_root / "PaulGrahamEssays.json"
        if not essay_corpus.is_file() or essay_corpus.stat().st_size == 0:
            raise FileNotFoundError(essay_corpus)
        assets["essay_corpus"] = {
            "path": str(essay_corpus),
            "bytes": essay_corpus.stat().st_size,
            "sha256": sha256_file(essay_corpus),
        }
    return assets


def main() -> None:
    args = parse_args()
    root = args.ruler_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    tasks = tuple(str(task) for task in args.tasks)
    lengths = tuple(int(length) for length in args.lengths)
    if len(set(tasks)) != len(tasks):
        raise RuntimeError("tasks must be unique")
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise RuntimeError("lengths must be unique and ascending")
    if any(length not in SUPPORTED_LENGTHS for length in lengths):
        raise RuntimeError(f"supported transfer lengths are {SUPPORTED_LENGTHS}")
    if not 1 <= int(args.samples_per_cell) <= 500:
        raise RuntimeError("samples-per-cell must be in [1, 500]")
    if not 1 <= int(args.workers) <= 32:
        raise RuntimeError("workers must be in [1, 32]")
    if output.exists():
        raise FileExistsError(output)

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

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    empty_raw = len(
        tokenizer("", add_special_tokens=False).input_ids
    )
    empty_chat = len(
        chat_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
        ))
    )
    chat_overhead = empty_chat - empty_raw
    if not 0 <= chat_overhead < min(lengths):
        raise RuntimeError(f"invalid chat-template overhead: {chat_overhead}")

    synthetic = root / "scripts" / "data" / "synthetic"
    sys.path.insert(0, str(synthetic))
    from constants import TASKS  # type: ignore[import-not-found]

    assets = verify_external_assets(synthetic, tasks)
    output.mkdir(parents=True)
    jobs: list[list[str]] = []
    for task in tasks:
        config = TASK_CONFIGS[task]
        base_task = str(config["base_task"])
        task_template = (
            TASKS[base_task]["template"]
            + TASKS[base_task].get("answer_prefix", "")
        )
        generator = synthetic / f"{base_task}.py"
        if not generator.is_file():
            raise FileNotFoundError(generator)
        for nominal_length in lengths:
            save_dir = output / f"L{nominal_length}"
            jobs.append(_generator_command(
                python=sys.executable,
                generator=generator,
                save_dir=save_dir,
                task=task,
                checkpoint=checkpoint,
                nominal_length=nominal_length,
                chat_overhead=chat_overhead,
                samples=int(args.samples_per_cell),
                seed=int(args.seed),
                template=task_template,
                config=config,
            ))

    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in jobs]
        for future in futures:
            future.result()

    cells: dict[str, dict[str, Any]] = {}
    for task in tasks:
        config = TASK_CONFIGS[task]
        cells[task] = {}
        for nominal_length in lengths:
            save_dir = output / f"L{nominal_length}"
            path = save_dir / task / "test.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != int(args.samples_per_cell):
                raise RuntimeError(
                    f"row-count drift for {task} at {nominal_length}"
                )
            prompt_lengths: list[int] = []
            for row in rows:
                chat_ids = chat_input_ids(tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["input"]}],
                    add_generation_prompt=True,
                ))
                prefix_ids = tokenizer(
                    row.get("answer_prefix", ""),
                    add_special_tokens=False,
                ).input_ids
                prompt_length = len(chat_ids) + len(prefix_ids)
                if (
                    prompt_length + int(config["tokens_to_generate"])
                    > nominal_length
                ):
                    raise RuntimeError(
                        f"{task} exceeds L{nominal_length}: "
                        f"{prompt_length}+{config['tokens_to_generate']}"
                    )
                prompt_lengths.append(prompt_length)
            cells[task][str(nominal_length)] = {
                "relative_path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "minimum_prompt_tokens": min(prompt_lengths),
                "maximum_prompt_tokens": max(prompt_lengths),
                "generation_tokens": int(
                    config["tokens_to_generate"]
                ),
                "official_metric": str(config["official_metric"]),
            }

    receipt = {
        "format_version": 1,
        "status": str(args.data_status),
        "metric": "official task-specific RULER scoring",
        "suite": "official RULER synthetic.yaml complete 13-task matrix",
        "ruler_commit": actual_commit,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "chat_overhead_tokens": chat_overhead,
        "tasks": list(tasks),
        "task_configs": {
            task: TASK_CONFIGS[task] for task in tasks
        },
        "lengths": list(lengths),
        "samples_per_cell": int(args.samples_per_cell),
        "seed": int(args.seed),
        "external_assets": assets,
        "cells": cells,
    }
    atomic_json(output / "manifest.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "tasks": list(tasks),
                "lengths": list(lengths),
                "cells": cells,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
