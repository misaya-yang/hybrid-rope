#!/usr/bin/env python3
"""Prepare a small official-RULER screen with the OLMo-2 chat tokenizer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from .prepare_data import atomic_json, sha256_file


RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
LENGTHS = (4_096, 8_192, 16_384)
TASK_CONFIGS = {
    "niah_single_1": {
        "type_haystack": "noise",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
    },
    "niah_single_2": {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
    },
}
TOKENS_TO_GENERATE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=tuple(TASK_CONFIGS),
        default="niah_single_1",
    )
    parser.add_argument("--samples-per-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.ruler_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    task = str(args.task)
    task_config = TASK_CONFIGS[task]
    if output.exists():
        raise FileExistsError(output)
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

    synthetic = root / "scripts" / "data" / "synthetic"
    sys.path.insert(0, str(synthetic))
    from constants import TASKS  # type: ignore[import-not-found]

    template = (
        TASKS["niah"]["template"]
        + TASKS["niah"]["answer_prefix"]
    )
    generator = synthetic / "niah.py"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
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

    files = {}
    for nominal_length in LENGTHS:
        save_dir = output / f"L{nominal_length}"
        command = [
            sys.executable,
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
            str(TOKENS_TO_GENERATE),
            "--num_samples",
            str(int(args.samples_per_length)),
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
            task_config["type_haystack"],
            "--type_needle_k",
            task_config["type_needle_k"],
            "--type_needle_v",
            task_config["type_needle_v"],
        ]
        subprocess.run(command, check=True)
        path = save_dir / task / "test.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(args.samples_per_length):
            raise RuntimeError("RULER row-count drift")
        prompt_lengths = []
        for row in rows:
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
            )
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
            ).input_ids
            prompt_length = len(chat_ids) + len(prefix_ids)
            if prompt_length + TOKENS_TO_GENERATE > nominal_length:
                raise RuntimeError(
                    f"RULER row exceeds L{nominal_length}: "
                    f"{prompt_length}+{TOKENS_TO_GENERATE}"
                )
            prompt_lengths.append(prompt_length)
        files[str(nominal_length)] = {
            "relative_path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "minimum_prompt_tokens": min(prompt_lengths),
            "maximum_prompt_tokens": max(prompt_lengths),
            "generation_tokens": TOKENS_TO_GENERATE,
        }

    receipt = {
        "format_version": 1,
        "status": "OLMO2_INSTRUCT_RULER_SCREEN_PREPARED",
        "task": task,
        "task_config": task_config,
        "ruler_commit": actual_commit,
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "chat_overhead_tokens": chat_overhead,
        "lengths": list(LENGTHS),
        "samples_per_length": int(args.samples_per_length),
        "seed": int(args.seed),
        "files": files,
    }
    atomic_json(output / "manifest.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
