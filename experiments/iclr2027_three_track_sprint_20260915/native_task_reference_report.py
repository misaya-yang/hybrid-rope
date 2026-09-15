#!/usr/bin/env python3
"""Summarize the one-arm Native 8K classic RULER reference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel = [row for row in rows(args.panel) if int(row["length_cap"]) == 8192]
    output = rows(args.run)
    expected = {row["prompt_sha256"] for row in panel}
    actual = {row["prompt_sha256"] for row in output}
    if len(panel) != 130 or len(output) != 130 or expected != actual:
        raise ValueError("Native 8K task reference is not the paired classic 13x10 panel")
    counts = Counter(row["task"] for row in output)
    if counts != Counter({task: 10 for task in TASKS}):
        raise ValueError("Native 8K task coverage drift")
    by_task = defaultdict(list)
    for row in output:
        by_task[row["task"]].append(float(row["ruler_official_score"]))
    task_scores = {task: float(np.mean(by_task[task])) for task in TASKS}
    result = {
        "status": "LLAMA_NATIVE_8K_RULER_REFERENCE_COMPLETE_V1",
        "length": 8192,
        "rows": 130,
        "rows_per_task": 10,
        "task_equal_macro": float(np.mean(list(task_scores.values()))),
        "by_task": task_scores,
        "output_health": {
            "ended_eos": sum(bool(row["ended_eos"]) for row in output),
            "hit_cap": sum(bool(row["hit_cap"]) for row in output),
            "empty": sum(bool(row["empty"]) for row in output),
        },
        "panel_sha256": sha256(args.panel),
        "raw_sha256": sha256(args.run),
        "claim_boundary": "One-arm Native reference on the classic 8K panel; comparisons require matched T/P rows and a declared non-inferiority margin.",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
