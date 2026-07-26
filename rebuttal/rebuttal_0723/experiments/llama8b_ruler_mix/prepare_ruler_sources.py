#!/usr/bin/env python3
"""Generate seed-separated official RULER sources for 8K training and evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_EVAL_LENGTHS,
    EVAL_SOURCE_STATUS,
    RULER_COMMIT,
    SOURCE_STATUS,
    TASKS,
    TRAIN_SOURCE_STATUS,
    TRAIN_LENGTH,
    atomic_json,
    load_jsonl,
    row_sha256,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path)
    parser.add_argument(
        "--scope",
        choices=("train", "eval", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows-per-task", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=20_420_726)
    parser.add_argument("--eval-seed", type=int, default=20_420_727)
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_LENGTHS),
    )
    return parser.parse_args()


def verify_existing(
    manifest_path: Path,
    expected_status: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise RuntimeError("existing RULER source manifest has wrong status")
    root = manifest_path.parent
    for record in manifest["files"].values():
        path = root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"existing source drift: {path}")
    return manifest


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def qa_query(row: dict[str, Any]) -> str:
    tail = str(row["input"]).rsplit("Question:", 1)[-1]
    return tail.split("<|eot_id|>", 1)[0].strip()


def run_task(
    *,
    ruler_root: Path,
    tokenizer: Path,
    destination: Path,
    task: str,
    length: int,
    rows: int,
    seed: int,
    qa_half: int | None,
    environment: dict[str, str],
    log_path: Path,
) -> Path:
    output_path = destination / task / "test.jsonl"
    if output_path.is_file():
        existing = load_jsonl(output_path)
        if len(existing) == rows:
            return output_path
        raise RuntimeError(
            f"partial source has {len(existing)} rows, expected {rows}: "
            f"{output_path}"
        )

    command = [
        sys.executable,
        str(ruler_root / "scripts/data/prepare.py"),
        "--save_dir",
        str(destination),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--subset",
        "test",
        "--tokenizer_path",
        str(tokenizer),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(length),
        "--model_template_type",
        "meta-llama3",
        "--random_seed",
        str(seed),
    ]
    if qa_half is None:
        command.extend(["--num_samples", str(rows)])
    else:
        command.extend(
            [
                "--num_samples",
                str(2 * rows),
                "--chunk_amount",
                "2",
                "--chunk_idx",
                str(qa_half),
            ]
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ruler_root / "scripts/data",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"official RULER preparation failed for {task} L{length}; "
            f"see {log_path}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            f"official RULER did not create {output_path}; see {log_path}"
        )
    generated = load_jsonl(output_path)
    if len(generated) != rows:
        raise RuntimeError(
            f"official RULER created {len(generated)} rows, expected {rows}"
        )
    return output_path


def validate_rows(
    path: Path,
    *,
    task: str,
    length: int,
    rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = load_jsonl(path)
    if len(values) != rows:
        raise RuntimeError(f"row-count drift: {path}")
    digests: set[str] = set()
    realized_lengths: list[int] = []
    for index, row in enumerate(values):
        required = {"index", "input", "outputs", "length", "answer_prefix"}
        if not required.issubset(row):
            raise RuntimeError(f"RULER schema drift at {path}:{index + 1}")
        digest = row_sha256(row)
        if digest in digests:
            raise RuntimeError(f"duplicate RULER row in {path}")
        digests.add(digest)
        realized = int(row["length_w_model_temp"])
        if realized > length:
            raise RuntimeError(
                f"RULER row exceeds requested length: {realized} > {length}"
            )
        realized_lengths.append(realized)
    return values, {
        "path": "",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(values),
        "task": task,
        "requested_length": length,
        "realized_length_min": min(realized_lengths),
        "realized_length_max": max(realized_lengths),
    }


def main() -> None:
    args = parse_args()
    ruler_root = args.ruler_root.resolve()
    tokenizer = args.tokenizer.resolve()
    output = args.output.resolve()
    manifest_name = {
        "train": "train_manifest.json",
        "eval": "eval_manifest.json",
        "all": "manifest.json",
    }[args.scope]
    expected_status = {
        "train": TRAIN_SOURCE_STATUS,
        "eval": EVAL_SOURCE_STATUS,
        "all": SOURCE_STATUS,
    }[args.scope]
    manifest_path = output / manifest_name
    if manifest_path.is_file():
        manifest = verify_existing(manifest_path, expected_status)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if int(args.rows_per_task) != 100:
        raise ValueError("registered protocol requires 100 rows per task")
    eval_lengths = tuple(sorted(set(int(x) for x in args.eval_lengths)))
    if eval_lengths != DEFAULT_EVAL_LENGTHS:
        raise ValueError(
            f"registered evaluation lengths are {DEFAULT_EVAL_LENGTHS}"
        )
    if int(args.train_seed) == int(args.eval_seed):
        raise ValueError("training and evaluation seeds must differ")
    if not 1 <= int(args.workers) <= 16:
        raise ValueError("workers must be in [1, 16]")
    if git_revision(ruler_root) != RULER_COMMIT:
        raise RuntimeError("official RULER revision drift")
    if not (tokenizer / "tokenizer.json").is_file():
        raise FileNotFoundError(tokenizer / "tokenizer.json")

    qa_root = ruler_root / "scripts/data/synthetic/json"
    qa_receipts = {}
    for name in ("squad.json", "hotpotqa.json"):
        path = qa_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        qa_receipts[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.nltk_data is not None:
        nltk_data = args.nltk_data.resolve()
        if not nltk_data.is_dir():
            raise FileNotFoundError(nltk_data)
        environment["NLTK_DATA"] = str(nltk_data)

    files: dict[str, Any] = {}
    rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    cells = []
    if args.scope in {"train", "all"}:
        cells.append(("train", TRAIN_LENGTH, int(args.train_seed), 0))
    if args.scope in {"eval", "all"}:
        cells.extend(
            ("eval", length, int(args.eval_seed), 1)
            for length in eval_lengths
        )
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        for split, length, seed, qa_half in cells:
            destination = output / split / f"L{length}"
            for task in TASKS:
                future = executor.submit(
                    run_task,
                    ruler_root=ruler_root,
                    tokenizer=tokenizer,
                    destination=destination,
                    task=task,
                    length=length,
                    rows=int(args.rows_per_task),
                    seed=seed,
                    qa_half=(
                        qa_half if task in {"qa_1", "qa_2"} else None
                    ),
                    environment=environment,
                    log_path=(
                        output
                        / "logs"
                        / f"{split}_L{length}_{task}.log"
                    ),
                )
                jobs.append((split, length, task, future))
        for split, length, task, future in jobs:
            path = future.result()
            key = f"{split}/L{length}/{task}/test.jsonl"
            values, receipt = validate_rows(
                path,
                task=task,
                length=length,
                rows=int(args.rows_per_task),
            )
            receipt["path"] = path.relative_to(output).as_posix()
            files[key] = receipt
            rows_by_cell[(split, length, task)] = values

    overlap: dict[str, Any] = {}
    if args.scope == "all":
        for task in TASKS:
            training = rows_by_cell[("train", TRAIN_LENGTH, task)]
            training_hashes = {row_sha256(row) for row in training}
            task_record: dict[str, Any] = {}
            train_queries = (
                {qa_query(row) for row in training}
                if task in {"qa_1", "qa_2"}
                else set()
            )
            for length in eval_lengths:
                evaluation = rows_by_cell[("eval", length, task)]
                exact = training_hashes & {
                    row_sha256(row) for row in evaluation
                }
                if exact:
                    raise RuntimeError(
                        f"training/evaluation exact-row overlap: "
                        f"{task} L{length}"
                    )
                record = {"exact_row_overlap": 0}
                if task in {"qa_1", "qa_2"}:
                    query_overlap = train_queries & {
                        qa_query(row) for row in evaluation
                    }
                    if query_overlap:
                        raise RuntimeError(
                            f"training/evaluation QA query overlap: "
                            f"{task} L{length}"
                        )
                    record["qa_query_overlap"] = 0
                task_record[f"L{length}"] = record
            overlap[task] = task_record

    manifest = {
        "status": expected_status,
        "protocol": {
            "scope": args.scope,
            "physical_training_length": TRAIN_LENGTH,
            "evaluation_lengths": list(eval_lengths),
            "tasks": list(TASKS),
            "rows_per_task": int(args.rows_per_task),
            "train_seed": int(args.train_seed),
            "eval_seed": int(args.eval_seed),
            "model_template_type": "meta-llama3",
            "qa_partition": {
                "train": "official source indices 0-99",
                "eval": "official source indices 100-199",
            },
        },
        "implementation": {
            "repository": "NVIDIA/RULER",
            "commit": RULER_COMMIT,
            "ruler_root_basename": ruler_root.name,
            "tokenizer_basename": tokenizer.name,
            "qa_sources": qa_receipts,
        },
        "separation": (
            overlap
            if args.scope == "all"
            else {
                "status": (
                    "cross-scope audit is performed before evaluation"
                )
            }
        ),
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
