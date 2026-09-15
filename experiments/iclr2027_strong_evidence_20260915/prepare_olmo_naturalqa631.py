#!/usr/bin/env python3
"""Freeze the historical OLMo-tokenized 631-row extended Natural-QA pool."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.data_prep.target_free_context_builder import (
    row_sha256,
    sha256_file,
    sha256_text,
    tokenizer_tree_sha256,
)


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")
EXPECTED = {
    "hotpotqa": 166,
    "2wikimqa": 173,
    "qasper": 119,
    "narrativeqa": 61,
    "multifieldqa_en": 112,
}
FROZEN_DIRS = (
    "prepared_natural_01",
    "prepared_natural_02",
    "prepared_natural_extra_01",
)
CONTRACT = "TAILSPLINE_OLMO_S4_NATURAL_QA_FROZEN631_V1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def token_digest(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--target-length", type=int, default=16384)
    args = parser.parse_args()

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") == "COMPLETE"
            and manifest.get("contract") == CONTRACT
            and manifest.get("rows") == 631
            and manifest.get("native_length") == args.native_length
            and manifest.get("target_length") == args.target_length
            and Path(manifest.get("model_path", "")).resolve() == args.model.resolve()
            and (output / "inputs.jsonl").is_file()
            and sha256_file(output / "inputs.jsonl") == manifest.get("inputs_sha256")
        ):
            print(json.dumps({"status": "SKIP_COMPLETE", "rows": 631}))
            return

    if args.native_length <= 0 or args.target_length <= args.native_length:
        raise ValueError("target length must exceed a positive native length")
    model = args.model.resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing: {model / 'config.json'}")

    frozen: dict[str, dict] = {}
    source_files: list[Path] = []
    for name in FROZEN_DIRS:
        directory = args.frozen_root / name
        screen = directory / "screen.jsonl"
        source_files.append(screen)
        source_manifest = json.loads((directory / "manifest.json").read_text())
        if Path(source_manifest["model_path"]).resolve() != model:
            raise ValueError(f"frozen tokenizer/model identity drift: {directory}")
        for row in read_jsonl(screen):
            row_id = str(row["row_id"])
            if row_id in frozen:
                raise ValueError(f"duplicate frozen row id: {row_id}")
            frozen[row_id] = row
    if len(frozen) != 778:
        raise ValueError(f"frozen Natural-QA pool has {len(frozen)} rows, expected 778")

    selected = [row for row in frozen.values() if row.get("native_stratum") == "extended"]
    counts = Counter(str(row["task"]) for row in selected)
    if len(selected) != 631 or counts != Counter(EXPECTED):
        raise ValueError(f"frozen extended-pool identity drift: rows={len(selected)} counts={counts}")

    archive_path = args.download_root / "longbench/data.zip"
    raw_by_task: dict[str, list[dict]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for task in TASKS:
            raw_by_task[task] = [
                json.loads(line)
                for line in archive.read(f"data/{task}.jsonl").decode().splitlines()
                if line
            ]

    rows = []
    for old in sorted(selected, key=lambda row: (TASKS.index(str(row["task"])), str(row["row_id"]))):
        task = str(old["task"])
        source_index = int(old["source_row_index"])
        original = raw_by_task[task][source_index]
        references = [str(value) for value in original["answers"]]
        source_id = str(original.get("_id", row_sha256(original)))
        prompt_ids = [int(value) for value in old["prompt_ids"]]
        budget = int(old["max_new_tokens"])
        if references != [str(value) for value in old["references"]]:
            raise ValueError(f"reference drift: {old['row_id']}")
        if source_id != str(old["source_id"]):
            raise ValueError(f"source identity drift: {old['row_id']}")
        if len(prompt_ids) != int(old["input_tokens"]):
            raise ValueError(f"input length drift: {old['row_id']}")
        if token_digest(prompt_ids) != str(old["prompt_sha256"]):
            raise ValueError(f"prompt digest drift: {old['row_id']}")
        if len(prompt_ids) <= args.native_length or len(prompt_ids) + budget > args.target_length:
            raise ValueError(f"row violates frozen length contract: {old['row_id']}")
        context_hash = sha256_text(original["context"])
        rows.append({
            "row_id": str(old["row_id"]),
            "task": task,
            "family": "longbench_natural_qa",
            "length_cap": args.target_length,
            "prompt_ids": prompt_ids,
            "prompt_sha256": str(old["prompt_sha256"]),
            "input_tokens": len(prompt_ids),
            "actual_length": len(prompt_ids),
            "max_new_tokens": budget,
            "references": references,
            "source_row_index": source_index,
            "source_document_id": source_id,
            "document_cluster_id": context_hash,
            "source_context_sha256": context_hash,
            "source_row_sha256": row_sha256(original),
            "historical_pool_stratum": "olmo_input_tokens_gt_4096",
            "olmo_native_stratum": "extended",
        })

    prompt_counts = Counter(row["prompt_sha256"] for row in rows)
    duplicate_prompt_groups = sum(count > 1 for count in prompt_counts.values())
    duplicate_prompt_rows = sum(count - 1 for count in prompt_counts.values() if count > 1)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "inputs.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, rows_path)

    manifest = {
        "status": "COMPLETE",
        "contract": CONTRACT,
        "rows": len(rows),
        "tasks": list(TASKS),
        "rows_by_task": dict(counts),
        "historical_selection": (
            "Exact prior 631-row OLMo-tokenizer >4096 source pool; no outcome-based reselection"
        ),
        "native_length": args.native_length,
        "target_length": args.target_length,
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "model_path": str(model),
        "tokenizer_tree_sha256": tokenizer_tree_sha256(model),
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": sha256_file(archive_path),
        "source_screens": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in source_files
        ],
        "document_cluster": "source_context_sha256; repeated source contexts form one bootstrap cluster",
        "prompt_identity_audit": {
            "unique_prompt_sha256": len(prompt_counts),
            "duplicate_prompt_groups": duplicate_prompt_groups,
            "duplicate_prompt_rows": duplicate_prompt_rows,
            "interpretation": (
                "Frozen Natural-QA rows are retained even when rendered prompts repeat; "
                "row identity and source-context clustering, not prompt uniqueness, define inference"
            ),
        },
        "inputs_sha256": sha256_file(rows_path),
        "scope": (
            "Five-task official-template LongBench Natural-QA subset on a frozen historical source pool; "
            "not full LongBench and not a newly sampled independent benchmark"
        ),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
