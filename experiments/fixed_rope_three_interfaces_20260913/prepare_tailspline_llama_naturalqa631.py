#!/usr/bin/env python3
"""Retokenize the frozen 631-row natural-QA source pool for Llama-3-8B."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.data_prep.target_free_context_builder import (
    load_longbench_prompt_assets,
    render_chat_prompt,
    row_sha256,
    sha256_file,
    sha256_text,
    tokenizer_tree_sha256,
)


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")
EXPECTED = {"hotpotqa": 166, "2wikimqa": 173, "qasper": 119, "narrativeqa": 61, "multifieldqa_en": 112}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def token_digest(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETE" and manifest.get("rows") == 631:
            print(json.dumps({"status": "SKIP_COMPLETE", "rows": 631}))
            return

    frozen_dirs = (
        args.frozen_root / "prepared_natural_01",
        args.frozen_root / "prepared_natural_02",
        args.frozen_root / "prepared_natural_extra_01",
    )
    frozen: dict[str, dict] = {}
    for directory in frozen_dirs:
        for row in read_jsonl(directory / "screen.jsonl"):
            row_id = str(row["row_id"])
            if row_id in frozen:
                raise ValueError(f"duplicate frozen row id: {row_id}")
            frozen[row_id] = row
    if len(frozen) != 778:
        raise ValueError(f"frozen natural-QA pool has {len(frozen)} rows, expected 778")
    selected = [row for row in frozen.values() if row.get("native_stratum") == "extended"]
    counts = Counter(str(row["task"]) for row in selected)
    if len(selected) != 631 or counts != Counter(EXPECTED):
        raise ValueError(f"frozen extended pool identity drift: rows={len(selected)} counts={counts}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    tokenizer_hash = tokenizer_tree_sha256(args.model.resolve())
    prompts, _ = load_longbench_prompt_assets(args.download_root / "longbench")
    archive_path = args.download_root / "longbench/data.zip"
    raw_by_task = {}
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
        if list(old["references"]) != list(original["answers"]):
            raise ValueError(f"reference drift: {old['row_id']}")
        source_id = str(original.get("_id", row_sha256(original)))
        if source_id != str(old["source_id"]):
            raise ValueError(f"source identity drift: {old['row_id']}")
        content = prompts[task].format(context=original["context"], input=original["input"])
        prompt_ids, rendered = render_chat_prompt(tokenizer, content)
        budget = int(old["max_new_tokens"])
        if len(prompt_ids) + budget > 32768:
            raise ValueError(f"retokenized prompt exceeds 32K: {old['row_id']}/{len(prompt_ids)}+{budget}")
        rows.append({
            "row_id": str(old["row_id"]),
            "task": task,
            "family": "longbench_natural_qa",
            "length_cap": 32768,
            "prompt_ids": prompt_ids,
            "prompt_sha256": token_digest(prompt_ids),
            "input_tokens": len(prompt_ids),
            "max_new_tokens": budget,
            "references": [str(value) for value in original["answers"]],
            "source_row_index": source_index,
            "source_document_id": source_id,
            "document_cluster_id": source_id,
            "source_context_sha256": sha256_text(original["context"]),
            "source_row_sha256": row_sha256(original),
            "rendered_chat_prompt_sha256": sha256_text(rendered),
            "historical_pool_stratum": "olmo_input_tokens_gt_4096",
            "llama_native_stratum": "extended" if len(prompt_ids) > 8192 else "within_native",
        })

    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "inputs.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, rows_path)
    llama_counts = Counter(row["llama_native_stratum"] for row in rows)
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_LLAMA_NATURAL_QA_FROZEN631_V1",
        "rows": len(rows),
        "tasks": list(TASKS),
        "rows_by_task": dict(counts),
        "historical_selection": "exact prior 631-row OLMo-tokenizer >4096 source pool; no outcome-based reselection",
        "llama_native_length": 8192,
        "llama_strata": dict(llama_counts),
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "model_path": str(args.model.resolve()),
        "tokenizer_tree_sha256": tokenizer_hash,
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": sha256_file(archive_path),
        "inputs_sha256": sha256_file(rows_path),
        "scope": "Five-task official-template LongBench natural-QA subset on frozen source rows; not the full LongBench leaderboard.",
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
