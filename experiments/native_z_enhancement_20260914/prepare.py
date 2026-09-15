#!/usr/bin/env python3
"""Prepare disjoint Native-4K optimization and evaluation assets for OLMo-2-1B."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
NATURAL_TASKS = ("hotpotqa", "2wikimqa", "qasper")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def token_digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19-validation", type=Path, required=True)
    parser.add_argument("--ppl-manifest", type=Path, required=True)
    parser.add_argument("--ppl-array", type=Path, required=True)
    parser.add_argument("--ruler-panel", type=Path, required=True)
    parser.add_argument("--natural-dir", type=Path, action="append", required=True)
    parser.add_argument("--longbench-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETE" and manifest.get("contract") == "OLMO_NATIVE_Z5_ASSETS_V2":
            print(json.dumps({"status": "SKIP_COMPLETE", "manifest": str(manifest_path)}))
            return
    output.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    source = pq.read_table(args.pg19_validation.resolve()).to_pylist()
    books = []
    for row in source:
        text = str(row["text"])
        source_hash = sha256_text(text)
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) < 4097:
            continue
        span = len(ids) - 4097 + 1
        offset = int(source_hash[:16], 16) % span
        books.append((source_hash, row, ids[offset:offset + 4097], offset, len(ids)))
    books.sort(key=lambda item: item[0])
    if len(books) < 50:
        raise ValueError(f"PG19 validation has only {len(books)} books with 4097 OLMo tokens")
    books = books[:50]
    tokens = np.asarray([item[2] for item in books], dtype=np.int64)
    npy_path = output / "pg19_validation_50x4097.npy"
    temporary = npy_path.with_name(npy_path.name + ".incomplete")
    with temporary.open("wb") as stream:
        np.save(stream, tokens, allow_pickle=False)
    os.replace(temporary, npy_path)
    split_names = ["design"] * 16 + ["selection"] * 16 + ["internal_confirm"] * 18
    book_records = [
        {
            "row": index,
            "split": split_names[index],
            "source_sha256": source_hash,
            "short_book_title": str(row["short_book_title"]),
            "url": str(row["url"]),
            "token_offset": offset,
            "available_tokens": available,
            "used_tokens": 4097,
        }
        for index, (source_hash, row, _, offset, available) in enumerate(books)
    ]

    ppl_manifest = json.loads(args.ppl_manifest.read_text())
    ppl = np.load(args.ppl_array, mmap_mode="r", allow_pickle=False)
    if ppl_manifest.get("contract") != "TAILSPLINE_OLMO_PPL46_V1" or tuple(ppl.shape) != (46, 16385):
        raise ValueError("held-out PPL46 asset identity drift")
    if {record["dataset"] for record in ppl_manifest["document_records"]} != {"proofpile", "pg19"}:
        raise ValueError("held-out PPL46 source identity drift")

    ruler_all = read_jsonl(args.ruler_panel.resolve())
    ruler = [row for row in ruler_all if int(row["length_cap"]) == 4096]
    if len(ruler) != 130 or Counter(row["task"] for row in ruler) != Counter({task: 10 for task in TASKS}):
        raise ValueError("OLMo Native-4K RULER panel is not Full-13×10")

    natural_all = []
    seen = set()
    for directory in args.natural_dir:
        for row in read_jsonl(directory.resolve() / "screen.jsonl"):
            row_id = str(row["row_id"])
            if row_id in seen:
                raise ValueError(f"duplicate natural-QA source row: {row_id}")
            seen.add(row_id)
            natural_all.append(row)
    if len(natural_all) != 778:
        raise ValueError("frozen natural-QA union is not 778 rows")
    raw_by_task = {}
    with zipfile.ZipFile(args.longbench_archive.resolve()) as archive:
        for task in NATURAL_TASKS:
            raw_by_task[task] = [
                json.loads(line)
                for line in archive.read(f"data/{task}.jsonl").decode().splitlines()
                if line
            ]
    natural = []
    for row in natural_all:
        if row["task"] not in NATURAL_TASKS:
            continue
        if len(row["prompt_ids"]) + int(row["max_new_tokens"]) > 4096:
            continue
        original = raw_by_task[row["task"]][int(row["source_row_index"])]
        if list(original["answers"]) != list(row["references"]):
            raise ValueError(f"natural-QA source reference drift: {row['row_id']}")
        context_cluster = sha256_text(original["context"])
        natural.append({
            **row,
            "length_cap": 4096,
            "document_cluster_id": context_cluster,
            "source_document_id": context_cluster,
            "source_context_sha256": context_cluster,
        })
    natural_counts = Counter(row["task"] for row in natural)
    if len(natural) != 99 or natural_counts != Counter({"hotpotqa": 5, "2wikimqa": 24, "qasper": 70}):
        raise ValueError(f"Native-4K natural-QA identity drift: {natural_counts}")

    ruler_path = output / "ruler4k_full13x10.jsonl"
    natural_path = output / "natural4k_three_task.jsonl"
    write_jsonl(ruler_path, ruler)
    write_jsonl(natural_path, natural)
    manifest = {
        "status": "COMPLETE",
        "contract": "OLMO_NATIVE_Z5_ASSETS_V2",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model.resolve() / "config.json"),
        "native_length": 4096,
        "optimization": {
            "source": "PG19 validation split",
            "source_file": str(args.pg19_validation.resolve()),
            "source_file_sha256": sha256_file(args.pg19_validation.resolve()),
            "rows": 50,
            "split_counts": dict(Counter(split_names)),
            "tokens_shape": list(tokens.shape),
            "tokens_sha256": sha256_file(npy_path),
            "records": book_records,
            "selection_uses_task_outputs": False,
        },
        "heldout_nll": {
            "manifest": str(args.ppl_manifest.resolve()),
            "manifest_sha256": sha256_file(args.ppl_manifest.resolve()),
            "array": str(args.ppl_array.resolve()),
            "array_sha256": sha256_file(args.ppl_array.resolve()),
            "documents": 46,
            "lengths": [1024, 2048, 4096],
            "source_split": "ProofPile test and PG19 test; disjoint from PG19 validation optimization books",
        },
        "ruler": {"rows": 130, "rows_by_task": {task: 10 for task in TASKS}, "sha256": sha256_file(ruler_path)},
        "natural_qa": {
            "rows": 99,
            "rows_by_task": dict(natural_counts),
            "sha256": sha256_file(natural_path),
            "document_cluster": "source_context_sha256",
            "source_archive_sha256": sha256_file(args.longbench_archive.resolve()),
        },
        "scope": "Frozen OLMo-2-1B Native-window z-only optimization; no model-weight update and no extrapolation objective.",
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "optimization_rows": 50, "ruler_rows": 130, "natural_rows": 99}, sort_keys=True))


if __name__ == "__main__":
    main()
