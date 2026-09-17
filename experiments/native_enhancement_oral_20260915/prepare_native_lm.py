#!/usr/bin/env python3
"""Prepare 128 unseen Native-4K LM windows with source-document clustering."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .lm_context import build_manifest


ROWS = 128
TOKENS = 4097


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg19-parquet", type=Path, required=True)
    parser.add_argument("--proofpile-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Native LM confirmation assets are immutable; use a new --out")
    excluded_manifest = json.loads(args.exclude_manifest.read_text())
    excluded = {str(row["source_sha256"]) for row in excluded_manifest["document_records"]}
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    candidates = []
    table = pq.read_table(args.pg19_parquet, columns=["short_book_title", "url", "text"])
    for row in table.to_pylist():
        source_hash = text_sha256(row["text"])
        if source_hash not in excluded:
            candidates.append((source_hash, "pg19", row["short_book_title"], row["text"]))
    for path in sorted(args.proofpile_root.glob("proofpile_test_*.txt")):
        text = path.read_text(errors="strict")
        source_hash = text_sha256(text)
        if source_hash not in excluded:
            candidates.append((source_hash, "proofpile", path.name, text))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    tokenized = []
    for source_hash, dataset, label, text in candidates:
        ids = [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]
        if len(ids) >= TOKENS:
            tokenized.append((source_hash, dataset, label, ids))
    if len(tokenized) < 2 or sum(len(ids) // TOKENS for *_, ids in tokenized) < ROWS:
        raise ValueError("available unseen source documents do not provide 128 nonoverlapping windows")
    windows = []
    # First maximize independent documents, then add at most one second window
    # per document before allowing later windows.
    window_index = 0
    while len(windows) < ROWS:
        added = False
        for source_hash, dataset, label, ids in tokenized:
            start = window_index * TOKENS
            if start + TOKENS <= len(ids):
                windows.append((source_hash, dataset, label, window_index, ids[start:start + TOKENS]))
                added = True
                if len(windows) == ROWS:
                    break
        if not added:
            raise AssertionError("window capacity accounting drift")
        window_index += 1
    matrix = np.asarray([value[-1] for value in windows], dtype=np.int64)
    document_ids = [value[0] for value in windows]
    if set(document_ids) & excluded:
        raise AssertionError("Native LM confirmation overlaps a declared development document")
    args.out.mkdir(parents=True)
    array_path = args.out / "tokens_128x4097.npy"
    np.save(array_path, matrix, allow_pickle=False)
    document_path = args.out / "document_ids.json"
    document_path.write_text(json.dumps(document_ids, indent=2) + "\n")
    excluded_path = args.out / "excluded_document_ids.json"
    excluded_path.write_text(json.dumps(sorted(excluded), indent=2) + "\n")
    source_rows = [{
        "row": index, "document_id": source_hash, "dataset": dataset,
        "source_label": label, "nonoverlapping_window_index": window,
    } for index, (source_hash, dataset, label, window, _ids) in enumerate(windows)]
    (args.out / "sources.json").write_text(json.dumps(source_rows, indent=2, sort_keys=True) + "\n")
    manifest = build_manifest(
        array_path, native_length=4096, split="confirmation",
        document_ids=document_ids, excluded_document_ids=sorted(excluded),
        exclusions_verified=True, recent_history=512, target_tokens=256,
    )
    manifest.update({
        "source_windows": ROWS,
        "independent_source_documents": len(set(document_ids)),
        "source_document_clustering_required": True,
        "pg19_parquet_sha256": file_sha256(args.pg19_parquet),
        "exclude_manifest_sha256": file_sha256(args.exclude_manifest),
        "tokens_sha256": file_sha256(array_path),
        "source_policy": (
            "Use every eligible unseen document once before adding nonoverlapping later "
            "windows; repeated windows retain one document_id statistical cluster."
        ),
    })
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "rows": ROWS,
        "documents": manifest["independent_source_documents"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
