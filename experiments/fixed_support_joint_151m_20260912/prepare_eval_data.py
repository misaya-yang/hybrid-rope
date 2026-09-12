#!/usr/bin/env python3
"""Prepare an independent 16K-capable document panel from frozen shard 004."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


SOURCE_SHA256 = "33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b"
TOKENIZER_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
TOKENS_PER_DOCUMENT = 16_385
DOCUMENTS = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("validation source shard identity changed")
    tokenizer_path = args.tokenizer / "tokenizer.json"
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity changed")
    excluded = set(
        json.loads(args.exclude_manifest.read_text())["validation"]["document_ids"]
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    documents = []
    document_ids = []
    content_hashes = set()
    offset = 0
    parquet = pq.ParquetFile(args.source)
    for batch in parquet.iter_batches(batch_size=512, columns=["text"]):
        texts = batch.column(0).to_pylist()
        rows = tokenizer.encode_batch(texts, add_special_tokens=False)
        for local_index, row in enumerate(rows):
            document_id = offset + local_index
            if document_id in excluded or len(row.ids) < TOKENS_PER_DOCUMENT:
                continue
            ids = np.asarray(row.ids[:TOKENS_PER_DOCUMENT], dtype=np.int64)
            if ids.min() < 0 or ids.max() >= 50_304:
                raise ValueError("token outside the locked vocabulary")
            stored = ids.astype(np.uint16)
            content_sha = hashlib.sha256(stored.tobytes()).hexdigest()
            if content_sha in content_hashes:
                continue
            content_hashes.add(content_sha)
            documents.append(stored)
            document_ids.append(document_id)
            if len(documents) == DOCUMENTS:
                break
        offset += len(rows)
        if len(documents) == DOCUMENTS:
            break
    if len(documents) != DOCUMENTS:
        raise RuntimeError(f"only found {len(documents)} independent 16K documents")
    data_path = args.output / "documents.npy"
    temporary = args.output / "documents.incomplete.npy"
    np.save(temporary, np.stack(documents))
    os.replace(temporary, data_path)
    manifest = {
        "status": "READY",
        "selection": "first 512 unique shard004 documents with >=16385 tokens, excluding prior 8K panel document IDs",
        "source": str(args.source.resolve()),
        "source_sha256": SOURCE_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "excluded_manifest": str(args.exclude_manifest.resolve()),
        "documents": str(data_path.resolve()),
        "documents_sha256": sha256_file(data_path),
        "document_ids": document_ids,
        "count": DOCUMENTS,
        "tokens_per_document": TOKENS_PER_DOCUMENT,
        "evaluation_lengths": [2_048, 4_096, 8_192, 16_384],
        "shared_endpoint": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "READY", "documents": DOCUMENTS}, sort_keys=True))


if __name__ == "__main__":
    main()
