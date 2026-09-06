#!/usr/bin/env python3
"""Prepare deterministic pure-text rows for the mature OLMo R0 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.length < 8 or args.rows < 1:
        raise ValueError("length must be >=8 and rows must be positive")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    selected: list[torch.Tensor] = []
    documents: list[dict[str, object]] = []
    row_index = 0
    parquet = pq.ParquetFile(args.parquet)
    for batch in parquet.iter_batches(columns=["text"], batch_size=64):
        for text in batch.column(0).to_pylist():
            current = row_index
            row_index += 1
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < args.length:
                continue
            selected.append(torch.tensor(token_ids[: args.length], dtype=torch.long))
            documents.append({
                "parquet_row": current,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_tokens": len(token_ids),
            })
            if len(selected) == args.rows:
                break
        if len(selected) == args.rows:
            break
    if len(selected) != args.rows:
        raise RuntimeError(
            f"found only {len(selected)} documents with at least {args.length} tokens"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(selected), args.output)
    receipt = {
        "status": "PURE_TEXT_ROWS_READY",
        "selection": "first deterministic parquet rows with at least length tokenizer tokens",
        "model": str(args.model.resolve()),
        "parquet": str(args.parquet.resolve()),
        "parquet_sha256": sha256(args.parquet),
        "length": args.length,
        "rows": args.rows,
        "tensor": str(args.output.resolve()),
        "tensor_sha256": sha256(args.output),
        "shape": [args.rows, args.length],
        "documents": documents,
        "training_or_parameter_updates": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": receipt["status"],
        "tensor": receipt["tensor"],
        "tensor_sha256": receipt["tensor_sha256"],
        "receipt": str(args.receipt.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
