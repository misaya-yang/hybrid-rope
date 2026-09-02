#!/usr/bin/env python3
"""Prepare fixed fresh Qwen K32 32K/64K packed-natural NLL rows; no model use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]

GRID = (32768, 65536)
DOCUMENTS = 32
TARGET_TOKENS = 256
STATUS = "QWEN_K32_PACKED_NATURAL_NLL_DATA_READY_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def tokenizer_file_receipts(checkpoint: Path) -> list[dict]:
    paths = sorted(path for path in checkpoint.iterdir()
                   if path.is_file() and (path.name.startswith("tokenizer")
                                          or path.name == "special_tokens_map.json"))
    if not paths:
        raise ValueError("checkpoint tokenizer files are unavailable")
    return [{"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths]


def iter_source_texts(source: Path, start_row: int) -> Iterable[tuple[int, str]]:
    import pyarrow.parquet as pq

    cursor = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64, columns=["text"]):
        for offset, text in enumerate(batch.column(0).to_pylist()):
            row = cursor + offset
            if row >= start_row and isinstance(text, str) and text:
                yield row, text
        cursor += batch.num_rows


def pack_streams(rows: Iterable[tuple[int, str]], tokenizer: Any) -> list[dict]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer must provide EOS for document packing")
    eos = int(tokenizer.eos_token_id)
    streams: list[dict] = []
    current: list[int] = []
    sources: list[dict] = []
    seen: set[str] = set()
    for source_row, text in rows:
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if source_hash in seen:
            continue
        seen.add(source_hash)
        encoded = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if not encoded:
            continue
        remaining = GRID[-1] - len(current)
        complete_document = len(encoded) + 1 <= remaining
        contribution = len(encoded) if complete_document else remaining
        current.extend(encoded[:contribution])
        if complete_document:
            current.append(eos)
        sources.append({"source_row": int(source_row), "source_text_sha256": source_hash,
                        "document_tokens_used": contribution, "document_tokens_available": len(encoded),
                        "truncated_to_finish_stream": not complete_document})
        if len(current) == GRID[-1]:
            streams.append({"input_ids": current, "sources": sources})
            current, sources = [], []
            if len(streams) == DOCUMENTS:
                return streams
    raise ValueError(f"need {DOCUMENTS} complete 65536-token streams; found {len(streams)}")


def paired_rows(streams: list[dict]) -> list[dict]:
    if len(streams) != DOCUMENTS:
        raise ValueError("packed-natural contract requires exactly 32 streams")
    rows = []
    for index, stream in enumerate(streams):
        long_ids = stream["input_ids"]
        if len(long_ids) != GRID[-1]:
            raise ValueError("invalid packed stream length")
        source_hashes = [item["source_text_sha256"] for item in stream["sources"]]
        source_rows = [item["source_row"] for item in stream["sources"]]
        for length in GRID:
            ids = list(long_ids[-length:])
            rows.append({"family": "natural", "variant": "packed_natural",
                         "split": "holdout", "sample_id": f"qwen-k32-natural-{index:03d}",
                         "length": length, "input_ids": ids,
                         "target_start": length - TARGET_TOKENS,
                         "target_tokens": TARGET_TOKENS,
                         "prompt_ids_sha256": ids_hash(ids),
                         "target_ids_sha256": ids_hash(ids[-TARGET_TOKENS:]),
                         "source_row_start": min(source_rows), "source_row_end": max(source_rows),
                         "source_document_count": len(source_rows),
                         "source_set_sha256": canonical_hash(source_hashes)})
    return rows


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("output directory must be fresh")
    source, checkpoint = args.source.resolve(), args.checkpoint.resolve()
    if sha256_file(source) != args.expected_source_sha256:
        raise ValueError("source parquet SHA-256 mismatch")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    head_dim = int(head_dim)
    if (config.get("model_type") != "qwen2" or head_dim != 64
            or int(config.get("max_position_embeddings", 0)) != GRID[0]
            or config.get("rope_scaling") not in (None, {})
            or config.get("use_sliding_window", False) is not False):
        raise ValueError("checkpoint is not the admitted unscaled Qwen K32 geometry")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=False)
    streams = pack_streams(iter_source_texts(source, args.start_row), tokenizer)
    rows = paired_rows(streams)
    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / "rows.jsonl"
    with rows_path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    all_sources = [source for stream in streams for source in stream["sources"]]
    manifest = {
        "status": STATUS, "model_evaluation_status": "NOT_RUN", "model_outcomes_read": False,
        "grid": list(GRID), "natural_streams": DOCUMENTS, "rows": len(rows),
        "target_tokens": TARGET_TOKENS, "config_sha256": sha256_file(config_path),
        "tokenizer_files": tokenizer_file_receipts(checkpoint),
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "source": {"name": source.name, "sha256": sha256_file(source),
                   "start_row": int(args.start_row),
                   "consumed_row_range": [min(item["source_row"] for item in all_sources),
                                          max(item["source_row"] for item in all_sources)]},
        "file": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
        "source_document_set_sha256": canonical_hash(sorted(item["source_text_sha256"]
                                                              for item in all_sources)),
        "packing_contract": (
            "source-order unique documents; insert one EOS between complete documents; truncate only "
            "the last document to finish each 65536-token stream and discard its unused suffix; never "
            "reuse a source document; 32768 is the suffix of the paired 65536 stream"
        ),
        "stream_receipts": [{"sample_id": f"qwen-k32-natural-{index:03d}",
                             "source_row_range": [items[0]["source_row"], items[-1]["source_row"]],
                             "source_documents": len(items),
                             "source_set_sha256": canonical_hash([item["source_text_sha256"]
                                                                   for item in items]),
                             "long_ids_sha256": ids_hash(stream["input_ids"])}
                            for index, stream in enumerate(streams) for items in [stream["sources"]]],
        "selection": "first 32 complete packed streams from the fixed source row; no model scores",
        "script_sha256": sha256_file(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
