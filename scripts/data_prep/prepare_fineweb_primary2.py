#!/usr/bin/env python3
"""Freeze the pinned FineWeb-Edu stream used by fresh Primary-II reruns.

The selection intentionally mirrors ``run_evq_sweep.py``:

* training reads the unshuffled stream until a document crosses the requested
  token budget, then keeps every complete ``seq_len`` chunk;
* validation applies the historical iterable shuffle (seed 99999, buffer
  10000), stops after the crossing document, and keeps that full document;
* documents are tokenized independently with no added special tokens.

Unlike the historical online loader, this tool is local-only, rejects fallback
datasets, validates the pinned source receipts, and writes a provenance
manifest with content hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from array import array
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import __version__ as datasets_version
from datasets import load_dataset
from transformers import AutoTokenizer, __version__ as transformers_version


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
REPO_ID = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SHARDS = (
    (
        "000_00000.parquet",
        2_152_819_114,
        "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
    ),
    (
        "001_00000.parquet",
        2_152_222_432,
        "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693",
    ),
    (
        "002_00000.parquet",
        2_151_796_315,
        "547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7",
    ),
    (
        "003_00000.parquet",
        2_152_437_524,
        "22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb",
    ),
)
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
TOKENIZER_REPO_ID = "EleutherAI/gpt-neox-20b"
TOKENIZER_FILES = {
    "special_tokens_map.json": "c0b3c279b6ecdb71996a86ffb4d4ab94dfdb5df95f00bac9515688faef2ff5dd",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "tokenizer_config.json": "6c50f7b37042a059c71a347be27bc53cc4fcb0f6c8166b00712f3937c91e6bc7",
}
PRIMARY2_TRAIN_TOKENS = 15_000_000
PRIMARY2_VAL_TOKENS = 5_000_000
PRIMARY2_SEQ_LEN = 128
PRIMARY2_VAL_SHUFFLE_SEED = 99_999
PRIMARY2_VAL_SHUFFLE_BUFFER = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=15_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--val-shuffle-seed", type=int, default=99_999)
    parser.add_argument("--val-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--progress-tokens", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_tokens <= 0 or args.val_tokens <= 0 or args.seq_len <= 0:
        parser.error("token budgets and seq_len must be positive")
    locked = (
        args.train_tokens == PRIMARY2_TRAIN_TOKENS
        and args.val_tokens == PRIMARY2_VAL_TOKENS
        and args.seq_len == PRIMARY2_SEQ_LEN
        and args.val_shuffle_seed == PRIMARY2_VAL_SHUFFLE_SEED
        and args.val_shuffle_buffer == PRIMARY2_VAL_SHUFFLE_BUFFER
    )
    if not locked:
        parser.error(
            "Primary-II preparation is protocol-locked to train=15M, val=5M, "
            "seq_len=128, val seed=99999, and val buffer=10000"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(source_root: Path) -> list[dict[str, Any]]:
    source_root = source_root.resolve()
    verified_root = source_root / ".verified"
    records: list[dict[str, Any]] = []
    for name, expected_size, expected_hash in SHARDS:
        path = source_root / "sample" / "10BT" / name
        receipt = verified_root / f"{name}.sha256"
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned shard: {path}")
        stat = path.stat()
        if stat.st_size != expected_size:
            raise ValueError(
                f"size mismatch for {path}: {stat.st_size} != {expected_size}"
            )
        expected_receipt = (
            f"{expected_hash} {stat.st_size} {int(stat.st_mtime)} "
            f"{int(stat.st_ctime)} {stat.st_ino}"
        )
        if not receipt.is_file() or receipt.read_text().strip() != expected_receipt:
            raise ValueError(
                f"verified receipt missing or stale for {path}; run the downloader verify action"
            )
        records.append(
            {
                "name": name,
                "size": expected_size,
                "sha256": expected_hash,
                "receipt": receipt.name,
                "receipt_content": expected_receipt,
            }
        )
    return records


def validate_tokenizer(tokenizer_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, expected_hash in TOKENIZER_FILES.items():
        path = tokenizer_path / name
        if not path.is_file():
            raise FileNotFoundError(f"missing tokenizer file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"tokenizer hash mismatch for {path}: {actual_hash} != {expected_hash}"
            )
        records.append(
            {"name": name, "size": path.stat().st_size, "sha256": actual_hash}
        )
    return records


def local_stream(shard_paths: list[str]) -> Iterable[dict[str, Any]]:
    return load_dataset(
        "parquet",
        data_files={"train": shard_paths},
        split="train",
        streaming=True,
    )


def collect_tokens(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    requested_tokens: int,
    progress_tokens: int,
    label: str,
) -> tuple[array, int, float]:
    ids = array("I")
    documents = 0
    next_report = progress_tokens
    started = time.monotonic()
    for row in rows:
        text = row.get("text")
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.extend(encoded)
        documents += 1
        if len(ids) >= next_report:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = len(ids) / elapsed
            remaining = max(requested_tokens - len(ids), 0)
            print(
                f"[{label}] {len(ids):,}/{requested_tokens:,} tokens; "
                f"documents={documents:,}; rate={rate/1e6:.3f}M tok/s; "
                f"ETA={remaining/rate:.0f}s",
                flush=True,
            )
            while next_report <= len(ids):
                next_report += progress_tokens
        if len(ids) >= requested_tokens:
            break
    elapsed = time.monotonic() - started
    if len(ids) < requested_tokens:
        raise RuntimeError(
            f"{label} exhausted after {len(ids):,} tokens, below {requested_tokens:,}"
        )
    return ids, documents, elapsed


def tensor_from_ids(ids: array) -> torch.Tensor:
    values = np.frombuffer(ids, dtype=np.uint32).astype(np.int64, copy=True)
    return torch.from_numpy(values)


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".incomplete")
    torch.save(tensor, temp)
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source_root = args.source_root.resolve()
    tokenizer_path = args.tokenizer_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / (
        f"train_fineweb-edu_{args.train_tokens}_{args.seq_len}.pt"
    )
    val_path = output_dir / f"val_fineweb-edu_{args.val_tokens}.pt"
    manifest_path = output_dir / "manifest_primary2_fineweb.json"
    for path in (train_path, val_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    source_records = validate_sources(source_root)
    tokenizer_records = validate_tokenizer(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    shard_paths = [str(source_root / "sample" / "10BT" / row[0]) for row in SHARDS]

    train_ids, train_documents, train_seconds = collect_tokens(
        local_stream(shard_paths),
        tokenizer,
        args.train_tokens,
        args.progress_tokens,
        "train",
    )
    train_flat = tensor_from_ids(train_ids)
    train_chunks = train_flat.numel() // args.seq_len
    train_tensor = train_flat[: train_chunks * args.seq_len].view(
        train_chunks, args.seq_len
    )
    atomic_torch_save(train_tensor, train_path)
    del train_flat, train_tensor, train_ids

    val_rows = local_stream(shard_paths).shuffle(
        seed=args.val_shuffle_seed,
        buffer_size=args.val_shuffle_buffer,
    )
    val_ids, val_documents, val_seconds = collect_tokens(
        val_rows,
        tokenizer,
        args.val_tokens,
        args.progress_tokens,
        "validation",
    )
    val_tensor = tensor_from_ids(val_ids)
    atomic_torch_save(val_tensor, val_path)
    del val_ids

    train_saved = torch.load(train_path, map_location="cpu", weights_only=True)
    val_saved = torch.load(val_path, map_location="cpu", weights_only=True)
    manifest = {
        "schema": "evq_cosh.primary2_fineweb.v1",
        "created_at_unix": int(time.time()),
        "source": {
            "repo_id": REPO_ID,
            "config": CONFIG,
            "revision": REVISION,
            "field": "text",
            "local_root_basename": source_root.name,
            "shards_in_order": source_records,
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "local_root_basename": tokenizer_path.name,
            "files": tokenizer_records,
            "vocab_size": len(tokenizer),
            "add_special_tokens": False,
        },
        "selection": {
            "train": "unshuffled rows; stop after crossing document; keep full seq_len chunks",
            "validation": "iterable shuffle; stop after crossing document; keep crossing document",
            "validation_shuffle_seed": args.val_shuffle_seed,
            "validation_shuffle_buffer": args.val_shuffle_buffer,
            "train_validation_disjointness": "not guaranteed; mirrors historical loader",
        },
        "consumer_contract": {
            "artifact_role": "fresh pinned Primary-II reconstruction",
            "historical_byte_identity": False,
            "historical_identity_limit": (
                "the original Hub revision, shard topology, and datasets/transformers "
                "versions were not recorded"
            ),
            "required_run_policy": (
                "rerun every compared method and seed, including seed 42, on this one "
                "manifest; validate manifest and tensor hashes before launch"
            ),
            "cache_is_immutable": True,
            "incompatible_entrypoint": (
                "phase11b_125m_dape.py is L=256/100M and must not consume this cache"
            ),
        },
        "outputs": {
            "train": {
                "filename": train_path.name,
                "requested_tokens": args.train_tokens,
                "actual_tokens": int(train_saved.numel()),
                "shape": list(train_saved.shape),
                "dtype": str(train_saved.dtype),
                "documents_consumed": train_documents,
                "seconds": train_seconds,
                "file_size": train_path.stat().st_size,
                "file_sha256": sha256_file(train_path),
                "content_sha256_int64": tensor_content_sha256(train_saved),
            },
            "validation": {
                "filename": val_path.name,
                "requested_tokens": args.val_tokens,
                "actual_tokens": int(val_saved.numel()),
                "shape": list(val_saved.shape),
                "dtype": str(val_saved.dtype),
                "documents_consumed": val_documents,
                "seconds": val_seconds,
                "file_size": val_path.stat().st_size,
                "file_sha256": sha256_file(val_path),
                "content_sha256_int64": tensor_content_sha256(val_saved),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "datasets": datasets_version,
            "numpy": np.__version__,
            "byteorder": sys.byteorder,
        },
    }
    temp_manifest = manifest_path.with_suffix(".json.incomplete")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
