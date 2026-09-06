#!/usr/bin/env python3
"""Compile pinned FineWeb-Edu parquet into three contiguous 1B-token parts.

The output is a pure, unshuffled GPT-NeoX core-text stream. Each part contains
exactly 1,000,000,000 tokens as a one-dimensional int32 PyTorch tensor. Parts
are consecutive half-open ranges of one global stream; they are not independent
seed datasets. Consumers must reshape a shared prefix for their sequence length
and cast training batches to int64.

Tokenization is CPU-only. The compiler reads parquet shards and rows in a fixed
order, batches the fast tokenizer without reordering results, writes resumable
raw int32 files, then converts each independent backing file to a Zip64-capable
``torch.save`` artifact. It never downloads or falls back to another dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers
import torch
import transformers
from transformers import AutoTokenizer

from prepare_fineweb_primary2 import (
    CONFIG,
    REPO_ID,
    REVISION,
    SHARDS as PRIMARY2_SHARDS,
    TOKENIZER_FILES,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    sha256_file,
    validate_tokenizer,
)


PART_TOKENS = 1_000_000_000
NUM_PARTS = 3
TOTAL_TOKENS = PART_TOKENS * NUM_PARTS
MODEL_VOCAB_SIZE = 50_304
RAW_BYTES_PER_PART = PART_TOKENS * np.dtype("<i4").itemsize
PROTOCOL_SCHEMA = "evq_cosh.fineweb_neox_3x1b.v1"
FOUR_SHARD_CONFIG_SHA256 = (
    "78af5959fd15ea13c6ab04b72045acbd13e6bad8232c710e6eb4972498f91030"
)
FOUR_SHARD_SCRIPT_SHA256 = (
    "031e1dffd8a81b8d5053f27178d0245f4d2f65c7a46b9b9aecac9ebcdf7fe9de"
)
FOUR_SHARD_EXHAUSTED_TOKENS = 2_991_788_221
SHARDS = PRIMARY2_SHARDS + (
    (
        "004_00000.parquet",
        2_152_338_550,
        "33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--checkpoint-tokens", type=int, default=10_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--accept-four-shard-extension",
        action="store_true",
        help=(
            "resume the audited four-shard exhausted checkpoint by appending "
            "pinned shard 004; all state fields must match exactly"
        ),
    )
    args = parser.parse_args()
    if args.batch_rows <= 0 or args.checkpoint_tokens <= 0:
        parser.error("batch-rows and checkpoint-tokens must be positive")
    return args


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".incomplete")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def validate_sources(source_root: Path) -> list[dict[str, Any]]:
    """Require verified receipts for every ordered source shard."""
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
                f"verified receipt missing or stale for {path}; run downloader verify"
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


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "source_repo": REPO_ID,
        "source_config": CONFIG,
        "source_revision": REVISION,
        "ordered_shards": [row[0] for row in SHARDS],
        "tokenizer_repo": TOKENIZER_REPO_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files": TOKENIZER_FILES,
        "parts": NUM_PARTS,
        "tokens_per_part": PART_TOKENS,
        "dtype": "int32",
        "byteorder": "little",
        "text_field": "text",
        "row_order": "shard -> row_group -> row",
        "document_separator": None,
        "add_special_tokens": False,
        "batch_rows": args.batch_rows,
    }


def config_sha256(args: argparse.Namespace) -> str:
    blob = json.dumps(config_payload(args), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def part_bin(output_dir: Path, index: int) -> Path:
    return output_dir / f"fineweb-edu_neox_part{index + 1:02d}_1b.int32.incomplete"


def part_pt(output_dir: Path, index: int) -> Path:
    return output_dir / f"fineweb-edu_neox_part{index + 1:02d}_1b.pt"


def part_receipt(output_dir: Path, index: int) -> Path:
    return output_dir / f"fineweb-edu_neox_part{index + 1:02d}_1b.receipt.json"


def progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.json"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest_fineweb_neox_3x1b.json"


def new_progress(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "phase": "tokenizing",
        "config_sha256": config_sha256(args),
        "script_sha256": script_sha256(),
        "created_at_unix": int(time.time()),
        "updated_at_unix": int(time.time()),
        "source_cursor": {"shard_index": 0, "next_row": 0},
        "documents_consumed": 0,
        "part_tokens": [0] * NUM_PARTS,
        "part_boundaries": [
            {"start": None, "end_exclusive": None} for _ in range(NUM_PARTS)
        ],
        "completed_parts": [],
        "part_artifacts": {},
    }


def progress_total(state: dict[str, Any]) -> int:
    return sum(int(x) for x in state["part_tokens"])


def print_status(output_dir: Path) -> None:
    state_file = progress_path(output_dir)
    manifest_file = manifest_path(output_dir)
    if manifest_file.is_file():
        payload = json.loads(manifest_file.read_text())
        print("FineWeb-Edu GPT-NeoX 3x1B: COMPLETE")
        print(f"manifest_sha256: {sha256_file(manifest_file)}")
        for part in payload["parts"]:
            print(
                f"part{part['part_index']:02d} "
                f"{part['token_count']:,}/{PART_TOKENS:,} verified "
                f"file_sha256={part['file_sha256']}"
            )
        return
    if not state_file.is_file():
        print("FineWeb-Edu GPT-NeoX 3x1B: not started")
        return
    state = json.loads(state_file.read_text())
    print(f"FineWeb-Edu GPT-NeoX 3x1B: {state['phase']}")
    for index, count in enumerate(state["part_tokens"], 1):
        pct = 100.0 * int(count) / PART_TOKENS
        width = 30
        filled = min(width, int(pct * width / 100.0))
        print(
            f"part{index:02d} [{'#' * filled}{'-' * (width - filled)}] "
            f"{pct:6.2f}% {int(count):,}/{PART_TOKENS:,}"
        )
    print(f"total: {progress_total(state):,}/{TOTAL_TOKENS:,}")
    print(f"documents: {state['documents_consumed']:,}")
    print(f"source_cursor: {state['source_cursor']}")


def validate_or_initialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = progress_path(output_dir)
    manifest_file = manifest_path(output_dir)
    if args.overwrite:
        for index in range(NUM_PARTS):
            for path in (
                part_bin(output_dir, index),
                part_pt(output_dir, index),
                part_pt(output_dir, index).with_suffix(".pt.incomplete"),
                part_receipt(output_dir, index),
            ):
                path.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)
        manifest_file.unlink(missing_ok=True)
    if manifest_file.is_file():
        raise FileExistsError(f"dataset already complete: {manifest_file}")
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        if state.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError("progress schema mismatch")
        hashes_match = (
            state.get("config_sha256") == config_sha256(args)
            and state.get("script_sha256") == script_sha256()
        )
        if not hashes_match:
            expected_legacy = {
                "phase": "tokenizing",
                "config_sha256": FOUR_SHARD_CONFIG_SHA256,
                "script_sha256": FOUR_SHARD_SCRIPT_SHA256,
                "source_cursor": {"shard_index": 3, "next_row": 734_000},
                "documents_consumed": 2_916_000,
                "part_tokens": [
                    PART_TOKENS,
                    PART_TOKENS,
                    FOUR_SHARD_EXHAUSTED_TOKENS - 2 * PART_TOKENS,
                ],
                "completed_parts": [],
            }
            observed_legacy = {key: state.get(key) for key in expected_legacy}
            if not args.accept_four_shard_extension or observed_legacy != expected_legacy:
                raise ValueError(
                    "progress configuration/script mismatch; refusing unsafe resume"
                )
            for index, tokens in enumerate(expected_legacy["part_tokens"]):
                raw_path = part_bin(output_dir, index)
                expected_bytes = int(tokens) * 4
                if not raw_path.is_file() or raw_path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"four-shard checkpoint backing file mismatch: {raw_path}"
                    )
            migration = {
                "type": "append_pinned_source_shard",
                "created_at_unix": int(time.time()),
                "old_config_sha256": state["config_sha256"],
                "old_script_sha256": state["script_sha256"],
                "exhausted_total_tokens": progress_total(state),
                "old_terminal_cursor": state["source_cursor"],
                "added_shard": SHARDS[-1][0],
                "added_shard_sha256": SHARDS[-1][2],
            }
            state.setdefault("migrations", []).append(migration)
            state["source_cursor"] = {"shard_index": 4, "next_row": 0}
            state["config_sha256"] = config_sha256(args)
            state["script_sha256"] = script_sha256()
            state["updated_at_unix"] = int(time.time())
            atomic_json(state_file, state)
    else:
        state = new_progress(args)
        atomic_json(state_file, state)
    for index, tokens in enumerate(state["part_tokens"]):
        path = part_bin(output_dir, index)
        expected_bytes = int(tokens) * 4
        if index in state["completed_parts"]:
            continue
        if not path.exists():
            if expected_bytes:
                raise FileNotFoundError(f"missing partial file: {path}")
            path.touch()
        actual_bytes = path.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(f"partial file shorter than checkpoint: {path}")
        if actual_bytes != expected_bytes:
            with path.open("r+b") as handle:
                handle.truncate(expected_bytes)
    return state


def ensure_disk_space(output_dir: Path, state: dict[str, Any]) -> None:
    remaining_raw = (TOTAL_TOKENS - progress_total(state)) * 4
    final_missing = (NUM_PARTS - len(state["completed_parts"])) * RAW_BYTES_PER_PART
    # At conversion peak, all raw parts plus one additional final tensor coexist.
    required = remaining_raw + min(RAW_BYTES_PER_PART, final_missing) + 2 * 1024**3
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise OSError(
            f"insufficient disk: {free/1024**3:.2f} GiB free, "
            f"{required/1024**3:.2f} GiB required"
        )


def iter_shard_batches(
    path: Path, start_row: int, batch_rows: int
) -> Iterable[tuple[int, int, list[str]]]:
    parquet = pq.ParquetFile(path)
    row_cursor = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
        batch_start = row_cursor
        row_cursor += batch.num_rows
        if row_cursor <= start_row:
            continue
        if batch_start < start_row:
            batch = batch.slice(start_row - batch_start)
            batch_start = start_row
        texts = batch.column(0).to_pylist()
        yield batch_start, batch_start + len(texts), texts


def source_location(
    shard_index: int,
    row_index: int,
    document_ordinal: int,
    within_document_token: int,
) -> dict[str, int | str]:
    return {
        "shard_index": shard_index,
        "shard_name": SHARDS[shard_index][0],
        "row_index": row_index,
        "document_ordinal": document_ordinal,
        "within_document_token": within_document_token,
    }


def write_document_tokens(
    encoded: list[int],
    handles: list[Any],
    state: dict[str, Any],
    shard_index: int,
    row_index: int,
) -> None:
    document_ordinal = int(state["documents_consumed"])
    position = 0
    while position < len(encoded) and progress_total(state) < TOTAL_TOKENS:
        part_index = next(
            i for i, count in enumerate(state["part_tokens"]) if int(count) < PART_TOKENS
        )
        current = int(state["part_tokens"][part_index])
        if current == 0 and state["part_boundaries"][part_index]["start"] is None:
            state["part_boundaries"][part_index]["start"] = source_location(
                shard_index, row_index, document_ordinal, position
            )
        take = min(PART_TOKENS - current, len(encoded) - position)
        values = np.asarray(encoded[position : position + take], dtype="<i4")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= MODEL_VOCAB_SIZE):
            raise ValueError("token id outside padded model vocabulary")
        handles[part_index].write(memoryview(values))
        state["part_tokens"][part_index] = current + take
        position += take
        if int(state["part_tokens"][part_index]) == PART_TOKENS:
            state["part_boundaries"][part_index]["end_exclusive"] = source_location(
                shard_index, row_index, document_ordinal, position
            )
    state["documents_consumed"] = document_ordinal + 1


def checkpoint_bins(
    handles: list[Any], state: dict[str, Any], state_file: Path
) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())
    state["updated_at_unix"] = int(time.time())
    atomic_json(state_file, state)


def tokenize_stream(
    args: argparse.Namespace,
    state: dict[str, Any],
    tokenizer: Any,
    source_root: Path,
) -> None:
    if state["phase"] != "tokenizing":
        return
    output_dir = args.output_dir.resolve()
    state_file = progress_path(output_dir)
    if progress_total(state) == TOTAL_TOKENS:
        # A crash can occur after the final token checkpoint but before the
        # phase transition. Resume directly at conversion without advancing
        # the source cursor or counting another document.
        state["phase"] = "converting"
        state["updated_at_unix"] = int(time.time())
        atomic_json(state_file, state)
        return
    handles = [part_bin(output_dir, index).open("ab", buffering=0) for index in range(NUM_PARTS)]
    last_checkpoint = progress_total(state)
    started = time.monotonic()
    start_total = progress_total(state)
    try:
        initial_shard = int(state["source_cursor"]["shard_index"])
        initial_row = int(state["source_cursor"]["next_row"])
        for shard_index in range(initial_shard, len(SHARDS)):
            shard_path = source_root / "sample" / "10BT" / SHARDS[shard_index][0]
            start_row = initial_row if shard_index == initial_shard else 0
            for batch_start, batch_end, texts in iter_shard_batches(
                shard_path, start_row, args.batch_rows
            ):
                nonempty_positions = [i for i, text in enumerate(texts) if text]
                nonempty_texts = [texts[i] for i in nonempty_positions]
                encoded_batch = tokenizer(
                    nonempty_texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                encoded_by_position = dict(zip(nonempty_positions, encoded_batch))
                for offset in range(len(texts)):
                    encoded = encoded_by_position.get(offset)
                    if encoded is None:
                        continue
                    write_document_tokens(
                        encoded,
                        handles,
                        state,
                        shard_index,
                        batch_start + offset,
                    )
                    if progress_total(state) >= TOTAL_TOKENS:
                        break
                state["source_cursor"] = {
                    "shard_index": shard_index,
                    "next_row": batch_end,
                }
                total = progress_total(state)
                if total - last_checkpoint >= args.checkpoint_tokens or total >= TOTAL_TOKENS:
                    checkpoint_bins(handles, state, state_file)
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = (total - start_total) / elapsed
                    eta = (TOTAL_TOKENS - total) / rate if rate > 0 else 0.0
                    print(
                        f"[tokenize] {total:,}/{TOTAL_TOKENS:,} "
                        f"({100*total/TOTAL_TOKENS:.2f}%) "
                        f"docs={state['documents_consumed']:,} "
                        f"rate={rate/1e6:.3f}M tok/s ETA={eta/60:.1f}m",
                        flush=True,
                    )
                    last_checkpoint = total
                if total >= TOTAL_TOKENS:
                    break
            initial_row = 0
            if progress_total(state) >= TOTAL_TOKENS:
                break
        if progress_total(state) != TOTAL_TOKENS:
            checkpoint_bins(handles, state, state_file)
            raise RuntimeError(
                f"pinned shards exhausted at {progress_total(state):,} tokens; "
                f"need {TOTAL_TOKENS:,}; no padding or fallback is allowed"
            )
        state["phase"] = "converting"
        checkpoint_bins(handles, state, state_file)
    finally:
        for handle in handles:
            handle.close()


def hash_raw_int32(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    minimum = MODEL_VOCAB_SIZE
    maximum = -1
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024 * 1024):
            if len(chunk) % 4:
                raise ValueError(f"unaligned int32 file: {path}")
            digest.update(chunk)
            values = np.frombuffer(chunk, dtype="<i4")
            if values.size:
                minimum = min(minimum, int(values.min()))
                maximum = max(maximum, int(values.max()))
    return digest.hexdigest(), minimum, maximum


def convert_parts(args: argparse.Namespace, state: dict[str, Any]) -> None:
    if state["phase"] not in ("converting", "complete"):
        return
    output_dir = args.output_dir.resolve()
    state_file = progress_path(output_dir)
    for index in range(NUM_PARTS):
        if index in state["completed_parts"]:
            part_bin(output_dir, index).unlink(missing_ok=True)
            continue
        raw_path = part_bin(output_dir, index)
        if raw_path.stat().st_size != RAW_BYTES_PER_PART:
            raise ValueError(f"raw part has wrong size: {raw_path}")
        content_hash, minimum, maximum = hash_raw_int32(raw_path)
        if minimum < 0 or maximum >= MODEL_VOCAB_SIZE:
            raise ValueError(f"token range invalid in {raw_path}: {minimum}..{maximum}")
        tensor = torch.from_file(
            str(raw_path), shared=False, size=PART_TOKENS, dtype=torch.int32
        )
        output_path = part_pt(output_dir, index)
        temp_path = output_path.with_suffix(".pt.incomplete")
        print(f"[convert] part {index + 1}/{NUM_PARTS}: torch.save", flush=True)
        torch.save(tensor, temp_path)
        with temp_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        loaded = torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
        if loaded.shape != (PART_TOKENS,) or loaded.dtype != torch.int32:
            raise ValueError(f"saved tensor contract mismatch: {output_path}")
        artifact = {
            "part_index": index + 1,
            "global_token_range": [index * PART_TOKENS, (index + 1) * PART_TOKENS],
            "token_count": PART_TOKENS,
            "shape": [PART_TOKENS],
            "dtype": "torch.int32",
            "byteorder": "little",
            "file": output_path.name,
            "file_size": output_path.stat().st_size,
            "file_sha256": sha256_file(output_path),
            "content_sha256_int32": content_hash,
            "min_token_id": minimum,
            "max_token_id": maximum,
            "boundary": state["part_boundaries"][index],
        }
        atomic_json(part_receipt(output_dir, index), artifact)
        state["part_artifacts"][str(index + 1)] = artifact
        state["completed_parts"].append(index)
        state["completed_parts"] = sorted(set(state["completed_parts"]))
        state["updated_at_unix"] = int(time.time())
        atomic_json(state_file, state)
        del loaded, tensor
        raw_path.unlink()


def aggregate_content_sha256(parts: list[Path]) -> str:
    digest = hashlib.sha256()
    elements_per_chunk = 16 * 1024 * 1024
    for path in parts:
        tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for start in range(0, tensor.numel(), elements_per_chunk):
            values = tensor[start : start + elements_per_chunk].numpy()
            digest.update(memoryview(values))
        del tensor
    return digest.hexdigest()


def finalize_manifest(
    args: argparse.Namespace,
    state: dict[str, Any],
    source_records: list[dict[str, Any]],
    tokenizer_records: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    if len(state["completed_parts"]) != NUM_PARTS:
        return
    output_dir = args.output_dir.resolve()
    parts = [state["part_artifacts"][str(index + 1)] for index in range(NUM_PARTS)]
    for index in range(NUM_PARTS - 1):
        if parts[index]["global_token_range"][1] != parts[index + 1]["global_token_range"][0]:
            raise ValueError("part ranges are not continuous")
        if parts[index]["boundary"]["end_exclusive"] != parts[index + 1]["boundary"]["start"]:
            raise ValueError("part source boundaries are not continuous")
    part_paths = [output_dir / part["file"] for part in parts]
    manifest = {
        "schema": PROTOCOL_SCHEMA,
        "created_at_unix": int(time.time()),
        "source": {
            "repo_id": REPO_ID,
            "config": CONFIG,
            "revision": REVISION,
            "field": "text",
            "shards_in_order": source_records,
            "ordering": "shard -> row_group -> row",
            "empty_text_rule": "skip",
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "files": tokenizer_records,
            "implementation": "PreTrainedTokenizerFast",
            "is_fast": bool(tokenizer.is_fast),
            "vocab_size": len(tokenizer),
            "add_special_tokens": False,
            "document_separator": None,
        },
        "transform": {
            "script": Path(__file__).name,
            "script_sha256": script_sha256(),
            "config": config_payload(args),
            "config_sha256": config_sha256(args),
            "passkey_mix": 0.0,
            "split_rule": "exact global token offsets; a boundary may split a document",
            "continuity": "three consecutive parts; no gaps and no overlaps",
            "documents_consumed": state["documents_consumed"],
            "terminal_source_position": parts[-1]["boundary"]["end_exclusive"],
            "progress_migrations": state.get("migrations", []),
        },
        "parts": parts,
        "aggregate": {
            "total_tokens": TOTAL_TOKENS,
            "ordered_part_files": [path.name for path in part_paths],
            "content_sha256_int32_concatenated": aggregate_content_sha256(part_paths),
        },
        "consumer_contract": {
            "lineage": "GPT-NeoX core-text Primary-I/II/454M",
            "model_vocab_size": MODEL_VOCAB_SIZE,
            "storage": "1D int32 torch tensor",
            "load_rule": "mmap, select a shared prefix, truncate to seq_len multiple, reshape, cast batch to int64",
            "multiseed_rule": "all seeds must consume the same frozen prefix; parts are not seed-specific datasets",
            "validation": "separate artifact required; no train/validation disjointness claim is implied",
            "incompatible_with": "historical GPT-2-tokenized MLA 500M/1B rows",
            "historical_byte_identity": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "byteorder": sys.byteorder,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "unset"),
        },
    }
    final_manifest = manifest_path(output_dir)
    atomic_json(final_manifest, manifest)
    state["phase"] = "complete"
    state["updated_at_unix"] = int(time.time())
    state["manifest_sha256"] = sha256_file(final_manifest)
    atomic_json(progress_path(output_dir), state)
    print_status(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.status:
        print_status(output_dir)
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_root = args.source_root.resolve()
    tokenizer_path = args.tokenizer_path.resolve()
    source_records = validate_sources(source_root)
    tokenizer_records = validate_tokenizer(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast or len(tokenizer) > MODEL_VOCAB_SIZE:
        raise ValueError("tokenizer is not compatible with the core-text model vocabulary")
    state = validate_or_initialize(args)
    ensure_disk_space(output_dir, state)
    tokenize_stream(args, state, tokenizer, source_root)
    convert_parts(args, state)
    finalize_manifest(args, state, source_records, tokenizer_records, tokenizer)


if __name__ == "__main__":
    main()
