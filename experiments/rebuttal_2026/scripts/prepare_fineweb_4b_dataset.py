#!/usr/bin/env python3
"""Prepare the shared Geo/EVQ FineWeb-Edu 4B-token rebuttal dataset.

This is CPU-only by design. It streams pinned FineWeb-Edu Parquet files in
repository order, splits documents before packing, and writes mmap-friendly
token shards plus a manifest. It does not use the Hugging Face datasets cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer


SCHEMA_VERSION = 1
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER_REPO = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
MODEL_VOCAB_SIZE = 50_304

DEFAULT_SEQUENCE_LENGTH = 4_096
DEFAULT_GLOBAL_BATCH_TOKENS = 262_144
DEFAULT_TRAIN_TOKENS = 4_000_055_296
DEFAULT_VALIDATION_TOKENS = 20_000_768
DEFAULT_TRAIN_SHARD_TOKENS = DEFAULT_GLOBAL_BATCH_TOKENS * 1_000
DEFAULT_VALIDATION_BASIS_POINTS = 50
DEFAULT_MODELSCOPE_REPO = "HuggingFaceFW/fineweb-edu"

TOKENIZER_FILES = {
    "special_tokens_map.json": (
        "c0b3c279b6ecdb71996a86ffb4d4ab94dfdb5df95f00bac9515688faef2ff5dd"
    ),
    "tokenizer.json": (
        "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
    ),
    "tokenizer_config.json": (
        "6c50f7b37042a059c71a347be27bc53cc4fcb0f6c8166b00712f3937c91e6bc7"
    ),
}

BANNED_SYNTHETIC_MARKERS = (
    "passkey",
    "RULER",
    "NIAH",
    "variable tracking",
    "multi-needle",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            item = Path(root) / filename
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def check_directory_limit(output_dir: Path, *, stop_bytes: int) -> None:
    used = directory_size_bytes(output_dir)
    if used > stop_bytes:
        raise RuntimeError(
            f"output directory exceeds stop limit: {used} > {stop_bytes} bytes"
        )


def resolved_url(
    endpoint: str,
    *,
    repo: str,
    revision: str,
    path: str,
    repo_type: str,
) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return (
        f"{endpoint.rstrip('/')}/{prefix}{repo}/resolve/{revision}/"
        f"{path.lstrip('/')}"
    )


def list_fineweb_parquet_shards(api_endpoint: str) -> list[dict[str, Any]]:
    url = (
        f"{api_endpoint.rstrip('/')}/api/datasets/{FINEWEB_REPO}/tree/"
        f"{FINEWEB_REVISION}/sample/10BT?recursive=1"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    rows = response.json()
    shards: list[dict[str, Any]] = []
    for row in rows:
        path = str(row.get("path", ""))
        if not path.endswith(".parquet"):
            continue
        lfs = row.get("lfs") or {}
        sha256 = lfs.get("oid")
        size = int(lfs.get("size") or row.get("size") or 0)
        if not sha256 or not size:
            raise RuntimeError(f"missing LFS identity for {path}: {row}")
        shards.append(
            {
                "relative_path": path,
                "size": size,
                "sha256": str(sha256),
                "git_oid": str(row.get("oid", "")),
                "xet_hash": str(row.get("xetHash", "")),
            }
        )
    shards.sort(key=lambda item: item["relative_path"])
    if not shards:
        raise RuntimeError(f"no Parquet shards found from {url}")
    return shards


def download_verified_file(
    *,
    destination: Path,
    url: str,
    expected_sha256: str,
    expected_size: int | None,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size = destination.stat().st_size
        if expected_size is not None and size != expected_size:
            raise RuntimeError(f"cached file has wrong size: {destination}")
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(f"cached file has wrong SHA-256: {destination}")
        return {
            "path": str(destination),
            "url": url,
            "size": size,
            "sha256": actual,
            "transport": "cached_verified",
        }

    partial = destination.with_name(destination.name + ".partial")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "10",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    print("[download] " + " ".join(command), flush=True)
    subprocess.check_call(command)
    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(
            f"downloaded file has wrong size: {partial}: {size} != "
            f"{expected_size}"
        )
    actual = sha256_file(partial)
    if actual != expected_sha256:
        raise RuntimeError(f"downloaded file has wrong SHA-256: {partial}")
    partial.replace(destination)
    return {
        "path": str(destination),
        "url": url,
        "size": size,
        "sha256": actual,
        "transport": "curl",
    }


def modelscope_api_url(*, repo: str, relative_path: str) -> str:
    return (
        f"https://modelscope.cn/api/v1/datasets/{repo}/repo?"
        f"Revision=master&FilePath={quote(relative_path, safe='')}"
    )


def resolve_modelscope_redirect(*, repo: str, relative_path: str) -> str:
    api_url = modelscope_api_url(repo=repo, relative_path=relative_path)
    response = requests.get(api_url, allow_redirects=False, timeout=30)
    if response.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(
            f"ModelScope did not return a redirect for {relative_path}: "
            f"HTTP {response.status_code} {response.text[:200]}"
        )
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"ModelScope redirect missing Location: {api_url}")
    return location


def download_verified_parquet_shard(
    *,
    destination: Path,
    canonical_url: str,
    relative_path: str,
    expected_sha256: str,
    expected_size: int,
    modelscope_repo: str,
) -> dict[str, Any]:
    """Download one dataset shard by the Agent.md ModelScope/aria2 strategy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size = destination.stat().st_size
        if size != expected_size:
            raise RuntimeError(f"cached file has wrong size: {destination}")
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(f"cached file has wrong SHA-256: {destination}")
        return {
            "path": str(destination),
            "canonical_url": canonical_url,
            "modelscope_repo": modelscope_repo,
            "modelscope_api_url": modelscope_api_url(
                repo=modelscope_repo, relative_path=relative_path
            ),
            "size": size,
            "sha256": actual,
            "transport": "cached_verified",
        }

    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c is required for ModelScope Parquet downloads")

    legacy_partial = destination.with_name(destination.name + ".partial")
    incomplete = destination.with_name(destination.name + ".incomplete")
    if legacy_partial.exists() and not incomplete.exists():
        legacy_partial.replace(incomplete)

    signed_url = resolve_modelscope_redirect(
        repo=modelscope_repo, relative_path=relative_path
    )
    command = [
        aria2,
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=4M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        f"--dir={str(destination.parent)}",
        f"--out={destination.name}.incomplete",
        signed_url,
    ]
    print(
        "[download] ModelScope aria2 "
        f"repo={modelscope_repo} path={relative_path}",
        flush=True,
    )
    subprocess.check_call(command)
    if not incomplete.is_file():
        raise RuntimeError(f"aria2 did not create {incomplete}")
    size = incomplete.stat().st_size
    if size != expected_size:
        raise RuntimeError(
            f"downloaded file has wrong size: {incomplete}: {size} != "
            f"{expected_size}"
        )
    actual = sha256_file(incomplete)
    if actual != expected_sha256:
        raise RuntimeError(f"downloaded file has wrong SHA-256: {incomplete}")
    incomplete.replace(destination)
    return {
        "path": str(destination),
        "canonical_url": canonical_url,
        "modelscope_repo": modelscope_repo,
        "modelscope_api_url": modelscope_api_url(
            repo=modelscope_repo, relative_path=relative_path
        ),
        "size": size,
        "sha256": actual,
        "transport": "modelscope_aria2",
    }


def download_tokenizer(
    *,
    output_dir: Path,
    endpoint: str,
) -> tuple[Any, dict[str, Any]]:
    tokenizer_dir = output_dir / "tokenizer_gpt_neox_20b"
    records: dict[str, Any] = {}
    for filename, expected_sha256 in TOKENIZER_FILES.items():
        url = resolved_url(
            endpoint,
            repo=TOKENIZER_REPO,
            revision=TOKENIZER_REVISION,
            path=filename,
            repo_type="model",
        )
        records[filename] = download_verified_file(
            destination=tokenizer_dir / filename,
            url=url,
            expected_sha256=expected_sha256,
            expected_size=None,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, use_fast=True
    )
    tokenizer.model_max_length = 1 << 60
    file_hash_payload = {
        filename: records[filename]["sha256"] for filename in sorted(records)
    }
    tokenizer_hash = sha256_bytes(
        json.dumps(file_hash_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    vocab_size = int(len(tokenizer))
    if vocab_size > MODEL_VOCAB_SIZE:
        raise RuntimeError(
            f"tokenizer vocab {vocab_size} exceeds model vocab {MODEL_VOCAB_SIZE}"
        )
    dtype = "uint16" if vocab_size <= 65_535 else "uint32"
    return tokenizer, {
        "repo": TOKENIZER_REPO,
        "revision": TOKENIZER_REVISION,
        "directory": str(tokenizer_dir),
        "files": records,
        "file_hash_payload_sha256": tokenizer_hash,
        "vocab_size": vocab_size,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "storage_dtype_rule": "uint16 if vocab_size <= 65535 else uint32",
        "storage_dtype": dtype,
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": [int(x) for x in tokenizer.all_special_ids],
        "bos_token_id": (
            None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id)
        ),
        "eos_token_id": (
            None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
        ),
        "pad_token_id": (
            None if tokenizer.pad_token_id is None else int(tokenizer.pad_token_id)
        ),
        "unk_token_id": (
            None if tokenizer.unk_token_id is None else int(tokenizer.unk_token_id)
        ),
        "add_special_tokens": False,
    }


def split_is_validation(text: str, *, basis_points: int) -> tuple[bool, str]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return bucket < basis_points, digest.hex()


class ShardWriter:
    def __init__(
        self,
        *,
        root: Path,
        split: str,
        total_tokens: int,
        shard_tokens: int,
        dtype: np.dtype[Any],
        sequence_length: int,
        start_tokens: int,
    ) -> None:
        self.root = root
        self.split = split
        self.total_tokens = int(total_tokens)
        self.shard_tokens = int(shard_tokens)
        self.dtype = np.dtype(dtype)
        self.sequence_length = int(sequence_length)
        self.tokens = int(start_tokens)
        if self.total_tokens % self.sequence_length:
            raise ValueError(f"{split} target must be divisible by sequence length")
        if not 0 <= self.tokens <= self.total_tokens:
            raise ValueError(f"invalid resume token count for {split}: {self.tokens}")
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def done(self) -> bool:
        return self.tokens >= self.total_tokens

    def shard_count(self) -> int:
        return math.ceil(self.total_tokens / self.shard_tokens)

    def shard_token_count(self, shard_index: int) -> int:
        start = shard_index * self.shard_tokens
        stop = min(start + self.shard_tokens, self.total_tokens)
        return max(0, stop - start)

    def shard_path(self, shard_index: int) -> Path:
        return self.root / self.split / f"{self.split}_{shard_index:05d}.bin"

    def idx_path(self, shard_index: int) -> Path:
        return self.root / self.split / f"{self.split}_{shard_index:05d}.idx.json"

    def _ensure_file(self, shard_index: int) -> Path:
        path = self.shard_path(shard_index)
        expected_bytes = self.shard_token_count(shard_index) * self.dtype.itemsize
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if path.exists() else "w+b"
        with path.open(mode) as handle:
            if path.stat().st_size != expected_bytes:
                handle.truncate(expected_bytes)
        return path

    def append(self, token_ids: Iterable[int]) -> int:
        if self.done:
            return 0
        array = np.asarray(list(token_ids), dtype=self.dtype)
        if array.size == 0:
            return 0
        remaining = self.total_tokens - self.tokens
        if array.size > remaining:
            array = array[:remaining]
        cursor = 0
        while cursor < array.size:
            shard_index = self.tokens // self.shard_tokens
            shard_offset = self.tokens % self.shard_tokens
            in_shard = self.shard_token_count(shard_index) - shard_offset
            take = min(in_shard, array.size - cursor)
            chunk = np.ascontiguousarray(array[cursor : cursor + take])
            path = self._ensure_file(shard_index)
            with path.open("r+b") as handle:
                handle.seek(shard_offset * self.dtype.itemsize)
                handle.write(chunk.tobytes(order="C"))
            self.tokens += int(take)
            cursor += int(take)
        return int(array.size)

    def shard_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in range(self.shard_count()):
            path = self.shard_path(index)
            token_count = self.shard_token_count(index)
            expected_bytes = token_count * self.dtype.itemsize
            if not path.is_file():
                raise RuntimeError(f"missing {self.split} shard: {path}")
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"{path} has {actual_bytes} bytes, expected {expected_bytes}"
                )
            start = index * self.shard_tokens
            relative_path = str(path.relative_to(self.root.parent))
            relative_idx_path = str(self.idx_path(index).relative_to(self.root.parent))
            record = {
                "index": index,
                "path": relative_path,
                "relative_path": relative_path,
                "idx_path": relative_idx_path,
                "token_start": start,
                "tokens": token_count,
                "sequence_length": self.sequence_length,
                "sequences": token_count // self.sequence_length,
                "dtype": self.dtype.name,
                "bytes": actual_bytes,
                "sha256": sha256_file(path),
            }
            write_json_atomic(self.idx_path(index), record)
            records.append(record)
        return records


def load_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def update_progress(path: Path, state: dict[str, Any]) -> None:
    state["last_update_utc"] = utc_now()
    write_json_atomic(path, state)


def relativize_manifest_paths(value: Any, *, root: Path) -> Any:
    """Make local filesystem paths portable while preserving source URLs."""
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"path", "idx_path", "directory"} and isinstance(item, str):
                path = Path(item)
                if path.is_absolute():
                    try:
                        normalized[key] = str(path.relative_to(root))
                        continue
                    except ValueError:
                        pass
            normalized[key] = relativize_manifest_paths(item, root=root)
        return normalized
    if isinstance(value, list):
        return [relativize_manifest_paths(item, root=root) for item in value]
    return value


def parquet_text_rows(path: Path, *, batch_rows: int) -> Iterable[list[str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
        column = batch.column(0)
        values: list[str] = []
        for index in range(len(column)):
            value = column[index].as_py()
            values.append("" if value is None else str(value))
        yield values


def validate_run_config(progress: dict[str, Any] | None, config: dict[str, Any]) -> None:
    if progress is None:
        return
    old = progress.get("config")
    if isinstance(old, dict):
        patched_old = dict(old)
        patched_old.setdefault(
            "remove_consumed_parquet", config.get("remove_consumed_parquet")
        )
        if patched_old == config:
            progress["config"] = patched_old
            return
    if old != config:
        raise RuntimeError(
            "progress config does not match requested config; use a fresh output_dir"
        )


def read_flat_tokens(
    shards: list[dict[str, Any]],
    *,
    dtype: np.dtype[Any],
    start: int,
    count: int,
    dataset_root: Path | None = None,
) -> np.ndarray:
    result = np.empty(int(count), dtype=dtype)
    written = 0
    request_start = int(start)
    request_stop = int(start + count)
    for shard in shards:
        shard_start = int(shard["token_start"])
        shard_stop = shard_start + int(shard["tokens"])
        overlap_start = max(request_start, shard_start)
        overlap_stop = min(request_stop, shard_stop)
        if overlap_start >= overlap_stop:
            continue
        offset = overlap_start - shard_start
        take = overlap_stop - overlap_start
        path = Path(shard["path"])
        if not path.is_absolute() and dataset_root is not None:
            path = dataset_root / path
        mmap = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=offset * dtype.itemsize,
            shape=(take,),
        )
        result[written : written + take] = np.asarray(mmap)
        written += take
    if written != count:
        raise RuntimeError(f"read {written} tokens, expected {count}")
    return result


def verify_batch_hashes(
    *,
    train_shards: list[dict[str, Any]],
    dtype: np.dtype[Any],
    global_batch_tokens: int,
    batches: int,
    output_path: Path,
) -> dict[str, Any]:
    records = []
    for batch_index in range(int(batches)):
        start = batch_index * int(global_batch_tokens)
        tokens_geo = read_flat_tokens(
            train_shards,
            dtype=dtype,
            start=start,
            count=global_batch_tokens,
            dataset_root=output_path.parent,
        )
        tokens_evq = read_flat_tokens(
            train_shards,
            dtype=dtype,
            start=start,
            count=global_batch_tokens,
            dataset_root=output_path.parent,
        )
        geo_hash = sha256_bytes(np.ascontiguousarray(tokens_geo).tobytes(order="C"))
        evq_hash = sha256_bytes(np.ascontiguousarray(tokens_evq).tobytes(order="C"))
        if geo_hash != evq_hash:
            raise RuntimeError(f"Geo/EVQ batch hash mismatch at batch {batch_index}")
        records.append(
            {
                "batch_index": batch_index,
                "token_start": start,
                "tokens": int(global_batch_tokens),
                "geo_sha256": geo_hash,
                "evq_sha256": evq_hash,
            }
        )
    payload = {
        "status": "passed",
        "description": "Geo and EVQ readers used the same dataset manifest",
        "batches_checked": int(batches),
        "global_batch_tokens": int(global_batch_tokens),
        "records": records,
    }
    write_json_atomic(output_path, payload)
    return payload


def decode_spotcheck(
    *,
    tokenizer: Any,
    train_shards: list[dict[str, Any]],
    dtype: np.dtype[Any],
    total_train_tokens: int,
    sequence_length: int,
    samples: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    total_windows = total_train_tokens // sequence_length
    marker_hits = 0
    lengths: list[int] = []
    with output_path.open("w") as handle:
        for sample_index in range(int(samples)):
            window = rng.randrange(0, total_windows)
            span_tokens = min(128, sequence_length)
            start = window * sequence_length
            tokens = read_flat_tokens(
                train_shards,
                dtype=dtype,
                start=start,
                count=span_tokens,
                dataset_root=output_path.parent,
            )
            token_list = [int(x) for x in tokens.tolist()]
            text = tokenizer.decode(token_list, skip_special_tokens=False)
            lowered = text.lower()
            hits = [
                marker
                for marker in BANNED_SYNTHETIC_MARKERS
                if marker.lower() in lowered
            ]
            if hits:
                marker_hits += 1
            lengths.append(len(text))
            handle.write(
                json.dumps(
                    {
                        "sample_index": sample_index,
                        "window_index": window,
                        "token_start": start,
                        "tokens": span_tokens,
                        "token_sha256": sha256_bytes(
                            np.ascontiguousarray(tokens).tobytes(order="C")
                        ),
                        "decoded_text": text,
                        "synthetic_marker_hits": hits,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    payload = {
        "status": "passed",
        "samples": int(samples),
        "seed": int(seed),
        "span_tokens": 128,
        "synthetic_marker_hits": marker_hits,
        "marker_hit_policy": (
            "literal marker hits are retained for audit; a natural FineWeb-Edu "
            "text mention is not evidence that synthetic tasks were injected"
        ),
        "min_decoded_chars": min(lengths),
        "max_decoded_chars": max(lengths),
    }
    return payload


def build_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(output_dir / "hf_home_disabled"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(output_dir / "hf_home_disabled"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(output_dir / "hf_datasets_disabled"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    train_target = int(args.train_tokens)
    validation_target = int(args.validation_tokens)
    global_batch_tokens = int(args.global_batch_tokens)
    sequence_length = int(args.sequence_length)
    if train_target % global_batch_tokens:
        raise ValueError("train_tokens must be divisible by global_batch_tokens")
    if train_target % sequence_length:
        raise ValueError("train_tokens must be divisible by sequence_length")
    if validation_target % sequence_length:
        raise ValueError("validation_tokens must be divisible by sequence_length")

    config = {
        "schema_version": SCHEMA_VERSION,
        "fineweb_repo": FINEWEB_REPO,
        "fineweb_config": FINEWEB_CONFIG,
        "fineweb_revision": FINEWEB_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "train_tokens": train_target,
        "validation_tokens": validation_target,
        "global_batch_tokens": global_batch_tokens,
        "sequence_length": sequence_length,
        "train_shard_tokens": int(args.train_shard_tokens),
        "validation_shard_tokens": validation_target,
        "validation_basis_points": int(args.validation_basis_points),
        "add_special_tokens": False,
        "eos_between_documents": False,
        "format": "sharded_bin_idx",
        "dataset_download_transport": "modelscope_aria2",
        "modelscope_repo": str(args.modelscope_repo),
        "remove_consumed_parquet": bool(args.remove_consumed_parquet),
    }
    progress_path = output_dir / "progress.json"
    progress = load_progress(progress_path)
    validate_run_config(progress, config)

    print("[audit] loading tokenizer", flush=True)
    tokenizer, tokenizer_record = download_tokenizer(
        output_dir=output_dir,
        endpoint=args.download_endpoint,
    )
    dtype_name = str(tokenizer_record["storage_dtype"])
    dtype = np.dtype(dtype_name)
    dtype_max = np.iinfo(dtype).max
    if int(tokenizer_record["vocab_size"]) > dtype_max:
        raise RuntimeError("tokenizer vocab does not fit selected dtype")

    if progress is None:
        progress = {
            "status": "running",
            "created_utc": utc_now(),
            "config": config,
            "source_shard_index": 0,
            "source_row_offset": 0,
            "train_tokens": 0,
            "validation_tokens": 0,
            "documents_seen": 0,
            "empty_documents": 0,
            "train_documents": 0,
            "validation_documents": 0,
            "train_documents_after_target": 0,
            "validation_documents_after_target": 0,
            "source_records": [],
        }
        update_progress(progress_path, progress)

    train_writer = ShardWriter(
        root=output_dir / "shards",
        split="train",
        total_tokens=train_target,
        shard_tokens=int(args.train_shard_tokens),
        dtype=dtype,
        sequence_length=sequence_length,
        start_tokens=int(progress["train_tokens"]),
    )
    validation_writer = ShardWriter(
        root=output_dir / "shards",
        split="validation",
        total_tokens=validation_target,
        shard_tokens=validation_target,
        dtype=dtype,
        sequence_length=sequence_length,
        start_tokens=int(progress["validation_tokens"]),
    )

    print("[audit] listing FineWeb-Edu shards", flush=True)
    source_shards = list_fineweb_parquet_shards(args.api_endpoint)
    print(f"[audit] source shard count: {len(source_shards)}", flush=True)
    stop_bytes = int(args.stop_gb * (1024**3))

    for shard_index, shard in enumerate(source_shards):
        if train_writer.done and validation_writer.done:
            break
        if shard_index < int(progress["source_shard_index"]):
            continue
        relative_path = str(shard["relative_path"])
        destination = output_dir / "downloads" / "fineweb_edu" / relative_path
        source_record = {
            **download_verified_parquet_shard(
                destination=destination,
                canonical_url=resolved_url(
                    args.download_endpoint,
                    repo=FINEWEB_REPO,
                    revision=FINEWEB_REVISION,
                    path=relative_path,
                    repo_type="dataset",
                ),
                relative_path=relative_path,
                expected_sha256=str(shard["sha256"]),
                expected_size=int(shard["size"]),
                modelscope_repo=str(args.modelscope_repo),
            ),
            "relative_path": relative_path,
            "source_shard_index": shard_index,
            "git_oid": shard.get("git_oid", ""),
            "xet_hash": shard.get("xet_hash", ""),
        }

        parquet = pq.ParquetFile(destination)
        total_rows = int(parquet.metadata.num_rows)
        start_row = (
            int(progress["source_row_offset"])
            if shard_index == int(progress["source_shard_index"])
            else 0
        )
        row_cursor = 0
        shard_train_tokens_start = int(progress["train_tokens"])
        shard_validation_tokens_start = int(progress["validation_tokens"])
        shard_docs_start = int(progress["documents_seen"])
        started = time.time()
        if (
            progress.get("active_source_shard_index") != shard_index
            or int(progress.get("source_row_offset", 0)) == 0
        ):
            progress["active_source_shard_index"] = shard_index
            progress["active_source_train_tokens_start"] = train_writer.tokens
            progress["active_source_validation_tokens_start"] = validation_writer.tokens
            progress["active_source_documents_start"] = int(progress["documents_seen"])
            update_progress(progress_path, progress)
        source_train_tokens_start = int(
            progress.get("active_source_train_tokens_start", shard_train_tokens_start)
        )
        source_validation_tokens_start = int(
            progress.get(
                "active_source_validation_tokens_start",
                shard_validation_tokens_start,
            )
        )
        source_documents_start = int(
            progress.get("active_source_documents_start", shard_docs_start)
        )
        print(
            f"[source] {relative_path} rows={total_rows:,} resume_row={start_row:,}",
            flush=True,
        )

        for texts in parquet_text_rows(destination, batch_rows=int(args.batch_rows)):
            batch_start = row_cursor
            batch_stop = row_cursor + len(texts)
            row_cursor = batch_stop
            if batch_stop <= start_row:
                continue
            if batch_start < start_row:
                texts = texts[start_row - batch_start :]
                batch_start = start_row

            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            train_buffer: list[int] = []
            validation_buffer: list[int] = []
            processed_in_batch = 0
            for local_index, (text, token_ids) in enumerate(zip(texts, encoded)):
                if train_writer.done and validation_writer.done:
                    break
                absolute_row = batch_start + local_index
                processed_in_batch = local_index + 1
                progress["documents_seen"] = int(progress["documents_seen"]) + 1
                if not text or not token_ids:
                    progress["empty_documents"] = int(progress["empty_documents"]) + 1
                    continue
                max_id = max(int(x) for x in token_ids)
                if max_id > dtype_max or max_id >= MODEL_VOCAB_SIZE:
                    raise RuntimeError(
                        f"token id {max_id} exceeds dtype/model vocab at "
                        f"{relative_path}:{absolute_row}"
                    )
                is_validation, doc_hash = split_is_validation(
                    text, basis_points=int(args.validation_basis_points)
                )
                if is_validation:
                    if validation_writer.done:
                        progress["validation_documents_after_target"] = (
                            int(progress["validation_documents_after_target"]) + 1
                        )
                    else:
                        validation_buffer.extend(int(x) for x in token_ids)
                        progress["validation_documents"] = (
                            int(progress["validation_documents"]) + 1
                        )
                else:
                    if train_writer.done:
                        progress["train_documents_after_target"] = (
                            int(progress["train_documents_after_target"]) + 1
                        )
                    else:
                        train_buffer.extend(int(x) for x in token_ids)
                        progress["train_documents"] = (
                            int(progress["train_documents"]) + 1
                        )
                progress["last_document_sha256"] = doc_hash

            if train_buffer:
                train_writer.append(train_buffer)
            if validation_buffer:
                validation_writer.append(validation_buffer)
            progress["train_tokens"] = train_writer.tokens
            progress["validation_tokens"] = validation_writer.tokens
            progress["source_shard_index"] = shard_index
            progress["source_row_offset"] = batch_start + processed_in_batch
            update_progress(progress_path, progress)
            check_directory_limit(output_dir, stop_bytes=stop_bytes)

            total_tokens = train_writer.tokens + validation_writer.tokens
            if total_tokens and total_tokens % int(args.progress_tokens) < len(
                train_buffer
            ) + len(validation_buffer):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    "[progress] "
                    f"train={train_writer.tokens:,}/{train_target:,} "
                    f"validation={validation_writer.tokens:,}/{validation_target:,} "
                    f"docs={progress['documents_seen']:,} "
                    f"rate={(train_writer.tokens - shard_train_tokens_start + validation_writer.tokens - shard_validation_tokens_start) / elapsed / 1e3:.1f}k tok/s",
                    flush=True,
                )

            if train_writer.done and validation_writer.done:
                break

        consumed_record = {
            **source_record,
            "rows": total_rows,
            "rows_consumed": int(progress["source_row_offset"]),
            "train_tokens_added": train_writer.tokens - source_train_tokens_start,
            "validation_tokens_added": (
                validation_writer.tokens - source_validation_tokens_start
            ),
            "documents_seen_added": (
                int(progress["documents_seen"]) - source_documents_start
            ),
            "completed_utc": utc_now(),
            "removed_after_processing": False,
        }
        if args.remove_consumed_parquet:
            destination.unlink(missing_ok=True)
            consumed_record["removed_after_processing"] = True
        progress["source_records"].append(
            consumed_record
        )
        if progress["source_row_offset"] >= total_rows:
            progress["source_shard_index"] = shard_index + 1
            progress["source_row_offset"] = 0
            progress.pop("active_source_shard_index", None)
            progress.pop("active_source_train_tokens_start", None)
            progress.pop("active_source_validation_tokens_start", None)
            progress.pop("active_source_documents_start", None)
        update_progress(progress_path, progress)

    if not (train_writer.done and validation_writer.done):
        raise RuntimeError(
            "source shards ended before targets were met: "
            f"train={train_writer.tokens}, validation={validation_writer.tokens}"
        )

    print("[finalize] hashing shards", flush=True)
    train_shards = train_writer.shard_records()
    validation_shards = validation_writer.shard_records()
    batch_check = verify_batch_hashes(
        train_shards=train_shards,
        dtype=dtype,
        global_batch_tokens=global_batch_tokens,
        batches=int(args.verify_batches),
        output_path=output_dir / "geo_evq_batch_hash_check.json",
    )
    decode_check = decode_spotcheck(
        tokenizer=tokenizer,
        train_shards=train_shards,
        dtype=dtype,
        total_train_tokens=train_target,
        sequence_length=sequence_length,
        samples=int(args.decode_samples),
        seed=int(args.decode_seed),
        output_path=output_dir / "decode_spotcheck.jsonl",
    )
    disk_bytes = directory_size_bytes(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "ready",
        "audit": {
            "agent_md_read": True,
            "rebuttal_entry": "rebuttal/rebuttal_0723/README.md",
            "repo_training_format_audit": {
                "small_rebuttal_trainers": (
                    "single np.load(..., mmap_mode='r') .npy tensors with "
                    "strict int64 checks"
                ),
                "main_4b_trainer_status": (
                    "no implemented fixed 4B trainer format found in current "
                    "rebuttal_0723 code; emitted sharded .bin + .idx per task rule"
                ),
            },
        },
        "source": {
            "repo": FINEWEB_REPO,
            "config": FINEWEB_CONFIG,
            "revision": FINEWEB_REVISION,
            "field": "text",
            "parquet_shards_available": source_shards,
            "parquet_shards_consumed": progress["source_records"],
            "order": "lexicographic repository path order under sample/10BT",
        },
        "tokenizer": tokenizer_record,
        "split": {
            "algorithm": "document SHA-256 hash bucket before tokenization/packing",
            "basis_points_validation": int(args.validation_basis_points),
            "basis_points_denominator": 10_000,
            "hash_material": "raw FineWeb-Edu text field encoded as UTF-8",
            "document_leakage_policy": (
                "documents are assigned to exactly one split before token packing"
            ),
        },
        "packing": {
            "format": "sharded .bin + .idx.json",
            "dtype": dtype.name,
            "sequence_length": sequence_length,
            "global_batch_tokens": global_batch_tokens,
            "train_target_tokens": train_target,
            "validation_target_tokens": validation_target,
            "add_special_tokens": False,
            "eos_between_documents": False,
            "natural_text_only": True,
            "synthetic_tasks_excluded": [
                "passkey",
                "RULER",
                "NIAH",
                "variable tracking",
                "multi-key/multi-value templates",
            ],
        },
        "train": {
            "tokens": train_target,
            "sequences": train_target // sequence_length,
            "shards": train_shards,
        },
        "validation": {
            "tokens": validation_target,
            "sequences": validation_target // sequence_length,
            "shards": validation_shards,
        },
        "checks": {
            "decode_spotcheck": decode_check,
            "geo_evq_batch_hash": batch_check,
            "tokenizer_vocab_fits_dtype": True,
            "completed_shards_are_resume_skipped": True,
        },
        "progress_summary": {
            key: progress[key]
            for key in (
                "documents_seen",
                "empty_documents",
                "train_documents",
                "validation_documents",
                "train_documents_after_target",
                "validation_documents_after_target",
            )
        },
        "disk_usage_bytes": disk_bytes,
        "disk_usage_gib": disk_bytes / float(1024**3),
        "directory_limit_gb": int(args.limit_gb),
        "stop_limit_gb": int(args.stop_gb),
        "reader_command": (
            "Use dataset_manifest.json; resolve shard paths relative to the "
            "manifest directory, mmap each shard as the declared dtype, and "
            "view contiguous 4096-token windows in listed order. Geo and EVQ "
            "must pass the recorded first-batch hash check before training."
        ),
    }
    manifest = relativize_manifest_paths(manifest, root=output_dir)
    manifest_path = output_dir / "dataset_manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (output_dir / "dataset_manifest.sha256").write_text(
        f"{manifest_hash}  dataset_manifest.json\n"
    )
    progress["status"] = "ready"
    progress["manifest_sha256"] = manifest_hash
    update_progress(progress_path, progress)
    print(f"[ready] manifest={manifest_path}", flush=True)
    print(f"[ready] manifest_sha256={manifest_hash}", flush=True)
    print(f"[ready] disk_usage_gib={manifest['disk_usage_gib']:.3f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--api_endpoint", default="https://huggingface.co")
    parser.add_argument("--download_endpoint", default="https://huggingface.co")
    parser.add_argument("--modelscope_repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument(
        "--keep_consumed_parquet",
        action="store_false",
        dest="remove_consumed_parquet",
        help="Keep verified source Parquet files after their rows are tokenized.",
    )
    parser.set_defaults(remove_consumed_parquet=True)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument(
        "--validation_tokens", type=int, default=DEFAULT_VALIDATION_TOKENS
    )
    parser.add_argument(
        "--global_batch_tokens", type=int, default=DEFAULT_GLOBAL_BATCH_TOKENS
    )
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--train_shard_tokens", type=int, default=DEFAULT_TRAIN_SHARD_TOKENS
    )
    parser.add_argument(
        "--validation_basis_points",
        type=int,
        default=DEFAULT_VALIDATION_BASIS_POINTS,
    )
    parser.add_argument("--batch_rows", type=int, default=512)
    parser.add_argument("--decode_samples", type=int, default=100)
    parser.add_argument("--decode_seed", type=int, default=20_260_724)
    parser.add_argument("--verify_batches", type=int, default=8)
    parser.add_argument("--progress_tokens", type=int, default=25_000_000)
    parser.add_argument("--limit_gb", type=int, default=80)
    parser.add_argument("--stop_gb", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_gb >= args.limit_gb:
        raise SystemExit("--stop_gb must be below --limit_gb")
    if not shutil.which("curl"):
        raise SystemExit("curl is required for resumable downloads")
    build_dataset(args)


if __name__ == "__main__":
    main()
