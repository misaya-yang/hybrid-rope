#!/usr/bin/env python3
"""CPU-only manifest preparation for the FMRoPE / EVQ L=256 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from rebuttal.rebuttal_0723.fmrope_125m_l256.protocol import (
    SPEC,
    frequency_contract,
)


SCHEMA_VERSION = 2
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
FRESH_VALIDATION_TOKENS = 5_000_000
FRESH_SOURCE_FILES = {
    "train": {
        "relative_path": "sample/10BT/000_00000.parquet",
        "size": 2_152_819_114,
        "sha256": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
    },
    "validation": {
        "relative_path": "sample/10BT/004_00000.parquet",
        "size": 2_152_338_550,
        "sha256": "33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b",
    },
}
TOKENIZER_REPO = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_independent_validation(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    max_prefix_tokens: int = 5_000_000,
) -> int:
    """Reject the historical failure where validation was the train prefix."""
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if train.ndim != 2 or validation.ndim != 1:
        raise ValueError(
            f"expected train=(N,L), validation=(T,), got "
            f"{train.shape}, {validation.shape}"
        )
    flat_train = train.reshape(-1)
    checked = min(len(flat_train), len(validation), int(max_prefix_tokens))
    if checked <= 0:
        raise ValueError("cannot validate empty train or validation tensor")
    if np.array_equal(
        np.asarray(flat_train[:checked]), np.asarray(validation[:checked])
    ):
        raise ValueError(
            f"validation overlaps the training prefix for all {checked} "
            "checked tokens"
        )
    return checked


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolved_url(
    endpoint: str,
    *,
    repo: str,
    revision: str,
    relative_path: str,
    repo_type: str,
) -> str:
    root = endpoint.rstrip("/")
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{root}/{prefix}{repo}/resolve/{revision}/{relative_path}"


def download_verified_file(
    *,
    destination: Path,
    url: str,
    expected_sha256: str,
    expected_size: int | None = None,
) -> dict[str, Any]:
    """Resume one public download, then reject anything but the pinned bytes."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_size = destination.stat().st_size
        if expected_size is not None and actual_size != int(expected_size):
            raise ValueError(
                f"existing source has wrong size: {destination}: "
                f"{actual_size} != {expected_size}"
            )
        actual_sha = sha256_file(destination)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"existing source has wrong SHA-256: {destination}: "
                f"{actual_sha} != {expected_sha256}"
            )
        print(f"[download] verified cached file: {destination}", flush=True)
        return {
            "path": str(destination),
            "canonical_url": url,
            "download_transport": "preexisting cache accepted after full hash",
            "size": actual_size,
            "sha256": actual_sha,
        }

    partial = destination.with_name(destination.name + ".partial")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    print("[download] " + " ".join(command), flush=True)
    subprocess.check_call(command)
    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != int(expected_size):
        raise ValueError(
            f"downloaded source has wrong size: {partial}: "
            f"{actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(partial)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"downloaded source has wrong SHA-256: {partial}: "
            f"{actual_sha} != {expected_sha256}"
        )
    partial.replace(destination)
    return {
        "path": str(destination),
        "canonical_url": url,
        "download_transport": "curl",
        "size": actual_size,
        "sha256": actual_sha,
    }


def _parquet_text_batches(
    path: Path, *, batch_rows: int = 128
) -> Iterable[list[str]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=int(batch_rows), columns=["text"]
    ):
        column = batch.column(0)
        texts = []
        for index in range(len(column)):
            value = column[index].as_py()
            if value:
                texts.append(str(value))
        if texts:
            yield texts


def tokenize_batches_to_npy(
    *,
    text_batches: Iterable[list[str]],
    tokenizer: Any,
    output_path: Path,
    token_count: int,
    seq_len: int | None,
    progress_tokens: int = 5_000_000,
) -> dict[str, Any]:
    """Tokenize a deterministic document prefix directly into an int64 NPY."""
    count = int(token_count)
    if count <= 0:
        raise ValueError("token_count must be positive")
    if seq_len is not None and count % int(seq_len):
        raise ValueError("token_count must be divisible by seq_len")
    shape = (count,) if seq_len is None else (count // int(seq_len), int(seq_len))
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.int64, shape=shape
    )
    flat = target.reshape(-1)
    position = 0
    documents = 0
    next_report = int(progress_tokens)
    started = time.time()
    last_document_tokens_used = 0
    for texts in text_batches:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for token_ids in encoded:
            documents += 1
            if not token_ids:
                continue
            take = min(len(token_ids), count - position)
            flat[position : position + take] = np.asarray(
                token_ids[:take], dtype=np.int64
            )
            position += take
            last_document_tokens_used = take
            if position >= next_report:
                target.flush()
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[tokenize] {position:,}/{count:,} tokens "
                    f"({position / elapsed / 1e3:.1f}k tok/s)",
                    flush=True,
                )
                while next_report <= position:
                    next_report += int(progress_tokens)
            if position == count:
                break
        if position == count:
            break
    if position != count:
        del flat, target
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"source produced {position:,} tokens; expected {count:,}"
        )
    target.flush()
    del flat, target
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "tokens": count,
        "shape": list(shape),
        "dtype": "int64",
        "documents_consumed": documents,
        "last_document_tokens_used": last_document_tokens_used,
        "seconds": time.time() - started,
        "sha256": sha256_file(output_path),
    }


def sha256_token_prefix(
    path: str | Path,
    token_count: int,
    *,
    chunk_tokens: int = 1_000_000,
) -> str:
    """Hash a flat int64 prefix without copying the whole tensor."""
    tokens = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if tokens.dtype != np.int64:
        raise ValueError(f"training tensor must be int64, got {tokens.dtype}")
    flat = tokens.reshape(-1)
    count = int(token_count)
    if count <= 0 or len(flat) < count:
        raise ValueError(
            f"training tensor has {len(flat)} tokens, needs at least {count}"
        )
    digest = hashlib.sha256()
    for start in range(0, count, int(chunk_tokens)):
        stop = min(start + int(chunk_tokens), count)
        value = np.ascontiguousarray(flat[start:stop])
        digest.update(value.tobytes())
    return digest.hexdigest()


def token_prefix_bounds(
    path: str | Path,
    token_count: int,
    *,
    chunk_tokens: int = 1_000_000,
) -> tuple[int, int]:
    tokens = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if tokens.dtype != np.int64:
        raise ValueError(f"training tensor must be int64, got {tokens.dtype}")
    flat = tokens.reshape(-1)
    count = int(token_count)
    if count <= 0 or len(flat) < count:
        raise ValueError(
            f"training tensor has {len(flat)} tokens, needs at least {count}"
        )
    minimum: int | None = None
    maximum: int | None = None
    for start in range(0, count, int(chunk_tokens)):
        stop = min(start + int(chunk_tokens), count)
        value = flat[start:stop]
        chunk_min = int(value.min())
        chunk_max = int(value.max())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    assert minimum is not None and maximum is not None
    return minimum, maximum


def choose_eval_anchors(
    validation_tokens: int,
    *,
    count: int = SPEC.eval_anchor_count,
    max_length: int = max(SPEC.eval_lengths),
    seed: int = SPEC.eval_anchor_seed,
) -> np.ndarray:
    """Choose separated shared endpoints for every method and context length."""
    total = int(validation_tokens)
    lower = int(max_length)
    bins = int(count)
    usable = total - lower
    bin_width = usable // bins
    if bin_width <= int(max_length):
        raise ValueError(
            f"validation has {total} tokens; cannot choose {count} separated "
            f"anchors with max_length={max_length}"
        )
    rng = random.Random(int(seed))
    anchors = []
    for index in range(bins):
        bin_start = lower + index * bin_width
        bin_stop = (
            total
            if index == bins - 1
            else lower + (index + 1) * bin_width
        )
        # Leave one longest-window guard before the next stratum. This prevents
        # the reported standard error from silently treating overlapping
        # target windows as independent observations.
        upper = bin_stop - int(max_length)
        anchors.append(rng.randrange(bin_start, upper))
    if any(
        right - left < int(max_length)
        for left, right in zip(anchors, anchors[1:])
    ):
        raise RuntimeError("internal error: evaluation anchors overlap")
    return np.asarray(anchors, dtype=np.int64)


def validate_experiment_manifest(
    manifest: dict[str, Any],
    *,
    check_files: bool,
    check_content_hashes: bool,
) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("experiment manifest schema_version mismatch")
    if manifest.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("experiment manifest protocol fingerprint mismatch")
    if manifest.get("frequency_contract") != frequency_contract():
        raise ValueError("experiment manifest frequency contract mismatch")

    train = manifest.get("train", {})
    validation = manifest.get("validation", {})
    anchors = manifest.get("anchors", {})
    if int(train.get("used_rows", -1)) != SPEC.train_rows:
        raise ValueError("training row count does not match protocol")
    if int(train.get("used_tokens", -1)) != SPEC.train_tokens:
        raise ValueError("training token count does not match protocol")
    if int(train.get("seq_len", -1)) != SPEC.train_length:
        raise ValueError("training sequence length does not match protocol")
    if int(validation.get("tokens", -1)) <= max(SPEC.eval_lengths):
        raise ValueError("validation tensor is too short")
    if int(anchors.get("count", -1)) != SPEC.eval_anchor_count:
        raise ValueError("evaluation anchor count does not match protocol")
    if int(anchors.get("seed", -1)) != SPEC.eval_anchor_seed:
        raise ValueError("evaluation anchor seed does not match protocol")

    if not check_files:
        return
    train_path = Path(str(train.get("path", "")))
    validation_path = Path(str(validation.get("path", "")))
    anchor_path = Path(str(anchors.get("path", "")))
    for label, path in (
        ("training tensor", train_path),
        ("validation tensor", validation_path),
        ("anchor tensor", anchor_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    train_array = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation_array = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    anchor_array = np.load(anchor_path, allow_pickle=False)
    if train_array.dtype != np.int64 or train_array.ndim != 2:
        raise ValueError(
            f"training tensor must be two-dimensional int64, got "
            f"{train_array.dtype} {train_array.shape}"
        )
    if validation_array.dtype != np.int64 or validation_array.ndim != 1:
        raise ValueError(
            f"validation tensor must be one-dimensional int64, got "
            f"{validation_array.dtype} {validation_array.shape}"
        )
    if anchor_array.dtype != np.int64 or anchor_array.shape != (
        SPEC.eval_anchor_count,
    ):
        raise ValueError("anchor tensor shape/dtype mismatch")
    if int(anchor_array.min()) < max(SPEC.eval_lengths):
        raise ValueError("an evaluation anchor cannot supply the longest window")
    if int(anchor_array.max()) >= len(validation_array):
        raise ValueError("an evaluation anchor is outside validation")
    if len(set(anchor_array.tolist())) != len(anchor_array):
        raise ValueError("evaluation anchors contain duplicates")
    if np.any(np.diff(anchor_array) < max(SPEC.eval_lengths)):
        raise ValueError("evaluation anchor windows overlap")
    if anchor_array.tolist() != list(anchors.get("values", [])):
        raise ValueError("evaluation anchor values do not match manifest")

    if not check_content_hashes:
        return
    if sha256_file(train_path) != train.get("sha256"):
        raise ValueError("training tensor SHA-256 mismatch")
    if sha256_file(validation_path) != validation.get("sha256"):
        raise ValueError("validation tensor SHA-256 mismatch")
    if sha256_file(anchor_path) != anchors.get("sha256"):
        raise ValueError("anchor tensor SHA-256 mismatch")
    prefix_sha = sha256_token_prefix(train_path, SPEC.train_tokens)
    if prefix_sha != train.get("used_prefix_sha256"):
        raise ValueError("used training prefix SHA-256 mismatch")
    token_min, token_max = token_prefix_bounds(train_path, SPEC.train_tokens)
    if token_min != int(train.get("used_token_min", -1)):
        raise ValueError("used training prefix minimum token mismatch")
    if token_max != int(train.get("used_token_max", -1)):
        raise ValueError("used training prefix maximum token mismatch")
    if token_min < 0 or token_max >= SPEC.vocab_size:
        raise ValueError("used training token id is outside model vocabulary")
    validation_min = int(validation_array.min())
    validation_max = int(validation_array.max())
    if validation_min != int(validation.get("token_min", -1)):
        raise ValueError("validation minimum token mismatch")
    if validation_max != int(validation.get("token_max", -1)):
        raise ValueError("validation maximum token mismatch")
    if validation_min < 0 or validation_max >= SPEC.vocab_size:
        raise ValueError("validation token id is outside model vocabulary")
    assert_independent_validation(train_path, validation_path)


def _finalize_manifest(
    *,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    data_origin: dict[str, Any],
    validation_source_shard: str,
    validation_source_revision: str,
) -> dict[str, Any]:
    train_path = train_path.resolve()
    validation_path = validation_path.resolve()
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if train.dtype != np.int64 or train.ndim != 2:
        raise ValueError("source training tensor must be two-dimensional int64")
    if validation.dtype != np.int64 or validation.ndim != 1:
        raise ValueError("source validation tensor must be one-dimensional int64")
    if train.size < SPEC.train_tokens:
        raise ValueError(
            f"source has {train.size} tokens, protocol needs {SPEC.train_tokens}"
        )
    prefix_checked = assert_independent_validation(train_path, validation_path)

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    anchor_path = output / "eval_anchors.npy"
    anchors = choose_eval_anchors(len(validation))
    _atomic_save_npy(anchor_path, anchors)
    used_token_min, used_token_max = token_prefix_bounds(
        train_path, SPEC.train_tokens
    )
    validation_token_min = int(validation.min())
    validation_token_max = int(validation.max())
    if used_token_min < 0 or used_token_max >= SPEC.vocab_size:
        raise ValueError("used training token id is outside model vocabulary")
    if validation_token_min < 0 or validation_token_max >= SPEC.vocab_size:
        raise ValueError("validation token id is outside model vocabulary")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": SPEC.fingerprint(),
        "frequency_contract": frequency_contract(),
        "data_origin": data_origin,
        "train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "source_shape": list(train.shape),
            "source_dtype": str(train.dtype),
            "used_rows": SPEC.train_rows,
            "used_tokens": SPEC.train_tokens,
            "prediction_tokens": SPEC.prediction_tokens,
            "seq_len": SPEC.train_length,
            "used_prefix_sha256": sha256_token_prefix(
                train_path, SPEC.train_tokens
            ),
            "used_token_min": used_token_min,
            "used_token_max": used_token_max,
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "tokens": int(len(validation)),
            "source_shard": validation_source_shard,
            "source_revision": validation_source_revision,
            "prefix_tokens_checked": int(prefix_checked),
            "token_min": validation_token_min,
            "token_max": validation_token_max,
        },
        "anchors": {
            "path": str(anchor_path),
            "sha256": sha256_file(anchor_path),
            "count": SPEC.eval_anchor_count,
            "seed": SPEC.eval_anchor_seed,
            "max_length": max(SPEC.eval_lengths),
            "values": anchors.tolist(),
        },
    }
    manifest_path = output / "data_manifest.json"
    _write_json(manifest_path, manifest)
    validate_experiment_manifest(
        manifest, check_files=True, check_content_hashes=True
    )
    return manifest


def prepare_from_source_manifest(
    source_manifest_path: Path, output_dir: Path
) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError(
            "data preparation must run before the paid GPU is enabled"
        )
    source_path = source_manifest_path.resolve()
    source = json.loads(source_path.read_text())
    from experiments.native_rope_evq_150m.prepare_data import (
        validate_data_manifest as validate_source_manifest,
    )

    validate_source_manifest(source, check_files=True)

    train_path = Path(source["train"]["path"]).resolve()
    validation_path = Path(source["validation"]["path"]).resolve()
    return _finalize_manifest(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=output_dir,
        data_origin={
            "mode": "reused_validated_native_150m_manifest",
            "source_manifest": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "role": (
                    "validated 500M FineWeb-Edu train plus disjoint "
                    "held-out shard"
                ),
            },
        },
        validation_source_shard=str(source["validation"]["source_shard"]),
        validation_source_revision=str(source["validation"]["source_revision"]),
    )


def prepare_fresh(
    *,
    output_dir: Path,
    download_dir: Path,
    tokenizer_dir: Path | None,
    endpoint: str,
) -> dict[str, Any]:
    """Rebuild the registered tensors from pinned, disjoint public shards."""
    if torch.cuda.is_available():
        raise RuntimeError(
            "fresh data preparation must run before the paid GPU is enabled"
        )
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloads = download_dir.resolve()
    tokenizer_root = (
        tokenizer_dir.resolve()
        if tokenizer_dir is not None
        else downloads / "tokenizer_gpt_neox_20b"
    )
    train_path = (
        output
        / f"train_fineweb-edu_shard000_{SPEC.train_tokens}_l{SPEC.train_length}.npy"
    )
    validation_path = (
        output
        / f"val_fineweb-edu_shard004_{FRESH_VALIDATION_TOKENS}.npy"
    )
    for path in (train_path, validation_path, output / "data_manifest.json"):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite prepared artifact: {path}"
            )

    source_records: dict[str, dict[str, Any]] = {}
    for role, expected in FRESH_SOURCE_FILES.items():
        relative_path = str(expected["relative_path"])
        destination = downloads / "fineweb_edu" / relative_path
        source_records[role] = {
            **download_verified_file(
                destination=destination,
                url=_resolved_url(
                    endpoint,
                    repo=FINEWEB_REPO,
                    revision=FINEWEB_REVISION,
                    relative_path=relative_path,
                    repo_type="dataset",
                ),
                expected_sha256=str(expected["sha256"]),
                expected_size=int(expected["size"]),
            ),
            "relative_path": relative_path,
        }

    tokenizer_records: dict[str, dict[str, Any]] = {}
    for filename, expected_sha in TOKENIZER_FILES.items():
        tokenizer_records[filename] = download_verified_file(
            destination=tokenizer_root / filename,
            url=_resolved_url(
                endpoint,
                repo=TOKENIZER_REPO,
                revision=TOKENIZER_REVISION,
                relative_path=filename,
                repo_type="model",
            ),
            expected_sha256=expected_sha,
        )

    import pyarrow
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True, use_fast=True
    )
    if len(tokenizer) > SPEC.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {len(tokenizer)} exceeds model vocabulary "
            f"{SPEC.vocab_size}"
        )
    tokenizer.model_max_length = 1 << 60

    train_tokenization = tokenize_batches_to_npy(
        text_batches=_parquet_text_batches(
            Path(source_records["train"]["path"])
        ),
        tokenizer=tokenizer,
        output_path=train_path,
        token_count=SPEC.train_tokens,
        seq_len=SPEC.train_length,
    )
    validation_tokenization = tokenize_batches_to_npy(
        text_batches=_parquet_text_batches(
            Path(source_records["validation"]["path"])
        ),
        tokenizer=tokenizer,
        output_path=validation_path,
        token_count=FRESH_VALIDATION_TOKENS,
        seq_len=None,
        progress_tokens=500_000,
    )

    return _finalize_manifest(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=output,
        data_origin={
            "mode": "fresh_pinned_public_sources",
            "dataset": {
                "repo": FINEWEB_REPO,
                "config": "sample-10BT",
                "revision": FINEWEB_REVISION,
                "field": "text",
                "files": source_records,
            },
            "tokenizer": {
                "repo": TOKENIZER_REPO,
                "revision": TOKENIZER_REVISION,
                "vocab_size": len(tokenizer),
                "files": tokenizer_records,
                "add_special_tokens": False,
            },
            "software": {
                "numpy": np.__version__,
                "pyarrow": pyarrow.__version__,
                "tokenizers": tokenizers.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "selection": {
                "train": (
                    "unshuffled shard-000 document order; concatenate document "
                    "token IDs without added separators; exact registered prefix"
                ),
                "validation": (
                    "unshuffled shard-004 document order; concatenate document "
                    "token IDs without added separators; exact 5M-token prefix"
                ),
                "train_tokenization": train_tokenization,
                "validation_tokenization": validation_tokenization,
                "disjointness": (
                    "train and validation use different pinned source shards"
                ),
            },
        },
        validation_source_shard=str(
            FRESH_SOURCE_FILES["validation"]["relative_path"]
        ),
        validation_source_revision=FINEWEB_REVISION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source_manifest", type=Path)
    source.add_argument("--fresh", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--download_dir", type=Path)
    parser.add_argument("--tokenizer_dir", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--full_hash_check", action="store_true")
    args = parser.parse_args()
    if args.validate is not None:
        manifest = json.loads(args.validate.resolve().read_text())
        validate_experiment_manifest(
            manifest,
            check_files=True,
            check_content_hashes=bool(args.full_hash_check),
        )
        print("FMRoPE L256 data manifest: PASS")
        return
    if args.output_dir is None:
        parser.error("--output_dir is required")
    if args.source_manifest is not None:
        manifest = prepare_from_source_manifest(
            args.source_manifest, args.output_dir
        )
    elif args.fresh:
        if args.download_dir is None:
            parser.error("--download_dir is required with --fresh")
        manifest = prepare_fresh(
            output_dir=args.output_dir,
            download_dir=args.download_dir,
            tokenizer_dir=args.tokenizer_dir,
            endpoint=args.endpoint,
        )
    else:
        parser.error("choose --fresh or --source_manifest")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
