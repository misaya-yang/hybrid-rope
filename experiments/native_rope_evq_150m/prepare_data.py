#!/usr/bin/env python3
"""CPU-only data preparation for the native-RoPE/endpoint-EVQ 150M pair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from experiments.native_rope_evq_150m.protocol import (
    FORBIDDEN_LEAKED_VAL_SHA256,
    SPEC,
    TRAIN_NPY_SHA256,
    legacy_passkey_indices,
)
from scripts.supporting_eval.eval_passkey_scratch import make_passkey_training_sample


FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
HELDOUT_SHARD = "004_00000.parquet"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"tokenizer directory has no files: {root}")
    for file_path in files:
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


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


def compile_passkey_cache(
    *,
    train_path: str | Path,
    tokenizer,
    output_dir: str | Path,
    seq_len: int = SPEC.seq_len,
    ratio: float = SPEC.passkey_mix_ratio,
) -> dict[str, Any]:
    """Precompile exactly the rows selected by the prior ``MixedDataset``."""
    source_path = Path(train_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if train.ndim != 2 or train.shape[1] != int(seq_len):
        raise ValueError(
            f"training tensor must have shape (N, {seq_len}), got {train.shape}"
        )
    if train.dtype != np.int64:
        raise ValueError(f"training tensor must use int64, got {train.dtype}")

    selected = legacy_passkey_indices(int(train.shape[0]), ratio=float(ratio))
    index_path = out_dir / "passkey_indices.npy"
    passkey_path = out_dir / "passkey_train_2pct.npy"
    _atomic_save_npy(index_path, np.asarray(selected, dtype=np.int64))

    temporary = passkey_path.with_name(passkey_path.name + ".incomplete.npy")
    if temporary.exists():
        temporary.unlink()
    if selected:
        target = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.int64,
            shape=(len(selected), int(seq_len)),
        )
        started = time.time()
        for out_index, train_index in enumerate(selected):
            filler = torch.from_numpy(np.array(train[train_index], copy=True))
            sample = make_passkey_training_sample(
                filler,
                tokenizer,
                seq_len=int(seq_len),
                seed=int(train_index),
            )
            target[out_index] = sample.numpy()
            if out_index and out_index % 500 == 0:
                target.flush()
                rate = out_index / max(time.time() - started, 1e-6)
                print(
                    f"[passkey] {out_index}/{len(selected)} rows "
                    f"({rate:.1f} rows/s)",
                    flush=True,
                )
        target.flush()
        del target
        temporary.replace(passkey_path)
    else:
        _atomic_save_npy(
            passkey_path, np.empty((0, int(seq_len)), dtype=np.int64)
        )

    return {
        "selector": "legacy_hash_v1",
        "target_ratio": float(ratio),
        "actual_ratio": len(selected) / max(int(train.shape[0]), 1),
        "rows": len(selected),
        "tokens": len(selected) * int(seq_len),
        "indices_path": str(index_path.resolve()),
        "indices_sha256": sha256_file(index_path),
        "passkey_path": str(passkey_path.resolve()),
        "passkey_sha256": sha256_file(passkey_path),
        "template": "legacy_duplicate_marker_5_dash_digits",
        "filler_source": "same_training_row",
    }


def assert_independent_validation(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    max_prefix_tokens: int = 5_000_000,
) -> int:
    """Reject the observed failure where validation is the train prefix."""
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if train.ndim != 2 or validation.ndim != 1:
        raise ValueError(
            f"expected train=(N,L), validation=(T,), got {train.shape}, {validation.shape}"
        )
    flat_train = train.reshape(-1)
    checked = min(len(flat_train), len(validation), int(max_prefix_tokens))
    if checked <= 0:
        raise ValueError("cannot validate empty train or validation tensor")
    if np.array_equal(np.asarray(flat_train[:checked]), np.asarray(validation[:checked])):
        raise ValueError(
            f"validation overlaps the training prefix for all {checked} checked tokens"
        )
    return checked


def _manifest_train_sha(manifest: dict[str, Any]) -> str:
    train = manifest.get("train")
    if not isinstance(train, dict):
        return ""
    return str(train.get("sha256_npy") or train.get("sha256") or "")


def validate_data_manifest(
    manifest: dict[str, Any], *, check_files: bool = True
) -> None:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("data manifest schema_version must be 1")
    train = manifest.get("train", {})
    validation = manifest.get("validation", {})
    passkey = manifest.get("passkey", {})
    train_sha = str(train.get("sha256", ""))
    if train_sha != TRAIN_NPY_SHA256:
        raise ValueError(f"unexpected training tensor sha256: {train_sha}")
    val_sha = str(validation.get("sha256", ""))
    if val_sha == FORBIDDEN_LEAKED_VAL_SHA256:
        raise ValueError("forbidden leaked validation tensor hash")
    if len(val_sha) != 64:
        raise ValueError("validation sha256 must contain 64 hexadecimal characters")
    source = str(validation.get("source_shard", ""))
    train_sources = {Path(str(x)).name for x in manifest.get("train_source_shards", [])}
    if Path(source).name in train_sources:
        raise ValueError(f"validation shard {source} is also listed as a training source")
    if passkey.get("selector") != "legacy_hash_v1":
        raise ValueError("passkey selector must be legacy_hash_v1")
    if abs(float(passkey.get("ratio", -1.0)) - SPEC.passkey_mix_ratio) > 1e-12:
        raise ValueError("passkey ratio does not match the registered protocol")

    if not check_files:
        return
    for label, record, key in (
        ("train", train, "path"),
        ("validation", validation, "path"),
        ("passkey indices", passkey, "indices_path"),
        ("passkey tensor", passkey, "passkey_path"),
    ):
        path = Path(str(record.get(key, "")))
        if not path.is_file():
            raise ValueError(f"{label} file is missing: {path}")
    if sha256_file(train["path"]) != train_sha:
        raise ValueError("training tensor bytes do not match manifest sha256")
    if sha256_file(validation["path"]) != val_sha:
        raise ValueError("validation tensor bytes do not match manifest sha256")
    if sha256_file(passkey["indices_path"]) != passkey["indices_sha256"]:
        raise ValueError("passkey index bytes do not match manifest sha256")
    if sha256_file(passkey["passkey_path"]) != passkey["passkey_sha256"]:
        raise ValueError("passkey tensor bytes do not match manifest sha256")


def heldout_shard_url(endpoint: str = DEFAULT_ENDPOINT) -> str:
    root = endpoint.rstrip("/")
    return (
        f"{root}/datasets/{FINEWEB_REPO}/resolve/{FINEWEB_REVISION}/"
        f"sample/10BT/{HELDOUT_SHARD}"
    )


def download_heldout_shard(destination: str | Path, endpoint: str) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 1_000_000_000:
        print(f"[download] reuse {output} ({output.stat().st_size / 1e9:.2f} GB)")
        return output
    partial = output.with_name(output.name + ".partial")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(partial),
        heldout_shard_url(endpoint),
    ]
    print("[download] " + " ".join(command), flush=True)
    subprocess.check_call(command)
    partial.replace(output)
    return output


def _parquet_texts(path: Path, batch_rows: int = 64) -> Iterator[str]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
        column = batch.column(0)
        for index in range(len(column)):
            value = column[index].as_py()
            if value:
                yield str(value)
        del batch, column


def tokenize_validation_shard(
    *,
    parquet_path: str | Path,
    tokenizer,
    output_path: str | Path,
    token_count: int = 5_000_000,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete.npy")
    if temporary.exists():
        temporary.unlink()
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.int64, shape=(int(token_count),)
    )
    position = 0
    next_report = 500_000
    started = time.time()
    for text in _parquet_texts(Path(parquet_path)):
        if len(text) > 80_000:
            text = text[:80_000]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        take = min(len(token_ids), int(token_count) - position)
        target[position : position + take] = np.asarray(
            token_ids[:take], dtype=np.int64
        )
        position += take
        if position >= next_report:
            target.flush()
            rate = position / max(time.time() - started, 1e-6)
            print(
                f"[validation] {position}/{token_count} tokens "
                f"({rate / 1e3:.1f}k tok/s)",
                flush=True,
            )
            next_report += 500_000
        if position >= int(token_count):
            break
    if position != int(token_count):
        del target
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"held-out shard produced {position} tokens, expected {token_count}"
        )
    target.flush()
    del target
    temporary.replace(output)
    return {
        "path": str(output.resolve()),
        "tokens": int(token_count),
        "sha256": sha256_file(output),
        "source_shard": Path(parquet_path).name,
        "source_sha256": sha256_file(parquet_path),
        "source_revision": FINEWEB_REVISION,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.train_npy).resolve()
    source_manifest = json.loads(Path(args.train_manifest).read_text())
    if _manifest_train_sha(source_manifest) != TRAIN_NPY_SHA256:
        raise ValueError("source manifest does not identify the registered 500M tensor")
    if sha256_file(train_path) != TRAIN_NPY_SHA256:
        raise ValueError("500M tensor bytes do not match the registered SHA-256")

    from transformers import AutoTokenizer

    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(args.parquet_dir).resolve() / HELDOUT_SHARD
    if not args.skip_download:
        download_heldout_shard(parquet_path, args.endpoint)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"held-out parquet is missing: {parquet_path}")

    validation_path = out_dir / "val_fineweb-edu_shard004_5000000.npy"
    validation = tokenize_validation_shard(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        output_path=validation_path,
        token_count=int(args.val_tokens),
    )
    prefix_checked = assert_independent_validation(train_path, validation_path)
    passkey = compile_passkey_cache(
        train_path=train_path,
        tokenizer=tokenizer,
        output_dir=out_dir,
        seq_len=SPEC.seq_len,
        ratio=SPEC.passkey_mix_ratio,
    )
    passkey["ratio"] = SPEC.passkey_mix_ratio
    train_sources = [Path(str(x)).name for x in source_manifest.get("shards", [])]
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train": {
            "path": str(train_path),
            "sha256": TRAIN_NPY_SHA256,
            "shape": [SPEC.train_rows, SPEC.seq_len],
            "tokens": SPEC.train_tokens,
        },
        "train_source_shards": train_sources,
        "validation": {**validation, "prefix_tokens_checked": prefix_checked},
        "passkey": passkey,
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256_directory(tokenizer_path),
            "vocab_size": len(tokenizer),
        },
    }
    validate_data_manifest(manifest, check_files=True)
    _write_json(out_dir / "data_manifest.json", manifest)
    return manifest


def self_test() -> None:
    class TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [1_000 + (ord(ch) % 200) for ch in str(text)]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train = np.arange(20 * 64, dtype=np.int64).reshape(20, 64)
        np.save(root / "train.npy", train)
        result = compile_passkey_cache(
            train_path=root / "train.npy",
            tokenizer=TinyTokenizer(),
            output_dir=root / "out",
            seq_len=64,
            ratio=0.25,
        )
        if result["rows"] != len(legacy_passkey_indices(20, ratio=0.25)):
            raise RuntimeError("passkey self-test row count mismatch")
    print("prepare_data self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npy", type=Path)
    parser.add_argument("--train_manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--parquet_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--val_tokens", type=int, default=5_000_000)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        "train_npy",
        "train_manifest",
        "tokenizer",
        "parquet_dir",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    manifest = prepare(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    gc.collect()


if __name__ == "__main__":
    main()
