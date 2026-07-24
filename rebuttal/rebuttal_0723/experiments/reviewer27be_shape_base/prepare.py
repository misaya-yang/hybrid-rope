#!/usr/bin/env python3
"""CPU-only preparation of a shared-data manifest for Reviewer 27bE runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.protocol import (
    SPECS,
    frequency_contract,
)


SCHEMA_VERSION = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_prefix(path: str | Path, token_count: int) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    flat = array.reshape(-1)
    if array.dtype != np.int64 or len(flat) < int(token_count):
        raise ValueError("training tensor is not a sufficient int64 token array")
    digest = hashlib.sha256()
    for start in range(0, int(token_count), 1_000_000):
        stop = min(start + 1_000_000, int(token_count))
        digest.update(np.ascontiguousarray(flat[start:stop]).tobytes())
    return digest.hexdigest()


def _atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _materialize_source(source: Path, destination: Path) -> Path:
    """Keep an experiment-owned hardlink (or copy across filesystems)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"refusing to replace different file: {destination}")
        return destination
    temporary = destination.with_name(destination.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return destination


def choose_disjoint_anchors(
    validation_tokens: int,
    *,
    count: int,
    max_length: int,
    seed: int,
) -> np.ndarray:
    """Choose non-overlapping endpoints from deterministic equal-width strata."""
    total = int(validation_tokens)
    width = (total - max_length) // int(count)
    if width <= max_length:
        raise ValueError("validation tensor is too short for disjoint anchors")
    rng = random.Random(int(seed))
    values = []
    for index in range(int(count)):
        lower = max_length + index * width
        upper = (
            total - max_length
            if index == count - 1
            else max_length + (index + 1) * width - max_length
        )
        if upper <= lower:
            raise RuntimeError("invalid anchor stratum")
        values.append(rng.randrange(lower, upper))
    anchors = np.asarray(sorted(values), dtype=np.int64)
    if np.any(np.diff(anchors) < max_length):
        raise RuntimeError("generated anchor windows overlap")
    return anchors


def _source_paths(source: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    train = source.get("train", {})
    validation = source.get("validation", {})
    train_path = Path(str(train.get("path", ""))).resolve()
    validation_path = Path(str(validation.get("path", ""))).resolve()
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("source manifest train/validation files are missing")
    return train_path, validation_path, {
        "path": str(Path(str(source.get("_manifest_path", ""))).resolve()),
        "sha256": source.get("_manifest_sha256"),
        "schema_version": source.get("schema_version"),
    }


def build_manifest(source_manifest: Path, output_dir: Path) -> dict[str, Any]:
    source_path = source_manifest.resolve()
    source = json.loads(source_path.read_text())
    source["_manifest_path"] = str(source_path)
    source["_manifest_sha256"] = sha256_file(source_path)
    train_path, validation_path, source_receipt = _source_paths(source)
    output = output_dir.resolve()
    train_path = _materialize_source(
        train_path, output / "source_train_tokens.npy"
    )
    validation_path = _materialize_source(
        validation_path, output / "source_validation_tokens.npy"
    )
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("source tensors must use int64 token ids")
    train_flat = train.reshape(-1)
    validation_flat = validation.reshape(-1)
    max_train_tokens = max(spec.train_tokens for spec in SPECS.values())
    if len(train_flat) < max_train_tokens:
        raise ValueError(
            f"source has {len(train_flat):,} train tokens; "
            f"{max_train_tokens:,} required"
        )
    max_eval_length = max(max(spec.eval_lengths) for spec in SPECS.values())
    total_anchors = max(
        spec.selection_anchor_count + spec.test_anchor_count
        for spec in SPECS.values()
    )
    anchors = choose_disjoint_anchors(
        len(validation_flat),
        count=total_anchors,
        max_length=max_eval_length,
        seed=SPECS["heldout_b1m_d128"].anchor_seed,
    )
    # Interleave the split across the shard instead of assigning an early
    # prefix to selection and a late suffix to test.
    selection_mask = np.arange(len(anchors)) % 3 == 0
    selection = anchors[selection_mask]
    test = anchors[~selection_mask]
    if len(selection) != SPECS["heldout_b1m_d128"].selection_anchor_count:
        raise RuntimeError("internal selection anchor count mismatch")
    if len(test) != SPECS["heldout_b1m_d128"].test_anchor_count:
        raise RuntimeError("internal test anchor count mismatch")
    selection_path = output / "selection_anchors.npy"
    test_path = output / "test_anchors.npy"
    _atomic_save(selection_path, selection)
    _atomic_save(test_path, test)

    suites: dict[str, Any] = {}
    for name, spec in SPECS.items():
        suites[name] = {
            "protocol_sha256": spec.fingerprint(),
            "frequency_contract": frequency_contract(name, spec=spec),
            "train_tokens": spec.train_tokens,
            "train_prefix_sha256": sha256_token_prefix(
                train_path, spec.train_tokens
            ),
            "train_rows": spec.train_rows,
            "train_length": spec.train_length,
            "max_eval_length": max(spec.eval_lengths),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity": "reviewer27be_shared_fineweb_fresh_holdout",
        "source_manifest": source_receipt,
        "train": {
            "path": str(train_path),
            "tokens_available": len(train_flat),
            "sha256": sha256_file(train_path),
            "token_min": int(train_flat[:max_train_tokens].min()),
            "token_max": int(train_flat[:max_train_tokens].max()),
        },
        "validation": {
            "path": str(validation_path),
            "tokens": len(validation_flat),
            "sha256": sha256_file(validation_path),
            "token_min": int(validation_flat.min()),
            "token_max": int(validation_flat.max()),
        },
        "selection_anchors": {
            "path": str(selection_path),
            "count": len(selection),
            "values": selection.tolist(),
            "sha256": sha256_file(selection_path),
        },
        "test_anchors": {
            "path": str(test_path),
            "count": len(test),
            "values": test.tolist(),
            "sha256": sha256_file(test_path),
        },
        "suites": suites,
        "claim_boundary": (
            "Fresh FineWeb-Edu train shard and disjoint held-out shard. "
            "This is a new matched ablation and must not be numerically merged "
            "with historical Primary-II rows."
        ),
    }
    path = output / "data_manifest.json"
    _atomic_json(path, manifest)
    validate_manifest(manifest, check_hashes=True)
    return manifest


def validate_manifest(manifest: dict[str, Any], *, check_hashes: bool) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("manifest schema mismatch")
    train = manifest.get("train", {})
    validation = manifest.get("validation", {})
    train_path = Path(str(train.get("path", "")))
    validation_path = Path(str(validation.get("path", "")))
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("manifest token tensors are missing")
    train_array = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_array = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if train_array.dtype != np.int64 or val_array.dtype != np.int64:
        raise ValueError("manifest token tensors must be int64")
    for key in ("selection_anchors", "test_anchors"):
        record = manifest.get(key, {})
        path = Path(str(record.get("path", "")))
        anchors = np.load(path, allow_pickle=False)
        if anchors.dtype != np.int64 or anchors.ndim != 1:
            raise ValueError(f"{key} must be a one-dimensional int64 array")
        if anchors.tolist() != record.get("values"):
            raise ValueError(f"{key} values differ from manifest")
        if int(anchors.max()) >= val_array.size:
            raise ValueError(f"{key} is outside validation tensor")
        if check_hashes and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{key} SHA-256 mismatch")
    for name, spec in SPECS.items():
        record = manifest.get("suites", {}).get(name, {})
        if record.get("protocol_sha256") != spec.fingerprint():
            raise ValueError(f"{name} protocol fingerprint mismatch")
        if record.get("frequency_contract") != frequency_contract(
            name, spec=spec
        ):
            raise ValueError(f"{name} frequency contract mismatch")
        if int(record.get("train_tokens", -1)) != spec.train_tokens:
            raise ValueError(f"{name} token budget mismatch")
        if check_hashes:
            actual = sha256_token_prefix(train_path, spec.train_tokens)
            if actual != record.get("train_prefix_sha256"):
                raise ValueError(f"{name} train prefix SHA-256 mismatch")
    if check_hashes:
        if sha256_file(train_path) != train.get("sha256"):
            raise ValueError("training tensor SHA-256 mismatch")
        if sha256_file(validation_path) != validation.get("sha256"):
            raise ValueError("validation tensor SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--full_hash_check", action="store_true")
    args = parser.parse_args()
    if args.validate:
        manifest = json.loads(args.validate.resolve().read_text())
        validate_manifest(manifest, check_hashes=bool(args.full_hash_check))
        print("Reviewer 27bE data manifest: PASS")
        return
    if not args.source_manifest or not args.output_dir:
        parser.error("--source_manifest and --output_dir are required")
    manifest = build_manifest(args.source_manifest, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
