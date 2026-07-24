#!/usr/bin/env python3
"""CPU-only fresh-holdout preparation for the operator-parity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.protocol import (
    BASE_TRAINING_PROTOCOL_SHA256,
    SPEC,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_prefix(path: Path, token_count: int) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
    if array.dtype != np.int64 or len(array) < int(token_count):
        raise ValueError("training prefix is not a sufficient int64 array")
    digest = hashlib.sha256()
    for start in range(0, int(token_count), 1_000_000):
        stop = min(start + 1_000_000, int(token_count))
        digest.update(
            np.ascontiguousarray(array[start:stop]).tobytes()
        )
    return digest.hexdigest()


def validate_source_manifest(
    source: dict[str, Any],
    *,
    check_hashes: bool,
    check_prefix: bool,
) -> None:
    if source.get("protocol_sha256") != BASE_TRAINING_PROTOCOL_SHA256:
        raise ValueError("source manifest training protocol mismatch")
    train_record = source.get("train", {})
    validation_record = source.get("validation", {})
    train_path = Path(str(train_record.get("path", "")))
    validation_path = Path(str(validation_record.get("path", "")))
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("source train/validation tensor is missing")
    train = np.load(train_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    validation = np.load(
        validation_path, mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("source tensors must contain int64 token ids")
    if (
        len(train) < SPEC.train_tokens
        or int(train_record.get("tokens_used", -1)) != SPEC.train_tokens
    ):
        raise ValueError("source training prefix does not match protocol")
    if check_hashes:
        if sha256_file(train_path) != train_record.get("sha256"):
            raise ValueError("source training tensor hash mismatch")
        if (
            sha256_file(validation_path)
            != validation_record.get("sha256")
        ):
            raise ValueError("source validation tensor hash mismatch")
    if check_prefix and (
        sha256_token_prefix(train_path, SPEC.train_tokens)
        != train_record.get("token_prefix_sha256")
    ):
        raise ValueError("source training prefix hash mismatch")
    anchors: list[np.ndarray] = []
    for split, count in (
        ("selection", SPEC.selection_anchor_count),
        ("test", SPEC.test_anchor_count),
    ):
        record = source.get(f"{split}_anchors", {})
        path = Path(str(record.get("path", "")))
        current = np.load(path, allow_pickle=False)
        if (
            current.dtype != np.int64
            or current.ndim != 1
            or len(current) != count
            or current.tolist() != record.get("values")
        ):
            raise ValueError(f"source {split} anchor identity mismatch")
        if check_hashes and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source {split} anchor hash mismatch")
        anchors.append(current)
    combined = np.sort(np.concatenate(anchors))
    if np.any(np.diff(combined) < max(SPEC.eval_lengths)):
        raise ValueError("source evaluation windows overlap")
    if int(combined.min()) < max(SPEC.eval_lengths):
        raise ValueError("source anchor cannot cover maximum length")
    if int(combined.max()) > len(validation):
        raise ValueError("source anchor exceeds validation tensor")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _materialize(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        if os.path.samefile(source, destination):
            return destination
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(
                f"destination differs from source: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return destination


def _windows(
    endpoints: Iterable[int], max_length: int
) -> list[tuple[int, int]]:
    length = int(max_length)
    return [
        (int(endpoint) - length, int(endpoint))
        for endpoint in endpoints
    ]


def _merge_intervals(
    intervals: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if start < 0 or end <= start:
            raise ValueError(f"invalid forbidden interval: {(start, end)}")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def choose_fresh_disjoint_anchor_endpoints(
    validation_tokens: int,
    *,
    previous_endpoints: Iterable[int],
    count: int,
    max_length: int,
    seed: int,
) -> np.ndarray:
    """Choose fixed-size windows disjoint from all old and new windows."""
    total = int(validation_tokens)
    needed = int(count)
    length = int(max_length)
    if total < length or needed <= 0 or length <= 0:
        raise ValueError("invalid validation/anchor dimensions")
    forbidden = _merge_intervals(
        _windows(previous_endpoints, max_length=length)
    )
    if forbidden and forbidden[-1][1] > total:
        raise ValueError("previous evaluation window exceeds validation data")

    free_segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in forbidden:
        if cursor < start:
            free_segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        free_segments.append((cursor, total))

    slots: list[tuple[int, int]] = []
    for start, end in free_segments:
        current = start
        while current + length <= end:
            slots.append((current, current + length))
            current += length
    if len(slots) < needed:
        raise ValueError(
            f"only {len(slots)} fresh non-overlapping windows available; "
            f"{needed} required"
        )

    rng = random.Random(int(seed))
    chosen = rng.sample(slots, needed)
    endpoints = np.asarray(
        sorted(end for _, end in chosen), dtype=np.int64
    )
    new_windows = _windows(endpoints.tolist(), max_length=length)
    all_windows = sorted([*forbidden, *new_windows])
    if any(
        right[0] < left[1]
        for left, right in zip(all_windows, all_windows[1:])
    ):
        raise RuntimeError("fresh and previous evaluation windows overlap")
    return endpoints


def _anchor_values(manifest: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for split in ("selection", "test"):
        record = manifest.get(f"{split}_anchors", {})
        current = record.get("values")
        if not isinstance(current, list):
            raise ValueError(f"source manifest lacks {split} anchor values")
        values.extend(int(value) for value in current)
    return values


def build_manifest(source_manifest: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    output = output_dir / "data_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(source_manifest.read_text())
    if source.get("protocol_sha256") != BASE_TRAINING_PROTOCOL_SHA256:
        raise ValueError("source manifest training protocol mismatch")
    validate_source_manifest(
        source,
        check_hashes=True,
        check_prefix=True,
    )

    train_record = source.get("train", {})
    validation_record = source.get("validation", {})
    source_train = Path(str(train_record.get("path", ""))).resolve()
    source_validation = Path(
        str(validation_record.get("path", ""))
    ).resolve()
    if not source_train.is_file() or not source_validation.is_file():
        raise FileNotFoundError("source train/validation tensor is missing")
    train_path = _materialize(
        source_train, output_dir / "source_train_tokens.npy"
    )
    validation_path = _materialize(
        source_validation, output_dir / "source_validation_tokens.npy"
    )
    train = np.load(train_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    validation = np.load(
        validation_path, mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("source tensors must contain int64 token ids")
    if len(train) < SPEC.train_tokens:
        raise ValueError("training tensor is shorter than frozen prefix")

    previous = _anchor_values(source)
    total_count = SPEC.selection_anchor_count + SPEC.test_anchor_count
    anchors = choose_fresh_disjoint_anchor_endpoints(
        len(validation),
        previous_endpoints=previous,
        count=total_count,
        max_length=max(SPEC.eval_lengths),
        seed=SPEC.anchor_seed,
    )
    selection = anchors[np.arange(total_count) % 3 == 0]
    test = anchors[np.arange(total_count) % 3 != 0]
    if len(selection) != SPEC.selection_anchor_count:
        raise RuntimeError("selection anchor count mismatch")
    if len(test) != SPEC.test_anchor_count:
        raise RuntimeError("test anchor count mismatch")
    selection_path = output_dir / "selection_anchors.npy"
    test_path = output_dir / "test_anchors.npy"
    _atomic_npy(selection_path, selection)
    _atomic_npy(test_path, test)

    train_copy = dict(train_record)
    train_copy["path"] = str(train_path)
    validation_copy = dict(validation_record)
    validation_copy["path"] = str(validation_path)
    manifest = {
        "schema_version": 1,
        "identity": "mla_yarn_operator_parity_fresh_holdout",
        # The underlying trainer validates this fingerprint.
        "protocol_sha256": BASE_TRAINING_PROTOCOL_SHA256,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "train": train_copy,
        "validation": validation_copy,
        "selection_anchors": {
            "path": str(selection_path),
            "count": int(len(selection)),
            "values": selection.tolist(),
            "sha256": sha256_file(selection_path),
        },
        "test_anchors": {
            "path": str(test_path),
            "count": int(len(test)),
            "values": test.tolist(),
            "sha256": sha256_file(test_path),
        },
        "operator_parity": {
            "protocol_sha256": SPEC.fingerprint(),
            "previous_anchor_count": len(previous),
            "previous_anchor_values": previous,
            "previous_selection_sha256": source[
                "selection_anchors"
            ]["sha256"],
            "previous_test_sha256": source["test_anchors"]["sha256"],
            "maximum_window_length": max(SPEC.eval_lengths),
            "evaluation_batch_sizes": {
                str(length): batch
                for length, batch in SPEC.eval_batch_size_by_length.items()
            },
            "fresh_windows_disjoint_from_previous": True,
            "claim_boundary": (
                "Fresh selection/test windows for a shared-operator "
                "mechanism experiment; no production-scale claim."
            ),
        },
    }
    _atomic_json(output, manifest)
    validate_manifest(
        manifest,
        source_manifest=source_manifest,
        check_tensor_hashes=False,
    )
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    source_manifest: Path,
    check_tensor_hashes: bool,
) -> None:
    source_manifest = source_manifest.resolve()
    source = json.loads(source_manifest.read_text())
    validate_source_manifest(
        source,
        check_hashes=check_tensor_hashes,
        check_prefix=check_tensor_hashes,
    )
    if manifest.get("protocol_sha256") != BASE_TRAINING_PROTOCOL_SHA256:
        raise ValueError("underlying training protocol mismatch")
    parity = manifest.get("operator_parity", {})
    if parity.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("operator-parity protocol mismatch")
    if parity.get("evaluation_batch_sizes") != {
        str(length): batch
        for length, batch in SPEC.eval_batch_size_by_length.items()
    }:
        raise ValueError("operator-parity evaluation batch table mismatch")
    if manifest.get("source_manifest", {}).get(
        "sha256"
    ) != sha256_file(source_manifest):
        raise ValueError("source manifest hash mismatch")

    train_path = Path(str(manifest["train"]["path"]))
    validation_path = Path(str(manifest["validation"]["path"]))
    train = np.load(train_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    validation = np.load(
        validation_path, mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("manifest tensors must contain int64 ids")
    if len(train) < SPEC.train_tokens:
        raise ValueError("training tensor is shorter than frozen prefix")
    if check_tensor_hashes:
        if sha256_file(train_path) != manifest["train"]["sha256"]:
            raise ValueError("training tensor hash mismatch")
        if (
            sha256_file(validation_path)
            != manifest["validation"]["sha256"]
        ):
            raise ValueError("validation tensor hash mismatch")

    fresh: list[int] = []
    for split, count in (
        ("selection", SPEC.selection_anchor_count),
        ("test", SPEC.test_anchor_count),
    ):
        record = manifest[f"{split}_anchors"]
        path = Path(str(record["path"]))
        anchors = np.load(path, allow_pickle=False)
        if (
            anchors.dtype != np.int64
            or anchors.ndim != 1
            or len(anchors) != count
            or anchors.tolist() != record["values"]
        ):
            raise ValueError(f"{split} anchor identity mismatch")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{split} anchor hash mismatch")
        fresh.extend(int(value) for value in anchors)

    previous = _anchor_values(source)
    maximum = max(SPEC.eval_lengths)
    all_windows = sorted(
        [
            *_windows(previous, maximum),
            *_windows(fresh, maximum),
        ]
    )
    if any(
        right[0] < left[1]
        for left, right in zip(all_windows, all_windows[1:])
    ):
        raise ValueError("fresh/previous evaluation windows overlap")
    if min(fresh) < maximum or max(fresh) > len(validation):
        raise ValueError("fresh anchor exceeds validation bounds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.source_manifest, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
