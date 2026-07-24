#!/usr/bin/env python3
"""CPU-only data preparation for the MLA scarcity experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.mla_scarcity_5090.protocol import SPEC
from rebuttal.rebuttal_0723.reviewer27be_shape_base.prepare import (
    sha256_file,
    sha256_token_prefix,
)


SCHEMA_VERSION = 1


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
        if sha256_file(destination) != sha256_file(source):
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


def choose_disjoint_anchor_endpoints(
    validation_tokens: int,
    *,
    count: int,
    max_length: int,
    seed: int,
) -> np.ndarray:
    total = int(validation_tokens)
    required = int(count) * int(max_length)
    if total < required + int(max_length):
        raise ValueError(
            f"validation tensor has {total:,} tokens; "
            f"at least {required + max_length:,} required"
        )
    width = (total - max_length) // int(count)
    if width <= max_length:
        raise ValueError("validation tensor is too short for disjoint anchors")
    rng = random.Random(int(seed))
    endpoints = []
    for index in range(int(count)):
        lower = max_length + index * width
        upper = (
            total
            if index == count - 1
            else max_length + (index + 1) * width
        )
        upper -= max_length
        if upper <= lower:
            raise RuntimeError("invalid anchor stratum")
        endpoints.append(rng.randrange(lower, upper))
    result = np.asarray(sorted(endpoints), dtype=np.int64)
    if np.any(np.diff(result) < max_length):
        raise RuntimeError("generated evaluation windows overlap")
    return result


def _source_paths(source_manifest: Path) -> tuple[Path, Path]:
    payload = json.loads(source_manifest.resolve().read_text())
    train = Path(str(payload.get("train", {}).get("path", ""))).resolve()
    validation = Path(
        str(payload.get("validation", {}).get("path", ""))
    ).resolve()
    if not train.is_file() or not validation.is_file():
        raise FileNotFoundError(
            "source manifest train/validation tensor is missing"
        )
    return train, validation


def build_manifest(source_manifest: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    source_train, source_validation = _source_paths(source_manifest)
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
        raise ValueError(
            f"training tensor has {len(train):,} tokens; "
            f"{SPEC.train_tokens:,} required"
        )
    total_anchors = SPEC.selection_anchor_count + SPEC.test_anchor_count
    anchors = choose_disjoint_anchor_endpoints(
        len(validation),
        count=total_anchors,
        max_length=max(SPEC.eval_lengths),
        seed=SPEC.anchor_seed,
    )
    selection_mask = np.arange(total_anchors) % 3 == 0
    selection = anchors[selection_mask]
    test = anchors[~selection_mask]
    if len(selection) != SPEC.selection_anchor_count:
        raise RuntimeError("selection-anchor count mismatch")
    if len(test) != SPEC.test_anchor_count:
        raise RuntimeError("test-anchor count mismatch")
    selection_path = output_dir / "selection_anchors.npy"
    test_path = output_dir / "test_anchors.npy"
    _atomic_npy(selection_path, selection)
    _atomic_npy(test_path, test)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity": "mla_scarcity_fineweb_fixed_prefix_and_holdout",
        "protocol_sha256": SPEC.fingerprint(),
        "source_manifest": {
            "sha256": sha256_file(source_manifest),
            "schema_version": json.loads(source_manifest.read_text()).get(
                "schema_version"
            ),
        },
        "train": {
            "path": str(train_path),
            "tokens_available": int(len(train)),
            "tokens_used": SPEC.train_tokens,
            "rows_used": SPEC.train_rows,
            "sha256": sha256_file(train_path),
            "token_prefix_sha256": sha256_token_prefix(
                train_path, SPEC.train_tokens
            ),
            "token_min": int(train[: SPEC.train_tokens].min()),
            "token_max": int(train[: SPEC.train_tokens].max()),
        },
        "validation": {
            "path": str(validation_path),
            "tokens": int(len(validation)),
            "sha256": sha256_file(validation_path),
            "token_min": int(validation.min()),
            "token_max": int(validation.max()),
        },
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
        "claim_boundary": (
            "Fresh matched 50M-class MLA mechanism experiment. Seed-42 gate "
            "uses only selection anchors; confirmatory claims use test anchors."
        ),
    }
    _atomic_json(manifest_path, manifest)
    validate_manifest(manifest, check_hashes=True, check_prefix=True)
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    check_hashes: bool,
    check_prefix: bool,
) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("manifest schema mismatch")
    if manifest.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("manifest protocol fingerprint mismatch")
    train_record = manifest.get("train", {})
    val_record = manifest.get("validation", {})
    train_path = Path(str(train_record.get("path", "")))
    val_path = Path(str(val_record.get("path", "")))
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("manifest tensors are missing")
    train = np.load(train_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    validation = np.load(
        val_path, mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("manifest tensors must contain int64 token ids")
    if len(train) < SPEC.train_tokens:
        raise ValueError("training tensor is shorter than frozen prefix")
    if int(train_record.get("tokens_used", -1)) != SPEC.train_tokens:
        raise ValueError("registered training-token count mismatch")
    if check_hashes:
        if sha256_file(train_path) != train_record.get("sha256"):
            raise ValueError("training tensor hash mismatch")
        if sha256_file(val_path) != val_record.get("sha256"):
            raise ValueError("validation tensor hash mismatch")
    if check_prefix and (
        sha256_token_prefix(train_path, SPEC.train_tokens)
        != train_record.get("token_prefix_sha256")
    ):
        raise ValueError("training prefix hash mismatch")
    all_anchors: list[np.ndarray] = []
    for split, count in (
        ("selection", SPEC.selection_anchor_count),
        ("test", SPEC.test_anchor_count),
    ):
        record = manifest.get(f"{split}_anchors", {})
        path = Path(str(record.get("path", "")))
        anchors = np.load(path, allow_pickle=False)
        if (
            anchors.dtype != np.int64
            or anchors.ndim != 1
            or len(anchors) != count
        ):
            raise ValueError(f"{split} anchors have an invalid shape")
        if anchors.tolist() != record.get("values"):
            raise ValueError(f"{split} anchor values differ from manifest")
        if int(anchors.min()) < max(SPEC.eval_lengths):
            raise ValueError(f"{split} anchor cannot cover maximum length")
        if int(anchors.max()) > len(validation):
            raise ValueError(f"{split} anchor exceeds validation tensor")
        if check_hashes and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{split} anchor hash mismatch")
        all_anchors.append(anchors)
    combined = np.sort(np.concatenate(all_anchors))
    if np.any(np.diff(combined) < max(SPEC.eval_lengths)):
        raise ValueError("selection/test evaluation windows overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.source_manifest, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
