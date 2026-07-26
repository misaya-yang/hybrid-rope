"""Serialization helpers for source-causal probe sets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    FormalProbeSet,
)

from .prepare_data import atomic_json, sha256_file


ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "sourced": np.dtype(np.int64),
    "deleted": np.dtype(np.int64),
    "swapped": np.dtype(np.int64),
    "gold": np.dtype(np.int64),
    "alternate": np.dtype(np.int64),
    "source_fraction": np.dtype(np.float64),
    "distractor_count": np.dtype(np.int64),
    "document_rows": np.dtype(np.int64),
    "crop_offsets": np.dtype(np.int64),
}

LIST_FIELDS = (
    "keys",
    "value_words",
    "document_sources",
    "template_ids",
)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def save_probe_set(
    output_dir: Path,
    data: FormalProbeSet,
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(getattr(data, name), dtype=dtype)
        for name, dtype in ARRAY_DTYPES.items()
    }
    count = len(data.gold)
    if any(value.shape[0] != count for value in arrays.values()):
        raise RuntimeError("probe array row-count drift")
    lists = {
        name: list(getattr(data, name))
        for name in LIST_FIELDS
    }
    if any(len(value) != count for value in lists.values()):
        raise RuntimeError("probe metadata row-count drift")

    files: dict[str, dict[str, Any]] = {}
    for name, value in arrays.items():
        path = output_dir / f"{name}.npy"
        _atomic_numpy(path, value)
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metadata_path = output_dir / "rows.json"
    atomic_json(metadata_path, lists)
    files[metadata_path.name] = {
        "bytes": metadata_path.stat().st_size,
        "sha256": sha256_file(metadata_path),
    }
    manifest = {
        "format_version": 1,
        "status": "OLMO2_SOURCE_CAUSAL_SET_PREPARED",
        "count": count,
        "sequence_length": int(data.sourced.shape[1] + 1),
        "dataset_sha256": data.digest(),
        "receipt": dict(receipt),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        },
        "files": files,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def load_probe_set(
    input_dir: Path,
    *,
    mmap_mode: str | None = "r",
) -> FormalProbeSet:
    manifest = json.loads(
        (input_dir / "manifest.json").read_text(encoding="utf-8")
    )
    arrays = {
        name: np.load(
            input_dir / f"{name}.npy",
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        for name in ARRAY_DTYPES
    }
    metadata = json.loads(
        (input_dir / "rows.json").read_text(encoding="utf-8")
    )
    data = FormalProbeSet(
        sourced=arrays["sourced"],
        deleted=arrays["deleted"],
        swapped=arrays["swapped"],
        gold=arrays["gold"],
        alternate=arrays["alternate"],
        source_fraction=arrays["source_fraction"],
        distractor_count=arrays["distractor_count"],
        keys=list(metadata["keys"]),
        value_words=list(metadata["value_words"]),
        document_sources=list(metadata["document_sources"]),
        document_rows=arrays["document_rows"],
        crop_offsets=arrays["crop_offsets"],
        template_ids=list(metadata["template_ids"]),
    )
    if len(data.gold) != int(manifest["count"]):
        raise RuntimeError("probe manifest count drift")
    return data


def verify_probe_set(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, file_receipt in manifest["files"].items():
        path = input_dir / name
        if path.stat().st_size != int(file_receipt["bytes"]):
            raise RuntimeError(f"probe file size mismatch: {path}")
        if sha256_file(path) != file_receipt["sha256"]:
            raise RuntimeError(f"probe file hash mismatch: {path}")
    data = load_probe_set(input_dir)
    count = int(manifest["count"])
    expected_context = int(manifest["sequence_length"]) - 1
    for name in ("sourced", "deleted", "swapped"):
        value = getattr(data, name)
        if value.shape != (count, expected_context):
            raise RuntimeError(f"probe shape drift: {name}")
        if value.dtype != ARRAY_DTYPES[name]:
            raise RuntimeError(f"probe dtype drift: {name}")
    if data.digest() != manifest["dataset_sha256"]:
        raise RuntimeError("probe dataset digest mismatch after serialization")
    if len(set(data.keys)) != count:
        raise RuntimeError("probe keys are not unique")
    return {
        "status": "OLMO2_SOURCE_CAUSAL_SET_VERIFIED",
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "count": count,
        "sequence_length": int(manifest["sequence_length"]),
    }
