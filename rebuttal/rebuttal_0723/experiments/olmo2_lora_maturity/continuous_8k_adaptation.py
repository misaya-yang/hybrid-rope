"""Hash-checked physical-8K natural-span adaptation contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .prepare_data import sha256_file


LENGTH = 8_192
FORMAT_VERSION = 1
DATA_STATUS = "OLMO2_8K_CONTINUOUS_NATURAL_SPAN_DATA_PREPARED_V1"


class Continuous8KView:
    """Physical 8K answer-plus-EOS view with contiguous position IDs."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("status") != DATA_STATUS
            or int(self.manifest.get("format_version", -1))
            != FORMAT_VERSION
            or int(self.manifest.get("physical_storage_length", -1))
            != LENGTH
            or int(
                self.manifest.get("hard_maximum_training_length", -1)
            )
            != LENGTH
            or self.manifest.get("labels_only_cover_answer_and_final_eos")
            is not True
            or self.manifest.get("position_ids")
            != "contiguous_zero_based"
        ):
            raise RuntimeError("continuous-8K data contract drift")
        for name, entry in self.manifest["files"].items():
            path = self.root / name
            if (
                path.stat().st_size != int(entry["bytes"])
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError(f"continuous-8K file drift: {path}")
        self.input_ids = np.load(
            self.root / "input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            self.root / "labels.npy", mmap_mode="r", allow_pickle=False
        )
        self.query_starts = np.load(
            self.root / "query_starts.npy", mmap_mode="r", allow_pickle=False
        )
        self.active_lengths = np.load(
            self.root / "active_lengths.npy", mmap_mode="r", allow_pickle=False
        )
        self.split = np.load(
            self.root / "split.npy", mmap_mode="r", allow_pickle=False
        )
        expected = tuple(int(value) for value in self.manifest["shape"])
        if (
            tuple(self.input_ids.shape) != expected
            or tuple(self.labels.shape) != expected
            or expected[1] != LENGTH
            or tuple(self.query_starts.shape) != (expected[0],)
            or tuple(self.active_lengths.shape) != (expected[0],)
            or tuple(self.split.shape) != (expected[0],)
            or self.input_ids.dtype != np.uint32
            or self.labels.dtype != np.int32
        ):
            raise RuntimeError("continuous-8K array contract drift")
        if np.any(self.active_lengths != LENGTH):
            raise RuntimeError("continuous-8K rows must be physically full")
        label_mask = self.labels != -100
        first_supervised = np.argmax(label_mask, axis=1)
        if (
            np.any(label_mask.sum(axis=1) < 2)
            or np.any(first_supervised <= self.query_starts)
        ):
            raise RuntimeError("continuous-8K supervision drift")
        self.training_rows = np.flatnonzero(self.split == 0).astype(np.int64)
        self.validation_rows = np.flatnonzero(self.split == 1).astype(np.int64)
        if (
            len(self.training_rows) != int(self.manifest["training_rows"])
            or len(self.validation_rows)
            != int(self.manifest["validation_rows"])
        ):
            raise RuntimeError("continuous-8K split count drift")


def continuous_batch(
    *, view: Continuous8KView, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[dict[str, Any]]]:
    host_ids = np.asarray(view.input_ids[indices], dtype=np.int64)
    host_labels = np.asarray(view.labels[indices], dtype=np.int64)
    contexts = torch.from_numpy(host_ids[:, :-1].copy()).to(
        "cuda", non_blocking=True
    )
    labels = torch.from_numpy(host_labels[:, 1:].copy()).to(
        "cuda", non_blocking=True
    )
    positions = torch.arange(
        LENGTH - 1, device="cuda", dtype=torch.int64
    )[None, :].expand(len(indices), -1)
    supervised = int((labels != -100).sum())
    if supervised <= 0:
        raise RuntimeError("continuous-8K batch has no targets")
    receipts = [
        {
            "row": int(row_index),
            "query_start": int(view.query_starts[int(row_index)]),
            "active_length": LENGTH,
            "answer_prediction_position": int(
                np.argmax(view.labels[int(row_index)] != -100) - 1
            ),
            "maximum_position_id": LENGTH - 2,
        }
        for row_index in indices.tolist()
    ]
    return contexts, labels, positions, supervised, receipts
