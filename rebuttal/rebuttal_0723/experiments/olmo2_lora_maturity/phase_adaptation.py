"""Shared contracts for physical-4K realized-phase task adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .prepare_data import sha256_file


LENGTH = 4_096
MAXIMUM_POSITION_ID = 4 * LENGTH - 1
FORMAT_VERSION = 1
READY_STATUS = "OLMO2_4K_PHASE_ADAPTATION_READY_V1"
DATA_STATUSES = {
    "OLMO2_4K_2WIKI_PHASE_DATA_PREPARED_V1",
    "OLMO2_4K_RULER13_PHASE_DATA_PREPARED_V1",
    "OLMO2_4K_NATURAL_SPAN_PHASE_DATA_PREPARED_V1",
}


class PhaseAdaptationView:
    """Hash-checked answer-plus-EOS view with a semantic query boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("status") not in DATA_STATUSES
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
        ):
            raise RuntimeError("phase-adaptation data contract drift")
        for name, entry in self.manifest["files"].items():
            path = self.root / name
            if (
                path.stat().st_size != int(entry["bytes"])
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError(f"phase-adaptation file drift: {path}")
        self.input_ids = np.load(
            self.root / "input_ids.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.labels = np.load(
            self.root / "labels.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.query_starts = np.load(
            self.root / "query_starts.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.active_lengths = np.load(
            self.root / "active_lengths.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.split = np.load(
            self.root / "split.npy",
            mmap_mode="r",
            allow_pickle=False,
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
            raise RuntimeError("phase-adaptation array contract drift")
        if np.any(self.active_lengths <= 1) or np.any(
            self.active_lengths > LENGTH
        ):
            raise RuntimeError("phase-adaptation active length drift")
        if np.any(self.query_starts <= 0) or np.any(
            self.query_starts >= self.active_lengths - 1
        ):
            raise RuntimeError("phase-adaptation query boundary drift")
        label_mask = self.labels != -100
        if np.any(label_mask.sum(axis=1) < 2):
            raise RuntimeError("answer-plus-EOS supervision is missing")
        first_supervised = np.argmax(label_mask, axis=1)
        if np.any(first_supervised <= self.query_starts):
            raise RuntimeError("answer supervision precedes query boundary")
        self.training_rows = np.flatnonzero(self.split == 0).astype(
            np.int64
        )
        self.validation_rows = np.flatnonzero(self.split == 1).astype(
            np.int64
        )
        if (
            len(self.training_rows)
            != int(self.manifest["training_rows"])
            or len(self.validation_rows)
            != int(self.manifest["validation_rows"])
        ):
            raise RuntimeError("phase-adaptation split count drift")


def phase_batch(
    *,
    view: PhaseAdaptationView,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    host_ids = np.asarray(view.input_ids[indices], dtype=np.int64)
    host_labels = np.asarray(view.labels[indices], dtype=np.int64)
    contexts = torch.from_numpy(host_ids[:, :-1].copy()).to(
        "cuda", non_blocking=True
    )
    labels = torch.from_numpy(host_labels[:, 1:].copy()).to(
        "cuda", non_blocking=True
    )
    supervised = int((labels != -100).sum())
    if supervised <= 0:
        raise RuntimeError("phase-adaptation batch has no targets")
    return contexts, labels, supervised


def position_ids_for_offsets(
    *,
    view: PhaseAdaptationView,
    indices: np.ndarray,
    offsets: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, int]], bytes]:
    """Shift only the semantic query/answer block into a long phase."""

    indices = np.asarray(indices, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if indices.shape != offsets.shape:
        raise RuntimeError("phase offset shape drift")
    positions = np.broadcast_to(
        np.arange(LENGTH - 1, dtype=np.int64),
        (len(indices), LENGTH - 1),
    ).copy()
    receipts: list[dict[str, int]] = []
    for slot, (row_index, offset) in enumerate(
        zip(indices.tolist(), offsets.tolist())
    ):
        query_start = int(view.query_starts[row_index])
        active_length = int(view.active_lengths[row_index])
        if not 0 <= int(offset) <= 3 * LENGTH + 1:
            raise RuntimeError("phase offset exceeds 16K support")
        positions[slot, query_start:] += int(offset)
        if (
            int(positions[slot, -1]) > MAXIMUM_POSITION_ID
            or np.any(np.diff(positions[slot]) <= 0)
            or np.any(np.diff(positions[slot, :query_start]) != 1)
            or np.any(np.diff(positions[slot, query_start:]) != 1)
        ):
            raise RuntimeError("phase position contract drift")
        answer_prediction_index = int(
            np.argmax(view.labels[row_index] != -100) - 1
        )
        if answer_prediction_index < query_start:
            raise RuntimeError("answer prediction precedes query")
        receipts.append(
            {
                "row": int(row_index),
                "query_start": query_start,
                "active_length": active_length,
                "query_offset": int(offset),
                "answer_prediction_index": answer_prediction_index,
                "virtual_answer_prediction_position": int(
                    positions[slot, answer_prediction_index]
                ),
                "maximum_active_position_id": int(
                    positions[slot, min(active_length - 1, LENGTH - 2)]
                ),
            }
        )
    payload = positions.astype("<i8", copy=False).tobytes(order="C")
    return (
        torch.from_numpy(positions).to("cuda", non_blocking=True),
        receipts,
        payload,
    )


def deterministic_offset_batch(
    *,
    seed: int,
    optimizer_step: int,
    accumulation_index: int,
    micro_batch_size: int,
) -> np.ndarray:
    """Return a locked 1:1:2 contiguous/8K/16K phase curriculum."""

    if int(micro_batch_size) != 4:
        raise RuntimeError("locked phase curriculum requires micro-batch 4")
    label = (
        f"evq-phase-adaptation-v1\0{int(seed)}\0"
        f"{int(optimizer_step)}\0{int(accumulation_index)}"
    )
    digest = hashlib.sha256(label.encode("ascii")).digest()

    def choose(low: int, high: int, start: int) -> int:
        width = high - low + 1
        return low + int.from_bytes(
            digest[start : start + 4], "little"
        ) % width

    values = [
        0,
        choose(1, LENGTH, 0),
        choose(LENGTH + 1, 3 * LENGTH + 1, 4),
        choose(LENGTH + 1, 3 * LENGTH + 1, 8),
    ]
    order = sorted(range(4), key=lambda index: (digest[12 + index], index))
    return np.asarray([values[index] for index in order], dtype=np.int64)


def offset_bucket(offset: int) -> str:
    if offset == 0:
        return "contiguous_4k"
    if 1 <= offset <= LENGTH:
        return "phase_to_8k"
    if LENGTH + 1 <= offset <= 3 * LENGTH + 1:
        return "phase_to_16k"
    raise RuntimeError("unregistered phase offset")


def file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
