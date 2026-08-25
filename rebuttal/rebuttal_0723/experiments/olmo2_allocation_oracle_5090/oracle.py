"""Shared contracts for the fixed-support allocation oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


METHOD_ID = "olmo2_fixed_support_allocation_oracle_v1"
NATIVE_LENGTH = 4_096
PAIR_COUNT = 64
PHASE_MAX_MULTIPLIER = 16
FAMILY_PATTERN = ("phase", "phase", "natural")


def protocol(*, steps: int, seed: int, smoke: bool = False) -> dict[str, Any]:
    return {
        "method": METHOD_ID,
        "steps": 1 if smoke else int(steps),
        "seed": int(seed),
        "physical_length": NATIVE_LENGTH,
        "phase_maximum_position": PHASE_MAX_MULTIPLIER * NATIVE_LENGTH - 1,
        "phase_risk_prior": "one contiguous plus three deterministic log-shell offsets",
        "family_pattern": list(FAMILY_PATTERN),
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "qk_lora": {"rank": 64, "alpha": 128.0, "dropout": 0.0},
        "learning_rates": {"qk_lora": 5e-5, "allocation": 1e-3},
        "warmup_steps": 20,
        "allocation": "positive normalized gaps; exact Native endpoints; shared across layers and heads",
        "allocation_gap_logit_bound": 2.0,
        "attention_scaling": 1.0,
        "target_length_used_by_construction": False,
    }


def deterministic_offsets(*, seed: int, step: int, accumulation: int) -> np.ndarray:
    """One Native exposure and one sample from each wider log shell."""

    label = f"{METHOD_ID}\0{int(seed)}\0{int(step)}\0{int(accumulation)}"
    digest = hashlib.sha256(label.encode("ascii")).digest()

    def choose(low: int, high: int, start: int) -> int:
        return low + int.from_bytes(digest[start : start + 8], "little") % (high - low + 1)

    values = np.asarray(
        [
            0,
            choose(1, NATIVE_LENGTH, 0),
            choose(NATIVE_LENGTH + 1, 3 * NATIVE_LENGTH, 8),
            choose(3 * NATIVE_LENGTH + 1, 15 * NATIVE_LENGTH, 16),
        ],
        dtype=np.int64,
    )
    order = np.argsort(np.frombuffer(digest[24:28], dtype=np.uint8), kind="stable")
    return values[order]


def position_ids_for_offsets(
    *,
    query_starts: np.ndarray,
    offsets: np.ndarray,
    sequence_length: int = NATIVE_LENGTH - 1,
) -> np.ndarray:
    """Shift only each row's query/answer suffix; token count is unchanged."""

    starts = np.asarray(query_starts, dtype=np.int64)
    shifts = np.asarray(offsets, dtype=np.int64)
    if starts.ndim != 1 or shifts.shape != starts.shape:
        raise ValueError("query starts and offsets must be aligned vectors")
    if np.any(starts <= 0) or np.any(starts >= int(sequence_length)):
        raise ValueError("query start is outside the physical sequence")
    if np.any(shifts < 0) or np.any(shifts > 15 * NATIVE_LENGTH):
        raise ValueError("phase offset is outside the registered risk support")
    positions = np.broadcast_to(
        np.arange(int(sequence_length), dtype=np.int64),
        (len(starts), int(sequence_length)),
    ).copy()
    for row, (start, shift) in enumerate(zip(starts.tolist(), shifts.tolist())):
        positions[row, int(start) :] += int(shift)
    if np.any(np.diff(positions, axis=1) <= 0):
        raise RuntimeError("phase position ids must stay strictly increasing")
    if int(positions.max(initial=0)) >= PHASE_MAX_MULTIPLIER * NATIVE_LENGTH:
        raise RuntimeError("phase position id exceeds the registered maximum")
    return positions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().numpy().astype("<f4", copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    repo = root.parents[3]
    paths = {
        "oracle": root / "oracle.py",
        "trainer": root / "train.py",
        "preflight": root / "preflight.py",
        "fixed_support_z": repo / "scripts/lib/rope/fixed_support_z.py",
        "phase_data": root.parent / "olmo2_lora_maturity/phase_adaptation.py",
        "lora": root.parent / "olmo2_lora_conversion.py",
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


__all__ = (
    "FAMILY_PATTERN",
    "METHOD_ID",
    "NATIVE_LENGTH",
    "PAIR_COUNT",
    "PHASE_MAX_MULTIPLIER",
    "code_hashes",
    "deterministic_offsets",
    "position_ids_for_offsets",
    "protocol",
    "sha256_file",
    "tensor_sha256",
)

