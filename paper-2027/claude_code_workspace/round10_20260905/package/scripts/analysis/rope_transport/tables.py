"""Frequency-table sourcing with explicit identity.

Frozen tables are loaded from an existing target manifest and never rebuilt.
Derived operators (official YaRN / position interpolation) are computed from a
loaded frozen table by a named, pinned rule and are labelled as derived. Every
table carries the float32 SHA-256 that the training and evaluation code uses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class Table:
    name: str
    origin: str
    inv_freq: np.ndarray
    sha256: str
    meta: Dict[str, Any]

    def as_record(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "sha256": self.sha256,
            "pairs": int(self.inv_freq.size),
            "min_inv_freq": float(self.inv_freq.min()),
            "max_inv_freq": float(self.inv_freq.max()),
            "meta": self.meta,
        }


def float32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()
    ).hexdigest()


def _check(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name}: table too short")
    if not np.isfinite(arr).all() or not (arr > 0.0).all():
        raise ValueError(f"{name}: inverse frequencies must be finite and positive")
    if not np.all(arr[:-1] > arr[1:]):
        raise ValueError(f"{name}: inverse frequencies must be strictly decreasing")
    return arr


def load_manifest_tables(path: str | Path) -> List[Table]:
    """Load ``native`` plus every candidate from a frozen target manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {manifest_path}")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    out: List[Table] = []

    def add(name: str, record: Any) -> None:
        if not isinstance(record, dict) or "inv_freq" not in record:
            return
        arr = _check(np.asarray(record["inv_freq"], dtype="<f4"), name)
        out.append(
            Table(
                name=name,
                origin="frozen_manifest",
                inv_freq=arr,
                sha256=float32_sha256(arr),
                meta={
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                    "manifest_key": name,
                },
            )
        )

    add("native", payload.get("native"))
    candidates = payload.get("candidates")
    if isinstance(candidates, dict):
        for key in sorted(candidates):
            add(key, candidates[key])
    if not out:
        raise ValueError(f"manifest contains no frequency tables: {manifest_path}")
    return out


def position_interpolation(base_table: Table, scale: float) -> Table:
    """Uniform position interpolation: every frequency divided by ``scale``."""
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    arr = base_table.inv_freq / float(scale)
    return Table(
        name=f"pi_s{scale:g}",
        origin="derived",
        inv_freq=arr,
        sha256=float32_sha256(arr),
        meta={
            "rule": "position_interpolation",
            "scale": float(scale),
            "source_table": base_table.name,
            "source_sha256": base_table.sha256,
        },
    )


def official_yarn(
    base_table: Table,
    *,
    scale: float,
    head_dim: int,
    rope_base: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Table:
    """Official YaRN ramp (jquesnelle/yarn@995db5b equations) on a loaded table.

    The ramp is expressed in the virtual channel coordinate
    ``j_v = -d * log(w) / (2 * log(b))`` so it applies unchanged to non-native
    grids. ``mscale`` is an amplitude factor on cos/sin and is reported, never
    folded into the phase, so it does not enter the frequency table.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    omega = base_table.inv_freq
    pairs = omega.size

    def correction_dim(rotations: float) -> float:
        return (
            head_dim
            * math.log(original_max_position_embeddings / (rotations * 2.0 * math.pi))
        ) / (2.0 * math.log(rope_base))

    low = max(int(math.floor(correction_dim(beta_fast))), 0)
    high = min(int(math.ceil(correction_dim(beta_slow))), pairs - 1)
    if high == low:
        high = low + 1

    virtual = -float(head_dim) * np.log(omega) / (2.0 * math.log(float(rope_base)))
    ramp = np.clip((virtual - low) / float(high - low), 0.0, 1.0)
    extrapolation_mask = 1.0 - ramp
    arr = (omega / float(scale)) * (1.0 - extrapolation_mask) + omega * extrapolation_mask
    mscale = 0.1 * math.log(float(scale)) + 1.0
    return Table(
        name=f"yarn_s{scale:g}",
        origin="derived",
        inv_freq=_check(arr, f"yarn_s{scale:g}"),
        sha256=float32_sha256(arr),
        meta={
            "rule": "official_yarn_equations",
            "pinned_source": "jquesnelle/yarn@995db5b",
            "scale": float(scale),
            "beta_fast": float(beta_fast),
            "beta_slow": float(beta_slow),
            "low": int(low),
            "high": int(high),
            "mscale_amplitude": float(mscale),
            "head_dim": int(head_dim),
            "rope_base": float(rope_base),
            "original_max_position_embeddings": int(original_max_position_embeddings),
            "source_table": base_table.name,
            "source_sha256": base_table.sha256,
            "untouched_fast_pairs": int((extrapolation_mask >= 1.0).sum()),
            "fully_interpolated_pairs": int((extrapolation_mask <= 0.0).sum()),
        },
    )


def budgeted_transport(
    base_table: Table,
    uniqueness: np.ndarray,
    *,
    scale: float,
    exponent: float = 1.0,
    name: str | None = None,
) -> Table:
    """Displacement budget inversely proportional to measured in-window uniqueness.

    Each pair is interpolated toward ``omega / scale`` by an amount
    ``(1 - u_k) ** exponent``, where ``u_k`` is the normalised conditional
    uniqueness of that pair on the training window. Pairs the model can actually
    resolve in-window stay put; pairs that are redundant on that window absorb
    the full range extension. YaRN's fixed wavelength ramp is the crude binary
    special case of this rule.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    u = np.asarray(uniqueness, dtype=np.float64).reshape(-1)
    if u.shape != base_table.inv_freq.shape:
        raise ValueError("uniqueness must match the table length")
    span = float(u.max()) - float(u.min())
    norm = np.zeros_like(u) if span <= 0.0 else (u - float(u.min())) / span
    move = (1.0 - norm) ** float(exponent)
    omega = base_table.inv_freq
    arr = omega * (1.0 - move) + (omega / float(scale)) * move
    label = name or f"budgeted_s{scale:g}_p{exponent:g}"
    return Table(
        name=label,
        origin="derived",
        inv_freq=_check(arr, label),
        sha256=float32_sha256(arr),
        meta={
            "rule": "uniqueness_budgeted_transport",
            "scale": float(scale),
            "exponent": float(exponent),
            "source_table": base_table.name,
            "source_sha256": base_table.sha256,
            "mean_move": float(move.mean()),
            "max_move": float(move.max()),
            "min_move": float(move.min()),
        },
    )
