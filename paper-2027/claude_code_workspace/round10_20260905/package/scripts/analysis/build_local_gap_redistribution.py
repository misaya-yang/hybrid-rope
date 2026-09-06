#!/usr/bin/env python3
"""Build monotone local-gap redistribution candidates from one frozen table.

The construction works in ``x=-log(omega)`` gap space.  It protects every pair
through ``critical_start_gap``, compresses a smooth critical-band gap window,
and pays back exactly the same log-span in a slower donor window.  Positive
gaps, both endpoints, support, pair count, and the serving gain are fixed; only
interior allocation ``z`` changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critical-start-gap", type=int, default=35)
    parser.add_argument("--critical-stop-gap", type=int, default=43)
    parser.add_argument("--donor-start-gap", type=int, default=54)
    parser.add_argument("--donor-stop-gap", type=int, default=62)
    parser.add_argument("--mass", nargs="+", type=float, default=(0.05, 0.10, 0.20))
    parser.add_argument("--attention-scaling", type=float, default=1.138629436111989)
    parser.add_argument("--label", default="", help="short identity suffix for band scans")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes()
    ).hexdigest()


def smooth_weights(start: int, stop: int, size: int) -> np.ndarray:
    if not 0 <= start <= stop < size:
        raise ValueError(f"invalid gap window [{start}, {stop}] for {size} gaps")
    count = stop - start + 1
    local = np.hanning(count + 2)[1:-1]
    local /= local.sum()
    weights = np.zeros(size, dtype=np.float64)
    weights[start : stop + 1] = local
    return weights


def main() -> int:
    args = parse_args()
    base_path = args.base_table.resolve()
    base = np.load(base_path, allow_pickle=False)
    if (
        base.dtype != np.dtype("float32")
        or base.ndim != 1
        or base.size < 8
        or not np.isfinite(base).all()
        or not np.all(base[:-1] > base[1:])
    ):
        raise RuntimeError("base table must be finite, decreasing float32")
    if not math.isfinite(float(args.attention_scaling)) or args.attention_scaling <= 0.0:
        raise ValueError("attention scaling must be finite and positive")

    x = -np.log(base.astype(np.float64))
    gaps = np.diff(x)
    critical = smooth_weights(
        int(args.critical_start_gap), int(args.critical_stop_gap), gaps.size
    )
    donor = smooth_weights(
        int(args.donor_start_gap), int(args.donor_stop_gap), gaps.size
    )
    if np.any((critical > 0.0) & (donor > 0.0)):
        raise RuntimeError("critical and donor gap windows must be disjoint")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, Any]] = []
    for value in args.mass:
        mass = float(value)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("redistribution masses must be finite and positive")
        active_gaps = gaps + mass * (donor - critical)
        if not np.all(active_gaps > 0.0):
            raise RuntimeError(f"mass={mass:g} creates a non-positive log gap")
        active_x = np.concatenate(([x[0]], x[0] + np.cumsum(active_gaps)))
        table = np.exp(-active_x).astype("<f4")
        protected_stop = min(
            int(args.critical_start_gap), int(args.donor_start_gap)
        )
        table[: protected_stop + 1] = base[: protected_stop + 1]
        table[-1] = base[-1]
        if not np.all(table[:-1] > table[1:]):
            raise RuntimeError(f"mass={mass:g} creates a frequency crossing")
        label = str(args.label).strip()
        stem = f"s4_localgap_{label}_m{mass:g}" if label else f"s4_localgap_m{mass:g}"
        name = stem.replace(".", "p")
        path = output / f"{name}.npy"
        np.save(path, table, allow_pickle=False)
        candidates.append({
            "name": name,
            "path": str(path),
            "table_sha256_float32": float32_sha256(table),
            "mass": mass,
            "critical_gap_window": [
                int(args.critical_start_gap), int(args.critical_stop_gap)
            ],
            "donor_gap_window": [int(args.donor_start_gap), int(args.donor_stop_gap)],
            "protected_through_pair": protected_stop,
            "minimum_log_gap": float(active_gaps.min()),
            "support_fixed": bool(table[0] == base[0] and table[-1] == base[-1]),
            "attention_scaling": float(args.attention_scaling),
        })

    payload = {
        "status": "LOCAL_GAP_REDISTRIBUTION_FROZEN_V1",
        "construction": (
            "compress smooth critical gaps; expand smooth ultra-slow donor gaps; "
            "preserve positive gaps and total log span"
        ),
        "base_table": str(base_path),
        "base_table_sha256_float32": float32_sha256(base),
        "base_file_sha256": sha256_file(base_path),
        "pair_count": int(base.size),
        "candidates": candidates,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(candidates), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
