#!/usr/bin/env python3
"""Deterministic model-free diagnostics for the frozen MLA schedules.

These diagnostics are preflight checks, not language-model evidence. They
verify endpoint/range matching and quantify the cosine-feature collision proxy
for the active frequencies only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.mla_scarcity_5090.protocol import (
    ARMS,
    FREQUENCY_PAIRS,
    SPEC,
    schedule_phi,
    training_inv_freq,
)


REGIONS = (
    ("in_domain", 1, 4_096),
    ("one_to_two_x", 4_097, 8_192),
    ("two_to_four_x", 8_193, 16_384),
    ("four_to_eight_x", 16_385, 32_768),
)


def _kernel_stats(
    inv_freq: np.ndarray, start: int, stop: int
) -> dict[str, Any]:
    deltas = np.arange(start, stop + 1, dtype=np.float64)
    kernel = np.cos(deltas[:, None] * inv_freq[None, :]).mean(axis=1)
    absolute = np.abs(kernel)
    return {
        "delta_start": int(start),
        "delta_stop": int(stop),
        "count": int(len(deltas)),
        "mean_kernel": float(kernel.mean()),
        "mean_abs_kernel": float(absolute.mean()),
        "rms_kernel": float(np.sqrt(np.mean(kernel**2))),
        "p95_abs_kernel": float(np.quantile(absolute, 0.95)),
        "max_abs_kernel": float(absolute.max()),
    }


def _schedule_record(arm: str, pairs: int) -> dict[str, Any]:
    phi_tensor, metadata = schedule_phi(arm, pairs)
    inv_tensor, inv_metadata = training_inv_freq(arm, pairs)
    phi = phi_tensor.detach().cpu().numpy().astype(np.float64)
    active = (
        inv_tensor[:pairs].detach().cpu().numpy().astype(np.float64)
    )
    gaps = np.diff(phi)
    return {
        "arm": arm,
        "active_frequency_pairs": pairs,
        "phi": phi.tolist(),
        "active_inv_freq": active.tolist(),
        "phi_min": float(phi[0]),
        "phi_max": float(phi[-1]),
        "phi_span": float(phi[-1] - phi[0]),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "gap_mean": float(gaps.mean()),
        "gap_cv": float(gaps.std(ddof=0) / gaps.mean()),
        "metadata": metadata,
        "padded_schedule_metadata": inv_metadata,
        "collision_proxy": {
            label: _kernel_stats(active, start, stop)
            for label, start, stop in REGIONS
        },
    }


def build_diagnostics() -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for pairs in FREQUENCY_PAIRS:
        records = {arm: _schedule_record(arm, pairs) for arm in ARMS}
        native = records["native_geo"]
        control = records["range_matched_uniform"]
        evq = records["evq_cosh"]
        if not (
            control["phi_min"] == evq["phi_min"]
            and control["phi_max"] == evq["phi_max"]
            and control["phi_span"] == evq["phi_span"]
        ):
            raise RuntimeError(f"K={pairs} range control is not endpoint matched")
        phi_control = np.asarray(control["phi"], dtype=np.float64)
        phi_evq = np.asarray(evq["phi"], dtype=np.float64)
        schedules[f"k{pairs}"] = records
        region_contrasts = {}
        for label, _, _ in REGIONS:
            native_rms = native["collision_proxy"][label]["rms_kernel"]
            control_rms = control["collision_proxy"][label]["rms_kernel"]
            evq_rms = evq["collision_proxy"][label]["rms_kernel"]
            region_contrasts[label] = {
                "native_minus_range_rms": native_rms - control_rms,
                "range_minus_evq_rms": control_rms - evq_rms,
                "evq_minus_native_rms": evq_rms - native_rms,
            }
        contrasts[f"k{pairs}"] = {
            "evq_vs_range_phi_rms": float(
                np.sqrt(np.mean((phi_evq - phi_control) ** 2))
            ),
            "regions": region_contrasts,
        }
    scarcity_proxy = {}
    for label, _, _ in REGIONS:
        k8 = contrasts["k8"]["regions"][label]["range_minus_evq_rms"]
        k32 = contrasts["k32"]["regions"][label]["range_minus_evq_rms"]
        scarcity_proxy[label] = {
            "k8_range_minus_evq_rms": k8,
            "k32_range_minus_evq_rms": k32,
            "k8_minus_k32": k8 - k32,
        }
    return {
        "schema_version": 1,
        "status": "MODEL_FREE_DIAGNOSTIC_ONLY",
        "protocol_sha256": SPEC.fingerprint(),
        "definition": (
            "K(delta)=mean_k cos(delta*omega_k), computed over active "
            "frequency pairs only. Lower RMS is treated only as a collision "
            "proxy; it is not an LM loss or a preregistered success metric."
        ),
        "schedules": schedules,
        "contrasts": contrasts,
        "scarcity_proxy": scarcity_proxy,
        "limitations": [
            "No learned attention weights, token distribution, or task loss.",
            "Inactive identity pairs and MLA content projections are excluded.",
            "Proxy signs do not predict or override the seed-42 GPU gate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_diagnostics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
