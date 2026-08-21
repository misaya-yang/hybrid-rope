#!/usr/bin/env python3
"""Turn measured attention-distance mass into a RoPE phase-demand profile.

The primary kernel is the exact squared chord energy of one rotary pair,
``2 * (1 - cos(omega * Delta))``.  The factor two cancels on normalization.
This is a numerical candidate generator, not an identified LM-risk derivative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _demand(mass: np.ndarray, inv_freq: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    mass = np.asarray(mass, dtype=np.float64).copy()
    mass[0] = 0.0
    if mass.sum() <= 0:
        raise ValueError("non-self attention mass is zero")
    distance_probability = mass / mass.sum()
    phi = np.linspace(0.0, 1.0, bins)
    log_span = float(np.log(inv_freq[0] / inv_freq[-1]))
    omega = inv_freq[0] * np.exp(-log_span * phi)
    delta = np.arange(len(mass), dtype=np.float64)
    demand = distance_probability @ (1.0 - np.cos(delta[:, None] * omega[None, :]))
    demand = np.maximum(demand, np.finfo(np.float64).tiny)
    demand /= np.trapezoid(demand, phi)
    return phi, demand, log_span


def phase_demand(collection: Path, bins: int = 64) -> dict[str, object]:
    payload = np.load(collection, allow_pickle=False)
    inv_freq = np.asarray(payload["inv_freq"], dtype=np.float64)
    if inv_freq.ndim != 1 or len(inv_freq) < 2 or np.any(inv_freq <= 0):
        raise ValueError("collection has no valid inverse-frequency table")
    phi, demand, log_span = _demand(
        np.asarray(payload["mass"], dtype=np.float64).sum(axis=(0, 1)), inv_freq, bins
    )

    return {
        "schema_version": 1,
        "status": "PHASE_CHORD_DEMAND_COMPLETE",
        "delta": phi.tolist(),
        "m": demand.tolist(),
        "measurement": {
            "source_collection": str(collection.resolve()),
            "distance_mass": "all layers and heads; self distance excluded",
            "kernel": "1-cos(omega(phi)*Delta)",
            "interpretation": "bounded RoPE phase-discrimination energy, not an LM-risk derivative",
            "log_span": log_span,
            "bins": bins,
        },
    }


def layerwise_plan(collection: Path, *, bins: int, lam: float, base: float, k: int) -> dict[str, object]:
    from rebuttal.rebuttal_0723.experiments.demand_companding_5090.schedule import (
        endpoint_anchored_omega,
        quantile_phi,
    )

    payload = np.load(collection, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"].item()))
    inv_freq = np.asarray(payload["inv_freq"], dtype=np.float64)
    rows = []
    for layer_mass in np.asarray(payload["mass"], dtype=np.float64).sum(axis=1):
        phi, demand, _ = _demand(layer_mass, inv_freq, bins)
        rho = np.cbrt((1.0 - lam) * demand + lam)
        rho /= np.trapezoid(rho, phi)
        rows.append(endpoint_anchored_omega(quantile_phi(phi, rho, k), base, k=k).tolist())
    return {
        "schema_version": 1,
        "source_label": "phase_chord_layerwise",
        "num_layers": int(metadata["layers"]),
        "attention_type": "mha",
        "rope_dim": int(metadata["head_dim"]),
        "head_dim": int(metadata["head_dim"]),
        "num_heads": int(metadata["heads"]),
        "base": base,
        "train_length": int(metadata["length"]),
        "effective_dim": int(metadata["head_dim"]),
        "per_layer_inv_freq": rows,
        "measurement": {
            "source_collection": str(collection.resolve()),
            "kernel": "1-cos(omega(phi)*Delta)",
            "lambda": lam,
            "bins": bins,
        },
    }


def self_test() -> None:
    distance_probability = np.array([0.0, 1.0])
    phi = np.linspace(0.0, 1.0, 64)
    omega = np.exp(-5.0 * phi)
    demand = distance_probability @ (1.0 - np.cos(np.arange(2)[:, None] * omega[None, :]))
    assert np.all(demand > 0)
    assert demand[0] > demand[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--r3-output", type=Path)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--base", type=float, default=256.0)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('{"status":"PASS"}')
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    result = phase_demand(args.input, args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.r3_output is not None:
        plan = layerwise_plan(
            args.input,
            bins=args.bins,
            lam=args.lambda_value,
            base=args.base,
            k=args.K,
        )
        args.r3_output.parent.mkdir(parents=True, exist_ok=True)
        args.r3_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
