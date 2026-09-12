"""Exact E3 C42/C42V24 constructions and their deployed FP32 tables."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Any

import numpy as np


K = 64
BASE = 500_000
SCALE = 4
GAIN = 1.0 + 0.1 * math.log(SCALE)


def exact_increments(arm: str) -> list[Fraction]:
    if arm not in ("C42", "C42V24"):
        raise ValueError("arm must be C42 or C42V24")
    values = []
    for r in range(1, 19):
        p = Fraction(6 * r * (19 - r), 18 * 19 * 20)
        v = Fraction(2 * r - 19, 2)
        factor = 1 - Fraction(10, 119) * v
        if arm == "C42V24":
            factor += Fraction(35, 1496) * (v * v - Fraction(357, 20))
        values.append(p * factor)
    return values


def exact_exponents(arm: str) -> list[Fraction]:
    values = [Fraction(0) for _ in range(K)]
    cumulative = Fraction(0)
    for slot, increment in enumerate(exact_increments(arm), start=15):
        cumulative += increment
        values[slot] = cumulative
    for slot in range(33, K):
        values[slot] = Fraction(1)
    return values


def tensor_sha(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f4").copy(order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def construction_receipt(arm: str) -> dict[str, Any]:
    increments = exact_increments(arm)
    exponents = exact_exponents(arm)
    if sum(increments) != 1 or sum(Fraction(r) * value for r, value in enumerate(increments, 1)) != 8:
        raise AssertionError(f"{arm} increment mass/centroid changed")
    if sum(exponents) != 42 or exponents[:15] != [0] * 15 or exponents[32] != 1:
        raise AssertionError(f"{arm} fixed-total/band contract changed")
    native64 = np.power(float(BASE), -np.arange(K, dtype=np.float64) / K)
    m64 = np.asarray([float(value) for value in exponents], dtype=np.float64)
    deployed = np.asarray(native64 * np.power(float(SCALE), -m64), dtype=np.float32)
    return {
        "values_float32": deployed.tolist(),
        "tensor_sha256": tensor_sha(deployed),
        "gain": GAIN,
        "construction": {
            "method": "P1 section G.7 exact polynomial reconstruction",
            "identity_scope": "formula-reconstructed preregistered replication; not claimed as byte replay",
            "base": BASE,
            "scale": SCALE,
            "head_dim": 128,
            "transition_slots": [15, 32],
            "sum_m_exact": "42",
            "increment_mass_exact": "1",
            "increment_centroid_r_exact": "8",
            "increments_rational": [f"{value.numerator}/{value.denominator}" for value in increments],
            "cumulative_exponents_float64": m64.tolist(),
        },
    }


def tables() -> dict[str, Any]:
    native = np.power(float(BASE), -np.arange(K, dtype=np.float64) / K).astype(np.float32)
    return {
        "Native": {
            "values_float32": native.tolist(), "tensor_sha256": tensor_sha(native),
            "gain": 1.0, "construction": {"method": "identity", "base": BASE},
        },
        "C42": construction_receipt("C42"),
        "C42V24": construction_receipt("C42V24"),
    }


if __name__ == "__main__":
    print(json.dumps(tables(), indent=2, sort_keys=True))
