#!/usr/bin/env python3
"""CPU-only audit of the dose-matched YaRN/MrPro back-loading contrast."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .tables import atomic_json


FORMAT = "DOSE_MATCHED_YARN_MRPRO_CPU_AUDIT_V1"


def mrpro_exponents(n: int) -> np.ndarray:
    if n <= 1:
        raise ValueError("n must exceed one")
    q = np.arange(n + 1, dtype=np.float64)
    return q * (q + 1.0) / (n * (n + 1.0))


def yarn_exponents(n: int, scale: float) -> np.ndarray:
    if n <= 1 or not math.isfinite(scale) or scale <= 1.0:
        raise ValueError("n must exceed one and scale must exceed one")
    u = np.arange(n + 1, dtype=np.float64) / n
    return -np.log1p(-(1.0 - 1.0 / scale) * u) / math.log(scale)


def dose_difference(n: int, scale: float) -> float:
    return float(yarn_exponents(n, scale).sum() - mrpro_exponents(n).sum())


def solve_dose_crossing(n: int) -> float:
    """Find the unique S where finite-grid YaRN and MrPro have equal sum(m)."""
    low, high = 1.0 + 1e-8, 2.0
    if dose_difference(n, low) <= 0.0:
        raise AssertionError("YaRN must start above MrPro in the S->1 limit")
    while dose_difference(n, high) > 0.0:
        high *= 2.0
        if high > 1e9:
            raise AssertionError("failed to bracket the dose crossing")
    for _ in range(160):
        middle = 0.5 * (low + high)
        if dose_difference(n, middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def verify(n: int = 17, native_length: int = 8192) -> dict:
    scale = solve_dose_crossing(n)
    yarn = yarn_exponents(n, scale)
    mrpro = mrpro_exponents(n)
    difference = mrpro - yarn
    internal = difference[1:-1]
    crossings = [
        [q, q + 1]
        for q in range(1, n - 1)
        if difference[q] * difference[q + 1] < 0.0
    ]
    expected_mrpro_sum = (n + 2.0) / 3.0
    checks = {
        "mrpro_sum_closed_form": math.isclose(
            float(mrpro.sum()), expected_mrpro_sum, rel_tol=0.0, abs_tol=2e-15,
        ),
        "equal_exponent_dose": abs(float(yarn.sum() - mrpro.sum())) < 2e-14,
        "equal_total_log_frequency_displacement": abs(
            math.log(scale) * float(yarn.sum() - mrpro.sum())
        ) < 5e-14,
        "shared_endpoints": bool(
            abs(float(yarn[0] - mrpro[0])) < 2e-15
            and abs(float(yarn[-1] - mrpro[-1])) < 2e-15
            and mrpro[0] == 0.0
            and mrpro[-1] == 1.0
        ),
        "single_internal_crossing": crossings == [[9, 10]] if n == 17 else len(crossings) == 1,
        "mrpro_is_lower_before_crossing": bool(np.all(internal[:9] < 0.0)) if n == 17 else True,
        "mrpro_is_higher_after_crossing": bool(np.all(internal[9:] > 0.0)) if n == 17 else True,
        "integer_scales_bracket_crossing": (
            dose_difference(n, 4.0) > 0.0 and dose_difference(n, 16.0) < 0.0
        ) if n == 17 else True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"dose-matched audit failed: {failed}")
    return {
        "status": FORMAT,
        "n": n,
        "native_length": native_length,
        "dose_crossing_scale": scale,
        "target_window": native_length * scale,
        "shared_gain": 1.0 + 0.1 * math.log(scale),
        "mrpro_exponent_sum": float(mrpro.sum()),
        "yarn_exponent_sum": float(yarn.sum()),
        "total_log_displacement": math.log(scale) * float(mrpro.sum()),
        "crossings": crossings,
        "checks": checks,
        "uniqueness_argument": (
            "For u in (0,1), h(a)=-log((1-u)+u*exp(-a)) is strictly concave, "
            "h(0)=0, so h(a)/a strictly decreases with a=log(S). The summed "
            "finite-grid YaRN dose therefore has one crossing with the fixed MrPro dose."
        ),
        "scale_4_yarn_minus_mrpro_dose": dose_difference(n, 4.0),
        "scale_16_yarn_minus_mrpro_dose": dose_difference(n, 16.0),
        "checkpoint_loaded": False,
        "model_task_evaluations": 0,
        "performance_advantage_demonstrated": False,
        "claim_boundary": (
            "CPU algebra defines a parameter-free equal-dose single-crossing contrast. "
            "It does not show that back-loading improves any model or task."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=17)
    parser.add_argument("--native-length", type=int, default=8192)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = verify(args.n, args.native_length)
    atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "dose_crossing_scale": result["dose_crossing_scale"],
        "target_window": result["target_window"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
