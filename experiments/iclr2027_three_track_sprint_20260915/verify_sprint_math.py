#!/usr/bin/env python3
"""CPU-only exact checks for the sprint's allocation identities."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path


def uniform(n: int) -> list[Fraction]:
    return [Fraction(q, n) for q in range(n + 1)]


def mrpro(n: int) -> list[Fraction]:
    return [Fraction(q * (q + 1), n * (n + 1)) for q in range(n + 1)]


def frontloaded(n: int) -> list[Fraction]:
    u, p = uniform(n), mrpro(n)
    return [2 * left - right for left, right in zip(u, p)]


def bm(n: int) -> list[Fraction]:
    return [
        Fraction(q * (q + 1) * (3 * n + 2 - 2 * q), n * (n + 1) * (n + 2))
        for q in range(n + 1)
    ]


def tailspline(n: int) -> list[Fraction]:
    return [
        Fraction(q * (3 * n * n + 3 * n + 1 - q * q), n * (n + 1) * (2 * n + 1))
        for q in range(n + 1)
    ]


def control_c(n: int) -> list[Fraction]:
    weight = Fraction(3 * n, 2 * (2 * n + 1))
    u, f = uniform(n), frontloaded(n)
    return [(1 - weight) * left + weight * right for left, right in zip(u, f)]


def increments(curve: list[Fraction]) -> list[Fraction]:
    return [right - left for left, right in zip(curve, curve[1:])]


def exact_profile(n: int, salt: int) -> list[Fraction]:
    weights = [Fraction(((index + 1) * (salt + 3)) % 17 + 1) for index in range(n)]
    total = sum(weights)
    return [value / total for value in weights]


def check_n(n: int) -> dict:
    t, c, b, u = tailspline(n), control_c(n), bm(n), uniform(n)
    weight = Fraction(3 * n, 2 * (2 * n + 1))
    expected_difference = [
        Fraction(q * (n - q) * (2 * q - n), 2 * n * (n + 1) * (2 * n + 1))
        for q in range(n + 1)
    ]
    assert t == [(1 - weight) * left + weight * right for left, right in zip(b, frontloaded(n))]
    assert [left - right for left, right in zip(t, c)] == expected_difference
    assert sum(expected_difference) == 0
    assert all(expected_difference[n - q] == -expected_difference[q] for q in range(n + 1))
    expected_sum = Fraction((n - 1) * (5 * n + 2), 4 * (2 * n + 1))
    assert sum(t[1:-1]) == expected_sum == sum(c[1:-1])
    assert t[0] == c[0] == 0 and t[-1] == c[-1] == 1
    assert all(left <= right for left, right in zip(t, t[1:]))
    if n <= 2:
        assert t == c
    else:
        boundary = -Fraction((n - 1) * (n - 2), 2 * n * (n + 1) * (2 * n + 1))
        et, ec = increments(t), increments(c)
        assert et[0] - ec[0] == boundary
        assert et[-1] - ec[-1] == boundary
    for salt in range(6):
        epsilon = exact_profile(n, salt)
        cumulative = [Fraction(0)]
        for value in epsilon:
            cumulative.append(cumulative[-1] + value)
        mu = sum(Fraction(index + 1) * value for index, value in enumerate(epsilon))
        assert sum(cumulative[1:-1]) == n - mu
    return {
        "n": n,
        "max_abs_t_minus_c": float(max(abs(value) for value in expected_difference)),
        "same_for_n_le_2": bool(t == c),
    }


def check_coordinate_relations() -> list[dict]:
    checks = []
    for pair_count in (16, 32, 64):
        for base in (10_000.0, 500_000.0, 1_000_000.0):
            for scale in (2.0, 4.0, 8.0):
                native_x = [index * math.log(base) / pair_count for index in range(pair_count)]
                native_span = native_x[-1] - native_x[0]
                m = [index / (pair_count - 1) for index in range(pair_count)]
                transformed = [x + math.log(scale) * exponent for x, exponent in zip(native_x, m)]
                span = native_span + math.log(scale)
                z = [(value - transformed[0]) / span for value in transformed]
                expected = [
                    (native_span * (x - native_x[0]) / native_span + math.log(scale) * exponent) / span
                    for x, exponent in zip(native_x, m)
                ]
                gaps = [right - left for left, right in zip(transformed, transformed[1:])]
                expected_gaps = [
                    native_x[index] - native_x[index - 1]
                    + (m[index] - m[index - 1]) * math.log(scale)
                    for index in range(1, pair_count)
                ]
                assert max(abs(left - right) for left, right in zip(z, expected)) < 1e-14
                assert max(abs(left - right) for left, right in zip(gaps, expected_gaps)) < 1e-14
                checks.append({"pair_count": pair_count, "base": base, "scale": scale})
    return checks


def check_quadratic_identity(n: int) -> None:
    optimum = increments(tailspline(n))

    def objective(value: list[Fraction]) -> Fraction:
        return sum((right - left) ** 2 for left, right in zip(value, value[1:])) + value[-1] ** 2

    for salt in range(6):
        perturbation = exact_profile(n, salt)
        mean = sum(perturbation) / n
        perturbation = [value - mean for value in perturbation]
        candidate = [left + Fraction(1, 100) * right for left, right in zip(optimum, perturbation)]
        residual = [left - right for left, right in zip(candidate, optimum)]
        quadratic = sum((right - left) ** 2 for left, right in zip(residual, residual[1:])) + residual[-1] ** 2
        assert objective(candidate) - objective(optimum) == quadratic


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [check_n(n) for n in range(1, 65)]
    for n in range(2, 65):
        check_quadratic_identity(n)
    coordinate = check_coordinate_relations()
    result = {
        "status": "ICLR2027_SPRINT_MATH_CHECKS_COMPLETE_V1",
        "exact_n_range": [1, 64],
        "profiles_per_n": 6,
        "coordinate_cases": len(coordinate),
        "verified": [
            "T1/T3 native-relative coordinate and gap relations",
            "T4/T5 displacement-centroid identity",
            "T6/T7 exact TailSpline-control difference",
            "T8 equal interior displacement",
            "T9 boundary increment differences and n=1/2 degeneracy",
            "T11 quadratic excess identity",
        ],
        "n17": rows[16],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_execution": False,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
