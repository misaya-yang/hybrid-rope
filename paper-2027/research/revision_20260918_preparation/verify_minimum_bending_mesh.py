"""CPU-only audit of TailSpline's minimum-bending mesh interpretation.

This script uses public RoPE parameters only.  It loads no checkpoint, CUDA,
task output, or calibration score.  The frame diagnostics are descriptive:
they do not turn a finite-grid lower eigenvalue into a task-performance claim.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("TailSpline", "MrPro", "C", "BM", "Uniform")
RELATIVE_EIGEN_TOLERANCE = 1e-12


def profiles(n: int) -> dict[str, list[F]]:
    q = range(n + 1)
    uniform = [F(i, n) for i in q]
    mrpro = [F(i * (i + 1), n * (n + 1)) for i in q]
    bm = [F(i * (i + 1) * (3 * n + 2 - 2 * i), n * (n + 1) * (n + 2)) for i in q]
    tailspline = [
        F(i * (3 * n * n + 3 * n + 1 - i * i), n * (n + 1) * (2 * n + 1))
        for i in q
    ]
    weight = F(3 * n, 2 * (2 * n + 1))
    front = [2 * u - p for u, p in zip(uniform, mrpro)]
    control = [(1 - weight) * u + weight * f for u, f in zip(uniform, front)]
    return {
        "TailSpline": tailspline,
        "MrPro": mrpro,
        "C": control,
        "BM": bm,
        "Uniform": uniform,
    }


def bending_energy(profile: list[F]) -> F:
    increments = [profile[i] - profile[i - 1] for i in range(1, len(profile))]
    return sum((increments[i + 1] - increments[i]) ** 2 for i in range(len(increments) - 1)) + increments[-1] ** 2


def exact_bending_checks() -> dict:
    checked = 0
    for n in range(1, 129):
        values = profiles(n)
        tail = values["TailSpline"]
        increments = [tail[i] - tail[i - 1] for i in range(1, n + 1)]
        he = [F(0) for _ in increments]
        for i in range(n - 1):
            delta = increments[i + 1] - increments[i]
            he[i] -= delta
            he[i + 1] += delta
        he[-1] += increments[-1]
        optimum = F(6, n * (n + 1) * (2 * n + 1))
        assert all(value == optimum for value in he)
        assert bending_energy(tail) == optimum
        assert min(increments) > 0 and sum(increments) == 1
        checked += 1
    cases = {}
    for name, n in (("Llama", 17), ("OLMo", 18)):
        energies = {method: bending_energy(profile) for method, profile in profiles(n).items()}
        mrpro = energies["MrPro"]
        cases[name] = {
            "transition_width_n": n,
            "normalized_bending_energy": {method: float(energies[method]) for method in METHODS},
            "ratio_to_mrpro": {method: float(energies[method] / mrpro) for method in METHODS},
            "tailspline_reduction_vs_mrpro_percent": float(100 * (1 - energies["TailSpline"] / mrpro)),
        }
    return {"exact_kkt_widths_checked": checked, "cases": cases}


def public_table(pair_count: int, base: float, native_length: int, scale: float) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
    native = np.exp(-math.log(base) * np.arange(pair_count, dtype=float) / pair_count)
    turns = native * native_length / (2 * math.pi)
    low = int(np.flatnonzero(turns > 32)[-1])
    high = int(np.flatnonzero(turns < 1)[0])
    n = high - low
    result = {}
    for method, exact_profile in profiles(n).items():
        profile = np.asarray([float(value) for value in exact_profile])
        whole = np.concatenate((np.zeros(low), profile, np.ones(pair_count - 1 - high)))
        assert len(whole) == pair_count
        table = native * np.power(scale, -whole)
        assert np.all(table > 0) and np.all(np.diff(table) < 0)
        result[method] = table
    return (low, high), result


def centered_discrete_pair_whitened_gram(frequencies: np.ndarray, length: int) -> np.ndarray:
    difference = frequencies[:, None] - frequencies[None, :]
    total = frequencies[:, None] + frequencies[None, :]

    def mean_cosine(value: np.ndarray) -> np.ndarray:
        return np.sinc(length * value / (2 * np.pi)) / np.sinc(value / (2 * np.pi))

    cos_cos = (mean_cosine(difference) + mean_cosine(total)) / 2
    sin_sin = (mean_cosine(difference) - mean_cosine(total)) / 2
    cos_cos /= np.sqrt(np.diag(cos_cos)[:, None] * np.diag(cos_cos)[None, :])
    sin_sin /= np.sqrt(np.diag(sin_sin)[:, None] * np.diag(sin_sin)[None, :])
    zero = np.zeros_like(cos_cos)
    gram = np.block([[cos_cos, zero], [zero, sin_sin]])
    return (gram + gram.T) / 2


def spectrum_summary(frequencies: np.ndarray, length: int) -> dict:
    gram = centered_discrete_pair_whitened_gram(frequencies, length)
    eigenvalues = np.linalg.eigvalsh(gram)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    tolerance = largest * RELATIVE_EIGEN_TOLERANCE
    resolved = smallest > tolerance
    clipped = np.maximum(eigenvalues, 0)
    return {
        "dimension": len(eigenvalues),
        "lambda_min_float64": smallest,
        "lambda_max": largest,
        "lambda_min_resolved_at_relative_1e_12": resolved,
        "condition_number_if_resolved": float(largest / smallest) if resolved else None,
        "numerical_rank_relative_1e_12": int(np.count_nonzero(eigenvalues > tolerance)),
        "r2_effective_rank": float(clipped.sum() ** 2 / np.square(clipped).sum()),
    }


def frame_audit() -> dict:
    models = (
        ("Llama", 64, 500_000.0, 8192),
        ("OLMo", 64, 500_000.0, 4096),
    )
    output = {}
    tail_below_mrpro_r2 = True
    for name, pair_count, base, native_length in models:
        band, tables = public_table(pair_count, base, native_length, 4.0)
        model = {"native_length": native_length, "band_inclusive": list(band), "windows": {}}
        for length in (native_length, 2 * native_length, 4 * native_length):
            rows = {}
            for method in METHODS:
                table = tables[method]
                rows[method] = {
                    "full": spectrum_summary(table, length),
                    "transition_inclusive": spectrum_summary(table[band[0] : band[1] + 1], length),
                }
            tail_below_mrpro_r2 &= rows["TailSpline"]["full"]["r2_effective_rank"] < rows["MrPro"]["full"]["r2_effective_rank"]
            model["windows"][str(length)] = rows
        output[name] = model
    assert tail_below_mrpro_r2
    unresolved_full = sum(
        not metrics["full"]["lambda_min_resolved_at_relative_1e_12"]
        for model in output.values()
        for window in model["windows"].values()
        for metrics in window.values()
    )
    return {
        "convention": "centered discrete positions; real cosine/sine pairs; each pair block-whitened; band endpoints included",
        "relative_eigen_tolerance": RELATIVE_EIGEN_TOLERANCE,
        "models": output,
        "checks": {
            "tailspline_full_r2_below_mrpro_at_all_six_windows": tail_below_mrpro_r2,
            "unresolved_full_lambda_min_count": unresolved_full,
            "total_full_spectra": 30,
        },
        "frame_admissibility_conclusion": "not_identified: no independent lower-bound threshold was specified, and many full-Gram minima are below float64 resolution",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "MINIMUM_BENDING_MESH_CPU_AUDIT_COMPLETE_V1",
        "scope": "public RoPE parameters only; no checkpoint, CUDA, task output, or fitted threshold",
        "bending": exact_bending_checks(),
        "frame": frame_audit(),
        "interpretation": {
            "supported": "TailSpline is the unique minimizer of the stated one-sided low-tail-junction bending functional.",
            "not_supported": "Nyquist optimality, a task-performance mechanism, or a frame-stability advantage over MrPro.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "status": report["status"],
        "llama_reduction_percent": report["bending"]["cases"]["Llama"]["tailspline_reduction_vs_mrpro_percent"],
        "olmo_reduction_percent": report["bending"]["cases"]["OLMo"]["tailspline_reduction_vs_mrpro_percent"],
        "unresolved_full_lambda_min_count": report["frame"]["checks"]["unresolved_full_lambda_min_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
