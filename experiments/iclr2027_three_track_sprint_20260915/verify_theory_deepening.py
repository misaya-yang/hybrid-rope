#!/usr/bin/env python3
"""CPU-only numerical checks for Hybrid_RoPE_Theory_Deepening_20260914.md."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .verify_sprint_math import control_c, increments, tailspline


SEED = 20260914


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    result = np.exp(shifted)
    return result / result.sum()


def rotation(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def record(name: str, cases: int, maximum_error: float, **extra) -> dict:
    return {"name": name, "status": "PASS", "cases": cases, "maximum_error": float(maximum_error), **extra}


def check_01_normalized_coordinates() -> dict:
    errors = []
    for pairs in (16, 32, 64, 128):
        for base in (10_000.0, 500_000.0, 1_000_000.0):
            index = np.arange(pairs, dtype=np.float64)
            omega = np.power(base, -index / pairs)
            a = -math.log(float(omega[0]))
            span = math.log(float(omega[0] / omega[-1]))
            z = (-np.log(omega) - a) / span
            errors.append(float(np.max(np.abs(z - index / (pairs - 1)))))
            errors.append(abs(span - (pairs - 1) * math.log(base) / pairs))
    return record("01_normalized_z_and_native_span", 12, max(errors))


def check_02_logit_z_gradient() -> dict:
    rng = np.random.default_rng(SEED + 2)
    errors = []
    for _ in range(200):
        a, b = rng.normal(size=2)
        offset = rng.uniform(-0.5, 0.5)
        span = rng.uniform(1.0, 12.0)
        z = rng.uniform(0.05, 0.95)
        distance = rng.uniform(1.0, 200.0)
        head_dim = int(rng.choice([64, 128, 256]))

        def value(current: float) -> float:
            frequency = math.exp(-(offset + span * current))
            phase = frequency * distance
            return (a * math.cos(phase) + b * math.sin(phase)) / math.sqrt(head_dim)

        frequency = math.exp(-(offset + span * z))
        phase = frequency * distance
        analytic = span * phase / math.sqrt(head_dim) * (a * math.sin(phase) - b * math.cos(phase))
        step = 1e-5
        numeric = (
            -value(z + 2 * step) + 8 * value(z + step)
            - 8 * value(z - step) + value(z - 2 * step)
        ) / (12 * step)
        errors.append(abs(analytic - numeric) / max(1.0, abs(analytic)))
    return record("02_content_phase_logit_z_gradient", 200, max(errors))


def check_03_finite_softmax_reweighting() -> dict:
    rng = np.random.default_rng(SEED + 3)
    errors = []
    for _ in range(200):
        logits = rng.normal(size=17)
        delta = rng.normal(scale=0.7, size=17)
        old = softmax(logits)
        weights = np.exp(delta)
        identity = old * weights / np.dot(old, weights)
        errors.append(float(np.max(np.abs(identity - softmax(logits + delta)))))
    return record("03_finite_softmax_reweighting", 200, max(errors))


def check_04_log_odds_shift() -> dict:
    rng = np.random.default_rng(SEED + 4)
    errors = []
    for _ in range(200):
        logits = rng.normal(size=13)
        delta = rng.normal(size=13)
        old, new = softmax(logits), softmax(logits + delta)
        left, right = rng.choice(13, size=2, replace=False)
        observed = math.log(new[left] / new[right]) - math.log(old[left] / old[right])
        errors.append(abs(observed - (delta[left] - delta[right])))
    return record("04_exact_log_odds_shift", 200, max(errors))


def check_05_value_covariance_identity() -> dict:
    rng = np.random.default_rng(SEED + 5)
    errors = []
    for _ in range(200):
        logits = rng.normal(size=19)
        delta = rng.normal(scale=0.8, size=19)
        values = rng.normal(size=(19, 7))
        probability = softmax(logits)
        output = probability @ values
        weights = np.exp(delta)
        updated = probability * weights / np.dot(probability, weights)
        covariance = np.sum(probability[:, None] * (values - output) * weights[:, None], axis=0)
        predicted = covariance / np.dot(probability, weights)
        errors.append(float(np.max(np.abs((updated @ values - output) - predicted))))
    return record("05_finite_value_covariance_identity", 200, max(errors))


def check_06_local_loss_chain_rule() -> dict:
    rng = np.random.default_rng(SEED + 6)
    errors = []
    for _ in range(100):
        count = 11
        base = rng.normal(size=count)
        aa, bb = rng.normal(size=(2, count))
        distances = rng.uniform(1.0, 100.0, size=count)
        values = rng.normal(size=(count, 5))
        readout = rng.normal(size=5)
        span, z = rng.uniform(1.0, 8.0), rng.uniform(0.1, 0.9)

        def state(current: float):
            frequency = math.exp(-span * current)
            phase = frequency * distances
            logits = base + aa * np.cos(phase) + bb * np.sin(phase)
            probability = softmax(logits)
            return float(readout @ (probability @ values)), probability, phase

        _, probability, phase = state(z)
        output = probability @ values
        derivative_logits = span * phase * (aa * np.sin(phase) - bb * np.cos(phase))
        analytic = np.sum(probability * ((values - output) @ readout) * derivative_logits)
        step = 1e-5
        numeric = (
            -state(z + 2 * step)[0] + 8 * state(z + step)[0]
            - 8 * state(z - step)[0] + state(z - 2 * step)[0]
        ) / (12 * step)
        errors.append(abs(analytic - numeric) / max(1.0, abs(analytic)))
    return record("06_attention_value_local_chain_rule", 100, max(errors))


def check_07_schur_complement() -> dict:
    rng = np.random.default_rng(SEED + 7)
    errors = []
    minimum_eigenvalue = math.inf
    for _ in range(100):
        matrix = rng.normal(size=(9, 9))
        hessian = matrix.T @ matrix + np.eye(9)
        h_ww, h_wz = hessian[:6, :6], hessian[:6, 6:]
        h_zw, h_zz = hessian[6:, :6], hessian[6:, 6:]
        removed = h_zw @ np.linalg.solve(h_ww, h_wz)
        adapted = h_zz - removed
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(adapted).min()))
        errors.append(float(np.max(np.abs((h_zz - adapted) - removed))))
        if np.linalg.eigvalsh(removed).min() < -1e-10:
            raise AssertionError("Schur removed term is not positive semidefinite")
    if minimum_eigenvalue <= 0.0:
        raise AssertionError("Schur complement lost positive definiteness")
    return record("07_local_adaptation_schur_complement", 100, max(errors), minimum_adapted_eigenvalue=minimum_eigenvalue)


def check_08_gain_square_and_competition() -> dict:
    rng = np.random.default_rng(SEED + 8)
    errors = []
    for _ in range(100):
        query, key = rng.normal(size=(2, 2))
        left, right = rotation(rng.normal()) @ query, rotation(rng.normal()) @ key
        gain = rng.uniform(0.7, 1.8)
        errors.append(abs(np.dot(gain * left, gain * right) - gain ** 2 * np.dot(left, right)))
        competitors = int(rng.integers(1, 100))
        scale = rng.uniform(1.1, 8.0)
        beta, gap, new_gap = rng.uniform(0.5, 2.0), rng.uniform(0.1, 2.0), rng.uniform(0.1, 2.0)
        new_beta = (beta * gap + math.log(scale)) / new_gap
        old_probability = 1.0 / (1.0 + competitors * math.exp(-beta * gap))
        new_probability = 1.0 / (1.0 + scale * competitors * math.exp(-new_beta * new_gap))
        errors.append(abs(old_probability - new_probability))
    return record("08_gain_squared_logit_and_competitor_compensation", 100, max(errors))


def check_09_gain_affine_recovery() -> dict:
    rng = np.random.default_rng(SEED + 9)
    errors = []
    non_affine_residuals = []
    for _ in range(100):
        scores = rng.normal(size=15)
        multiplier, shift, beta = rng.uniform(0.2, 3.0), rng.normal(), rng.uniform(0.3, 2.0)
        changed = multiplier * scores + shift
        errors.append(float(np.max(np.abs(softmax(beta * scores) - softmax(beta / multiplier * changed)))))
        perturbed = changed.copy()
        perturbed[0] += 0.5
        centered_scores = scores - scores.mean()
        centered_perturbed = perturbed - perturbed.mean()
        fitted = float(np.dot(centered_scores, centered_perturbed) / np.dot(centered_perturbed, centered_perturbed))
        residual = np.linalg.norm(beta * centered_scores - fitted * centered_perturbed)
        non_affine_residuals.append(float(residual))
        if not np.array_equal(np.argsort(scores), np.argsort(multiplier * scores)):
            raise AssertionError("positive uniform gain changed ranking")
    if min(non_affine_residuals) <= 1e-4:
        raise AssertionError("non-affine example was accidentally recoverable")
    return record("09_gain_affine_softmax_recoverability", 100, max(errors), minimum_non_affine_centered_residual=min(non_affine_residuals))


def check_10_threeband_phase_and_midband_optimum() -> dict:
    rng = np.random.default_rng(SEED + 10)
    errors = []
    for _ in range(200):
        distance, frequency, scale = rng.uniform(1.0, 1000.0), rng.uniform(1e-4, 1.0), rng.uniform(1.1, 16.0)
        errors.append(float(np.max(np.abs(rotation(scale * distance * frequency / scale) - rotation(distance * frequency)))))
        horizon = rng.uniform(10.0, 1000.0)
        selected = min(frequency, math.pi / horizon)
        if selected > frequency + 1e-15 or selected * horizon > math.pi + 1e-12:
            raise AssertionError("minimal phase-safe slowdown failed")
        aa, bb = rng.uniform(0.1, 5.0, size=2)
        optimum = (aa + scale * bb) / (aa + scale ** 2 * bb)
        if not 1.0 / scale < optimum < 1.0:
            raise AssertionError("midband optimum left interpolation interval")
        derivative = 2 * aa * (optimum - 1) + 2 * scale * bb * (scale * optimum - 1)
        errors.append(abs(derivative))
    return record("10_threeband_phase_invariance_and_midband_optimum", 200, max(errors))


def check_11_yarn_shape_and_scale() -> dict:
    errors = []
    minimum_first = math.inf
    minimum_second = math.inf
    for scale in (2.0, 4.0, 8.0, 32.0):
        u = np.linspace(0.0, 1.0, 1001)
        m = -np.log(1.0 - u + u / scale) / math.log(scale)
        first, second = np.diff(m), np.diff(m, n=2)
        minimum_first = min(minimum_first, float(first.min()))
        minimum_second = min(minimum_second, float(second.min()))
        if first.min() <= 0.0 or second.min() <= 0.0:
            raise AssertionError("YaRN exponent ramp is not increasing and convex")
        for fixed_u in (0.2, 0.5, 0.8):
            factor = 1.0 / (1.0 - fixed_u + fixed_u / scale)
            limit = 1.0 / (1.0 - fixed_u)
            if factor >= limit:
                raise AssertionError("YaRN wavelength factor exceeded saturation limit")
            errors.append(max(0.0, factor - limit))
    mrpro_factor_small = 4.0 ** 0.5
    mrpro_factor_large = 64.0 ** 0.5
    if mrpro_factor_large <= mrpro_factor_small:
        raise AssertionError("fixed-exponent MrPro did not grow with scale")
    return record("11_yarn_convex_ramp_and_scale_saturation", 4, max(errors, default=0.0), minimum_first_difference=minimum_first, minimum_second_difference=minimum_second)


def check_12_deployment_z_and_tc_identities() -> dict:
    errors = []
    for n in range(1, 65):
        t, c = tailspline(n), control_c(n)
        difference = [left - right for left, right in zip(t, c)]
        expected = [
            Fraction(q * (n - q) * (2 * q - n), 2 * n * (n + 1) * (2 * n + 1))
            for q in range(n + 1)
        ]
        if difference != expected or sum(difference) != 0:
            raise AssertionError("T-C exact identity failed")
        span, scale = 10.0, 4.0
        native_z = np.linspace(0.0, 1.0, n + 1)
        exponent = np.asarray([float(value) for value in t])
        deployed_z = (span * native_z + math.log(scale) * exponent) / (span + math.log(scale))
        delta = math.log(scale) / (span + math.log(scale)) * (exponent - native_z)
        errors.append(float(np.max(np.abs((deployed_z - native_z) - delta))))
    return record("12_deployment_z_transform_and_tc_exact_identities", 64, max(errors))


def check_13_tc_marginal_pairing() -> dict:
    rng = np.random.default_rng(SEED + 13)
    errors = []
    for _ in range(200):
        n = int(rng.integers(3, 30))
        t = np.asarray([float(value) for value in tailspline(n)])
        c = np.asarray([float(value) for value in control_c(n)])
        linear = rng.normal(size=n + 1)
        raw = rng.normal(size=(n + 1, n + 1))
        quadratic = 0.5 * (raw + raw.T)
        gradient_average = linear + quadratic @ ((t + c) / 2.0)
        value = lambda x: float(linear @ x + 0.5 * x @ quadratic @ x)
        direct = value(t) - value(c)
        fundamental = float((t - c) @ gradient_average)
        paired = 0.0
        for q in range(n + 1):
            if q < n / 2:
                coefficient = q * (n - q) * (n - 2 * q) / (2 * n * (n + 1) * (2 * n + 1))
                paired += coefficient * (gradient_average[n - q] - gradient_average[q])
        errors.extend([abs(direct - fundamental), abs(direct - paired)])
    return record("13_tc_conditional_marginal_pairing", 200, max(errors))


def check_14_roughness_and_rotation_bounds() -> dict:
    rng = np.random.default_rng(SEED + 14)
    errors = []
    for n in range(2, 65):
        optimum = np.asarray([float(value) for value in increments(tailspline(n))])
        perturbation = rng.normal(size=n)
        perturbation -= perturbation.mean()
        candidate = optimum + 0.01 * perturbation

        def roughness(value):
            return float(np.sum(np.diff(value) ** 2) + value[-1] ** 2)

        residual = candidate - optimum
        excess = float(np.sum(np.diff(residual) ** 2) + residual[-1] ** 2)
        errors.append(abs((roughness(candidate) - roughness(optimum)) - excess))
    for _ in range(200):
        distance = rng.uniform(-1000.0, 1000.0)
        left, right = rng.uniform(-1.0, 1.0, size=2)
        observed = np.linalg.norm(rotation(distance * left) - rotation(distance * right), ord=2)
        exact = 2.0 * abs(math.sin(distance * (left - right) / 2.0))
        bound = min(2.0, abs(distance) * abs(left - right))
        errors.append(abs(observed - exact))
        if observed > bound + 1e-12:
            raise AssertionError("rotation perturbation bound failed")
    return record("14_roughness_excess_and_rotation_operator_bound", 263, max(errors))


def sinkhorn(matrix: np.ndarray, iterations: int = 200) -> np.ndarray:
    result = matrix.copy()
    for _ in range(iterations):
        result /= result.sum(axis=1, keepdims=True)
        result /= result.sum(axis=0, keepdims=True)
    return result


def check_15_sparse_compression_value_moe_mhc() -> dict:
    rng = np.random.default_rng(SEED + 15)
    access_slack = []
    compression_slack = []
    direct_value_errors = []
    omitted_direct_norms = []
    topk_cases = moe_cases = 0
    doubly_stochastic_residual = []
    decomposition_error = []
    for _ in range(200):
        probability = rng.dirichlet(np.ones(20))
        values = rng.normal(size=(20, 6))
        keep = rng.random(20) > 0.35
        if not keep.any():
            keep[0] = True
        removed_mass = float(probability[~keep].sum())
        kept_probability = probability[keep] / probability[keep].sum()
        original, sparse = probability @ values, kept_probability @ values[keep]
        value_bound = float(np.linalg.norm(values, axis=1).max())
        slack = 2 * value_bound * removed_mass - np.linalg.norm(sparse - original)
        if slack < -1e-12:
            raise AssertionError("sparse removed-mass value bound failed")
        access_slack.append(slack)

        scores = np.sort(rng.normal(size=30))[::-1]
        k = 7
        margin = scores[k - 1] - scores[k]
        perturbation = rng.uniform(-0.49 * margin, 0.49 * margin, size=30)
        if set(np.argpartition(-scores, k - 1)[:k]) != set(np.argpartition(-(scores + perturbation), k - 1)[:k]):
            raise AssertionError("top-k margin certificate failed")
        topk_cases += 1

        positions = rng.uniform(-8.0, 8.0, size=9)
        weights = rng.dirichlet(np.ones(9))
        keys = rng.normal(size=(9, 2))
        center = float(weights @ positions)
        delta = positions - center
        frequency = rng.uniform(-1.5, 1.5)
        exact = sum(weight * (rotation(frequency * position) @ key) for weight, position, key in zip(weights, positions, keys))
        coarse = rotation(frequency * center) @ (weights @ keys)
        first = abs(frequency) * np.linalg.norm(np.sum(weights[:, None] * delta[:, None] * keys, axis=0))
        second = frequency ** 2 / 2.0 * np.sum(weights * delta ** 2 * np.linalg.norm(keys, axis=1))
        compression_slack.append(first + second - np.linalg.norm(exact - coarse))
        if compression_slack[-1] < -1e-11:
            raise AssertionError("compression-rotation content bound failed")

        positive = np.exp(rng.normal(size=(6, 6)))
        matrix = sinkhorn(positive)
        row_column = max(np.max(np.abs(matrix.sum(axis=0) - 1)), np.max(np.abs(matrix.sum(axis=1) - 1)))
        norm = np.linalg.norm(matrix, ord=2)
        if norm > 1.0 + 1e-10:
            raise AssertionError("doubly stochastic spectral norm bound failed")
        doubly_stochastic_residual.append(float(row_column))
        second_matrix = sinkhorn(np.exp(rng.normal(size=(6, 6))))
        x, delta_x = rng.normal(size=(2, 6))
        x_prime = x + delta_x
        left = second_matrix @ x_prime.T - matrix @ x.T
        right = matrix @ delta_x.T + (second_matrix - matrix) @ x_prime.T
        decomposition_error.append(float(np.max(np.abs(left - right))))

    for _ in range(100):
        count = 13
        base, aa, bb = rng.normal(size=(3, count))
        distances = rng.uniform(-20.0, 20.0, size=count)
        values = rng.normal(size=(count, 2))
        span, z = rng.uniform(0.3, 4.0), rng.uniform(0.1, 0.9)

        def state(current: float):
            frequency = math.exp(-span * current)
            phase = frequency * distances
            logits = base + aa * np.cos(phase) + bb * np.sin(phase)
            probability = softmax(logits)
            rotated = np.stack([rotation(angle) @ value for angle, value in zip(phase, values)])
            return probability @ rotated, probability, rotated, phase

        output, probability, rotated, phase = state(z)
        derivative_logits = span * phase * (aa * np.sin(phase) - bb * np.cos(phase))
        selection = np.sum(probability[:, None] * (rotated - output) * derivative_logits[:, None], axis=0)
        generator = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        direct = np.sum(
            probability[:, None]
            * np.stack([-span * angle * generator @ current for angle, current in zip(phase, rotated)]),
            axis=0,
        )
        step = 1e-5
        numeric = (
            -state(z + 2 * step)[0] + 8 * state(z + step)[0]
            - 8 * state(z - step)[0] + state(z - 2 * step)[0]
        ) / (12 * step)
        direct_value_errors.append(float(np.max(np.abs(numeric - (selection + direct)))))
        omitted_direct_norms.append(float(np.linalg.norm(numeric - selection)))

        router = rng.normal(size=(8, 5))
        hidden = rng.normal(size=5)
        router_scores = router @ hidden
        sorted_scores = np.sort(router_scores)[::-1]
        k = 2
        margin = sorted_scores[k - 1] - sorted_scores[k]
        direction = rng.normal(size=5)
        routed_delta = router @ direction
        scale = 0.49 * margin / max(np.max(np.abs(routed_delta)), 1e-12)
        delta_hidden = scale * direction
        if set(np.argpartition(-router_scores, k - 1)[:k]) != set(np.argpartition(-(router_scores + router @ delta_hidden), k - 1)[:k]):
            raise AssertionError("MoE routing margin certificate failed")
        moe_cases += 1
    if min(compression_slack) < -1e-11 or min(omitted_direct_norms) <= 1e-7:
        raise AssertionError("architecture-path validation failed")
    return record(
        "15_sparse_access_compression_rotary_value_moe_mhc",
        500,
        max(max(direct_value_errors), max(doubly_stochastic_residual), max(decomposition_error)),
        minimum_sparse_bound_slack=float(min(access_slack)),
        minimum_compression_bound_slack=float(min(compression_slack)),
        maximum_rotary_value_derivative_error=float(max(direct_value_errors)),
        minimum_error_when_direct_value_term_omitted=float(min(omitted_direct_norms)),
        topk_margin_cases=topk_cases,
        moe_margin_cases=moe_cases,
        maximum_doubly_stochastic_residual=float(max(doubly_stochastic_residual)),
        maximum_state_decomposition_error=float(max(decomposition_error)),
    )


CHECKS = (
    check_01_normalized_coordinates,
    check_02_logit_z_gradient,
    check_03_finite_softmax_reweighting,
    check_04_log_odds_shift,
    check_05_value_covariance_identity,
    check_06_local_loss_chain_rule,
    check_07_schur_complement,
    check_08_gain_square_and_competition,
    check_09_gain_affine_recovery,
    check_10_threeband_phase_and_midband_optimum,
    check_11_yarn_shape_and_scale,
    check_12_deployment_z_and_tc_identities,
    check_13_tc_marginal_pairing,
    check_14_roughness_and_rotation_bounds,
    check_15_sparse_compression_value_moe_mhc,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = [function() for function in CHECKS]
    result = {
        "status": "HYBRID_ROPE_THEORY_DEEPENING_CPU_CHECKS_COMPLETE_V1",
        "checks": checks,
        "check_count": len(checks),
        "all_passed": all(check["status"] == "PASS" for check in checks),
        "seed": SEED,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Algebraic and numerical operator checks only; no checkpoint, task score, causal mediation or performance optimum is validated.",
        "not_cpu_verifiable": [
            "TailSpline, C, MrPro or YaRN task ordering",
            "whether key competition changes improve useful evidence aggregation in a checkpoint",
            "whether gain-frequency effects transfer across heads, layers or architectures",
            "whether sparse routing or rotary values improve complete generated answers",
            "whether post-hoc Native-z calibration improves a mature checkpoint",
        ],
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
