#!/usr/bin/env python3
"""Training-free finite-K EVQ-Cosh tau selector and Phase16 audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def evq_phi(num_pairs: int, tau: float, *, midpoint: bool = True) -> np.ndarray:
    """Actual selected-grid Cosh quantiles, with a stable large-tau path."""
    offset = 0.5 if midpoint else 0.0
    u = (np.arange(num_pairs, dtype=np.float64) + offset) / num_pairs
    if tau == 0.0:
        return u
    if math.isinf(tau):
        return np.zeros_like(u)
    if tau < 20.0:
        return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
    log_sinh = tau - math.log(2.0) + math.log1p(-math.exp(-2.0 * tau))
    log_x = np.log1p(-u) + log_sinh
    asinh_x = log_x + np.log1p(np.sqrt(1.0 + np.exp(-2.0 * log_x)))
    return 1.0 - asinh_x / tau


def distance_features(
    base: float,
    num_pairs: int,
    tau: float,
    max_delta: int,
    *,
    midpoint: bool = True,
) -> np.ndarray:
    """z_tau(delta)=[cos(w_k delta),sin(w_k delta)]/sqrt(K)."""
    inv_freq = np.power(
        float(base), -evq_phi(num_pairs, tau, midpoint=midpoint)
    )
    phase = np.outer(np.arange(max_delta + 1, dtype=np.float64), inv_freq)
    return np.concatenate((np.cos(phase), np.sin(phase)), axis=1) / math.sqrt(
        num_pairs
    )


def pair_distance_distribution(length: int, max_delta: int) -> np.ndarray:
    """Distance law for a uniformly sampled distinct causal token pair."""
    if length < 2 or length - 1 > max_delta:
        raise ValueError("length must be in [2, max_delta + 1]")
    probability = np.zeros(max_delta + 1, dtype=np.float64)
    delta = np.arange(1, length, dtype=np.float64)
    probability[1:length] = 2.0 * (length - delta) / (
        length * (length - 1)
    )
    return probability


def target_distribution(
    lengths: list[int], weights: list[float], max_delta: int
) -> np.ndarray:
    if len(lengths) != len(weights) or not lengths:
        raise ValueError("target lengths and weights must have equal nonzero size")
    weight = np.asarray(weights, dtype=np.float64)
    if np.any(weight < 0.0) or not float(weight.sum()) > 0.0:
        raise ValueError("target weights must be nonnegative with positive sum")
    weight /= weight.sum()
    return sum(
        w * pair_distance_distribution(length, max_delta)
        for length, w in zip(lengths, weight, strict=True)
    )


def _feature_covariance(features: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return (features * probability[:, None]).T @ features


def _distinct_collision(
    left_cov: np.ndarray,
    right_cov: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
) -> float:
    same_delta = float(left_probability @ right_probability)
    return (
        float(np.sum(left_cov * right_cov)) - same_delta
    ) / (1.0 - same_delta)


def collision_risk(
    *,
    base: float,
    num_pairs: int,
    train_length: int,
    target_lengths: list[int],
    target_weights: list[float],
    tau: float,
    midpoint: bool = True,
) -> dict[str, float]:
    """Worst expected squared Gram collision across train/target domains."""
    max_delta = max(target_lengths) - 1
    train = pair_distance_distribution(train_length, max_delta)
    target = target_distribution(target_lengths, target_weights, max_delta)
    features = distance_features(
        base, num_pairs, tau, max_delta, midpoint=midpoint
    )
    train_cov = _feature_covariance(features, train)
    target_cov = _feature_covariance(features, target)
    parts = {
        "train": _distinct_collision(train_cov, train_cov, train, train),
        "target": _distinct_collision(target_cov, target_cov, target, target),
        "cross": _distinct_collision(train_cov, target_cov, train, target),
    }
    return {"risk": max(parts.values()), **parts}


def select_tau(
    *,
    base: float,
    num_pairs: int,
    train_length: int,
    target_lengths: list[int],
    target_weights: list[float],
    midpoint: bool = True,
    coarse_points: int = 257,
) -> dict[str, float]:
    """Deterministic global grid on q=tau/(1+tau), then golden refinement."""
    if coarse_points < 17:
        raise ValueError("coarse_points must be at least 17")

    def objective(q: float) -> float:
        tau = math.inf if q >= 1.0 else q / (1.0 - q)
        return collision_risk(
            base=base,
            num_pairs=num_pairs,
            train_length=train_length,
            target_lengths=target_lengths,
            target_weights=target_weights,
            tau=tau,
            midpoint=midpoint,
        )["risk"]

    grid = np.linspace(0.0, 1.0, coarse_points)
    values = np.asarray([objective(float(q)) for q in grid])
    best = int(values.argmin())
    lo = float(grid[max(0, best - 1)])
    hi = float(grid[min(coarse_points - 1, best + 1)])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(48):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    q = x1 if f1 <= f2 else x2
    tau = q / (1.0 - q)
    return {
        "tau": tau,
        **collision_risk(
            base=base,
            num_pairs=num_pairs,
            train_length=train_length,
            target_lengths=target_lengths,
            target_weights=target_weights,
            tau=tau,
            midpoint=midpoint,
        ),
    }


def _weighted_extrapolation_nll(row: dict[str, str]) -> float:
    length = int(row["seq_len"])
    ppl = {int(k): float(v) for k, v in json.loads(row["ppl_json"]).items()}
    ratios = (2, 4, 8)
    weights = [math.log2(ratio + 1) for ratio in ratios]
    return sum(
        weight * math.log(ppl[ratio * length])
        for ratio, weight in zip(ratios, weights, strict=True)
    ) / sum(weights)


def validate_phase16(manifest: Path, base: float) -> dict:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    configurations = sorted({row["config_id"] for row in rows})
    details = []
    for config in configurations:
        pilot = [
            row
            for row in rows
            if row["config_id"] == config and row["stage"] == "pilot"
        ]
        if len(pilot) != 5:
            raise ValueError(f"{config}: expected five pilot rows, found {len(pilot)}")
        train_length = int(pilot[0]["seq_len"])
        num_pairs = int(pilot[0]["head_dim"]) // 2
        ratios = (2, 4, 8)
        target_lengths = [ratio * train_length for ratio in ratios]
        target_weights = [math.log2(ratio + 1) for ratio in ratios]
        selected = select_tau(
            base=base,
            num_pairs=num_pairs,
            train_length=train_length,
            target_lengths=target_lengths,
            target_weights=target_weights,
            midpoint=True,
        )
        observed = sorted(
            (float(row["tau"]), _weighted_extrapolation_nll(row)) for row in pilot
        )
        oracle_tau, oracle_nll = min(observed, key=lambda item: item[1])
        proxy_tau, proxy_nll = min(
            observed, key=lambda item: (abs(item[0] - selected["tau"]), item[0])
        )
        formula_tau = float(pilot[0]["theory_tau"])
        used_formula_tau, formula_nll = min(
            observed, key=lambda item: (abs(item[0] - formula_tau), item[0])
        )
        _, geo_nll = next(item for item in observed if item[0] == 0.0)
        ordered = [tau for tau, _ in sorted(observed, key=lambda item: item[1])]
        details.append(
            {
                "config_id": config,
                "base": base,
                "num_frequency_pairs": num_pairs,
                "train_length": train_length,
                "target_lengths": target_lengths,
                "target_weights": target_weights,
                "selector_tau": selected["tau"],
                "selector_risk": selected["risk"],
                "nearest_observed_tau": proxy_tau,
                "formula_tau": used_formula_tau,
                "oracle_tau": oracle_tau,
                "selector_rank": ordered.index(proxy_tau) + 1,
                "formula_rank": ordered.index(used_formula_tau) + 1,
                "selector_nll_regret": proxy_nll - oracle_nll,
                "formula_nll_regret": formula_nll - oracle_nll,
                "fixed_tau0_nll_regret": geo_nll - oracle_nll,
            }
        )

    def summarize(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in details])
        return {
            "mean_nll_regret": float(values.mean()),
            "mean_relative_ppl_regret": float(np.mean(np.exp(values) - 1.0)),
            "median_relative_ppl_regret": float(np.median(np.exp(values) - 1.0)),
        }

    selector = summarize("selector_nll_regret")
    formula = summarize("formula_nll_regret")
    fixed = summarize("fixed_tau0_nll_regret")
    better = sum(
        row["selector_nll_regret"] < row["formula_nll_regret"] for row in details
    )
    top_two = sum(row["selector_rank"] <= 2 for row in details)
    formula_top_two = sum(row["formula_rank"] <= 2 for row in details)
    gate = (
        selector["mean_relative_ppl_regret"]
        < formula["mean_relative_ppl_regret"]
        and top_two >= formula_top_two
    )
    return {
        "method": "finite_K_worst_domain_gram_collision",
        "base": base,
        "target_semantics": {
            "grid": "midpoint",
            "length_ratios": [2, 4, 8],
            "length_weights": "log2(ratio + 1), matching Phase16 NLL metric",
            "within_length_distance_law": "uniform distinct causal token pairs",
            "train_target_aggregation": "maximum of train, target, cross collision",
        },
        "configuration_holdout": (
            "No PPL-fitted parameters: every configuration is evaluated without "
            "using any other configuration's outcomes."
        ),
        "selector": selector,
        "old_formula": formula,
        "fixed_tau0": fixed,
        "selector_better_than_formula_configs": better,
        "selector_top2_configs": top_two,
        "formula_top2_configs": formula_top_two,
        "historical_gate_pass": gate,
        "prospective_training_authorized": gate,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--base", type=float, required=True)
    select.add_argument("--pairs", type=int, required=True)
    select.add_argument("--train-length", type=int, required=True)
    select.add_argument("--target-lengths", type=int, nargs="+", required=True)
    select.add_argument("--target-weights", type=float, nargs="+", required=True)
    select.add_argument(
        "--grid", choices=("midpoint", "endpoint"), default="midpoint"
    )
    validate = subparsers.add_parser("validate-phase16")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--base", type=float, default=500_000.0)
    validate.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "select":
        result = select_tau(
            base=args.base,
            num_pairs=args.pairs,
            train_length=args.train_length,
            target_lengths=args.target_lengths,
            target_weights=args.target_weights,
            midpoint=args.grid == "midpoint",
        )
    else:
        result = validate_phase16(args.manifest, args.base)
    text = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
