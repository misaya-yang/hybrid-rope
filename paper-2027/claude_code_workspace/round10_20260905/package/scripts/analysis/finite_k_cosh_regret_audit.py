#!/usr/bin/env python3
"""Audit finite-K EVQ-Cosh histogram regret for the stated convex surrogate.

This is a CPU-only numerical certificate for one mathematical realization:
the equal-mass quantile histogram of the continuous Cosh minimizer.  It checks
that the surrogate regret scales as K^-2 and that

    K^2 (J[rho_K] - J[rho_*])
      -> alpha / 24 * (tau^2 - tau * tanh(tau)).

The audit does not evaluate r2, language-model loss, extrapolation, or any
frequency-table selector.  Other discretizations require separate analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PAIRS = (8, 16, 32, 64, 128)
DEFAULT_TAUS = (0.5, 1.0, 2.0, 4.0)


def cosh_quantile(mass: float, tau: float) -> float:
    """Inverse CDF of rho(phi)=tau*cosh(tau*(1-phi))/sinh(tau)."""
    if not 0.0 <= float(mass) <= 1.0:
        raise ValueError("mass must lie in [0, 1]")
    tau = float(tau)
    if tau < 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and nonnegative")
    if tau == 0.0:
        return float(mass)
    return 1.0 - math.asinh((1.0 - float(mass)) * math.sinh(tau)) / tau


def continuous_components(tau: float) -> tuple[float, float]:
    """Return (||rho_*||_2^2, ||V rho_*||_2^2) analytically."""
    tau = float(tau)
    if tau < 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and nonnegative")
    if tau == 0.0:
        return 1.0, 1.0 / 3.0
    denominator = math.sinh(tau) ** 2
    l2 = (tau * tau / denominator) * (
        0.5 + math.sinh(2.0 * tau) / (4.0 * tau)
    )
    volterra_l2 = (
        math.sinh(2.0 * tau) / (4.0 * tau) - 0.5
    ) / denominator
    return l2, volterra_l2


def histogram_components(pairs: int, tau: float) -> tuple[float, float]:
    """Return exact quadratic components for the equal-mass histogram."""
    pairs = int(pairs)
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    boundaries = [cosh_quantile(index / pairs, tau) for index in range(pairs + 1)]
    if boundaries[0] != 0.0 or boundaries[-1] != 1.0:
        raise AssertionError("quantile support drift")

    density_l2 = 0.0
    volterra_l2 = 0.0
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        width = right - left
        if not width > 0.0:
            raise AssertionError("quantile cells must be strictly ordered")
        density_l2 += 1.0 / (pairs * pairs * width)

        # On this cell V rho_K falls linearly from remaining/pairs to
        # (remaining-1)/pairs, so its squared integral is exact.
        remaining = pairs - index
        volterra_l2 += (
            width
            / (pairs * pairs)
            * (remaining * remaining - remaining + 1.0 / 3.0)
        )
    return density_l2, volterra_l2


def surrogate_value(
    density_l2: float,
    volterra_l2: float,
    *,
    alpha: float,
    tau: float,
) -> float:
    """Evaluate J with beta=alpha*tau^2, the Cosh-minimizer relation."""
    return 0.5 * float(alpha) * (
        float(density_l2) + float(tau) ** 2 * float(volterra_l2)
    )


def asymptotic_constant(alpha: float, tau: float) -> float:
    tau = float(tau)
    return float(alpha) / 24.0 * (tau * tau - tau * math.tanh(tau))


def audit_tau(*, tau: float, pairs: Iterable[int], alpha: float) -> dict[str, Any]:
    pair_grid = tuple(sorted({int(value) for value in pairs}))
    if len(pair_grid) < 2 or pair_grid[0] <= 0:
        raise ValueError("pairs must contain at least two positive values")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and positive")

    continuous = continuous_components(tau)
    optimum = surrogate_value(*continuous, alpha=alpha, tau=tau)
    target = asymptotic_constant(alpha, tau)
    rows: list[dict[str, float | int | None]] = []
    for count in pair_grid:
        histogram = histogram_components(count, tau)
        value = surrogate_value(*histogram, alpha=alpha, tau=tau)
        regret = value - optimum
        if regret < -1e-13:
            raise AssertionError("histogram beat the continuous minimizer numerically")
        regret = max(regret, 0.0)
        scaled = count * count * regret
        relative_error = None if target == 0.0 else scaled / target - 1.0
        rows.append(
            {
                "pairs": count,
                "surrogate_value": value,
                "regret": regret,
                "scaled_regret_k2": scaled,
                "relative_constant_error": relative_error,
            }
        )

    previous, final = rows[-2], rows[-1]
    if float(previous["regret"]) == 0.0 or float(final["regret"]) == 0.0:
        slope = None
    else:
        slope = math.log(float(final["regret"]) / float(previous["regret"])) / math.log(
            int(final["pairs"]) / int(previous["pairs"])
        )
    return {
        "tau": float(tau),
        "alpha": float(alpha),
        "beta": float(alpha) * float(tau) ** 2,
        "continuous_surrogate_value": optimum,
        "asymptotic_k2_constant": target,
        "tail_log_log_slope": slope,
        "rows": rows,
    }


def build_report(
    *,
    taus: Iterable[float],
    pairs: Iterable[int],
    alpha: float,
    max_relative_error: float,
    max_slope_error: float,
) -> dict[str, Any]:
    tau_grid = tuple(float(value) for value in taus)
    pair_grid = tuple(sorted({int(value) for value in pairs}))
    if not tau_grid:
        raise ValueError("taus must be nonempty")
    audits = [audit_tau(tau=tau, pairs=pair_grid, alpha=alpha) for tau in tau_grid]
    failures: list[str] = []
    for audit in audits:
        final = audit["rows"][-1]
        relative = final["relative_constant_error"]
        slope = audit["tail_log_log_slope"]
        if relative is not None and abs(float(relative)) > float(max_relative_error):
            failures.append(f"tau={audit['tau']}: asymptotic constant tolerance failed")
        if slope is not None and abs(float(slope) + 2.0) > float(max_slope_error):
            failures.append(f"tau={audit['tau']}: K^-2 slope tolerance failed")
    return {
        "status": "PASS" if not failures else "FAIL",
        "scope": (
            "equal-mass quantile histogram; Cosh convex-surrogate value only; "
            "not r2, LM loss, extrapolation, or a table selector"
        ),
        "pair_grid": list(pair_grid),
        "tau_grid": list(tau_grid),
        "max_relative_error": float(max_relative_error),
        "max_slope_error": float(max_slope_error),
        "audits": audits,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-relative-error", type=float, default=0.01)
    parser.add_argument("--max-slope-error", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        taus=args.taus,
        pairs=args.pairs,
        alpha=args.alpha,
        max_relative_error=args.max_relative_error,
        max_slope_error=args.max_slope_error,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
