#!/usr/bin/env python3
"""CPU-only checks for demand companding and conditional ``C_app``.

This module is deliberately a numerical companion, not a new paper claim.  It
keeps three objects separate:

* ``m`` is an R0 demand profile on a one-dimensional coordinate ``x`` (it is
  normalized when entering a probability-density functional);
* ``rho`` is a normalized channel point density; and
* a protected-frequency calculation is a finite, discretized optimization.

For the high-rate squared-error surrogate,

    D[rho] = (12 K**2)**-1 integral m(x) / rho(x)**2 dx,

the Euler solution is ``rho = m**(1/3) / Z`` and
``D* = Z**3 / (12 K**2)``, where ``Z = integral m**(1/3)``.  ``H_m13``
below is this normalizer.  On a unit coordinate, the requested compression
headroom is ``H(m) = 1 - Z**3``; the differential entropy is reported
separately as ``H_differential`` so that the two cannot be conflated.

The lambda family mixes the R0 profile with the unit-coordinate uniform prior,
``m_lambda = (1-lambda) m + lambda / |I|``.  It is a diagnostic family, not a
claim that the mixture is an optimum for another objective.

The pointwise quartic helper diagnoses the alpha-regularized equation

    alpha*rho**4 + nu*rho**3 - 2*(m + epsilon) = 0.

Its left side is monotone for non-negative ``alpha`` and ``nu``; fixed-ν roots
are found by bracketed bisection.  The density wrapper outer-solves ν only on
that guaranteed branch and otherwise returns ``BLOCKED_DIAGNOSTIC``.  A
separate scalar quartic helper solves the
stationary equation for

    F(t) = alpha*t**4 + beta*t**2 - gain*t,
    F'(t) = 4*alpha*t**3 + 2*beta*t - gain.

The derivative is monotone for non-negative ``alpha`` and ``beta``; its root is
also found by bracketed bisection rather than an unstable polynomial formula.
Neither helper promotes the stiffness diagnostic ``p ~= 0.85`` into the
``C_app`` derivation.

The protected-frequency routine uses a Voronoi/cell discretization of the
continuous ``C_app`` functional and SciPy SLSQP when available.  Its result is
always labelled ``NUMERICAL_SMALL_SCALE_ONLY`` (or a blocked status), with no
global-optimum or convexity claim.

Examples::

    python scripts/analysis/demand_companding_numeric.py --self-test
    python scripts/analysis/demand_companding_numeric.py --synthetic both \
        --output-json /tmp/companding.json --output-csv /tmp/companding.csv \
        --output-plot /tmp/companding.png
    python scripts/analysis/demand_companding_numeric.py --r0-json R0.json \
        --output-json /tmp/r0-companding.json

No training, GPU, repository mutation, or paper compilation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Return a scalar trapezoidal integral on NumPy 1.x or 2.x."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy 1.x
        trapezoid = np.trapz
    integral = trapezoid(y, x)
    return float(integral)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid with the same length as ``x``."""

    dx = np.diff(x)
    increments = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(increments)))


def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with >=2 entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_grid(x: ArrayLike, density: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and, if necessary, sort a strictly increasing integration grid."""

    grid = _as_float_array(x, "x")
    if np.any(np.diff(grid) <= 0):
        order = np.argsort(grid, kind="mergesort")
        grid = grid[order]
        if np.any(np.diff(grid) <= 0):
            raise ValueError("x must contain distinct values")
        if density is None:
            return grid, None
        values = _as_float_array(density, "density")
        if values.size != grid.size:
            raise ValueError("x and density must have the same length")
        return grid, values[order]
    if density is None:
        return grid, None
    values = _as_float_array(density, "density")
    if values.size != grid.size:
        raise ValueError("x and density must have the same length")
    return grid, values


def normalize_density(x: ArrayLike, density: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, density / integral(density))``.

    The input is required to be non-negative.  Keeping normalization in one
    function prevents an arbitrary R0 scale from changing ``D*`` or lambda
    comparisons.
    """

    grid, values = validate_grid(x, density)
    if np.any(values < 0):
        raise ValueError("density must be non-negative")
    mass = _trapz(values, grid)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("density must have positive finite integral")
    return grid, values / mass


def uniform_density(x: ArrayLike) -> np.ndarray:
    """Uniform probability density on the interval covered by ``x``."""

    grid = _as_float_array(x, "x")
    width = float(grid[-1] - grid[0])
    if width <= 0:
        raise ValueError("x must span a positive interval")
    return np.full_like(grid, 1.0 / width)


def mix_demand(x: ArrayLike, demand: ArrayLike, lam: float) -> np.ndarray:
    """Return the raw R0 mixture ``(1-lam)m + lam/|I|``.

    R0 attention-demand profiles are allowed to have arbitrary positive scale;
    that scale must be preserved before lambda mixing because the uniform
    prior is the unit density on the normalized ``[0,1]`` coordinate.  The
    downstream companding functions normalize the mixture when treating it as
    a probability density.  ``lam=0`` is pure demand and ``lam=1`` is uniform.
    """

    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    grid, m = validate_grid(x, demand)
    if np.any(m < 0):
        raise ValueError("demand must be non-negative")
    width = float(grid[-1] - grid[0])
    if width <= 0:
        raise ValueError("x must span a positive interval")
    return (1.0 - float(lam)) * m + float(lam) / width


def lambda_mix(x: ArrayLike, demand: ArrayLike, lam: float) -> np.ndarray:
    """Alias for the raw R0 lambda mixture."""

    return mix_demand(x, demand, lam)


def high_rate_density(x: ArrayLike, demand: ArrayLike) -> dict[str, Any]:
    """Compute ``rho proportional to m**(1/3)`` and its high-rate constants."""

    grid, m = normalize_density(x, demand)
    z = _trapz(np.cbrt(m), grid)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("m**(1/3) must have positive finite integral")
    rho = np.cbrt(m) / z
    phi, width = _unit_coordinate(grid)
    m_phi = m * width
    z_unit = _trapz(np.cbrt(m_phi), phi)
    # Jensen gives 0 <= 1-z_unit**3 < 1 for a probability density on a unit
    # interval.  Clip only round-off at the uniform endpoint.
    headroom = float(np.clip(1.0 - z_unit**3, 0.0, 1.0))
    return {
        "x": grid,
        "m": m,
        "rho": rho,
        "H_m13": float(z),
        "H_unit_m13": float(z_unit),
        "H": headroom,
        "H_differential": differential_entropy(grid, m),
        "H_rho": differential_entropy(grid, rho),
    }


def high_rate_distortion(x: ArrayLike, demand: ArrayLike, K: int) -> float:
    """Return ``D* = (integral m**(1/3))**3 / (12 K**2)``."""

    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer")
    result = high_rate_density(x, demand)
    return float(result["H_m13"] ** 3 / (12.0 * int(K) ** 2))


def rho_m13(x: ArrayLike, demand: ArrayLike) -> np.ndarray:
    """Convenience API for the normalized ``m**(1/3)`` point density."""

    return np.asarray(high_rate_density(x, demand)["rho"], dtype=float)


def D_star(x: ArrayLike, demand: ArrayLike, K: int | None = None) -> float:
    """Return dimensionless ``Z**3`` or full high-rate ``D*`` when ``K`` is set.

    The paper-independent shape quantity is ``Z**3``.  Passing ``K`` includes
    the quantization factor ``1/(12 K**2)``.
    """

    result = high_rate_density(x, demand)
    shape = float(result["H_m13"] ** 3)
    if K is None:
        return shape
    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer")
    return float(shape / (12.0 * int(K) ** 2))


def differential_entropy(x: ArrayLike, density: ArrayLike) -> float:
    """Return differential entropy ``-integral p log p``.

    This quantity is distinct from ``H_m13``.  Zero-density grid values are
    assigned an integrand of zero using the continuous ``p log p`` limit.
    """

    grid, p = normalize_density(x, density)
    integrand = np.zeros_like(p)
    positive = p > 0
    integrand[positive] = p[positive] * np.log(p[positive])
    return float(-_trapz(integrand, grid))


def compression_headroom(x: ArrayLike, demand: ArrayLike) -> float:
    """Return ``H(m)=1-(integral m**(1/3))**3`` on the unit coordinate.

    The coordinate is normalized to ``phi in [0,1]`` before integration.  This
    is the dimensionless headroom used for the Jensen bound; it is not Shannon
    or differential entropy.
    """

    result = high_rate_density(x, demand)
    return float(result["H"])


def H(x: ArrayLike, demand: ArrayLike) -> float:
    """Convenience API for the dimensionless compression headroom ``H(m)``."""

    return compression_headroom(x, demand)


def quantile_from_density(
    x: ArrayLike,
    density: ArrayLike,
    K: int,
    *,
    endpoint: bool = True,
) -> np.ndarray:
    """Invert a trapezoidal CDF using endpoint or midpoint quantiles.

    Endpoint mode uses ``u_k=k/(K-1)`` and therefore returns the exact domain
    endpoints.  Midpoint mode uses ``u_k=(k+1/2)/K`` and never forces endpoint
    frequencies.  Flat CDF segments are retained as repeated quantiles; this
    makes zero-demand/collision behavior visible rather than adding a hidden
    positive floor.
    """

    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer")
    if endpoint and K < 2:
        raise ValueError("endpoint quantiles require K >= 2")
    grid, p = normalize_density(x, density)
    cdf = _cumtrapz(p, grid)
    total = float(cdf[-1])
    if total <= 0 or not np.isfinite(total):
        raise ValueError("CDF has invalid total mass")
    cdf = cdf / total
    cdf = np.maximum.accumulate(cdf)
    if endpoint:
        u = np.linspace(0.0, 1.0, K)
    else:
        u = (np.arange(K, dtype=float) + 0.5) / K
    q = np.interp(u, cdf, grid)
    if endpoint:
        q[0] = grid[0]
        q[-1] = grid[-1]
    return np.maximum.accumulate(q)


def endpoint_quantiles(x: ArrayLike, density: ArrayLike, K: int) -> np.ndarray:
    """Alias for endpoint-inclusive inverse-CDF quantiles."""

    return quantile_from_density(x, density, K, endpoint=True)


def minimum_spacing(values: ArrayLike) -> float:
    """Return minimum adjacent spacing, including zero for collisions."""

    array = np.sort(_as_float_array(values, "values"))
    if array.size < 2:
        return float("inf")
    return float(np.min(np.diff(array)))


def min_spacing(values: ArrayLike) -> float:
    """Alias for the collision-sensitive minimum adjacent spacing."""

    return minimum_spacing(values)


def frequency_from_coordinate(x: ArrayLike, *, base: float = 5.0e5, theta_hi: float = 1.0) -> np.ndarray:
    """Map normalized RoPE-style coordinate ``phi`` to ``theta_hi*base**(-phi)``."""

    if base <= 1 or theta_hi <= 0:
        raise ValueError("base must be >1 and theta_hi must be positive")
    return float(theta_hi) * np.power(float(base), -np.asarray(x, dtype=float))


def coordinate_from_frequency(
    theta: ArrayLike,
    *,
    base: float = 5.0e5,
    theta_hi: float = 1.0,
) -> np.ndarray:
    """Map positive native frequencies back to normalized ``phi``."""

    if base <= 1 or theta_hi <= 0:
        raise ValueError("base must be >1 and theta_hi must be positive")
    values = _as_float_array(theta, "theta")
    if np.any(values <= 0):
        raise ValueError("native frequencies must be positive")
    phi = -np.log(values / float(theta_hi)) / math.log(float(base))
    if np.any(phi < -1.0e-10) or np.any(phi > 1.0 + 1.0e-10):
        raise ValueError("native frequencies lie outside the configured support")
    return np.clip(phi, 0.0, 1.0)


def phase_collision_score(
    theta: ArrayLike,
    distances: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    """Compute a transparent pairwise phase-coherence collision proxy.

    For each pair ``i<j`` this uses
    ``(sum_r w_r cos((theta_i-theta_j) r))**2`` and averages over pairs.  A
    distance kernel is required; callers should report unavailable rather than
    silently inventing one.  Lower values indicate less coherence under the
    supplied kernel, but this is not an LM-quality metric.
    """

    frequencies = np.sort(_as_float_array(theta, "theta"))
    r = _as_float_array(distances, "distances")
    if weights is None:
        w = np.full_like(r, 1.0 / r.size)
    else:
        w = _as_float_array(weights, "weights")
        if w.size != r.size:
            raise ValueError("distances and weights must have the same length")
        if np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        w = w / w.sum()
    if frequencies.size < 2:
        return 0.0
    values: list[float] = []
    for i in range(frequencies.size - 1):
        differences = frequencies[i + 1 :] - frequencies[i]
        coherence = np.cos(differences[:, None] * r[None, :]) @ w
        values.extend(np.square(coherence).tolist())
    return float(np.mean(values))


def coordinate_collision_score(values: ArrayLike, reference_scale: float | None = None) -> float:
    """Return a bounded spacing-only collision proxy.

    This is available even when an R0 file has no distance kernel.  It averages
    ``exp(-|x_i-x_j|/s)`` with ``s=1/(K-1)`` by default, so exact collisions
    score one and well-separated points score lower.  It is explicitly a
    coordinate diagnostic, not an attention or language-model metric.
    """

    points = np.sort(_as_float_array(values, "values"))
    if reference_scale is None:
        reference_scale = 1.0 / max(points.size - 1, 1)
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("reference_scale must be finite and positive")
    differences = points[:, None] - points[None, :]
    mask = np.triu(np.ones_like(differences, dtype=bool), k=1)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.exp(-np.abs(differences[mask]) / reference_scale)))


def _unit_coordinate(x: ArrayLike) -> tuple[np.ndarray, float]:
    grid = _as_float_array(x, "x")
    width = float(grid[-1] - grid[0])
    if width <= 0:
        raise ValueError("x must span a positive interval")
    return (grid - grid[0]) / width, width


def cosh_density(phi: ArrayLike, tau: float) -> np.ndarray:
    """Stable evaluation of the normalized Cosh density on ``[0, 1]``."""

    z = np.asarray(phi, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(z < 0) or np.any(z > 1):
        raise ValueError("phi must be finite and lie in [0, 1]")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if tau < 1.0e-4:
        t2 = tau * tau
        sinhc = 1.0 + t2 / 6.0 + t2 * t2 / 120.0
        return np.cosh(tau * (1.0 - z)) / sinhc
    if tau < 40.0:
        return tau * np.cosh(tau * (1.0 - z)) / np.sinh(tau)
    # Exponentially scaled form avoids overflow in sinh(tau).
    a = tau * (1.0 - z)
    numerator = np.exp(a - tau) * (1.0 + np.exp(-2.0 * a))
    denominator = 1.0 - np.exp(-2.0 * tau)
    return tau * numerator / denominator


def capp_components(
    x: ArrayLike,
    rho: ArrayLike,
    *,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Evaluate the two terms of ``C_app`` on a normalized coordinate.

    The input density is interpreted with respect to ``x`` and transformed to
    a density on ``phi=(x-x_min)/(x_max-x_min)`` before evaluating the paper's
    ``min(phi, psi)`` kernel.
    """

    if alpha < 0 or beta < 0 or not np.isfinite(alpha + beta):
        raise ValueError("alpha and beta must be finite and non-negative")
    grid, values = normalize_density(x, rho)
    phi, width = _unit_coordinate(grid)
    rho_phi = values * width
    cumulative = _cumtrapz(rho_phi, phi)
    tail = cumulative[-1] - cumulative
    # The density is normalized, so this is the integral from phi to one.
    tail = np.maximum(tail, 0.0)
    alpha_term = 0.5 * float(alpha) * _trapz(rho_phi * rho_phi, phi)
    beta_kernel = _trapz(tail * tail, phi)
    beta_term = 0.5 * float(beta) * beta_kernel
    return {
        "alpha_term": float(alpha_term),
        "beta_term": float(beta_term),
        "kernel_integral": float(beta_kernel),
        "value": float(alpha_term + beta_term),
    }


def capp_quartic_alpha_coefficient(alpha: float) -> float:
    """Return the small-``tau`` coefficient ``alpha/90`` of Cosh's alpha term."""

    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    # rho_tau = 1 + tau^2 eta + O(tau^4), integral eta^2 = 1/45;
    # alpha/2 * integral rho_tau^2 therefore has coefficient alpha/90.
    return float(alpha / 90.0)


def solve_pointwise_quartic(
    demand: ArrayLike,
    *,
    alpha: float,
    nu: float,
    epsilon: float = 0.0,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Diagnose fixed-``nu`` roots of ``alpha*rho**4 + nu*rho**3 = 2*(m + epsilon)``.

    ``m`` is normalized before solving.  For non-negative ``alpha`` and
    ``nu`` the left-hand side is continuous and monotone on ``rho >= 0``.
    The vectorized bracketed bisection remains stable when one coefficient is
    zero and avoids Ferrari/Cardano branch choices.  The returned ``rho_raw``
    is the direct pointwise root.  Because this function has no ``x`` grid,
    ``rho`` is only a sum-normalized convenience; use
    :func:`quartic_density_solution` for any normalization claim.
    """

    for name, value in (("alpha", alpha), ("nu", nu), ("epsilon", epsilon)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    m = np.asarray(demand, dtype=float)
    if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("demand must be a finite non-negative one-dimensional array")
    rhs = 2.0 * (m + float(epsilon))
    if alpha == 0.0 and nu == 0.0:
        if np.any(rhs > 0.0):
            raise ValueError("alpha=nu=0 has no finite positive quartic root")
        raw = np.zeros_like(rhs)
    elif alpha == 0.0:
        raw = np.cbrt(rhs / nu)
    elif nu == 0.0:
        raw = np.power(rhs / alpha, 0.25)
    else:
        # Both lower bounds are valid; their maximum is a scale-aware bracket
        # seed.  Doubling is unnecessary because this bound is conservative,
        # but a finite guard protects pathological inputs.
        hi = np.maximum(np.cbrt(rhs / nu), np.power(rhs / alpha, 0.25))
        hi = np.maximum(hi, 1.0)
        lo = np.zeros_like(rhs)

        def equation(value: np.ndarray) -> np.ndarray:
            return alpha * value**4 + nu * value**3 - rhs

        if np.any(~np.isfinite(hi)):
            raise ValueError("quartic bracket overflow")
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            residual = equation(mid)
            positive = residual >= 0.0
            hi = np.where(positive, mid, hi)
            lo = np.where(positive, lo, mid)
            if np.max(hi - lo) <= atol + rtol * np.maximum(1.0, mid).max():
                break
        raw = 0.5 * (lo + hi)
    residual = alpha * raw**4 + nu * raw**3 - rhs
    mass = float(np.sum(raw))
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(residual)):
        raise ValueError("quartic root produced non-finite values")
    return {
        "rho_raw": raw,
        "rho": raw / mass if mass > 0 else raw,
        "rhs": rhs,
        "residual": residual,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "status": "CONVERGED_MONOTONE_BISECTION" if alpha and nu else "DIRECT_MONOTONE_ROOT",
        "equation": "alpha*rho^4 + nu*rho^3 - 2*(m+epsilon) = 0",
    }


def quartic_density_solution(
    x: ArrayLike,
    demand: ArrayLike,
    *,
    alpha: float,
    nu: float | None = None,
    epsilon: float = 0.0,
    mass_atol: float = 2.0e-10,
    mass_rtol: float = 2.0e-10,
    max_outer_iter: int = 100,
) -> dict[str, Any]:
    """Solve the quartic only when its normalization is actually satisfied.

    With ``nu=None`` this searches the guaranteed monotone branch ``nu>=0``
    for an outer multiplier satisfying ``integral rho = 1``.  If that branch
    cannot reach unit mass, the required multiplier is negative; its local
    quartic can then have multiple positive roots, so this function returns
    ``BLOCKED_DIAGNOSTIC`` instead of selecting an arbitrary branch.

    With an explicit ``nu``, the pointwise roots are retained only when their
    numerical integral is already one.  No post-hoc normalization is applied,
    because that would invalidate the stationarity equation.
    """

    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    grid, m = normalize_density(x, demand)

    def evaluate(multiplier: float) -> tuple[dict[str, Any], float]:
        roots = solve_pointwise_quartic(m, alpha=alpha, nu=multiplier, epsilon=epsilon)
        raw = np.asarray(roots["rho_raw"], dtype=float)
        return roots, float(_trapz(raw, grid))

    def blocked(reason: str, multiplier: float | None, roots: dict[str, Any] | None, mass: float | None) -> dict[str, Any]:
        raw = None if roots is None else np.asarray(roots["rho_raw"], dtype=float)
        residual = None if roots is None else float(roots["max_abs_residual"])
        return {
            "status": "BLOCKED_DIAGNOSTIC",
            "reason": reason,
            "equation": "alpha*rho^4 + nu*rho^3 - 2*(m+epsilon) = 0",
            "x": grid,
            "m": m,
            "nu": multiplier,
            "rho_raw": raw,
            "rho": None,
            "rho_integral": mass,
            "max_abs_residual": residual,
            "global_solution_claim": False,
        }

    if nu is not None:
        if not np.isfinite(nu) or nu < 0:
            return blocked("negative_nu_branch_not_supported", float(nu) if np.isfinite(nu) else None, None, None)
        roots, mass = evaluate(float(nu))
        if not np.isfinite(mass) or mass <= 0:
            return blocked("nonpositive_or_nonfinite_mass", float(nu), roots, mass)
        tolerance = mass_atol + mass_rtol
        if abs(mass - 1.0) > tolerance:
            return blocked("fixed_nu_does_not_satisfy_integral_constraint", float(nu), roots, mass)
        raw = np.asarray(roots["rho_raw"], dtype=float)
        return {
            **roots,
            "status": "CONVERGED_FIXED_NU_NORMALIZED",
            "x": grid,
            "m": m,
            "nu": float(nu),
            "rho": raw,
            "rho_integral": mass,
            "global_solution_claim": False,
        }

    # Search the only branch on which pointwise roots are single-valued and
    # mass is monotone.  At nu=0, alpha>0 has a finite maximum mass; if it is
    # already below one, a negative multiplier would be required and we stop.
    if alpha > 0:
        roots_zero, mass_zero = evaluate(0.0)
        if mass_zero < 1.0 - (mass_atol + mass_rtol):
            return blocked("required_nu_negative_branch_not_supported", 0.0, roots_zero, mass_zero)
        if abs(mass_zero - 1.0) <= mass_atol + mass_rtol:
            raw = np.asarray(roots_zero["rho_raw"], dtype=float)
            return {
                **roots_zero,
                "status": "CONVERGED_OUTER_NU",
                "x": grid,
                "m": m,
                "nu": 0.0,
                "rho": raw,
                "rho_integral": mass_zero,
                "global_solution_claim": False,
            }

    lo = 0.0
    hi = 1.0
    roots_hi, mass_hi = evaluate(hi)
    while mass_hi > 1.0 and hi < np.finfo(float).max / 4.0:
        hi *= 2.0
        roots_hi, mass_hi = evaluate(hi)
    if mass_hi > 1.0:
        return blocked("outer_nu_bracket_overflow", hi, roots_hi, mass_hi)
    roots_mid: dict[str, Any] = roots_hi
    mass_mid = mass_hi
    for _ in range(int(max_outer_iter)):
        mid = 0.5 * (lo + hi)
        roots_mid, mass_mid = evaluate(mid)
        if abs(mass_mid - 1.0) <= mass_atol + mass_rtol:
            lo = hi = mid
            break
        if mass_mid > 1.0:
            lo = mid
        else:
            hi = mid
    multiplier = 0.5 * (lo + hi)
    roots_final, mass_final = evaluate(multiplier)
    if abs(mass_final - 1.0) > mass_atol + mass_rtol:
        return blocked("outer_nu_mass_tolerance_not_reached", multiplier, roots_final, mass_final)
    raw = np.asarray(roots_final["rho_raw"], dtype=float)
    return {
        **roots_final,
        "status": "CONVERGED_OUTER_NU",
        "x": grid,
        "m": m,
        "nu": float(multiplier),
        "rho": raw,
        "rho_integral": mass_final,
        "global_solution_claim": False,
    }


def quartic_root(
    demand: ArrayLike,
    *,
    alpha: float,
    nu: float,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Return only the pointwise monotone quartic roots."""

    return np.asarray(
        solve_pointwise_quartic(
            demand,
            alpha=alpha,
            nu=nu,
            epsilon=epsilon,
        )["rho_raw"],
        dtype=float,
    )


def _quartic_derivative(t: float, alpha: float, beta: float, gain: float) -> float:
    return 4.0 * alpha * t * t * t + 2.0 * beta * t - gain


def solve_quartic_balance(
    alpha: float,
    beta: float,
    gain: float,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Solve the monotone stationary equation of a quartic-plus-quadratic.

    ``alpha`` and ``beta`` are non-negative.  ``gain`` is the coefficient of
    the linear utility term.  If both regularizers vanish and gain is positive
    there is no finite stationary point; this is returned explicitly.
    """

    for name, value in (("alpha", alpha), ("beta", beta), ("gain", gain)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if alpha == 0 and beta == 0:
        if gain == 0:
            return {"status": "ROOT_AT_ZERO", "root": 0.0, "residual": 0.0, "objective": 0.0}
        return {
            "status": "NO_FINITE_ROOT",
            "root": None,
            "residual": None,
            "objective": None,
        }
    if gain == 0:
        return {"status": "ROOT_AT_ZERO", "root": 0.0, "residual": 0.0, "objective": 0.0}

    # A scale-aware upper bracket, followed by doubling if necessary.
    if beta > 0:
        scale = gain / (2.0 * beta)
    else:
        scale = (gain / (4.0 * alpha)) ** (1.0 / 3.0)
    hi = max(1.0, 2.0 * scale)
    f_hi = _quartic_derivative(hi, alpha, beta, gain)
    while f_hi < 0.0 and hi < np.finfo(float).max / 4.0:
        hi *= 2.0
        f_hi = _quartic_derivative(hi, alpha, beta, gain)
    if f_hi < 0.0:
        return {"status": "BRACKET_OVERFLOW", "root": None, "residual": None, "objective": None}

    lo = 0.0
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        f_mid = _quartic_derivative(mid, alpha, beta, gain)
        if abs(f_mid) <= atol + rtol * max(1.0, gain):
            lo = hi = mid
            break
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= atol + rtol * max(1.0, mid):
            break
    root = 0.5 * (lo + hi)
    residual = _quartic_derivative(root, alpha, beta, gain)
    objective = alpha * root**4 + beta * root**2 - gain * root
    return {
        "status": "CONVERGED",
        "root": float(root),
        "residual": float(residual),
        "objective": float(objective),
        "derivative_monotone_on_domain": True,
        "iterations_max": int(max_iter),
    }


def capp_discrete_objective(values: ArrayLike, *, alpha: float, beta: float) -> float:
    """Evaluate the documented Voronoi/cell discretization of ``C_app``."""

    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    points = np.sort(_as_float_array(values, "values"))
    if points[0] < 0 or points[-1] > 1 or np.any(np.diff(points) <= 0):
        raise ValueError("values must be strictly increasing in [0, 1]")
    edges = np.concatenate(([0.0], 0.5 * (points[:-1] + points[1:]), [1.0]))
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("Voronoi cells must have positive width")
    n = points.size
    rho = 1.0 / (n * widths)
    alpha_term = 0.5 * alpha * float(np.sum(rho * rho * widths))
    masses = rho * widths
    kernel = np.minimum(points[:, None], points[None, :])
    beta_term = 0.5 * beta * float(masses @ kernel @ masses)
    return float(alpha_term + beta_term)


def _gap_counts(protected: np.ndarray, free_count: int) -> list[int]:
    anchors = np.concatenate(([0.0], protected, [1.0]))
    widths = np.diff(anchors)
    if free_count <= 0:
        return [0] * widths.size
    ideal = free_count * widths / max(float(widths.sum()), np.finfo(float).tiny)
    counts = np.floor(ideal).astype(int)
    remainder = int(free_count - counts.sum())
    order = np.argsort(-(ideal - counts))
    for index in order[:remainder]:
        counts[index] += 1
    return counts.tolist()


def _build_gap_initial(protected: np.ndarray, counts: Sequence[int], min_gap: float) -> np.ndarray:
    anchors = np.concatenate(([0.0], protected, [1.0]))
    values: list[float] = []
    for gap, count in enumerate(counts):
        lo, hi = float(anchors[gap]), float(anchors[gap + 1])
        if count and hi - lo <= (count + 1) * min_gap:
            raise ValueError("protected points leave no room for the requested min_gap")
        if count:
            values.extend(np.linspace(lo + min_gap, hi - min_gap, count).tolist())
        if gap < protected.size:
            values.append(float(protected[gap]))
    return np.asarray(values, dtype=float)


def conditional_capp_optimize(
    K: int,
    protected_values: ArrayLike,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    protected_indices: Sequence[int] | None = None,
    min_gap: float = 1.0e-6,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Numerically optimize free points while preserving native frequencies.

    ``protected_values`` are normalized coordinates in ``[0,1]``.  When
    ``protected_indices`` is omitted, the protected points remain fixed and
    free points are allocated into the intervening gaps.  With indices, the
    values are fixed at those ranks.  The objective is the explicit Voronoi
    discretization in :func:`capp_discrete_objective`.

    This function intentionally returns a bounded numerical status, not an
    analytic or global optimum.  The discrete parameterization's convexity is
    not established.  If SciPy is unavailable or the constrained solve fails,
    ``status`` is ``BLOCKED_NONCONVEX_OR_SOLVER``.
    """

    if int(K) != K or K < 2:
        raise ValueError("K must be an integer >= 2")
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min_gap must be finite and non-negative")
    protected = np.sort(_as_float_array(protected_values, "protected_values"))
    if protected.size >= K:
        raise ValueError("at least one free frequency is required")
    if protected[0] < 0 or protected[-1] > 1 or np.any(np.diff(protected) <= 0):
        raise ValueError("protected_values must be distinct and in [0, 1]")

    if protected_indices is not None:
        indices = np.asarray(protected_indices, dtype=int)
        if indices.ndim != 1 or indices.size != protected.size or np.any(indices < 0) or np.any(indices >= K):
            raise ValueError("protected_indices must match protected_values and lie in [0,K)")
        if np.unique(indices).size != indices.size:
            raise ValueError("protected_indices must be distinct")
        order = np.argsort(indices)
        indices = indices[order]
        protected_at_index = protected[order]
        if np.any(np.diff(protected_at_index) <= 0):
            raise ValueError("protected values must increase with protected indices")
        anchors_i = np.concatenate(([0], indices, [K - 1]))
        anchors_v = np.concatenate(([0.0], protected_at_index, [1.0]))
        if np.any(np.diff(anchors_v) < 0):
            raise ValueError("protected values conflict with domain endpoints")
        initial = np.empty(K, dtype=float)
        initial[indices] = protected_at_index
        free_indices = np.asarray([i for i in range(K) if i not in set(indices)], dtype=int)
        for left in range(anchors_i.size - 1):
            lo_i, hi_i = int(anchors_i[left]), int(anchors_i[left + 1])
            slots = np.arange(lo_i + 1, hi_i, dtype=int)
            if slots.size:
                initial[slots] = np.linspace(anchors_v[left], anchors_v[left + 1], slots.size + 2)[1:-1]
    else:
        counts = _gap_counts(protected, K - protected.size)
        initial = _build_gap_initial(protected, counts, min_gap)
        free_indices = np.asarray([i for i in range(K) if i not in set(range(1, K))], dtype=int)
    def objective(points: np.ndarray) -> float:
        try:
            return capp_discrete_objective(points, alpha=alpha, beta=beta)
        except ValueError:
            # SLSQP can probe an infeasible finite-difference point before the
            # inequality callback rejects it.  A finite penalty keeps the
            # numerical diagnostic alive and does not pretend the point is a
            # valid optimum.
            points_array = np.asarray(points, dtype=float)
            violations = np.maximum(0.0, -(np.diff(np.concatenate(([0.0], points_array, [1.0]))) - min_gap))
            return float(1.0e12 + np.sum(violations**2) * 1.0e12)

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - depends on environment
        return {
            "status": "BLOCKED_NONCONVEX_OR_SOLVER",
            "reason": f"scipy unavailable: {exc}",
            "values": initial.tolist(),
            "protected_values": protected.tolist(),
            "global_optimum_claim": False,
            "convexity": "not_established",
        }

    if protected_indices is None:
        # In gap mode, each free group has fixed membership between two
        # protected anchors.  This makes all iterates sorted by construction.
        counts = _gap_counts(protected, K - protected.size)
        nfree = int(sum(counts))
        free0: list[float] = []
        anchors = np.concatenate(([0.0], protected, [1.0]))
        for gap, count in enumerate(counts):
            lo, hi = float(anchors[gap]), float(anchors[gap + 1])
            if count:
                free0.extend(np.linspace(lo + min_gap, hi - min_gap, count).tolist())
        free0_array = np.asarray(free0, dtype=float)

        def reconstruct(free: np.ndarray) -> np.ndarray:
            result: list[float] = []
            cursor = 0
            for gap, count in enumerate(counts):
                result.extend(free[cursor : cursor + count].tolist())
                cursor += count
                if gap < protected.size:
                    result.append(float(protected[gap]))
            return np.asarray(result, dtype=float)

        def gap_constraints(free: np.ndarray) -> np.ndarray:
            points = reconstruct(free)
            return np.diff(np.concatenate(([0.0], points, [1.0]))) - min_gap

        bounds = []
        for gap, count in enumerate(counts):
            lo, hi = float(anchors[gap]), float(anchors[gap + 1])
            bounds.extend([(lo + min_gap, hi - min_gap)] * count)
        if nfree == 0:
            return {
                "status": "BLOCKED_NONCONVEX_OR_SOLVER",
                "reason": "no free points after protected allocation",
                "values": initial.tolist(),
                "protected_values": protected.tolist(),
                "global_optimum_claim": False,
                "convexity": "not_established",
            }
        initial_objective = objective(initial)
        result = minimize(
            lambda free: objective(reconstruct(free)),
            free0_array,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": gap_constraints}],
            options={"maxiter": int(max_iter), "ftol": 1.0e-12, "disp": False},
        )
        points = reconstruct(np.asarray(result.x, dtype=float))
    else:
        free_indices = np.asarray([i for i in range(K) if i not in set(indices)], dtype=int)
        free0_array = initial[free_indices]

        def reconstruct_indexed(free: np.ndarray) -> np.ndarray:
            points = initial.copy()
            points[free_indices] = free
            points[indices] = protected_at_index
            return points

        def indexed_constraints(free: np.ndarray) -> np.ndarray:
            points = reconstruct_indexed(free)
            return np.diff(np.concatenate(([0.0], points, [1.0]))) - min_gap

        initial_objective = objective(initial)
        result = minimize(
            lambda free: objective(reconstruct_indexed(free)),
            free0_array,
            method="SLSQP",
            bounds=[(min_gap, 1.0 - min_gap)] * free_indices.size,
            constraints=[{"type": "ineq", "fun": indexed_constraints}],
            options={"maxiter": int(max_iter), "ftol": 1.0e-12, "disp": False},
        )
        points = reconstruct_indexed(np.asarray(result.x, dtype=float))

    success = bool(getattr(result, "success", False)) and np.all(np.diff(points) >= min_gap - 1.0e-8)
    return {
        "status": "NUMERICAL_SMALL_SCALE_ONLY" if success else "BLOCKED_NONCONVEX_OR_SOLVER",
        "solver": "scipy.optimize.SLSQP",
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "")),
        "values": points.tolist(),
        "protected_values": protected.tolist(),
        "initial_objective": float(initial_objective),
        "final_objective": float(objective(points)),
        "min_spacing": float(minimum_spacing(points)),
        "global_optimum_claim": False,
        "convexity": "not_established_for_discrete_parameterization",
        "objective_definition": "Voronoi cell discretization of C_app on [0,1]",
    }


def conditional_capp_from_native_frequencies(
    K: int,
    protected_frequencies: ArrayLike,
    *,
    base: float = 5.0e5,
    theta_hi: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    protected_indices: Sequence[int] | None = None,
    min_gap: float = 1.0e-6,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Run conditional C_app optimization from native ``theta`` values.

    The returned coordinates remain normalized ``phi`` values and the status
    boundary is identical to :func:`conditional_capp_optimize`.
    """

    protected_phi = coordinate_from_frequency(protected_frequencies, base=base, theta_hi=theta_hi)
    result = conditional_capp_optimize(
        K,
        protected_phi,
        alpha=alpha,
        beta=beta,
        protected_indices=protected_indices,
        min_gap=min_gap,
        max_iter=max_iter,
    )
    result["protected_frequencies"] = np.asarray(protected_frequencies, dtype=float).tolist()
    result["protected_values"] = protected_phi.tolist()
    return result


def synthetic_case(name: str, n: int = 801) -> dict[str, Any]:
    """Return a deterministic bimodal or trimodal normalized demand case."""

    if name not in {"bimodal", "trimodal"}:
        raise ValueError("synthetic case must be 'bimodal' or 'trimodal'")
    x = np.linspace(0.0, 1.0, int(n))
    if name == "bimodal":
        centers = np.array([0.26, 0.76])
        widths = np.array([0.045, 0.075])
        weights = np.array([0.55, 0.45])
    else:
        centers = np.array([0.18, 0.50, 0.82])
        widths = np.array([0.035, 0.055, 0.045])
        weights = np.array([0.30, 0.40, 0.30])
    demand = np.zeros_like(x)
    for center, width, weight in zip(centers, widths, weights):
        demand += weight * np.exp(-0.5 * ((x - center) / width) ** 2)
    demand += 1.0e-10  # retain a positive CDF while preserving the modes
    _, demand = normalize_density(x, demand)
    distances = np.geomspace(1.0, 512.0, 256)
    return {
        "name": name,
        "x": x,
        "m": demand,
        "distances": distances,
        "distance_weights": np.full_like(distances, 1.0 / distances.size),
        "metadata": {"synthetic": True, "centers": centers.tolist(), "widths": widths.tolist()},
    }


def _find_mapping_with_demand(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        for demand_key in ("m", "demand", "p_dem", "density"):
            if demand_key in value:
                return value
        for child in value.values():
            found = _find_mapping_with_demand(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping_with_demand(child)
            if found is not None:
                return found
    return None


def load_r0_json(path: str | Path) -> dict[str, Any]:
    """Load common R0 JSON spellings without assuming a repository-specific schema."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    record = _find_mapping_with_demand(raw)
    if record is None:
        raise ValueError("R0 JSON contains no m/demand/p_dem/density array")
    demand_key = next(key for key in ("m", "demand", "p_dem", "density") if key in record)
    raw_m = record[demand_key]
    if isinstance(raw_m, Mapping):
        if "values" in raw_m:
            raw_m = raw_m["values"]
        else:
            raw_m = list(raw_m.values())
    m = _as_float_array(raw_m, demand_key)
    raw_x = None
    for key in ("x", "delta", "grid", "phi", "coordinate", "coordinates", "positions"):
        if key in record:
            raw_x = record[key]
            break
    x = np.arange(m.size, dtype=float) if raw_x is None else _as_float_array(raw_x, "x")
    if x.size != m.size:
        raise ValueError("R0 x/grid and m/demand lengths differ")
    x, m = validate_grid(x, m)
    if np.any(m < 0):
        raise ValueError("R0 demand must be non-negative")
    raw_x_min, raw_x_max = float(x[0]), float(x[-1])
    if raw_x_max <= raw_x_min:
        raise ValueError("R0 coordinate must span a positive interval")
    x = (x - raw_x_min) / (raw_x_max - raw_x_min)
    x[0], x[-1] = 0.0, 1.0
    result: dict[str, Any] = {
        "name": str(record.get("name", source.stem)),
        "x": x,
        "m": m,
        "raw_x_min": raw_x_min,
        "raw_x_max": raw_x_max,
    }
    for key in ("distances", "r", "distance_grid"):
        if key in record:
            result["distances"] = _as_float_array(record[key], key)
            break
    for key in ("distance_weights", "weights", "r_weights"):
        if key in record:
            result["distance_weights"] = _as_float_array(record[key], key)
            break
    if "distances" in result and "distance_weights" in result and result["distances"].size != result["distance_weights"].size:
        raise ValueError("R0 distance grid and weights lengths differ")
    raw_protected = record.get("protected_values", record.get("protected_x"))
    if raw_protected is not None:
        protected_array = _as_float_array(raw_protected, "protected_values")
        result["protected_values"] = (
            (protected_array - raw_x_min) / (raw_x_max - raw_x_min)
        ).tolist()
    else:
        result["protected_values"] = None
    for key in ("protected_native_frequencies", "protected_frequencies", "protected_theta", "protected_omega"):
        if key in record:
            result["protected_frequencies"] = _as_float_array(record[key], key)
            break
    result["protected_indices"] = record.get("protected_indices")
    result["base"] = float(record.get("base", 5.0e5))
    result["metadata"] = record.get("metadata", {})
    result["source"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return result


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def analyze_case(
    case: Mapping[str, Any],
    *,
    K: int = 32,
    lambda_values: Iterable[float] = (0.0, 0.1, 0.3, 1.0),
    alpha: float = 1.0,
    beta: float = 1.0,
    quartic_nu: float | None = None,
    quartic_epsilon: float = 0.0,
    base: float = 5.0e5,
) -> dict[str, Any]:
    """Analyze demand and lambda families for one R0/synthetic case."""

    x, m = validate_grid(case["x"], case["m"])
    if np.any(m < 0):
        raise ValueError("case demand must be non-negative")
    distances = case.get("distances")
    distance_weights = case.get("distance_weights")
    records: list[dict[str, Any]] = []
    for lam in lambda_values:
        lam = float(lam)
        mixed = mix_demand(x, m, lam)
        high = high_rate_density(x, mixed)
        q = quantile_from_density(x, high["rho"], K, endpoint=True)
        theta = frequency_from_coordinate(q, base=base)
        capp = capp_components(x, high["rho"], alpha=alpha, beta=beta)
        quartic = quartic_density_solution(
            x,
            mixed,
            alpha=alpha,
            nu=quartic_nu,
            epsilon=quartic_epsilon,
        )
        collision = None
        collision_status = "UNAVAILABLE_NO_DISTANCE_KERNEL"
        if distances is not None:
            collision = phase_collision_score(theta, distances, distance_weights)
            collision_status = "COMPUTED_PHASE_COHERENCE_PROXY"
        row = {
            "case": str(case.get("name", "r0")),
            "lambda": lam,
            "K": int(K),
            "H": float(high["H"]),
            "H_m13": float(high["H_m13"]),
            "H_unit_m13": float(high["H_unit_m13"]),
            "H_differential": float(high["H_differential"]),
            "H_rho": float(high["H_rho"]),
            "D_star": float(high["H_m13"] ** 3 / (12.0 * K**2)),
            "D_star_shape": float(high["H_m13"] ** 3),
            "rho_integral": float(_trapz(high["rho"], x)),
            "rho_min": float(np.min(high["rho"])),
            "rho_max": float(np.max(high["rho"])),
            "endpoint_left": float(q[0]),
            "endpoint_right": float(q[-1]),
            "min_coordinate_spacing": minimum_spacing(q),
            "min_frequency_spacing": minimum_spacing(theta),
            "coordinate_collision_score": coordinate_collision_score(q),
            "collision_score": collision,
            "collision_status": collision_status,
            "capp_value": float(capp["value"]),
            "capp_alpha_term": float(capp["alpha_term"]),
            "capp_beta_term": float(capp["beta_term"]),
            "quartic_alpha": float(alpha),
            "quartic_nu": None if quartic_nu is None else float(quartic_nu),
            "quartic_epsilon": float(quartic_epsilon),
            "quartic_status": quartic["status"],
            "quartic_max_abs_residual": _finite_or_none(quartic.get("max_abs_residual")),
            "quartic_rho_integral": _finite_or_none(quartic.get("rho_integral")),
            "rho": high["rho"].tolist(),
            "quartic_rho": None if quartic.get("rho") is None else np.asarray(quartic["rho"]).tolist(),
            "quantiles": q.tolist(),
            "theta": theta.tolist(),
        }
        records.append(row)
    result: dict[str, Any] = {
        "name": str(case.get("name", "r0")),
        "x": x.tolist(),
        "m": m.tolist(),
        "records": records,
        "definitions": {
            "rho": "m^(1/3) / integral(m^(1/3))",
            "D_star": "(integral(m^(1/3)))^3 / (12 K^2)",
            "lambda": "(1-lambda)m + lambda/|I|",
            "H": "1 - (integral_phi(m_phi^(1/3)))^3 on unit coordinate",
            "H_m13": "integral(m^(1/3))",
            "H_differential": "differential entropy; separate diagnostic",
            "D_star_shape": "(integral(m^(1/3)))^3 before 1/(12 K^2)",
            "quartic": "outer nu>=0 root only when integral rho=1; otherwise BLOCKED_DIAGNOSTIC",
            "coordinate_collision": "mean exp(-coordinate spacing / nominal cell width)",
            "collision": "mean pairwise squared phase coherence under supplied distance kernel",
            "conditional": "Voronoi/cell C_app discretization; numerical small-scale only",
        },
    }
    protected = case.get("protected_values")
    protected_frequencies = case.get("protected_frequencies")
    if protected is not None or protected_frequencies is not None:
        x0, width = float(x[0]), float(x[-1] - x[0])
        if protected_frequencies is not None:
            conditional = conditional_capp_from_native_frequencies(
                K,
                protected_frequencies,
                base=float(case.get("base", base)),
                alpha=alpha,
                beta=beta,
                protected_indices=case.get("protected_indices"),
            )
            conditional["protected_values_input_coordinate"] = conditional["protected_values"]
            conditional["optimized_values_input_coordinate"] = conditional.get("values")
        else:
            protected_array = np.asarray(protected, dtype=float)
            # Protected values in an R0 file are in the same x-coordinate as
            # the demand grid; conditional_capp_optimize works on [0,1].
            protected_phi = (protected_array - x0) / width
            conditional = conditional_capp_optimize(
                K,
                protected_phi,
                alpha=alpha,
                beta=beta,
                protected_indices=case.get("protected_indices"),
            )
            conditional["protected_values_input_coordinate"] = protected_array.tolist()
            conditional["optimized_values_input_coordinate"] = (
                (x0 + width * np.asarray(conditional["values"])).tolist()
                if conditional.get("values") is not None
                else None
            )
        result["conditional_capp"] = conditional
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_csv(results: Mapping[str, Any], path: str | Path) -> None:
    """Write one scalar row per case/lambda; arrays remain in JSON only."""

    rows: list[dict[str, Any]] = []
    for case in results.get("cases", []):
        rows.extend({k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in case["records"])
    if not rows:
        raise ValueError("no scalar records to write")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(results: Mapping[str, Any], path: str | Path) -> None:
    """Plot demand and lambda densities; imports Matplotlib only on request."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = results.get("cases", [])
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.0, 3.2 * max(1, len(cases))), squeeze=False)
    for axis, case in zip(axes[:, 0], cases):
        x = np.asarray(case["x"], dtype=float)
        axis.plot(x, case["m"], color="black", linewidth=1.8, label="R0 demand m")
        for row in case["records"]:
            axis.plot(x, row["rho"], linewidth=1.0, label=f"rho lambda={row['lambda']:g}")
        axis.set_title(str(case["name"]))
        axis.set_xlabel("coordinate x")
        axis.set_ylabel("density")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_self_checks() -> dict[str, Any]:
    """Run deterministic CPU checks and raise ``AssertionError`` on failure."""

    x = np.linspace(0.0, 1.0, 2001)
    uniform = np.ones_like(x)
    high_uniform = high_rate_density(x, uniform)
    assert np.max(np.abs(high_uniform["rho"] - 1.0)) < 2.0e-12
    assert abs(high_uniform["H"]) < 2.0e-12
    assert abs(high_rate_distortion(x, uniform, 32) - 1.0 / (12.0 * 32**2)) < 2.0e-12

    demand = 0.2 + 1.5 * x**2
    high = high_rate_density(x, demand)
    assert abs(_trapz(high["rho"], x) - 1.0) < 2.0e-12
    assert 0.0 < high["H"] < 1.0
    numerical_d = _trapz(high["m"] / np.maximum(high["rho"], 1.0e-14) ** 2, x) / (12.0 * 32**2)
    assert abs(numerical_d - high_rate_distortion(x, demand, 32)) < 2.0e-8
    assert np.allclose(mix_demand(x, demand, 1.0), uniform, atol=2.0e-12)

    endpoints = quantile_from_density(x, high["rho"], 32, endpoint=True)
    assert endpoints[0] == x[0] and endpoints[-1] == x[-1]
    assert np.all(np.diff(endpoints) >= 0)
    assert minimum_spacing(np.array([0.0, 0.0, 1.0])) == 0.0

    alpha = 2.5
    # tau=1e-3 makes the O(tau^4) signal comparable to float64 subtraction
    # round-off; 5e-3 remains asymptotic while giving a stable numerical check.
    tau = 5.0e-3
    capp = capp_components(x, cosh_density(x, tau), alpha=alpha, beta=0.0)
    uniform_alpha = capp_components(x, np.ones_like(x), alpha=alpha, beta=0.0)["value"]
    observed_coeff = (capp["value"] - uniform_alpha) / tau**4
    assert abs(observed_coeff - capp_quartic_alpha_coefficient(alpha)) < 2.0e-4
    root = solve_quartic_balance(0.7, 0.4, 1.3)
    assert root["status"] == "CONVERGED"
    assert abs(float(root["residual"])) < 1.0e-10
    assert solve_quartic_balance(0.0, 0.0, 1.0)["status"] == "NO_FINITE_ROOT"
    pointwise = solve_pointwise_quartic(high["m"], alpha=0.8, nu=1.1, epsilon=0.01)
    assert pointwise["status"] == "CONVERGED_MONOTONE_BISECTION"
    assert pointwise["max_abs_residual"] < 1.0e-9
    fixed_bad = quartic_density_solution(x, high["m"], alpha=0.8, nu=1.1, epsilon=0.01)
    assert fixed_bad["status"] == "BLOCKED_DIAGNOSTIC"
    assert fixed_bad["rho"] is None
    uniform_fixed = quartic_density_solution(x, np.ones_like(x), alpha=1.0, nu=1.0)
    assert uniform_fixed["status"] == "CONVERGED_FIXED_NU_NORMALIZED"
    assert abs(uniform_fixed["rho_integral"] - 1.0) < 2.0e-12
    uniform_outer = quartic_density_solution(x, np.ones_like(x), alpha=1.0, nu=None)
    assert uniform_outer["status"] == "CONVERGED_OUTER_NU"
    assert abs(uniform_outer["rho_integral"] - 1.0) < 2.0e-12

    for name in ("bimodal", "trimodal"):
        case = synthetic_case(name, n=401)
        result = analyze_case(case, K=16)
        assert len(result["records"]) == 4
        assert all(row["collision_status"].startswith("COMPUTED") for row in result["records"])
        assert all(row["endpoint_left"] == 0.0 and row["endpoint_right"] == 1.0 for row in result["records"])
        assert all(row["quartic_status"] in {"CONVERGED_OUTER_NU", "BLOCKED_DIAGNOSTIC"} for row in result["records"])

    conditional = conditional_capp_optimize(12, [0.22, 0.74], alpha=1.0, beta=1.0)
    assert conditional["status"] in {"NUMERICAL_SMALL_SCALE_ONLY", "BLOCKED_NONCONVEX_OR_SOLVER"}
    values = np.asarray(conditional["values"], dtype=float)
    assert np.min(np.abs(values - 0.22)) < 1.0e-12
    assert np.min(np.abs(values - 0.74)) < 1.0e-12
    assert conditional["global_optimum_claim"] is False
    return {
        "status": "PASS",
        "checks": [
            "rho proportional to m^(1/3)",
            "D* high-rate identity",
            "H and H_m13 reported separately",
            "lambda uniform endpoint",
            "endpoint quantiles and collision spacing",
            "C_app alpha quartic coefficient",
            "monotone quartic root",
            "bimodal and trimodal synthetic cases",
            "protected conditional numerical-only boundary",
        ],
    }


def _parse_lambda_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("lambda list cannot be empty")
    if any(item < 0 or item > 1 for item in values):
        raise argparse.ArgumentTypeError("lambda values must be in [0,1]")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-json", type=Path, help="R0 JSON containing x/grid and m/demand")
    parser.add_argument(
        "--synthetic",
        choices=("none", "bimodal", "trimodal", "both"),
        default="both",
        help="also run deterministic synthetic cases (default: both)",
    )
    parser.add_argument("--K", type=int, default=32, help="number of endpoint-anchored channels")
    parser.add_argument("--lambda-values", type=_parse_lambda_values, default=[0.0, 0.1, 0.3, 1.0])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--quartic-nu",
        type=float,
        default=None,
        help="fixed non-negative nu for diagnostics; omit to outer-solve the supported nu>=0 branch",
    )
    parser.add_argument("--quartic-epsilon", type=float, default=0.0, help="epsilon floor in pointwise quartic")
    parser.add_argument("--base", type=float, default=5.0e5, help="RoPE base for theta spacing diagnostics")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--self-test", action="store_true", help="run CPU numerical self-checks and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        report = run_self_checks()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.K < 2:
        parser.error("--K must be >= 2 for endpoint quantiles")
    cases: list[dict[str, Any]] = []
    if args.r0_json is not None:
        cases.append(load_r0_json(args.r0_json))
    synthetic_names = {
        "bimodal": ["bimodal"],
        "trimodal": ["trimodal"],
        "both": ["bimodal", "trimodal"],
        "none": [],
    }[args.synthetic]
    cases.extend(synthetic_case(name) for name in synthetic_names)
    if not cases:
        parser.error("provide --r0-json or choose a synthetic case")
    analyzed = [
        analyze_case(
            case,
            K=args.K,
            lambda_values=args.lambda_values,
            alpha=args.alpha,
            beta=args.beta,
            quartic_nu=args.quartic_nu,
            quartic_epsilon=args.quartic_epsilon,
            base=args.base,
        )
        for case in cases
    ]
    output: dict[str, Any] = {
        "schema": "demand-companding-numeric-v1",
        "status": "NUMERICAL_DIAGNOSTIC_ONLY",
        "gpu_used": False,
        "cases": analyzed,
        "scalar_quartic_balance_example": solve_quartic_balance(args.alpha, args.beta, max(args.lambda_values)),
    }
    safe_output = _json_safe(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(safe_output, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(safe_output, args.output_csv)
    if args.output_plot is not None:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        write_plot(safe_output, args.output_plot)
    print(json.dumps({"status": output["status"], "cases": [case["name"] for case in analyzed]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI checks
    sys.exit(main())
