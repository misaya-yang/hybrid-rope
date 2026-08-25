#!/usr/bin/env python3
"""Best-found third-axis value on one static geometric metric.

The causal decomposition ``x_k = -log omega_k = a + R z_k`` separates the
sampled support ``(a, R)`` from the normalized interior allocation ``z``.  This
script answers one bounded question: **holding ``(a, R)`` exactly fixed, how
far can a declared numerical search move the block-whitened stable rank**

    r2(z) = 2K / (1 + (K - 1) * cbar(z))

**be made, and how much of that range does a given construction capture?**

Scope, stated once so no caller mistakes it for more:

* ``r2`` is a static, phase-invariant property of the positional basis under a
  declared distance measure.  It is **not** an LM-quality or extrapolation
  predictor; see ``AGENTS.md`` claim ceilings and ``INDEX.md`` §3.4.
* The reported value is the best point found by Adam from the declared
  restarts.  It is a lower bound on the true supremum, not a global ceiling or
  an upper bound on the continuous allocation family.
* ``2K`` is the algebraic maximum of ``r2`` (reached only at ``cbar = 0``).
  A numerical gap to ``2K`` cannot be attributed to the distance measure or
  support without a global certificate that this script does not provide.

CPU only; no GPU, no training, no paid compute.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.experiments.geo_rope_contract import std_geo_inv_freq  # noqa: E402
from scripts.lib.rope.schedules import evq_cosh_phi  # noqa: E402

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# distance measure
# --------------------------------------------------------------------------
def measure_weights(length: int, kind: str) -> torch.Tensor:
    """Distribution of the relative distance ``d`` over ``{0, ..., length-1}``."""
    d = torch.arange(length, dtype=torch.float64)
    if kind == "causal":
        # p(d) proportional to (L - d): the exact relative-distance histogram of
        # a causal window of length L.  Matches the M4 target-free screens.
        return (length - d) / (length * (length + 1) / 2.0)
    if kind == "uniform":
        return torch.full((length,), 1.0 / length, dtype=torch.float64)
    raise ValueError(f"unknown measure {kind!r}")


def characteristic(weights: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``E[exp(i t d)]`` by direct summation: stable at every ``t``, including 0."""
    d = torch.arange(weights.numel(), dtype=torch.float64)
    phase = t.reshape(-1, 1) * d.reshape(1, -1)
    return (weights * torch.exp(torch.complex(torch.zeros_like(phase), phase))).sum(1).reshape(t.shape)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def gram_blocks(omega: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``E[[cos(w_i d), sin(w_i d)]^T [cos(w_j d), sin(w_j d)]]`` as 2x2 blocks."""
    difference = omega[:, None] - omega[None, :]
    total = omega[:, None] + omega[None, :]
    chi_d = characteristic(weights, difference)
    chi_s = characteristic(weights, total)
    return torch.stack(
        (
            torch.stack((0.5 * (chi_d.real + chi_s.real), 0.5 * (chi_s.imag - chi_d.imag)), -1),
            torch.stack((0.5 * (chi_s.imag + chi_d.imag), 0.5 * (chi_d.real - chi_s.real)), -1),
        ),
        -2,
    )


def inverse_sqrt_2x2(matrix: torch.Tensor) -> torch.Tensor:
    """Closed-form ``M^{-1/2}`` for batched symmetric positive-definite 2x2."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]
    root_det = (a * d - b * c).clamp_min(1e-300).sqrt()
    scale = root_det * (a + d + 2.0 * root_det).clamp_min(1e-300).sqrt()
    adjugate = torch.stack(
        (torch.stack((d + root_det, -b), -1), torch.stack((-c, a + root_det), -1)), -2
    )
    return adjugate / scale[..., None, None]


def stable_rank(omega: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(r2, cbar)`` for a realized frequency vector."""
    blocks = gram_blocks(omega, weights)
    index = torch.arange(omega.numel())
    root = inverse_sqrt_2x2(blocks[index, index])
    whitened = torch.einsum("aik,abkl,blj->abij", root, blocks, root)
    upper = torch.triu_indices(omega.numel(), omega.numel(), 1)
    collision = (0.5 * (whitened**2).sum((-1, -2)))[upper[0], upper[1]].mean()
    pairs = float(omega.numel())
    return 2.0 * pairs / (1.0 + (pairs - 1.0) * collision), collision


def omega_from_z(z: torch.Tensor, log_max: float, span: float) -> torch.Tensor:
    """``x = a + R z`` with the support endpoints held exactly."""
    return torch.exp(log_max - span * z)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def search_best_found(
    pairs: int,
    log_max: float,
    span: float,
    weights: torch.Tensor,
    *,
    restarts: int,
    steps: int,
) -> tuple[float, torch.Tensor, list[float]]:
    """Maximize ``r2`` over monotone ``z`` with ``z_0 = 0`` and ``z_{K-1} = 1``."""
    attained: list[float] = []
    best_value, best_z = -math.inf, None
    for seed in range(restarts):
        generator = torch.Generator().manual_seed(seed)
        logits = (0.3 * torch.randn(pairs - 1, generator=generator)).requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=0.05)
        for step in range(steps):
            if step == int(0.8 * steps):
                for group in optimizer.param_groups:
                    group["lr"] = 0.005
            optimizer.zero_grad()
            z = torch.cat((torch.zeros(1), torch.cumsum(torch.softmax(logits, 0), 0)))
            _, collision = stable_rank(omega_from_z(z, log_max, span), weights)
            collision.backward()
            optimizer.step()
        with torch.no_grad():
            z = torch.cat((torch.zeros(1), torch.cumsum(torch.softmax(logits, 0), 0)))
            value = float(stable_rank(omega_from_z(z, log_max, span), weights)[0])
        attained.append(value)
        if value > best_value:
            best_value, best_z = value, z.detach()
    assert best_z is not None
    return best_value, best_z, attained


def search_free_support(
    pairs: int,
    weights: torch.Tensor,
    *,
    restarts: int,
    steps: int,
) -> tuple[float, torch.Tensor]:
    """Maximize ``r2`` over all ``K`` frequencies in ``(0, 1)`` with no support pinned.

    Comparing this value with a fixed-support search is descriptive only: both
    are optimizer outcomes and neither certifies a global optimum.
    """
    best_value, best_omega = -math.inf, None
    for seed in range(restarts):
        generator = torch.Generator().manual_seed(seed)
        logits = (0.5 * torch.randn(pairs, generator=generator)).requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=0.03)
        for step in range(steps):
            if step == int(0.8 * steps):
                for group in optimizer.param_groups:
                    group["lr"] = 0.003
            optimizer.zero_grad()
            _, collision = stable_rank(torch.sigmoid(logits), weights)
            collision.backward()
            optimizer.step()
        with torch.no_grad():
            omega = torch.sigmoid(logits)
            value = float(stable_rank(omega, weights)[0])
        if value > best_value:
            best_value, best_omega = value, omega.detach()
    assert best_omega is not None
    return best_value, best_omega


def measure_comparison(
    pairs: int, length: int, *, restarts: int, steps: int
) -> dict[str, object]:
    """Compare best-found values under uniform and causal distance measures.

    Under a *uniform* distance measure and free support, ``K`` rotary pairs can
    be made exactly orthogonal whenever they sit on one parity class of the
    lattice ``omega_k = pi a_k / L`` (independently established in
    ``analysis/full_rope_audit/``).  The causal window's distance distribution
    is triangular rather than uniform, so that lattice does not exist there.
    This function does not attribute the remaining gap causally because the
    numerical searches have no global optimality certificate.
    """
    out: dict[str, object] = {"algebraic_max_r2": 2.0 * pairs}
    for kind in ("uniform", "causal"):
        weights = measure_weights(length, kind)
        value, omega = search_free_support(pairs, weights, restarts=restarts, steps=steps)
        lattice = (omega * length / math.pi).sort(descending=True).values
        out[kind] = {
            "free_support_r2": value,
            "lattice_index_a_k": [round(float(x), 4) for x in lattice],
            "max_distance_from_integer": float(
                (lattice - lattice.round()).abs().max()
            ),
        }
    return out


def recurrence_gap(omega: torch.Tensor, length: int) -> float:
    """``max_d |Phi(d + L) - Phi(d)|`` for ``Phi(d) = mean_k cos(w_k d)``.

    A table whose collision is minimized by an exact Fourier comb repeats after
    ``L``; this reports how close a table is to that degeneracy.
    """
    distance = torch.arange(2 * length, dtype=torch.float64)
    phi = torch.cos(omega[:, None] * distance[None, :]).mean(0)
    return float((phi[length:] - phi[:length]).abs().max())


# --------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------
def self_check(pairs: int, length: int, base: float) -> dict[str, float]:
    """Agree with the tracked numpy owner and with the pairwise identity."""
    from scripts.analysis.full_rope_collision_audit import (
        geometric_phi,
        schedule_metrics,
        whitened_blocks,
        gram_blocks as numpy_gram_blocks,
        _block_matrix,
    )

    phi = geometric_phi(pairs)
    omega = torch.tensor(np.power(base, -phi))
    mine = float(stable_rank(omega, measure_weights(length, "uniform"))[0])
    # The numpy owner integrates d over the continuous interval [0, L]; this
    # script sums over the discrete grid {0, ..., L-1}.  Compare against the
    # owner's own continuous kernel to validate the whitening algebra itself.
    blocks = numpy_gram_blocks(np.power(base, -phi), length)
    white = _block_matrix(whitened_blocks(blocks))
    reference = float(np.trace(white) ** 2 / np.sum(white**2))
    owner = schedule_metrics(phi, length, base)["full_whitened_stable_rank"]
    if not math.isclose(reference, owner, rel_tol=1e-12):
        raise AssertionError("numpy owner disagrees with its own identity")
    continuous = float(
        stable_rank(
            omega,
            # trapezoid weights on {0..L} reproduce the continuous mean closely
            torch.cat((torch.tensor([0.5]), torch.ones(length - 1), torch.tensor([0.5]))) / length,
        )[0]
    )
    return {
        "discrete_uniform_r2": mine,
        "continuous_owner_r2": owner,
        "trapezoid_vs_owner_abs_diff": abs(continuous - owner),
    }


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--base", type=float, default=256.0)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--measure", choices=("causal", "uniform"), default="causal")
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="also compare free-support searches under uniform and causal measures",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    weights = measure_weights(args.length, args.measure)
    native = std_geo_inv_freq(2 * args.pairs, args.base, dtype=torch.float64)
    omega_max, omega_min = float(native[0]), float(native[-1])
    log_max = math.log(omega_max)
    span = math.log(omega_max / omega_min)

    geometric_z = torch.linspace(0.0, 1.0, args.pairs, dtype=torch.float64)
    cosh_phi = evq_cosh_phi(args.pairs, tau=args.tau, midpoint=True, dtype=torch.float64)
    cosh_z = (cosh_phi - cosh_phi[0]) / (cosh_phi[-1] - cosh_phi[0])

    arms = {"FMRoPE (geometric)": geometric_z, "anchored EVQ-Cosh": cosh_z}
    arm_report = {}
    for name, z in arms.items():
        omega = omega_from_z(z, log_max, span)
        value, collision = stable_rank(omega, weights)
        arm_report[name] = {
            "r2": float(value),
            "cbar": float(collision),
            "recurrence_gap": recurrence_gap(omega, args.length),
        }

    best_found, z_star, attained = search_best_found(
        args.pairs, log_max, span, weights, restarts=args.restarts, steps=args.steps
    )
    omega_star = omega_from_z(z_star, log_max, span)
    floor = arm_report["FMRoPE (geometric)"]["r2"]
    report = {
        "config": {
            "pairs": args.pairs,
            "length": args.length,
            "base": args.base,
            "tau": args.tau,
            "measure": args.measure,
            "restarts": args.restarts,
            "steps": args.steps,
        },
        "support": {
            "omega_max": omega_max,
            "omega_min": omega_min,
            "omega_min_times_length": omega_min * args.length,
            "log_span": span,
        },
        "algebraic_max_r2": 2.0 * args.pairs,
        "arms": arm_report,
        "best_found_static_rank": {
            "r2": best_found,
            "restart_values": attained,
            "restart_spread": max(attained) - min(attained),
            "recurrence_gap": recurrence_gap(omega_star, args.length),
            "fraction_of_algebraic_max": best_found / (2.0 * args.pairs),
            "omega_star_times_length": (omega_star * args.length).tolist(),
            "z_star": z_star.tolist(),
        },
        "fraction_of_best_found_improvement_from_geometric": {
            name: (row["r2"] - floor) / (best_found - floor)
            for name, row in arm_report.items()
        },
        "self_check": self_check(args.pairs, args.length, args.base),
    }
    if args.decompose:
        report["measure_comparison"] = measure_comparison(
            args.pairs, args.length, restarts=args.restarts, steps=args.steps
        )
    text = json.dumps(report, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
