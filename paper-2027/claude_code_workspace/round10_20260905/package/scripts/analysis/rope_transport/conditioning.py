"""Per-channel in-window uniqueness and far-range resolvability.

The RoPE basis restricted to a distance window is

    Phi(D) = [cos(w_1 D), sin(w_1 D), ..., cos(w_K D), sin(w_K D)] .

``pair_uniqueness`` reports, for each rotary pair, how much of its 2-D block is
*not* explained by every other pair on that window. It is the per-channel form
of the low-frequency-collapse result: pairs with ``w * L << 1`` are mutually
near-redundant and therefore nearly free to move, while pairs with ``w * L >> 1``
are near-orthogonal and cannot be moved without unrecoverable damage.

``range_resolvability`` reports what a table can still discriminate over a
target range: stable rank, entropy-effective rank, and the fraction of channels
whose phase at the target length leaves the arc seen during training.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def basis_matrix(omega: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Return the ``(n, 2K)`` sin/cos design matrix."""
    theta = np.outer(np.asarray(support, dtype=np.float64), np.asarray(omega, dtype=np.float64))
    out = np.empty((theta.shape[0], 2 * theta.shape[1]), dtype=np.float64)
    out[:, 0::2] = np.cos(theta)
    out[:, 1::2] = np.sin(theta)
    return out


def weighted_gram(omega: np.ndarray, support: np.ndarray, weight: np.ndarray) -> np.ndarray:
    phi = basis_matrix(omega, support)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    w = w / float(w.sum())
    return phi.T @ (phi * w[:, None])


def pair_uniqueness(
    omega: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    *,
    ridge: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """Return per-pair conditional uniqueness in ``(0, 1]``.

    For pair ``k`` the conditional Gram given every other pair is the inverse of
    the corresponding block of the inverse Gram (a Schur complement). The
    reported value normalises it by the pair's own marginal Gram, so a value
    near zero means the pair carries no information the rest of the table does
    not already carry on this window.
    """
    phi = basis_matrix(omega, support)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    w = w / float(w.sum())
    design = phi * np.sqrt(w)[:, None]
    d = design.shape[1]
    k = d // 2
    values = np.empty(k, dtype=np.float64)
    conditional = np.empty(k, dtype=np.float64)
    marginal = np.empty(k, dtype=np.float64)
    for idx in range(k):
        cols = np.arange(2 * idx, 2 * idx + 2)
        rest = np.delete(np.arange(d), cols)
        target = design[:, cols]
        others = design[:, rest]
        # Least squares, not an explicit inverse: the Gram is deliberately
        # near-singular in the collapsed slow block, which is the measurement.
        coef, *_ = np.linalg.lstsq(others, target, rcond=ridge)
        residual = target - others @ coef
        cond_energy = float((residual ** 2).sum()) / 2.0
        marg_energy = float((target ** 2).sum()) / 2.0
        conditional[idx] = cond_energy
        marginal[idx] = marg_energy
        values[idx] = 0.0 if marg_energy <= 0.0 else cond_energy / marg_energy
    return {
        "uniqueness": np.clip(values, 0.0, 1.0),
        "conditional_energy": conditional,
        "marginal_energy": marginal,
    }


def range_resolvability(
    omega: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    *,
    trained_omega: np.ndarray | None = None,
    trained_length: int | None = None,
) -> Dict[str, Any]:
    """Discriminating power of a table over a distance range."""
    gram = weighted_gram(omega, support, weight)
    evals = np.clip(np.linalg.eigvalsh(0.5 * (gram + gram.T)), 0.0, None)
    total = float(evals.sum())
    if total <= 0.0:
        raise ValueError("degenerate Gram spectrum")
    probs = evals / total
    nonzero = probs[probs > 0.0]
    entropy_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    stable_rank = float(total / max(float(evals.max()), 1e-300))
    out: Dict[str, Any] = {
        "entropy_effective_rank": entropy_rank,
        "stable_rank": stable_rank,
        "nominal_dimension": int(gram.shape[0]),
        "min_eigenvalue": float(evals.min()),
        "max_eigenvalue": float(evals.max()),
    }
    if trained_omega is not None and trained_length is not None:
        src = np.asarray(trained_omega, dtype=np.float64).reshape(-1)
        dst = np.asarray(omega, dtype=np.float64).reshape(-1)
        if src.shape != dst.shape:
            raise ValueError("trained_omega must match omega length")
        target = float(np.asarray(support, dtype=np.float64).max())
        trained_arc = src * float(trained_length)
        deployed_arc = dst * target
        # A channel is phase-safe when its deployed arc stays inside the arc it
        # saw during training, or when it already wrapped a full circle then.
        safe = (deployed_arc <= trained_arc + 1e-12) | (trained_arc >= 2.0 * np.pi)
        out["phase_safe_fraction"] = float(safe.mean())
        out["phase_safe_count"] = int(safe.sum())
        out["trained_length"] = int(trained_length)
        out["target_length"] = float(target)
    return out
