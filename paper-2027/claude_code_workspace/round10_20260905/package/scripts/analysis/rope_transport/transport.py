"""Static-reparameterization transport residual between two RoPE tables.

Scientific object
-----------------
For one attention head with rotary table ``Omega`` the logit is

    l(q, k, D) = q^T R_Omega(D) k .

Replacing the table by ``Omega'`` and granting the model arbitrary fixed,
position-independent content maps ``M`` (queries) and ``N`` (keys) -- the exact
operator class of the post-hoc transplant obstruction, and a strict superset of
any Q/K LoRA of any rank including cross-pair mixing -- the best achievable
in-window function preservation is

    D* = min_{M,N} E_{D~w} || M^T R_Omega'(D) N - R_Omega(D) ||_F^2 .

For isotropic content, E_{q,k}[(q^T A k - q^T B k)^2] = ||A - B||_F^2 exactly,
so D* is the expected squared logit error of the best static repair, and the
unrepaired value D0 (M = N = I) is the expected squared logit error of a hard
table swap. Because ||R_Omega(D)||_F^2 = d for every D, dividing by d gives a
dimensionless fraction of logit energy.

The obstruction theorem states D* > 0 whenever the frequency multisets differ
up to permutation/sign. It gives no upper bound on D*. This module measures it.

A LoRA of rank r on ``q_proj`` induces a content map with rank(M - I) <= r, and
that rank budget is shared by every head in the layer. The reduced-rank solve
here therefore reports, per head, the exact minimum rank a LoRA must spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

import numpy as np


CHUNK = 256


def _validate_table(omega: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(omega, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError(f"{name}: empty frequency table")
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError(f"{name}: frequencies must be finite and positive")
    return values


def rotation_blocks(omega: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Return ``(n, K, 2, 2)`` rotation blocks ``R(omega_k * D)``."""
    theta = np.outer(np.asarray(support, dtype=np.float64), np.asarray(omega, dtype=np.float64))
    cos = np.cos(theta)
    sin = np.sin(theta)
    out = np.empty(theta.shape + (2, 2), dtype=np.float64)
    out[..., 0, 0] = cos
    out[..., 0, 1] = -sin
    out[..., 1, 0] = sin
    out[..., 1, 1] = cos
    return out


def _blockdiag_matmul(blocks: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """``R @ X`` for block-diagonal ``R`` given as ``(n, K, 2, 2)``.

    ``dense`` is ``(d, d)`` (shared) or ``(n, d, d)`` (per sample).
    """
    n, k = blocks.shape[0], blocks.shape[1]
    d = 2 * k
    if dense.ndim == 2:
        view = dense.reshape(k, 2, d)
        out = np.einsum("nkab,kbj->nkaj", blocks, view, optimize=True)
    elif dense.ndim == 3:
        view = dense.reshape(n, k, 2, d)
        out = np.einsum("nkab,nkbj->nkaj", blocks, view, optimize=True)
    else:
        raise ValueError(f"dense must be rank 2 or 3, got {dense.ndim}")
    return out.reshape(n, d, d)


def _chunks(total: int, size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(total, start + size)


def hard_swap_residual(
    omega_src: np.ndarray,
    omega_dst: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Analytic ``D0 = E||R_dst - R_src||_F^2``; no optimization involved."""
    src = _validate_table(omega_src, "omega_src")
    dst = _validate_table(omega_dst, "omega_dst")
    if src.shape != dst.shape:
        raise ValueError("frequency tables must have equal length")
    d = 2 * src.size
    delta = np.outer(np.asarray(support, dtype=np.float64), dst - src)
    inner = 4.0 * np.cos(delta).sum(axis=1)
    return float(2.0 * d - float(np.asarray(weight, dtype=np.float64) @ inner))


def _accumulate(
    blocks_src: np.ndarray,
    blocks_dst: np.ndarray,
    weight: np.ndarray,
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(gram, cross)`` for one half-step.

    With ``right = N`` (key map) and ``left = None`` this returns
    ``Q = sum w C C^T`` and ``P = sum w R_src C^T`` for ``C = R_dst N``.

    With ``left = M`` and ``right = None`` it returns ``S = sum w D^T D`` and
    ``T = sum w D^T R_src`` for ``D = M^T R_dst``.
    """
    d = blocks_src.shape[1] * 2
    gram = np.zeros((d, d), dtype=np.float64)
    cross = np.zeros((d, d), dtype=np.float64)
    n_total = blocks_src.shape[0]
    for lo, hi in _chunks(n_total, CHUNK):
        w = weight[lo:hi]
        sw = np.sqrt(w)[:, None, None]
        rs = blocks_src[lo:hi]
        rd = blocks_dst[lo:hi]
        if right is not None:
            mat = _blockdiag_matmul(rd, right)  # C = R_dst N
            cw = mat * sw
            flat = np.ascontiguousarray(np.transpose(cw, (1, 0, 2))).reshape(d, -1)
            gram += flat @ flat.T
            # P = sum_n w_n R_src C^T
            ct = np.ascontiguousarray(np.transpose(mat, (0, 2, 1)))
            cross += np.tensordot(w, _blockdiag_matmul(rs, ct), axes=(0, 0))
        else:
            if left is None:
                raise ValueError("one of left/right must be provided")
            # D = M^T R_dst = (R_dst^T M)^T
            mat = np.ascontiguousarray(
                np.transpose(_blockdiag_matmul(np.swapaxes(rd, -1, -2), left), (0, 2, 1))
            )
            dw = (mat * sw).reshape(-1, d)
            gram += dw.T @ dw
            # T = sum_n w_n D^T R_src = sum_n w_n (R_src^T D)^T
            stacked = _blockdiag_matmul(np.swapaxes(rs, -1, -2), mat)
            cross += np.tensordot(w, np.transpose(stacked, (0, 2, 1)), axes=(0, 0))
    return gram, cross


def _sym_inv_sqrt(mat: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    sym = 0.5 * (mat + mat.T)
    scale = float(np.trace(sym)) / max(1, sym.shape[0])
    sym = sym + ridge * max(scale, 1e-300) * np.eye(sym.shape[0])
    evals, evecs = np.linalg.eigh(sym)
    evals = np.clip(evals, 1e-300, None)
    root = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    inv_root = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return root, inv_root


def _truncate(mat: np.ndarray, rank: int) -> np.ndarray:
    if rank >= min(mat.shape):
        return mat
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    s = s.copy()
    s[rank:] = 0.0
    return (u * s) @ vt


def _solve_left(gram: np.ndarray, cross: np.ndarray, rank: int | None, ridge: float) -> np.ndarray:
    """Return ``M`` minimizing ``sum w ||M^T C - R||^2`` with rank(M - I) <= rank."""
    d = gram.shape[0]
    eye = np.eye(d)
    if rank is None:
        y = np.linalg.solve(
            gram + ridge * max(float(np.trace(gram)) / d, 1e-300) * eye, cross.T
        ).T
        return y.T
    root, inv_root = _sym_inv_sqrt(gram, ridge)
    target = (cross - gram) @ inv_root
    z = _truncate(target, int(rank)) @ inv_root
    return (eye + z).T


def _solve_right(gram: np.ndarray, cross: np.ndarray, rank: int | None, ridge: float) -> np.ndarray:
    """Return ``N`` minimizing ``sum w ||D N - R||^2`` with rank(N - I) <= rank."""
    d = gram.shape[0]
    eye = np.eye(d)
    if rank is None:
        return np.linalg.solve(gram + ridge * max(float(np.trace(gram)) / d, 1e-300) * eye, cross)
    root, inv_root = _sym_inv_sqrt(gram, ridge)
    target = inv_root @ (cross - gram)
    z = inv_root @ _truncate(target, int(rank))
    return eye + z


def _objective_left(gram: np.ndarray, cross: np.ndarray, mat: np.ndarray, d: int) -> float:
    y = mat.T
    return float(np.trace(y @ gram @ y.T) - 2.0 * np.trace(y @ cross.T) + d)


def _objective_right(gram: np.ndarray, cross: np.ndarray, mat: np.ndarray, d: int) -> float:
    return float(np.trace(mat.T @ gram @ mat) - 2.0 * np.trace(mat.T @ cross) + d)


@dataclass
class TransportResult:
    hard_swap: float
    repaired: float
    relative_hard_swap: float
    relative_repaired: float
    repairability: float
    iterations: int
    converged: bool
    rank: int | None
    query_map: np.ndarray = field(repr=False)
    key_map: np.ndarray = field(repr=False)
    history: list = field(default_factory=list, repr=False)

    def summary(self) -> Dict[str, Any]:
        return {
            "hard_swap": self.hard_swap,
            "repaired": self.repaired,
            "relative_hard_swap": self.relative_hard_swap,
            "relative_repaired": self.relative_repaired,
            "repairability": self.repairability,
            "iterations": self.iterations,
            "converged": self.converged,
            "rank": self.rank,
        }


def rank_matching_permutation(omega_src: np.ndarray, omega_dst: np.ndarray) -> np.ndarray:
    """Block permutation sending each target pair to the same-rank source pair.

    This is the physically correct starting point: content should move to the
    pair whose new frequency is closest to its old one. For the strictly
    decreasing tables used in practice it is the identity, but it is the exact
    optimum in the permuted case the obstruction theorem singles out, and it
    keeps the bilinear search out of the trivial ``M -> 0`` basin.
    """
    src = _validate_table(omega_src, "omega_src")
    dst = _validate_table(omega_dst, "omega_dst") if np.all(np.diff(omega_dst) < 0) else np.asarray(
        omega_dst, dtype=np.float64
    ).reshape(-1)
    k = src.size
    order_src = np.argsort(-src, kind="stable")
    order_dst = np.argsort(-dst, kind="stable")
    sigma = np.empty(k, dtype=np.int64)
    sigma[order_dst] = order_src
    out = np.zeros((2 * k, 2 * k), dtype=np.float64)
    for j in range(k):
        out[2 * j : 2 * j + 2, 2 * sigma[j] : 2 * sigma[j] + 2] = np.eye(2)
    return out


def _run_als(
    blocks_src: np.ndarray,
    blocks_dst: np.ndarray,
    weight: np.ndarray,
    d: int,
    start: np.ndarray,
    rank: int | None,
    max_iter: int,
    tol: float,
    ridge: float,
) -> Tuple[float, np.ndarray, np.ndarray, int, bool, list]:
    query_map = start.copy()
    key_map = start.copy()
    previous = float("inf")
    history: list = []
    converged = False
    used = 0
    for step in range(int(max_iter)):
        used = step + 1
        gram, cross = _accumulate(blocks_src, blocks_dst, weight, None, key_map)
        query_map = _solve_left(gram, cross, rank, ridge)
        value_q = _objective_left(gram, cross, query_map, d)

        gram, cross = _accumulate(blocks_src, blocks_dst, weight, query_map, None)
        key_map = _solve_right(gram, cross, rank, ridge)
        value = _objective_right(gram, cross, key_map, d)

        if not np.isfinite(value):
            raise FloatingPointError("transport objective became non-finite")
        history.append({"iteration": used, "after_query": value_q, "after_key": value})
        if abs(previous - value) <= tol * max(1.0, abs(previous)):
            previous = value
            converged = True
            break
        previous = value
    return float(previous), query_map, key_map, used, converged, history


def transport_residual(
    omega_src: np.ndarray,
    omega_dst: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    *,
    rank: int | None = None,
    max_iter: int = 60,
    tol: float = 1e-12,
    ridge: float = 1e-12,
) -> TransportResult:
    """Alternating exact least squares for ``D*``.

    Each half-step is the closed-form (reduced-rank when ``rank`` is set)
    minimizer, so the objective is monotone within a start. The bilinear problem
    is not jointly convex and has a trivial ``M, N -> 0`` basin whose value is
    exactly ``d`` (a model that emits no positional signal at all), so the
    search runs from both the identity and the rank-matching permutation and
    keeps the better result. ``D*`` is therefore always an upper bound on the
    true optimum, never an underestimate of achievable repair.
    """
    src = _validate_table(omega_src, "omega_src")
    dst = _validate_table(omega_dst, "omega_dst") if np.all(np.diff(omega_dst) < 0) else np.asarray(
        omega_dst, dtype=np.float64
    ).reshape(-1)
    if src.shape != dst.shape:
        raise ValueError("frequency tables must have equal length")
    support = np.asarray(support, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if support.shape != weight.shape:
        raise ValueError("support and weight must have equal length")
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weight must be finite and positive in total")
    weight = weight / total

    d = 2 * src.size
    blocks_src = rotation_blocks(src, support)
    blocks_dst = rotation_blocks(dst, support)
    hard = hard_swap_residual(src, dst, support, weight)

    starts = [("identity", np.eye(d))]
    permutation = rank_matching_permutation(src, dst)
    if not np.array_equal(permutation, np.eye(d)):
        starts.append(("rank_matching_permutation", permutation))

    best: Tuple[float, np.ndarray, np.ndarray, int, bool, list, str] | None = None
    for label, start in starts:
        value, qmap, kmap, used, converged, history = _run_als(
            blocks_src, blocks_dst, weight, d, start, rank, max_iter, tol, ridge
        )
        if best is None or value < best[0]:
            best = (value, qmap, kmap, used, converged, history, label)
    assert best is not None

    repaired = max(0.0, float(best[0]))
    result = TransportResult(
        hard_swap=float(hard),
        repaired=repaired,
        relative_hard_swap=float(hard) / d,
        relative_repaired=repaired / d,
        repairability=float("nan") if hard <= 0.0 else 1.0 - repaired / float(hard),
        iterations=best[3],
        converged=best[4],
        rank=None if rank is None else int(rank),
        query_map=best[1],
        key_map=best[2],
        history=best[5],
    )
    result.history.append({"selected_start": best[6], "starts_tried": [name for name, _ in starts]})
    return result


def residual_by_pair(
    omega_src: np.ndarray,
    omega_dst: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    query_map: np.ndarray,
    key_map: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Attribute the repaired residual energy to source rotary pairs."""
    src = _validate_table(omega_src, "omega_src")
    dst = _validate_table(omega_dst, "omega_dst")
    support = np.asarray(support, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight = weight / float(weight.sum())
    k = src.size
    d = 2 * k
    query_rows = np.zeros(k, dtype=np.float64)
    key_cols = np.zeros(k, dtype=np.float64)
    blocks_src = rotation_blocks(src, support)
    blocks_dst = rotation_blocks(dst, support)
    for lo, hi in _chunks(support.size, CHUNK):
        w = weight[lo:hi]
        # M^T R_dst N = (R_dst^T M)^T N
        approx = (
            np.transpose(
                _blockdiag_matmul(np.swapaxes(blocks_dst[lo:hi], -1, -2), query_map), (0, 2, 1)
            )
            @ key_map
        )
        rs = blocks_src[lo:hi]
        exact = np.zeros_like(approx)
        for idx in range(k):
            exact[:, 2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = rs[:, idx]
        sq = (approx - exact) ** 2
        n = sq.shape[0]
        query_rows += w @ sq.reshape(n, k, 2, d).sum(axis=(2, 3))
        key_cols += w @ sq.reshape(n, d, k, 2).sum(axis=(1, 3))
    return {"query_pair_energy": query_rows, "key_pair_energy": key_cols}
