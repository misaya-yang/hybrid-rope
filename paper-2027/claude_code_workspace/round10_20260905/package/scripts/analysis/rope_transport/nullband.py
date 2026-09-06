"""Null-band reallocation: the phase-safety box and the operators inside it.

The retrofit problem has an exact feasible-set description. A channel is
phase-safe at a deployed length when its deployed arc stays inside the arc it
saw in training, **or** when it already completed a full turn in training:

    w'_k * L_tgt <= w_k * L_nat        or        w_k * L_nat >= 2 pi .

The second clause depends only on the *native* table, so a channel that wrapped
during training is unconditionally safe wherever it is moved to. The first
clause is a plain box on `w'`. Writing the table in the log coordinate
`phi = -log(w) / log(base)` (for a geometric native table `phi_k = k / K`), the
box becomes a **lower bound** `phi'_k >= phi_k + log(s)/log(base)` on the
unwrapped block and nothing at all elsewhere.

So a retrofit operator is any monotone `phi'` above that floor, and the whole
design question is which one minimises the unrepairable in-window residual
`D*`. Sorting is without loss of generality: a pure frequency permutation is
exactly compensable by the content map (see `tests/test_transport.py`).

Three named blocks matter and they are nested:

* resolvable      -- `uniqueness >= threshold`; moving these is what destroys
                     in-window behaviour;
* redundant       -- the rest; free displacement budget;
* unwrapped       -- the ones the safety box actually constrains.

For OLMo-2 (`base = 5e5`, `L = 4096`, `K = 64`) the unwrapped block is a strict
subset of the redundant block, which is why a training-free retrofit is
possible at all, and which leaves the redundant-but-already-safe channels as a
degree of freedom no wavelength ramp uses.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import numpy as np

from .tables import Table, _check, float32_sha256

TWO_PI = 2.0 * math.pi


def phi_coordinate(omega: np.ndarray, rope_base: float) -> np.ndarray:
    """Log-frequency coordinate; equals ``k / K`` for a geometric native table."""
    return -np.log(np.asarray(omega, dtype=np.float64)) / math.log(float(rope_base))


def omega_from_phi(phi: np.ndarray, rope_base: float) -> np.ndarray:
    return np.power(float(rope_base), -np.asarray(phi, dtype=np.float64))


def phase_safety_box(
    omega_native: np.ndarray, *, native_length: int, target_length: int
) -> Dict[str, np.ndarray]:
    """Exact feasible box for full phase safety at ``target_length``."""
    omega = np.asarray(omega_native, dtype=np.float64).reshape(-1)
    trained_arc = omega * float(native_length)
    wrapped = trained_arc >= TWO_PI
    bound = np.where(wrapped, np.inf, trained_arc / float(target_length))
    return {
        "omega_upper_bound": bound,
        "wrapped": wrapped,
        "unwrapped_index": np.flatnonzero(~wrapped),
        "trained_arc": trained_arc,
    }


def phi_floor(
    omega_native: np.ndarray,
    *,
    native_length: int,
    target_length: int,
    rope_base: float,
) -> np.ndarray:
    """Lower bound on ``phi'`` implied by the safety box (``-inf`` where free)."""
    box = phase_safety_box(
        omega_native, native_length=native_length, target_length=target_length
    )
    bound = box["omega_upper_bound"]
    out = np.full(bound.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(bound)
    out[finite] = phi_coordinate(bound[finite], rope_base)
    return out


# --------------------------------------------------------------------------
# one-parameter warps of the band coordinate, all fixing u=0 -> 0, u=1 -> 1
# --------------------------------------------------------------------------


def warp_identity(u: np.ndarray, param: float) -> np.ndarray:
    return np.asarray(u, dtype=np.float64)


def warp_evq_cosh(u: np.ndarray, param: float) -> np.ndarray:
    """The paper's EVQ-Cosh quantile warp, restricted to a band."""
    tau = float(param)
    u = np.asarray(u, dtype=np.float64)
    if abs(tau) < 1e-9:
        return u.copy()
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * math.sinh(tau))


def warp_power(u: np.ndarray, param: float) -> np.ndarray:
    p = float(param)
    if p <= 0.0:
        raise ValueError(f"power warp needs a positive exponent, got {p}")
    return np.power(np.asarray(u, dtype=np.float64), p)


def warp_logistic(u: np.ndarray, param: float) -> np.ndarray:
    c = float(param)
    u = np.asarray(u, dtype=np.float64)
    if abs(c) < 1e-9:
        return u.copy()
    lo = 1.0 / (1.0 + math.exp(0.5 * c))
    hi = 1.0 / (1.0 + math.exp(-0.5 * c))
    s = 1.0 / (1.0 + np.exp(-c * (u - 0.5)))
    return (s - lo) / (hi - lo)


WARPS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "identity": warp_identity,
    "evq_cosh": warp_evq_cosh,
    "power": warp_power,
    "logistic": warp_logistic,
}


def _finalise(
    phi_new: np.ndarray,
    *,
    rope_base: float,
    name: str,
    meta: Dict[str, Any],
) -> Table:
    arr = _check(omega_from_phi(phi_new, rope_base), name)
    return Table(
        name=name,
        origin="derived",
        inv_freq=arr,
        sha256=float32_sha256(arr),
        meta=meta,
    )


def band_warp_table(
    native: Table,
    *,
    warp: str,
    param: float,
    scale: float,
    band_start: int,
    rope_base: float,
    name: str | None = None,
) -> Table:
    """Apply a one-parameter warp to the log-frequency band ``[band_start, K)``.

    The band's first channel is pinned to its native value (so the resolvable
    block joins continuously) and its last channel lands on ``w_min / scale``.
    ``warp='identity'`` is a linear stretch of the band and is the natural
    zero-parameter reference; official YaRN is a different, binary-ramp member
    of the same feasible set.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    omega = native.inv_freq
    pairs = omega.size
    j0 = int(band_start)
    if not 0 <= j0 <= pairs - 2:
        raise ValueError(f"band_start out of range: {j0}")
    phi = phi_coordinate(omega, rope_base)
    a = float(phi[j0])
    b = float(phi[pairs - 1]) + math.log(float(scale)) / math.log(float(rope_base))
    u = np.linspace(0.0, 1.0, pairs - j0)
    shaped = WARPS[warp](u, param)
    phi_new = phi.copy()
    phi_new[j0:] = a + (b - a) * shaped
    label = name or f"band_{warp}_{param:g}_s{scale:g}_j{j0}"
    return _finalise(
        phi_new,
        rope_base=rope_base,
        name=label,
        meta={
            "rule": "band_warp",
            "warp": warp,
            "param": float(param),
            "scale": float(scale),
            "band_start": j0,
            "band_pairs": int(pairs - j0),
            "rope_base": float(rope_base),
            "source_table": native.name,
            "source_sha256": native.sha256,
        },
    )


def free_band_table(
    native: Table,
    z: np.ndarray,
    *,
    scale: float,
    band_start: int,
    rope_base: float,
    floor: np.ndarray | None = None,
    name: str = "free_band",
) -> Table:
    """Free monotone re-layout of the band from unconstrained coordinates.

    ``z`` has ``K - band_start - 1`` entries. Softplus increments are normalised
    to span the band, then the safety floor is applied by a running maximum, so
    every returned table is feasible by construction.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    omega = native.inv_freq
    pairs = omega.size
    j0 = int(band_start)
    n = pairs - j0
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != n - 1:
        raise ValueError(f"z must have {n - 1} entries, got {z.size}")
    phi = phi_coordinate(omega, rope_base)
    a = float(phi[j0])
    b = float(phi[pairs - 1]) + math.log(float(scale)) / math.log(float(rope_base))
    increments = np.logaddexp(0.0, np.clip(z, -60.0, 60.0))
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("degenerate free-band increments")
    phi_band = a + (b - a) * (cumulative / total)
    if floor is not None:
        f = np.asarray(floor, dtype=np.float64).reshape(-1)[j0:]
        phi_band = np.maximum(phi_band, np.where(np.isfinite(f), f, -np.inf))
    phi_band = np.maximum.accumulate(phi_band)
    # strict monotonicity for the frozen-table contract
    eps = 1e-9
    for i in range(1, phi_band.size):
        if phi_band[i] <= phi_band[i - 1]:
            phi_band[i] = phi_band[i - 1] + eps
    phi_new = phi.copy()
    phi_new[j0:] = phi_band
    return _finalise(
        phi_new,
        rope_base=rope_base,
        name=name,
        meta={
            "rule": "free_band_monotone",
            "scale": float(scale),
            "band_start": j0,
            "band_pairs": int(n),
            "rope_base": float(rope_base),
            "floor_applied": floor is not None,
            "source_table": native.name,
            "source_sha256": native.sha256,
        },
    )


def floor_table(
    native: Table,
    *,
    scale: float,
    native_length: int,
    target_length: int,
    rope_base: float,
    name: str = "safety_floor",
) -> Table:
    """The feasible point sitting exactly on the safety box.

    Every unwrapped channel is divided by exactly the amount phase safety
    demands and nothing else moves. This is position interpolation applied to
    the unwrapped block alone, and it is the natural zero-design baseline.
    """
    phi = phi_coordinate(native.inv_freq, rope_base)
    fl = phi_floor(
        native.inv_freq,
        native_length=native_length,
        target_length=target_length,
        rope_base=rope_base,
    )
    phi_new = np.maximum(phi, np.where(np.isfinite(fl), fl, -np.inf))
    phi_new = np.maximum.accumulate(phi_new)
    return _finalise(
        phi_new,
        rope_base=rope_base,
        name=name,
        meta={
            "rule": "safety_floor",
            "scale": float(scale),
            "native_length": int(native_length),
            "target_length": int(target_length),
            "source_table": native.name,
            "source_sha256": native.sha256,
        },
    )


# --------------------------------------------------------------------------
# graded out-of-window risk
# --------------------------------------------------------------------------


def phase_excess_risk(
    omega_native: np.ndarray,
    omega_new: np.ndarray,
    *,
    native_length: int,
    target_length: int,
) -> Dict[str, Any]:
    """Unseen phase per channel, in turns.

    The binary phase-safe fraction makes the retrofit problem degenerate: the
    feasible set is a box, `D0` is separable, and the constrained optimum is
    simply "move as little as the box demands". The graded version is the one
    with structure.

    A channel explored `min(w_k * L_nat, 2 pi)` radians during training -- past
    a full turn there is no new phase to see. At deployment it sweeps
    `w'_k * L_tgt`. The excess is the phase the model was never shown, capped at
    one turn. It peaks for channels that *just* failed to wrap in training and
    vanishes for near-DC channels, which is the correct mechanical weighting and
    is what a wavelength ramp is crudely approximating.
    """
    src = np.asarray(omega_native, dtype=np.float64).reshape(-1)
    dst = np.asarray(omega_new, dtype=np.float64).reshape(-1)
    if src.shape != dst.shape:
        raise ValueError("tables must have equal length")
    trained = src * float(native_length)
    deployed = dst * float(target_length)
    wrapped = trained >= TWO_PI
    # Unseen phase is what the deployed arc reaches that training never did.
    # Past one turn a channel revisits phases it has already shown, so both
    # arcs are capped at 2 pi and a channel that wrapped in training is at no
    # risk wherever it is deployed.
    unseen = np.clip(np.minimum(deployed, TWO_PI) - trained, 0.0, TWO_PI)
    excess = np.where(wrapped, 0.0, unseen)
    turns = excess / TWO_PI
    return {
        "per_channel_turns": turns,
        "mean_turns": float(turns.mean()),
        "max_turns": float(turns.max()),
        "channels_at_risk": int((turns > 1e-9).sum()),
    }


def free_band_delta_table(
    native: Table,
    z: np.ndarray,
    delta: float,
    *,
    band_start: int,
    rope_base: float,
    start_delta: float = 0.0,
    name: str = "free_band_delta",
) -> Table:
    """Free monotone band re-layout with a free total extension ``delta``.

    ``delta`` and ``start_delta`` are measured in the log-frequency coordinate:
    the band runs from ``phi[band_start] + start_delta`` to ``phi[-1] + delta``,
    so the slowest channel is divided by ``base ** delta`` and the fastest band
    channel by ``base ** start_delta``. Without ``start_delta`` the band's first
    channel would be pinned and uniform operators such as position
    interpolation would not be representable at all.

    Unlike :func:`free_band_table` there is no safety floor: feasibility on the
    risk axis is expressed by the objective, not by the parametrisation, which
    is what makes the two-objective frontier searchable.
    """
    omega = native.inv_freq
    pairs = omega.size
    j0 = int(band_start)
    n = pairs - j0
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != n - 1:
        raise ValueError(f"z must have {n - 1} entries, got {z.size}")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"delta must be finite and non-negative, got {delta}")
    if not np.isfinite(start_delta) or start_delta < 0.0:
        raise ValueError(f"start_delta must be finite and non-negative, got {start_delta}")
    phi = phi_coordinate(omega, rope_base)
    a = float(phi[j0]) + float(start_delta)
    b = float(phi[pairs - 1]) + float(delta)
    if b <= a:
        raise ValueError("degenerate band: start_delta exceeds the available span")
    increments = np.logaddexp(0.0, np.clip(z, -60.0, 60.0))
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("degenerate free-band increments")
    phi_band = a + (b - a) * (cumulative / total)
    eps = 1e-9
    for i in range(1, phi_band.size):
        if phi_band[i] <= phi_band[i - 1]:
            phi_band[i] = phi_band[i - 1] + eps
    phi_new = phi.copy()
    phi_new[j0:] = phi_band
    return _finalise(
        phi_new,
        rope_base=rope_base,
        name=name,
        meta={
            "rule": "free_band_delta",
            "delta": float(delta),
            "start_delta": float(start_delta),
            "effective_scale": float(rope_base ** float(delta)),
            "effective_start_scale": float(rope_base ** float(start_delta)),
            "band_start": j0,
            "band_pairs": int(n),
            "rope_base": float(rope_base),
            "source_table": native.name,
            "source_sha256": native.sha256,
        },
    )


def turn_budget_table(
    native: Table,
    *,
    scale: float,
    beta: float,
    native_length: int,
    rope_base: float,
    ramp_turns: float = 0.0,
    name: str | None = None,
) -> Table:
    """Interpolate exactly the channels that saw fewer than ``beta`` turns.

    This is the one integer of disagreement between the published operators.
    Official YaRN keeps channels above ``beta_fast = 32`` rotations, ramps
    between 32 and 1, and fully interpolates below 1. The phase-argument that
    licenses a training-free swap only requires **one** turn: past a full
    rotation a channel has already presented every phase to the model.

    ``beta = 1`` with ``ramp_turns = 0`` is the exact safety floor.
    ``beta = 32`` with ``ramp_turns = 31`` reproduces YaRN's schedule shape.
    Sweeping ``beta`` therefore interpolates between the two hypotheses and
    makes the disagreement a measurable one-parameter curve.
    """
    if scale <= 1.0:
        raise ValueError(f"scale must exceed one, got {scale}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    omega = native.inv_freq
    turns = omega * float(native_length) / TWO_PI
    if ramp_turns <= 0.0:
        keep = (turns >= float(beta)).astype(np.float64)
    else:
        lo = float(beta) - float(ramp_turns)
        keep = np.clip((turns - lo) / float(ramp_turns), 0.0, 1.0)
        keep = keep * keep * (3.0 - 2.0 * keep)
    arr = omega * keep + (omega / float(scale)) * (1.0 - keep)
    label = name or f"turnbudget_b{beta:g}_r{ramp_turns:g}_s{scale:g}"
    return _finalise(
        phi_coordinate(_check(arr, label), rope_base),
        rope_base=rope_base,
        name=label,
        meta={
            "rule": "turn_budget",
            "beta_turns": float(beta),
            "ramp_turns": float(ramp_turns),
            "scale": float(scale),
            "native_length": int(native_length),
            "untouched_pairs": int((keep >= 1.0).sum()),
            "fully_interpolated_pairs": int((keep <= 0.0).sum()),
            "source_table": native.name,
            "source_sha256": native.sha256,
        },
    )


def encode_band(
    native: Table,
    table: Table,
    *,
    band_start: int,
    rope_base: float,
) -> Tuple[np.ndarray, float, float]:
    """Invert :func:`free_band_delta_table` so a known table can seed a search.

    The parametrisation is scale-invariant in the increments, so fixing the
    increment total at the band span gives ``softplus(z_i) = d_i`` exactly.
    Returns ``(z, delta, start_delta)`` such that rebuilding reproduces ``table``
    on the band up to float round-off. A derivative-free search over 40+ dimensions will not
    find the useful region unaided, so every published operator is injected into
    the initial population and the search can only improve on them.
    """
    phi_src = phi_coordinate(native.inv_freq, rope_base)
    phi_dst = phi_coordinate(table.inv_freq, rope_base)
    j0 = int(band_start)
    # The parametrisation pins everything below the band to native, so a table
    # that moved the resolvable block cannot be represented here. Refuse rather
    # than return a vector that silently decodes to a different operator.
    if j0 > 0 and not np.allclose(phi_dst[:j0], phi_src[:j0], rtol=0.0, atol=1e-12):
        moved = int(np.count_nonzero(~np.isclose(phi_dst[:j0], phi_src[:j0], rtol=0.0, atol=1e-12)))
        raise ValueError(
            f"table moves {moved} channel(s) below band_start={j0}; not encodable on this band"
        )
    band = phi_dst[j0:]
    if band.size < 2:
        raise ValueError("band too short to encode")
    increments = np.diff(band)
    if not np.all(increments > 0.0):
        raise ValueError("table is not strictly increasing on the band")
    delta = max(0.0, float(band[-1] - phi_src[-1]))
    start_delta = max(0.0, float(band[0] - phi_src[j0]))
    # softplus inverse, guarded for very small increments
    z = np.log(np.expm1(np.clip(increments, 1e-12, 60.0)))
    return np.clip(z, -60.0, 60.0), delta, start_delta


def coverage_residual(
    omega_native: np.ndarray,
    omega_new: np.ndarray,
    *,
    native_length: int,
    target_length: int,
    deployed_points: int = 512,
    trained_points: int = 8192,
    weight: np.ndarray | None = None,
) -> Dict[str, Any]:
    """How far the deployed operator leaves the trained operator's trajectory.

    Per-channel phase coverage is the wrong benefit axis, and the repository's
    own RULER numbers say so: the minimal-displacement table that is perfectly
    phase-safe by that measure scores zero. What the model actually saw is the
    whole block-diagonal operator ``R_Omega(D')`` traced over ``D' in [0, L]``.
    At deployment it is shown ``R_Omega'(D)`` for ``D in [0, L_tgt]``. Being
    in-distribution requires a **single** ``D'`` to match every channel at once:

        c(D) = min_{D'} || R_Omega'(D) - R_Omega(D') ||_F^2
             = 4K - 4 max_{D'} sum_k cos(w'_k D - w_k D') .

    This is what makes YaRN's ``beta_fast`` meaningful without any appeal to
    phase safety. A channel that wraps 32 times inside the training window has
    32 preimages and so barely constrains the joint match, which is why leaving
    the fast block untouched is free. A channel that wraps once or twice pins
    ``D'`` sharply, so leaving the middle band untouched makes the joint match
    impossible however safe each channel looks on its own.

    Position interpolation drives this to exactly zero by construction, so the
    quantity is only informative together with the in-window cost ``D*``.
    """
    src = np.asarray(omega_native, dtype=np.float64).reshape(-1)
    dst = np.asarray(omega_new, dtype=np.float64).reshape(-1)
    if src.shape != dst.shape:
        raise ValueError("tables must have equal length")
    pairs = src.size
    deployed = np.linspace(0.0, float(target_length), int(deployed_points))
    trained = np.linspace(0.0, float(native_length), int(trained_points))
    if weight is None:
        w = np.maximum(float(target_length) - deployed, 0.0)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if w.size != deployed.size:
            raise ValueError("weight must match deployed_points")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weight must be positive in total")
    w = w / total

    trained_phase = np.outer(trained, src)                       # (T, K)
    cos_t, sin_t = np.cos(trained_phase), np.sin(trained_phase)
    best = np.empty(deployed.size, dtype=np.float64)
    best_at = np.empty(deployed.size, dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(1, trained.size)))
    for lo in range(0, deployed.size, chunk):
        hi = min(lo + chunk, deployed.size)
        dp = np.outer(deployed[lo:hi], dst)                      # (n, K)
        # sum_k cos(a_k - b_k) = sum_k (cos a cos b + sin a sin b)
        score = np.cos(dp) @ cos_t.T + np.sin(dp) @ sin_t.T      # (n, T)
        idx = np.argmax(score, axis=1)
        best[lo:hi] = score[np.arange(hi - lo), idx]
        best_at[lo:hi] = trained[idx]
    residual = 2.0 - 2.0 * (best / float(pairs))                 # in units of ||R||_F^2 / d
    return {
        "coverage_residual": float(np.dot(w, residual)),
        "max_residual": float(residual.max()),
        "per_distance": residual,
        "deployed": deployed,
        "matched_trained_distance": best_at,
    }
