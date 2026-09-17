#!/usr/bin/env python3
"""CPU reference mathematics for carrier-aligned NCP.

The complex layout is Hugging Face split-half: ``z[k] = x[k] + 1j*x[k+K]``.
No checkpoint, task output, loss gradient, or GPU is used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from . import CARRIER_RATIO


def tensor_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_table(values: Any, *, pairs: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 1:
        raise ValueError("frequency table must be a one-dimensional float32 array")
    if pairs is not None and array.size != pairs:
        raise ValueError("frequency table pair count differs")
    if (
        array.size < 4
        or not np.isfinite(array).all()
        or np.any(array <= 0)
        or np.any(array[:-1] <= array[1:])
    ):
        raise ValueError("frequency table must be finite, positive, and strictly descending")
    return np.ascontiguousarray(array)


def build_carrier_table(
    native_fp32: Any,
    ncp_fp32: Any,
    length: int,
    *,
    carrier_ratio: float = CARRIER_RATIO,
) -> dict[str, Any]:
    """Return the fixed active set, carrier slot, and carrier-NCP table."""
    native = validate_table(native_fp32)
    ncp = validate_table(ncp_fp32, pairs=native.size)
    if type(length) is not int or length < 2:
        raise ValueError("length must be an integer of at least two")
    if not math.isfinite(carrier_ratio) or carrier_ratio <= 0:
        raise ValueError("carrier ratio must be finite and positive")
    changed = np.flatnonzero(ncp != native)
    indices = np.arange(native.size)
    active = changed[(changed > 0) & (changed < native.size - 1)]
    active = active[length * native[active].astype(np.float64) <= 2.0 * math.pi]
    if active.size < 2:
        raise ValueError("carrier alignment needs at least two protected active dimensions")
    wc64 = 2.0 * math.pi / (carrier_ratio * length)
    wc32 = np.float32(wc64)
    candidates = [
        int(k)
        for k in active
        if ncp[k - 1] > wc32 > ncp[k + 1]
    ]
    if not candidates:
        raise ValueError("the carrier frequency is unsupported by this NCP geometry")
    carrier = min(candidates, key=lambda k: (abs(math.log(float(ncp[k]) / wc64)), k))
    carrier_table = ncp.copy()
    carrier_table[carrier] = wc32
    validate_table(carrier_table, pairs=native.size)
    local = int(np.flatnonzero(active == carrier)[0])
    return {
        "native": native,
        "ncp": ncp,
        "carrier_table": carrier_table,
        "changed_indices": changed.astype(np.int64),
        "active_indices": active.astype(np.int64),
        "carrier_slot": carrier,
        "carrier_local": local,
        "carrier_ratio": float(carrier_ratio),
        "carrier_frequency_float64": float(wc64),
        "carrier_frequency_float32": float(wc32),
        "native_table_sha256_float32": tensor_sha256(native),
        "ncp_table_sha256_float32": tensor_sha256(ncp),
        "carrier_table_sha256_float32": tensor_sha256(carrier_table),
        "indices": indices,
    }


def split_half_to_complex(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] % 2:
        raise ValueError("split-half input needs an even final dimension")
    pairs = array.shape[-1] // 2
    return array[..., :pairs].astype(np.float64) + 1j * array[..., pairs:].astype(np.float64)


def complex_to_split_half(values: Any, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(values)
    if not np.iscomplexobj(array):
        raise ValueError("complex input required")
    return np.concatenate((array.real, array.imag), axis=-1).astype(dtype)


def signed_moment(
    q_complex: Any,
    k_complex: Any,
    lag: Any,
    wc: float,
    *,
    weights: Any | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return ``-E Herm((exp(-i*d*wc)-1) k q^H) * scale``."""
    q = np.asarray(q_complex, dtype=np.complex128)
    k = np.asarray(k_complex, dtype=np.complex128)
    d = np.asarray(lag, dtype=np.float64)
    if q.ndim != 2 or q.shape != k.shape or d.shape != (q.shape[0],):
        raise ValueError("q/k must be [samples,dimensions] and lag [samples]")
    if q.shape[0] == 0 or q.shape[1] < 2 or np.any(d < 0):
        raise ValueError("signed moment needs nonempty causal samples")
    if not np.isfinite(q).all() or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("signed moment inputs must be finite")
    if not math.isfinite(wc) or wc <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("carrier frequency and attention scale must be positive")
    if weights is None:
        probability = np.full(q.shape[0], 1.0 / q.shape[0], dtype=np.float64)
    else:
        probability = np.asarray(weights, dtype=np.float64)
        if probability.shape != (q.shape[0],) or np.any(probability < 0) or not np.isfinite(probability).all():
            raise ValueError("weights must be finite and nonnegative")
        total = float(probability.sum())
        if total <= 0:
            raise ValueError("weights must have positive total")
        probability = probability / total
    gamma = np.expm1(-1j * d * float(wc))
    raw = np.einsum(
        "n,ni,nj->ij", probability * gamma, k, q.conj(), optimize=True,
    )
    matrix = -float(scale) * (raw + raw.conj().T) / 2.0
    return np.ascontiguousarray(matrix, dtype=np.complex128)


def _fix_phase(vector: np.ndarray, carrier_local: int, tolerance: float) -> np.ndarray:
    value = vector[carrier_local]
    if abs(value) <= tolerance:
        index = int(np.argmax(np.abs(vector)))
        value = vector[index]
    if abs(value) > 0:
        vector = vector * np.exp(-1j * np.angle(value))
    if vector[carrier_local].real < 0:
        vector = -vector
    vector[carrier_local] = complex(float(vector[carrier_local].real), 0.0)
    return vector


def choose_direction(matrix: Any, carrier_local: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the deterministic top-eigenspace direction specified by the plan."""
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError("matrix must be square with dimension at least two")
    if not 0 <= carrier_local < value.shape[0] or not np.isfinite(value).all():
        raise ValueError("invalid carrier index or matrix values")
    hermitian = (value + value.conj().T) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    norm = float(np.linalg.norm(hermitian, ord=2))
    tolerance = 1e-10 * max(norm, 1e-30)
    maximum = float(values[-1])
    second = float(values[-2])
    if maximum <= tolerance:
        direction = np.eye(value.shape[0], dtype=np.complex128)[:, carrier_local]
        objective = float(np.vdot(direction, hermitian @ direction).real)
        return direction, {
            "identity": True,
            "identity_reason": "nonpositive_maximum_eigenvalue",
            "lambda_max": maximum,
            "lambda_second": second,
            "eigen_gap": maximum - second,
            "top_cluster_dimension": 0,
            "tolerance": tolerance,
            "objective": objective,
            "objective_gap_to_lambda_max": maximum - objective,
        }
    top = np.flatnonzero(values >= maximum - tolerance)
    basis = vectors[:, top]
    e = np.eye(value.shape[0], dtype=np.complex128)[:, carrier_local]
    projection = basis @ (basis.conj().T @ e)
    if np.linalg.norm(projection) > tolerance:
        direction = projection / np.linalg.norm(projection)
        selection = "carrier_projection_into_top_space"
    else:
        projector = basis @ basis.conj().T
        diagonal = projector.diagonal().real
        index = int(np.flatnonzero(diagonal >= diagonal.max() - tolerance)[0])
        direction = projector[:, index] / math.sqrt(max(float(diagonal[index]), 1e-300))
        selection = f"largest_top_projector_diagonal_column_{index}"
    direction = _fix_phase(direction, carrier_local, tolerance)
    direction = direction / np.linalg.norm(direction)
    objective = float(np.vdot(direction, hermitian @ direction).real)
    return np.ascontiguousarray(direction), {
        "identity": False,
        "identity_reason": None,
        "selection": selection,
        "lambda_max": maximum,
        "lambda_second": second,
        "eigen_gap": maximum - second,
        "top_cluster_dimension": int(top.size),
        "tolerance": tolerance,
        "objective": objective,
        "objective_gap_to_lambda_max": maximum - objective,
    }


@dataclass(frozen=True)
class Plane:
    a: float
    b: float
    v: np.ndarray
    u: np.ndarray
    carrier_local: int
    identity: bool

    def dense(self) -> np.ndarray:
        size = self.u.size
        identity = np.eye(size, dtype=np.complex128)
        if self.identity:
            return identity
        e = identity[:, self.carrier_local]
        return (
            identity
            + (self.a - 1.0) * (np.outer(e, e.conj()) + np.outer(self.v, self.v.conj()))
            + self.b * (np.outer(e, self.v.conj()) - np.outer(self.v, e.conj()))
        )


def minimal_plane(direction: Any, carrier_local: int, *, tolerance: float = 1e-13) -> Plane:
    u = np.asarray(direction, dtype=np.complex128).copy()
    if u.ndim != 1 or u.size < 2 or not 0 <= carrier_local < u.size or not np.isfinite(u).all():
        raise ValueError("invalid direction")
    norm = float(np.linalg.norm(u))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have positive norm")
    u /= norm
    u = _fix_phase(u, carrier_local, tolerance)
    a = float(np.clip(u[carrier_local].real, 0.0, 1.0))
    e = np.eye(u.size, dtype=np.complex128)[:, carrier_local]
    residual = u - a * e
    b = float(np.linalg.norm(residual))
    if b <= tolerance:
        u = e.copy()
        return Plane(a=1.0, b=0.0, v=np.zeros_like(u), u=u,
                     carrier_local=carrier_local, identity=True)
    v = residual / b
    v[carrier_local] = 0.0
    v /= np.linalg.norm(v)
    plane = Plane(a=a, b=math.sqrt(max(0.0, 1.0 - a * a)), v=v, u=u,
                  carrier_local=carrier_local, identity=False)
    dense = plane.dense()
    if np.linalg.norm(dense.conj().T @ dense - np.eye(u.size)) > 1e-10:
        raise RuntimeError("constructed plane is not unitary")
    if np.linalg.norm(dense @ u - e) > 1e-10:
        raise RuntimeError("constructed plane maps the wrong direction")
    return plane


def apply_plane(values: Any, plane: Plane) -> np.ndarray:
    z = np.asarray(values, dtype=np.complex128)
    if z.shape[-1] != plane.u.size:
        raise ValueError("plane dimension differs")
    if plane.identity:
        return z.copy()
    t = z[..., plane.carrier_local]
    r = np.einsum("i,...i->...", plane.v.conj(), z, optimize=True)
    output = z.copy()
    output[..., plane.carrier_local] += (plane.a - 1.0) * t + plane.b * r
    output += np.multiply.outer((plane.a - 1.0) * r - plane.b * t, plane.v)
    return output


def apply_group_planes_torch(
    x,
    active,
    carrier_local,
    a,
    b,
    vr,
    vi,
    head_to_group,
):
    """Apply grouped rank-2 maps to ``[...,heads,2K]`` split-half tensors."""
    import torch

    if x.ndim < 2 or x.shape[-1] % 2:
        raise ValueError("x must end in [heads,2K]")
    pairs = x.shape[-1] // 2
    active_t = torch.as_tensor(active, dtype=torch.long, device=x.device)
    head_map = torch.as_tensor(head_to_group, dtype=torch.long, device=x.device)
    if head_map.numel() != x.shape[-2] or active_t.numel() < 2:
        raise ValueError("head map or active set differs")
    a_t = torch.as_tensor(a, dtype=torch.float32, device=x.device)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=x.device)
    # Runtime alignment buffers already live on the model device.  Converting a
    # CUDA tensor through NumPy is invalid, so keep tensors on-device and only
    # use ``as_tensor`` for the CPU/NumPy reference path.
    vr_t = vr.to(device=x.device, dtype=torch.float32) if torch.is_tensor(vr) else torch.as_tensor(
        vr, dtype=torch.float32, device=x.device,
    )
    vi_t = vi.to(device=x.device, dtype=torch.float32) if torch.is_tensor(vi) else torch.as_tensor(
        vi, dtype=torch.float32, device=x.device,
    )
    if vr_t.shape != vi_t.shape or vr_t.shape != (a_t.numel(), active_t.numel()):
        raise ValueError("plane arrays differ")
    if not 0 <= int(carrier_local) < active_t.numel():
        raise ValueError("carrier local index differs")
    nonidentity = (b_t.abs() > 0) | ((a_t - 1.0).abs() > 0)
    if not bool(nonidentity.any()):
        return x
    output = x.clone()
    for head in range(x.shape[-2]):
        group = int(head_map[head])
        if group < 0 or group >= a_t.numel() or not bool(nonidentity[group]):
            continue
        real = x[..., head, :pairs].float().index_select(-1, active_t)
        imag = x[..., head, pairs:].float().index_select(-1, active_t)
        vreal, vimag = vr_t[group], vi_t[group]
        rreal = (real * vreal + imag * vimag).sum(-1)
        rimag = (imag * vreal - real * vimag).sum(-1)
        treal = real[..., int(carrier_local)]
        timag = imag[..., int(carrier_local)]
        aa, bb = a_t[group], b_t[group]
        sreal = (aa - 1.0) * rreal - bb * treal
        simag = (aa - 1.0) * rimag - bb * timag
        new_real = real + sreal[..., None] * vreal - simag[..., None] * vimag
        new_imag = imag + sreal[..., None] * vimag + simag[..., None] * vreal
        new_real[..., int(carrier_local)] += (aa - 1.0) * treal + bb * rreal
        new_imag[..., int(carrier_local)] += (aa - 1.0) * timag + bb * rimag
        out_real = output[..., head, :pairs]
        out_imag = output[..., head, pairs:]
        out_real.index_copy_(-1, active_t, new_real.to(dtype=x.dtype))
        out_imag.index_copy_(-1, active_t, new_imag.to(dtype=x.dtype))
    return output


def sample_causal_pairs(valid_positions: Any, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(valid_positions, dtype=np.int64)
    if positions.ndim != 1 or positions.size < 2 or len(np.unique(positions)) != positions.size:
        raise ValueError("valid positions must be a unique one-dimensional set")
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("count and seed must be nonnegative integers")
    rng = np.random.default_rng(seed)
    query = np.empty(count, dtype=np.int64)
    key = np.empty(count, dtype=np.int64)
    for index in range(count):
        pair = rng.choice(positions, size=2, replace=False)
        query[index] = max(int(pair[0]), int(pair[1]))
        key[index] = min(int(pair[0]), int(pair[1]))
    return query, key
