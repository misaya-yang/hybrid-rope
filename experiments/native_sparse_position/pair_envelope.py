"""Cached rotary-pair extrema and Quest controls; no reader or PE-table changes.

Keys are [KV, NB, B, D], queries [H, D] already include attention scaling.
Native split-half pairs are (i, i + K) in the first 2*K dimensions. GQA query
heads are contiguous groups of H/KV. Builders accept NumPy arrays or tensors.
All stored floating tensors are FP32. Construction uses FP64 temporaries to
round enclosing widths outwards; scores use FP32 elementwise operations with
conservative roundoff padding, not tensor-core matrix multiplication.

The mathematical descriptor bounds max-token logits, NOT block log mass.
Upper bounds and common-rotation equivariance do not imply better retrieval or
answers than Quest. No CUDA execution is required by this module.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


RANDOM_PAIR_SEED = 20260908


def _as_keys(keys):
    original = torch.as_tensor(keys)
    if original.ndim != 4 or min(original.shape) < 1:
        raise ValueError("keys must have nonempty shape [KV, NB, B, D]")
    if not original.is_floating_point():
        raise ValueError("keys must be floating point")
    source_bytes = original.numel() * original.element_size()
    keys = original.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(keys).all()):
        raise ValueError("keys must be finite when represented in FP32")
    return keys, source_bytes


def _principal_axis_2x2(cov):
    """Analytic symmetric 2x2 eigensystem; no cuSOLVER batched-eigh call.

    Eigenvalues are ascending. The principal direction is defined up to sign.
    At a repeated eigenvalue atan2(0,0) returns an arbitrary axis; the builder
    replaces it with its rotation-invariant disk descriptor.
    """
    a, c = cov[..., 0, 0], cov[..., 1, 1]
    b = (cov[..., 0, 1] + cov[..., 1, 0]) * 0.5
    trace = a + c
    gap = torch.hypot(a - c, 2.0 * b)
    angle = 0.5 * torch.atan2(2.0 * b, a - c)
    values = torch.stack(((trace - gap) * 0.5, (trace + gap) * 0.5), dim=-1)
    principal = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    return values, principal


def _outward_width(x):
    """Upper-round a nonnegative FP64 width, allowing FP64 construction error."""
    inflated = x * (1.0 + 16.0 * torch.finfo(torch.float64).eps)
    result = inflated.to(torch.float32)
    result = torch.where(
        x == 0, result,
        torch.nextafter(result, torch.full_like(result, torch.inf)),
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("enclosing widths exceed finite FP32 range")
    return result


def _query(q, kv, dim, device):
    q = torch.as_tensor(q, device=device).detach().to(dtype=torch.float32)
    if q.ndim != 2 or q.shape[1] != dim or q.shape[0] < kv or q.shape[0] % kv:
        raise ValueError("q must be [H,D], with H a positive multiple of KV")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("q must be finite when represented in FP32")
    return q.reshape(kv, q.shape[0] // kv, dim)


def _padded_upper(value, magnitude, dim):
    # Dot products are length two, followed by at most D scalar accumulations.
    # Magnitude bounds all intermediate absolute terms, avoiding cancellation
    # based padding. The pair contribution to it is rotation invariant in R.
    factor = 8.0 * (dim + 8) * torch.finfo(torch.float32).eps
    upper = value + factor * magnitude
    return torch.nextafter(upper, torch.full_like(upper, torch.inf))


def _byte_info(tensors, *, source_bytes, kv, blocks, block_size, dim, method):
    by_tensor = {name: t.numel() * t.element_size() for name, t in tensors.items()}
    descriptor = sum(v for name, v in by_tensor.items() if name != "pair_indices")
    total = sum(by_tensor.values())
    return {
        "method": method,
        "cache_dtype": "float32",
        "tensor_bytes": by_tensor,
        "descriptor_bytes": descriptor,
        "shared_pair_index_bytes": by_tensor.get("pair_indices", 0),
        "total_tensor_bytes": total,
        "source_key_bytes": source_bytes,
        "cache_to_source_key_ratio": total / source_bytes,
        "bytes_per_kv_block": descriptor / (kv * blocks),
        "shape": {"kv_heads": kv, "blocks": blocks, "block_size": block_size, "dim": dim},
        "excludes": "original reader KV, build temporaries, Python object overhead",
    }


@dataclass(frozen=True)
class PairEnvelopeCache:
    center: torch.Tensor       # [KV, NB, K, 2]
    axis: torch.Tensor         # [KV, NB, K, 2]; second axis is J axis
    halfwidth: torch.Tensor    # [KV, NB, K, 2]; disk radius in slot zero
    isotropic: torch.Tensor    # [KV, NB, K], bool
    nonrot_min: torch.Tensor   # [KV, NB, D-2K]
    nonrot_max: torch.Tensor
    pair_indices: torch.Tensor # [K, 2], int64, shared by all blocks/heads
    dim: int
    block_size: int
    source_key_bytes: int
    pairing: str
    seed: int
    eigengap_rtol: float

    def byte_info(self):
        names = ("center", "axis", "halfwidth", "isotropic", "nonrot_min", "nonrot_max", "pair_indices")
        return _byte_info(
            {name: getattr(self, name) for name in names},
            source_bytes=self.source_key_bytes, kv=self.center.shape[0],
            blocks=self.center.shape[1], block_size=self.block_size,
            dim=self.dim, method="RPEE-" + self.pairing,
        )


@torch.no_grad()
def build_pair_envelope(keys, omega=None, *, K: Optional[int] = None,
                        pairing="native", seed=RANDOM_PAIR_SEED,
                        eigengap_rtol=1e-6):
    """Build native or fixed-seed random-pair extrema; no query is consulted.

    Specify omega (only its length is needed) and/or K. Frequencies themselves
    are not refitted or used to unrotate keys. Numerical isotropy switches to a
    centered disk and avoids choosing arbitrary eigenvectors at a repeated root.
    """
    keys, source_bytes = _as_keys(keys)
    kv, nb, block, dim = keys.shape
    if omega is not None:
        omega = torch.as_tensor(omega)
        if omega.ndim != 1 or not bool(torch.isfinite(omega).all()):
            raise ValueError("omega must be a finite one-dimensional frequency array")
        if K is not None and K != omega.numel():
            raise ValueError("K and len(omega) disagree")
        K = omega.numel()
    if K is None or not isinstance(K, int) or not 0 <= 2 * K <= dim:
        raise ValueError("provide K or omega, with 0 <= 2*K <= D")
    if pairing not in ("native", "random"):
        raise ValueError("pairing must be native or random")
    if not 0 <= eigengap_rtol < 1:
        raise ValueError("eigengap_rtol must be in [0,1)")
    if pairing == "native":
        indices = np.stack((np.arange(K), np.arange(K) + K), axis=-1)
    else:
        indices = np.random.default_rng(seed).permutation(2 * K).reshape(K, 2)
    indices = torch.as_tensor(indices, dtype=torch.int64, device=keys.device)
    nonrot_min = keys[..., 2*K:].amin(dim=2)
    nonrot_max = keys[..., 2*K:].amax(dim=2)
    if K == 0:
        center = keys.new_empty(kv, nb, 0, 2)
        return PairEnvelopeCache(center, center.clone(), center.clone(),
            torch.empty(kv, nb, 0, dtype=torch.bool, device=keys.device),
            nonrot_min, nonrot_max, indices, dim, block, source_bytes,
            pairing, int(seed), float(eigengap_rtol))

    # FP64 build work makes containment depend on the stored FP32 axes/center,
    # rather than assuming an FP32 eigenvector has exactly unit length.
    x = keys[..., indices].permute(0, 1, 3, 2, 4).double()  # KV,NB,K,B,2
    mean = x.mean(dim=-2)
    centered = x - mean.unsqueeze(-2)
    cov = (centered.unsqueeze(-1) * centered.unsqueeze(-2)).mean(dim=-3)
    eigenvalues, principal = _principal_axis_2x2(cov)
    gap = eigenvalues[..., 1] - eigenvalues[..., 0]
    trace = eigenvalues.sum(dim=-1).clamp_min(0)
    isotropic = (trace == 0) | (gap <= eigengap_rtol * trace)
    axis = principal.float()
    fixed_axis = torch.zeros_like(axis)
    fixed_axis[..., 0] = 1
    axis = torch.where(isotropic.unsqueeze(-1), fixed_axis, axis)
    u = axis.double()
    v = torch.stack((-u[..., 1], u[..., 0]), dim=-1)
    basis = torch.stack((u, v), dim=-1)
    norm2 = u.square().sum(dim=-1)
    origin = mean.float().double()
    projected = torch.einsum("...bi,...ij->...bj", x - origin.unsqueeze(-2), basis)
    projected = projected / norm2[..., None, None]
    midpoint = (projected.amin(dim=-2) + projected.amax(dim=-2)) / 2
    rectangle_center = origin + (basis * midpoint.unsqueeze(-2)).sum(dim=-1)
    center = torch.where(isotropic.unsqueeze(-1), origin, rectangle_center).float()
    # Recompute extrema about the rounded center; the stored basis is exact
    # input to this calculation. Both axis signs describe the same envelope.
    delta = x - center.double().unsqueeze(-2)
    projected = torch.einsum("...bi,...ij->...bj", delta, basis) / norm2[..., None, None]
    widths = _outward_width(projected.abs().amax(dim=-2))
    radius = _outward_width(torch.linalg.vector_norm(delta, dim=-1).amax(dim=-1))
    disk_widths = torch.stack((radius, torch.zeros_like(radius)), dim=-1)
    widths = torch.where(isotropic.unsqueeze(-1), disk_widths, widths)
    return PairEnvelopeCache(center, axis, widths, isotropic,
        nonrot_min, nonrot_max, indices, dim, block, source_bytes,
        pairing, int(seed), float(eigengap_rtol))


@torch.no_grad()
def score_pair_envelope(q, cache):
    """Return (FP32 upper bounds [H,NB], exact cache tensor-byte metadata)."""
    kv, nb, K, _ = cache.center.shape
    query = _query(q, kv, cache.dim, cache.center.device)
    groups = query.shape[1]
    value = query.new_zeros(kv, groups, nb)
    magnitude = torch.zeros_like(value)
    if K:
        qp = query[..., cache.pair_indices].unsqueeze(2)  # KV,G,1,K,2
        c = cache.center.unsqueeze(1)
        u = cache.axis.unsqueeze(1)
        v = torch.stack((-u[..., 1], u[..., 0]), dim=-1)
        a, b = cache.halfwidth.unbind(dim=-1)
        a, b = a.unsqueeze(1), b.unsqueeze(1)
        center_term = (qp * c).sum(dim=-1)
        qu = (qp * u).sum(dim=-1)
        qv = (qp * v).sum(dim=-1)
        qnorm = torch.linalg.vector_norm(qp, dim=-1)
        rectangle = a * qu.abs() + b * qv.abs()
        disk = a * qnorm
        spread = torch.where(cache.isotropic.unsqueeze(1), disk, rectangle)
        value += (center_term + spread).sum(dim=-1)
        # Cauchy bounds intermediate terms and respects pair rotations.
        rectangle_radius = (a + b) * torch.linalg.vector_norm(u, dim=-1)
        extent = torch.where(cache.isotropic.unsqueeze(1), a, rectangle_radius)
        magnitude += (qnorm * (torch.linalg.vector_norm(c, dim=-1) + extent)).sum(dim=-1)
    if 2*K < cache.dim:
        qn = query[..., 2*K:].unsqueeze(2)
        lo, hi = cache.nonrot_min.unsqueeze(1), cache.nonrot_max.unsqueeze(1)
        value += torch.maximum(qn * lo, qn * hi).sum(dim=-1)
        magnitude += (qn.abs() * torch.maximum(lo.abs(), hi.abs())).sum(dim=-1)
    upper = _padded_upper(value, magnitude, cache.dim).reshape(kv * groups, nb)
    return upper, cache.byte_info()


@dataclass(frozen=True)
class QuestCache:
    minimum: torch.Tensor
    maximum: torch.Tensor
    block_size: int
    source_key_bytes: int

    def byte_info(self):
        kv, nb, dim = self.minimum.shape
        return _byte_info({"minimum": self.minimum, "maximum": self.maximum},
            source_bytes=self.source_key_bytes, kv=kv, blocks=nb,
            block_size=self.block_size, dim=dim, method="Quest")


@torch.no_grad()
def build_quest(keys):
    keys, source_bytes = _as_keys(keys)
    return QuestCache(keys.amin(dim=2), keys.amax(dim=2), keys.shape[2], source_bytes)


@torch.no_grad()
def score_quest(q, cache):
    """Original axis-aligned Quest score, with the same FP32 padding policy."""
    kv, nb, dim = cache.minimum.shape
    query = _query(q, kv, dim, cache.minimum.device).unsqueeze(2)
    lo, hi = cache.minimum.unsqueeze(1), cache.maximum.unsqueeze(1)
    value = torch.maximum(query * lo, query * hi).sum(dim=-1)
    magnitude = (query.abs() * torch.maximum(lo.abs(), hi.abs())).sum(dim=-1)
    upper = _padded_upper(value, magnitude, dim).reshape(-1, nb)
    return upper, cache.byte_info()
