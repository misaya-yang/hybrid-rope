"""The operator-family method, including its FreqFold/PCA initialization.

The grouping/permutation in ``freqfold_rotation`` is adapted from TransMLA,
MIT (c) 2025 Fanxu Meng. See THIRD_PARTY_LICENSE.txt for the complete notice.
No upstream runtime or model implementation is imported.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class Shape:
    heads: int
    kv_heads: int
    head_dim: int
    content_rank: int
    rotary_dim: int
    theta: float = 1_000_000.0

    def __post_init__(self):
        if self.heads % self.kv_heads or self.head_dim % 2 or self.rotary_dim % 2:
            raise ValueError("GQA groups and rotary pairs must be integral")
        if min(self.heads, self.kv_heads, self.head_dim, self.content_rank, self.rotary_dim) <= 0:
            raise ValueError("positive dimensions required")

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def payload_width(self) -> int:
        return self.content_rank + self.rotary_dim


def split_to_pairs(x: Tensor) -> Tensor:
    """HF split-half layout to interleaved real/imaginary pairs."""
    return x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2).transpose(-1, -2).flatten(-2)


def rotate(x: Tensor, positions: Tensor, frequencies: Tensor) -> Tensor:
    """x is [tokens, heads, rotary_dim]; position/frequency math stays FP32+."""
    phase_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    phase = positions.to(phase_dtype)[:, None] * frequencies.to(phase_dtype)[None, :]
    cos, sin = phase.cos().to(x.dtype)[:, None, :], phase.sin().to(x.dtype)[:, None, :]
    real, imag = x[..., 0::2], x[..., 1::2]
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)


def native_frequencies(shape: Shape, *, device=None, dtype=torch.float32) -> Tensor:
    return shape.theta ** (-torch.arange(0, shape.head_dim, 2, device=device, dtype=dtype) / shape.head_dim)


def head_selection(shape: Shape, *, device=None, dtype=torch.float32) -> Tensor:
    """Maps the shared concatenation of native KV heads to each Q head."""
    eye = torch.eye(shape.kv_width, device=device, dtype=dtype)
    indices = torch.arange(shape.heads, device=device) // (shape.heads // shape.kv_heads)
    return eye.reshape(shape.kv_heads, shape.head_dim, shape.kv_width)[indices]


def principal_basis(samples: Tensor, rank: int) -> Tensor:
    """Uncentered second moment, as used by TransMLA's projection stage."""
    if not 0 < rank <= samples.shape[-1]:
        raise ValueError("PCA rank exceeds the joint representation width")
    matrix = samples.double().T @ samples.double()
    matrix += torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype) * (0.01 * matrix.diag().mean())
    _, basis = torch.linalg.eigh(matrix)
    return basis[:, -rank:].flip(-1).to(samples.dtype)


def freqfold_rotation(keys: Tensor, shape: Shape, fold: int) -> Tensor:
    """Return J such that transformed keys are k @ J.T, in upstream order."""
    h, d, width = shape.kv_heads, shape.head_dim, shape.kv_width
    if shape.rotary_dim > d or d % shape.rotary_dim:
        raise ValueError("FreqFold initialization needs rotary_dim to divide native head_dim")
    collapse = d // shape.rotary_dim
    if fold < collapse or fold % collapse or (d // 2) % fold:
        raise ValueError("fold must divide native pair count and be a multiple of collapse")
    z = keys.reshape(-1, h, 2, d // 2 // fold, fold // collapse, collapse)
    z = z.permute(0, 2, 5, 1, 4, 3).reshape(-1, h * fold, d // 2 // fold)
    bases = []
    for index in range(d // 2 // fold):
        bases.append(principal_basis(z[:, :, index], h * fold))
    u = torch.stack(bases + bases)
    weight = torch.eye(width, dtype=keys.dtype, device=keys.device)
    weight = weight.reshape(h, d // fold, fold // collapse, collapse, width)
    weight = weight.permute(3, 0, 2, 1, 4).reshape(h * fold, d // fold, width)
    weight = torch.einsum("dhc,hdw->cdw", u, weight)
    weight = weight.reshape(collapse, h, d // fold // 2, fold // collapse, 2, width)
    return weight.permute(0, 1, 4, 2, 3, 5).reshape(width, width)


class OperatorFactors(nn.Module):
    """q_R=q A, k_R=k B, c=[k,v] C, q_C=q P, values=c U.

    Frequencies are represented as phase / phase_scale for stable optimization.
    Matrices are real; the evolution uses ordinary unitary 2x2 rotation blocks.
    """
    def __init__(self, shape: Shape, phase_scale: float = 16384.0):
        super().__init__()
        self.shape, self.phase_scale = shape, phase_scale
        h, d, w, r, p = shape.heads, shape.head_dim, shape.kv_width, shape.content_rank, shape.rotary_dim
        self.A = nn.Parameter(torch.zeros(h, d, p))
        self.B = nn.Parameter(torch.zeros(w, p))
        self.C = nn.Parameter(torch.zeros(2 * w, r))
        self.P = nn.Parameter(torch.zeros(h, d, r))
        self.U = nn.Parameter(torch.zeros(h, r, d))
        self.phase = nn.Parameter(torch.zeros(p // 2))

    @property
    def frequencies(self) -> Tensor:
        return self.phase / self.phase_scale

    def metadata(self) -> dict[str, Any]:
        return {"shape": asdict(self.shape), "phase_scale": self.phase_scale}

    @classmethod
    def from_freqfold(cls, keys: Tensor, values: Tensor, shape: Shape, fold: int | None = None):
        """Initialize one candidate. This is not a competing-method sweep."""
        if shape.content_rank > 2 * shape.kv_width - shape.rotary_dim:
            raise ValueError("content rank exceeds remaining K plus V width")
        fold = fold or shape.head_dim // shape.rotary_dim
        result = cls(shape).to(device=keys.device, dtype=keys.dtype)
        with torch.no_grad():
            j = freqfold_rotation(keys, shape, fold)
            selection = head_selection(shape, device=keys.device, dtype=keys.dtype)
            p, w = shape.rotary_dim, shape.kv_width
            # No balancing is applied here: identical to upstream balance=None.
            remainder = keys @ j[p:].T
            basis = principal_basis(torch.cat((remainder, values), dim=-1), shape.content_rank)
            result.B.copy_(j[:p].T)
            result.A.copy_(selection @ result.B)
            transform = torch.zeros(2 * w - p, 2 * w, device=keys.device, dtype=keys.dtype)
            transform[:w - p, :w] = j[p:]
            transform[w - p:, w:] = torch.eye(w, device=keys.device, dtype=keys.dtype)
            result.C.copy_(transform.T @ basis)
            result.P.copy_((selection @ j[p:].T) @ basis[:w - p])
            result.U.copy_(basis[w - p:].T.unsqueeze(0) @ selection.transpose(-1, -2))
            collapse = shape.head_dim // p
            result.phase.copy_(native_frequencies(shape, device=keys.device, dtype=keys.dtype)[::collapse] * result.phase_scale)
        return result

    @classmethod
    def identity(cls, heads: int, kv_heads: int, head_dim: int, theta=1_000_000.0):
        """Exact full native GQA represented as shared compact factors; parity only."""
        width = kv_heads * head_dim
        shape = Shape(heads, kv_heads, head_dim, width, width, theta)
        result = cls(shape)
        with torch.no_grad():
            j = split_to_pairs(torch.eye(width).reshape(width, kv_heads, head_dim)).reshape(width, width).T
            selection = head_selection(shape)
            result.B.copy_(j.T)
            result.A.copy_(selection @ j.T)
            result.C[width:].copy_(torch.eye(width))
            result.U.copy_(selection.transpose(-1, -2))
            result.phase.copy_(native_frequencies(shape).repeat(kv_heads) * result.phase_scale)
        return result

    def response(self, q: Tensor, k: Tensor, v: Tensor, query_positions: Tensor, key_positions: Tensor, valid: Tensor | None = None):
        """Full causal row response. q [M,H,D], k/v [N,KV*D]."""
        content = torch.cat((k, v), dim=-1) @ self.C
        q_c = torch.einsum("mhd,hdr->mhr", q, self.P)
        q_r = rotate(torch.einsum("mhd,hdp->mhp", q, self.A), query_positions, self.frequencies)
        k_r = rotate((k @ self.B)[:, None, :], key_positions, self.frequencies)[:, 0]
        scores = (torch.einsum("mhr,nr->hmn", q_c, content) + torch.einsum("mhp,np->hmn", q_r, k_r)) / math.sqrt(self.shape.head_dim)
        valid = key_positions[None, :] <= query_positions[:, None] if valid is None else valid
        probabilities = scores.masked_fill(~valid[None], float("-inf")).softmax(-1)
        projected_values = torch.einsum("nr,hrd->hnd", content, self.U)
        output = probabilities @ projected_values
        return scores, output, projected_values, valid


def native_response(q: Tensor, k: Tensor, v: Tensor, query_positions: Tensor, key_positions: Tensor, shape: Shape, valid: Tensor | None = None):
    frequencies = native_frequencies(shape, device=q.device, dtype=q.dtype)
    qr = rotate(split_to_pairs(q), query_positions, frequencies)
    kr = rotate(split_to_pairs(k.reshape(-1, shape.kv_heads, shape.head_dim)), key_positions, frequencies)
    groups = torch.arange(shape.heads, device=q.device) // (shape.heads // shape.kv_heads)
    scores = torch.einsum("mhd,nhd->hmn", qr, kr[:, groups]) / math.sqrt(shape.head_dim)
    valid = key_positions[None, :] <= query_positions[:, None] if valid is None else valid
    probability = scores.masked_fill(~valid[None], float("-inf")).softmax(-1)
    values = v.reshape(-1, shape.kv_heads, shape.head_dim)[:, groups].transpose(0, 1)
    return scores, probability @ values, values, valid
