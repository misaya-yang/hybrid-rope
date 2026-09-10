"""CPU reference for sparse reads conditioned on earlier read addresses.

This is a candidate composition of known addressing and mixture operations, not
a new RoPE theorem or a validated language-model method. Keys, values, and sparse
support remain fixed. It also supports a native NoPE reader because the optional
relative-position bias is additive and initialized to zero.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class References:
    positions: torch.Tensor
    masses: torch.Tensor
    residual_mass: torch.Tensor


def phases(positions, inv_freq):
    # Addresses/frequencies here are fixed metadata. Reducing FP64 phases before
    # casting prevents a large common origin from corrupting a small offset.
    # This cost is real; a production implementation may cache shared features.
    phase = positions.to(torch.float64)[..., None] * inv_freq.to(torch.float64)
    return torch.remainder(phase, 2 * math.pi).to(inv_freq.dtype)


def retain_references(positions, attention, count, query_position):
    """Keep actual writer probability mass; omitted mass returns to native read.

    A sink probability can be absent from `attention`, so its sum may be < 1.
    Positions are original logical addresses, never ranks in a selected subset.
    This single-row reference intentionally does not implement a model hook.
    """
    if positions.ndim != 1 or attention.shape != positions.shape:
        raise ValueError("one writer attention row and aligned positions required")
    if count < 1 or not bool(torch.isfinite(attention).all()):
        raise ValueError("positive count and finite attention required")
    if bool((attention < 0).any()) or attention.sum().item() > 1 + 1e-6:
        raise ValueError("writer masses must form a sub-probability distribution")
    if bool((positions > query_position).any()) or bool((positions < 0).any()):
        raise ValueError("references must be observed causal addresses")
    if positions.unique().numel() != positions.numel():
        raise ValueError("writer addresses must not be duplicated")
    # Stable ties by original address, independent of sparse gather order.
    address_order = positions.argsort(stable=True)
    mass_order = attention[address_order].argsort(descending=True, stable=True)
    selected = address_order[mass_order[:count]]
    masses = attention[selected]
    return References(positions[selected], masses, (1 - masses.sum()).clamp_min(0))


def relative_features(key_positions, reference_positions, inv_freq):
    """[reference, selected key, cosine/sine channel], using immutable positions."""
    delta = key_positions[None, :] - reference_positions[:, None]
    phase = phases(delta, inv_freq)
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def position_features(positions, inv_freq):
    """Immutable features that can be shared across layers with the same basis."""
    phase = phases(positions, inv_freq)
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def factored_bias(coefficients, key_features, reference_positions, inv_freq):
    """Compute u^T phi(p-a) without materializing [R,S,2F] features.

    This is ordinary Fourier translation, not a novel representation theorem.
    The native content QK scores are computed once outside this helper.
    """
    uc, us = coefficients.chunk(2, dim=-1)
    phase = phases(reference_positions, inv_freq)
    cosine, sine = phase.cos(), phase.sin()
    shifted = torch.cat((uc*cosine-us*sine, uc*sine+us*cosine), dim=-1)
    return shifted @ key_features.T


def normalized_mixture(native_logits, reference_logits, values, masses,
                       sink_logit=None, return_delta=False):
    """Mix individually normalized reads over exactly the same visible keys.

    native_logits: [S], reference_logits: [R,S], values: [S,D], masses: [R].
    All remaining mass uses the native read, without renormalizing references.
    A sink, when supplied, has zero value and its native logit in every branch.
    Returns the output and key probabilities; omitted sink mass explains a sum
    below one. Dense diagnostic only: this implementation is not a fast kernel.
    """
    if native_logits.ndim != 1 or reference_logits.shape != (masses.numel(), native_logits.numel()):
        raise ValueError("branch logits and masses do not align")
    if values.ndim != 2 or values.shape[0] != native_logits.numel():
        raise ValueError("all branches must read the same selected value rows")
    if bool((masses < 0).any()) or masses.sum().item() > 1 + 1e-6:
        raise ValueError("reference masses must sum to at most one")
    if not bool(torch.isfinite(masses).all()):
        raise ValueError("reference masses must be finite")
    if not torch.equal(torch.isneginf(reference_logits), torch.isneginf(native_logits)[None].expand_as(reference_logits)):
        raise ValueError("reference branches may not change the sparse/causal support")
    logits = torch.cat((native_logits[None], reference_logits), dim=0)
    if sink_logit is not None:
        sink = torch.as_tensor(sink_logit, device=logits.device, dtype=logits.dtype)
        logits = torch.cat((logits, sink.expand(logits.shape[0], 1)), dim=-1)
    probs = logits.softmax(-1)
    if not bool(torch.isfinite(probs).all()):
        raise ValueError("every branch needs finite probability mass")
    weights = torch.cat(((1 - masses.sum()).clamp_min(0).reshape(1), masses))
    # Accumulate around the native read: zero bias is exactly native in this
    # reference arithmetic, not just mathematically native after re-summation.
    delta = (weights[1:, None] * (probs[1:] - probs[0])).sum(0)
    mixture = probs[0] + delta
    key_probs = mixture[:native_logits.numel()]
    value_weights = delta[:native_logits.numel()] if return_delta else key_probs
    return value_weights @ values, key_probs


class ReferenceBias(nn.Module):
    """One-row candidate adapter; zero initialization preserves native attention.

    No pretrained Q/K is reinterpreted as already expressed in a new frame.
    The frequency basis is supplied explicitly and is shared with matched
    controls. It is not optimized or claimed to be a novel frequency allocation.
    """
    def __init__(self, hidden_size, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone())
        self.projection = nn.Linear(hidden_size, 2 * inv_freq.numel(), bias=False,
                                    dtype=inv_freq.dtype, device=inv_freq.device)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden, native_logits, values, key_positions, references,
                sink_logit=None, pooling="arithmetic", return_delta=False):
        key_features = position_features(key_positions, self.inv_freq)
        coefficients = self.projection(hidden) / math.sqrt(key_features.shape[-1])
        bias = factored_bias(coefficients, key_features, references.positions, self.inv_freq)
        if pooling == "geometric":
            # Same parameters, references and native residual mass. The only
            # changed operation is averaging bias before normalization.
            logits = native_logits + (references.masses[:, None]*bias).sum(0)
            unit = torch.ones(1, dtype=logits.dtype, device=logits.device)
            return normalized_mixture(native_logits, logits[None], values, unit,
                                      sink_logit, return_delta)
        if pooling != "arithmetic":
            raise ValueError("unknown read pooling")
        return normalized_mixture(native_logits, native_logits[None] + bias,
                                  values, references.masses, sink_logit, return_delta)
