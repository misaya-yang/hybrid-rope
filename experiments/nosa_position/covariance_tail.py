"""Choose one exact key to reduce PC2's measured covariance omission.

This is a candidate, not a demonstrated language-model improvement. It has the
same stored tensors and scoring rule as tail_pair; only the prefix-only choice
of the exact key changes. A block keeps plain PC2 when extraction does not
improve the criterion, except for a zero-cross-covariance tail that leaves it
exactly unchanged. Building the criterion uses a B-by-B centered-key Gram
matrix, with no eigendecomposition and no future query or answer.

Let E = Sigma - pairdiag(Sigma), r_i = k_i - mu, w_i = softmax(CIS)_i,
and a_i = w_i/(1-w_i). Exact extraction of i leaves covariance error
E - offpair(a_i r_i r_i^T). Its squared Frobenius loss reduction is
2 a_i r_i^T E r_i - a_i^2 ||offpair(r_i r_i^T)||_F^2.
We evaluate that reduction and compare with zero (no extraction). This is an
exact covariance identity, not a bound on finite-query logmass or generation.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import torch

from . import run as base
from . import exact_probe, tail_pair
from .selector_controls import _concat_summary, build_summary
from .tail_pair import TailPairSelector, tail_logmass


MODE = "pc2_covariance_tail1"


def extraction_gains(keys, cis):
    """Full split-half rotary pairs; returns gains, index, and activation mask.

    Near-singleton CIS distributions skip ill-conditioned exclusions. A small
    arithmetic tolerance sends unresolved zero-gain blocks to unchanged PC2.
    In exact arithmetic, the stated minimization is over all valid exclusions
    plus the no-extraction option. GPU costs still need measurement.
    """
    if keys.ndim != 4 or cis.shape != keys.shape[:-1] or keys.shape[-2] < 2 or keys.shape[-1] % 2:
        raise ValueError("expected [KV,blocks,B>=2,even full-rotary D] and CIS")
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    x, weights = keys.to(dtype), cis.to(dtype).softmax(-1)
    mean = (weights[..., None] * x).sum(-2)
    centered = x - mean[..., None, :]
    gram = centered @ centered.transpose(-1, -2)
    full_quad = (gram.square() * weights[..., None, :]).sum(-1)
    left, right = centered.chunk(2, dim=-1)
    w = weights[..., None]
    xx = (w * left.square()).sum(-2)
    yy = (w * right.square()).sum(-2)
    xy = (w * left * right).sum(-2)
    pair_quad = (left.square() * xx[..., None, :]
                 + 2 * left * right * xy[..., None, :]
                 + right.square() * yy[..., None, :]).sum(-1)
    pair_energy = left.square() + right.square()
    off_outer_norm2 = (pair_energy.sum(-1).square() - pair_energy.square().sum(-1)).clamp_min(0)
    eps = torch.finfo(dtype).eps
    remainder = 1 - weights
    alpha = weights / remainder.clamp_min(64 * eps)
    cross_term = 2 * alpha * (full_quad - pair_quad)
    self_term = alpha.square() * off_outer_norm2
    gain = (cross_term - self_term).masked_fill(remainder <= 64 * eps, -torch.inf)
    tolerance = 64 * eps * (cross_term.abs() + self_term.abs() + 1)
    index = gain.argmax(-1)
    best = gain.gather(-1, index[..., None]).squeeze(-1)
    margin = tolerance.gather(-1, index[..., None]).squeeze(-1)
    active = torch.isfinite(best) & (best > margin)
    # A residual confined to one rotary pair has offpair(r r^T)=0: its exact
    # extraction changes none of the covariance error. Preserve the known
    # single-pair rare-key repair in this exact tie, choosing greatest radius.
    zero_cross = (off_outer_norm2 == 0) & (remainder > 64 * eps)
    safe_index = pair_energy.sum(-1).masked_fill(~zero_cross, -torch.inf).argmax(-1)
    use_safe_tie = (~active) & zero_cross.any(-1)
    index = torch.where(use_safe_tie, safe_index, index)
    active = active | use_safe_tie
    return gain, index, active


def build_covariance_tail_summary(keys, cis, *, storage_dtype=None):
    _, index, active = extraction_gains(keys, cis)
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    x, bias = keys.to(dtype), cis.to(dtype)
    tail_key = x.gather(-2, index[..., None, None].expand(*index.shape, 1, x.shape[-1])).squeeze(-2)
    tail_bias = bias.gather(-1, index[..., None]).squeeze(-1).masked_fill(~active, -torch.inf)
    remove = torch.zeros_like(bias, dtype=torch.bool).scatter_(-1, index[..., None], active[..., None])
    summary = build_summary(x, bias.masked_fill(remove, -torch.inf), "pc2", storage_dtype=storage_dtype)
    summary.extra.update(tail_key=tail_key, tail_bias=tail_bias)
    return summary


class CovarianceTailSelector(TailPairSelector):
    def __init__(self, mode=MODE, **kwargs):
        self.covariance_tail_enabled = mode == MODE
        super().__init__(tail_pair.MODE if self.covariance_tail_enabled else mode, **kwargs)

    @torch.no_grad()
    def logmass(self, context):
        if not self.covariance_tail_enabled:
            return super().logmass(context)
        # Match tail_pair's unchanged cache/current-block handling. Kept local
        # so the running owner's implementation and controls are never patched.
        q, k, cis, settings = context.q, context.k, context.cis, context.settings
        kvh, length, dim = k.shape
        group, queries = q.shape[0] // kvh, q.shape[1]
        query = q.reshape(kvh, group, queries, dim).float() / math.sqrt(dim)
        count, full = math.ceil(length / settings.block_size), length // settings.block_size
        if int(context.query_positions[0]) == 0:
            self.cache.pop(context.layer_idx, None)
        previous = self.cache.get(context.layer_idx)
        done = previous.blocks if previous else 0
        if done > full:
            raise ValueError("cache sequence shrank without reset")
        if full > done:
            begin, end = done * settings.block_size, full * settings.block_size
            started = time.perf_counter()
            added = build_covariance_tail_summary(
                k[:, begin:end].reshape(kvh, full - done, settings.block_size, dim),
                cis[:, begin:end].reshape(kvh, full - done, settings.block_size),
                storage_dtype=self.storage_dtype)
            previous = _concat_summary(previous, added)
            self.cache[context.layer_idx] = previous
            self.metrics["summary_build_calls"] += 1
            self.metrics["summary_build_wall_seconds"] += time.perf_counter() - started
        result = query.new_full((kvh, group, queries, count), -torch.inf)
        if previous is not None and full:
            result[..., :full] = tail_logmass(query, previous)
        endpoints = (torch.arange(count, device=q.device) + 1) * settings.block_size - 1
        result.masked_fill_(~(endpoints[None] <= context.query_positions[:, None])[None, None], -torch.inf)
        current = context.query_positions // settings.block_size
        for block in current.unique().tolist():
            rows = (current == block).nonzero(as_tuple=True)[0]
            begin, end = block * settings.block_size, min((block + 1) * settings.block_size, length)
            logits = torch.einsum("hgqd,htd->hgqt", query[:, :, rows], k[:, begin:end].float())
            logits += cis[:, None, None, begin:end].float()
            visible = torch.arange(begin, end, device=q.device)[None] <= context.query_positions[rows, None]
            result[:, :, rows, block] = logits.masked_fill(~visible[None, None], -torch.inf).logsumexp(-1)
            self.metrics["exact_raw_key_scores"] += q.shape[0] * len(rows) * (end - begin)
        self.metrics["max_metadata_bytes"] = max(
            self.metrics["max_metadata_bytes"], sum(value.nbytes() for value in self.cache.values()))
        return result


def main():
    original_hashes, original_modes, original_factory = base.source_hashes, base.MODES, base.BlockSummarySelector
    def source_hashes():
        return {**original_hashes(), **{Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, tail_pair.__file__, exact_probe.__file__)}}
    try:
        base.source_hashes = source_hashes
        base.MODES = (*original_modes, tail_pair.MODE, MODE)
        base.BlockSummarySelector = CovarianceTailSelector
        base.main()
    finally:
        base.source_hashes, base.MODES, base.BlockSummarySelector = original_hashes, original_modes, original_factory


if __name__ == "__main__":
    main()
