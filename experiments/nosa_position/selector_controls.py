"""Cacheable block selectors for the NOSA position experiment.

PC2 is an unvalidated, RoPE-pair covariance approximation, NOT a new cumulant
expansion. COBS-inspired low-rank and split-mean controls use the identical
64-token blocks, CIS exponential tilt, GQA normalization, and NOSA quotas.
Every mode retains raw KV and the pretrained reader. No cache-reduction or
production-kernel speed claim follows from this reference implementation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .runtime import SelectionContext, select_with_scores


MODES = ("weighted_mean", "pc2", "pc2_unweighted", "pc2_rank1", "cobs_rank1", "cobs_rank2", "split2", "split4", "quest", "exact_mass")


@dataclass
class Summary:
    mean: Tensor
    log_weight: Tensor
    extra: dict[str, Tensor]
    blocks: int

    def nbytes(self):
        # Count backing storage, not only view numel: even a zero-length view
        # can retain a nonempty allocation. Deduplicate shared backing storage.
        storage = {}
        for value in (self.mean, self.log_weight, *self.extra.values()):
            allocation = value.untyped_storage()
            storage[(str(value.device), allocation.data_ptr())] = allocation.nbytes()
        return sum(storage.values())


def build_summary(keys: Tensor, cis: Tensor, mode: str, *, rotary_dim: int | None = None,
                  storage_dtype: torch.dtype | None = None) -> Summary:
    """keys [KV, blocks, B, D]; all tokens in each block are already visible.

    exp(CIS) is incorporated exactly as a reference measure: log Z_cis plus
    log E_{softmax(CIS)} exp(q.k/sqrt(D)). It must not be added a second time.
    Covariances use the population convention. All accumulation is at least
    float32; float64 input remains float64 for independent numerical checks.
    """
    if mode not in MODES or mode == "exact_mass":
        raise ValueError(f"unsupported cached mode {mode}")
    if keys.ndim != 4 or cis.shape != keys.shape[:-1] or keys.shape[-2] < 1:
        raise ValueError("keys [KV,blocks,B,D] and CIS [KV,blocks,B] are required")
    dim, size = keys.shape[-1], keys.shape[-2]
    rd = dim if rotary_dim is None else rotary_dim
    if rd < 0 or rd > dim or rd % 2:
        raise ValueError("rotary dimension must be even and within head dimension")
    if mode == "pc2_unweighted":
        mode, cis = "pc2", torch.zeros_like(cis)
    acc = torch.float64 if keys.dtype == torch.float64 else torch.float32
    x, b = keys.to(acc), cis.to(acc)
    # Split/extrema controls retain and compute only their own statistics.
    mean = x.new_empty((*x.shape[:2], 0))
    log_weight = b.new_empty((b.shape[0], 0))
    if mode in ("weighted_mean", "pc2", "pc2_rank1", "cobs_rank1", "cobs_rank2"):
        log_weight = torch.logsumexp(b, dim=-1)
        w = torch.softmax(b, dim=-1)
        mean = (w[..., None] * x).sum(-2)
        if mode != "weighted_mean":
            centered = x - mean[..., None, :]
    extra = {}
    if mode in ("pc2", "pc2_rank1"):
        half = rd // 2
        # NOSA and HF split-half basis: (f, f + rotary_dim/2) is one pair.
        u, v = centered[..., :half], centered[..., half:rd]
        extra["var_x"] = (w[..., None] * u.square()).sum(-2)
        extra["cov_xy"] = (w[..., None] * u * v).sum(-2)
        extra["var_y"] = (w[..., None] * v.square()).sum(-2)
        if rd < dim:
            # Non-rotating channels carry a diagonal covariance; invariant
            # under RoPE's identity action, but not a full covariance.
            extra["var_nope"] = (w[..., None] * centered[..., rd:].square()).sum(-2)
        if mode == "pc2_rank1":
            # Prepared failure branch: preserve one cross-frequency direction
            # and pair-project only its PSD residual to avoid double counting.
            _, singular, vh = torch.linalg.svd(centered * w.sqrt()[..., None], full_matrices=False)
            factor = vh[..., :1, :] * singular[..., :1, None]
            extra["factors"] = factor
            a, z = factor[..., 0, :half], factor[..., 0, half:rd]
            extra["var_x"] = (extra["var_x"] - a.square()).clamp_min(0)
            extra["cov_xy"] = extra["cov_xy"] - a * z
            extra["var_y"] = (extra["var_y"] - z.square()).clamp_min(0)
            if rd < dim:
                extra["var_nope"] = (extra["var_nope"] - factor[..., 0, rd:].square()).clamp_min(0)
    elif mode.startswith("cobs_rank"):
        rank = int(mode[-1])
        weighted = centered * w.sqrt()[..., None]
        # Per-block full-space SVD: a controlled COBS-inspired variant, without
        # its learned query subspace or FP4 implementation. This truncation is
        # also orthogonally equivariant away from a degenerate cutoff.
        _, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
        r = min(rank, singular.shape[-1])
        extra["factors"] = vh[..., :r, :] * singular[..., :r, None]
    elif mode.startswith("split"):
        pieces = int(mode[-1])
        if size % pieces:
            raise ValueError("split control requires a divisible block size")
        xx = x.reshape(*x.shape[:-2], pieces, size // pieces, dim)
        bb = b.reshape(*b.shape[:-1], pieces, size // pieces)
        extra["sub_mean"] = (bb.softmax(-1)[..., None] * xx).sum(-2)
        extra["sub_log_weight"] = torch.logsumexp(bb, -1)
    elif mode == "quest":
        extra = {"key_min": x.amin(-2), "key_max": x.amax(-2), "cis_max": b.amax(-1)}
    dtype = storage_dtype or acc
    return Summary(mean.to(dtype), log_weight, {k: v.to(dtype) for k, v in extra.items()}, keys.shape[1])


def summary_logmass(q: Tensor, summary: Summary, mode: str, *, rotary_dim: int | None = None) -> Tensor:
    """Per-head approximate log mass, [KV,G,Q,blocks]; q is already scaled."""
    if mode == "pc2_unweighted":
        mode = "pc2"
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    query = q.to(dtype)
    dim = query.shape[-1]
    if mode.startswith("split"):
        logits = torch.einsum("hgqd,hbmd->hgqbm", query, summary.extra["sub_mean"].to(dtype))
        return torch.logsumexp(logits + summary.extra["sub_log_weight"].to(dtype)[:, None, None], -1)
    if mode == "quest":
        logits = (torch.einsum("hgqd,hbd->hgqb", query.clamp_min(0), summary.extra["key_max"].to(dtype))
                  + torch.einsum("hgqd,hbd->hgqb", query.clamp_max(0), summary.extra["key_min"].to(dtype)))
        # This is an extrema ranking heuristic, not an unbiased logmass.
        return logits + summary.extra["cis_max"].to(dtype)[:, None, None]
    logits = torch.einsum("hgqd,hbd->hgqb", query, summary.mean.to(dtype))
    logits = logits + summary.log_weight.to(dtype)[:, None, None]
    if mode in ("pc2", "pc2_rank1"):
        rd = dim if rotary_dim is None else rotary_dim
        half = rd // 2
        u, v = query[..., :half], query[..., half:rd]
        variance = (torch.einsum("hgqd,hbd->hgqb", u.square(), summary.extra["var_x"].to(dtype))
                    + 2 * torch.einsum("hgqd,hbd->hgqb", u * v, summary.extra["cov_xy"].to(dtype))
                    + torch.einsum("hgqd,hbd->hgqb", v.square(), summary.extra["var_y"].to(dtype)))
        if rd < dim:
            variance = variance + torch.einsum("hgqd,hbd->hgqb", query[..., rd:].square(), summary.extra["var_nope"].to(dtype))
        if mode == "pc2_rank1":
            projection = torch.einsum("hgqd,hbrd->hgqbr", query, summary.extra["factors"].to(dtype))
            variance = variance + projection.square().sum(-1)
        logits = logits + 0.5 * variance.clamp_min(0)
    elif mode.startswith("cobs_rank"):
        factors = summary.extra["factors"].to(dtype)
        projected = torch.einsum("hgqd,hbrd->hgqbr", query, factors)
        logits = logits + 0.5 * projected.square().sum(-1)
    return logits


def _concat_summary(old: Summary | None, new: Summary) -> Summary:
    if old is None:
        return new
    return Summary(torch.cat((old.mean, new.mean), 1),
                   torch.cat((old.log_weight, new.log_weight), 1),
                   {name: torch.cat((old.extra[name], value), 1) for name, value in new.extra.items()},
                   old.blocks + new.blocks)


class BlockSummarySelector:
    """Cached per-layer summaries, reset when a new sequence begins.

    Only fully completed 64-token blocks enter the persistent summary. The
    current partial block's log mass is evaluated from its visible raw keys,
    so GQA normalizers neither omit it nor see future tokens. Full raw K/V
    remain in the parent cache; metadata bytes are reported separately.
    """
    def __init__(self, mode="pc2", *, rotary_dim: int | None = None, storage_dtype=torch.float32):
        if mode not in MODES:
            raise ValueError(mode)
        self.cis_tilt = mode != "pc2_unweighted"
        self.mode = "pc2" if mode == "pc2_unweighted" else mode
        self.rotary_dim, self.storage_dtype = rotary_dim, storage_dtype
        self.cache: dict[int, Summary] = {}
        self.metrics = {"summary_build_calls": 0, "summary_build_wall_seconds": 0.0,
                        "max_metadata_bytes": 0, "exact_raw_key_scores": 0, "calls": 0}

    def reset(self):
        self.cache.clear()
        self.metrics = {"summary_build_calls": 0, "summary_build_wall_seconds": 0.0,
                        "max_metadata_bytes": 0, "exact_raw_key_scores": 0, "calls": 0}

    @torch.no_grad()
    def logmass(self, context: SelectionContext) -> Tensor:
        q, k, s = context.q, context.k, context.settings
        # The unweighted control changes only the query-aware statistic. The
        # original context still reaches select_with_scores (real CIS quota)
        # and the pretrained final reader (real per-token CIS key bias).
        cis = context.cis if self.cis_tilt else torch.zeros_like(context.cis)
        kvh, length, dim = k.shape
        queries, group = q.shape[1], q.shape[0] // kvh
        scaled_q = q.reshape(kvh, group, queries, dim).float() / math.sqrt(dim)
        count, full = math.ceil(length / s.block_size), length // s.block_size
        if int(context.query_positions[0]) == 0:
            self.cache.pop(context.layer_idx, None)
        if self.mode == "exact_mass":
            out = scaled_q.new_full((kvh, group, queries, count), -torch.inf)
            for j in range(count):
                begin, end = j * s.block_size, min((j + 1) * s.block_size, length)
                logits = torch.einsum("hgqd,htd->hgqt", scaled_q, k[:, begin:end].float())
                logits += cis[:, None, None, begin:end].float()
                visible = torch.arange(begin, end, device=q.device)[None] <= context.query_positions[:, None]
                out[..., j] = torch.logsumexp(logits.masked_fill(~visible[None, None], -torch.inf), -1)
            self.metrics["exact_raw_key_scores"] += q.shape[0] * queries * length
            return out
        old = self.cache.get(context.layer_idx)
        done = old.blocks if old else 0
        if done > full:
            raise ValueError("cache sequence shrank without a position-zero reset")
        if full > done:
            begin, end = done * s.block_size, full * s.block_size
            start = time.perf_counter()
            new = build_summary(k[:, begin:end].reshape(kvh, full - done, s.block_size, dim),
                                cis[:, begin:end].reshape(kvh, full - done, s.block_size),
                                self.mode, rotary_dim=self.rotary_dim, storage_dtype=self.storage_dtype)
            old = _concat_summary(old, new)
            self.cache[context.layer_idx] = old
            # CPU wall timing is diagnostic; GPU kernel timings require events
            # in the runner, as asynchronous launches cannot be timed this way.
            self.metrics["summary_build_wall_seconds"] += time.perf_counter() - start
            self.metrics["summary_build_calls"] += 1
        out = scaled_q.new_full((kvh, group, queries, count), -torch.inf)
        if old is not None and full:
            out[..., :full] = summary_logmass(scaled_q, old, self.mode, rotary_dim=self.rotary_dim)
        endpoints = (torch.arange(count, device=q.device) + 1) * s.block_size - 1
        out.masked_fill_(~(endpoints[None] <= context.query_positions[:, None])[None, None], -torch.inf)
        # Every query's currently incomplete block is mandatory, but its exact
        # normalizer still matters when different query heads share one route.
        current_ids = context.query_positions // s.block_size
        for j in current_ids.unique().tolist():
            rows = (current_ids == j).nonzero(as_tuple=True)[0]
            begin, end = j * s.block_size, min((j + 1) * s.block_size, length)
            logits = torch.einsum("hgqd,htd->hgqt", scaled_q[:, :, rows], k[:, begin:end].float())
            logits += cis[:, None, None, begin:end].float()
            visible = torch.arange(begin, end, device=q.device)[None] <= context.query_positions[rows, None]
            out[:, :, rows, j] = torch.logsumexp(logits.masked_fill(~visible[None, None], -torch.inf), -1)
            self.metrics["exact_raw_key_scores"] += q.shape[0] * len(rows) * (end - begin)
        self.metrics["max_metadata_bytes"] = max(self.metrics["max_metadata_bytes"],
                                                sum(value.nbytes() for value in self.cache.values()))
        return out

    @torch.no_grad()
    def __call__(self, context: SelectionContext) -> Tensor:
        self.metrics["calls"] += 1
        # Match native's all-visible shortcut before sparsification is active.
        if math.ceil(context.k.shape[1] / context.settings.block_size) <= context.settings.topk:
            if int(context.query_positions[0]) == 0:
                self.cache.pop(context.layer_idx, None)
            return select_with_scores(context, context.q.new_empty(0))
        logmass = self.logmass(context)
        group_score = torch.softmax(logmass, -1).sum(1)
        return select_with_scores(context, group_score)
