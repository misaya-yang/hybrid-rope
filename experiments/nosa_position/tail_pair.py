"""One exact extreme key plus a pair-covariance bulk approximation.

The extra D-dimensional slot replaces the rank-one covariance direction with
an original key, plus one CIS scalar. Selection is prefix-only and needs no
SVD. This is a candidate for the specific failure where one key dominates an
exponential block response; it is not a guarantee for arbitrary query tails.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import torch

from . import run as base
from . import exact_probe
from .exact_probe import ExactBlockSelector
from .selector_controls import Summary, _concat_summary, build_summary, summary_logmass


MODE = "pc2_tail1"


def build_tail_summary(keys, cis, *, storage_dtype=None):
    """Remove the largest-radius key around the CIS-weighted block center.

    This deterministic, query-independent rule isolates the worst residual
    norm around that center. It does not assert which key an unseen Q needs.
    The tail's original K and CIS are retained exactly in accumulator dtype.
    """
    if keys.ndim != 4 or cis.shape != keys.shape[:-1] or keys.shape[-2] < 2:
        raise ValueError("expected [KV, blocks, B>=2, D] keys and matching CIS")
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    x, bias = keys.to(dtype), cis.to(dtype)
    mean = (bias.softmax(-1)[..., None] * x).sum(-2)
    index = (x - mean[..., None, :]).square().sum(-1).argmax(-1)
    tail_key = x.gather(-2, index[..., None, None].expand(*index.shape, 1, x.shape[-1])).squeeze(-2)
    tail_bias = bias.gather(-1, index[..., None]).squeeze(-1)
    bulk_bias = bias.clone().scatter_(-1, index[..., None], -torch.inf)
    summary = build_summary(x, bulk_bias, "pc2", storage_dtype=storage_dtype)
    # Keep original-tail precision; bulk statistics follow the usual setting.
    summary.extra.update(tail_key=tail_key, tail_bias=tail_bias)
    return summary


def tail_logmass(query, summary):
    """Query is scaled by 1/sqrt(D), as in summary_logmass."""
    bulk = summary_logmass(query, summary, "pc2")
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    tail = torch.einsum("hgqd,hbd->hgqb", query.to(dtype), summary.extra["tail_key"].to(dtype))
    tail = tail + summary.extra["tail_bias"].to(dtype)[:, None, None]
    return torch.logaddexp(tail, bulk)


class TailPairSelector(ExactBlockSelector):
    def __init__(self, mode=MODE, **kwargs):
        self.tail_enabled = mode == MODE
        super().__init__("pc2" if self.tail_enabled else mode, **kwargs)

    @torch.no_grad()
    def logmass(self, context):
        if not self.tail_enabled:
            return super().logmass(context)
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
            added = build_tail_summary(
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
        # The query's current block is exact and causal, as in original PC2.
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
    original = base.source_hashes
    base.source_hashes = lambda: {**original(),
        "exact_probe.py": hashlib.sha256(Path(exact_probe.__file__).read_bytes()).hexdigest(),
        "tail_pair.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    base.MODES = (*base.MODES, MODE)
    base.BlockSummarySelector = TailPairSelector
    base.main()


if __name__ == "__main__":
    main()
