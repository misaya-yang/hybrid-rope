"""E01 INT8 tiled logmass with normalized-GQA intervals and raw-K refinement.

Real-arithmetic bounds plus numerical empirical comparison, not a bitwise
floating-point certificate. Raw K/V remain resident; all extra scans are counted.
"""
import math

import torch

from .exact_probe import ExactBlockSelector
from .runtime import mandatory_blocks, select_with_scores, _stable_topk


def _kernel():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def score(Q, K, CIS, POS, INDEX, SCALE, ERROR, REFINE, OUT, DELTA,
              length, queries, blocks, quant_blocks, GROUP: tl.constexpr,
              D: tl.constexpr, B: tl.constexpr, TQ: tl.constexpr):
        head, qt, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        h = head // GROUP
        qi = qt * TQ + tl.arange(0, TQ)
        di = tl.arange(0, D)
        ki = block * B + tl.arange(0, B)
        qp = tl.load(POS + qi, qi < queries, other=-1)
        query = tl.load(Q + (head * queries + qi[:, None]) * D + di[None, :], qi[:, None] < queries, other=0).to(tl.float32) * (D ** -0.5)
        refine = tl.load(REFINE + (h * queries + qi) * blocks + block, qi < queries, other=0)
        raw = (block >= quant_blocks) | (tl.max(refine, 0) > 0)
        if raw:
            keys = tl.load(K + (h * length + ki[None, :]) * D + di[:, None], ki[None, :] < length, other=0).to(tl.float32)
            error = tl.full((D,), 0, tl.float32)
        else:
            scale = tl.load(SCALE + (h * quant_blocks + block) * D + di)
            error = tl.load(ERROR + (h * quant_blocks + block) * D + di)
            keys = tl.load(INDEX + (h * quant_blocks * B + ki[None, :]) * D + di[:, None]).to(tl.float32) * scale[:, None]
        logits = tl.dot(query, keys, input_precision="ieee")
        bias = tl.load(CIS + h * length + ki, ki < length, other=-float('inf')).to(tl.float32)
        logits += bias[None, :]
        logits = tl.where((ki[None, :] <= qp[:, None]) & (ki[None, :] < length) & (qi[:, None] < queries), logits, -float('inf'))
        maximum = tl.max(logits, 1)
        safe_maximum = tl.where(maximum == -float('inf'), 0., maximum)
        logmass = safe_maximum + tl.log(tl.sum(tl.exp(logits - safe_maximum[:, None]), 1))
        delta = tl.sum(tl.abs(query) * error[None, :], 1)
        offset = (head * queries + qi) * blocks + block
        tl.store(OUT + offset, logmass, qi < queries)
        tl.store(DELTA + offset, delta, qi < queries)
    return score


_SCORE = None


class BoundedInt8Selector(ExactBlockSelector):
    def __init__(self, mode="e01_int8", **kwargs):
        self.bounded = mode == "e01_int8"
        super().__init__("exact_mass" if self.bounded else mode, **kwargs)
        self.index = {}

    def _build(self, context):
        size = context.settings.block_size
        # Only blocks completed before the first current query enter the index.
        full = int(context.query_positions[0]) // size
        if full == 0:
            self.index.pop(context.layer_idx, None)
        old = self.index.get(context.layer_idx)
        done = old[0].shape[1] // size if old else 0
        if full > done:
            h, _, d = context.k.shape
            x = context.k[:, done*size:full*size].float().reshape(h, full-done, size, d)
            scale = (x.abs().amax(-2) / 127).clamp_min(torch.finfo(torch.float32).tiny)
            index = (x / scale[:, :, None]).round().clamp(-127, 127).to(torch.int8)
            error = (x - index.float() * scale[:, :, None]).abs().amax(-2)
            fresh = (index.flatten(1, 2), scale, error)
            old = tuple(torch.cat((a, b), 1) for a, b in zip(old, fresh)) if old else fresh
            self.index[context.layer_idx] = old
            self.metrics["e01_build_raw_k_elements"] = self.metrics.get("e01_build_raw_k_elements", 0) + x.numel()
        if old is None:
            h, _, d = context.k.shape
            old = (torch.empty((h, 0, d), device=context.k.device, dtype=torch.int8),
                   torch.empty((h, 0, d), device=context.k.device), torch.empty((h, 0, d), device=context.k.device))
        self.metrics["max_metadata_bytes"] = sum(t.nbytes for entry in self.index.values() for t in entry)
        return old

    def _score(self, context, index, refine):
        global _SCORE
        if _SCORE is None:
            _SCORE = _kernel()
        q = context.q.contiguous()
        h, length, d = context.k.shape
        queries = q.shape[1]
        b = context.settings.block_size
        blocks = math.ceil(length/b)
        group = q.shape[0] // h
        out = torch.empty((h, group, queries, blocks), device=q.device, dtype=torch.float32)
        delta = torch.empty_like(out)
        quant_blocks = index[0].shape[1] // b
        _SCORE[(q.shape[0], math.ceil(queries/16), blocks)](
            q, context.k.contiguous(), context.cis.contiguous(), context.query_positions,
            *index, refine.contiguous(), out, delta, length, queries, blocks, quant_blocks,
            GROUP=group, D=d, B=b, TQ=16, num_warps=4)
        # Refine is executed for the whole 16-query tile if any lane needs it.
        padded = torch.nn.functional.pad(refine, (0, 0, 0, (-queries) % 16))
        raw_tiles = padded.reshape(h, -1, 16, blocks).any(2)
        raw_tiles[..., quant_blocks:] = True
        raw = int(raw_tiles.sum()) * b * d * group
        self.metrics["e01_raw_k_tile_elements"] = self.metrics.get("e01_raw_k_tile_elements", 0) + raw
        self.metrics["e01_scoring_passes"] = self.metrics.get("e01_scoring_passes", 0) + 1
        self.metrics["e01_int8_tile_elements"] = self.metrics.get("e01_int8_tile_elements", 0) + int((~raw_tiles).sum()) * b * d * group
        return out, delta

    def _decision(self, context, logmass, delta):
        visible = torch.arange(logmass.shape[-1], device=logmass.device)[None] <= context.query_positions[:, None] // context.settings.block_size
        lo, hi = logmass-delta, logmass+delta
        offset = hi.amax(-1, keepdim=True)
        low, high = (lo-offset).exp(), (hi-offset).exp()
        # The full causal denominator includes mandatory mass. Each block uses
        # its own bound and the opposite bound for every other block.
        lower = (low / (high.sum(-1, keepdim=True)-high+low).clamp_min(1e-30)).sum(1)
        upper = (high / (low.sum(-1, keepdim=True)-low+high).clamp_min(1e-30)).sum(1)
        score = logmass.softmax(-1).sum(1)
        mandatory = mandatory_blocks(context, logmass.shape[-1])
        ranked = score.masked_fill(mandatory, torch.inf).masked_fill(~visible, -torch.inf)
        count = min(context.settings.init_blocks+context.settings.local_blocks+context.settings.select_blocks, logmass.shape[-1])
        chosen = _stable_topk(ranked, count)
        mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, chosen, True)
        cutoff = lower.masked_fill(mandatory, torch.inf).gather(-1, chosen).amin(-1)
        outside = upper.masked_fill(mask | mandatory | ~visible, -torch.inf).amax(-1)
        certified = cutoff > outside
        ambiguous = ((upper >= cutoff[..., None]) | (mask & (lower <= outside[..., None]))) & visible & ~mandatory
        return score, certified, ambiguous

    @torch.no_grad()
    def __call__(self, context):
        if not self.bounded:
            return super().__call__(context)
        self.metrics["calls"] += 1
        blocks = math.ceil(context.k.shape[1]/context.settings.block_size)
        if blocks <= context.settings.topk:
            return select_with_scores(context, context.q.new_empty(0))
        index = self._build(context)
        h, queries = context.k.shape[0], context.q.shape[1]
        flags = torch.zeros((h, queries, blocks), device=context.q.device, dtype=torch.bool)
        logmass, delta = self._score(context, index, flags)
        score, certified, ambiguous = self._decision(context, logmass, delta)
        initial = certified.clone()
        if not bool(certified.all()):
            flags = ambiguous & ~certified[..., None]
            refined, error = self._score(context, index, flags)
            # Keep already certified states unchanged, including their intervals.
            logmass = torch.where(certified[:, None, :, None], logmass, refined)
            delta = torch.where(certified[:, None, :, None], delta, error)
            score, certified, _ = self._decision(context, logmass, delta)
        if not bool(certified.all()):
            flags = (~certified)[..., None].expand(-1, -1, blocks)
            exact, _ = self._score(context, index, flags)
            logmass = torch.where(certified[:, None, :, None], logmass, exact)
            score = logmass.softmax(-1).sum(1)
        self.metrics["e01_states"] = self.metrics.get("e01_states", 0) + h * queries
        self.metrics["e01_initial_certified"] = self.metrics.get("e01_initial_certified", 0) + int(initial.sum())
        self.metrics["e01_fallback_states"] = self.metrics.get("e01_fallback_states", 0) + int((~certified).sum())
        return select_with_scores(context, score)
