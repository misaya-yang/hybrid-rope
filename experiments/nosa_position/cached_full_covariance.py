"""Exact full-second-order reparameterization, not another approximation.

Cache [logZ, mean, upper-triangle covariance] in FP32. The packed covariance
coefficient is .5*Sigma_ii on diagonal and Sigma_ij off diagonal. Query features
[1, a_i, a_i*a_j], a=q/sqrt(D), give logmass in one batched GEMM per query tile.
Current blocks remain exact, causal raw-key reads. No SVD, low rank or quantization.

D=128 stores 8,385 FP32 numbers = 33,540 bytes per completed block/KV head.
At T=16,384/B=64, 2 KV heads and 28 layers, descriptors alone are 480,829,440
bytes (0.481 GB / 0.448 GiB), IN ADDITION to raw KV/CIS. Build/concatenation and
query-feature scratch also cost memory/time. Query FLOPs are close to raw QK;
no low-memory or asymptotic speedup claim follows. CPU FP64 is verification only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .runtime import select_with_scores


@dataclass
class PackedCovariance:
    coefficients: torch.Tensor  # [KV, completed blocks, 1+D+D(D+1)/2]
    dim: int

    @property
    def blocks(self):
        return self.coefficients.shape[1]

    @property
    def nbytes(self):
        return self.coefficients.untyped_storage().nbytes()


@torch.no_grad()
def build_packed(keys, cis):
    """Build completed blocks [KV,blocks,B,D]; no precision compression."""
    if keys.ndim != 4 or cis.shape != keys.shape[:-1]:
        raise ValueError("keys [KV,blocks,B,D] and matching CIS required")
    dtype = torch.float64 if keys.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=keys.device.type, enabled=False):
        x, b = keys.to(dtype), cis.to(dtype)
        w = b.softmax(-1)
        mean = (x * w[..., None]).sum(-2)
        centered = x - mean[..., None, :]
        covariance = centered.transpose(-1, -2) @ (centered * w[..., None])
        row, col = torch.triu_indices(x.shape[-1], x.shape[-1], device=x.device)
        upper = covariance[..., row, col]
        upper = upper * torch.where(row == col, .5, 1.).to(dtype)
        packed = torch.cat((b.logsumexp(-1)[..., None], mean, upper), -1)
    return PackedCovariance(packed, keys.shape[-1])


@torch.no_grad()
def packed_logmass(scaled_query, summary):
    """scaled_query [KV,G,Q,D] already includes the native 1/sqrt(D)."""
    if scaled_query.shape[-1] != summary.dim:
        raise ValueError("query and descriptor dimensions disagree")
    dtype = torch.float64 if scaled_query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=scaled_query.device.type, enabled=False):
        q = scaled_query.to(dtype)
        row, col = torch.triu_indices(summary.dim, summary.dim, device=q.device)
        features = torch.cat((torch.ones_like(q[..., :1]), q, q[..., row] * q[..., col]), -1)
        shape = q.shape[:-1]
        result = features.flatten(1, 2) @ summary.coefficients.to(dtype).transpose(-1, -2)
    return result.reshape(*shape, summary.blocks)


class CachedFullCovarianceSelector:
    """Importable selector; root owns scheduling and two-input runtime parity."""
    mode = "cached_full_covariance"

    def __init__(self):
        self.reset()

    def reset(self):
        self.cache = {}
        self.metrics = dict(max_metadata_bytes=0, summary_build_calls=0,
                            exact_raw_key_scores=0, calls=0,
                            descriptor_dtype="float32", mathematical_target="full_second_cumulant",
                            deployment_speed_verified=False)

    @torch.no_grad()
    def logmass(self, context):
        q, k, cis, s = context.q, context.k, context.cis, context.settings
        kvh, length, dim = k.shape
        group, queries = q.shape[0] // kvh, q.shape[1]
        full, count = length // s.block_size, math.ceil(length / s.block_size)
        if int(context.query_positions[0]) == 0:
            self.cache.pop(context.layer_idx, None)
        old = self.cache.get(context.layer_idx)
        done = old.blocks if old is not None else 0
        if done > full:
            raise ValueError("prefix shrank without a position-zero reset")
        if full > done:
            begin, end = done * s.block_size, full * s.block_size
            added = build_packed(k[:, begin:end].float().reshape(kvh, full-done, s.block_size, dim),
                                 cis[:, begin:end].float().reshape(kvh, full-done, s.block_size))
            old = added if old is None else PackedCovariance(
                torch.cat((old.coefficients, added.coefficients), 1), dim)
            self.cache[context.layer_idx] = old
            self.metrics["summary_build_calls"] += 1
        with torch.autocast(device_type=q.device.type, enabled=False):
            query = q.float().reshape(kvh, group, queries, dim) / math.sqrt(dim)
            result = query.new_full((kvh, group, queries, count), -torch.inf)
            if full:
                for begin in range(0, queries, s.attention_query_chunk_size):
                    end = min(queries, begin + s.attention_query_chunk_size)
                    result[:, :, begin:end, :full] = packed_logmass(query[:, :, begin:end], old)
            endpoints = (torch.arange(count, device=q.device) + 1) * s.block_size - 1
            result.masked_fill_(~(endpoints[None] <= context.query_positions[:, None])[None, None], -torch.inf)
            current = context.query_positions // s.block_size
            for block in current.unique().tolist():
                rows = (current == block).nonzero(as_tuple=True)[0]
                begin, end = block * s.block_size, min((block + 1) * s.block_size, length)
                logits = torch.einsum("hgqd,htd->hgqt", query[:, :, rows], k[:, begin:end].float())
                logits += cis[:, None, None, begin:end].float()
                visible = torch.arange(begin, end, device=q.device)[None] <= context.query_positions[rows, None]
                result[:, :, rows, block] = logits.masked_fill(~visible[None, None], -torch.inf).logsumexp(-1)
                self.metrics["exact_raw_key_scores"] += q.shape[0] * len(rows) * (end - begin)
        if bool(torch.isnan(result).any()) or bool(torch.isposinf(result).any()):
            raise FloatingPointError("invalid packed full-covariance score")
        self.metrics["max_metadata_bytes"] = max(self.metrics["max_metadata_bytes"], sum(x.nbytes for x in self.cache.values()))
        return result

    @torch.no_grad()
    def __call__(self, context):
        self.metrics["calls"] += 1
        if int(context.query_positions[0]) == 0:
            self.cache.pop(context.layer_idx, None)
        if math.ceil(context.k.shape[1] / context.settings.block_size) <= context.settings.topk:
            return select_with_scores(context, context.q.new_empty(0))
        return select_with_scores(context, self.logmass(context).softmax(-1).sum(1))
