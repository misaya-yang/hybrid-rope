"""Exact causal block-mass reference for diagnosing PC2 approximation errors.

Uses full raw K to score blocks, then the unchanged NOSA quota and reader.
This is an expensive reference, not an asserted deployable approximation.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import run as base
from .selector_controls import BlockSummarySelector


class ExactBlockSelector(BlockSummarySelector):
    @torch.no_grad()
    def logmass(self, context):
        if self.mode != "exact_mass":
            return super().logmass(context)
        q, k, settings = context.q, context.k, context.settings
        kvh, length, dim = k.shape
        group, queries = q.shape[0] // kvh, q.shape[1]
        blocks = math.ceil(length / settings.block_size)
        parts = []
        with torch.autocast(device_type=q.device.type, enabled=False):
            query = q.reshape(kvh, group, queries, dim).float() / math.sqrt(dim)
            keys = k.float()[:, None].transpose(-1, -2)
            for begin in range(0, queries, settings.attention_query_chunk_size):
                end = min(queries, begin + settings.attention_query_chunk_size)
                logits = query[:, :, begin:end] @ keys
                logits += context.cis.float()[:, None, None]
                visible = torch.arange(length, device=q.device)[None] <= context.query_positions[begin:end, None]
                logits.masked_fill_(~visible[None, None], -torch.inf)
                logits = F.pad(logits, (0, blocks * settings.block_size - length), value=-torch.inf)
                parts.append(logits.reshape(kvh, group, end-begin, blocks, settings.block_size).logsumexp(-1))
        self.metrics["exact_raw_key_scores"] += q.shape[0] * queries * length
        return torch.cat(parts, dim=2)


def main():
    original_hashes = base.source_hashes
    def source_hashes():
        return {**original_hashes(), "exact_probe.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    base.source_hashes = source_hashes
    base.BlockSummarySelector = ExactBlockSelector
    base.main()


if __name__ == "__main__":
    main()
