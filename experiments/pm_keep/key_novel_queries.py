"""Prefix-Q sampling weighted by the observed key-direction novelty.

Uses the same cosine-to-mean geometry as the existing KeyDiff score, with
nonnegative squared spherical distance 1-cos as the sampling measure. This is
a content-proxy candidate, not a new position mechanism or a claim that a key's
own query must represent a future question about that key.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from . import run as base
from .adapter import AdapterConfig
from .ops import _seed
from .run_followup import ValueAwareSession


def novelty_weights(keys, sink_tokens=4):
    """K[KV,T,D] -> non-sink sampling weights[KV,T-sink]."""
    if keys.ndim != 3 or not 0 <= sink_tokens < keys.shape[1]:
        raise ValueError("native keys and a nonempty non-sink prefix are required")
    values = keys.float()
    anchor = F.normalize(values, p=2, dim=-1).mean(dim=1, keepdim=True)
    distance = (1 - F.cosine_similarity(values, anchor, dim=-1)).clamp_min(0)
    distance = distance[:, sink_tokens:].double()
    sums = distance.sum(-1, keepdim=True)
    uniform = torch.full_like(distance, 1 / distance.shape[-1])
    return torch.where(sums > 0, distance / sums.clamp_min(torch.finfo(torch.float64).tiny), uniform)


def novelty_indices(keys, plan, *, sink_tokens):
    weights = novelty_weights(keys, sink_tokens).cpu()
    heads, count = plan.query_indices.shape
    if heads % keys.shape[0]:
        raise ValueError("native contiguous GQA grouping required")
    group = heads // keys.shape[0]
    indices = []
    for head in range(heads):
        # Couple the random stream to the existing type-balanced experiment.
        rng = torch.Generator().manual_seed(_seed(plan.seed, plan.layer_idx, head,
                                                  "query_type_balanced_v1"))
        indices.append(torch.multinomial(weights[head // group], count, replacement=True,
                                         generator=rng) + sink_tokens)
    return torch.stack(indices)


@dataclass(frozen=True)
class KeyNovelConfig(AdapterConfig):
    value_objective: str = "raw_norm"
    sampling_override: str = "prefix_key_spherical_distance_v1"
    sampling_source_sha256: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class KeyNovelSession(ValueAwareSession):
    def __init__(self, model, prefix_ids, config=None):
        super().__init__(model, prefix_ids, config or KeyNovelConfig())
        if self.config.query_policy != "uniform_prefix":
            raise ValueError("key novelty sampling uses the whole non-sink prefix")

    @torch.inference_mode()
    def prefill(self, external_scorers=None):
        raw_queries, handles = {}, []
        for index, layer in enumerate(self.model.model.layers):
            def capture(module, args, output, index=index):
                if index not in raw_queries:
                    raw_queries[index] = output.detach()
            handles.append(layer.self_attn.q_proj.register_forward_hook(capture))

        def sample(data):
            index = data.layer_idx
            plan = self.plans[index]
            take = novelty_indices(data.keys[0], plan, sink_tokens=self.config.sink_tokens)
            self.plans[index] = replace(plan, query_indices=take)
            raw = raw_queries.pop(index)
            heads = self.model.config.num_attention_heads
            query = raw.view(1, self.prefix_length, heads, -1).transpose(1, 2)
            selected = take.to(self.device)[None, :, :, None].expand(1, heads, -1, query.shape[-1])
            self.query_samples[index] = query.gather(2, selected).detach()
            return novelty_weights(data.keys[0], 0).float()

        callbacks = dict(external_scorers or {})
        callbacks["_key_novel_sampling"] = sample
        try:
            super().prefill(callbacks)
        finally:
            for handle in handles:
                handle.remove()
            raw_queries.clear()
        # Charge sampler construction to P/C/U standalone costs; no hidden
        # uncounted external-scorer cost after subtracting native prefill time.
        self.timings["query_capture_seconds"] += self.timings["score_seconds"].pop("_key_novel_sampling")
        self.scores.pop("_key_novel_sampling")
        return self


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--broad-scoring", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.broad_scoring:
        from experiments.broad_position_eval.scoring import VERSION, score
        base.score = score
        base.BASELINE_VERSION += "_" + VERSION
    base.AdapterConfig = KeyNovelConfig
    base.PrefillSession = KeyNovelSession
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
