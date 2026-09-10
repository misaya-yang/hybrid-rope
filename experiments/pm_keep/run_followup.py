"""Matched value-aware P/C/U ablation; fixed EA/F/KeyDiff remain reusable.

Example: python -m experiments.pm_keep.run_followup --value-objective raw_norm
         --data DATA --output NEW_RUN --baseline-cache CACHE --split dev

This changes a scoring objective, not RoPE or the read operator. It is a
prepared follow-up, not an established improvement or a new value-norm idea.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

import torch

from . import run as base
from .adapter import AdapterConfig, PrefillSession, _sync


def value_weights(values, objective):
    """[1, Hkv, T, D] -> [Hkv, T]; uses only already-visible prefix V."""
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("Expected the native batch-one prefix values")
    x = values[0].float()
    if objective == "centered_norm":
        x = x - x.mean(dim=-2, keepdim=True)
    elif objective != "raw_norm":
        raise ValueError(objective)
    return x.norm(dim=-1)


class ValueAwareSession(PrefillSession):
    @torch.inference_mode()
    def score(self, arm):
        if arm in self.scores or arm not in ("P", "C", "U"):
            return super().score(arm)
        if self.config.value_objective == "value_only":
            _sync(self.device)
            start = time.perf_counter()
            self.scores[arm] = [value_weights(layer.values, "raw_norm") for layer in self.cache.layers]
            _sync(self.device)
            self.timings["score_seconds"][arm] = time.perf_counter() - start
            self.score_metrics[arm] = [dict(value_objective="value_only", query_or_position_scoring=False)
                                      for _ in self.cache.layers]
            return self.scores[arm]
        scores = super().score(arm)
        _sync(self.device)
        start = time.perf_counter()
        for i, (scores_i, layer) in enumerate(zip(scores, self.cache.layers)):
            scores[i] = scores_i * value_weights(layer.values, self.config.value_objective)
            self.score_metrics[arm][i].update(
                value_norm_weighting=True, value_objective=self.config.value_objective,
                score_objective="expected_prefix_mass_times_fixed_value_magnitude",
                normalized_output_optimality_claim=False,
            )
        _sync(self.device)
        self.timings["score_seconds"][arm] += time.perf_counter() - start
        self.scores[arm] = scores
        return scores


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--value-objective", choices=("raw_norm", "centered_norm", "value_only"), required=True)
    args, remaining = p.parse_known_args()
    # The existing manifest includes this config and the source hash. Old
    # output directories reject changed contracts. Fixed baseline keys use
    # only their own scientifically relevant fields, so remain reusable.
    @dataclass(frozen=True)
    class FollowupConfig(AdapterConfig):
        value_objective: str = args.value_objective
        followup_source_sha256: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    base.AdapterConfig = FollowupConfig
    base.PrefillSession = ValueAwareSession
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
