"""Type-balanced prefix Q proxy with the fixed raw-value PM objective.

Each observed non-sink token type has equal total sampling probability.
This tests whether repeated background dominates uniform prefix sampling;
no question, answer, new position range, or trained weights are used.
"""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import torch

from . import run as base
from .adapter import AdapterConfig
from .ops import _seed
from .run_followup import ValueAwareSession


def type_weights(prefix_ids, sink_tokens):
    ids = torch.as_tensor(prefix_ids).detach().cpu().flatten()[sink_tokens:]
    _, inverse, counts = torch.unique(ids, return_inverse=True, return_counts=True)
    weights = counts[inverse].double().reciprocal()
    return weights / weights.sum()


def balanced_plan(plan, weights, sink_tokens):
    indices = []
    for head in range(plan.query_indices.shape[0]):
        rng = torch.Generator().manual_seed(_seed(plan.seed, plan.layer_idx, head, "query_type_balanced_v1"))
        indices.append(torch.multinomial(weights, plan.query_indices.shape[1], replacement=True,
                                         generator=rng) + sink_tokens)
    return replace(plan, query_indices=torch.stack(indices))


class BalancedValueSession(ValueAwareSession):
    def __init__(self, model, prefix_ids, config=None):
        super().__init__(model, prefix_ids, config)
        if self.config.query_policy != "uniform_prefix":
            raise ValueError("Type-balanced experiment uses the whole non-sink prefix, not recent-prefix")
        weights = type_weights(self.prefix_ids, self.config.sink_tokens)
        self.plans = [balanced_plan(plan, weights, self.config.sink_tokens) for plan in self.plans]


@dataclass(frozen=True)
class BalancedConfig(AdapterConfig):
    value_objective: str = "raw_norm"
    sampling_override: str = "inverse_prefix_token_frequency_v1"
    sampling_source_sha256: str = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    base.AdapterConfig = BalancedConfig
    base.PrefillSession = BalancedValueSession
    base.main()


if __name__ == "__main__":
    main()
