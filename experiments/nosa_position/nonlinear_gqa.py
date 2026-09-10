"""E02 reference: two risk-decreasing swaps within the original QK quota.

Exact scoring, CIS-only fillers, mandatory blocks and the raw-KV reader stay
unchanged. This Python reference makes no kernel-speed claim.
"""
import math

import torch

from .exact_probe import ExactBlockSelector
from .runtime import mandatory_blocks, select_with_scores, _stable_topk


@torch.no_grad()
def improve_set(probabilities, selected, fixed, visible):
    """One state: probabilities[G,B], selected[m], fixed/visible[B].

    Probabilities include mandatory mass in their full causal denominator.
    Return physical IDs and the accepted risk trace (initial plus <=2 entries).
    Candidate-pool ties and exchange ties prefer lower physical block IDs.
    """
    if probabilities.ndim != 2 or fixed.shape != visible.shape or fixed.numel() != probabilities.shape[1]:
        raise ValueError("expected probabilities[G,B] and fixed/visible[B]")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    selected = selected[selected >= 0].sort().values.clone()
    if selected.unique().numel() != selected.numel() or not visible[selected].all():
        raise ValueError("selected blocks must be unique and visible")
    if not torch.isin(torch.where(fixed)[0], selected).all():
        raise ValueError("fixed blocks must be selected")
    p = probabilities.to(torch.float64)
    mass = p[:, selected].sum(-1)
    if not (mass > 0).all():
        raise ValueError("each head needs positive retained mass")
    risk = (mass.reciprocal() - 1).sum()
    trace = [float(risk)]
    for _ in range(2):
        inside = torch.zeros_like(visible).scatter_(0, selected, True)
        outside = torch.where(visible & ~inside)[0]
        outgoing = selected[~fixed[selected]]
        if outside.numel() == 0 or outgoing.numel() == 0:
            break
        # Rebuild the finite candidate pool after an accepted swap.
        additive = p[:, outside].sum(0)
        pool = [outside[_stable_topk(additive, min(selected.numel(), outside.numel()))]]
        pool.extend(outside[_stable_topk(head[outside], min(2, outside.numel()))] for head in p)
        incoming = torch.cat(pool).unique(sorted=True)
        proposed_mass = mass[:, None, None] - p[:, outgoing, None] + p[:, None, incoming]
        risks = (proposed_mass.reciprocal() - 1).sum(0)
        legal = (proposed_mass > 0).all(0) & torch.isfinite(risks)
        risks = risks.masked_fill(~legal, torch.inf)
        index = int(risks.argmin())
        i, j = index // incoming.numel(), index % incoming.numel()
        if not risks[i, j] < risk:
            break
        proposal = selected.clone()
        proposal[proposal == outgoing[i]] = incoming[j]
        proposal = proposal.sort().values
        # Re-sum the actual final set, avoiding cancellation in the acceptance.
        new_mass = p[:, proposal].sum(-1)
        new_risk = (new_mass.reciprocal() - 1).sum()
        if not torch.isfinite(new_risk) or not new_risk < risk:
            break
        selected, mass, risk = proposal, new_mass, new_risk
        trace.append(float(risk))
    return selected, trace


class NonlinearGQASelector(ExactBlockSelector):
    """E02 constrained adaptation, using B0 exact mass in every live state."""

    @torch.no_grad()
    def __call__(self, context):
        if self.mode != "exact_mass":
            return super().__call__(context)
        self.metrics["calls"] += 1
        settings = context.settings
        blocks = math.ceil(context.k.shape[1] / settings.block_size)
        if blocks <= settings.topk:
            return select_with_scores(context, context.q.new_empty(0))
        p = self.logmass(context).softmax(-1)
        additive = p.sum(1)
        baseline = select_with_scores(context, additive)
        visible = torch.arange(blocks, device=p.device)[None] <= context.query_positions[:, None] // settings.block_size
        mandatory = mandatory_blocks(context, blocks)
        qk = additive.masked_fill(mandatory, torch.inf).masked_fill(~visible, -torch.inf)
        qk_count = min(settings.init_blocks + settings.local_blocks + settings.select_blocks, blocks)
        qk_ids = _stable_topk(qk, qk_count)
        qk_mask = torch.zeros_like(additive, dtype=torch.bool).scatter_(-1, qk_ids, True)
        result = baseline.clone()
        for h in range(p.shape[0]):
            for t in range(p.shape[2]):
                chosen = baseline[h, t]
                valid = chosen[chosen >= 0]
                baseline_mask = torch.zeros(blocks, device=p.device, dtype=torch.bool).scatter_(0, valid, True)
                fixed = baseline_mask & (~qk_mask[h, t] | mandatory[0, t])
                new, trace = improve_set(p[h, :, t], valid, fixed, visible[t])
                result[h, t].fill_(-1)
                result[h, t, :new.numel()] = new
                self.metrics["e02_states"] = self.metrics.get("e02_states", 0) + 1
                self.metrics["e02_swaps"] = self.metrics.get("e02_swaps", 0) + len(trace) - 1
                self.metrics["e02_risk_reduction"] = self.metrics.get("e02_risk_reduction", 0.) + trace[0] - trace[-1]
        return result
