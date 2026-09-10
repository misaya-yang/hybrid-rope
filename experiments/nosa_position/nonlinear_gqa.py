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


@torch.no_grad()
def improve_sets_batched(probabilities, selected, fixed, visible):
    """Batched equivalent of improve_set, without a host sync per query/head."""
    p = probabilities.double()  # [states,G,B]
    states, groups, blocks = p.shape
    slots = selected.shape[-1]
    ids = selected.clamp_min(0).clone()
    valid = selected >= 0
    rows = torch.arange(states, device=p.device)
    def retained(index):
        return (p.gather(2, index[:, None].expand(-1, groups, -1)) * valid[:, None]).sum(-1)
    mass = retained(ids)
    initial = (mass.reciprocal() - 1).sum(-1)
    risk = initial.clone()
    swaps = torch.zeros(states, device=p.device, dtype=torch.long)
    for _ in range(2):
        inside = torch.zeros_like(visible, dtype=torch.long).scatter_add_(1, ids, valid.long()) > 0
        outside = visible & ~inside
        additive = p.sum(1).masked_fill(~outside, -torch.inf)
        linear_ids = _stable_topk(additive, min(slots, blocks))
        head_ids = _stable_topk(p.masked_fill(~outside[:, None], -torch.inf), min(2, blocks)).flatten(1)
        # Unique pool via a block mask, then physical-ID order for deterministic ties.
        pool_mask = torch.zeros_like(visible).scatter_(1, torch.cat((linear_ids, head_ids), 1), True) & outside
        width = min(blocks, slots + 2 * groups)
        block_ids = torch.arange(blocks, device=p.device).expand(states, -1)
        pool = block_ids.masked_fill(~pool_mask, blocks).sort(-1).values[:, :width]
        pool_valid = pool < blocks
        pool = pool.clamp_max(blocks - 1)
        out_mass = p.gather(2, ids[:, None].expand(-1, groups, -1))
        in_mass = p.gather(2, pool[:, None].expand(-1, groups, -1))
        next_mass = mass[:, :, None, None] - out_mass[:, :, :, None] + in_mass[:, :, None, :]
        risks = (next_mass.reciprocal() - 1).sum(1)
        movable = valid & ~fixed.gather(1, ids)
        legal = movable[:, :, None] & pool_valid[:, None] & (next_mass > 0).all(1) & torch.isfinite(risks)
        risks = risks.masked_fill(~legal, torch.inf)
        best = risks.flatten(1).argmin(-1)
        outgoing, incoming = best // width, best % width
        proposal = ids.clone()
        proposal[rows, outgoing] = pool[rows, incoming]
        new_mass = retained(proposal)
        new_risk = (new_mass.reciprocal() - 1).sum(-1)
        accept = (risks.flatten(1)[rows, best] < risk) & (new_risk < risk) & torch.isfinite(new_risk)
        ids = torch.where(accept[:, None], proposal, ids)
        # Keep valid slots first; invalid placeholders must never duplicate block 0.
        ids = ids.masked_fill(~valid, blocks).sort(-1).values.clamp_max(blocks - 1)
        mass = torch.where(accept[:, None], new_mass, mass)
        risk = torch.where(accept, new_risk, risk)
        swaps += accept.long()
    return ids.masked_fill(~valid, -1), swaps, initial - risk


class NonlinearGQASelector(ExactBlockSelector):
    """E02 constrained adaptation, using B0 exact mass in every live state."""

    def __init__(self, mode="e02_nonlinear", **kwargs):
        self.e02 = mode == "e02_nonlinear"
        super().__init__("exact_mass" if self.e02 else mode, **kwargs)

    @torch.no_grad()
    def __call__(self, context):
        if not self.e02:
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
        kvh, queries = baseline.shape[:2]
        selected_mask = torch.zeros_like(additive, dtype=torch.long).scatter_add_(
            -1, baseline.clamp_min(0), (baseline >= 0).long()) > 0
        fixed = selected_mask & (~qk_mask | mandatory[None])
        chosen, swaps, decrease = improve_sets_batched(
            p.permute(0, 2, 1, 3).reshape(kvh * queries, p.shape[1], blocks),
            baseline.reshape(kvh * queries, -1), fixed.reshape(kvh * queries, blocks),
            visible.expand(kvh, -1, -1).reshape(kvh * queries, blocks))
        self.metrics["e02_states"] = self.metrics.get("e02_states", 0) + kvh * queries
        self.metrics["e02_swaps"] = self.metrics.get("e02_swaps", 0) + int(swaps.sum())
        self.metrics["e02_risk_reduction"] = self.metrics.get("e02_risk_reduction", 0.) + float(decrease.sum())
        return chosen.reshape_as(baseline)


def main():
    import hashlib
    from pathlib import Path
    from . import run as base
    original_hashes = base.source_hashes
    def source_hashes():
        names = ("nonlinear_gqa.py", "exact_probe.py")
        return {**original_hashes(), **{n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}}
    base.source_hashes = source_hashes
    base.MODES = (*base.MODES, "e02_nonlinear")
    base.BlockSummarySelector = NonlinearGQASelector
    base.main()


if __name__ == "__main__":
    main()
