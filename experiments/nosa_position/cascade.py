"""E04 candidate recall with candidate-only exact mass and mixed GQA denominators."""
import math
import torch
from .projected_distribution import ProjectedDistributionSelector
from .runtime import mandatory_blocks, select_with_scores, _stable_topk


def candidate_ids(context, approximate, multiplier):
    h, _, queries, blocks = approximate.shape
    mandatory = mandatory_blocks(context, blocks)
    visible = torch.arange(blocks, device=approximate.device)[None] <= context.query_positions[:, None] // context.settings.block_size
    slots = min(context.settings.init_blocks + context.settings.local_blocks + context.settings.select_blocks, blocks)
    free = (slots - mandatory.sum(-1)).clamp_min(0)
    score = approximate.softmax(-1).sum(1).masked_fill(mandatory | ~visible, -torch.inf)
    order = _stable_topk(score, min(blocks, slots * multiplier))
    valid = torch.arange(order.shape[-1], device=order.device)[None, None] < (free * multiplier)[None, :, None]
    valid = valid.expand(h, -1, -1).clone()
    valid &= visible[None].expand(h, -1, -1).gather(-1, order)
    valid &= ~mandatory[None].expand(h, -1, -1).gather(-1, order)
    mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, order, valid) | mandatory[None]
    width = int(mask.sum(-1).max())
    ids = torch.arange(blocks, device=order.device).expand_as(score).masked_fill(~mask, blocks).sort(-1).values[..., :width]
    return ids.masked_fill(ids == blocks, -1), mask


@torch.no_grad()
def exact_candidates(context, ids):
    """Read raw K only at supplied candidate IDs (including mandatory once)."""
    h, length, d = context.k.shape
    group = context.q.shape[0] // h
    queries, width = ids.shape[1:]
    q = context.q.float().reshape(h, group, queries, d) / math.sqrt(d)
    size = context.settings.block_size
    result = torch.full((h, group, queries, width), -torch.inf, device=q.device)
    reads = 0
    for head in range(h):
        for begin in range(0, queries, 16):
            end = min(begin + 16, queries)
            block_ids = ids[head, begin:end]
            positions = block_ids[..., None] * size + torch.arange(size, device=q.device)
            valid = (block_ids[..., None] >= 0) & (positions < length) & (positions <= context.query_positions[begin:end, None, None])
            safe = positions.clamp(0, length - 1)
            keys = context.k[head, safe].float()
            logits = torch.einsum('gqd,qctd->gqct', q[head, :, begin:end], keys)
            logits += context.cis[head, safe].float()[None]
            result[head, :, begin:end] = logits.masked_fill(~valid[None], -torch.inf).logsumexp(-1)
            reads += keys.numel()  # includes padding/repeated gathers, not an ideal count
    return result, reads


def replace_candidate_mass(approximate, ids, exact):
    mixed = approximate.clone()
    # Padded IDs must not overwrite block zero.
    for h in range(ids.shape[0]):
        q, c = torch.where(ids[h] >= 0)
        mixed[h, :, q, ids[h, q, c]] = exact[h, :, q, c]
    return mixed


def final_selection(context, mass, mask):
    scores = mass.softmax(-1).sum(1).masked_fill(~mask, -torch.inf)
    return select_with_scores(context, scores)


def cascade_select(context, approximate, multiplier, exact_callback):
    """The exact provider receives candidate IDs, never an all-block score request."""
    ids, mask = candidate_ids(context, approximate, multiplier)
    exact, reads = exact_callback(ids)
    mixed = replace_candidate_mass(approximate, ids, exact)
    return final_selection(context, mixed, mask), ids, reads


class CascadeSelector(ProjectedDistributionSelector):
    def __init__(self, multiplier=2, **kwargs):
        super().__init__('e04_empirical', **kwargs)
        self.multiplier = multiplier

    @torch.no_grad()
    def __call__(self, context):
        self.metrics['calls'] += 1
        if math.ceil(context.k.shape[1] / context.settings.block_size) <= context.settings.topk:
            return select_with_scores(context, context.q.new_empty(0))
        approximate = self.logmass(context)
        selected, ids, reads = cascade_select(
            context, approximate, self.multiplier,
            lambda candidate_ids: exact_candidates(context, candidate_ids))
        self.metrics['cascade_raw_k_elements'] = self.metrics.get('cascade_raw_k_elements', 0) + reads
        self.metrics['cascade_candidates'] = self.metrics.get('cascade_candidates', 0) + int((ids >= 0).sum())
        return selected
