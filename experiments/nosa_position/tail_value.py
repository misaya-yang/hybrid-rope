"""E08: B0 exact selection with CIS-weighted per-block tail value means."""
import hashlib
import math
from pathlib import Path

import torch

from .exact_probe import ExactBlockSelector
from .runtime import select_with_scores


class TailValueSelector(ExactBlockSelector):
    active = None

    def __init__(self, mode="e08_tail_value", **kwargs):
        self.tail_value = mode == "e08_tail_value"
        super().__init__("exact_mass" if self.tail_value else mode, **kwargs)
        self.value_means = {}
        self.current = None

    @torch.no_grad()
    def __call__(self, context):
        TailValueSelector.active = self
        self.current = None
        if not self.tail_value:
            return super().__call__(context)
        self.metrics["calls"] += 1
        size = context.settings.block_size
        blocks = math.ceil(context.k.shape[1] / size)
        if int(context.query_positions[0]) == 0:
            self.value_means.pop(context.layer_idx, None)
        if blocks <= context.settings.topk:
            return select_with_scores(context, context.q.new_empty(0))
        p = self.logmass(context).softmax(-1)
        selected = select_with_scores(context, p.sum(1))
        full = context.k.shape[1] // size
        old = self.value_means.get(context.layer_idx)
        done = old.shape[1] if old is not None else 0
        if full > done:
            kvh, _, dim = context.v.shape
            values = context.v[:, done*size:full*size].float().reshape(kvh, full-done, size, dim)
            weights = context.cis[:, done*size:full*size].float().reshape(kvh, full-done, size).softmax(-1)
            new = (weights[..., None] * values).sum(-2)
            old = new if old is None else torch.cat((old, new), 1)
            self.value_means[context.layer_idx] = old
            self.metrics["e08_build_raw_v_elements"] = self.metrics.get("e08_build_raw_v_elements", 0) + values.numel()
        # Incomplete current blocks are mandatory; their omitted mass is zero.
        means = torch.nn.functional.pad(old, (0, 0, 0, blocks-full))
        self.current = (context, p, means)
        self.metrics["max_metadata_bytes"] = sum(v.nbytes for v in self.value_means.values())
        return selected

    @torch.no_grad()
    def read(self, context, selected, original_reader):
        sparse = original_reader(context, selected)
        if self.current is None:
            return sparse
        if self.current[0] is not context:
            raise RuntimeError("tail response must use the same live Q/K/V state")
        _, p, means = self.current
        valid = selected >= 0
        mask = torch.zeros_like(p[:, 0], dtype=torch.long).scatter_add_(
            -1, selected.clamp_min(0), valid.long()) > 0
        retained = (p * mask[:, None]).sum(-1)
        omitted = p.masked_fill(mask[:, None], 0)
        tail = torch.einsum("hgqb,hbd->hgqd", omitted, means)
        result = sparse.float().reshape(*p.shape[:3], context.q.shape[-1]) * retained[..., None] + tail
        self.metrics["e08_tail_mean_elements"] = self.metrics.get("e08_tail_mean_elements", 0) + means.numel()
        return result.reshape_as(sparse).to(sparse.dtype)


def main():
    from . import run as base, runtime
    original_hashes = base.source_hashes
    original_reader = runtime.selected_causal_attention
    def hashes():
        return {**original_hashes(), **{n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                      for n in ("tail_value.py", "exact_probe.py")}}
    def reader(context, selected):
        selector = TailValueSelector.active
        return selector.read(context, selected, original_reader) if selector is not None and selector.tail_value else original_reader(context, selected)
    base.MODES = (*base.MODES, "e08_tail_value")
    base.BlockSummarySelector = TailValueSelector
    base.source_hashes = hashes
    runtime.selected_causal_attention = reader
    base.main()


if __name__ == "__main__":
    main()
