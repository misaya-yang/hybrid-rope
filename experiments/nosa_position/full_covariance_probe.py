"""Diagnostic full second-cumulant routing without PCA/SVD or covariance truncation.

For every query, this reference reads all raw keys and computes CIS-tilted
mean/variance of q.K directly. It keeps the existing PC2 current-block exact
normalizer, causality, GQA normalization, NOSA quota and final reader. It is NOT
a cheap deployable selector. Comparing it with pair-PC2 and exact logmass
separates covariance projection error from higher-cumulant error.

The CLI delegates to the existing NOSA runner (including baseline/reuse flags).
It defaults to dry-run; --execute owns ROOT/queue.lock and respects ROOT/STOP.
Do not wrap it in another holder of the same lock. Use a new --output directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from . import run as base
from .exact_probe import ExactBlockSelector
from .runtime import (SelectionContext, dense_causal_attention, mandatory_blocks,
                      select_with_scores, selected_causal_attention)
from .selector_controls import BlockSummarySelector


@torch.no_grad()
def full_covariance_logmass(context: SelectionContext):
    """Return [KV,G,Q,blocks]; no explicit D×D matrix and no SVD.

    Population Var_w(q.K/sqrt(D)) equals q.T Cov_w(K) q / D. All historical
    completed blocks use this FULL variance. Each query's current block uses
    exact causal logmass, matching the original pair-PC2 normalization rule.
    FP64 input stays FP64 for independent CPU identity checks; runtime uses FP32.
    """
    q, k, cis, settings = context.q, context.k, context.cis, context.settings
    if q.ndim != 3 or k.ndim != 3 or cis.shape != k.shape[:2]:
        raise ValueError("expected Q[Hq,Q,D], K[KV,T,D], CIS[KV,T]")
    kvh, length, dim = k.shape
    heads, queries, qdim = q.shape
    if min(kvh, length, dim, heads, queries) < 1 or qdim != dim or heads % kvh:
        raise ValueError("invalid query/key dimensions or contiguous GQA grouping")
    if context.query_positions.shape != (queries,) or bool((context.query_positions < 0).any()) or bool((context.query_positions >= length).any()):
        raise ValueError("query positions must identify visible positions within K")
    dtype = torch.float64 if q.dtype == torch.float64 or k.dtype == torch.float64 else torch.float32
    group, size = heads // kvh, settings.block_size
    blocks = math.ceil(length / size)
    padding = blocks * size - length
    parts = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        query = q.to(dtype).reshape(kvh, group, queries, dim) / math.sqrt(dim)
        keys = k.to(dtype)[:, None].transpose(-1, -2)
        block_cis = F.pad(cis.to(dtype), (0, padding), value=-torch.inf).reshape(kvh, blocks, size)
        weights = block_cis.softmax(-1)
        log_cis_weight = block_cis.logsumexp(-1)
        endpoints = (torch.arange(blocks, device=q.device) + 1) * size - 1
        offsets = torch.arange(size, device=q.device)
        for begin in range(0, queries, settings.attention_query_chunk_size):
            end = min(queries, begin + settings.attention_query_chunk_size)
            positions = context.query_positions[begin:end]
            scores = query[:, :, begin:end] @ keys
            scores = F.pad(scores, (0, padding), value=0).reshape(kvh, group, end - begin, blocks, size)
            w = weights[:, None, None]
            mean = (w * scores).sum(-1)
            variance = (w * (scores - mean[..., None]).square()).sum(-1)
            logmass = log_cis_weight[:, None, None] + mean + 0.5 * variance
            logmass.masked_fill_(~(endpoints[None] <= positions[:, None])[None, None], -torch.inf)
            # Current blocks are exact even when this query ends a full block.
            current = positions // size
            index = current[None, None, :, None, None].expand(kvh, group, -1, 1, size)
            current_scores = scores.gather(-2, index).squeeze(-2)
            current_logits = current_scores + block_cis[:, current][:, None]
            visible = offsets[None] <= positions[:, None] % size
            exact_partial = current_logits.masked_fill(~visible[None, None], -torch.inf).logsumexp(-1)
            logmass.scatter_(-1, current[None, None, :, None].expand(kvh, group, -1, 1), exact_partial[..., None])
            parts.append(logmass)
    output = torch.cat(parts, dim=2)
    if bool(torch.isnan(output).any()) or bool(torch.isposinf(output).any()):
        raise FloatingPointError("invalid full second-order diagnostic score")
    return output


class FullCovarianceSelector(ExactBlockSelector):
    """Drop-in runner factory, preserving all unchanged control implementations."""
    def __init__(self, mode="full_covariance", **kwargs):
        super().__init__("pc2" if mode == "full_covariance" else mode, **kwargs)
        self.mode = mode
        self.metrics.update(full_covariance_raw_key_scores=0, diagnostic_dense_qk=mode == "full_covariance")

    def reset(self):
        super().reset()
        self.metrics.update(full_covariance_raw_key_scores=0, diagnostic_dense_qk=self.mode == "full_covariance")

    @torch.no_grad()
    def logmass(self, context):
        if self.mode != "full_covariance":
            return super().logmass(context)
        output = full_covariance_logmass(context)
        self.metrics["full_covariance_raw_key_scores"] += context.q.shape[0] * context.q.shape[1] * context.k.shape[1]
        return output


@torch.no_grad()
def diagnose_query_state(context: SelectionContext, *, max_queries=16):
    """Same-state pair/full/exact scores, error decomposition, quota and mass.

    Returns tensors for a caller to save alongside its row/layer provenance.
    Only competitive nonmandatory blocks enter scalar error summaries. This
    function receives no answer labels and does not run or change a model.
    """
    if context.q.shape[1] > max_queries:
        raise ValueError("slice the desired real query states explicitly; diagnostic is bounded")
    pair = BlockSummarySelector("pc2").logmass(context)
    full = full_covariance_logmass(context)
    exact = ExactBlockSelector("exact_mass").logmass(context)
    valid = torch.isfinite(exact)
    covariance_error = torch.where(valid, full - pair, 0)
    cumulant_error = torch.where(valid, exact - full, 0)
    total_error = torch.where(valid, exact - pair, 0)
    protected = mandatory_blocks(context, exact.shape[-1])
    competitive = valid & ~protected[None, None]
    scores = {"pair": pair, "full_covariance": full, "exact": exact}
    group_scores = {name: value.softmax(-1).sum(1) for name, value in scores.items()}
    selected = {name: select_with_scores(context, value) for name, value in group_scores.items()}
    support_masks = {}
    for name, ids in selected.items():
        mask = torch.zeros_like(group_scores[name], dtype=torch.int32)
        mask.scatter_add_(-1, ids.clamp_min(0), (ids >= 0).to(torch.int32))
        support_masks[name] = mask > 0
    changed_support = ((support_masks["pair"] ^ support_masks["exact"])
                       | (support_masks["full_covariance"] ^ support_masks["exact"]))
    decision_band = competitive & changed_support[:, None]
    exact_group_mean = group_scores["exact"] / exact.shape[1]
    retained = {}
    for name, ids in selected.items():
        mass = exact_group_mean.gather(-1, ids.clamp_min(0)) * (ids >= 0)
        retained[name] = mass.sum(-1)
    def average_absolute(value):
        return float(value[competitive].abs().mean()) if bool(competitive.any()) else None
    metrics = {"competitive_head_query_block_count": int(competitive.sum()),
        "cross_covariance_logmass_mae": average_absolute(covariance_error),
        "higher_cumulant_logmass_mae": average_absolute(cumulant_error),
        "total_pair_logmass_mae": average_absolute(total_error),
        "decomposition_residual_max": float((total_error - covariance_error - cumulant_error)[valid].abs().max()),
        "exact_retained_mass_mean": {name: float(value.mean()) for name, value in retained.items()},
        "interpretation": "full-pair includes projection plus numerical error; exact-full is second-cumulant remainder",
        "scope": "same actual Q/K/CIS state; quota/reader unchanged; no generation or deployability claim"}
    dense_output = dense_causal_attention(context).float()
    metrics["attention_output_relative_l2_error"] = {
        name: float((selected_causal_attention(context, ids).float() - dense_output).norm()
                    / dense_output.norm().clamp_min(1e-12))
        for name, ids in selected.items()}
    metrics["changed_support_head_query_block_count"] = int(decision_band.sum())
    metrics["changed_support_cross_covariance_mae"] = (
        float(covariance_error[decision_band].abs().mean()) if bool(decision_band.any()) else None)
    metrics["changed_support_higher_cumulant_mae"] = (
        float(cumulant_error[decision_band].abs().mean()) if bool(decision_band.any()) else None)
    return dict(logmass=scores, covariance_error=covariance_error, higher_cumulant_error=cumulant_error,
                total_error=total_error, competitive_mask=competitive, selected_blocks=selected,
                exact_retained_mass=retained, metrics=metrics,
                layer_idx=context.layer_idx, query_positions=context.query_positions.detach().clone())


def _run_existing(argv):
    """Register only in this process; original source files remain untouched."""
    original_hashes, original_modes, original_factory = base.source_hashes, base.MODES, base.BlockSummarySelector
    original_argv = sys.argv
    def source_hashes():
        extra = (Path(__file__), Path(__file__).with_name("exact_probe.py"))
        return {**original_hashes(), **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in extra}}
    try:
        base.source_hashes = source_hashes
        base.MODES = (*original_modes, "full_covariance")
        base.BlockSummarySelector = FullCovarianceSelector
        supplied = list(argv)
        if not any(arg == "--selectors" or arg.startswith("--selectors=") for arg in supplied):
            supplied += ["--selectors", "full_covariance"]
        sys.argv = [str(Path(__file__)), *supplied]
        return base.main()
    finally:
        base.source_hashes, base.MODES, base.BlockSummarySelector = original_hashes, original_modes, original_factory
        sys.argv = original_argv


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--queue-root", default="/root/autodl-tmp/position_overnight_20260909")
    parser.add_argument("--split", choices=("dev",), default="dev")
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    remaining = ["--split", options.split, *remaining]
    if "--help" in remaining or "-h" in remaining:
        return _run_existing(remaining)
    report = dict(status="EXECUTE_REQUESTED" if options.execute else "DRY_RUN",
                  runner_argv=remaining, selector="full_covariance",
                  lock=str(Path(options.queue_root) / "queue.lock"),
                  scope="dense-QK second-order diagnostic; preserves PC2 current-block/quotas/reader",
                  reuse="existing --baseline-cache for native/COBS; --reuse-from exact DEV directory")
    print(json.dumps(report), flush=True)
    if not options.execute:
        return report
    root = Path(options.queue_root)
    if not root.is_dir():
        raise ValueError("shared queue root must exist")
    with (root / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("shared queue busy; no model loaded") from exc
        if (root / "STOP").exists():
            raise RuntimeError("global STOP present; root must authorize and clear it before execution")
        return _run_existing(remaining)


if __name__ == "__main__":
    main()
