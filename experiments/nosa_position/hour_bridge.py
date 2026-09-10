"""Full-covariance DEV generations plus bounded common-state diagnostics.

Observe the first two predetermined DEV inputs per task, at the last prompt Q
and first actually ingested answer Q. No oracle state affects any selector.
All requested rows still complete full free generation with the full-cov arm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from . import run as base
from .full_covariance_probe import diagnose_query_state, main as full_cov_main
from .runtime import dense_causal_attention, select_with_scores, selected_causal_attention
from .tail_pair import TailPairSelector


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    cells = defaultdict(list)
    for row in rows:
        if row["split"] == "dev" and row["length_cap"] == 16384 and row["task"] in {
                "niah_single_1", "niah_multikey_1", "niah_multiquery", "vt"}:
            cells[row["task"]].append(row)
    wanted = {r["row_id"] for cell in cells.values() for r in sorted(cell, key=lambda x: x["row_id"])[:2]}
    by_tokens = {tuple(r["prompt_ids"]): r for r in rows if r["row_id"] in wanted}
    out = Path(args.output)
    original_generate, original_hashes = base.generate, base.source_hashes

    def source_hashes():
        files = (Path(__file__), Path(__file__).with_name("tail_pair.py"))
        return {**original_hashes(), **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

    def generate(model, ids, max_new_tokens, eos_ids, chunk_size, device):
        row = by_tokens.get(tuple(ids[0].tolist()))
        if row is None or getattr(model.selector, "mode", None) != "full_covariance":
            tokens, timing = original_generate(model, ids, max_new_tokens, eos_ids, chunk_size, device)
            return tokens, {**timing, "timing_includes_diagnostics": False}
        targets = {ids.numel() - 1, ids.numel()}
        records, diagnostic_seconds = [], 0.0
        previous = model.trace_callback

        def observe(context, selected):
            nonlocal diagnostic_seconds
            if previous is not None:
                previous(context, selected)
            for position in targets:
                found = (context.query_positions == position).nonzero(as_tuple=True)[0]
                if not found.numel():
                    continue
                if context.q.is_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                index = int(found[0])
                point = replace(context, q=context.q[:, index:index + 1],
                                query_positions=context.query_positions[index:index + 1])
                result = diagnose_query_state(point)
                tail_scores = TailPairSelector().logmass(point).softmax(-1).sum(1)
                tail_set = select_with_scores(point, tail_scores)
                exact_mass = result["logmass"]["exact"].softmax(-1).mean(1)
                tail_mass = (exact_mass.gather(-1, tail_set.clamp_min(0)) * (tail_set >= 0)).sum(-1)
                dense = dense_causal_attention(point).float()
                error = (selected_causal_attention(point, tail_set).float() - dense).norm() / dense.norm().clamp_min(1e-12)
                metrics = result["metrics"]
                metrics["exact_retained_mass_mean"]["tail1"] = float(tail_mass.mean())
                metrics["attention_output_relative_l2_error"]["tail1"] = float(error)
                records.append({"row_id": row["row_id"], "task": row["task"],
                    "layer_idx": point.layer_idx, "query_position": position,
                    "state_trajectory": "full_covariance free generation",
                    "query_kind": "last_prompt" if position == ids.numel() - 1 else "first_ingested_answer",
                    **metrics})
                if context.q.is_cuda:
                    torch.cuda.synchronize()
                diagnostic_seconds += time.perf_counter() - start

        model.trace_callback = observe
        try:
            tokens, timing = original_generate(model, ids, max_new_tokens, eos_ids, chunk_size, device)
        finally:
            model.trace_callback = previous
        with (out / "common_state.jsonl").open("a") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        return tokens, {**timing, "timing_includes_diagnostics": True,
                        "diagnostic_observer_seconds": diagnostic_seconds,
                        "common_state_observations": len(records)}

    try:
        base.source_hashes = source_hashes
        base.generate = generate
        full_cov_main()
    finally:
        base.source_hashes, base.generate = original_hashes, original_generate


if __name__ == "__main__":
    main()
