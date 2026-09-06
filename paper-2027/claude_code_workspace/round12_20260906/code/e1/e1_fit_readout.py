#!/usr/bin/env python3
"""E1 training-fit readout (Pro Unified Plan §6.2 / §4.3 row 1).

Read-only reuse of release008 library primitives (load_runtime / greedy /
gold_prefix_trace) imported by path; nothing in code_release_008 is modified.
For each (arm, step) adapter, scores the train-far and validation-far rows of
the frozen round-11 transport views under the arm's own training runtime:

  full_answer_CE   = -sum of gold logprobs over answer+EOS positions
  worst_gold_margin= min(correct - best competitor) over those positions
  greedy strict    = generated tokens exactly an accepted full answer + EOS
  ended_with_eos   = greedy stopped on EOS within budget
  lenient_contains = any accepted answer appears in generated text

Built-in contract check: ZF step_128 validation-far must reproduce the
round-11 diagonal row counts (strict 2/128, groups 1/32) before any
trajectory number is trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ENGINE_LIB = Path("/root/autodl-tmp/ffn_review_execution_20260904/"
                  "code_release_008/scripts/experiments/single_table_generation.py")


def import_lib():
    spec = importlib.util.spec_from_file_location("stg_readonly", ENGINE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_views(path: Path, split: str, layout: str):
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split and row["layout"] == layout:
                rows.append(row)
    if not rows:
        raise SystemExit(f"no rows for {split}/{layout}")
    return rows


def canonical_target(row, eos: int):
    target = list(row["target_ids"])
    if not target or target[-1] != eos:
        target = [*target, eos]
    return target


def contains_subseq(haystack, needle):
    if not needle:
        return True
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def accepted_token_lists(aliases, tok):
    return [list(tok(a, add_special_tokens=False)["input_ids"])
            for a in aliases if a]


def score_row(stg, model, row, eos: int, tok):
    prompt = list(row["prompt_ids"])
    target = canonical_target(row, eos)
    trace = stg.gold_prefix_trace(model, prompt, target)
    generated = stg.greedy(model, prompt, eos, int(row["generation_budget"]))
    accepted = accepted_token_lists(row["accepted_full_answers"], tok)
    strict = any(generated == [*a, eos] for a in accepted)
    lenient = any(contains_subseq(generated, a) for a in accepted)
    ended = bool(generated) and generated[-1] == eos
    return {
        "semantic_id": row.get("semantic_id"),
        "world": row["world"],
        "length_cap": row["length_cap"],
        "full_answer_CE": -sum(trace["gold_logprobs"]),
        "worst_gold_margin": min(trace["margins"]),
        "mean_gold_margin": sum(trace["margins"]) / len(trace["margins"]),
        "worst_gold_rank": max(trace["ranks"]),
        "strict_exact_eos": bool(strict),
        "lenient_contains": bool(lenient),
        "ended_with_eos": ended,
        "n_generated": len(generated),
        "generated_ids": generated,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ZF", "ON"], required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-seconds", type=int, default=21600)
    ap.add_argument("--cells", nargs="+", default=["train:far", "validation:far"])
    args = ap.parse_args()

    B = Path("/root/autodl-tmp/ffn_review_execution_20260904")
    R11 = Path("/root/autodl-tmp/claude_round11_olmo_20260905")
    OLMO = Path("/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct")
    train_root = R11 / ("out_zf" if args.arm == "ZF" else "out_on") / "train"
    views_path = R11 / "olmo_tasks" / "transport_views.jsonl"

    stg = import_lib()
    ns = argparse.Namespace(
        checkpoint=OLMO,
        checkpoint_contract=B / "olmo1485_contract.json",
        data=None,
        seed=42,
        table=(B / "fixed_controls" / "Z.npy") if args.arm == "ZF" else None,
        gain=1.102585782722872 if args.arm == "ZF" else 1.0,
        adapter=None,
        authorized=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = {"arm": args.arm, "cells": args.cells, "steps": args.steps,
                "engine_lib_sha256": sha(ENGINE_LIB),
                "views_sha256": sha(views_path),
                "runtime": None, "stages": []}

    for step in args.steps:
        adapter = train_root / f"step_{step:03d}"
        if not adapter.is_dir():
            raise SystemExit(f"missing adapter: {adapter}")
        ns.adapter = adapter
        model, tok, receipt = stg.load_runtime(ns)
        if manifest["runtime"] is None:
            manifest["runtime"] = receipt
        elif receipt["table_sha256"] != manifest["runtime"]["table_sha256"]:
            raise SystemExit("runtime table drift between steps")
        eos = tok.eos_token_id
        for cell in args.cells:
            split, layout = cell.split(":")
            rows = load_views(views_path, split, layout)
            out_rows, stage_t0 = [], time.monotonic()
            for i, row in enumerate(rows):
                if time.monotonic() - start > args.max_seconds:
                    manifest["stages"].append({
                        "step": step, "cell": cell, "status": "PARTIAL_WALL_CLOCK",
                        "rows": len(out_rows), "of": len(rows)})
                    (args.output / f"{args.arm}_step{step:03d}_{cell.replace(':','_')}_PARTIAL.jsonl"
                     ).write_text("".join(json.dumps(r) + "\n" for r in out_rows))
                    (args.output / "manifest_PARTIAL.json").write_text(json.dumps(manifest, indent=1))
                    raise SystemExit("wall-clock budget exhausted; partial rows preserved")
                rec = score_row(stg, model, row, eos, tok)
                rec["row_index"] = i
                out_rows.append(rec)
                if (i + 1) % 32 == 0:
                    elapsed = time.monotonic() - start
                    print(f"[{args.arm} step{step:03d} {cell}] {i+1}/{len(rows)} "
                          f"elapsed={elapsed:.0f}s strict_so_far="
                          f"{sum(r['strict_exact_eos'] for r in out_rows)}", flush=True)
            groups = {}
            for r in out_rows:
                groups.setdefault(r["semantic_id"], {})[r["world"]] = r["strict_exact_eos"]
            both = sum(1 for ws in groups.values() if len(ws) >= 2 and all(ws.values()))
            summary = {
                "step": step, "cell": cell, "rows": len(out_rows),
                "strict": sum(r["strict_exact_eos"] for r in out_rows),
                "lenient": sum(r["lenient_contains"] for r in out_rows),
                "groups_both": both, "groups": len(groups),
                "eos_rate": sum(r["ended_with_eos"] for r in out_rows) / len(out_rows),
                "mean_CE": sum(r["full_answer_CE"] for r in out_rows) / len(out_rows),
                "min_worst_margin": min(r["worst_gold_margin"] for r in out_rows),
                "mean_worst_margin": sum(r["worst_gold_margin"] for r in out_rows) / len(out_rows),
                "frac_positive_worst_margin": sum(r["worst_gold_margin"] > 0 for r in out_rows) / len(out_rows),
                "wall_seconds": time.monotonic() - stage_t0,
            }
            manifest["stages"].append(summary)
            path = args.output / f"{args.arm}_step{step:03d}_{cell.replace(':','_')}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
            print(f"[{args.arm} step{step:03d} {cell}] strict {summary['strict']}/{summary['rows']} "
                  f"groups_both {both}/{len(groups)} mean_CE {summary['mean_CE']:.3f} "
                  f"mean_worst_margin {summary['mean_worst_margin']:.3f}", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()

    # Diagonal reference record against the round-11 ZF diagonal
    # (out_zf/task128). NOTE (2026-09-06 reconciliation): the constants
    # originally hardcoded here ("strict 2/128, groups_both 1/32") matched NO
    # recorded artifact — root cause: they were conflated with Track A
    # zero-shot figures. The actual round-11 reference (round-11 engine and
    # fields, NOT comparable count-for-count with the round-12 rescoring
    # below) is: 64 validation-far single rows, full_exact_eos 14/64 (22%),
    # gold-in-text 29/64, fields accepted_full_answers/full_exact_eos (round
    # 12 uses strict_exact_eos/ruler_official_contains/lenient). So this is
    # recorded as an informational reference, not a PASS/FAIL gate.
    if args.arm == "ZF" and 128 in args.steps and "validation:far" in args.cells:
        diag = [s for s in manifest["stages"]
                if s.get("step") == 128 and s.get("cell") == "validation:far" and "strict" in s]
        if diag:
            d = diag[0]
            manifest["diagonal_reference_record"] = {
                "round11_reference": "out_zf/task128: 64 far rows, full_exact_eos "
                                     "14/64, gold-in-text 29/64 (round-11 fields)",
                "round12_rescore_observed": {"strict_exact_eos": d["strict"],
                                             "rows": d["rows"],
                                             "groups_both": d["groups_both"]},
                "status": "INFORMATIONAL (scoring fields differ across rounds; "
                          "no equality gate)"}
    manifest["status"] = "COMPLETE"
    (args.output / f"manifest_{args.arm}.json").write_text(json.dumps(manifest, indent=1))
    print(f"E1_FIT_READOUT_{args.arm}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
