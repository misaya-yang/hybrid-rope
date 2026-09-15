#!/usr/bin/env python3
"""Report Native, Adam-Z5, and signed consensus-direction controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.native_z_enhancement_20260914.report import (
    NATURAL_TASKS,
    RULER_TASKS,
    atomic_json,
    interval,
    read_jsonl,
    task_bootstrap,
    task_macro,
)
from scripts.eval.longbench_metrics import qa_f1_score


ARMS = ("native", "native_z5", "consensus_plus", "consensus_minus")


def paired_nll_contrast(by_arm, candidate, baseline, *, draws=20_000, seed=20260918):
    rng = np.random.default_rng(seed)
    result = {}
    for length in (1024, 2048, 4096):
        candidate_rows = by_arm[candidate]
        baseline_rows = by_arm[baseline]
        documents = sorted({document for document, current in candidate_rows if current == length})
        delta = np.asarray([
            candidate_rows[(document, length)]["nll"] - baseline_rows[(document, length)]["nll"]
            for document in documents
        ])
        sampled = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
        result[str(length)] = {
            "documents": len(documents),
            "candidate_nll": float(np.mean([
                candidate_rows[(document, length)]["nll"] for document in documents
            ])),
            "baseline_nll": float(np.mean([
                baseline_rows[(document, length)]["nll"] for document in documents
            ])),
            "delta_candidate_minus_baseline": float(delta.mean()),
            "paired_document_bootstrap_ci95": interval(sampled),
            "probability_delta_lt_zero": float(np.mean(sampled < 0.0)),
        }
    return result


def load_heldout(original_path: Path, consensus_path: Path):
    by_arm = defaultdict(dict)
    for row in read_jsonl(original_path) + read_jsonl(consensus_path):
        arm = str(row["arm"])
        key = (int(row["document"]), int(row["length"]))
        if key in by_arm[arm]:
            raise ValueError(f"duplicate held-out NLL row: {arm}/{key}")
        by_arm[arm][key] = row
    if set(by_arm) != set(ARMS):
        raise ValueError(f"held-out arms drift: {sorted(by_arm)}")
    reference = set(by_arm["native"])
    if any(set(by_arm[arm]) != reference for arm in ARMS):
        raise ValueError("held-out NLL rows are not exactly paired")
    return by_arm


def load_task_runs(paths: dict[str, Path]):
    runs = {}
    for arm, directory in paths.items():
        status = json.loads((directory / "status.json").read_text())
        rows = read_jsonl(directory / "generations.jsonl")
        if status != {"status": "COMPLETE", "rows": 229, "lm_rows": 0} or len(rows) != 229:
            raise ValueError(f"incomplete task arm: {arm}/{status}/{len(rows)}")
        mapping = {str(row["row_id"]): row for row in rows}
        if len(mapping) != 229:
            raise ValueError(f"duplicate task rows: {arm}")
        runs[arm] = mapping
    prompts = set(runs["native"])
    if any(set(runs[arm]) != prompts for arm in ARMS):
        raise ValueError("task arms are not exactly paired")
    for row_id in prompts:
        reference = runs["native"][row_id]
        for arm in ARMS[1:]:
            row = runs[arm][row_id]
            for key in ("task", "prompt_sha256", "references", "input_tokens"):
                if row.get(key) != reference.get(key):
                    raise ValueError(f"task input drift: {arm}/{row_id}/{key}")
        if reference["task"] in NATURAL_TASKS:
            for arm in ARMS:
                row = runs[arm][row_id]
                if abs(qa_f1_score(row["output_text"], row["references"]) - row["whole_response_f1"]) > 1e-12:
                    raise ValueError(f"natural-QA score drift: {arm}/{row_id}")
    return runs


def task_panel(runs, ids, tasks, score, *, cluster_documents, seed):
    arms = {}
    for arm in ARMS:
        macro, by_task = task_macro(runs[arm], ids, tasks, score)
        arms[arm] = {"macro": macro, "by_task": by_task}
    contrasts = {}
    for offset, (candidate, baseline) in enumerate((
        ("consensus_plus", "native"),
        ("consensus_minus", "native"),
        ("consensus_plus", "consensus_minus"),
        ("consensus_plus", "native_z5"),
    )):
        inference = task_bootstrap(
            runs[baseline], runs[candidate], ids, tasks, score,
            draws=20_000, seed=seed + offset, cluster_documents=cluster_documents,
        )
        contrasts[f"{candidate}_minus_{baseline}"] = {
            "estimate": arms[candidate]["macro"] - arms[baseline]["macro"],
            "by_task": {
                task: arms[candidate]["by_task"][task] - arms[baseline]["by_task"][task]
                for task in tasks
            },
            **inference,
        }
    return {"rows": len(ids), "arms": arms, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--native-optimization", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--z5-run", type=Path, required=True)
    parser.add_argument("--plus-run", type=Path, required=True)
    parser.add_argument("--minus-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.assets / "manifest.json").read_text())
    consensus = json.loads((args.consensus / "consensus_result.json").read_text())
    if consensus.get("decision") != "ADVANCE_TO_HELDOUT_TASKS":
        raise ValueError("consensus direction did not advance to held-out tasks")
    heldout = load_heldout(
        args.native_optimization / "heldout_nll_rows.jsonl",
        args.consensus / "heldout_nll_rows.jsonl",
    )
    nll = {}
    for offset, (candidate, baseline) in enumerate((
        ("native_z5", "native"),
        ("consensus_plus", "native"),
        ("consensus_minus", "native"),
        ("consensus_plus", "consensus_minus"),
        ("consensus_plus", "native_z5"),
    )):
        nll[f"{candidate}_minus_{baseline}"] = paired_nll_contrast(
            heldout, candidate, baseline, seed=20260918 + offset,
        )

    runs = load_task_runs({
        "native": args.native_run,
        "native_z5": args.z5_run,
        "consensus_plus": args.plus_run,
        "consensus_minus": args.minus_run,
    })
    ids = list(runs["native"])
    ruler_ids = [row_id for row_id in ids if runs["native"][row_id]["task"] in RULER_TASKS]
    natural_ids = [row_id for row_id in ids if runs["native"][row_id]["task"] in NATURAL_TASKS]
    if len(ruler_ids) != 130 or len(natural_ids) != 99:
        raise ValueError("consensus task-panel coverage drift")
    ruler = task_panel(
        runs, ruler_ids, RULER_TASKS, lambda row: float(row["ruler_official_score"]),
        cluster_documents=False, seed=20260923,
    )
    natural = task_panel(
        runs, natural_ids, NATURAL_TASKS, lambda row: float(row["whole_response_f1"]),
        cluster_documents=True, seed=20260927,
    )

    internal = consensus["split_evaluations"]
    native_segments = np.asarray(internal["native"]["internal_confirm"]["segment_nll"])
    plus_segments = np.asarray(internal["consensus_plus"]["internal_confirm"]["segment_nll"])
    segment_delta = plus_segments - native_segments
    plus_native_nll = nll["consensus_plus_minus_native"]["4096"]
    plus_minus_nll = nll["consensus_plus_minus_consensus_minus"]["4096"]
    task_success = (
        ruler["contrasts"]["consensus_plus_minus_native"]["ci95"][0] > 0.0
        or natural["contrasts"]["consensus_plus_minus_native"]["ci95"][0] > 0.0
    )
    decision = {
        "positive_first_order_margin": bool(consensus["certificate"]["margin"] > 0.0),
        "selection_chose_nonzero_alpha": bool(consensus["best_alpha"] > 0.0),
        "heldout_4k_nll_ci_below_zero": bool(plus_native_nll["paired_document_bootstrap_ci95"][1] < 0.0),
        "plus_beats_minus_4k_nll_ci": bool(plus_minus_nll["paired_document_bootstrap_ci95"][1] < 0.0),
        "internal_confirm_segments_improved": int(np.sum(segment_delta < 0.0)),
        "ruler_or_natural_ci_above_zero": bool(task_success),
    }
    decision["strong_consensus_native_enhancement"] = bool(
        decision["positive_first_order_margin"]
        and decision["selection_chose_nonzero_alpha"]
        and decision["heldout_4k_nll_ci_below_zero"]
        and decision["plus_beats_minus_4k_nll_ci"]
        and decision["internal_confirm_segments_improved"] >= 3
        and decision["ruler_or_natural_ci_above_zero"]
    )
    report = {
        "status": "OLMO_NATIVE_Z5_CONSENSUS_CONFIRMATION_V1",
        "scientific_question": (
            "Does a split-consistent signed LM-gradient direction improve a frozen "
            "checkpoint inside the same five-dimensional Native Z5 space?"
        ),
        "model": manifest["model"],
        "consensus": consensus,
        "heldout_nll": nll,
        "internal_confirm_segment_delta_plus_minus_native": segment_delta.tolist(),
        "ruler4k": ruler,
        "natural_qa4k": natural,
        "decision": decision,
        "claim_boundary": (
            "Single checkpoint and checkpoint-calibrated direction. The max-min margin is "
            "a local first-order certificate; held-out results test, rather than follow from, it."
        ),
    }
    atomic_json(args.out, report)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
