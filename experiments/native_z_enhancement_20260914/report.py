#!/usr/bin/env python3
"""Build the paired Native-vs-Z5 Native-window confirmation report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from scripts.eval.longbench_metrics import qa_f1_score


RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
NATURAL_TASKS = ("hotpotqa", "2wikimqa", "qasper")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def interval(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_nll(rows, *, draws=20_000, seed=20260914):
    by_arm = defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][(int(row["document"]), int(row["length"]))] = row
    if set(by_arm) != {"native", "native_z5"} or set(by_arm["native"]) != set(by_arm["native_z5"]):
        raise ValueError("held-out NLL rows are not paired")
    rng = np.random.default_rng(seed)
    result = {}
    for length in (1024, 2048, 4096):
        documents = sorted({document for document, current in by_arm["native"] if current == length})
        delta = np.asarray([
            by_arm["native_z5"][(document, length)]["nll"] - by_arm["native"][(document, length)]["nll"]
            for document in documents
        ])
        sampled = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
        result[str(length)] = {
            "documents": len(documents),
            "native_nll": float(np.mean([by_arm["native"][(document, length)]["nll"] for document in documents])),
            "candidate_nll": float(np.mean([by_arm["native_z5"][(document, length)]["nll"] for document in documents])),
            "delta_candidate_minus_native": float(delta.mean()),
            "paired_document_bootstrap_ci95": interval(sampled),
            "probability_delta_lt_zero": float(np.mean(sampled < 0)),
        }
    return result


def task_macro(mapping, ids, tasks, score):
    by_task = {}
    for task in tasks:
        selected = [score(mapping[row_id]) for row_id in ids if mapping[row_id]["task"] == task]
        if not selected:
            raise ValueError(f"missing task {task}")
        by_task[task] = float(np.mean(selected))
    return float(np.mean(list(by_task.values()))), by_task


def task_bootstrap(
    native, candidate, ids, tasks, score, *, draws=20_000, seed=20260915,
    cluster_documents=False,
):
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in tasks:
        selected = [row_id for row_id in ids if native[row_id]["task"] == task]
        if cluster_documents:
            clusters = defaultdict(list)
            for row_id in selected:
                clusters[native[row_id].get("document_cluster_id") or row_id].append(row_id)
            sums = np.asarray([
                sum(score(candidate[row_id]) - score(native[row_id]) for row_id in group)
                for group in clusters.values()
            ])
            counts = np.asarray([len(group) for group in clusters.values()])
            sampled = rng.integers(len(sums), size=(draws, len(sums)))
            task_draws.append(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
        else:
            delta = np.asarray([score(candidate[row_id]) - score(native[row_id]) for row_id in selected])
            task_draws.append(delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1))
    values = np.mean(task_draws, axis=0)
    return {"bootstrap_mean": float(values.mean()), "ci95": interval(values), "probability_delta_gt_zero": float(np.mean(values > 0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.assets / "manifest.json").read_text())
    optimization = json.loads((args.optimization / "optimization_result.json").read_text())
    nll = paired_nll(read_jsonl(args.optimization / "heldout_nll_rows.jsonl"))
    runs = {}
    for name, directory in (("native", args.native_run), ("native_z5", args.candidate_run)):
        status = json.loads((directory / "status.json").read_text())
        values = read_jsonl(directory / "generations.jsonl")
        if status != {"status": "COMPLETE", "rows": 229, "lm_rows": 0} or len(values) != 229:
            raise ValueError(f"incomplete task run: {name}")
        mapping = {str(row["row_id"]): row for row in values}
        if len(mapping) != 229:
            raise ValueError(f"duplicate task row: {name}")
        runs[name] = mapping
    if set(runs["native"]) != set(runs["native_z5"]):
        raise ValueError("Native/Z5 task prompts are not paired")
    for row_id in runs["native"]:
        left, right = runs["native"][row_id], runs["native_z5"][row_id]
        for key in ("task", "prompt_sha256", "references", "input_tokens"):
            if left.get(key) != right.get(key):
                raise ValueError(f"task input drift: {row_id}/{key}")
        if left["task"] in NATURAL_TASKS:
            for row in (left, right):
                if abs(qa_f1_score(row["output_text"], row["references"]) - row["whole_response_f1"]) > 1e-12:
                    raise ValueError(f"natural-QA score drift: {row_id}")

    ruler_ids = [row_id for row_id, row in runs["native"].items() if row["task"] in RULER_TASKS]
    natural_ids = [row_id for row_id, row in runs["native"].items() if row["task"] in NATURAL_TASKS]
    if len(ruler_ids) != 130 or len(natural_ids) != 99:
        raise ValueError("task panel row counts drift")
    ruler_score = lambda row: float(row["ruler_official_score"])
    natural_score = lambda row: float(row["whole_response_f1"])
    ruler_native, ruler_native_tasks = task_macro(runs["native"], ruler_ids, RULER_TASKS, ruler_score)
    ruler_z5, ruler_z5_tasks = task_macro(runs["native_z5"], ruler_ids, RULER_TASKS, ruler_score)
    natural_native, natural_native_tasks = task_macro(runs["native"], natural_ids, NATURAL_TASKS, natural_score)
    natural_z5, natural_z5_tasks = task_macro(runs["native_z5"], natural_ids, NATURAL_TASKS, natural_score)
    ruler_inference = task_bootstrap(runs["native"], runs["native_z5"], ruler_ids, RULER_TASKS, ruler_score)
    natural_inference = task_bootstrap(
        runs["native"], runs["native_z5"], natural_ids, NATURAL_TASKS,
        natural_score, seed=20260916, cluster_documents=True,
    )

    nll_success = nll["4096"]["paired_document_bootstrap_ci95"][1] < 0.0
    ruler_success = ruler_inference["ci95"][0] > 0.0
    natural_success = natural_inference["ci95"][0] > 0.0
    report = {
        "status": "OLMO_NATIVE_Z5_CONFIRMATION_REPORT_V1",
        "scientific_question": "Can post-hoc z-only calibration improve a frozen mature checkpoint inside its Native window?",
        "model": manifest["model"],
        "parameterization": "Z5; five effective interior z degrees, exact Native endpoints/support, gain=1, zero model-weight updates",
        "optimization": optimization,
        "heldout_nll": nll,
        "ruler4k": {
            "rows": len(ruler_ids), "tasks": len(RULER_TASKS),
            "native_macro": ruler_native, "candidate_macro": ruler_z5,
            "delta": ruler_z5 - ruler_native,
            "by_task_delta": {task: ruler_z5_tasks[task] - ruler_native_tasks[task] for task in RULER_TASKS},
            "inference": ruler_inference,
        },
        "natural_qa4k": {
            "rows": len(natural_ids), "rows_by_task": manifest["natural_qa"]["rows_by_task"],
            "native_macro": natural_native, "candidate_macro": natural_z5,
            "delta": natural_z5 - natural_native,
            "by_task_delta": {task: natural_z5_tasks[task] - natural_native_tasks[task] for task in NATURAL_TASKS},
            "inference": natural_inference,
            "limitation": "Untruncated frozen in-window pool is unbalanced and HotpotQA has only five rows.",
        },
        "decision": {
            "heldout_4k_nll_ci_below_zero": nll_success,
            "ruler_ci_above_zero": ruler_success,
            "natural_qa_ci_above_zero": natural_success,
            "strong_native_enhancement": bool(nll_success and (ruler_success or natural_success)),
            "directional_native_enhancement": bool(nll["4096"]["delta_candidate_minus_native"] < 0.0 and ((ruler_z5 - ruler_native) > 0.0 or (natural_z5 - natural_native) > 0.0)),
        },
        "claim_boundary": "Single checkpoint and checkpoint-calibrated table. A positive result shows Native geometric RoPE is not optimal for this frozen checkpoint; it does not provide a target-free universal z rule.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
