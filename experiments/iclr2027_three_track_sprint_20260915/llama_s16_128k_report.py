#!/usr/bin/env python3
"""Build the paired Llama S=16 TailSpline/MrPro 128K gate report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
ARMS = ("tailspline", "mrpro")
LENGTH = 131072


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_run(path: Path, arm: str):
    status = json.loads((path / "status.json").read_text())
    contract = json.loads((path / "contract.json").read_text())
    generated = read_jsonl(path / "generations.jsonl")
    lm = read_jsonl(path / "lm_rows.jsonl")
    if status != {"status": "COMPLETE", "rows": 130, "lm_rows": 10}:
        raise ValueError(f"incomplete S16 arm {arm}: {status}")
    if len(generated) != 130 or len(lm) != 10:
        raise ValueError(f"S16 row coverage drift: {arm}")
    mapping = {str(row["row_id"]): row for row in generated}
    if len(mapping) != 130:
        raise ValueError(f"duplicate S16 generation row: {arm}")
    expected_lm_chunk = int(contract.get("lm_prefill_chunk_size", -1))
    actual_lm_chunks = {int(row.get("lm_prefill_chunk_size", -1)) for row in lm}
    normalized_expected = 0 if expected_lm_chunk == 0 or expected_lm_chunk >= LENGTH else expected_lm_chunk
    if actual_lm_chunks != {normalized_expected}:
        raise ValueError(f"S16 LM strategy receipt drift: {arm}/{actual_lm_chunks}/{normalized_expected}")
    return mapping, lm, contract


def task_summary(mapping):
    by_task = {}
    for task in TASKS:
        rows = [row for row in mapping.values() if row["task"] == task]
        if len(rows) != 10:
            raise ValueError(f"S16 task coverage drift: {task}")
        by_task[task] = float(np.mean([row["ruler_official_score"] for row in rows]))
    return {"macro": float(np.mean(list(by_task.values()))), "by_task": by_task}


def task_bootstrap(runs, *, draws=20_000, seed=20261001):
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in TASKS:
        row_ids = sorted(row_id for row_id, row in runs["tailspline"][0].items() if row["task"] == task)
        delta = np.asarray([
            runs["tailspline"][0][row_id]["ruler_official_score"]
            - runs["mrpro"][0][row_id]["ruler_official_score"]
            for row_id in row_ids
        ])
        task_draws.append(delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1))
    values = np.mean(task_draws, axis=0)
    return {
        "draws": draws, "seed": seed,
        "resampling": "paired rows within task; 13 tasks fixed and equally weighted",
        "mean_delta": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def ppl_summary(rows):
    loss = sum(float(row["whole_loss_sum"]) for row in rows)
    tokens = sum(int(row["whole_target_count"]) for row in rows)
    nll = loss / tokens
    return {"documents": len(rows), "target_tokens": tokens, "whole_nll": nll, "whole_ppl": math.exp(nll)}


def ppl_bootstrap(runs, *, draws=20_000, seed=20261002):
    mappings = {
        arm: {int(row["document"]): row for row in runs[arm][1]} for arm in ARMS
    }
    if set(mappings["tailspline"]) != set(range(10)) or set(mappings["mrpro"]) != set(range(10)):
        raise ValueError("S16 PPL documents are not exactly paired")
    rng = np.random.default_rng(seed)
    delta_nll = np.empty(draws)
    delta_ppl = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(10, size=10)
        values = {}
        for arm in ARMS:
            selected = [mappings[arm][int(document)] for document in sampled]
            nll = sum(float(row["whole_loss_sum"]) for row in selected) / sum(
                int(row["whole_target_count"]) for row in selected
            )
            values[arm] = nll
        delta_nll[draw] = values["tailspline"] - values["mrpro"]
        delta_ppl[draw] = math.exp(values["tailspline"]) - math.exp(values["mrpro"])
    return {
        "draws": draws, "seed": seed,
        "resampling": "paired ProofPile documents",
        "delta_nll": {
            "mean": float(delta_nll.mean()),
            "ci95": [float(value) for value in np.quantile(delta_nll, [0.025, 0.975])],
            "probability_lt_zero": float(np.mean(delta_nll < 0.0)),
        },
        "delta_ppl": {
            "mean": float(delta_ppl.mean()),
            "ci95": [float(value) for value in np.quantile(delta_ppl, [0.025, 0.975])],
        },
    }


def advance_to_independent_confirmation(ruler_inference: dict) -> bool:
    ci95 = ruler_inference.get("ci95")
    if not isinstance(ci95, list) or len(ci95) != 2:
        raise ValueError("S16 RULER gate lacks a paired 95% interval")
    return bool(float(ci95[0]) > 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="ARM=RUN_DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.run:
        arm, value = item.split("=", 1)
        paths[arm] = Path(value)
    if set(paths) != set(ARMS):
        raise ValueError("S16 report requires exactly tailspline and mrpro")
    runs = {arm: load_run(path, arm) for arm, path in paths.items()}
    if set(runs["tailspline"][0]) != set(runs["mrpro"][0]):
        raise ValueError("S16 generation prompts are not paired")
    for row_id in runs["tailspline"][0]:
        left, right = runs["tailspline"][0][row_id], runs["mrpro"][0][row_id]
        for key in ("task", "length_cap", "prompt_sha256", "references"):
            if left.get(key) != right.get(key):
                raise ValueError(f"S16 prompt identity drift: {row_id}/{key}")
    runtime_keys = (
        "prefill_chunk_size", "generation_prefill_strategy",
        "lm_prefill_chunk_size", "lm_execution_strategy",
        "batch_size", "runtime_versions",
    )
    runtime_contracts = {
        arm: {key: runs[arm][2].get(key) for key in runtime_keys} for arm in ARMS
    }
    if runtime_contracts["tailspline"] != runtime_contracts["mrpro"]:
        raise ValueError("S16 arms used different runtime contracts")
    summaries = {arm: task_summary(runs[arm][0]) for arm in ARMS}
    ppl = {arm: ppl_summary(runs[arm][1]) for arm in ARMS}
    ruler_delta = summaries["tailspline"]["macro"] - summaries["mrpro"]["macro"]
    ppl_delta = ppl["tailspline"]["whole_nll"] - ppl["mrpro"]["whole_nll"]
    ruler_inference = task_bootstrap(runs)
    ppl_inference = ppl_bootstrap(runs)
    report = {
        "status": "TAILSPLINE_MRPRO_LLAMA_S16_128K_GATE_COMPLETE_V1",
        "model": "Meta-Llama-3-8B-Instruct",
        "scale": 16,
        "native_length": 8192,
        "evaluation_length": LENGTH,
        "runtime_contract": runtime_contracts["tailspline"],
        "rows_per_arm": 130,
        "ruler": {
            "tasks": list(TASKS), "rows_per_task": 10,
            "arms": summaries, "delta_tailspline_minus_mrpro": ruler_delta,
            "paired_inference": ruler_inference,
        },
        "proofpile_ppl": {
            "arms": ppl, "delta_nll_tailspline_minus_mrpro": ppl_delta,
            "paired_inference": ppl_inference,
        },
        "decision": {
            "advance_to_independent_128k_confirmation": advance_to_independent_confirmation(ruler_inference),
            "rule": "advance only if the paired task-equal 95% interval at 128K is strictly positive",
            "ppl_role": "independent language-modeling health signal; not averaged with or used to gate the task confirmation",
        },
        "claim_boundary": (
            "Single 128K endpoint gate with 10 rows/task and 10 ProofPile documents; "
            "a positive gate authorizes but does not replace the 8/16/32/64/128K curve."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "decision": report["decision"]}))


if __name__ == "__main__":
    main()
