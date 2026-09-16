#!/usr/bin/env python3
"""Report paired TailSpline/MrPro 128K (or added 64K) NIAH and PPL."""

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
    "niah_multivalue", "niah_multiquery",
)
ARMS = ("tailspline", "mrpro")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--native", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--rows-per-task", type=int, default=5)
    parser.add_argument("--ppl-documents", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.target != args.native * args.scale:
        raise ValueError("target/native is not the declared scale")

    runs = {}
    contracts = {}
    for arm in ARMS:
        run = args.root / f"runs/{arm}"
        status = json.loads((run / "status.json").read_text())
        expected_rows = len(TASKS) * args.rows_per_task
        if status != {"status": "COMPLETE", "rows": expected_rows, "lm_rows": args.ppl_documents}:
            raise ValueError(f"incomplete {arm} run: {status}")
        generations = read_jsonl(run / "generations.jsonl")
        lm = read_jsonl(run / "lm_rows.jsonl")
        if len(generations) != expected_rows or len(lm) != args.ppl_documents:
            raise ValueError(f"raw row count drift: {arm}")
        if {row["task"] for row in generations} != set(TASKS):
            raise ValueError(f"NIAH task coverage drift: {arm}")
        if any(int(row["length_cap"]) != args.target for row in generations):
            raise ValueError(f"generation length drift: {arm}")
        if any(int(row["length"]) != args.target for row in lm):
            raise ValueError(f"PPL length drift: {arm}")
        runs[arm] = ({row["eval_id"]: row for row in generations}, lm)
        contracts[arm] = json.loads((run / "contract.json").read_text())
        receipt = json.loads((args.root / f"tables/{arm}.json").read_text())
        frozen = receipt.get("table", receipt)
        active = contracts[arm].get("static_table") or {}
        if (
            active.get("values_float32") != frozen.get("values_float32")
            or active.get("gain") != frozen.get("gain")
        ):
            raise ValueError(f"{arm} runtime table differs from its frozen receipt")
    runtime_keys = (
        "batch_size", "prefill_chunk_size", "lm_prefill_chunk_size",
        "generation_order", "runtime_versions",
    )
    runtime_contracts = {
        arm: {key: contracts[arm].get(key) for key in runtime_keys} for arm in ARMS
    }
    if runtime_contracts["tailspline"] != runtime_contracts["mrpro"]:
        raise ValueError("TailSpline and MrPro used different runtime contracts")
    if set(runs["tailspline"][0]) != set(runs["mrpro"][0]):
        raise ValueError("generation prompts are not paired")
    if [(r["document"], r["length"]) for r in runs["tailspline"][1]] != [
        (r["document"], r["length"]) for r in runs["mrpro"][1]
    ]:
        raise ValueError("PPL documents are not paired")

    task_scores = {arm: defaultdict(list) for arm in ARMS}
    paired_by_task = defaultdict(list)
    for row_id in sorted(runs["tailspline"][0]):
        left = runs["tailspline"][0][row_id]
        right = runs["mrpro"][0][row_id]
        if any(left.get(key) != right.get(key) for key in ("task", "prompt_sha256", "references")):
            raise ValueError(f"paired prompt drift: {row_id}")
        task = left["task"]
        a = float(left["ruler_official_score"])
        b = float(right["ruler_official_score"])
        task_scores["tailspline"][task].append(a)
        task_scores["mrpro"][task].append(b)
        paired_by_task[task].append(a - b)
    summaries = {}
    for arm in ARMS:
        by_task = {task: float(np.mean(task_scores[arm][task])) for task in TASKS}
        summaries[arm] = {
            "passkey_niah_single_1": by_task["niah_single_1"],
            "niah8_task_macro": float(np.mean(list(by_task.values()))),
            "by_task": by_task,
        }
    rng = np.random.default_rng(20260915)
    draws = np.empty(10000, dtype=np.float64)
    passkey = np.empty(10000, dtype=np.float64)
    for draw in range(len(draws)):
        task_means = []
        for task in TASKS:
            values = np.asarray(paired_by_task[task], dtype=np.float64)
            task_means.append(float(np.mean(rng.choice(values, len(values), replace=True))))
        draws[draw] = float(np.mean(task_means))
        values = np.asarray(paired_by_task["niah_single_1"], dtype=np.float64)
        passkey[draw] = float(np.mean(rng.choice(values, len(values), replace=True)))

    ppl = {}
    document_nll = {}
    for arm in ARMS:
        rows = runs[arm][1]
        loss = sum(float(row["whole_loss_sum"]) for row in rows)
        count = sum(int(row["whole_target_count"]) for row in rows)
        nll = loss / count
        ppl[arm] = {"whole_nll": nll, "ppl": math.exp(nll), "documents": len(rows),
                    "target_tokens": count}
        document_nll[arm] = np.asarray([
            float(row["whole_loss_sum"]) / int(row["whole_target_count"]) for row in rows
        ])
    ppl_draws = np.empty(10000, dtype=np.float64)
    paired_docs = document_nll["tailspline"] - document_nll["mrpro"]
    for draw in range(len(ppl_draws)):
        ppl_draws[draw] = float(np.mean(rng.choice(paired_docs, len(paired_docs), replace=True)))

    report = {
        "status": "EXTREME_NIAH_PPL_REPORT_V1",
        "identity": {"model_id": args.model_id, "native_length": args.native,
                     "target_length": args.target, "scale": args.scale,
                     "target_over_native": args.target / args.native},
        "rows_per_task_per_arm": args.rows_per_task,
        "tasks": list(TASKS),
        "generation": {
            "arms": summaries,
            "delta_tailspline_minus_mrpro": {
                "passkey_niah_single_1": summaries["tailspline"]["passkey_niah_single_1"] - summaries["mrpro"]["passkey_niah_single_1"],
                "passkey_ci95": interval(passkey),
                "niah8_task_macro": summaries["tailspline"]["niah8_task_macro"] - summaries["mrpro"]["niah8_task_macro"],
                "niah8_task_macro_ci95": interval(draws),
            },
        },
        "ppl": {
            "arms": ppl,
            "delta_nll_tailspline_minus_mrpro": ppl["tailspline"]["whole_nll"] - ppl["mrpro"]["whole_nll"],
            "delta_nll_ci95": interval(ppl_draws),
        },
        "runtime": runtime_contracts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "model_id": args.model_id,
                      "target": args.target}))


if __name__ == "__main__":
    main()
