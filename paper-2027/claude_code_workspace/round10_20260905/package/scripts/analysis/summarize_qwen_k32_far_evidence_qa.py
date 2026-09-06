#!/usr/bin/env python3
"""Hash-bound paired summary for the frozen Qwen K32 far-evidence QA panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.analysis import summarize_qwen_k32_natural_nll as base
from scripts.data.prepare_qwen_k32_far_evidence_qa import ROWS_PER_TASK, TASKS

BOOTSTRAP_SEED = 202609031
BOOTSTRAP_SAMPLES = 10_000
STATUS = "QWEN_K32_FAR_EVIDENCE_QA_COMPLETE"


def summarize(root: Path) -> dict:
    paths = {name: root / filename for name, filename in (
        ("result", "results.json"), ("manifest", "run_manifest.json"),
        ("examples", "examples.jsonl"))}
    result = json.loads(paths["result"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    rows = [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()]
    hashes = {name + "_sha256": base.sha256(path) for name, path in paths.items()}
    if (
        result.get("status") != STATUS or result.get("stage") != "full"
        or manifest.get("status") != "QWEN_K32_FAR_EVIDENCE_QA_FROZEN"
        or manifest.get("stage") != "full"
        or manifest.get("checkpoint_weight_sha256") != base.WEIGHT_SHA256
        or manifest.get("config_sha256") != base.CONFIG_SHA256
        or manifest.get("tasks") != list(TASKS)
        or manifest.get("rows_per_task") != ROWS_PER_TASK
        or manifest.get("arm_order") != list(base.ARMS)
        or manifest.get("declared_arm_order") != list(base.ARMS)
        or manifest.get("use_cache") is not True or manifest.get("compile") is not False
        or manifest.get("model_updates") != 0 or manifest.get("profile_selection") is not False
        or manifest.get("all_profiles_loaded_before_inference") is not True
        or result.get("examples_sha256") != hashes["examples_sha256"]
        or result.get("run_manifest_sha256") != hashes["manifest_sha256"]
    ):
        raise ValueError("far-evidence QA run contract drift")
    bound = {key: base.checked_hash(manifest.get(key), key) for key in (
        "checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256",
        "data_rows_sha256", "nll_receipt_sha256", "script_sha256",
        "model_source_sha256", "attention_source_sha256")}
    evaluator = Path(__file__).resolve().parents[1] / "eval" / "eval_qwen_k32_far_evidence_qa.py"
    if bound["script_sha256"] != base.sha256(evaluator):
        raise ValueError("far-evidence evaluator hash differs from run")
    profiles = base.validate_profiles(manifest)
    expected, cells = set(), {}
    for task in TASKS:
        task_rows = sorted({row.get("row_sha256") for row in rows if row.get("task") == task})
        if len(task_rows) != ROWS_PER_TASK or None in task_rows:
            raise ValueError("far-evidence task row identity is incomplete")
        expected.update((arm, task, row_hash) for arm in base.ARMS for row_hash in task_rows)
    for row in rows:
        key = row.get("arm"), row.get("task"), row.get("row_sha256")
        profile = profiles.get(row.get("arm"))
        score = row.get("score")
        if (
            key not in expected or key in cells or profile is None
            or row.get("table_sha256_float32") != profile["tensor_sha256"]
            or row.get("attention_scaling") != profile["attention_scaling"]
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or type(row.get("generated_tokens")) is not int or row["generated_tokens"] <= 0
            or row["generated_tokens"] > row.get("generation_tokens_budget", 0)
            or not isinstance(row.get("prediction"), str)
        ):
            raise ValueError("invalid or duplicated far-evidence QA row")
        cells[key] = float(score)
    if set(cells) != expected or result.get("rows") != len(expected):
        raise ValueError("far-evidence QA grid is incomplete")

    per_task = []
    task_row_ids = {}
    for task in TASKS:
        ids = sorted({key[2] for key in expected if key[1] == task})
        task_row_ids[task] = ids
        per_task.append(np.array([[cells[arm, task, row_id] for arm in base.ARMS]
                                  for row_id in ids], dtype=np.float64))
    values = np.stack(per_task)
    means = values.mean(axis=1)
    stored = result.get("means")
    for arm_index, arm in enumerate(base.ARMS):
        for task_index, task in enumerate(TASKS):
            if not math.isclose(stored[arm][task], float(means[task_index, arm_index]),
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError("stored far-evidence mean differs from raw rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, ROWS_PER_TASK,
                           size=(BOOTSTRAP_SAMPLES, len(TASKS), ROWS_PER_TASK))
    draws = np.stack([values[task][indices[:, task]].mean(axis=1)
                      for task in range(len(TASKS))], axis=1)
    macro_draws, macro = draws.mean(axis=1), means.mean(axis=0)

    def contrast(left: int, right: int) -> dict:
        delta = macro_draws[:, left] - macro_draws[:, right]
        return {"mean_macro_delta": float(macro[left] - macro[right]),
                "paired_stratified_ci95": np.quantile(delta, [.025, .975]).tolist()}

    index_native = contrast(1, 0)
    index_yarn = contrast(1, 2)
    task_deltas = {task: float(means[index, 1] - means[index, 0])
                   for index, task in enumerate(TASKS)}
    positive_tasks = sum(value > 0 for value in task_deltas.values())
    utilization = index_native["paired_stratified_ci95"][0] > 0 and positive_tasks >= 2
    yarn_ci = index_yarn["paired_stratified_ci95"]
    ranking = ("INDEX_FAVORED" if yarn_ci[0] > 0 else
               "YARN_FAVORED" if yarn_ci[1] < 0 else "UNRESOLVED")
    return {
        "status": "QWEN_K32_FAR_EVIDENCE_QA_SUMMARIZED",
        "tasks": list(TASKS), "rows_per_task": ROWS_PER_TASK,
        "means": {task: {arm: float(means[task_index, arm_index])
                          for arm_index, arm in enumerate(base.ARMS)}
                  for task_index, task in enumerate(TASKS)},
        "macro": {arm: float(macro[index]) for index, arm in enumerate(base.ARMS)},
        "index_minus_native": index_native,
        "index_minus_yarn": index_yarn,
        "index_minus_native_by_task": task_deltas,
        "classification": {
            "natural_far_evidence_utilization": "PASS" if utilization else "NOT_PASS",
            "index_vs_yarn": ranking,
        },
        "decision_rule": (
            "PASS iff the paired stratified index-minus-Native macro CI lower bound is above "
            "zero and at least two of three task point deltas are positive."
        ),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES,
                      "unit": "rows resampled within each fixed task, jointly across arms"},
        "identity": {**bound, "profiles": profiles, "task_row_ids": task_row_ids},
        "raw_hashes": hashes,
        "summary_code_sha256": base.sha256(Path(__file__)),
        "evidence_limit": "Derived 64K far-evidence panel with 10 rows/task; official-style F1, not official LongBench.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.root)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_OR_INCOMPLETE_FAR_EVIDENCE_QA"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(report["status"])
    return 0 if report["status"] == "QWEN_K32_FAR_EVIDENCE_QA_SUMMARIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
