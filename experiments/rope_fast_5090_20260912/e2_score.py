#!/usr/bin/env python3
"""Audit raw E2 generations and report the prespecified long-pool contrasts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.eval.longbench_metrics import qa_f1_score
from scripts.experiments.olmo_fast_screen.prepare import sha_file


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def macro(data: dict[str, dict], ids: list[str]) -> tuple[float, dict[str, float]]:
    cells = defaultdict(list)
    for row_id in ids:
        row = data[row_id]
        cells[row["task"]].append(row["whole_response_f1"])
    if set(cells) != set(TASKS):
        raise ValueError("macro is missing a task")
    means = {task: sum(cells[task]) / len(cells[task]) for task in TASKS}
    return sum(means.values()) / len(TASKS), means


def stratified_cluster_draws(inputs, arms, coefficients, ids, *, draws=20_000, seed=20260912):
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in TASKS:
        clusters = defaultdict(list)
        for row_id in ids:
            if inputs[row_id]["task"] == task:
                clusters[inputs[row_id]["document_cluster_id"]].append(row_id)
        if not clusters:
            raise ValueError(f"missing bootstrap task: {task}")
        values = []
        for row_ids in clusters.values():
            values.append(np.mean([sum(weight * arms[arm][row_id]["whole_response_f1"]
                                             for arm, weight in coefficients.items())
                                   for row_id in row_ids]))
        values = np.asarray(values)
        task_draws.append(values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1))
    return np.mean(task_draws, axis=0)


def bootstrap_test(draws: np.ndarray) -> dict:
    p = min(1.0, 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0))))
    return {"ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "two_sided_p": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running, total = {}, 0.0, len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    generation = json.loads((args.prepared / "generation_config.json").read_text())
    eos_ids = generation["eos_token_id"]
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    status = json.loads((args.run / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status["completed_arms"] != manifest["arms"]:
        raise ValueError("seven-arm run is incomplete")
    inputs = {row["row_id"]: row for row in read_rows(args.prepared / "inputs.jsonl")}
    arms = {}
    for arm in manifest["arms"]:
        raw_path = args.run / f"{arm}.jsonl"
        receipt = json.loads((args.run / f"{arm}.json").read_text())
        if receipt["status"] != "COMPLETE" or receipt["raw_sha256"] != sha_file(raw_path):
            raise ValueError(f"raw receipt mismatch: {arm}")
        data = read_rows(raw_path)
        if len(data) != 778 or {row["row_id"] for row in data} != set(inputs):
            raise ValueError(f"unmatched rows: {arm}")
        mapping = {}
        for row in data:
            source = inputs[row["row_id"]]
            for key in ("task", "prompt_sha256", "input_tokens", "max_new_tokens", "references"):
                if row[key] != source[key]:
                    raise ValueError(f"input contract drift: {arm}/{row['row_id']}/{key}")
            if len(row["generated_ids"]) > row["max_new_tokens"]:
                raise ValueError("generation exceeded original task cap")
            ended = bool(row["generated_ids"] and row["generated_ids"][-1] in eos_ids)
            if row["ended_eos"] != ended:
                raise ValueError("terminal EOS record drift")
            hit_cap = len(row["generated_ids"]) == row["max_new_tokens"] and not ended
            if row["hit_cap"] != hit_cap:
                raise ValueError("generation cap record drift")
            score = qa_f1_score(row["output_text"], row["references"])
            if abs(score - row["whole_response_f1"]) > 1e-12:
                raise ValueError("whole-response scoring drift")
            mapping[row["row_id"]] = row
        arms[arm] = mapping
    long_ids = [row_id for row_id, row in inputs.items() if row["input_tokens"] > 4_096]
    short_ids = [row_id for row_id, row in inputs.items() if row["input_tokens"] <= 4_096]
    if len(long_ids) != 631:
        raise ValueError("long pool identity drift")
    summaries = {}
    for arm, data in arms.items():
        value, tasks = macro(data, long_ids)
        lengths = [len(data[x]["generated_ids"]) for x in long_ids]
        summaries[arm] = {"long_rows": 631, "five_task_equal_macro_f1": value,
                          "by_task": tasks, "eos": sum(data[x]["ended_eos"] for x in long_ids),
                          "hit_cap": sum(data[x]["hit_cap"] for x in long_ids),
                          "empty_outputs": sum(not data[x]["output_text"].strip() for x in long_ids),
                          "mean_generated_tokens_including_eos": float(np.mean(lengths))}
    def delta(left: str, right: str) -> float:
        return summaries[left]["five_task_equal_macro_f1"] - summaries[right]["five_task_equal_macro_f1"]
    contrast_coefficients = {
        "bm_g4_minus_mruni_g4": {"bm_g4": 1, "mruni_g4": -1},
        "bm_g4_minus_official_yarn_g4": {"bm_g4": 1, "official_yarn_g4": -1},
    }
    inference = {}
    for index, (name, coefficients) in enumerate(contrast_coefficients.items()):
        inference[name] = bootstrap_test(stratified_cluster_draws(inputs, arms, coefficients, long_ids, seed=20260912 + index))
    adjusted = holm({name: cell["two_sided_p"] for name, cell in inference.items()})
    for name in inference:
        inference[name]["holm_adjusted_p"] = adjusted[name]
    interaction_coefficients = {"bm_g4": 1, "mrpro_g4": -1, "bm_g1": -1, "mrpro_g1": 1}
    interaction_inference = bootstrap_test(stratified_cluster_draws(inputs, arms, interaction_coefficients, long_ids, seed=20260914))
    short_by_arm = {}
    for arm, data in arms.items():
        task_cells = {}
        for task in TASKS:
            task_ids = [x for x in short_ids if inputs[x]["task"] == task]
            if task_ids:
                task_cells[task] = {"n": len(task_ids), "mean_f1": float(np.mean([data[x]["whole_response_f1"] for x in task_ids]))}
        short_by_arm[arm] = {"rows": len(short_ids), "by_task": task_cells,
                             "eos": sum(data[x]["ended_eos"] for x in short_ids),
                             "hit_cap": sum(data[x]["hit_cap"] for x in short_ids),
                             "empty_outputs": sum(not data[x]["output_text"].strip() for x in short_ids),
                             "mean_generated_tokens_including_eos": float(np.mean([len(data[x]["generated_ids"]) for x in short_ids]))}
    result = {"status": "COMPLETE", "primary_pool": "631 rows with input_tokens > 4096",
              "metric": "five-task equal macro whole-response token F1",
              "arms": summaries,
              "primary_contrasts": {"bm_g4_minus_mruni_g4": delta("bm_g4", "mruni_g4"),
                                    "bm_g4_minus_official_yarn_g4": delta("bm_g4", "official_yarn_g4")},
              "primary_inference": inference,
              "gain_interaction": {"estimate": (delta("bm_g4", "mrpro_g4") - delta("bm_g1", "mrpro_g1")), **interaction_inference},
              "inference_method": "20,000 percentile paired cluster-bootstrap draws, resampling frozen document_cluster_id within each task and equal-macro averaging tasks; two-sided p=2*min(Pr(draw<=0),Pr(draw>=0)); Holm step-down adjustment over the two primary contrasts",
              "short_window_147": {"note": "Per-task only; NarrativeQA has no short rows, so no five-task macro is reported.", "arms": short_by_arm},
              "scope": "fixed previously evaluated eligible pool; strong-baseline completion, not a new independent sample"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["primary_contrasts"], sort_keys=True))


if __name__ == "__main__":
    main()
