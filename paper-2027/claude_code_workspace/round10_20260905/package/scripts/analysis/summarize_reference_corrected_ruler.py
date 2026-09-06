#!/usr/bin/env python3
"""Summarize completed paired reference-corrected RULER arms, never select one."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ARMS = ("native", "physical_x", "normalized_index", "official_yarn")
TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")
TERMINAL_STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
BOOTSTRAP_SEED = 202609024
BOOTSTRAP_SAMPLES = 10_000
CONTRASTS = {"physical_minus_index": "normalized_index",
             "physical_minus_native": "native", "physical_minus_yarn": "official_yarn"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_least(value: float, threshold: float) -> bool:
    # Only absorb floating-point subtraction at an exact inclusive threshold.
    return value >= threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def load_panel(root: Path) -> dict:
    panel = {}
    for name in ARMS:  # Deliberately never glob preflight/partial/other profile directories.
        directory = root / name
        result_path, examples_path = directory / "results.json", directory / "examples.jsonl"
        result = json.loads(result_path.read_text())
        if result.get("status") != TERMINAL_STATUS:
            raise ValueError(f"{name} is not terminal {TERMINAL_STATUS}")
        rows = [json.loads(line) for line in examples_path.read_text().splitlines() if line.strip()]
        item = {"result": result, "rows": rows,
                "hashes": {"results_sha256": file_hash(result_path), "examples_sha256": file_hash(examples_path)}}
        manifest_path = directory / "run_manifest.json"
        if manifest_path.exists():
            item["run_manifest"] = json.loads(manifest_path.read_text())
            item["hashes"]["run_manifest_sha256"] = file_hash(manifest_path)
        expected_hash = result.get("results", {}).get("examples_sha256")
        if expected_hash is not None and expected_hash != item["hashes"]["examples_sha256"]:
            raise ValueError(f"{name} raw examples hash differs from terminal receipt")
        panel[name] = item
    return panel


def validate_panel(panel: dict, reference_length: int, target_length: int):
    if set(panel) != set(ARMS):
        raise ValueError("panel must contain exactly the four registered arms")
    if reference_length <= 0 or target_length <= reference_length:
        raise ValueError("target length must exceed the positive reference length")
    protocols, keyed = {}, {}
    for arm in ARMS:
        item, result = panel[arm], panel[arm]["result"]
        if result.get("status") != TERMINAL_STATUS:
            raise ValueError(f"{arm} is incomplete or preflight-only")
        protocol = result.get("protocol", item.get("run_manifest"))
        if not isinstance(protocol, dict):
            raise ValueError(f"{arm} has no bound protocol")
        if "run_manifest" in item and protocol != item["run_manifest"]:
            raise ValueError(f"{arm} terminal protocol differs from run manifest")
        tasks, lengths, count = protocol.get("tasks"), protocol.get("lengths"), protocol.get("limit_per_cell")
        if not isinstance(tasks, list) or len(tasks) != 4 or set(tasks) != set(TASKS):
            raise ValueError("protocol must use exactly the four registered task families")
        if (not isinstance(lengths, list) or len(set(lengths)) != len(lengths)
                or any(type(length) is not int for length in lengths)
                or reference_length not in lengths or target_length not in lengths
                or min(lengths) != reference_length or max(lengths) != target_length):
            raise ValueError("requested reference/target disagree with protocol lengths")
        if type(count) is not int or count <= 0:
            raise ValueError("protocol limit_per_cell must be a positive integer")
        data_hash = protocol.get("data_manifest_sha256")
        if not isinstance(data_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", data_hash):
            raise ValueError("protocol must bind a data manifest SHA-256")
        if protocol.get("expected_data_manifest_sha256") not in (None, data_hash):
            raise ValueError("actual and expected data manifest hashes differ")
        if result.get("data", {}).get("manifest_sha256", data_hash) != data_hash:
            raise ValueError("terminal data receipt differs from protocol")
        if protocol.get("profile_target_length") not in (None, target_length):
            raise ValueError("profile target differs from requested target")
        if protocol.get("reference_length") not in (None, reference_length):
            raise ValueError("profile reference differs from requested reference")
        if protocol.get("table_factor") is not None and protocol["table_factor"] != target_length / reference_length:
            raise ValueError("profile scale differs from target/reference")
        protocols[arm] = protocol
        values = {}
        for row in item["rows"]:
            task, length, index = row.get("task"), row.get("nominal_length"), row.get("local_index")
            if (task not in TASKS or type(length) is not int or length not in lengths
                    or type(index) is not int or not 0 <= index < count):
                raise ValueError(f"{arm} has an unregistered raw row identity")
            key = task, length, index
            if key in values:
                raise ValueError(f"{arm} has duplicate raw row identities")
            score = row.get("official_task_score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{arm} has invalid official task scores")
            if not isinstance(row.get("prediction", row.get("fullpred")), str):
                raise ValueError(f"{arm} lacks complete prediction text")
            values[key] = float(score)
        expected = {(task, length, index) for task in TASKS for length in lengths for index in range(count)}
        if set(values) != expected:
            raise ValueError(f"{arm} lacks complete paired rows in every registered cell")
        aggregate = result.get("results", {})
        if aggregate.get("examples", len(values)) != len(values):
            raise ValueError(f"{arm} terminal row count differs from raw rows")
        for task, cells in aggregate.get("cells", {}).items():
            for length_string, cell in cells.items():
                length = int(length_string)
                if task not in TASKS or length not in lengths:
                    raise ValueError(f"{arm} terminal aggregate contains extra cells")
                mean = sum(values[task, length, index] for index in range(count)) / count
                if cell.get("rows") != count or not math.isclose(cell.get("official_task_score", math.inf), mean, abs_tol=1e-12):
                    raise ValueError(f"{arm} terminal aggregate differs from raw rows")
        keyed[arm] = values
    base = protocols["native"]
    for arm, protocol in protocols.items():
        for field in ("data_manifest_sha256", "tasks", "lengths", "limit_per_cell"):
            if protocol[field] != base[field]:
                raise ValueError(f"{arm} has mismatched paired protocol field: {field}")
        for field in ("checkpoint_sha256", "expected_native_sha256", "native_context_length", "model_type"):
            if any(field in item for item in protocols.values()) and protocol.get(field) != base.get(field):
                raise ValueError(f"{arm} has mismatched checkpoint identity: {field}")
    return keyed, protocols


def paired_bootstrap_cell(values: np.ndarray, rng) -> np.ndarray:
    """Equal task-macro, resampling row pairs within each task simultaneously for all arms."""
    if values.ndim != 3 or values.shape[0] != len(TASKS) or values.shape[2] != len(ARMS):
        raise ValueError("expected task-by-row-by-arm scores")
    draws = np.zeros((BOOTSTRAP_SAMPLES, len(ARMS)))
    for task_values in values:
        indices = rng.integers(0, len(task_values), size=(BOOTSTRAP_SAMPLES, len(task_values)))
        draws += task_values[indices].mean(axis=1) / len(TASKS)
    return draws


def summarize_panel(panel: dict, reference_length: int = 4096, target_length: int = 8192) -> dict:
    keyed, protocols = validate_panel(panel, reference_length, target_length)
    protocol = protocols["native"]
    count = protocol["limit_per_cell"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores, comparisons = {}, {}
    for length in sorted(protocol["lengths"]):
        values = np.array([[[keyed[arm][task, length, index] for arm in ARMS]
                            for index in range(count)] for task in TASKS])
        draws = paired_bootstrap_cell(values, rng)
        scores[str(length)] = {arm: {"macro": float(values[:, :, column].mean()),
            "per_task": {task: float(values[task_index, :, column].mean()) for task_index, task in enumerate(TASKS)},
            "rows_per_task": count} for column, arm in enumerate(ARMS)}
        comparisons[str(length)] = {}
        for name, comparator in CONTRASTS.items():
            index = ARMS.index(comparator)
            point = float((values[:, :, 1] - values[:, :, index]).mean())
            interval = [float(value) for value in np.quantile(draws[:, 1] - draws[:, index], [.025, .975])]
            tail = .05 / (2 * len(protocol["lengths"]))
            length_family = [float(value) for value in np.quantile(
                draws[:, 1] - draws[:, index], [tail, 1 - tail])]
            comparisons[str(length)][name] = {"delta": point, "paired_stratified_ci95": interval,
                "bonferroni_length_family_sensitivity_ci95": length_family,
                "per_task_delta": {task: float((values[t, :, 1] - values[t, :, index]).mean()) for t, task in enumerate(TASKS)}}
    reference = scores[str(reference_length)]
    target = scores[str(target_length)]
    single = target["physical_x"]["per_task"]["niah_single_1"]
    macro_gain = target["physical_x"]["macro"] - target["native"]["macro"]
    resolver_passed = at_least(single, .8) and at_least(macro_gain, .1)
    retention = {}
    for arm in ARMS:
        ratio = reference[arm]["macro"] / reference["native"]["macro"] if reference["native"]["macro"] > 0 else None
        retention[arm] = {"ratio": ratio, "threshold": .875,
                          "status": "UNDEFINED" if ratio is None else "PASS" if at_least(ratio, .875) else "FAIL"}
    return {"status": "REFERENCE_CORRECTED_RULER_PANEL_COMPLETE", "reference_length": reference_length,
        "target_length": target_length, "data_manifest_sha256": protocol["data_manifest_sha256"],
        "tasks": list(TASKS), "lengths": sorted(protocol["lengths"]), "rows_per_task": count,
        "scores": scores, "paired_comparisons": comparisons,
        "stage2_entrance": {"status": ("PASS" if resolver_passed else "FAIL")
            if target_length == 2 * reference_length else "NOT_APPLICABLE",
            "scope": "Registered s2-to-s4 entrance only; a completed s4 panel cannot authorize s8 or another stage",
            "resolver_thresholds_met": resolver_passed,
            "physical_single_score": single, "single_success_equivalent": single * count,
            "single_rows": count, "single_threshold": .8, "physical_macro_minus_native": macro_gain,
            "macro_gain_threshold": .1, "reference_retention_is_separate": True},
        "reference_retention": retention,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES, "confidence": .95,
            "unit": "paired rows resampled within each fixed task family; equal task-macro weights",
            "scope": "conditional on one checkpoint, fixed tasks, data and decoding; not checkpoint/training-seed uncertainty",
            "multiplicity_sensitivity": "Post hoc Bonferroni over tested lengths within each fixed contrast; not a global correction over all contrasts or confirmatory selection rule",
            "small_difference_policy": "report point and interval together; no coordinate identification from .01-.03 differences"},
        "raw_hashes": {arm: panel[arm].get("hashes", {}) for arm in ARMS},
        "arm_protocols": protocols, "profile_selection_performed": False,
        "evidence_limit": "Official RULER metrics unchanged; not strict full-string/EOS capability, natural-LM validation, K causality, or SOTA"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-length", type=int, default=4096)
    parser.add_argument("--target-length", type=int, default=8192)
    args = parser.parse_args()
    try:
        result = summarize_panel(load_panel(args.root), args.reference_length, args.target_length)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": str(error), "profile_selection_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["status"] == "INVALID_OR_INCOMPLETE_PANEL":
        print(json.dumps(result))
        return 2
    print("length  native  physical_x  normalized_index  official_yarn")
    for length, scores in result["scores"].items():
        print(f"{length:>6}  " + "  ".join(f"{scores[arm]['macro']:.4f}" for arm in ARMS))
    print("stage2 entrance:", result["stage2_entrance"]["status"],
          "| physical reference retention:", result["reference_retention"]["physical_x"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
