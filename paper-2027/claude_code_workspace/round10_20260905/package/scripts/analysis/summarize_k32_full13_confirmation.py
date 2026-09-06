#!/usr/bin/env python3
"""Hash-bound CPU summary of the fresh K32 normalized-index full-13 holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analysis import summarize_qwen_s2_baseline_completion as common  # noqa: E402

ARMS = ("native", "normalized_index", "official_yarn")
TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
         "vt", "cwe", "fwe", "qa_1", "qa_2")
METRICS = {task: "string_match_part" if task in ("qa_1", "qa_2") else "string_match_all"
           for task in TASKS}
LENGTHS = (32768, 65536)
ROWS_PER_CELL = 20
DATA_SEED = 202609027
BOOTSTRAP_SEED = 202609028
BOOTSTRAP_SAMPLES = 10000
TERMINAL_STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TENSOR_SHA256 = {
    "native": "6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3",
    "normalized_index": "8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f",
    "official_yarn": "d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59",
}
GAIN = {"native": 1.0, "normalized_index": 1.0512928913614359,
        "official_yarn": 1.0693147180559945}


def load_panel(root: Path) -> dict:
    panel = {}
    for arm in ARMS:
        directory = root / arm
        paths = {"results": directory / "results.json", "examples": directory / "examples.jsonl",
                 "run_manifest": directory / "run_manifest.json"}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"{arm}: missing complete terminal artifacts")
        result = json.loads(paths["results"].read_text())
        if result.get("status") != TERMINAL_STATUS:
            raise ValueError(f"{arm}: result is not terminal")
        hashes = {name + "_sha256": common.file_hash(path) for name, path in paths.items()}
        if common.require_hash(result.get("results", {}).get("examples_sha256"), "examples") != hashes["examples_sha256"]:
            raise ValueError(f"{arm}: terminal/raw examples SHA mismatch")
        panel[arm] = {"result": result, "manifest": json.loads(paths["run_manifest"].read_text()),
                      "rows": [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()],
                      "hashes": hashes}
    return panel


def validate_panel(panel: dict) -> tuple[dict, dict]:
    if set(panel) != set(ARMS):
        raise ValueError("full-13 confirmation requires exactly three registered arms")
    expected_keys = {(task, length, i) for task in TASKS for length in LENGTHS
                     for i in range(ROWS_PER_CELL)}
    keyed, identities = {}, {}
    for arm in ARMS:
        item, result, protocol = panel[arm], panel[arm]["result"], panel[arm]["manifest"]
        if result.get("status") != TERMINAL_STATUS or result.get("protocol") != protocol:
            raise ValueError(f"{arm}: terminal protocol/run manifest mismatch")
        if (protocol.get("tasks") != list(TASKS) or protocol.get("lengths") != list(LENGTHS)
                or protocol.get("limit_per_cell") != ROWS_PER_CELL
                or protocol.get("checkpoint_sha256") != WEIGHT_SHA256
                or protocol.get("model_type") != "qwen2"
                or protocol.get("native_context_length") != LENGTHS[0]
                or protocol.get("profile_target_length") != LENGTHS[1]
                or protocol.get("table_factor") != 2.0
                or protocol.get("expected_native_sha256") != TENSOR_SHA256["native"]):
            raise ValueError(f"{arm}: fixed checkpoint/full13/two-length protocol mismatch")
        method, data, aggregate = result["method"], result["data"], result["results"]
        if (data.get("seed") != DATA_SEED or data.get("ruler_commit") != RULER_COMMIT
                or method.get("model_type") != "qwen2"
                or method.get("native_context_length") != LENGTHS[0]
                or method.get("active_sha256_float32") != TENSOR_SHA256[arm]
                or method.get("method") != protocol.get("method")):
            raise ValueError(f"{arm}: seed/RULER/method/active tensor mismatch")
        if arm == "native":
            if method["method"] != "native" or method.get("attention_scaling") != 1.0:
                raise ValueError("Native arm is not unmodified Native")
            table_file = None
        else:
            gain = method.get("long_attention_scaling")
            if (method["method"] != "external_table_static" or method.get("table_factor") != 2.0
                    or method.get("native_sha256_float32") != TENSOR_SHA256["native"]
                    or protocol.get("expected_active_sha256") != TENSOR_SHA256[arm]
                    or protocol.get("table_sha256_float32") != TENSOR_SHA256[arm]
                    or protocol.get("table_support") != "native_div_factor"
                    or not math.isclose(gain, GAIN[arm], rel_tol=0, abs_tol=1e-12)
                    or protocol.get("long_attention_scaling") != gain):
                raise ValueError(f"{arm}: fixed tensor/factor/support/gain mismatch")
            table_file = common.require_hash(protocol.get("table_file_sha256"), "table file")
        data_hash = common.require_hash(protocol.get("data_manifest_sha256"), "data manifest")
        tokenizer_hash = common.require_hash(data.get("tokenizer_sha256"), "tokenizer")
        if (data.get("manifest_sha256") != data_hash
                or protocol.get("expected_data_manifest_sha256") != data_hash
                or set(data.get("cells", {})) != set(TASKS)
                or set(aggregate.get("cells", {})) != set(TASKS)):
            raise ValueError(f"{arm}: paired data/task identity mismatch")
        cell_hashes = {}
        for task in TASKS:
            if (set(data["cells"][task]) != {str(length) for length in LENGTHS}
                    or set(aggregate["cells"][task]) != {str(length) for length in LENGTHS}):
                raise ValueError("every task must contain exactly both lengths")
            cell_hashes[task] = {}
            for length in LENGTHS:
                cell = data["cells"][task][str(length)]
                if cell.get("rows") != 20 or cell.get("selected_rows") != 20:
                    raise ValueError("every input cell must contain twenty selected rows")
                cell_hashes[task][str(length)] = common.require_hash(cell.get("sha256"), "input cell")
        rows = {}
        for row in item["rows"]:
            key = row.get("task"), row.get("nominal_length"), row.get("local_index")
            if type(key[1]) is not int or type(key[2]) is not int or key not in expected_keys or key in rows:
                raise ValueError("missing, duplicate, or unregistered raw row identity")
            common.finite_score(row.get("official_task_score"), "official task")
            references, tokens = row.get("references"), row.get("generated_token_ids")
            if (row.get("official_metric") != METRICS[key[0]] or not isinstance(row.get("prediction"), str)
                    or not isinstance(references, list) or not references or not all(isinstance(x, str) for x in references)
                    or not isinstance(tokens, list) or not tokens or not all(type(x) is int and x >= 0 for x in tokens)
                    or type(row.get("ended_with_eos")) is not bool
                    or type(row.get("generation_budget")) is not int or row["generation_budget"] <= 0
                    or len(tokens) > row["generation_budget"]
                    or type(row.get("prompt_tokens")) is not int or not 0 < row["prompt_tokens"] <= key[1]):
                raise ValueError("full prediction/token/EOS/task-specific official metric contract invalid")
            rows[key] = row
        if set(rows) != expected_keys or aggregate.get("examples") != len(expected_keys):
            raise ValueError("all 520 terminal rows per arm are required")
        hashes = {name: common.require_hash(item["hashes"].get(name), name)
                  for name in ("results_sha256", "examples_sha256", "run_manifest_sha256")}
        if aggregate.get("examples_sha256") != hashes["examples_sha256"]:
            raise ValueError("terminal/raw examples binding mismatch")
        means = []
        for task in TASKS:
            for length in LENGTHS:
                mean = sum(rows[task, length, i]["official_task_score"] for i in range(20)) / 20
                cell = aggregate["cells"][task][str(length)]
                if cell.get("rows") != 20 or not math.isclose(
                        common.finite_score(cell.get("official_task_score"), "aggregate"), mean,
                        rel_tol=0, abs_tol=1e-12):
                    raise ValueError("terminal cell aggregate differs from raw scores")
                means.append(mean)
        if not math.isclose(common.finite_score(aggregate.get("macro_official_task_score"), "macro"),
                            sum(means) / len(means), rel_tol=0, abs_tol=1e-12):
            raise ValueError("terminal macro differs from raw scores")
        keyed[arm] = rows
        identities[arm] = {"raw_hashes": hashes,
            "runner_sha256": common.require_hash(protocol.get("script_sha256"), "runner"),
            "checkpoint_weight_sha256": WEIGHT_SHA256,
            "native_sha256_float32": TENSOR_SHA256["native"],
            "active_sha256_float32": TENSOR_SHA256[arm], "attention_scaling": GAIN[arm],
            "table_file_sha256": table_file, "data_manifest_sha256": data_hash,
            "tokenizer_sha256": tokenizer_hash, "input_cell_sha256": cell_hashes,
            "data_seed": DATA_SEED, "ruler_commit": RULER_COMMIT}
    anchor = identities["native"]
    for arm in ARMS:
        for field in ("runner_sha256", "checkpoint_weight_sha256", "native_sha256_float32",
                      "data_manifest_sha256", "tokenizer_sha256", "input_cell_sha256"):
            if identities[arm][field] != anchor[field]:
                raise ValueError(f"{arm}: paired {field} mismatch")
        for key, row in keyed[arm].items():
            native = keyed["native"][key]
            for field in ("references", "official_metric", "prompt_tokens", "generation_budget"):
                if row[field] != native[field]:
                    raise ValueError(f"{arm}: paired row {field} mismatch")
    return keyed, identities


def paired_bootstrap(values: np.ndarray, rng) -> np.ndarray:
    if values.shape != (len(TASKS), ROWS_PER_CELL, len(ARMS)):
        raise ValueError("expected 13 tasks by 20 paired rows by three arms")
    draws = np.zeros((BOOTSTRAP_SAMPLES, len(ARMS)))
    for cell in values:
        indices = rng.integers(0, ROWS_PER_CELL, size=(BOOTSTRAP_SAMPLES, ROWS_PER_CELL))
        draws += cell[indices].mean(axis=1) / len(TASKS)
    return draws


def decision(retention_pass: bool, index_native_ci, index_yarn_ci) -> str:
    extension_pass = index_native_ci[0] > 0
    if not retention_pass or not extension_pass or index_yarn_ci[1] < 0:
        return "BASELINE_LOSS"
    if index_yarn_ci[0] > 0:
        return "CLEAR_ADVANCE"
    return "COMPETITIVE_UNRESOLVED"


def summarize_panel(panel: dict) -> dict:
    keyed, identities = validate_panel(panel)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores, comparisons = {}, {}
    for length in LENGTHS:
        values = np.array([[[keyed[arm][task, length, i]["official_task_score"] for arm in ARMS]
                            for i in range(20)] for task in TASKS])
        draws = paired_bootstrap(values, rng)
        scores[str(length)] = {}
        for column, arm in enumerate(ARMS):
            vector = values[:, :, column].mean(axis=1)
            scores[str(length)][arm] = {"task_vector": vector.tolist(),
                "per_task": dict(zip(TASKS, vector.tolist())), "macro": float(vector.mean())}
        comparisons[str(length)] = {}
        for comparator, column in (("native", 0), ("official_yarn", 2)):
            contrast = values[:, :, 1] - values[:, :, column]
            comparisons[str(length)][f"normalized_index_minus_{comparator}"] = {
                "delta": float(contrast.mean()), "per_task_delta": contrast.mean(axis=1).tolist(),
                "paired_stratified_ci95": np.quantile(draws[:, 1] - draws[:, column], [.025, .975]).tolist()}
    native_macro = scores["32768"]["native"]["macro"]
    index_macro = scores["32768"]["normalized_index"]["macro"]
    ratio = index_macro / native_macro if native_macro > 0 else None
    retention = {"ratio": ratio, "threshold": .875,
        "status": "UNDEFINED" if ratio is None else "PASS" if (
            ratio >= .875 or math.isclose(ratio, .875, rel_tol=0, abs_tol=1e-12)) else "FAIL"}
    index_native_ci = comparisons["65536"]["normalized_index_minus_native"]["paired_stratified_ci95"]
    index_yarn_ci = comparisons["65536"]["normalized_index_minus_official_yarn"]["paired_stratified_ci95"]
    outcome = decision(retention["status"] == "PASS", index_native_ci, index_yarn_ci)
    return {"status": "K32_FULL13_CONFIRMATION_COMPLETE", "decision": outcome,
        "decision_rule": "CLEAR_ADVANCE requires 32K retention pass and both 64K index-minus-Native/index-minus-YaRN CI95 wholly positive; COMPETITIVE_UNRESOLVED requires retention and resolver pass with index-minus-YaRN CI containing zero; otherwise BASELINE_LOSS.",
        "arms": list(ARMS), "tasks": list(TASKS), "official_metrics": METRICS,
        "lengths": list(LENGTHS), "rows_per_task_length": 20, "rows_per_arm": 520,
        "data_seed": DATA_SEED, "scores": scores, "paired_comparisons": comparisons,
        "native_retention": {"length": 32768, "arm": "normalized_index", **retention},
        "extension_resolver": {"length": 65536, "status": "PASS" if index_native_ci[0] > 0 else "FAIL",
            "contrast": "normalized_index_minus_native", "ci95": index_native_ci},
        "baseline_contrast": {"length": 65536, "contrast": "normalized_index_minus_official_yarn",
            "ci95": index_yarn_ci},
        "bootstrap": {"replicates": 10000, "seed": BOOTSTRAP_SEED,
            "unit": "paired row resampling within each of 13 fixed tasks; equal task-macro weights",
            "confidence": .95, "scope": "conditional on one checkpoint/new seed/task suite; not training-seed or checkpoint uncertainty"},
        "arm_identities": identities, "profile_selection_performed": False,
        "evidence_scope": "Fresh full-13 confirmation of an outcome-selected engineering representative; no K causality, natural-LM, broad SOTA, or new profile claim",
        "summary_script_sha256": common.file_hash(Path(__file__))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize_panel(load_panel(args.root))
    except ValueError as error:
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": str(error),
                  "profile_selection_performed": False}
    except (KeyError, TypeError, OSError):
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL",
                  "reason": "missing or malformed required artifact/field",
                  "profile_selection_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "decision") if key in result}))
    return 2 if result["status"] == "INVALID_OR_INCOMPLETE_PANEL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
