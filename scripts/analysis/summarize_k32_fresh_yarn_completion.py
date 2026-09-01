#!/usr/bin/env python3
"""Complete the fresh K32 N80 panel with one fixed YaRN2 arm, CPU only."""

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
from scripts.analysis import summarize_k32_crossing_confirmation as crossing  # noqa: E402

ARMS = (*crossing.ARMS, "official_yarn")
YARN_TENSOR_SHA256 = "d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59"
YARN_GAIN = 1 + .1 * math.log(2)


def load_panel(root: Path) -> dict:
    panel = crossing.load_panel(root)
    arm, directory = "official_yarn", root / "official_yarn"
    paths = {"results": directory / "results.json", "examples": directory / "examples.jsonl",
             "run_manifest": directory / "run_manifest.json"}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("official_yarn: missing complete results/examples/run manifest")
    result = json.loads(paths["results"].read_text())
    if result.get("status") != crossing.TERMINAL_STATUS:
        raise ValueError("official_yarn: result is not terminal")
    hashes = {name + "_sha256": crossing.common.file_hash(path) for name, path in paths.items()}
    if crossing.common.require_hash(result.get("results", {}).get("examples_sha256"), "examples") != hashes["examples_sha256"]:
        raise ValueError("official_yarn: examples SHA mismatch")
    panel[arm] = {"result": result, "manifest": json.loads(paths["run_manifest"].read_text()),
                  "rows": [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()],
                  "hashes": hashes}
    return panel


def validate_yarn(item: dict, keyed: dict, identities: dict) -> tuple[dict, dict]:
    result, protocol = item["result"], item["manifest"]
    if result.get("status") != crossing.TERMINAL_STATUS or result.get("protocol") != protocol:
        raise ValueError("official_yarn: terminal protocol/run manifest mismatch")
    if (protocol.get("tasks") != list(crossing.TASKS)
            or protocol.get("lengths") != list(crossing.LENGTHS)
            or protocol.get("limit_per_cell") != crossing.ROWS_PER_CELL
            or protocol.get("checkpoint_sha256") != crossing.WEIGHT_SHA256
            or protocol.get("native_context_length") != crossing.LENGTHS[0]
            or protocol.get("profile_target_length") != crossing.LENGTHS[1]
            or protocol.get("model_type") != "qwen2" or protocol.get("table_factor") != 2.0
            or protocol.get("method") != "external_table_static"
            or protocol.get("expected_native_sha256") != crossing.TENSOR_SHA256["native"]
            or protocol.get("expected_active_sha256") != YARN_TENSOR_SHA256
            or protocol.get("table_sha256_float32") != YARN_TENSOR_SHA256
            or protocol.get("table_support") != "native_div_factor"
            or not math.isclose(protocol.get("long_attention_scaling", math.nan), YARN_GAIN,
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("official_yarn: fixed checkpoint/s2/tensor/gain protocol mismatch")
    method, data, aggregate = result["method"], result["data"], result["results"]
    if (method.get("method") != "external_table_static" or method.get("model_type") != "qwen2"
            or method.get("native_context_length") != crossing.LENGTHS[0]
            or method.get("active_sha256_float32") != YARN_TENSOR_SHA256
            or method.get("native_sha256_float32") != crossing.TENSOR_SHA256["native"]
            or method.get("table_factor") != 2.0
            or not math.isclose(method.get("long_attention_scaling", math.nan), YARN_GAIN,
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("official_yarn: realized method/tensor/gain mismatch")
    if data.get("seed") != crossing.DATA_SEED or data.get("ruler_commit") != crossing.RULER_COMMIT:
        raise ValueError("official_yarn: fresh seed or RULER revision mismatch; old pilot is inadmissible")
    data_hash = crossing.common.require_hash(protocol.get("data_manifest_sha256"), "data manifest")
    tokenizer_hash = crossing.common.require_hash(data.get("tokenizer_sha256"), "tokenizer")
    if (data.get("manifest_sha256") != data_hash
            or protocol.get("expected_data_manifest_sha256") != data_hash
            or set(data.get("cells", {})) != set(crossing.TASKS)
            or set(aggregate.get("cells", {})) != set(crossing.TASKS)):
        raise ValueError("official_yarn: bound data identity or task family mismatch")
    cell_hashes = {}
    for task in crossing.TASKS:
        if (set(data["cells"][task]) != {str(length) for length in crossing.LENGTHS}
                or set(aggregate["cells"][task]) != {str(length) for length in crossing.LENGTHS}):
            raise ValueError("official_yarn: each task requires both lengths")
        cell_hashes[task] = {}
        for length in crossing.LENGTHS:
            cell = data["cells"][task][str(length)]
            if cell.get("rows") != 80 or cell.get("selected_rows") != 80:
                raise ValueError("official_yarn: each input cell requires eighty rows")
            cell_hashes[task][str(length)] = crossing.common.require_hash(cell.get("sha256"), "input cell")
    expected = {(task, length, index) for task in crossing.TASKS
                for length in crossing.LENGTHS for index in range(crossing.ROWS_PER_CELL)}
    rows = {}
    for row in item["rows"]:
        key = row.get("task"), row.get("nominal_length"), row.get("local_index")
        if type(key[1]) is not int or type(key[2]) is not int or key not in expected or key in rows:
            raise ValueError("official_yarn: missing, duplicate, or unregistered row identity")
        crossing.common.finite_score(row.get("official_task_score"), "official task")
        references, tokens = row.get("references"), row.get("generated_token_ids")
        if (row.get("official_metric") != "string_match_all" or not isinstance(row.get("prediction"), str)
                or not isinstance(references, list) or not references or not all(isinstance(x, str) for x in references)
                or not isinstance(tokens, list) or not tokens or not all(type(x) is int and x >= 0 for x in tokens)
                or type(row.get("ended_with_eos")) is not bool
                or type(row.get("generation_budget")) is not int or row["generation_budget"] <= 0
                or len(tokens) > row["generation_budget"]
                or type(row.get("prompt_tokens")) is not int or not 0 < row["prompt_tokens"] <= key[1]):
            raise ValueError("official_yarn: full prediction/token/EOS/scorer contract invalid")
        rows[key] = row
    if set(rows) != expected or aggregate.get("examples") != 640:
        raise ValueError("official_yarn: all 640 terminal rows are required")
    hashes = {name: crossing.common.require_hash(item["hashes"].get(name), name)
              for name in ("results_sha256", "examples_sha256", "run_manifest_sha256")}
    if aggregate.get("examples_sha256") != hashes["examples_sha256"]:
        raise ValueError("official_yarn: terminal/raw examples SHA mismatch")
    means = []
    for task in crossing.TASKS:
        for length in crossing.LENGTHS:
            mean = sum(rows[task, length, i]["official_task_score"] for i in range(80)) / 80
            cell = aggregate["cells"][task][str(length)]
            if cell.get("rows") != 80 or not math.isclose(
                    crossing.common.finite_score(cell.get("official_task_score"), "cell aggregate"),
                    mean, rel_tol=0, abs_tol=1e-12):
                raise ValueError("official_yarn: terminal aggregate differs from raw rows")
            means.append(mean)
    if not math.isclose(crossing.common.finite_score(aggregate.get("macro_official_task_score"), "macro"),
                        sum(means) / len(means), rel_tol=0, abs_tol=1e-12):
        raise ValueError("official_yarn: terminal macro differs from raw rows")
    anchor = identities["native"]
    runner_hash = crossing.common.require_hash(protocol.get("script_sha256"), "runner")
    table_file_hash = crossing.common.require_hash(protocol.get("table_file_sha256"), "table file")
    identity = {"raw_hashes": hashes, "runner_sha256": runner_hash,
                "checkpoint_weight_sha256": crossing.WEIGHT_SHA256,
                "native_sha256_float32": crossing.TENSOR_SHA256["native"],
                "active_sha256_float32": YARN_TENSOR_SHA256, "table_file_sha256": table_file_hash,
                "table_factor": 2.0, "attention_scaling": YARN_GAIN,
                "data_manifest_sha256": data_hash, "tokenizer_sha256": tokenizer_hash,
                "input_cell_sha256": cell_hashes, "seed": crossing.DATA_SEED,
                "ruler_commit": crossing.RULER_COMMIT}
    for field in ("runner_sha256", "checkpoint_weight_sha256", "native_sha256_float32",
                  "data_manifest_sha256", "tokenizer_sha256", "input_cell_sha256"):
        if identity[field] != anchor[field]:
            raise ValueError(f"official_yarn: paired {field} mismatch")
    for key, row in rows.items():
        native = keyed["native"][key]
        for field in ("references", "official_metric", "prompt_tokens", "generation_budget"):
            if row[field] != native[field]:
                raise ValueError(f"official_yarn: paired row {field} mismatch")
    return rows, identity


def paired_bootstrap(values: np.ndarray, rng) -> np.ndarray:
    """Reuse the registered task-stratified paired-row algorithm for four arms."""
    if values.shape != (len(crossing.TASKS), 80, len(ARMS)):
        raise ValueError("expected four tasks by eighty paired rows by four arms")
    draws = np.zeros((crossing.BOOTSTRAP_SAMPLES, len(ARMS)))
    for cell in values:
        indices = rng.integers(0, 80, size=(crossing.BOOTSTRAP_SAMPLES, 80))
        draws += cell[indices].mean(axis=1) / len(crossing.TASKS)
    return draws


def summarize_panel(panel: dict) -> dict:
    if set(panel) != set(ARMS):
        raise ValueError("fresh completion requires exactly four named arms")
    original_panel = {arm: panel[arm] for arm in crossing.ARMS}
    original = crossing.summarize_panel(original_panel)
    keyed, identities = crossing.validate_panel(original_panel)
    yarn_rows, yarn_identity = validate_yarn(panel["official_yarn"], keyed, identities)
    keyed["official_yarn"], identities["official_yarn"] = yarn_rows, yarn_identity
    rng = np.random.default_rng(crossing.BOOTSTRAP_SEED)
    scores, comparison = {}, {}
    for length in crossing.LENGTHS:
        values = np.array([[[keyed[arm][task, length, i]["official_task_score"] for arm in ARMS]
                            for i in range(80)] for task in crossing.TASKS])
        draws = paired_bootstrap(values, rng)
        scores[str(length)] = {}
        for column, arm in enumerate(ARMS):
            vector = values[:, :, column].mean(axis=1)
            scores[str(length)][arm] = {"task_vector": vector.tolist(),
                "per_task": dict(zip(crossing.TASKS, vector.tolist())), "macro": float(vector.mean())}
        contrast = values[:, :, 2] - values[:, :, 3]
        comparison[str(length)] = {"contrast": "normalized_index_minus_official_yarn",
            "delta": float(contrast.mean()), "per_task_delta": contrast.mean(axis=1).tolist(),
            "paired_stratified_ci95": np.quantile(draws[:, 2] - draws[:, 3], [.025, .975]).tolist()}
    native_macro = scores["32768"]["native"]["macro"]
    retention = {}
    for arm in ARMS:
        ratio = scores["32768"][arm]["macro"] / native_macro if native_macro > 0 else None
        retention[arm] = {"ratio": ratio, "threshold": .875,
            "status": "UNDEFINED" if ratio is None else "PASS" if (
                ratio >= .875 or math.isclose(ratio, .875, rel_tol=0, abs_tol=1e-12)) else "FAIL"}
    long_ci = comparison["65536"]["paired_stratified_ci95"]
    verdict = "INDEX_FAVORED" if long_ci[0] > 0 else "YARN_FAVORED" if long_ci[1] < 0 else "UNRESOLVED"
    return {"status": "K32_FRESH_YARN_COMPLETION_COMPLETE",
        "three_arm_crossing_decision": original["decision"],
        "three_arm_crossing_decision_rule": original["decision_rule"],
        "three_arm_primary_receipt": original["primary_physical_minus_index"],
        "crossing_decision_changed_by_yarn": False, "arms": list(ARMS),
        "tasks": list(crossing.TASKS), "lengths": list(crossing.LENGTHS),
        "rows_per_task_length": 80, "scores": scores,
        "index_minus_yarn": comparison,
        "long_index_minus_yarn_verdict": {"length": 65536, "status": verdict,
            "rule": "95% paired interval wholly above zero favors index, wholly below favors YaRN, otherwise unresolved; no selector"},
        "native_retention": {"length": 32768, "arms": retention,
            "role": "separate point-estimate gate; does not modify the original crossing decision"},
        "bootstrap": {"replicates": 10000, "seed": crossing.BOOTSTRAP_SEED,
            "confidence": .95, "unit": "same row indices resampled within each fixed task; equal task-macro weights",
            "scope": "conditional on one checkpoint and the fresh registered rows; not checkpoint/training-seed uncertainty"},
        "arm_identities": identities, "old_pilot_read_or_pooled": False,
        "profile_selection_performed": False,
        "evidence_scope": "Matched fresh N80 baseline completion after three-arm outcomes were known; core4 official scores, not full RULER, K causality, strict EOS capability, natural LM, or SOTA",
        "summary_script_sha256": crossing.common.file_hash(Path(__file__)),
        "crossing_summary_script_sha256": crossing.common.file_hash(Path(crossing.__file__))}


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
    print(json.dumps({key: result[key] for key in ("status", "three_arm_crossing_decision") if key in result}))
    return 2 if result["status"] == "INVALID_OR_INCOMPLETE_PANEL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
