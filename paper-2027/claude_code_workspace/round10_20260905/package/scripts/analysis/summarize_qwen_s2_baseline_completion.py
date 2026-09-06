#!/usr/bin/env python3
"""Summarize hash-bound Qwen s2 core4 completion; no profile selection or GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")
TASK_LABELS = ("single1", "mk2", "mk3", "vt")
LENGTHS = (32768, 65536)
ROWS_PER_CELL = 20
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 202609025
TERMINAL_STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
ARMS = {"k32": ("native", "physical_x", "normalized_index", "official_yarn"),
        "k64": ("native", "c2", "official_yarn")}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"missing or invalid {label} SHA-256")
    return value


def directories(model: str) -> dict:
    if model == "k32":
        return {
            "native": {length: f"prior_k32_native_{length}" for length in LENGTHS},
            "physical_x": {length: f"prior_k32_physical/physical_x_s2_{length // 1024}k" for length in LENGTHS},
            "normalized_index": {length: f"prior_k32_index/index_s2_{length // 1024}k" for length in LENGTHS},
            "official_yarn": {length: f"k32/official_yarn_{length}" for length in LENGTHS},
        }
    if model == "k64":
        return {
            "native": {length: f"prior_k64_native_{length}" for length in LENGTHS},
            "c2": {length: f"k64_bound_data/c2_s2_{length}" for length in LENGTHS},
            "official_yarn": {length: f"k64_bound_data/official_yarn_{length}" for length in LENGTHS},
        }
    raise ValueError("model must be k32 or k64")


def load_panel(root: Path, model: str) -> dict:
    panel = {}
    for arm, cells in directories(model).items():
        panel[arm] = {}
        for length, relative in cells.items():
            directory = root / relative
            result_path = directory / "results.json"
            examples_path = directory / "examples.jsonl"
            manifest_path = directory / "run_manifest.json"
            if not all(path.is_file() for path in (result_path, examples_path, manifest_path)):
                raise ValueError(f"{arm}/{length}: missing terminal result, examples, or run manifest")
            result = json.loads(result_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            if result.get("status") != TERMINAL_STATUS:
                raise ValueError(f"{arm}/{length}: result is not terminal")
            examples_hash = file_hash(examples_path)
            expected = require_hash(result.get("results", {}).get("examples_sha256"), "examples")
            if examples_hash != expected:
                raise ValueError(f"{arm}/{length}: examples hash mismatch")
            rows = [json.loads(line) for line in examples_path.read_text().splitlines() if line.strip()]
            panel[arm][length] = {
                "result": result, "manifest": manifest, "rows": rows,
                "hashes": {"results_sha256": file_hash(result_path),
                           "examples_sha256": examples_hash,
                           "run_manifest_sha256": file_hash(manifest_path)},
            }
    return panel


def finite_score(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid {label} score")
    return float(value)


def validate_panel(panel: dict, model: str) -> tuple[dict, dict]:
    if set(panel) != set(ARMS[model]):
        raise ValueError("panel does not contain exactly its registered arms")
    keyed, identities = {}, {}
    for arm in ARMS[model]:
        if set(panel[arm]) != set(LENGTHS):
            raise ValueError("panel must contain both registered lengths")
        keyed[arm], identities[arm] = {}, {}
        for length in LENGTHS:
            item = panel[arm][length]
            result, protocol = item["result"], item["manifest"]
            if result.get("status") != TERMINAL_STATUS or result.get("protocol") != protocol:
                raise ValueError(f"{arm}/{length}: terminal protocol/run manifest mismatch")
            if (protocol.get("tasks") != list(TASKS) or protocol.get("lengths") != [length]
                    or type(protocol.get("limit_per_cell")) is not int
                    or protocol["limit_per_cell"] != ROWS_PER_CELL
                    or protocol.get("model_type") != "qwen2"
                    or protocol.get("native_context_length") != LENGTHS[0]):
                raise ValueError(f"{arm}/{length}: registered protocol mismatch")
            method, data, aggregate = result["method"], result["data"], result["results"]
            if (method.get("method") != protocol.get("method")
                    or method.get("model_type") != "qwen2"
                    or method.get("native_context_length") != LENGTHS[0]):
                raise ValueError(f"{arm}/{length}: method identity mismatch")
            active = require_hash(method.get("active_sha256_float32"), "active tensor")
            if arm == "native":
                if method["method"] != "native" or method.get("attention_scaling") != 1.0:
                    raise ValueError("Native comparator is not unmodified Native")
                native, gain = active, 1.0
            else:
                if (method["method"] != "external_table_static" or protocol.get("table_factor") != 2.0
                        or method.get("table_factor") != 2.0):
                    raise ValueError(f"{arm}/{length}: expected fixed external s2 profile")
                native = require_hash(method.get("native_sha256_float32"), "Native tensor")
                if require_hash(protocol.get("table_sha256_float32"), "table tensor") != active:
                    raise ValueError("active tensor differs from bound table")
                require_hash(protocol.get("table_file_sha256"), "table file")
                gain = method.get("long_attention_scaling")
                if (type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0
                        or protocol.get("long_attention_scaling") != gain):
                    raise ValueError("table amplitude is missing, invalid, or inconsistent")
                prescribed_gain = 1 + (.1 if arm == "official_yarn" else .074) * math.log(2)
                if not math.isclose(gain, prescribed_gain, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{arm}/{length}: amplitude differs from fixed s2 gain")
            if (protocol.get("expected_native_sha256") not in (None, native)
                    or protocol.get("expected_active_sha256") not in (None, active)
                    or protocol.get("profile_target_length") not in (None, LENGTHS[1])):
                raise ValueError("runtime tensor or declared maximum-profile identity mismatch")
            data_hash = require_hash(protocol.get("data_manifest_sha256"), "data manifest")
            if (data.get("manifest_sha256") != data_hash
                    or protocol.get("expected_data_manifest_sha256") not in (None, data_hash)):
                raise ValueError("bound data manifest identity mismatch")
            tokenizer_hash = require_hash(data.get("tokenizer_sha256"), "tokenizer")
            if set(data.get("cells", {})) != set(TASKS) or set(aggregate.get("cells", {})) != set(TASKS):
                raise ValueError("data/results do not contain exactly core4 task cells")
            cell_hashes = {}
            for task in TASKS:
                if set(data["cells"][task]) != {str(length)} or set(aggregate["cells"][task]) != {str(length)}:
                    raise ValueError("data/results contain unregistered length cells")
                cell = data["cells"][task][str(length)]
                if cell.get("rows") != ROWS_PER_CELL or cell.get("selected_rows") != ROWS_PER_CELL:
                    raise ValueError("data cell must bind exactly twenty input rows")
                cell_hashes[task] = require_hash(cell.get("sha256"), "input cell")
            expected_keys = {(task, length, i) for task in TASKS for i in range(ROWS_PER_CELL)}
            rows = {}
            for row in item["rows"]:
                key = row.get("task"), row.get("nominal_length"), row.get("local_index")
                if (type(key[1]) is not int or type(key[2]) is not int
                        or key not in expected_keys or key in rows):
                    raise ValueError("raw row identity is missing, duplicated, or unregistered")
                finite_score(row.get("official_task_score"), "official task")
                if (row.get("official_metric") != "string_match_all"
                        or not isinstance(row.get("references"), list) or not row["references"]
                        or not all(isinstance(ref, str) for ref in row["references"])
                        or not isinstance(row.get("prediction"), str)):
                    raise ValueError("raw reference/official metric/full prediction contract mismatch")
                rows[key] = row
            if set(rows) != expected_keys or aggregate.get("examples") != len(expected_keys):
                raise ValueError("raw/terminal examples are not a complete twenty-row-per-task panel")
            if aggregate.get("examples_sha256") != item["hashes"]["examples_sha256"]:
                raise ValueError("terminal examples binding differs from loaded raw hash")
            task_means = []
            for task in TASKS:
                mean = sum(rows[task, length, i]["official_task_score"] for i in range(ROWS_PER_CELL)) / ROWS_PER_CELL
                cell = aggregate["cells"][task][str(length)]
                if (cell.get("rows") != ROWS_PER_CELL or not math.isclose(
                        finite_score(cell.get("official_task_score"), "aggregate"), mean, rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("terminal task aggregate differs from raw scores")
                task_means.append(mean)
            if not math.isclose(finite_score(aggregate.get("macro_official_task_score"), "macro"),
                                sum(task_means) / len(TASKS), rel_tol=0, abs_tol=1e-12):
                raise ValueError("terminal macro differs from raw scores")
            keyed[arm][length] = rows
            # Deliberate whitelist: never copy data paths or an entire raw protocol.
            identities[arm][str(length)] = {
                "raw_hashes": item["hashes"],
                "runner_sha256": require_hash(protocol.get("script_sha256"), "runner"),
                "checkpoint_weight_sha256": require_hash(protocol.get("checkpoint_sha256"), "checkpoint weight"),
                "native_sha256_float32": native, "active_sha256_float32": active,
                "data_manifest_sha256": data_hash, "tokenizer_sha256": tokenizer_hash,
                "input_cell_sha256": cell_hashes, "attention_scaling": float(gain),
                "table_factor": None if arm == "native" else 2.0,
                "table_file_sha256": None if arm == "native" else protocol["table_file_sha256"],
                "native_context_length": LENGTHS[0],
            }
    anchor = identities["native"][str(LENGTHS[0])]
    for arm in ARMS[model]:
        for length in LENGTHS:
            identity = identities[arm][str(length)]
            for field in ("checkpoint_weight_sha256", "native_sha256_float32", "tokenizer_sha256"):
                if identity[field] != anchor[field]:
                    raise ValueError(f"{arm}/{length}: paired {field} mismatch")
            if identity["input_cell_sha256"] != identities["native"][str(length)]["input_cell_sha256"]:
                raise ValueError(f"{arm}/{length}: paired input cell hashes mismatch")
            for key, row in keyed[arm][length].items():
                baseline = keyed["native"][length][key]
                if row["references"] != baseline["references"] or row["official_metric"] != baseline["official_metric"]:
                    raise ValueError("paired row references or official metric mismatch")
        for field in ("active_sha256_float32", "attention_scaling", "table_file_sha256"):
            if identities[arm][str(LENGTHS[0])][field] != identities[arm][str(LENGTHS[1])][field]:
                raise ValueError(f"{arm}: profile changed between registered lengths")
    return keyed, identities


def paired_bootstrap(values: np.ndarray, rng) -> np.ndarray:
    """One paired row draw for all arms, stratified by the four fixed tasks."""
    if values.ndim != 3 or values.shape[:2] != (len(TASKS), ROWS_PER_CELL):
        raise ValueError("bootstrap requires four tasks by twenty paired rows by arms")
    draws = np.zeros((BOOTSTRAP_SAMPLES, values.shape[2]))
    for cell in values:
        indices = rng.integers(0, ROWS_PER_CELL, size=(BOOTSTRAP_SAMPLES, ROWS_PER_CELL))
        draws += cell[indices].mean(axis=1) / len(TASKS)
    return draws


def summarize_panel(panel: dict, model: str) -> dict:
    keyed, identities = validate_panel(panel, model)
    arms, candidate = ARMS[model], "physical_x" if model == "k32" else "c2"
    comparators = ["native", "official_yarn"] + (["normalized_index"] if model == "k32" else [])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores, contrasts = {}, {}
    for length in LENGTHS:
        values = np.array([[[keyed[arm][length][task, length, i]["official_task_score"] for arm in arms]
                            for i in range(ROWS_PER_CELL)] for task in TASKS], dtype=np.float64)
        draws = paired_bootstrap(values, rng)
        scores[str(length)] = {}
        for column, arm in enumerate(arms):
            vector = values[:, :, column].mean(axis=1)
            scores[str(length)][arm] = {"task_vector": vector.tolist(),
                "per_task": dict(zip(TASKS, vector.tolist())), "macro": float(vector.mean())}
        contrasts[str(length)] = {}
        ci_column = arms.index(candidate)
        for comparator in comparators:
            other = arms.index(comparator)
            paired = draws[:, ci_column] - draws[:, other]
            contrasts[str(length)][f"{candidate}_minus_{comparator}"] = {
                "delta": float((values[:, :, ci_column] - values[:, :, other]).mean()),
                "per_task_delta": (values[:, :, ci_column] - values[:, :, other]).mean(axis=1).tolist(),
                "paired_stratified_ci95": np.quantile(paired, [.025, .975]).tolist(),
                "bonferroni_two_lengths_sensitivity_ci95": np.quantile(paired, [.0125, .9875]).tolist(),
            }
    reference = scores[str(LENGTHS[0])]
    native_macro = reference["native"]["macro"]
    retention = {}
    for arm in arms:
        ratio = reference[arm]["macro"] / native_macro if native_macro > 0 else None
        retention[arm] = {"ratio": ratio, "threshold": .875,
                          "status": "UNDEFINED" if ratio is None else "PASS" if (
                              ratio >= .875 or math.isclose(ratio, .875, rel_tol=0, abs_tol=1e-12)) else "FAIL"}
    yarn_macro = scores[str(LENGTHS[1])]["official_yarn"]["macro"]
    resolved = native_macro > 0 and yarn_macro > 0
    return {
        "status": "QWEN_S2_BASELINE_PANEL_COMPLETE", "model": model, "arms": list(arms),
        "tasks": list(TASKS), "task_vector_order": list(TASK_LABELS), "lengths": list(LENGTHS),
        "rows_per_task_length": ROWS_PER_CELL, "scale": 2.0,
        "scores": scores, "paired_comparisons": contrasts,
        "native_reference_retention": {"length": LENGTHS[0], "denominator": "Native macro at 1x", "arms": retention},
        "resolver": {"native_1x_macro": native_macro, "yarn_2x_macro": yarn_macro,
                     "native_1x_nonzero": native_macro > 0, "yarn_2x_nonzero": yarn_macro > 0,
                     "status": "PASS" if resolved else "STOP_INTERPRETATION",
                     "rule": "both registered control macros must be strictly positive; no ref/scale adjustment on failure"},
        "bootstrap": {"replicates": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                      "confidence": .95, "numpy_version": np.__version__,
                      "quantile_interpolation": "linear", "unit": "paired rows within each fixed task; equal task-macro weights",
                      "assumption": "rows exchangeable within fixed tasks; all arms share resampled row indices; no normality assumption",
                      "scope": "conditional on checkpoint, input cells and recorded decoding; not training-seed/checkpoint uncertainty",
                      "multiplicity": "two-length Bonferroni sensitivity within each fixed contrast: 97.5% marginal percentile intervals; not all-contrast familywise control or new-sample confirmation",
                      "missing_data_policy": "reject incomplete panels; no exclusions or outlier filtering"},
        "arm_identities": identities, "profile_selection_performed": False,
        "execution_identity": "hash-bound historical matched-data comparison; K32 historical runner not fully recovered; no byte-identical execution claim",
        "coordinate_scope": "K32 physical-index is nondegenerate; K64 has one C2 arm, not an independent index arm",
        "evidence_limits": "Official RULER scores, not strict full-string/EOS capability. No K causality, SOTA, new-sample confirmation, or natural-LM double gate. Point contrasts and intervals do not establish small-effect rankings.",
        "summary_script_sha256": file_hash(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize_panel(load_panel(args.root, args.model), args.model)
    except ValueError as error:
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": str(error), "profile_selection_performed": False}
    except (KeyError, TypeError, OSError):
        # Never serialize OS exceptions containing private machine paths.
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": "missing or malformed required input artifact/field", "profile_selection_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "model": args.model}))
    return 0 if result["status"] == "QWEN_S2_BASELINE_PANEL_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
