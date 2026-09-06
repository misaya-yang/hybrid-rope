#!/usr/bin/env python3
"""Fixed, independently seeded K32 physical-index crossing confirmation."""

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

ARMS = ("native", "physical_x", "normalized_index")
TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")
LENGTHS = (32768, 65536)
ROWS_PER_CELL = 80
DATA_SEED = 202609026
BOOTSTRAP_SEED = 202609027
BOOTSTRAP_SAMPLES = 10000
TERMINAL_STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TENSOR_SHA256 = {
    "native": "6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3",
    "physical_x": "b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb",
    "normalized_index": "8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f",
}
GAIN = 1 + .074 * math.log(2)


def load_panel(root: Path) -> dict:
    panel = {}
    for arm in ARMS:
        directory = root / arm
        paths = {"results": directory / "results.json", "examples": directory / "examples.jsonl",
                 "run_manifest": directory / "run_manifest.json"}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"{arm}: missing complete results/examples/run manifest")
        result = json.loads(paths["results"].read_text())
        if result.get("status") != TERMINAL_STATUS:
            raise ValueError(f"{arm}: result is not terminal")
        hashes = {name + "_sha256": common.file_hash(path) for name, path in paths.items()}
        if common.require_hash(result.get("results", {}).get("examples_sha256"), "examples") != hashes["examples_sha256"]:
            raise ValueError(f"{arm}: examples SHA mismatch")
        panel[arm] = {
            "result": result, "manifest": json.loads(paths["run_manifest"].read_text()),
            "rows": [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()],
            "hashes": hashes,
        }
    return panel


def validate_panel(panel: dict) -> tuple[dict, dict]:
    if set(panel) != set(ARMS):
        raise ValueError("confirmation requires exactly Native, physical_x, normalized_index")
    keyed, identities = {}, {}
    expected_keys = {(task, length, i) for task in TASKS for length in LENGTHS for i in range(ROWS_PER_CELL)}
    for arm in ARMS:
        item, result = panel[arm], panel[arm]["result"]
        protocol = item["manifest"]
        if result.get("status") != TERMINAL_STATUS or result.get("protocol") != protocol:
            raise ValueError(f"{arm}: terminal protocol/run manifest mismatch")
        if (protocol.get("tasks") != list(TASKS) or protocol.get("lengths") != list(LENGTHS)
                or type(protocol.get("limit_per_cell")) is not int or protocol["limit_per_cell"] != ROWS_PER_CELL
                or protocol.get("checkpoint_sha256") != WEIGHT_SHA256
                or protocol.get("native_context_length") != LENGTHS[0]
                or protocol.get("profile_target_length") != LENGTHS[1]
                or protocol.get("model_type") != "qwen2" or protocol.get("table_factor") != 2.0):
            raise ValueError(f"{arm}: fixed checkpoint/two-length/80-row/s2 protocol mismatch")
        data, method, aggregate = result["data"], result["method"], result["results"]
        if type(data.get("seed")) is not int or data["seed"] != DATA_SEED or data.get("ruler_commit") != RULER_COMMIT:
            raise ValueError("fresh seed or official RULER revision mismatch; old pilot is not admissible")
        data_hash = common.require_hash(protocol.get("data_manifest_sha256"), "data manifest")
        if (data.get("manifest_sha256") != data_hash
                or protocol.get("expected_data_manifest_sha256") != data_hash):
            raise ValueError("data manifest is not consistently frozen")
        tokenizer_hash = common.require_hash(data.get("tokenizer_sha256"), "tokenizer")
        if (method.get("model_type") != "qwen2" or method.get("native_context_length") != LENGTHS[0]
                or method.get("method") != protocol.get("method")
                or method.get("active_sha256_float32") != TENSOR_SHA256[arm]
                or protocol.get("expected_native_sha256") != TENSOR_SHA256["native"]):
            raise ValueError("fixed Native/active tensor or method identity mismatch")
        if arm == "native":
            if method["method"] != "native" or method.get("attention_scaling") != 1.0:
                raise ValueError("Native arm is not unmodified Native")
            gain, table_file_hash = 1.0, None
        else:
            gain = method.get("long_attention_scaling")
            if (method["method"] != "external_table_static" or method.get("table_factor") != 2.0
                    or method.get("native_sha256_float32") != TENSOR_SHA256["native"]
                    or protocol.get("table_sha256_float32") != TENSOR_SHA256[arm]
                    or protocol.get("expected_active_sha256") != TENSOR_SHA256[arm]
                    or protocol.get("table_support") != "native_div_factor"
                    or type(gain) not in (int, float) or not math.isfinite(gain)
                    or not math.isclose(gain, GAIN, rel_tol=0, abs_tol=1e-12)
                    or protocol.get("long_attention_scaling") != gain):
                raise ValueError("frozen physical/index table, factor, support, or gain mismatch")
            table_file_hash = common.require_hash(protocol.get("table_file_sha256"), "table file")
        if set(data.get("cells", {})) != set(TASKS) or set(aggregate.get("cells", {})) != set(TASKS):
            raise ValueError("exactly four task families are required")
        cell_hashes = {}
        for task in TASKS:
            if (set(data["cells"][task]) != {str(length) for length in LENGTHS}
                    or set(aggregate["cells"][task]) != {str(length) for length in LENGTHS}):
                raise ValueError("each task must contain both registered lengths")
            cell_hashes[task] = {}
            for length in LENGTHS:
                cell = data["cells"][task][str(length)]
                if cell.get("rows") != ROWS_PER_CELL or cell.get("selected_rows") != ROWS_PER_CELL:
                    raise ValueError("each input cell must contain exactly eighty selected rows")
                cell_hashes[task][str(length)] = common.require_hash(cell.get("sha256"), "input cell")
        rows = {}
        for row in item["rows"]:
            key = row.get("task"), row.get("nominal_length"), row.get("local_index")
            if (type(key[1]) is not int or type(key[2]) is not int or key not in expected_keys or key in rows):
                raise ValueError("missing, duplicate, or unregistered paired row identity")
            common.finite_score(row.get("official_task_score"), "official task")
            refs, tokens = row.get("references"), row.get("generated_token_ids")
            budget, prompt_tokens = row.get("generation_budget"), row.get("prompt_tokens")
            if (row.get("official_metric") != "string_match_all" or not isinstance(row.get("prediction"), str)
                    or not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs)
                    or not isinstance(tokens, list) or not tokens or not all(type(token) is int and token >= 0 for token in tokens)
                    or type(row.get("ended_with_eos")) is not bool
                    or type(budget) is not int or budget <= 0 or len(tokens) > budget
                    or type(prompt_tokens) is not int or not 0 < prompt_tokens <= key[1]):
                raise ValueError("full prediction/token/EOS or official scorer contract missing or invalid")
            rows[key] = row
        if set(rows) != expected_keys or aggregate.get("examples") != 640:
            raise ValueError("all 640 confirmation rows per arm must be terminal; no pilot pooling")
        hashes = {name: common.require_hash(item["hashes"].get(name), name)
                  for name in ("results_sha256", "examples_sha256", "run_manifest_sha256")}
        if aggregate.get("examples_sha256") != hashes["examples_sha256"]:
            raise ValueError("terminal examples SHA differs from loaded raw artifact")
        means = []
        for task in TASKS:
            for length in LENGTHS:
                mean = sum(rows[task, length, i]["official_task_score"] for i in range(ROWS_PER_CELL)) / ROWS_PER_CELL
                cell = aggregate["cells"][task][str(length)]
                if (cell.get("rows") != ROWS_PER_CELL or not math.isclose(
                        common.finite_score(cell.get("official_task_score"), "cell aggregate"), mean, rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("terminal task aggregate differs from raw rows")
                means.append(mean)
        if not math.isclose(common.finite_score(aggregate.get("macro_official_task_score"), "macro"),
                            sum(means) / len(means), rel_tol=0, abs_tol=1e-12):
            raise ValueError("terminal macro differs from raw scores")
        keyed[arm] = rows
        identities[arm] = {
            "raw_hashes": hashes, "runner_sha256": common.require_hash(protocol.get("script_sha256"), "runner"),
            "checkpoint_weight_sha256": WEIGHT_SHA256, "native_sha256_float32": TENSOR_SHA256["native"],
            "active_sha256_float32": TENSOR_SHA256[arm], "table_file_sha256": table_file_hash,
            "table_factor": 2.0 if arm != "native" else None, "attention_scaling": float(gain),
            "data_manifest_sha256": data_hash, "tokenizer_sha256": tokenizer_hash,
            "input_cell_sha256": cell_hashes, "seed": DATA_SEED, "ruler_commit": RULER_COMMIT,
        }
    for arm in ARMS:
        for field in ("runner_sha256", "data_manifest_sha256", "tokenizer_sha256", "input_cell_sha256"):
            if identities[arm][field] != identities["native"][field]:
                raise ValueError(f"{arm}: paired {field} mismatch")
        for key, row in keyed[arm].items():
            native = keyed["native"][key]
            for field in ("references", "official_metric", "prompt_tokens", "generation_budget"):
                if row[field] != native[field]:
                    raise ValueError(f"{arm}: paired row {field} mismatch")
    return keyed, identities


def paired_bootstrap(values: np.ndarray, rng) -> np.ndarray:
    if values.shape != (len(TASKS), ROWS_PER_CELL, len(ARMS)):
        raise ValueError("expected four tasks by eighty paired rows by three arms")
    draws = np.zeros((BOOTSTRAP_SAMPLES, len(ARMS)))
    for cell in values:
        indices = rng.integers(0, ROWS_PER_CELL, size=(BOOTSTRAP_SAMPLES, ROWS_PER_CELL))
        draws += cell[indices].mean(axis=1) / len(TASKS)
    return draws


def crossing_decision(native_macro: float, short_ci: list[float], long_ci: list[float]) -> str:
    if native_macro == 0:
        return "STOP_INSTRUMENT"
    if short_ci[1] < 0 and long_ci[0] > 0:
        return "CONFIRMED_CROSSING"
    if short_ci[0] > 0 and long_ci[1] < 0:
        return "REVERSED"
    return "UNRESOLVED"


def summarize_panel(panel: dict) -> dict:
    keyed, identities = validate_panel(panel)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores, primary, secondary = {}, {}, {}
    for length in LENGTHS:
        values = np.array([[[keyed[arm][task, length, i]["official_task_score"] for arm in ARMS]
                            for i in range(ROWS_PER_CELL)] for task in TASKS], dtype=np.float64)
        draws = paired_bootstrap(values, rng)
        scores[str(length)] = {}
        for column, arm in enumerate(ARMS):
            vector = values[:, :, column].mean(axis=1).tolist()
            scores[str(length)][arm] = {"task_vector": vector, "per_task": dict(zip(TASKS, vector)),
                                        "macro": float(np.mean(vector))}
        differences = values[:, :, 1] - values[:, :, 2]
        primary[str(length)] = {
            "contrast": "physical_x_minus_normalized_index", "delta": float(differences.mean()),
            "per_task_delta": differences.mean(axis=1).tolist(),
            "paired_stratified_ci975": np.quantile(draws[:, 1] - draws[:, 2], [.0125, .9875]).tolist(),
        }
        secondary[str(length)] = {}
        for column, arm in enumerate(ARMS[1:], start=1):
            secondary[str(length)][f"{arm}_minus_native"] = {
                "delta": float((values[:, :, column] - values[:, :, 0]).mean()),
                "per_task_delta": (values[:, :, column] - values[:, :, 0]).mean(axis=1).tolist(),
                "paired_stratified_ci95": np.quantile(draws[:, column] - draws[:, 0], [.025, .975]).tolist(),
            }
    native_macro = scores["32768"]["native"]["macro"]
    decision = crossing_decision(native_macro, primary["32768"]["paired_stratified_ci975"],
                                primary["65536"]["paired_stratified_ci975"])
    retention = {}
    for arm in ARMS:
        ratio = scores["32768"][arm]["macro"] / native_macro if native_macro > 0 else None
        retention[arm] = {"ratio": ratio, "threshold": .875, "status": "UNDEFINED" if ratio is None else (
            "PASS" if ratio >= .875 or math.isclose(ratio, .875, rel_tol=0, abs_tol=1e-12) else "FAIL")}
    return {
        "status": "K32_CROSSING_CONFIRMATION_COMPLETE", "decision": decision,
        "decision_rule": "CONFIRMED_CROSSING requires corrected 32K interval wholly negative AND 64K wholly positive; REVERSED requires both opposite corrected signs; otherwise UNRESOLVED. Native 1x zero overrides as STOP_INSTRUMENT.",
        "arms": list(ARMS), "tasks": list(TASKS), "task_vector_order": ["single1", "mk2", "mk3", "vt"],
        "lengths": list(LENGTHS), "rows_per_task_length": ROWS_PER_CELL, "rows_per_arm": 640,
        "data_seed": DATA_SEED, "scores": scores, "primary_physical_minus_index": primary,
        "secondary_vs_native": secondary,
        "native_retention": {"length": 32768, "arms": retention, "role": "separate point-estimate gate, not the crossing decision"},
        "bootstrap": {"replicates": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                      "unit": "paired row resampling within each fixed task; equal task-macro weights",
                      "primary_marginal_confidence": .975, "primary_family_alpha": .05,
                      "primary_multiplicity": "Bonferroni over two prespecified lengths; quantiles .0125/.9875",
                      "secondary_confidence": .95, "secondary_role": "descriptive, not multiplicity-corrected confirmatory claims",
                      "numpy_version": np.__version__, "quantile_interpolation": "linear",
                      "assumptions": "exchangeable rows within fixed tasks; no normality assumption; no exclusions or optional stopping"},
        "arm_identities": identities, "old_pilot_pooled": False, "profile_selection_performed": False,
        "evidence_scope": "Independent-input confirmation at one frozen checkpoint, conditional on registered tasks/scorer. Not training-seed replication, K causality, SOTA, strict full-string/EOS capability, or validation of a Native KL selector.",
        "input_hash_verification_scope": "raw results/examples/run manifests hashed locally; input-cell identities compared through bound terminal data receipts",
        "summary_script_sha256": common.file_hash(Path(__file__)),
        "utility_script_sha256": common.file_hash(Path(common.__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize_panel(load_panel(args.root))
    except ValueError as error:
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": str(error)}
    except (KeyError, TypeError, OSError):
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": "missing or malformed required input artifact/field"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "decision") if key in result}))
    return 2 if result["status"] == "INVALID_OR_INCOMPLETE_PANEL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
