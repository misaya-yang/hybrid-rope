#!/usr/bin/env python3
"""Hash-bound summary of the fixed Gemma K128 physical/index N80 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("physical_x", "normalized_index")
TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt")
LENGTH = 16384
ROWS_PER_TASK = 80
ROWS_PER_ARM = len(TASKS) * ROWS_PER_TASK
GENERATION_BUDGET = {"niah_single_1": 128, "niah_multikey_2": 128,
                     "niah_multikey_3": 128, "vt": 30}
EOS_TOKEN_ID = 1
DATA_SEED = 202609028
BOOTSTRAP_SEED = 202609029
BOOTSTRAP_SAMPLES = 10000
WEIGHT_SHA256 = "584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933"
NATIVE_SHA256 = "cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15"
TABLE_SHA256 = {
    "physical_x": "be5c2b3b4ce01d7fe6020cb01d9041e10aad93b9cdb03e989e64b8fa17561423",
    "normalized_index": "1b908f90aebccc006521b3840b5662c217caa040d173c974527d33c7ea9e9849",
}
GAIN = 1.102585782722872
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TERMINAL_STATUS = "TARGET_FREE_RULER_SMOKE_COMPLETE"
FROZEN_STATUS = "TARGET_FREE_RULER_SMOKE_FROZEN"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {name} SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"invalid {name} SHA-256") from error
    return value


def finite_score(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid {name} score")
    return float(value)


def official_score(row: dict) -> float:
    if row.get("official_metric") != "string_match_all":
        raise ValueError("official metric drift")
    prediction, references = row.get("prediction"), row.get("references")
    if (not isinstance(prediction, str) or not isinstance(references, list) or not references
            or not all(isinstance(reference, str) for reference in references)):
        raise ValueError("full prediction or references missing")
    expected = sum(reference.lower() in prediction.lower() for reference in references) / len(references)
    recorded = finite_score(row.get("official_task_score"), "row")
    if not math.isclose(recorded, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError("row score is not reproducible from the complete prediction")
    if not math.isclose(finite_score(row.get("reference_recall"), "reference recall"), expected,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("reference recall differs from the complete prediction")
    return recorded


def load_panel(root: Path) -> dict:
    panel = {}
    for arm in ARMS:
        directory = root / arm
        paths = {name: directory / filename for name, filename in (
            ("results", "results.json"), ("examples", "examples.jsonl"),
            ("run_manifest", "run_manifest.json"))}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"{arm}: missing terminal result, examples, or run manifest")
        result = json.loads(paths["results"].read_text())
        examples_hash = file_hash(paths["examples"])
        if result.get("status") != TERMINAL_STATUS:
            raise ValueError(f"{arm}: result is not terminal")
        if require_hash(result.get("results", {}).get("examples_sha256"), "examples") != examples_hash:
            raise ValueError(f"{arm}: examples hash mismatch")
        panel[arm] = {
            "result": result,
            "manifest": json.loads(paths["run_manifest"].read_text()),
            "rows": [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()],
            "raw_hashes": {name + "_sha256": file_hash(path) for name, path in paths.items()},
        }
    return panel


def validate_panel(panel: dict) -> tuple[dict, dict]:
    if set(panel) != set(ARMS):
        raise ValueError("confirmation requires exactly physical_x and normalized_index")
    expected_keys = {(task, LENGTH, index) for task in TASKS for index in range(ROWS_PER_TASK)}
    keyed, identities = {}, {}
    for arm in ARMS:
        item, result, protocol = panel[arm], panel[arm]["result"], panel[arm]["manifest"]
        if (result.get("status") != TERMINAL_STATUS or protocol.get("status") != FROZEN_STATUS
                or result.get("protocol") != protocol):
            raise ValueError(f"{arm}: terminal protocol and run manifest differ")
        if (protocol.get("tasks") != list(TASKS) or protocol.get("lengths") != [LENGTH]
                or type(protocol.get("limit_per_cell")) is not int
                or protocol["limit_per_cell"] != ROWS_PER_TASK
                or protocol.get("checkpoint_sha256") != WEIGHT_SHA256
                or protocol.get("model_type") != "gemma"
                or protocol.get("native_context_length") != 8192
                or protocol.get("profile_target_length") != LENGTH
                or protocol.get("table_factor") != 4.0
                or protocol.get("method") != "external_table_static"):
            raise ValueError(f"{arm}: fixed Gemma K128 N80 protocol mismatch")
        data, method, aggregate = result["data"], result["method"], result["results"]
        if (type(data.get("seed")) is not int or data["seed"] != DATA_SEED
                or data.get("ruler_commit") != RULER_COMMIT):
            raise ValueError("fresh seed or official RULER revision mismatch")
        data_hash = require_hash(protocol.get("data_manifest_sha256"), "data manifest")
        if (data.get("manifest_sha256") != data_hash
                or protocol.get("expected_data_manifest_sha256") != data_hash):
            raise ValueError("data manifest is not consistently frozen")
        tokenizer_hash = require_hash(data.get("tokenizer_sha256"), "tokenizer")
        gain = method.get("long_attention_scaling")
        if (method.get("method") != "external_table_static" or method.get("model_type") != "gemma"
                or method.get("native_context_length") != 8192
                or method.get("profile_target_length") != LENGTH
                or method.get("native_sha256_float32") != NATIVE_SHA256
                or method.get("active_sha256_float32") != TABLE_SHA256[arm]
                or method.get("table_factor") != 4.0 or method.get("table_support") != "native_div_factor"
                or method.get("initial_branch") != "external_long"
                or protocol.get("expected_native_sha256") != NATIVE_SHA256
                or protocol.get("expected_active_sha256") != TABLE_SHA256[arm]
                or protocol.get("table_sha256_float32") != TABLE_SHA256[arm]
                or type(gain) not in (int, float) or not math.isfinite(gain)
                or not math.isclose(gain, GAIN, rel_tol=0, abs_tol=1e-12)
                or protocol.get("long_attention_scaling") != gain):
            raise ValueError("checkpoint, Native, active table, support, or gain drift")
        table_file_hash = require_hash(protocol.get("table_file_sha256"), "table file")
        if set(data.get("cells", {})) != set(TASKS) or set(aggregate.get("cells", {})) != set(TASKS):
            raise ValueError("exactly four registered task families are required")
        cell_hashes = {}
        for task in TASKS:
            if set(data["cells"][task]) != {str(LENGTH)} or set(aggregate["cells"][task]) != {str(LENGTH)}:
                raise ValueError("unexpected task length cell")
            cell = data["cells"][task][str(LENGTH)]
            if cell.get("rows") != ROWS_PER_TASK or cell.get("selected_rows") != ROWS_PER_TASK:
                raise ValueError("each input cell must contain exactly eighty selected rows")
            cell_hashes[task] = require_hash(cell.get("sha256"), "input cell")
        rows = {}
        for row in item["rows"]:
            key = row.get("task"), row.get("nominal_length"), row.get("local_index")
            tokens = row.get("generated_token_ids")
            budget, prompt_tokens = row.get("generation_budget"), row.get("prompt_tokens")
            if (type(key[1]) is not int or type(key[2]) is not int or key not in expected_keys or key in rows
                    or not isinstance(tokens, list) or not tokens
                    or not all(type(token) is int and token >= 0 for token in tokens)
                    or type(row.get("ended_with_eos")) is not bool
                    or row["ended_with_eos"] != (tokens[-1] == EOS_TOKEN_ID)
                    or type(budget) is not int or budget != GENERATION_BUDGET.get(key[0]) or len(tokens) > budget
                    or (not row["ended_with_eos"] and len(tokens) != budget)
                    or type(prompt_tokens) is not int or not 0 < prompt_tokens
                    or prompt_tokens + budget > LENGTH):
                raise ValueError("row identity or full token/EOS metadata is missing or invalid")
            official_score(row)
            rows[key] = row
        if set(rows) != expected_keys or aggregate.get("examples") != ROWS_PER_ARM:
            raise ValueError("each arm must contain exactly 320 registered rows")
        task_vector = []
        for task in TASKS:
            mean = sum(official_score(rows[task, LENGTH, index]) for index in range(ROWS_PER_TASK)) / ROWS_PER_TASK
            cell = aggregate["cells"][task][str(LENGTH)]
            if (cell.get("rows") != ROWS_PER_TASK
                    or not math.isclose(finite_score(cell.get("official_task_score"), "cell"), mean,
                                        rel_tol=0, abs_tol=1e-12)):
                raise ValueError("terminal task aggregate differs from raw rows")
            task_vector.append(mean)
        if not math.isclose(finite_score(aggregate.get("macro_official_task_score"), "macro"),
                            sum(task_vector) / len(TASKS), rel_tol=0, abs_tol=1e-12):
            raise ValueError("terminal macro differs from raw rows")
        raw_hashes = {name: require_hash(value, name) for name, value in item["raw_hashes"].items()}
        if aggregate.get("examples_sha256") != raw_hashes["examples_sha256"]:
            raise ValueError("terminal examples hash differs from loaded raw artifact")
        identities[arm] = {
            "raw_hashes": raw_hashes, "runner_sha256": require_hash(protocol.get("script_sha256"), "runner"),
            "checkpoint_weight_sha256": WEIGHT_SHA256, "native_sha256_float32": NATIVE_SHA256,
            "active_sha256_float32": TABLE_SHA256[arm], "table_file_sha256": table_file_hash,
            "attention_scaling": float(gain), "data_manifest_sha256": data_hash,
            "tokenizer_sha256": tokenizer_hash, "input_cell_sha256": cell_hashes,
            "seed": DATA_SEED, "ruler_commit": RULER_COMMIT,
            "torch": result["runtime"]["torch"], "cuda": result["runtime"]["cuda"],
        }
        keyed[arm] = rows
    shared = ("runner_sha256", "checkpoint_weight_sha256", "native_sha256_float32",
              "attention_scaling", "data_manifest_sha256", "tokenizer_sha256",
              "input_cell_sha256", "seed", "ruler_commit", "torch", "cuda")
    for field in shared:
        if identities["physical_x"][field] != identities["normalized_index"][field]:
            raise ValueError(f"paired {field} mismatch")
    for key in expected_keys:
        left, right = keyed["physical_x"][key], keyed["normalized_index"][key]
        for field in ("references", "official_metric", "prompt_tokens", "generation_budget"):
            if left[field] != right[field]:
                raise ValueError(f"paired row {field} mismatch")
    return keyed, identities


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if values.shape != (len(TASKS), ROWS_PER_TASK, len(ARMS)):
        raise ValueError("expected four tasks by eighty paired rows by two arms")
    draws = np.zeros((BOOTSTRAP_SAMPLES, len(ARMS)), dtype=np.float64)
    for task_values in values:
        indices = rng.integers(0, ROWS_PER_TASK, size=(BOOTSTRAP_SAMPLES, ROWS_PER_TASK))
        draws += task_values[indices].mean(axis=1) / len(TASKS)
    return draws


def decision(ci95: list[float]) -> str:
    if ci95[0] > 0:
        return "ABOVE_ZERO"
    if ci95[1] < 0:
        return "BELOW_ZERO"
    return "UNRESOLVED"


def summarize_panel(panel: dict) -> dict:
    keyed, identities = validate_panel(panel)
    values = np.array([[[keyed[arm][task, LENGTH, index]["official_task_score"] for arm in ARMS]
                        for index in range(ROWS_PER_TASK)] for task in TASKS], dtype=np.float64)
    vectors = values.mean(axis=1)
    scores = {arm: {"task_vector": vectors[:, column].tolist(),
                    "per_task": dict(zip(TASKS, vectors[:, column].tolist())),
                    "macro": float(vectors[:, column].mean())}
              for column, arm in enumerate(ARMS)}
    if scores["physical_x"]["macro"] == scores["normalized_index"]["macro"] == 0:
        raise ValueError("both arms collapsed to zero; confirmation instrument is unresolved")
    draws = paired_bootstrap(values, np.random.default_rng(BOOTSTRAP_SEED))
    delta = values[:, :, 1] - values[:, :, 0]
    ci95 = np.quantile(draws[:, 1] - draws[:, 0], [.025, .975]).tolist()
    return {
        "status": "GEMMA_K128_COORDINATE_CONFIRMATION_COMPLETE", "decision": decision(ci95),
        "decision_rule": "ABOVE_ZERO iff the 95% interval is wholly above zero; BELOW_ZERO iff wholly below zero; otherwise UNRESOLVED.",
        "arms": list(ARMS), "tasks": list(TASKS), "task_vector_order": list(TASKS),
        "length": LENGTH, "rows_per_task": ROWS_PER_TASK, "rows_per_arm": ROWS_PER_ARM,
        "data_seed": DATA_SEED, "scores": scores,
        "primary_index_minus_physical": {"delta": float(delta.mean()),
            "per_task_delta": delta.mean(axis=1).tolist(), "paired_task_stratified_ci95": ci95},
        "bootstrap": {"replicates": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
            "confidence": .95, "quantiles": [.025, .975], "quantile_interpolation": "linear",
            "unit": "paired local_index resampling within each fixed task; equal task weights",
            "numpy_version": np.__version__,
            "assumptions": "exchangeable rows within task; no exclusions, pooling, optional stopping, or normality assumption"},
        "arm_identities": identities, "old_pilot_pooled": False, "profile_selection_performed": False,
        "evidence_scope": "Independent-input Gemma K128 16K ordering for two frozen tables only; not K causality, universality, natural-text quality, SOTA, or token-level equality between arms.",
        "summary_script_sha256": file_hash(Path(__file__)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize_panel(load_panel(args.root))
    except ValueError as error:
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL", "reason": str(error)}
    except (KeyError, TypeError, OSError, json.JSONDecodeError):
        result = {"status": "INVALID_OR_INCOMPLETE_PANEL",
                  "reason": "missing or malformed required input artifact/field"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "decision") if key in result}))
    return 2 if result["status"] == "INVALID_OR_INCOMPLETE_PANEL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
