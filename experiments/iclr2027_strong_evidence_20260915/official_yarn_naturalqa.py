#!/usr/bin/env python3
"""Run and report one official-static-YaRN natural-QA arm per frozen model.

The four allowed conditions are deliberately heterogeneous at the benchmark
layer.  Llama and OLMo use the five-task frozen LongBench Natural-QA pool;
Qwen and GLM use the complete eligible InfiniteBench English long-book QA
pool.  Reports therefore remain per model and never pool across models.

The Llama generation is owned by ``run_naturalqa_yarn.sh``.  This module only
finalizes its three-arm report.  For OLMo, Qwen and GLM, execution installs the
deterministic official static YaRN table, runs exactly one resumable generation
arm, verifies it against the completed TailSpline/MrPro runtime contract, and
then creates a three-arm report with two family-adjusted primary contrasts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report import (
    TASKS as LONGBENCH_TASKS,
    bootstrap as longbench_bootstrap,
    pooled_cluster_bootstrap,
    task_macro,
)
from experiments.iclr2027_strong_evidence_20260915.run_natural_long import (
    acquire_gpu_lock,
    score_natural_output,
)
from scripts.eval.longbench_metrics import qa_f1_score


METHOD_IDENTITY = "official static YaRN, zero-training installation"
REPORT_CONTRACT = "FOUR_MODEL_OFFICIAL_STATIC_YARN_NATURAL_QA_TRIARM_V1"
RUN_CONTRACT = "OFFICIAL_STATIC_YARN_NATURAL_QA_SINGLE_ARM_V1"
ALLOWED_CONDITIONS = (
    "llama3_8b", "qwen25_3b", "olmo2_1b", "glm4_9b",
    "glm4_9b_second_books",
)
EXCLUDED_MODELS = ("Qwen2.5-1.5B-Instruct",)
ARMS = ("tailspline", "mrpro", "yarn")
PRIMARY_CONTRASTS = ("tailspline_minus_mrpro", "tailspline_minus_yarn")
BOOTSTRAP_DRAWS = 20_000


@dataclass(frozen=True)
class Condition:
    name: str
    public_model_name: str
    model_id: str
    table_model_id: str
    model: Path
    scale: float
    family: str
    panel: Path
    asset_manifest: Path
    baseline_runs: Path
    baseline_tables: Path
    yarn_root: Path
    yarn_table: Path
    data_manifest: Path
    length_cap: int
    expected_rows: int
    expected_clusters: int
    rows_per_task: int
    delegated_generation: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def condition_from_name(
    name: str, plan: Path, *, model_override: Path | None = None,
) -> Condition:
    if name not in ALLOWED_CONDITIONS:
        raise ValueError(
            f"condition must be one of {ALLOWED_CONDITIONS}; Qwen2.5-1.5B is excluded"
        )
    plan = Path(plan)
    if name == "llama3_8b":
        natural = plan / "tailspline_llama_s4_naturalqa631"
        return Condition(
            name=name,
            public_model_name="Meta-Llama-3-8B-Instruct",
            model_id="llama3_8b",
            table_model_id="meta_llama3_8b_instruct",
            model=model_override or Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"),
            scale=4.0,
            family="longbench_naturalqa631",
            panel=natural / "assets/inputs.jsonl",
            asset_manifest=natural / "assets/manifest.json",
            baseline_runs=natural / "runs",
            baseline_tables=plan / "tailspline_llama_s4_classic/tables",
            yarn_root=plan / "tailspline_llama_s4_naturalqa631_yarn_a1",
            yarn_table=plan / "tailspline_llama_s4_classic_strong_baselines/tables/yarn.json",
            data_manifest=plan / "tailspline_llama_s4_classic/assets/ppl46/manifest.json",
            length_cap=32768,
            expected_rows=631,
            expected_clusters=524,
            rows_per_task=0,
            delegated_generation=True,
        )
    if name == "olmo2_1b":
        natural = plan / "tailspline_olmo_s4_naturalqa631"
        output = plan / "tailspline_olmo_s4_naturalqa631_yarn_a1"
        return Condition(
            name=name,
            public_model_name="OLMo-2-0425-1B-Instruct",
            model_id="olmo2_1b",
            table_model_id="olmo2_1b",
            model=model_override or Path(
                "/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct"
            ),
            scale=4.0,
            family="longbench_naturalqa631",
            panel=natural / "assets/inputs.jsonl",
            asset_manifest=natural / "assets/manifest.json",
            baseline_runs=natural / "runs",
            baseline_tables=plan / "tailspline_olmo_s4_classic/tables",
            yarn_root=output,
            yarn_table=plan / "official_yarn_quick/olmo2_1b_s4_16k/tables/yarn.json",
            data_manifest=plan / "tailspline_olmo_s4_classic/assets/ppl46/manifest.json",
            length_cap=16384,
            expected_rows=631,
            expected_clusters=524,
            rows_per_task=0,
        )
    if name == "qwen25_3b":
        root = plan / "four_model_128k_extreme/qwen25_3b_128k"
        return Condition(
            name=name,
            public_model_name="Qwen2.5-3B-Instruct",
            model_id="qwen25_3b",
            table_model_id="qwen25_3b",
            model=model_override or Path("/root/autodl-tmp/rope_qwen_baseline_20260907/model"),
            scale=4.0,
            family="infinitebench_en_qa",
            panel=root / "infinitebench_en_qa_assets/inputs.jsonl",
            asset_manifest=root / "infinitebench_en_qa_assets/manifest.json",
            baseline_runs=root / "infinitebench_en_qa_evaluation/runs",
            baseline_tables=root / "infinitebench_en_qa_evaluation/tables",
            yarn_root=root / "infinitebench_en_qa_yarn_a1",
            yarn_table=root / "official_yarn/tables/yarn.json",
            data_manifest=root / "infinitebench_en_qa_assets/manifest.json",
            length_cap=131072,
            expected_rows=35,
            expected_clusters=7,
            rows_per_task=50,
        )
    root = plan / "glm4_9b_s4_128k"
    if name == "glm4_9b_second_books":
        assets = root / "en_qa_second_books_assets"
        evaluation = root / "en_qa_second_books_evaluation"
        return Condition(
            name=name,
            public_model_name="GLM-4-9B-0414",
            model_id="glm4_9b_0414",
            table_model_id="glm4_9b_0414",
            model=model_override or Path("/root/models/GLM-4-9B-0414"),
            scale=4.0,
            family="infinitebench_en_qa",
            panel=assets / "inputs.jsonl",
            asset_manifest=assets / "manifest.json",
            baseline_runs=evaluation / "runs",
            baseline_tables=evaluation / "tables",
            yarn_root=root / "en_qa_second_books_yarn_a1",
            yarn_table=plan / "official_yarn_quick/glm4_9b_s4_128k/tables/yarn.json",
            data_manifest=assets / "manifest.json",
            length_cap=131072,
            expected_rows=77,
            expected_clusters=15,
            rows_per_task=100,
        )
    output = root / "en_qa_yarn_a1"
    return Condition(
        name=name,
        public_model_name="GLM-4-9B-0414",
        model_id="glm4_9b_0414",
        table_model_id="glm4_9b_0414",
        model=model_override or Path("/root/models/GLM-4-9B-0414"),
        scale=4.0,
        family="infinitebench_en_qa",
        panel=root / "en_qa_assets/inputs.jsonl",
        asset_manifest=root / "en_qa_assets/manifest.json",
        baseline_runs=root / "en_qa_evaluation/runs",
        baseline_tables=root / "en_qa_evaluation/tables",
        yarn_root=output,
        yarn_table=plan / "official_yarn_quick/glm4_9b_s4_128k/tables/yarn.json",
        data_manifest=root / "en_qa_assets/manifest.json",
        length_cap=131072,
        expected_rows=35,
        expected_clusters=7,
        rows_per_task=50,
    )


def _cluster_id(row: Mapping[str, object]) -> str:
    value = row.get("document_cluster_id") or row.get("source_cluster_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row lacks a source-context cluster: {row.get('row_id')}")
    return value


def validate_panel(condition: Condition) -> tuple[list[dict], dict]:
    for path in (condition.panel, condition.asset_manifest, condition.model / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = read_json(condition.asset_manifest)
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"asset manifest is not COMPLETE: {condition.asset_manifest}")
    rows = read_jsonl(condition.panel)
    if len(rows) != condition.expected_rows:
        raise ValueError(f"{condition.name} panel has {len(rows)} rows")
    if len({str(row.get("row_id")) for row in rows}) != len(rows):
        raise ValueError("panel row IDs are missing or duplicated")
    if manifest.get("inputs_sha256") != sha256_file(condition.panel):
        raise ValueError("panel SHA256 differs from the frozen manifest")
    if any(
        not isinstance(row.get("prompt_sha256"), str)
        or int(row.get("input_tokens", -1)) != len(row.get("prompt_ids", []))
        or int(row.get("length_cap", -1)) != condition.length_cap
        for row in rows
    ):
        raise ValueError("panel prompt identity or length cap is invalid")
    clusters = {_cluster_id(row) for row in rows}
    if len(clusters) != condition.expected_clusters:
        raise ValueError(f"source-cluster count drift: {len(clusters)}")
    task_counts = Counter(str(row.get("task")) for row in rows)
    if condition.family == "longbench_naturalqa631":
        expected = Counter({
            "hotpotqa": 166,
            "2wikimqa": 173,
            "qasper": 119,
            "narrativeqa": 61,
            "multifieldqa_en": 112,
        })
        if task_counts != expected:
            raise ValueError(f"Natural-QA task-count drift: {task_counts}")
    elif task_counts != Counter({"longbook_qa_eng": 35}):
        raise ValueError(f"InfiniteBench En.QA task-count drift: {task_counts}")
    return rows, manifest


def _generation_rows(run: Path, panel: list[dict], arm: str) -> list[dict]:
    status = run / "status.json"
    generations = run / "generations.jsonl"
    if not status.is_file() or not generations.is_file():
        raise FileNotFoundError(f"{arm} is not complete: {run}")
    if read_json(status) != {"status": "COMPLETE", "rows": len(panel), "lm_rows": 0}:
        raise ValueError(f"{arm} status differs from the frozen panel")
    values = read_jsonl(generations)
    if len(values) != len(panel):
        raise ValueError(f"{arm} generation count differs from the frozen panel")
    for expected, actual in zip(panel, values):
        for key in ("row_id", "task", "prompt_sha256", "input_tokens", "references"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"{arm} input identity drift at {expected['row_id']}/{key}")
        if "generated_ids" not in actual or "output_text" not in actual:
            raise ValueError(f"{arm} does not retain its complete output")
    return values


def _recovery_runtime_signature(contract: Mapping[str, object]) -> dict:
    batch_size = int(contract.get("batch_size", -1))
    prefill = int(contract.get("prefill_chunk_size", -1))
    generation_order = contract.get("generation_order")
    if generation_order is None and batch_size == 1:
        generation_order = "panel_order_v1"
    strategy = contract.get("generation_prefill_strategy")
    if strategy is None and prefill >= 0:
        strategy = "direct_generate_v1" if prefill == 0 else "dynamic_cache_lower_right_v1"
    return {
        "base_arm": contract.get("base_arm"),
        "unadapted": contract.get("unadapted"),
        "row_ids": contract.get("row_ids"),
        "generation_length_caps": contract.get("generation_length_caps"),
        "lm_enabled": contract.get("lm_enabled"),
        "prefill_chunk_size": prefill,
        "generation_prefill_strategy": strategy,
        "batch_size": batch_size,
        "left_pad_batches": bool(contract.get("left_pad_batches", False)),
        "generation_order": generation_order,
        "runtime_versions": contract.get("runtime_versions"),
        "row_split": contract.get("row_split"),
    }


def _compact_runtime(runtime: dict) -> dict:
    return {
        "signature_sha256": sha256_json(runtime),
        "row_count": len(runtime.get("row_ids") or []),
        **{key: runtime.get(key) for key in (
            "base_arm", "unadapted", "generation_length_caps", "lm_enabled",
            "prefill_chunk_size", "generation_prefill_strategy", "batch_size",
            "left_pad_batches", "generation_order", "runtime_versions", "row_split",
        )},
    }


def _compact_wrapper(wrapper: dict | None) -> dict | None:
    if wrapper is None:
        return None
    return {
        "signature_sha256": sha256_json(wrapper),
        "row_count": len(wrapper.get("row_ids") or []),
        **{key: wrapper.get(key) for key in (
            "status", "model_id", "model_config_sha256", "scale", "lengths",
            "rows_per_task", "benchmark", "data_manifest_sha256", "inputs_sha256",
            "batch_size", "prefill_chunk_size", "generation_backend",
            "natural_score_source", "ruler_contains_is_natural_score",
        )},
    }


def _runtime_identity_level(runtime: dict) -> str:
    return (
        "EXACT_RECORDED_CONTRACT"
        if isinstance(runtime.get("runtime_versions"), dict)
        else "RECORDED_FIELDS_ONLY_LEGACY_CONTRACT"
    )


def _runtime_matches(expected: dict, actual: dict) -> bool:
    """Compare every recorded baseline field without inventing legacy metadata."""
    if expected.get("runtime_versions") is not None:
        return expected == actual
    return all(
        actual.get(key) == value
        for key, value in expected.items()
        if key != "runtime_versions"
    )


def _wrapper_runtime_signature(contract: Mapping[str, object]) -> dict:
    return {
        key: contract.get(key)
        for key in (
            "status", "model_id", "model_config_sha256", "scale", "lengths",
            "rows_per_task", "benchmark", "data_manifest_sha256", "inputs_sha256",
            "row_ids", "batch_size", "prefill_chunk_size", "generation_backend",
            "natural_score_source", "ruler_contains_is_natural_score",
        )
    }


def validate_baselines(
    condition: Condition, panel: list[dict],
) -> tuple[dict[str, list[dict]], dict, dict | None]:
    raw = {}
    recovery_contracts = {}
    wrapper_contracts = {}
    for arm in ("tailspline", "mrpro"):
        run = condition.baseline_runs / arm
        raw[arm] = _generation_rows(run, panel, arm)
        contract = run / "contract.json"
        if not contract.is_file():
            raise FileNotFoundError(contract)
        recovery_contracts[arm] = read_json(contract)
        wrapper = run / "wrapper_contract.json"
        if condition.family == "infinitebench_en_qa":
            if not wrapper.is_file():
                raise FileNotFoundError(wrapper)
            wrapper_contracts[arm] = read_json(wrapper)
    recovery = {
        arm: _recovery_runtime_signature(value)
        for arm, value in recovery_contracts.items()
    }
    if recovery["tailspline"] != recovery["mrpro"]:
        raise ValueError("TailSpline and MrPro recovery runtime contracts differ")
    if recovery["tailspline"]["base_arm"] != "Native" or not recovery["tailspline"]["unadapted"]:
        raise ValueError("baseline runs are not unadapted static-table installations")
    wrapper_signature = None
    if wrapper_contracts:
        wrappers = {
            arm: _wrapper_runtime_signature(value)
            for arm, value in wrapper_contracts.items()
        }
        if wrappers["tailspline"] != wrappers["mrpro"]:
            raise ValueError("TailSpline and MrPro natural wrapper contracts differ")
        wrapper_signature = wrappers["tailspline"]
        if (
            wrapper_signature["model_id"] != condition.model_id
            or wrapper_signature["inputs_sha256"] != sha256_file(condition.panel)
            or wrapper_signature["row_ids"] != [row["row_id"] for row in panel]
            or wrapper_signature["benchmark"] != "infinitebench"
        ):
            raise ValueError("natural wrapper identity differs from the frozen condition")
    return raw, recovery["tailspline"], wrapper_signature


def validate_yarn_table(path: Path, condition: Condition) -> dict:
    receipt = read_json(path)
    table = receipt.get("table", receipt)
    construction = table.get("construction", {})
    if (
        receipt.get("model_id") != condition.table_model_id
        or not math.isclose(float(receipt.get("scale", float("nan"))), condition.scale)
        or receipt.get("role") != "baseline"
        or construction.get("mode") != "official_yarn_native"
        or construction.get("identity")
        != "official static YaRN frequency map on a frozen checkpoint; no YaRN SFT"
        or construction.get("model_weight_updates") != 0
        or construction.get("same_table_all_layers_and_lengths") is not True
        or not isinstance(table.get("values_float32"), list)
        or not table["values_float32"]
        or not math.isfinite(float(table.get("gain", float("nan"))))
    ):
        raise ValueError(f"YaRN table is not the frozen zero-training installation: {path}")
    return receipt


def _create_yarn_table(
    condition: Condition, *, python: Path, repo: Path,
) -> None:
    condition.yarn_table.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), "-m",
        "experiments.fixed_rope_three_interfaces_20260913.tables", "analytic",
        "--config", str(condition.model / "config.json"),
        "--method", "yarn", "--scale", str(condition.scale),
        "--candidate-id", f"{condition.name}_s4_naturalqa_official_static_yarn",
        "--model-id", condition.table_model_id,
        "--role", "baseline",
        "--changed-variable", "internal_frequency_allocation",
        "--out", str(condition.yarn_table),
    ]
    subprocess.run(command, cwd=repo, check=True)


def _runtime_preflight(runtime: dict) -> None:
    versions = runtime.get("runtime_versions")
    if not isinstance(versions, dict):
        raise ValueError("baseline contract lacks runtime versions")
    import torch
    import transformers
    # Match the frozen recovery-v2 runtime setup before comparing the process
    # setting recorded by the completed baselines.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if versions.get("torch") != torch.__version__:
        raise ValueError("current torch version differs from the completed baselines")
    if versions.get("transformers") != transformers.__version__:
        raise ValueError("current transformers version differs from the completed baselines")
    if versions.get("model_dtype") != "bfloat16":
        raise ValueError("baseline model dtype is not bfloat16")
    if versions.get("attention_backend") != "torch_sdpa_flash_only":
        raise ValueError("baseline attention backend is not the frozen Flash-SDPA path")
    if bool(versions.get("allow_tf32")) != bool(torch.backends.cuda.matmul.allow_tf32):
        raise ValueError("current TF32 setting differs from the completed baselines")


def _expected_wrapper(
    condition: Condition, panel: list[dict], runtime: dict,
    wrapper: dict | None,
) -> dict:
    return {
        "status": RUN_CONTRACT,
        "condition": condition.name,
        "public_model_name": condition.public_model_name,
        "model_id": condition.model_id,
        "method_identity": METHOD_IDENTITY,
        "model_config_sha256": sha256_file(condition.model / "config.json"),
        "scale": condition.scale,
        "family": condition.family,
        "panel_sha256": sha256_file(condition.panel),
        "asset_manifest_sha256": sha256_file(condition.asset_manifest),
        "data_manifest_sha256": sha256_file(condition.data_manifest),
        "table_receipt_sha256": sha256_file(condition.yarn_table),
        "rows": len(panel),
        "row_ids": [row["row_id"] for row in panel],
        "runtime": runtime,
        "baseline_wrapper_runtime": wrapper,
        "cross_model_pooling_allowed": False,
        "excluded_models": list(EXCLUDED_MODELS),
    }


def _ensure_wrapper(path: Path, expected: dict) -> None:
    if path.is_file():
        if read_json(path) != expected:
            raise ValueError(f"existing YaRN wrapper contract differs: {path}")
        return
    owned = ("contract.json", "generations.jsonl", "status.json", "summary.json")
    if any((path.parent / name).exists() for name in owned):
        raise ValueError("YaRN outputs exist without their frozen wrapper contract")
    atomic_json(path, expected)


def _validate_yarn_recovery_contract(
    condition: Condition, baseline_runtime: dict,
) -> None:
    contract = read_json(condition.yarn_root / "runs/yarn/contract.json")
    if not _runtime_matches(baseline_runtime, _recovery_runtime_signature(contract)):
        raise ValueError("YaRN recovery runtime differs from TailSpline/MrPro")
    table = validate_yarn_table(condition.yarn_table, condition).get("table")
    table = table or validate_yarn_table(condition.yarn_table, condition)
    installed = contract.get("static_table") or {}
    if (
        installed.get("values_float32") != table.get("values_float32")
        or installed.get("gain") != table.get("gain")
    ):
        raise ValueError("installed YaRN table differs from the frozen receipt")


def _execute_yarn(
    condition: Condition, panel: list[dict], baseline_runtime: dict,
    baseline_wrapper: dict | None, *, python: Path, repo: Path,
) -> None:
    if condition.delegated_generation:
        raise ValueError("Llama generation must use run_naturalqa_yarn.sh")
    if not condition.yarn_table.is_file():
        _create_yarn_table(condition, python=python, repo=repo)
    validate_yarn_table(condition.yarn_table, condition)
    _runtime_preflight(baseline_runtime)
    run = condition.yarn_root / "runs/yarn"
    run.mkdir(parents=True, exist_ok=True)
    condition.yarn_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    expected = _expected_wrapper(condition, panel, baseline_runtime, baseline_wrapper)
    _ensure_wrapper(run / "wrapper_contract.json", expected)
    status = run / "status.json"
    if status.is_file():
        _generation_rows(run, panel, "yarn")
        _validate_yarn_recovery_contract(condition, baseline_runtime)
        return
    command = [
        str(python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(condition.data_manifest),
        "--model", str(condition.model),
        "--arm", "Native",
        "--extra-panel", str(condition.panel),
        "--only-extra-panels", "--skip-lm",
        "--length-cap", str(condition.length_cap),
        "--prefill-chunk-size", str(baseline_runtime["prefill_chunk_size"]),
        "--batch-size", str(baseline_runtime["batch_size"]),
        "--static-table-json", str(condition.yarn_table),
        "--table-label", f"{condition.name}_s4_naturalqa_official_static_yarn",
        "--out", str(run), "--execute",
    ]
    if baseline_runtime["left_pad_batches"]:
        command.append("--left-pad-batches")
    if baseline_runtime["generation_order"] == "longest_first_shape_sorted_v1":
        command.append("--longest-first")
    with (condition.yarn_root / "logs/yarn.log").open("a") as log:
        subprocess.run(
            command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    _generation_rows(run, panel, "yarn")
    _validate_yarn_recovery_contract(condition, baseline_runtime)


def _raw_map(rows: list[dict]) -> dict[str, dict]:
    return {str(row["row_id"]): row for row in rows}


def _output_health(panel: list[dict], values: Mapping[str, dict]) -> dict:
    tasks = sorted({str(row["task"]) for row in panel})
    ids_by_task = {
        task: [str(row["row_id"]) for row in panel if row["task"] == task]
        for task in tasks
    }

    def one(ids: list[str]) -> dict:
        lengths = [len(values[row_id]["generated_ids"]) for row_id in ids]
        return {
            "rows": len(ids),
            "ended_eos": sum(bool(values[row_id].get("ended_eos")) for row_id in ids),
            "hit_cap": sum(bool(values[row_id].get("hit_cap")) for row_id in ids),
            "empty": sum(bool(values[row_id].get("empty")) for row_id in ids),
            "generated_tokens": sum(lengths),
            "output_length_quantiles": (
                np.quantile(lengths, [0, 0.5, 0.95, 1]).tolist() if lengths else []
            ),
        }

    all_ids = [str(row["row_id"]) for row in panel]
    return {**one(all_ids), "by_task": {task: one(ids_by_task[task]) for task in tasks}}


def _longbench_scores(panel: list[dict], raw: dict[str, list[dict]]) -> tuple[dict, dict]:
    panel_map = _raw_map(panel)
    outputs = {arm: _raw_map(values) for arm, values in raw.items()}
    ids = [row["row_id"] for row in panel]
    arms = {}
    for arm, values in outputs.items():
        for row_id in ids:
            recomputed = qa_f1_score(values[row_id]["output_text"], panel_map[row_id]["references"])
            if abs(recomputed - float(values[row_id]["whole_response_f1"])) > 1e-12:
                raise ValueError(f"LongBench token-F1 drift: {arm}/{row_id}")
        macro, tasks = task_macro(values, ids)
        arms[arm] = {
            "macro_f1": macro,
            "by_task": tasks,
            "output_health": _output_health(panel, values),
        }

    def contrast(candidate: str, baseline: str, seed: int, family_size: int) -> dict:
        estimate = arms[candidate]["macro_f1"] - arms[baseline]["macro_f1"]
        result = longbench_bootstrap(
            panel_map, outputs[candidate], outputs[baseline], ids,
            draws=BOOTSTRAP_DRAWS, seed=seed, family_size=family_size,
        )
        result.update({
            "candidate": candidate,
            "baseline": baseline,
            "estimate": estimate,
            "by_task": {
                task: arms[candidate]["by_task"][task] - arms[baseline]["by_task"][task]
                for task in LONGBENCH_TASKS
            },
        })
        return result

    contrasts = {
        "tailspline_minus_mrpro": contrast("tailspline", "mrpro", 20261201, 2),
        "tailspline_minus_yarn": contrast("tailspline", "yarn", 20261202, 2),
        "mrpro_minus_yarn_descriptive": contrast("mrpro", "yarn", 20261203, 1),
    }
    sensitivity = {
        name: pooled_cluster_bootstrap(
            panel_map,
            outputs[value["candidate"]], outputs[value["baseline"]], ids,
            draws=BOOTSTRAP_DRAWS, seed=20261300 + index,
        )
        for index, (name, value) in enumerate(contrasts.items())
    }
    return arms, {"contrasts": contrasts, "question_equal_sensitivity": sensitivity}


def _cluster_bootstrap(
    panel: list[dict], candidate: Mapping[str, float], baseline: Mapping[str, float],
    *, seed: int, family_size: int,
) -> dict:
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in panel:
        clusters[_cluster_id(row)].append(str(row["row_id"]))
    keys = sorted(clusters)
    sums = np.asarray([
        sum(candidate[row_id] - baseline[row_id] for row_id in clusters[key])
        for key in keys
    ])
    counts = np.asarray([len(clusters[key]) for key in keys])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(keys), size=(BOOTSTRAP_DRAWS, len(keys)))
    draws = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "bootstrap_mean": float(draws.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "familywise_ci95_bonferroni": np.quantile(
            draws, [0.025 / family_size, 1.0 - 0.025 / family_size]
        ).tolist(),
        "comparison_family_size": family_size,
        "probability_delta_gt_zero": float(np.mean(draws > 0.0)),
        "resampling": "paired source-context clusters; rows within a selected book stay together",
    }


def _infinitebench_scores(
    condition: Condition, panel: list[dict], raw: dict[str, list[dict]],
) -> tuple[dict, dict]:
    score_maps = {}
    arms = {}
    for arm, values in raw.items():
        scored = []
        mapping = {}
        for prepared, generated in zip(panel, values):
            result = score_natural_output(prepared, generated["output_text"])
            if result["official_score_contract"] != "infinitebench_en_qa_rouge_f1_v1":
                raise ValueError("InfiniteBench En.QA used the wrong official scorer")
            record = dict(generated)
            record.update({key: prepared.get(key) for key in (
                "benchmark", "task", "source_id", "source_cluster_id",
                "input_tokens", "length_cap", "length_bucket", "references",
                "score_contract", "prompt_sha256",
            )})
            record.update(result)
            record["arm"] = arm
            record["ruler_contains_used_for_natural_score"] = False
            scored.append(record)
            mapping[str(prepared["row_id"])] = float(result["official_score"])
        atomic_jsonl(condition.yarn_root / f"scored/{arm}.jsonl", scored)
        score_maps[arm] = mapping
        arms[arm] = {
            "qa_f1": float(np.mean(list(mapping.values()))),
            "output_health": _output_health(panel, _raw_map(values)),
        }

    def contrast(candidate: str, baseline: str, seed: int, family_size: int) -> dict:
        estimate = arms[candidate]["qa_f1"] - arms[baseline]["qa_f1"]
        return {
            "candidate": candidate,
            "baseline": baseline,
            "estimate": estimate,
            **_cluster_bootstrap(
                panel, score_maps[candidate], score_maps[baseline],
                seed=seed, family_size=family_size,
            ),
        }

    contrasts = {
        "tailspline_minus_mrpro": contrast("tailspline", "mrpro", 20261401, 2),
        "tailspline_minus_yarn": contrast("tailspline", "yarn", 20261402, 2),
        "mrpro_minus_yarn_descriptive": contrast("mrpro", "yarn", 20261403, 1),
    }
    return arms, {"contrasts": contrasts}


def build_report(
    condition: Condition, panel: list[dict], raw: dict[str, list[dict]],
    baseline_runtime: dict, baseline_wrapper: dict | None,
) -> dict:
    if set(raw) != set(ARMS):
        raise ValueError("three-arm report requires TailSpline, MrPro and YaRN")
    if condition.family == "longbench_naturalqa631":
        arms, inference = _longbench_scores(panel, raw)
        metric = "five-task equal macro whole-response LongBench-normalized token F1"
        scorer = "scripts.eval.longbench_metrics.qa_f1_score"
    else:
        arms, inference = _infinitebench_scores(condition, panel, raw)
        metric = "InfiniteBench longbook_qa_eng official English-QA token F1"
        scorer = "run_natural_long.score_natural_output; ruler contains score is excluded"
    cluster_tasks: dict[str, set[str]] = defaultdict(set)
    for row in panel:
        cluster_tasks[_cluster_id(row)].add(str(row["task"]))
    report = {
        "status": "COMPLETE",
        "contract": REPORT_CONTRACT,
        "condition": condition.name,
        "public_model_name": condition.public_model_name,
        "model_id": condition.model_id,
        "method_identity": METHOD_IDENTITY,
        "benchmark_family": condition.family,
        "metric": metric,
        "official_scorer": scorer,
        "scale": condition.scale,
        "length_cap": condition.length_cap,
        "rows_per_arm": len(panel),
        "tasks": dict(Counter(row["task"] for row in panel)),
        "input_token_range": [
            min(int(row["input_tokens"]) for row in panel),
            max(int(row["input_tokens"]) for row in panel),
        ],
        "source_context_clusters": len(cluster_tasks),
        "arms": arms,
        **inference,
        "primary_comparison_family": {
            "size": 2,
            "contrasts": list(PRIMARY_CONTRASTS),
            "interval": "Bonferroni familywise 95% in addition to ordinary paired 95%",
        },
        "identity": {
            "panel_sha256": sha256_file(condition.panel),
            "asset_manifest_sha256": sha256_file(condition.asset_manifest),
            "model_config_sha256": sha256_file(condition.model / "config.json"),
            "yarn_table_receipt_sha256": sha256_file(condition.yarn_table),
            "raw_sha256": {
                arm: sha256_file(
                    (condition.yarn_root / "runs/yarn" if arm == "yarn" else condition.baseline_runs / arm)
                    / "generations.jsonl"
                )
                for arm in ARMS
            },
            "same_ordered_prompts": True,
            "runtime_identity_level": _runtime_identity_level(baseline_runtime),
            "recovery_runtime": _compact_runtime(baseline_runtime),
            "baseline_wrapper_runtime": _compact_wrapper(baseline_wrapper),
        },
        "cross_model_pooling_allowed": False,
        "cross_model_pooling_note": (
            "Models retain separate benchmark families, tokenizations, length ranges and reports; "
            "no four-model aggregate score is defined."
        ),
        "excluded_models": list(EXCLUDED_MODELS),
    }
    return report


def finalize(
    condition: Condition, panel: list[dict], baseline_raw: dict[str, list[dict]],
    baseline_runtime: dict, baseline_wrapper: dict | None,
) -> Path:
    validate_yarn_table(condition.yarn_table, condition)
    yarn_run = condition.yarn_root / "runs/yarn"
    yarn = _generation_rows(yarn_run, panel, "yarn")
    _validate_yarn_recovery_contract(condition, baseline_runtime)
    raw = {**baseline_raw, "yarn": yarn}
    report = build_report(condition, panel, raw, baseline_runtime, baseline_wrapper)
    report_path = condition.yarn_root / "reports/official_static_yarn_naturalqa_triarm.json"
    atomic_json(report_path, report)
    atomic_json(condition.yarn_root / "complete.json", {
        "status": "OFFICIAL_STATIC_YARN_NATURAL_QA_COMPLETE_V1",
        "condition": condition.name,
        "method_identity": METHOD_IDENTITY,
        "rows_per_arm": len(panel),
        "arms": list(ARMS),
        "report_sha256": sha256_file(report_path),
        "cross_model_pooling_allowed": False,
        "excluded_models": list(EXCLUDED_MODELS),
    })
    return report_path


def preflight(condition: Condition) -> dict:
    panel, _ = validate_panel(condition)
    _, runtime, wrapper = validate_baselines(condition, panel)
    table_state = "VALID"
    if condition.yarn_table.is_file():
        validate_yarn_table(condition.yarn_table, condition)
    elif condition.delegated_generation:
        raise FileNotFoundError(condition.yarn_table)
    else:
        table_state = "CONSTRUCTABLE_FROM_PUBLIC_CONFIG"
    return {
        "status": "READY",
        "condition": condition.name,
        "public_model_name": condition.public_model_name,
        "method_identity": METHOD_IDENTITY,
        "family": condition.family,
        "rows": len(panel),
        "source_context_clusters": condition.expected_clusters,
        "table_state": table_state,
        "runtime": _compact_runtime(runtime),
        "runtime_identity_level": _runtime_identity_level(runtime),
        "baseline_wrapper_runtime": _compact_wrapper(wrapper),
        "cross_model_pooling_allowed": False,
        "excluded_models": list(EXCLUDED_MODELS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=ALLOWED_CONDITIONS)
    parser.add_argument(
        "--plan-root", type=Path,
        default=Path("/root/autodl-tmp/today_rope_plan_20260914"),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/bin/python"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-ready", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--finalize-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    condition = condition_from_name(args.condition, args.plan_root, model_override=args.model)
    if not (args.check_ready or args.execute or args.finalize_only):
        print(json.dumps({
            "status": "PLAN_ONLY",
            "condition": condition.name,
            "public_model_name": condition.public_model_name,
            "method_identity": METHOD_IDENTITY,
            "family": condition.family,
            "rows": condition.expected_rows,
            "delegated_generation": condition.delegated_generation,
            "next_action": (
                "use run_naturalqa_yarn.sh --execute, then --finalize-only"
                if condition.delegated_generation else
                "run --check-ready first; --execute generates only YaRN and then reports"
            ),
            "cross_model_pooling_allowed": False,
            "excluded_models": list(EXCLUDED_MODELS),
        }, indent=2, sort_keys=True))
        return
    ready = preflight(condition)
    if args.check_ready:
        print(json.dumps(ready, indent=2, sort_keys=True))
        return
    panel, _ = validate_panel(condition)
    baseline_raw, runtime, wrapper = validate_baselines(condition, panel)
    repo = Path(__file__).resolve().parents[2]
    if args.execute:
        if condition.delegated_generation:
            raise ValueError("Llama must reuse run_naturalqa_yarn.sh; use --finalize-only here")
        if not args.python.is_file():
            raise FileNotFoundError(args.python)
        with acquire_gpu_lock():
            _execute_yarn(
                condition, panel, runtime, wrapper, python=args.python, repo=repo,
            )
    report = finalize(condition, panel, baseline_raw, runtime, wrapper)
    print(json.dumps({
        "status": "COMPLETE",
        "condition": condition.name,
        "report": str(report),
        "method_identity": METHOD_IDENTITY,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
