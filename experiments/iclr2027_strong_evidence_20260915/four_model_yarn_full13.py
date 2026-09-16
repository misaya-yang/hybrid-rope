#!/usr/bin/env python3
"""Complete and report four-model Full-13 x10 TailSpline/MrPro/static-YaRN.

Default mode is plan-only.  ``--execute`` reuses the completed Llama/OLMo
200-per-task runs, resumes the registered Qwen x10 TailSpline/MrPro runs, adds
only GLM's rows 5..9 per task to its completed x5 runs, and evaluates static
YaRN on the exact x10 panels.  No model training or table search occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913.pipeline import (
    bootstrap_point_contrast,
)
from experiments.fixed_rope_three_interfaces_20260913.tables import (
    find_table,
    tensor_sha256,
    validate_table,
)
from experiments.fixed_rope_three_interfaces_20260913.matched_generation_report import (
    normalize_arm,
    summarize,
)
from experiments.iclr2027_strong_evidence_20260915.run_natural_long import (
    acquire_gpu_lock,
)


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
ARMS = ("tailspline", "mrpro", "yarn")
PLAN_STATUS = "FOUR_MODEL_YARN_FULL13_STRICT_PLAN_V1"
REPORT_STATUS = "FULL13_THREE_ARM_STRICT_PAIRED_REPORT_V1"
COMPLETE_STATUS = "FOUR_MODEL_YARN_FULL13_STRICT_COMPLETE_V1"


@dataclass(frozen=True)
class RunSource:
    run: Path
    table: Path


@dataclass(frozen=True)
class Condition:
    name: str
    model: Path
    model_id: str
    target: int
    prefill_chunk: int
    data_manifest: Path
    source_panel: Path
    panel_mode: str
    tp_tables: dict[str, Path]
    existing_tp: dict[str, tuple[RunSource, ...]]
    yarn_table: Path
    current_glm_panel: Path | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
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


def write_once_or_equal(path: Path, value: dict | list[dict]) -> None:
    if path.exists():
        existing = read_json(path) if isinstance(value, dict) else read_jsonl(path)
        if existing != value:
            raise ValueError(f"frozen artifact drift: {path}")
        return
    if isinstance(value, dict):
        atomic_json(path, value)
    else:
        atomic_jsonl(path, value)


def prompt_sha256(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError([str(path) for path in paths])


def validate_panel(rows: list[dict], *, target: int, rows_per_task: int | None = None) -> None:
    if not rows:
        raise ValueError("Full-13 panel is empty")
    seen_rows: set[str] = set()
    seen_prompts: set[str] = set()
    counts = Counter()
    for row in rows:
        row_id = str(row.get("row_id", ""))
        prompt = row.get("prompt_ids")
        prompt_hash = str(row.get("prompt_sha256", ""))
        task = str(row.get("task", ""))
        if (
            not row_id or row_id in seen_rows or task not in TASKS
            or not isinstance(prompt, list) or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)
            or prompt_hash != prompt_sha256(prompt) or prompt_hash in seen_prompts
            or int(row.get("length_cap", -1)) != target
            or int(row.get("input_tokens", len(prompt))) != len(prompt)
            or int(row.get("max_new_tokens", 0)) <= 0
            or len(prompt) + int(row["max_new_tokens"]) > target
            or not isinstance(row.get("references"), list) or not row["references"]
        ):
            raise ValueError(f"invalid Full-13 row identity: {row_id!r}")
        seen_rows.add(row_id)
        seen_prompts.add(prompt_hash)
        counts[task] += 1
    if set(counts) != set(TASKS):
        raise ValueError("panel is not Full-13")
    if rows_per_task is not None and counts != Counter({task: rows_per_task for task in TASKS}):
        raise ValueError(f"panel is not Full-13 x{rows_per_task}: {dict(counts)}")


def first_rows_per_task(rows: list[dict], count: int) -> list[dict]:
    selected = []
    counts = Counter()
    for row in rows:
        task = str(row["task"])
        if task in TASKS and counts[task] < count:
            selected.append(row)
            counts[task] += 1
    if counts != Counter({task: count for task in TASKS}):
        raise ValueError(f"source panel lacks the first {count} rows for every Full-13 task")
    return selected


def task_prompt_lists(rows: list[dict]) -> dict[str, list[str]]:
    result = {task: [] for task in TASKS}
    for row in rows:
        result[str(row["task"])].append(str(row["prompt_sha256"]))
    return result


def task_identity_lists(rows: list[dict]) -> dict[str, list[tuple]]:
    result = {task: [] for task in TASKS}
    for row in rows:
        result[str(row["task"])].append((
            str(row["row_id"]), str(row["prompt_sha256"]),
            int(row["length_cap"]), tuple(row["references"]),
        ))
    return result


def freeze_panel(condition: Condition, root: Path) -> tuple[Path, Path | None, list[dict]]:
    source = read_jsonl(condition.source_panel)
    validate_panel(source, target=condition.target)
    back_panel = None
    if condition.panel_mode == "first10":
        selected = first_rows_per_task(source, 10)
    elif condition.panel_mode == "exact10":
        validate_panel(source, target=condition.target, rows_per_task=10)
        selected = source
    elif condition.panel_mode == "glm_assets10":
        validate_panel(source, target=condition.target, rows_per_task=10)
        selected = source
        if condition.current_glm_panel is None:
            raise ValueError("GLM x10 condition lacks the current x5 panel")
        current = read_jsonl(condition.current_glm_panel)
        validate_panel(current, target=condition.target, rows_per_task=5)
        expected_front = {
            task: identities[:5] for task, identities in task_identity_lists(selected).items()
        }
        if task_identity_lists(current) != expected_front:
            raise ValueError("GLM current assets are not exactly rows 0..4 of assets10")
        back_hashes = {
            prompt for task, prompts in task_prompt_lists(selected).items() for prompt in prompts[5:]
        }
        back_rows = [row for row in selected if row["prompt_sha256"] in back_hashes]
        validate_panel(back_rows, target=condition.target, rows_per_task=5)
        back_panel = root / "panels" / condition.name / "back5" / "inputs.jsonl"
        write_once_or_equal(back_panel, back_rows)
    else:
        raise ValueError(f"unknown panel mode: {condition.panel_mode}")
    validate_panel(selected, target=condition.target, rows_per_task=10)
    panel = root / "panels" / condition.name / "inputs.jsonl"
    write_once_or_equal(panel, selected)
    manifest = {
        "status": "FOUR_MODEL_FULL13_PANEL_FROZEN_V1",
        "condition": condition.name,
        "source_panel": str(condition.source_panel),
        "source_panel_sha256": sha256(condition.source_panel),
        "panel_sha256": sha256(panel),
        "rows": 130,
        "rows_per_task": 10,
        "tasks": list(TASKS),
        "target": condition.target,
        "selection": condition.panel_mode,
        "back5_panel": str(back_panel) if back_panel else None,
        "back5_panel_sha256": sha256(back_panel) if back_panel else None,
    }
    write_once_or_equal(panel.parent / "manifest.json", manifest)
    return panel, back_panel, selected


def table_identity(path: Path) -> dict:
    receipt = read_json(path)
    if receipt.get("status") != TABLE_FORMAT:
        raise ValueError(f"table is not a frozen static receipt: {path}")
    table = find_table(receipt)
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    values, gain = validate_table(table, pairs=len(values))
    digest = tensor_sha256(values)
    if receipt.get("table_sha256_float32") != digest or float(receipt.get("gain")) != gain:
        raise ValueError(f"static table receipt self-identity drift: {path}")
    return {
        "path": str(path), "receipt_sha256": sha256(path),
        "table_sha256_float32": digest, "gain": float(gain),
        "values_float32": values,
    }


def contract_matches_table(contract: dict, table: dict) -> None:
    static = contract.get("static_table")
    if not isinstance(static, dict):
        raise ValueError("run contract lacks an explicit static table")
    values = np.asarray(static.get("values_float32"), dtype=np.float32)
    if (
        tensor_sha256(values) != table["table_sha256_float32"]
        or float(static.get("gain", float("nan"))) != table["gain"]
        or contract.get("base_arm") != "Native"
        or contract.get("unadapted") is not True
    ):
        raise ValueError("run contract table/gain differs from its frozen receipt")


def validate_run_source(source: RunSource, *, require_complete: bool = True) -> tuple[list[dict], dict, dict]:
    generations = source.run / "generations.jsonl"
    contract_path = source.run / "contract.json"
    status_path = source.run / "status.json"
    if not generations.is_file() or not contract_path.is_file():
        raise FileNotFoundError(source.run)
    rows = read_jsonl(generations)
    contract = read_json(contract_path)
    table = table_identity(source.table)
    contract_matches_table(contract, table)
    status = read_json(status_path) if status_path.is_file() else None
    if require_complete and (
        not isinstance(status, dict)
        or status.get("status") != "COMPLETE"
        or int(status.get("rows", -1)) != len(rows)
        or int(status.get("lm_rows", -1)) < 0
    ):
        raise ValueError(f"run is not complete: {source.run}")
    row_ids = contract.get("row_ids")
    if not isinstance(row_ids, list) or len(rows) > len(row_ids):
        raise ValueError(f"run contract row list is invalid: {source.run}")
    for index, row in enumerate(rows):
        if (
            row.get("eval_id") != row_ids[index]
            or row.get("arm") != contract.get("arm")
            or int(row.get("length_cap", -1)) not in contract.get("generation_length_caps", [])
        ):
            raise ValueError(f"run row/contract identity drift at {source.run}:{index}")
    return rows, contract, table


def canonical_arm_rows(
    panel: list[dict], *, logical_arm: str, sources: tuple[RunSource, ...], table_path: Path,
) -> tuple[list[dict], list[dict]]:
    target_table = table_identity(table_path)
    panel_by_prompt = {row["prompt_sha256"]: row for row in panel}
    found: dict[str, dict] = {}
    source_receipts = []
    for source in sources:
        rows, contract, source_table = validate_run_source(source)
        if (
            source_table["table_sha256_float32"] != target_table["table_sha256_float32"]
            or source_table["gain"] != target_table["gain"]
        ):
            raise ValueError(f"{logical_arm} source uses another table/gain: {source.run}")
        selected = 0
        for row in rows:
            prompt = str(row.get("prompt_sha256", ""))
            if prompt not in panel_by_prompt:
                continue
            source_row = panel_by_prompt[prompt]
            if (
                prompt in found
                or str(row.get("row_id")) != str(source_row["row_id"])
                or str(row.get("task")) != str(source_row["task"])
                or int(row.get("length_cap", -1)) != int(source_row["length_cap"])
                or row.get("references") != source_row.get("references")
                or "ruler_official_score" not in row
            ):
                raise ValueError(f"{logical_arm} row is not paired to the frozen panel: {prompt}")
            found[prompt] = row
            selected += 1
        source_receipts.append({
            "run": str(source.run),
            "generations_sha256": sha256(source.run / "generations.jsonl"),
            "contract_sha256": sha256(source.run / "contract.json"),
            "source_arm": contract["arm"], "selected_rows": selected,
        })
    if set(found) != set(panel_by_prompt):
        missing = set(panel_by_prompt) - set(found)
        raise ValueError(f"{logical_arm} is missing {len(missing)} frozen prompts")
    canonical = []
    for source_row in panel:
        row = dict(found[source_row["prompt_sha256"]])
        row["source_arm"] = row["arm"]
        row["arm"] = logical_arm
        row["table_sha256_float32"] = target_table["table_sha256_float32"]
        row["gain"] = target_table["gain"]
        canonical.append(row)
    return canonical, source_receipts


def build_report(
    *, condition: Condition, panel_path: Path, sources: dict[str, tuple[RunSource, ...]],
    out: Path, draws: int, seed: int,
) -> dict:
    panel = read_jsonl(panel_path)
    validate_panel(panel, target=condition.target, rows_per_task=10)
    merged_dir = out / "merged"
    arm_rows = {}
    receipts = {}
    for arm in ARMS:
        table_path = condition.yarn_table if arm == "yarn" else condition.tp_tables[arm]
        rows, arm_receipts = canonical_arm_rows(
            panel, logical_arm=arm, sources=sources[arm], table_path=table_path,
        )
        path = merged_dir / arm / "generations.jsonl"
        write_once_or_equal(path, rows)
        arm_rows[arm] = normalize_arm([path])
        receipts[arm] = {
            "table": {key: value for key, value in table_identity(table_path).items()
                      if key != "values_float32"},
            "sources": arm_receipts,
            "merged_generations": str(path),
            "merged_generations_sha256": sha256(path),
        }
    prompt_sets = {arm: {row["prompt_sha256"] for row in rows} for arm, rows in arm_rows.items()}
    if any(values != prompt_sets["tailspline"] for values in prompt_sets.values()):
        raise AssertionError("canonical three-arm rows are not paired")
    tasks = sorted(TASKS)
    lengths = [condition.target]
    summaries = {arm: summarize(rows, tasks=tasks, lengths=lengths) for arm, rows in arm_rows.items()}
    contrasts = {}
    for offset, baseline in enumerate(("mrpro", "yarn")):
        candidate = summaries["tailspline"]
        reference = summaries[baseline]
        contrasts[baseline] = {
            "delta_task_macro_official": (
                candidate["by_length"][str(condition.target)]["task_macro_official"]
                - reference["by_length"][str(condition.target)]["task_macro_official"]
            ),
            "delta_by_task": {
                task: (
                    candidate["by_length"][str(condition.target)]["tasks"][task]["official"]
                    - reference["by_length"][str(condition.target)]["tasks"][task]["official"]
                )
                for task in tasks
            },
            "bootstrap": bootstrap_point_contrast(
                arm_rows["tailspline"], arm_rows[baseline], tasks=tasks,
                draws=draws, seed=seed + offset,
            ),
        }
    report = {
        "status": REPORT_STATUS,
        "condition": condition.name,
        "model_id": condition.model_id,
        "excluded_models": ["qwen25_1p5b"],
        "panel": str(panel_path), "panel_sha256": sha256(panel_path),
        "tasks": tasks, "lengths": lengths, "rows_per_task": 10,
        "paired_prompts": 130,
        "arms": list(ARMS),
        "identity_checks": ["row_id", "prompt_sha256", "task", "length_cap", "table", "gain"],
        "metric_contract": "RULER official score -> task-equal endpoint macro",
        "summaries": summaries, "contrasts": contrasts,
        "receipts": receipts,
        "uncertainty_unit": "paired prompts within task; tasks fixed; no cross-model pooling",
    }
    path = out / "report.json"
    if path.exists() and read_json(path) != report:
        raise ValueError(f"strict paired report drift: {path}")
    atomic_json(path, report)
    return report


def expected_eval_ids(panel: Path, *, limit_per_cell: int = 0) -> list[str]:
    rows = read_jsonl(panel)
    if limit_per_cell:
        rows = first_rows_per_task(rows, limit_per_cell)
    return [f"extra_{panel.parent.name}:{row['row_id']}" for row in rows]


def run_complete(run: Path, *, expected_rows: int) -> bool:
    status = run / "status.json"
    if not status.is_file():
        return False
    value = read_json(status)
    if value != {"status": "COMPLETE", "rows": expected_rows, "lm_rows": 0}:
        raise ValueError(f"completed run status drift: {run}")
    return True


def evaluation_command(
    *, python: Path, condition: Condition, panel: Path, table: Path, run: Path,
    label: str, limit_per_cell: int = 0,
) -> list[str]:
    prefill = condition.prefill_chunk
    batch = 1
    longest = False
    left_pad = False
    contract_path = run / "contract.json"
    if contract_path.is_file():
        contract = read_json(contract_path)
        contract_matches_table(contract, table_identity(table))
        if (
            contract.get("generation_length_caps") != [condition.target]
            or contract.get("lm_enabled") is not False
        ):
            raise ValueError(f"partial run length drift: {run}")
        label = str(contract["arm"])
        prefill = int(contract.get("prefill_chunk_size", 0))
        batch = int(contract.get("batch_size", 1))
        limit_per_cell = int(contract.get("limit_per_cell", 0))
        left_pad = bool(contract.get("left_pad_batches", False))
        if contract.get("row_ids") != expected_eval_ids(panel, limit_per_cell=limit_per_cell):
            raise ValueError(f"partial run prompt order drift: {run}")
        longest = contract.get("generation_order") == "longest_first_shape_sorted_v1"
    command = [
        str(python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(condition.data_manifest), "--model", str(condition.model),
        "--arm", "Native", "--extra-panel", str(panel), "--only-extra-panels",
        "--skip-lm", "--length-cap", str(condition.target),
        "--prefill-chunk-size", str(prefill), "--batch-size", str(batch),
        "--static-table-json", str(table), "--table-label", label,
        "--out", str(run), "--execute",
    ]
    if limit_per_cell:
        command[command.index("--prefill-chunk-size"):command.index("--prefill-chunk-size")] = [
            "--limit-per-cell", str(limit_per_cell),
        ]
    if longest:
        command.insert(command.index("--prefill-chunk-size"), "--longest-first")
    if left_pad:
        command.insert(command.index("--prefill-chunk-size"), "--left-pad-batches")
    return command


def ensure_run(
    *, python: Path, condition: Condition, panel: Path, table: Path, run: Path,
    label: str, expected_rows: int, limit_per_cell: int = 0, repo: Path,
) -> RunSource:
    if not run_complete(run, expected_rows=expected_rows):
        command = evaluation_command(
            python=python, condition=condition, panel=panel, table=table, run=run,
            label=label, limit_per_cell=limit_per_cell,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo)
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = run.parent.parent / "logs" / f"{run.parent.name}_{run.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    source = RunSource(run=run, table=table)
    validate_run_source(source)
    return source


def conditions(plan: Path) -> list[Condition]:
    llama_clean = plan / "tailspline_llama_s4_32k_ruler200_clean"
    llama_classic = plan / "tailspline_llama_s4_classic"
    olmo_clean = plan / "tailspline_olmo_s4_16k_ruler200_clean"
    olmo_classic = plan / "tailspline_olmo_s4_classic"
    qwen = plan / "tailspline_qwen25_s4_128k_ruler10_clean"
    qwen_clean = plan / "tailspline_qwen25_s4_64k128k_clean"
    qwen_extreme = plan / "four_model_128k_extreme/qwen25_3b_128k"
    glm = plan / "glm4_9b_s4_128k"
    quick = plan / "official_yarn_quick"
    return [
        Condition(
            name="llama3_8b_s4_32k", model=Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"),
            model_id="llama3_8b", target=32768, prefill_chunk=8192,
            data_manifest=llama_classic / "assets/ppl46/manifest.json",
            source_panel=llama_clean / "assets/inputs.jsonl", panel_mode="first10",
            tp_tables={arm: llama_classic / f"tables/{arm}.json" for arm in ("tailspline", "mrpro")},
            existing_tp={arm: (RunSource(llama_clean / f"runs/{arm}", llama_classic / f"tables/{arm}.json"),)
                         for arm in ("tailspline", "mrpro")},
            yarn_table=quick / "llama3_8b_s4_32k/tables/yarn.json",
        ),
        Condition(
            name="qwen25_3b_s4_128k", model=Path("/root/autodl-tmp/rope_qwen_baseline_20260907/model"),
            model_id="qwen25_3b", target=131072, prefill_chunk=65536,
            data_manifest=qwen_extreme / "ppl/manifest.json",
            source_panel=qwen / "assets/panels/131072/inputs.jsonl", panel_mode="exact10",
            tp_tables={
                "tailspline": qwen / "tables/tailspline.json",
                "mrpro": qwen_clean / "tables/mrpro.json",
            },
            existing_tp={
                "tailspline": (RunSource(qwen / "runs/tailspline", qwen / "tables/tailspline.json"),),
                "mrpro": (RunSource(qwen / "runs/mrpro", qwen_clean / "tables/mrpro.json"),),
            },
            yarn_table=qwen_extreme / "official_yarn/tables/yarn.json",
        ),
        Condition(
            name="olmo2_1b_s4_16k", model=Path("/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct"),
            model_id="olmo2_1b", target=16384, prefill_chunk=8192,
            data_manifest=olmo_classic / "assets/ppl46/manifest.json",
            source_panel=olmo_clean / "assets/panels/16384/inputs.jsonl", panel_mode="first10",
            tp_tables={arm: olmo_clean / f"tables/{arm}.json" for arm in ("tailspline", "mrpro")},
            existing_tp={arm: (RunSource(olmo_clean / f"runs/{arm}", olmo_clean / f"tables/{arm}.json"),)
                         for arm in ("tailspline", "mrpro")},
            yarn_table=quick / "olmo2_1b_s4_16k/tables/yarn.json",
        ),
        Condition(
            name="glm4_9b_s4_128k", model=Path("/root/models/GLM-4-9B-0414"),
            model_id="glm4_9b_0414", target=131072, prefill_chunk=65536,
            data_manifest=glm / "ppl5/manifest.json",
            source_panel=glm / "assets10/panels/131072/inputs.jsonl", panel_mode="glm_assets10",
            current_glm_panel=glm / "assets/panels/131072/inputs.jsonl",
            tp_tables={arm: glm / f"tables/{arm}.json" for arm in ("tailspline", "mrpro")},
            existing_tp={arm: (RunSource(glm / f"runs/{arm}", glm / f"tables/{arm}.json"),)
                         for arm in ("tailspline", "mrpro")},
            yarn_table=quick / "glm4_9b_s4_128k/tables/yarn.json",
        ),
    ]


def build_plan(plan: Path) -> dict:
    items = []
    for condition in conditions(plan):
        actions = ["reuse_tailspline", "reuse_mrpro", "run_yarn_130", "strict_three_arm_report"]
        if condition.name.startswith("qwen25_3b"):
            actions = ["resume_tailspline_130", "run_mrpro_130", "run_yarn_130", "strict_three_arm_report"]
        elif condition.name.startswith("glm4"):
            actions = [
                "verify_current_x5_equals_assets10_front5", "run_tailspline_back5_65",
                "run_mrpro_back5_65", "merge_tp_in_assets10_order", "run_yarn_130",
                "strict_three_arm_report",
            ]
        items.append({
            "condition": condition.name, "model_id": condition.model_id,
            "target": condition.target, "panel_mode": condition.panel_mode,
            "actions": actions,
        })
    return {
        "status": PLAN_STATUS,
        "models": [item["model_id"] for item in items],
        "excluded_models": ["qwen25_1p5b"],
        "rows_per_task": 10, "tasks": list(TASKS), "conditions": items,
        "gpu_started": False,
    }


def execute(args: argparse.Namespace) -> dict:
    repo = Path(__file__).resolve().parents[2]
    root = args.plan / "official_yarn_full13"
    frozen = {}
    for condition in conditions(args.plan):
        if condition.name.startswith("qwen25_3b") and not condition.source_panel.is_file():
            condition = replace(
                condition,
                source_panel=first_existing(
                    args.plan / "tailspline_qwen25_s4_128k_ruler10_clean/assets/inputs.jsonl",
                    args.plan / "tailspline_qwen25_s4_128k_ruler10_clean/assets/panels/131072/inputs.jsonl",
                ),
            )
        for path in (
            condition.model / "config.json", condition.data_manifest,
            condition.source_panel, condition.tp_tables["tailspline"],
            condition.tp_tables["mrpro"], condition.yarn_table,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        panel, back_panel, rows = freeze_panel(condition, root)
        if not condition.name.startswith("qwen25_3b"):
            for arm in ("tailspline", "mrpro"):
                for source in condition.existing_tp[arm]:
                    validate_run_source(source)
        frozen[condition.name] = (condition, panel, back_panel, rows)

    with acquire_gpu_lock(args.gpu_lock):
        report_paths = []
        for condition, panel, back_panel, rows in frozen.values():
            condition_root = root / condition.name
            sources = {arm: condition.existing_tp[arm] for arm in ("tailspline", "mrpro")}
            if condition.name.startswith("qwen25_3b"):
                sources = {}
                for arm in ("tailspline", "mrpro"):
                    sources[arm] = (ensure_run(
                        python=args.python, condition=condition, panel=condition.source_panel,
                        table=condition.tp_tables[arm], run=condition.existing_tp[arm][0].run,
                        label=f"{condition.name}_{arm}_full13x10", expected_rows=130,
                        repo=repo,
                    ),)
            elif condition.name.startswith("glm4"):
                if back_panel is None:
                    raise AssertionError("GLM supplement panel was not frozen")
                supplemented = {}
                for arm in ("tailspline", "mrpro"):
                    supplement = ensure_run(
                        python=args.python, condition=condition, panel=back_panel,
                        table=condition.tp_tables[arm],
                        run=condition_root / f"tp_back5/{arm}",
                        label=f"{condition.name}_{arm}_back5", expected_rows=65,
                        repo=repo,
                    )
                    supplemented[arm] = (*condition.existing_tp[arm], supplement)
                sources = supplemented
            yarn = ensure_run(
                python=args.python, condition=condition, panel=condition.source_panel,
                table=condition.yarn_table, run=condition_root / "run",
                label=f"{condition.name}_official_static_yarn_full13",
                expected_rows=130,
                limit_per_cell=10 if condition.panel_mode == "first10" else 0,
                repo=repo,
            )
            sources["yarn"] = (yarn,)
            report = build_report(
                condition=condition, panel_path=panel, sources=sources,
                out=condition_root, draws=args.bootstrap_draws,
                seed=args.bootstrap_seed,
            )
            report_paths.append({
                "condition": condition.name,
                "path": str(condition_root / "report.json"),
                "sha256": sha256(condition_root / "report.json"),
                "paired_prompts": report["paired_prompts"],
            })
    complete = {
        "status": COMPLETE_STATUS,
        "models": [item.model_id for item in conditions(args.plan)],
        "excluded_models": ["qwen25_1p5b"],
        "reports": report_paths,
    }
    write_once_or_equal(root / "complete.json", complete)
    return complete


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path,
        default=Path(os.environ.get("HYBRID_ROPE_PLAN_ROOT", "/root/autodl-tmp/today_rope_plan_20260914")),
    )
    parser.add_argument(
        "--python", type=Path,
        default=Path(os.environ.get("PYTHON_BIN", "/root/miniconda3/bin/python")),
    )
    parser.add_argument(
        "--gpu-lock", type=Path,
        default=Path(os.environ.get("GPU_LOCK_PATH", "/tmp/hybrid-rope-gpu0.lock")),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 1:
        raise ValueError("bootstrap draws must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(build_plan(args.plan), sort_keys=True))
        return
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    result = execute(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
