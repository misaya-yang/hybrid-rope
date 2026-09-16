#!/usr/bin/env python3
"""Run the frozen X1/X2/X5 clean RULER matrix through existing kernels.

The default mode is read-only and prints the complete command plan.  Passing
``--execute`` freezes analytic tables, takes one non-blocking GPU lock, resumes
validated arm prefixes, and creates a matched report only from complete arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913 import tables


RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
ARM_SPECS = {
    "tailspline": ("tailspline", "candidate", ["internal_frequency_allocation"]),
    "mrpro": ("mrpro", "baseline", ["internal_frequency_allocation"]),
    "native": ("native", "native", []),
}
PLAN_STATUS = "STRONG_CLEAN_MATRIX_PLAN_V1"
LAUNCH_STATUS = "STRONG_CLEAN_MATRIX_ARM_CONTRACT_V1"
MATRIX_SOURCE_SCHEMA = "STRONG_MATRIX_SOURCE_V1"
EVALUATION_CONTRACT_PREFIX = "full13-source-order-unpadded-ruler-official"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_sha256(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _absolute_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_panels(data_root: Path, lengths: tuple[int, ...]) -> tuple[dict, list[Path]]:
    """Resolve the portable manifest emitted by ``prepare_clean_transfer``."""
    root = Path(data_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"clean root manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "COMPLETE" or not isinstance(manifest.get("panels"), dict):
        raise ValueError("clean root manifest is incomplete")
    panels = []
    for length in lengths:
        entry = manifest["panels"].get(str(length))
        if not isinstance(entry, dict) or not entry.get("inputs"):
            raise ValueError(f"clean root manifest lacks length {length}")
        path = _absolute_from(root, entry["inputs"]).resolve()
        panel_manifest = _absolute_from(root, entry.get("manifest", path.parent / "manifest.json")).resolve()
        if not path.is_file() or not panel_manifest.is_file():
            raise FileNotFoundError(f"clean panel assets are missing for length {length}")
        recorded_inputs_sha = entry.get("inputs_sha256") or entry.get("sha256")
        if recorded_inputs_sha != sha256(path):
            raise ValueError(f"clean root manifest hash drift at length {length}")
        if entry.get("manifest_sha256") != sha256(panel_manifest):
            raise ValueError(f"clean child manifest hash drift at length {length}")
        child = read_json(panel_manifest)
        if (
            child.get("status") != "COMPLETE"
            or int(child.get("length_cap", -1)) != length
            or child.get("inputs_sha256") != sha256(path)
            or child.get("model_id") != manifest.get("model_id")
            or float(child.get("scale", float("nan"))) != float(manifest.get("scale", float("nan")))
            or int(child.get("rows_per_task", -1)) != int(manifest.get("rows_per_task", -1))
            or child.get("tasks") != list(RULER_TASKS)
            or child.get("selection_mode") != "source-order"
            or child.get("content_padding") is not False
        ):
            raise ValueError(f"clean child manifest drift at length {length}")
        panels.append(path)
    return manifest, panels


def validate_panels(
    manifest: dict, paths: list[Path], *, model_id: str, scale: float,
    lengths: tuple[int, ...], rows_per_task: int,
) -> list[dict]:
    if (
        str(manifest.get("model_id")) != model_id
        or float(manifest.get("scale", float("nan"))) != scale
        or [int(value) for value in manifest.get("lengths", [])] != list(lengths)
        or int(manifest.get("rows_per_task", -1)) != rows_per_task
        or manifest.get("selection_mode") != "source-order"
        or manifest.get("content_padding") is not False
    ):
        raise ValueError("clean root manifest does not match the requested model/scale/panel contract")
    rows: list[dict] = []
    seen: set[str] = set()
    for length, path in zip(lengths, paths):
        panel = read_jsonl(path)
        expected = len(RULER_TASKS) * rows_per_task
        if len(panel) != expected:
            raise ValueError(f"length {length} has {len(panel)} rows, expected {expected}")
        counts = Counter(str(row.get("task")) for row in panel)
        if counts != Counter({task: rows_per_task for task in RULER_TASKS}):
            raise ValueError(f"length {length} is not complete RULER-13")
        task_indices: dict[str, list[int]] = {task: [] for task in RULER_TASKS}
        for row in panel:
            prompt = row.get("prompt_ids")
            prompt_hash = str(row.get("prompt_sha256", ""))
            budget = int(row.get("max_new_tokens", 0))
            task = str(row.get("task"))
            if (
                not isinstance(prompt, list) or not prompt
                or any(not isinstance(value, int) or value < 0 for value in prompt)
                or prompt_hash != prompt_sha256(prompt)
                or prompt_hash in seen
                or int(row.get("length_cap", -1)) != length
                or int(row.get("input_tokens", -1)) != len(prompt)
                or budget <= 0 or len(prompt) + budget > length
                or not row.get("references")
                or row.get("selection_mode") != "source-order"
                or int(row.get("irrelevant_padding_tokens", -1)) != 0
                or row.get("row_id") in (None, "")
            ):
                raise ValueError(f"invalid clean prompt contract in {path}: {row.get('row_id')}")
            task_indices[task].append(int(row.get("source_order_index", -1)))
            seen.add(prompt_hash)
        if any(indices != list(range(rows_per_task)) for indices in task_indices.values()):
            raise ValueError(f"length {length} is not source-order within every task")
        rows.extend(panel)
    return rows


def candidate_id(model_id: str, scale: float, arm: str) -> str:
    scale_label = format(scale, "g").replace(".", "p")
    return f"strong_{model_id}_s{scale_label}_{arm}"


def table_command(args: argparse.Namespace, arm: str, table_path: Path) -> list[str]:
    method, role, changes = ARM_SPECS[arm]
    command = [
        str(args.python), "-m", "experiments.fixed_rope_three_interfaces_20260913.tables",
        "analytic", "--config", str(args.model / "config.json"), "--method", method,
        "--scale", format(args.scale, ".17g"), "--candidate-id", candidate_id(args.model_id, args.scale, arm),
        "--model-id", args.model_id, "--role", role, "--out", str(table_path),
    ]
    for value in changes:
        command.extend(("--changed-variable", value))
    return command


def expected_table(args: argparse.Namespace, arm: str) -> tuple[Any, float, dict]:
    config = read_json(args.model / "config.json")
    method, _, _ = ARM_SPECS[arm]
    return tables.build_analytic(
        config, method=method, scale=args.scale, low=None, high=None, depth=1.0, gain=None,
    )


def validate_table(args: argparse.Namespace, arm: str, path: Path) -> dict:
    receipt = read_json(path)
    method, role, changes = ARM_SPECS[arm]
    values, gain, _ = expected_table(args, arm)
    actual, actual_gain = tables.validate_table(
        tables.find_table(receipt), pairs=len(values),
    )
    geometry = tables.model_geometry(read_json(args.model / "config.json"))
    if (
        receipt.get("status") != TABLE_FORMAT
        or receipt.get("candidate_id") != candidate_id(args.model_id, args.scale, arm)
        or receipt.get("model_id") != args.model_id
        or receipt.get("role") != role
        or float(receipt.get("scale", float("nan"))) != args.scale
        or receipt.get("model_geometry") != geometry
        or receipt.get("changed_variables") != changes
        or receipt.get("source") != f"analytic:{method}"
        or receipt.get("table_sha256_float32") != tables.tensor_sha256(actual)
        or not __import__("numpy").array_equal(actual, values)
        or actual_gain != gain
    ):
        raise ValueError(f"frozen analytic table drift: {path}")
    return receipt


def eval_id(panel_path: Path, row: dict) -> str:
    return f"extra_{panel_path.parent.name}:{row['row_id']}"


def expected_eval_rows(panel_paths: list[Path]) -> list[tuple[Path, dict]]:
    return [(path, row) for path in panel_paths for row in read_jsonl(path)]


def launch_contract(
    args: argparse.Namespace, arm: str, table_path: Path,
    panel_paths: list[Path], rows: list[dict],
) -> dict:
    return {
        "status": LAUNCH_STATUS,
        "arm": arm,
        "model": str(args.model),
        "model_id": args.model_id,
        "model_config_sha256": sha256(args.model / "config.json"),
        "data_root": str(args.data_root),
        "data_manifest": str(args.data_manifest),
        "data_manifest_sha256": sha256(args.data_manifest),
        "scale": args.scale,
        "lengths": list(args.lengths),
        "execution_length_order": [int(path.parent.name) for path in panel_paths],
        "rows_per_task": args.rows_per_task,
        "rows": len(rows),
        "panel_files": [
            {"path": str(path), "sha256": sha256(path)} for path in panel_paths
        ],
        "prompt_sha256": [row["prompt_sha256"] for row in rows],
        "table": str(table_path),
        "table_receipt_sha256": sha256(table_path),
        "table_sha256_float32": read_json(table_path)["table_sha256_float32"],
        "python": str(args.python),
        "batch_size": args.batch_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "lm_enabled": False,
    }


def evaluation_command(
    args: argparse.Namespace, arm: str, table_path: Path,
    panel_paths: list[Path], run_dir: Path,
) -> list[str]:
    label = candidate_id(args.model_id, args.scale, arm)
    command = [
        str(args.python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(args.data_manifest), "--model", str(args.model), "--arm", "Native",
    ]
    for panel in panel_paths:
        command.extend(("--extra-panel", str(panel)))
    command.extend(("--only-extra-panels", "--skip-lm"))
    for length in args.lengths:
        command.extend(("--length-cap", str(length)))
    command.extend((
        "--prefill-chunk-size", str(args.prefill_chunk_size),
        "--batch-size", str(args.batch_size), "--static-table-json", str(table_path),
        "--table-label", label, "--out", str(run_dir), "--execute",
    ))
    return command


def validate_kernel_contract(
    args: argparse.Namespace, arm: str, receipt: dict,
    panel_paths: list[Path], path: Path,
) -> dict:
    contract = read_json(path)
    expected_ids = [eval_id(panel, row) for panel, row in expected_eval_rows(panel_paths)]
    if (
        contract.get("arm") != candidate_id(args.model_id, args.scale, arm)
        or contract.get("base_arm") != "Native"
        or contract.get("unadapted") is not True
        or contract.get("row_ids") != expected_ids
        or contract.get("generation_length_caps") != sorted(args.lengths)
        or contract.get("lm_enabled") is not False
        or int(contract.get("batch_size", -1)) != args.batch_size
        or int(contract.get("prefill_chunk_size", -1)) != args.prefill_chunk_size
        or contract.get("static_table") != receipt["table"]
    ):
        raise ValueError(f"evaluation kernel contract drift: {path}")
    return contract


def validate_generation_prefix(
    args: argparse.Namespace, arm: str, panel_paths: list[Path], path: Path,
    *, require_complete: bool,
) -> list[dict]:
    saved = read_jsonl(path) if path.is_file() else []
    expected = expected_eval_rows(panel_paths)
    if len(saved) > len(expected) or (require_complete and len(saved) != len(expected)):
        raise ValueError(f"generation row count drift: {path}")
    label = candidate_id(args.model_id, args.scale, arm)
    for index, row in enumerate(saved):
        panel, source = expected[index]
        fields = {
            "eval_id": eval_id(panel, source), "row_id": source["row_id"],
            "task": source["task"], "length_cap": source["length_cap"],
            "prompt_sha256": source["prompt_sha256"], "references": source["references"],
            "arm": label,
        }
        if any(row.get(key) != value for key, value in fields.items()):
            raise ValueError(f"saved generation is not the expected prompt prefix at row {index}")
    return saved


def arm_is_complete(
    args: argparse.Namespace, arm: str, table_path: Path,
    panel_paths: list[Path], rows: list[dict], run_dir: Path,
) -> bool:
    receipt = validate_table(args, arm, table_path)
    expected_launch = launch_contract(args, arm, table_path, panel_paths, rows)
    launch_path = run_dir / "launch_contract.json"
    kernel_path = run_dir / "contract.json"
    status_path = run_dir / "status.json"
    if not launch_path.exists() and not kernel_path.exists() and not status_path.exists() and not (run_dir / "generations.jsonl").exists():
        return False
    if not launch_path.is_file() or read_json(launch_path) != expected_launch:
        raise ValueError(f"arm launch contract drift: {run_dir}")
    if not kernel_path.is_file():
        if (run_dir / "generations.jsonl").exists() or status_path.exists():
            raise ValueError(f"arm has outputs without a kernel contract: {run_dir}")
        return False
    kernel = validate_kernel_contract(args, arm, receipt, panel_paths, kernel_path)
    status = read_json(status_path) if status_path.is_file() else None
    if status is not None and status != {"status": "COMPLETE", "rows": len(rows), "lm_rows": 0}:
        raise ValueError(f"invalid arm status: {run_dir}")
    saved = validate_generation_prefix(
        args, arm, panel_paths, run_dir / "generations.jsonl",
        require_complete=status is not None,
    )
    if status is None:
        return False
    summary = read_json(run_dir / "summary.json")
    if summary.get("status") != "COMPLETE" or summary.get("identity") != kernel or len(saved) != len(rows):
        raise ValueError(f"incomplete or mismatched arm summary: {run_dir}")
    return True


def report_path(args: argparse.Namespace) -> Path:
    baselines = "mrpro_native" if args.include_native else "mrpro"
    lengths = "_".join(str(value) for value in args.lengths)
    return args.out / "reports" / f"tailspline_vs_{baselines}_{lengths}.json"


def report_command(args: argparse.Namespace, complete: list[str], path: Path) -> list[str] | None:
    if not {"tailspline", "mrpro"}.issubset(complete):
        return None
    baselines = ["mrpro"] + (["native"] if "native" in complete else [])
    command = [
        str(args.python), "-m",
        "experiments.fixed_rope_three_interfaces_20260913.matched_generation_report",
    ]
    for arm in ["tailspline", *baselines]:
        command.extend(("--source", f"{arm}={args.out / 'runs' / arm / 'generations.jsonl'}"))
    command.extend(("--candidate", "tailspline"))
    for arm in baselines:
        command.extend(("--baseline", arm))
    for length in args.lengths:
        command.extend(("--length", str(length)))
    command.extend(("--out", str(path)))
    return command


def validate_report(args: argparse.Namespace, complete: list[str], path: Path, row_count: int) -> None:
    report = read_json(path)
    baselines = ["mrpro"] + (["native"] if "native" in complete else [])
    expected_arms = ["tailspline", *baselines]
    expected_sources = {
        arm: [str(args.out / "runs" / arm / "generations.jsonl")] for arm in expected_arms
    }
    if (
        report.get("status") != "MATCHED_GENERATION_RANGE_REPORT_V1"
        or report.get("candidate") != "tailspline"
        or report.get("baselines") != baselines
        or report.get("lengths") != sorted(args.lengths)
        or int(report.get("paired_prompts", -1)) != row_count
        or report.get("tasks") != sorted(RULER_TASKS)
        or report.get("source_files") != expected_sources
        or set(report.get("summaries", {})) != set(expected_arms)
        or set(report.get("contrasts", {})) != set(baselines)
        or any(int(report["summaries"][arm].get("rows", -1)) != row_count for arm in expected_arms)
        or any(report["summaries"][arm].get("rows_per_task_length") != [args.rows_per_task]
               for arm in expected_arms)
    ):
        raise ValueError(f"matched report contract drift: {path}")


def matrix_source_payload(
    args: argparse.Namespace, manifest: dict, complete: list[str], report: dict,
) -> dict:
    """Adapt one validated matched report to the strict matrix-source schema."""
    scale = int(args.scale)
    native_length = int(manifest["model_identity"]["native_length"])
    required_arms = ["tailspline", "mrpro"] + (["native"] if "native" in complete else [])
    required_contrasts = [
        {"candidate": "tailspline", "baseline": baseline}
        for baseline in required_arms if baseline != "tailspline"
    ]
    cells = []
    for length in args.lengths:
        length_key = str(length)
        arms = {}
        for arm in required_arms:
            summary = report["summaries"][arm]["by_length"].get(length_key)
            if not isinstance(summary, dict) or set(summary.get("tasks", {})) != set(RULER_TASKS):
                raise ValueError(f"matched report lacks complete RULER-13 at {length}/{arm}")
            task_rows = [int(summary["tasks"][task].get("rows", -1)) for task in RULER_TASKS]
            if task_rows != [args.rows_per_task] * len(RULER_TASKS):
                raise ValueError(f"matched report has incomplete task rows at {length}/{arm}")
            arms[arm] = {
                "status": "complete", "rows": sum(task_rows),
                "score": float(summary["task_macro_official"]),
            }
        contrasts = []
        for baseline in required_arms[1:]:
            source = report["contrasts"][baseline]
            delta = float(source["delta_by_length"][length_key])
            expected_delta = arms["tailspline"]["score"] - arms[baseline]["score"]
            if abs(delta - expected_delta) > 1e-10:
                raise ValueError(f"matched report contrast disagrees with scores at {length}/{baseline}")
            ci95 = source.get("bootstrap", {}).get("delta_by_length_interval95", {}).get(length_key)
            if not isinstance(ci95, list) or len(ci95) != 2:
                raise ValueError(f"matched report lacks paired interval at {length}/{baseline}")
            contrasts.append({
                "candidate": "tailspline", "baseline": baseline, "delta": delta,
                "ci95": [float(ci95[0]), float(ci95[1])],
                "uncertainty_unit": "paired_prompts",
            })
        cells.append({
            "length_tokens": length,
            "length_multiple": length / native_length,
            "expected_rows_per_arm": len(RULER_TASKS) * args.rows_per_task,
            "paired_rows": len(RULER_TASKS) * args.rows_per_task,
            "arms": arms,
            "contrasts": contrasts,
        })
    length_label = "_".join(f"{length / native_length:g}l" for length in args.lengths)
    payload = {
        "schema": MATRIX_SOURCE_SCHEMA,
        "status": "complete",
        "experiment_id": (
            f"strong_{args.model_id}_s{scale}_clean_ruler_{length_label}_r{args.rows_per_task}"
        ),
        "identity": {
            "model_id": args.model_id,
            "benchmark_family": "ruler",
            "data_contract": "clean",
            "scale": scale,
            "native_length_tokens": native_length,
            "evaluation_contract": f"{EVALUATION_CONTRACT_PREFIX}-batch{args.batch_size}-v1",
            "metric": {"name": "task_macro_official", "direction": "higher", "unit": "fraction"},
            "expected_length_multiples": [length / native_length for length in args.lengths],
            "required_arms": required_arms,
            "required_contrasts": required_contrasts,
        },
        "cells": cells,
    }
    from experiments.iclr2027_strong_evidence_20260915.summarize_matrix import normalize_source
    normalize_source(payload, source_name="matrix_source.json")
    return payload


def write_or_validate_matrix_source(path: Path, payload: dict) -> None:
    from experiments.iclr2027_strong_evidence_20260915.summarize_matrix import normalize_source
    if path.exists():
        existing = read_json(path)
        normalize_source(existing, source_name=path.name)
        if existing != payload:
            raise ValueError(f"existing matrix source belongs to a different complete report: {path}")
        return
    atomic_json(path, payload)
    normalize_source(read_json(path), source_name=path.name)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


class GpuLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise RuntimeError(f"another clean matrix owns the GPU lock: {self.path}") from error
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()}\n")
        self.stream.flush()
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def build_plan(args: argparse.Namespace) -> tuple[dict, list[Path], list[dict]]:
    if len(args.lengths) != len(set(args.lengths)) or any(value <= 0 for value in args.lengths):
        raise ValueError("--lengths must contain unique positive values")
    args.lengths = tuple(args.lengths)
    if (
        args.rows_per_task <= 0 or args.scale <= 1 or not float(args.scale).is_integer()
        or int(args.scale) not in {2, 4, 16} or args.batch_size not in {1, 2}
        or args.prefill_chunk_size < 0 or not _SLUG.fullmatch(args.model_id)
    ):
        raise ValueError("rows/scale/prefill are invalid; clean confirmation supports batch-size 1 or 2")
    if args.batch_size > 1 and args.prefill_chunk_size != 0:
        raise ValueError("exact-length batch=2 is supported only with direct prefill")
    for path in (args.model / "config.json", args.data_manifest, args.python):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest, panels = resolve_panels(args.data_root, args.lengths)
    if manifest.get("model_identity", {}).get("config_sha256") != sha256(args.model / "config.json"):
        raise ValueError("clean root manifest belongs to a different checkpoint config")
    geometry = tables.model_geometry(read_json(args.model / "config.json"))
    if int(manifest.get("model_identity", {}).get("native_length", -1)) != geometry["native_length"]:
        raise ValueError("clean root manifest has a different checkpoint Native length")
    rows = validate_panels(
        manifest, panels, model_id=args.model_id, scale=args.scale,
        lengths=args.lengths, rows_per_task=args.rows_per_task,
    )
    if args.longest_first:
        panels = list(reversed(panels))
        rows = [row for panel in panels for row in read_jsonl(panel)]
    arms = ["tailspline", "mrpro"] + (["native"] if args.include_native else [])
    table_paths = {arm: args.out / "tables" / f"{arm}.json" for arm in arms}
    eval_commands = {
        arm: evaluation_command(args, arm, table_paths[arm], panels, args.out / "runs" / arm)
        for arm in arms
    }
    planned_report = report_command(args, arms, report_path(args))
    plan = {
        "status": PLAN_STATUS, "execute": bool(args.execute),
        "model": str(args.model), "model_id": args.model_id, "data_root": str(args.data_root),
        "data_manifest": str(args.data_manifest), "out": str(args.out), "scale": args.scale,
        "lengths": list(args.lengths), "rows_per_task": args.rows_per_task,
        "rows": len(rows), "arms": arms, "batch_size": args.batch_size,
        "execution_length_order": [int(path.parent.name) for path in panels],
        "prefill_chunk_size": args.prefill_chunk_size, "gpu_lock": str(args.gpu_lock),
        "tables": {arm: table_command(args, arm, table_paths[arm]) for arm in arms},
        "evaluations": eval_commands, "matched_report": planned_report,
        "matrix_source": str(args.out / "reports" / "matrix_source.json"),
    }
    return plan, panels, rows


def execute(args: argparse.Namespace, plan: dict, panels: list[Path], rows: list[dict]) -> dict:
    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.out.mkdir(parents=True, exist_ok=True)
    with GpuLock(args.gpu_lock):
        complete = []
        for arm in plan["arms"]:
            table_path = args.out / "tables" / f"{arm}.json"
            if table_path.exists():
                validate_table(args, arm, table_path)
            else:
                _run(plan["tables"][arm], cwd=repo, env=environment)
                validate_table(args, arm, table_path)
            run_dir = args.out / "runs" / arm
            expected_launch = launch_contract(args, arm, table_path, panels, rows)
            launch_path = run_dir / "launch_contract.json"
            if launch_path.exists() and read_json(launch_path) != expected_launch:
                raise ValueError(f"arm launch contract drift: {run_dir}")
            if not launch_path.exists():
                atomic_json(launch_path, expected_launch)
            if arm_is_complete(args, arm, table_path, panels, rows, run_dir):
                complete.append(arm)
                continue
            _run(plan["evaluations"][arm], cwd=repo, env=environment)
            if not arm_is_complete(args, arm, table_path, panels, rows, run_dir):
                raise RuntimeError(f"evaluation returned without a complete arm: {arm}")
            complete.append(arm)
        output_report = report_path(args)
        command = report_command(args, complete, output_report)
        if command is None:
            raise RuntimeError("matched report requires complete TailSpline and MrPro arms")
        if output_report.exists():
            validate_report(args, complete, output_report, len(rows))
        else:
            _run(command, cwd=repo, env=environment)
            validate_report(args, complete, output_report, len(rows))
        manifest = read_json(args.data_root / "manifest.json")
        matrix_source = args.out / "reports" / "matrix_source.json"
        payload = matrix_source_payload(args, manifest, complete, read_json(output_report))
        write_or_validate_matrix_source(matrix_source, payload)
    return {
        "status": "COMPLETE", "arms": complete, "rows_per_arm": len(rows),
        "report": str(output_report), "matrix_source": str(matrix_source),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--lengths", nargs="+", required=True, help="space- or comma-separated token caps")
    parser.add_argument("--rows-per-task", type=int, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--longest-first", action="store_true",
        help="execute the largest frozen panel first without changing report aggregation",
    )
    parser.add_argument("--gpu-lock", type=Path, help="global lock shared by all jobs targeting this GPU")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.lengths = [
            int(value) for item in args.lengths for value in str(item).split(",") if value.strip()
        ]
    except ValueError as error:
        raise ValueError("--lengths must contain integers") from error
    args.model = args.model.resolve()
    args.data_root = args.data_root.resolve()
    args.out = args.out.resolve()
    args.data_manifest = args.data_manifest.resolve()
    args.python = args.python.resolve()
    if args.gpu_lock is None:
        args.gpu_lock = Path("/tmp/hybrid-rope-gpu0.lock")
    else:
        args.gpu_lock = args.gpu_lock.resolve()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan, panels, rows = build_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.execute:
        print(json.dumps(execute(args, plan, panels, rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
