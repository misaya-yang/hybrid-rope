#!/usr/bin/env python3
"""Plan and gate the corrected E0-E5 fixed-table pipeline.

The default is read-only plan generation.  Stages that load a model or use a
GPU are run only when ``--execute`` is present.  Existing implementation under
``experiments.olmo_recovery_20260912`` remains the single source of truth for
fit, generation, and scoring; this module audits identities and composes those
commands instead of duplicating them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np


BASELINE_ARMS = ("Native", "BM_g4", "MrPro_g4", "C42V24_g4")
SPLIT_ROWS = {"fit": (168, 128), "select": (84, 64), "internal_confirm": (84, 64)}
LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _artifact_error(reason: str, **details) -> dict:
    return {
        "status": "ENGINEERING_ERROR",
        "failure_class": "ENGINEERING_ERROR",
        "reason": reason,
        **details,
    }


def audit_prepared_inputs(data_dir: Path, source_cf_dir: Path, baselines_dir: Path) -> dict:
    """Verify only execution-critical identities; do not hash trusted assets."""
    try:
        data_manifest = read_json(data_dir / "manifest.json")
        data_rows = read_jsonl(data_dir / "rows.jsonl")
        source_manifest = read_json(source_cf_dir / "manifest.json")
        source_rows = read_jsonl(source_cf_dir / "rows.jsonl")
        baseline_status = read_json(baselines_dir / "status.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return _artifact_error(str(error))
    if data_manifest.get("status") != "RANGE_SOLVER_DATA_READY":
        return _artifact_error("main range data manifest is not ready")
    if source_manifest.get("status") != "RANGE_SOURCE_COUNTERFACTUAL_READY":
        return _artifact_error("source-counterfactual manifest is not ready")
    data_counts = Counter(row.get("split") for row in data_rows)
    source_counts = Counter(row.get("split") for row in source_rows)
    expected_data = {"fit": 168, "select": 84, "internal_confirm": 84}
    expected_source = {"fit": 128, "select": 64, "internal_confirm": 64}
    if data_counts != expected_data or source_counts != expected_source:
        return _artifact_error(
            "prepared split counts differ from the frozen 8/4/4 contract",
            data_counts=dict(data_counts),
            source_counts=dict(source_counts),
        )
    if len({row.get("row_id") for row in data_rows}) != len(data_rows):
        return _artifact_error("main data row IDs are not unique")
    if len({row.get("row_id") for row in source_rows}) != len(source_rows):
        return _artifact_error("source-counterfactual row IDs are not unique")
    if any(not row.get("target_includes_eos") for row in data_rows):
        return _artifact_error("main task target omits terminal EOS")
    if baseline_status.get("status") != "COMPLETE" or set(baseline_status.get("arms", ())) != set(BASELINE_ARMS):
        return _artifact_error("fit baselines are incomplete or have the wrong arm set")
    expected_fit_ids = [row["row_id"] for row in data_rows if row["split"] == "fit"]
    expected_source_ids = [row["row_id"] for row in source_rows if row["split"] == "fit"]
    for arm in BASELINE_ARMS:
        try:
            task_records = read_jsonl(baselines_dir / f"{arm}.jsonl")
            source_records = read_jsonl(baselines_dir / f"source_cf_{arm}.jsonl")
        except (FileNotFoundError, json.JSONDecodeError) as error:
            return _artifact_error(str(error), arm=arm)
        if [row.get("row_id") for row in task_records] != expected_fit_ids:
            return _artifact_error("baseline task rows do not match the prepared fit split", arm=arm)
        if [row.get("row_id") for row in source_records] != expected_source_ids:
            return _artifact_error("baseline source rows do not match the prepared fit split", arm=arm)
    return {
        "status": "READY",
        "failure_class": None,
        "main_rows": len(data_rows),
        "source_rows": len(source_rows),
        "split_counts": {split: {"main": data_counts[split], "source_cf": source_counts[split]} for split in SPLIT_ROWS},
        "baselines": list(BASELINE_ARMS),
        "target_contract": "canonical answer plus terminal EOS for fit proxy; free generation remains a separate endpoint",
        "asset_identity_policy": "reuse supplied clone; no repeated SHA gate",
    }


def classify_solver_run(run_dir: Path) -> dict:
    """Classify a solver directory without turning COMPLETE into task success."""
    result_path = run_dir / "result.json"
    status_path = run_dir / "status.json"
    if not result_path.exists():
        if run_dir.exists():
            return {
                "status": "RUNNING_OR_PENDING_RECEIPT",
                "failure_class": None,
                "run_dir": str(run_dir),
                "task_evidence": False,
            }
        return {
            "status": "NOT_STARTED",
            "failure_class": None,
            "run_dir": str(run_dir),
            "task_evidence": False,
        }
    try:
        result = read_json(result_path)
        status = read_json(status_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return _artifact_error(str(error), run_dir=str(run_dir), task_evidence=False)
    if result.get("status") != "COMPLETE" or status.get("status") != "COMPLETE":
        return _artifact_error("solver receipt is internally incomplete", run_dir=str(run_dir), task_evidence=False)
    table = result.get("table", {})
    allocation = result.get("allocation", {})
    validation = result.get("full_fit_validation", {})
    try:
        frequencies = np.asarray(table["values_float32"], dtype=np.float32)
        gain = float(table["gain"])
    except (KeyError, TypeError, ValueError) as error:
        return _artifact_error(f"invalid solver table: {error}", run_dir=str(run_dir), task_evidence=False)
    if (
        frequencies.shape != (64,)
        or not np.isfinite(frequencies).all()
        or not np.all(frequencies[:-1] > frequencies[1:])
        or not np.isfinite(gain)
        or gain <= 0.0
    ):
        return _artifact_error("solver table is non-finite, unordered, or has the wrong K", run_dir=str(run_dir), task_evidence=False)
    if not allocation.get("same_table_all_layers_and_lengths") or allocation.get("model_weight_updates") != 0:
        return _artifact_error("solver receipt violates the one-static-table frozen-model contract", run_dir=str(run_dir), task_evidence=False)
    accepted_steps = int(result.get("accepted_steps", -1))
    hard_constraints = validation.get("all_hard_constraints_hold") is True
    common = {
        "run_dir": str(run_dir),
        "result_json": str(result_path),
        "initial_arm": result.get("initial_arm"),
        "accepted_steps": accepted_steps,
        "all_fit_hard_constraints_hold": hard_constraints,
        "fit_range_objective": validation.get("range_objective"),
        "fit_feasibility_objective": validation.get("feasibility_objective"),
        "task_evidence": False,
        "evidence_tier": "teacher-forced fit proxy only",
    }
    if accepted_steps <= 0:
        return {
            "status": "TERMINAL_NO_NEW_CANDIDATE",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "reason": "solver terminated without an accepted table step; the initial arm is not a new candidate",
            **common,
        }
    if not hard_constraints:
        return {
            "status": "PARETO_FIT_CANDIDATE_READY_FOR_GENERATION",
            "failure_class": None,
            "reason": "accepted fit steps exist but the declared full-fit constraints do not all hold; retain as a Pareto candidate and let held-out generation expose the trade-off",
            **common,
        }
    return {
        "status": "FIT_CANDIDATE_READY_FOR_GENERATION",
        "failure_class": None,
        "reason": "candidate passed its fit receipt; free generation is still required",
        **common,
    }


def parse_solver_run(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("solver run must be LABEL=INITIAL_ARM=RUN_DIRECTORY")
    label, initial_arm, raw_path = parts
    if not LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError("solver label may contain only letters, digits, dot, underscore, or dash")
    if initial_arm not in ("BM_g4", "C42V24_g4", "BetaSym_gamma3_g4"):
        raise argparse.ArgumentTypeError("unsupported solver initial arm")
    return label, initial_arm, Path(raw_path)


def _generation_command(
    *,
    model: Path,
    empty_manifest: Path,
    data_dir: Path,
    source_cf_dir: Path,
    split: str | None,
    output: Path,
    arm: str,
    static_result: Path | None = None,
    table_label: str | None = None,
    extra_panels: Iterable[Path] = (),
    prefill_chunk_size: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data",
        str(empty_manifest),
        "--model",
        str(model),
        "--arm",
        arm,
        "--split",
        "dev",
        "--only-extra-panels",
        "--skip-lm",
        "--prefill-chunk-size",
        str(prefill_chunk_size),
    ]
    panels = list(extra_panels) or [data_dir / "rows.jsonl", source_cf_dir / "rows.jsonl"]
    for panel in panels:
        command.extend(("--extra-panel", str(panel)))
    if split:
        command.extend(("--row-split", split))
    if static_result is not None:
        command.extend(("--static-table-json", str(static_result), "--table-label", str(table_label)))
    command.extend(("--out", str(output), "--execute"))
    return command


def _complete_output(path: Path) -> bool:
    try:
        return read_json(path / "status.json").get("status") == "COMPLETE"
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False


def generation_stage_plan(
    *,
    stage: str,
    split: str,
    model: Path,
    data_dir: Path,
    source_cf_dir: Path,
    run_root: Path,
    candidates: dict[str, dict],
    selected_label: str | None = None,
    prefill_chunk_size: int = 0,
) -> dict:
    eligible = {
        label: record
        for label, record in candidates.items()
        if record["status"] in (
            "FIT_CANDIDATE_READY_FOR_GENERATION",
            "PARETO_FIT_CANDIDATE_READY_FOR_GENERATION",
        )
    }
    if selected_label is not None:
        eligible = {selected_label: eligible[selected_label]} if selected_label in eligible else {}
    if not eligible:
        return {
            "stage": stage,
            "status": "BLOCKED_NO_FIT_CANDIDATE",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "commands": [],
            "reason": "no solver receipt currently qualifies for task-level generation",
        }
    stage_root = run_root / {"e2": "select_r0", "e3": "internal_confirm_r0"}.get(stage, stage)
    empty_manifest = run_root / "empty_data_manifest.json"
    outputs: dict[str, Path] = {}
    commands: list[list[str]] = []
    for arm in BASELINE_ARMS:
        output = stage_root / arm
        outputs[arm] = output
        if not _complete_output(output):
            commands.append(_generation_command(
                model=model,
                empty_manifest=empty_manifest,
                data_dir=data_dir,
                source_cf_dir=source_cf_dir,
                split=split,
                output=output,
                arm=arm,
                prefill_chunk_size=prefill_chunk_size,
            ))
    for label, record in eligible.items():
        output = stage_root / label
        outputs[label] = output
        if not _complete_output(output):
            commands.append(_generation_command(
                model=model,
                empty_manifest=empty_manifest,
                data_dir=data_dir,
                source_cf_dir=source_cf_dir,
                split=split,
                output=output,
                arm=record["initial_arm"],
                static_result=Path(record["result_json"]),
                table_label=label,
                prefill_chunk_size=prefill_chunk_size,
            ))
    summary = stage_root / "summary.json"
    summary_command = [
        sys.executable,
        "-m",
        "experiments.olmo_recovery_20260912.summarize_range_generation",
        "--data",
        str(data_dir),
        "--source-cf",
        str(source_cf_dir),
        "--split",
        split,
    ]
    for label, output in outputs.items():
        summary_command.extend(("--arm", f"{label}={output}"))
    summary_command.extend(("--out", str(summary)))
    if commands or not summary.exists():
        commands.append(summary_command)
    return {
        "stage": stage,
        "status": "COMPLETE" if not commands else "READY_GPU",
        "failure_class": None,
        "split": split,
        "candidate_labels": list(eligible),
        "baseline_policy": "each baseline generated once on this split and reused in the summary",
        "fixed_table_policy": "each candidate uses one static solver tensor at all lengths and layers",
        "metrics_policy": "official score, exact-plus-EOS, source pair-follow, and proxy NLL remain separate",
        "commands": commands,
        "summary": str(summary),
    }


def _selection_from_lock(path: Path | None, candidates: dict[str, dict]) -> tuple[str | None, dict]:
    if path is None or not path.exists():
        return None, {
            "status": "BLOCKED_SELECTION_NOT_LOCKED",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "reason": "E3 requires an immutable E2 selection receipt; the pipeline does not auto-filter Pareto candidates",
        }
    try:
        receipt = read_json(path)
    except (json.JSONDecodeError, ValueError) as error:
        return None, _artifact_error(str(error))
    label = receipt.get("selected_candidate")
    if receipt.get("status") != "SELECTION_LOCKED" or label not in candidates:
        return None, _artifact_error("selection receipt is incomplete or names an unknown solver candidate")
    return str(label), receipt


def e4_plan(
    *,
    model: Path,
    data_dir: Path,
    source_cf_dir: Path,
    run_root: Path,
    candidates: dict[str, dict],
    selected_label: str | None,
    panels: list[Path],
    baseline_refs: list[Path],
    prefill_chunk_size: int,
) -> dict:
    if selected_label is None or candidates.get(selected_label, {}).get("status") not in (
        "FIT_CANDIDATE_READY_FOR_GENERATION",
        "PARETO_FIT_CANDIDATE_READY_FOR_GENERATION",
    ):
        return {"stage": "e4", "status": "BLOCKED_SELECTION_NOT_LOCKED", "failure_class": "SCIENTIFIC_UNRESOLVED", "commands": []}
    if not panels or not baseline_refs:
        return {
            "stage": "e4",
            "status": "BLOCKED_INPUT_IDENTITY",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "commands": [],
            "reason": "broad validation requires explicit archived panels and matching baseline result references; missing cells stay untested",
        }
    missing = [str(path) for path in (*panels, *baseline_refs) if not path.exists()]
    if missing:
        return _artifact_error("E4 input references do not exist", stage="e4", missing=missing, commands=[])
    record = candidates[selected_label]
    output = run_root / "broad_r0" / selected_label
    command = _generation_command(
        model=model,
        empty_manifest=run_root / "empty_data_manifest.json",
        data_dir=data_dir,
        source_cf_dir=source_cf_dir,
        split=None,
        output=output,
        arm=record["initial_arm"],
        static_result=Path(record["result_json"]),
        table_label=selected_label,
        extra_panels=panels,
        prefill_chunk_size=prefill_chunk_size,
    )
    return {
        "stage": "e4",
        "status": "COMPLETE" if _complete_output(output) else "READY_GPU",
        "failure_class": None,
        "commands": [] if _complete_output(output) else [command],
        "candidate": selected_label,
        "baseline_refs": [str(path) for path in baseline_refs],
        "scope": "candidate-only broad generation against explicitly supplied archived baselines; no cross-panel averaging",
    }


def e5_plan(lineage_path: Path | None, selected_label: str | None) -> dict:
    if lineage_path is None or not lineage_path.exists():
        return {
            "stage": "e5",
            "status": "BLOCKED_MATCHED_LINEAGE",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "commands": [],
            "reason": "no verified common pre-intervention parent and optimizer lineage is supplied; no pseudo-matched LoRA command is generated",
        }
    try:
        lineage = read_json(lineage_path)
    except (json.JSONDecodeError, ValueError) as error:
        return _artifact_error(str(error), stage="e5", commands=[])
    required = {
        "selected_candidate": selected_label,
        "rank": 32,
        "alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "same_pre_intervention_parent": True,
        "baseline_optimizer_state_available": True,
        "candidate_optimizer_state_available": True,
    }
    mismatches = {key: {"expected": value, "actual": lineage.get(key)} for key, value in required.items() if lineage.get(key) != value}
    if mismatches:
        return _artifact_error(
            "E5 lineage does not meet the declared matched contract",
            stage="e5",
            commands=[],
            mismatches=mismatches,
        )
    return {
        "stage": "e5",
        "status": "PLAN_ONLY_IMPLEMENTATION_BLOCKED",
        "failure_class": "SCIENTIFIC_UNRESOLVED",
        "commands": [],
        "reason": "lineage metadata passes, but the repository has no audited launcher that installs an arbitrary solver tensor into both matched r32/alpha32 branches; refusing a fake command",
        "lineage": str(lineage_path),
    }


def build_pipeline_state(args) -> dict:
    e0 = audit_prepared_inputs(args.data_dir, args.source_cf_dir, args.baselines_dir)
    solver_runs = {
        label: {**classify_solver_run(path), "declared_initial_arm": initial_arm}
        for label, initial_arm, path in args.solver_run
    }
    e1_commands = []
    if e0["status"] == "READY":
        for label, initial_arm, path in args.solver_run:
            if solver_runs[label]["status"] == "NOT_STARTED":
                e1_commands.append([
                    sys.executable,
                    "-m",
                    "experiments.olmo_recovery_20260912.solve_range_table",
                    "--model", str(args.model),
                    "--data", str(args.data_dir),
                    "--source-cf", str(args.source_cf_dir),
                    "--baselines", str(args.baselines_dir),
                    "--native-docs", str(args.native_docs),
                    "--initial-arm", initial_arm,
                    "--max-accepted-steps", str(args.max_accepted_steps),
                    "--out", str(path),
                    "--execute",
                ])
    e1 = {
        "stage": "e1",
        "status": (
            "BLOCKED_E0_ENGINEERING" if e0["status"] != "READY"
            else "READY_GPU" if e1_commands
            else "RUNNING" if any(record["status"] == "RUNNING_OR_PENDING_RECEIPT" for record in solver_runs.values())
            else "TERMINAL"
        ),
        "failure_class": "ENGINEERING_ERROR" if e0["status"] != "READY" else None,
        "solver_runs": solver_runs,
        "commands": e1_commands,
        "proxy_scope": "fit answer-plus-EOS NLL, counterfactual margin, and Native KL propose tables; they do not decide task success",
    }
    eligible = {
        label: record
        for label, record in solver_runs.items()
        if record["status"] in (
            "FIT_CANDIDATE_READY_FOR_GENERATION",
            "PARETO_FIT_CANDIDATE_READY_FOR_GENERATION",
        )
    }
    e2 = generation_stage_plan(
        stage="e2",
        split="select",
        model=args.model,
        data_dir=args.data_dir,
        source_cf_dir=args.source_cf_dir,
        run_root=args.run_root,
        candidates=solver_runs,
        prefill_chunk_size=args.prefill_chunk_size,
    ) if e0["status"] == "READY" else {
        "stage": "e2", "status": "BLOCKED_E0_ENGINEERING", "failure_class": "ENGINEERING_ERROR", "commands": []
    }
    selected_label, selection = _selection_from_lock(args.selection_lock, solver_runs)
    e3 = generation_stage_plan(
        stage="e3",
        split="internal_confirm",
        model=args.model,
        data_dir=args.data_dir,
        source_cf_dir=args.source_cf_dir,
        run_root=args.run_root,
        candidates=solver_runs,
        selected_label=selected_label,
        prefill_chunk_size=args.prefill_chunk_size,
    ) if selected_label is not None else {
        "stage": "e3", **selection, "commands": []
    }
    e4 = e4_plan(
        model=args.model,
        data_dir=args.data_dir,
        source_cf_dir=args.source_cf_dir,
        run_root=args.run_root,
        candidates=solver_runs,
        selected_label=selected_label,
        panels=args.e4_panel,
        baseline_refs=args.e4_baseline_ref,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    e5 = e5_plan(args.e5_lineage, selected_label)
    return {
        "status": "PLAN_ONLY" if not args.execute else "EXECUTION_REQUESTED",
        "problem": "one fixed Native-relative table at every layer and every runtime length inside the declared horizon",
        "dynamic_table_switching": False,
        "stages": {
            "e0": {"stage": "e0", **e0, "commands": []},
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
            "e5": e5,
        },
        "eligible_fit_candidates": list(eligible),
        "selection_lock": str(args.selection_lock) if args.selection_lock else None,
        "claim_boundary": "CPU geometry and fit losses are proposal signals; free generation on held-out prompts is the task endpoint",
    }


def execute_stage(args, state: dict) -> None:
    stage = state["stages"][args.stage]
    commands = stage.get("commands", [])
    if not commands:
        if args.stage == "e0" and stage.get("status") == "READY":
            return
        raise RuntimeError(f"stage {args.stage} has no executable commands: {stage.get('status')}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    empty_manifest = args.run_root / "empty_data_manifest.json"
    if not empty_manifest.exists():
        empty_manifest.write_text("{}\n")
    for command in commands:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e0", "e1", "e2", "e3", "e4", "e5"), default="e0")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-cf-dir", type=Path, required=True)
    parser.add_argument("--baselines-dir", type=Path, required=True)
    parser.add_argument("--native-docs", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--solver-run", type=parse_solver_run, action="append", default=[])
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--e4-panel", type=Path, action="append", default=[])
    parser.add_argument("--e4-baseline-ref", type=Path, action="append", default=[])
    parser.add_argument("--e5-lineage", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--max-accepted-steps", type=int, default=8)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prefill_chunk_size < 0 or args.max_accepted_steps <= 0:
        raise ValueError("prefill chunk size must be nonnegative and accepted steps positive")
    state = build_pipeline_state(args)
    if not args.execute:
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    execute_stage(args, state)
    refreshed = build_pipeline_state(args)
    if args.state_out:
        if args.state_out.exists():
            raise FileExistsError(args.state_out)
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        args.state_out.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n")
    print(json.dumps(refreshed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
