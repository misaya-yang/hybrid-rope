import json
from pathlib import Path

import numpy as np

from experiments.rope_z_ood_collision_20260913.pipeline import (
    BASELINE_ARMS,
    audit_prepared_inputs,
    classify_solver_run,
    e5_plan,
    generation_stage_plan,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prepared_inputs(tmp_path: Path):
    data = tmp_path / "data"
    source = tmp_path / "source"
    baselines = tmp_path / "baselines"
    split_counts = {"fit": 168, "select": 84, "internal_confirm": 84}
    data_rows = [
        {"row_id": f"main_{split}_{index}", "split": split, "target_includes_eos": True}
        for split, count in split_counts.items()
        for index in range(count)
    ]
    source_counts = {"fit": 128, "select": 64, "internal_confirm": 64}
    source_rows = [
        {"row_id": f"source_{split}_{index}", "split": split}
        for split, count in source_counts.items()
        for index in range(count)
    ]
    _write_json(data / "manifest.json", {"status": "RANGE_SOLVER_DATA_READY"})
    _write_jsonl(data / "rows.jsonl", data_rows)
    _write_json(source / "manifest.json", {"status": "RANGE_SOURCE_COUNTERFACTUAL_READY"})
    _write_jsonl(source / "rows.jsonl", source_rows)
    _write_json(baselines / "status.json", {"status": "COMPLETE", "arms": list(BASELINE_ARMS)})
    fit_ids = [row["row_id"] for row in data_rows if row["split"] == "fit"]
    source_fit_ids = [row["row_id"] for row in source_rows if row["split"] == "fit"]
    for arm in BASELINE_ARMS:
        _write_jsonl(baselines / f"{arm}.jsonl", [{"row_id": row_id} for row_id in fit_ids])
        _write_jsonl(baselines / f"source_cf_{arm}.jsonl", [{"row_id": row_id} for row_id in source_fit_ids])
    return data, source, baselines


def _solver_result(run_dir: Path, *, accepted_steps: int, feasible: bool) -> None:
    values = (1.0 / (1.1 ** np.arange(64))).tolist()
    result = {
        "status": "COMPLETE",
        "initial_arm": "BM_g4",
        "accepted_steps": accepted_steps,
        "allocation": {"same_table_all_layers_and_lengths": True, "model_weight_updates": 0},
        "table": {"values_float32": values, "gain": 1.1},
        "full_fit_validation": {
            "all_hard_constraints_hold": feasible,
            "range_objective": [0.1, 0.05],
            "feasibility_objective": [0.0, 0.0] if feasible else [0.2, 0.1],
        },
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "status.json", {"status": "COMPLETE", "accepted_steps": accepted_steps})


def test_e0_audit_checks_split_and_baseline_identity_without_hash_gate(tmp_path):
    data, source, baselines = _prepared_inputs(tmp_path)
    result = audit_prepared_inputs(data, source, baselines)
    assert result["status"] == "READY"
    assert result["split_counts"]["select"] == {"main": 84, "source_cf": 64}
    assert result["asset_identity_policy"] == "reuse supplied clone; no repeated SHA gate"


def test_completed_zero_step_solver_is_not_mislabeled_as_candidate(tmp_path):
    run = tmp_path / "solver"
    _solver_result(run, accepted_steps=0, feasible=False)
    result = classify_solver_run(run)
    assert result["status"] == "TERMINAL_NO_NEW_CANDIDATE"
    assert result["failure_class"] == "SCIENTIFIC_UNRESOLVED"
    assert result["task_evidence"] is False


def test_fit_candidate_only_qualifies_after_nonzero_step_and_full_fit_contract(tmp_path):
    run = tmp_path / "solver"
    _solver_result(run, accepted_steps=2, feasible=True)
    result = classify_solver_run(run)
    assert result["status"] == "FIT_CANDIDATE_READY_FOR_GENERATION"
    assert result["task_evidence"] is False


def test_nonzero_infeasible_fit_is_retained_as_pareto_generation_candidate(tmp_path):
    run = tmp_path / "solver"
    _solver_result(run, accepted_steps=1, feasible=False)
    result = classify_solver_run(run)
    assert result["status"] == "PARETO_FIT_CANDIDATE_READY_FOR_GENERATION"
    plan = generation_stage_plan(
        stage="e2",
        split="select",
        model=tmp_path / "model",
        data_dir=tmp_path / "data",
        source_cf_dir=tmp_path / "source",
        run_root=tmp_path / "runs",
        candidates={"pareto_candidate": result},
    )
    assert plan["status"] == "READY_GPU"


def test_generation_plan_is_static_and_execute_gated(tmp_path):
    run = tmp_path / "solver"
    _solver_result(run, accepted_steps=2, feasible=True)
    candidate = classify_solver_run(run)
    plan = generation_stage_plan(
        stage="e2",
        split="select",
        model=tmp_path / "model",
        data_dir=tmp_path / "data",
        source_cf_dir=tmp_path / "source",
        run_root=tmp_path / "runs",
        candidates={"candidate_a": candidate},
    )
    assert plan["status"] == "READY_GPU"
    commands = [token for command in plan["commands"] for token in command]
    assert "--execute" in commands
    assert "--static-table-json" in commands
    assert "--row-split" in commands and "select" in commands
    assert plan["fixed_table_policy"].startswith("each candidate uses one static")


def test_existing_summary_is_refreshed_when_a_new_candidate_output_is_missing(tmp_path):
    run = tmp_path / "solver"
    _solver_result(run, accepted_steps=1, feasible=False)
    candidate = classify_solver_run(run)
    run_root = tmp_path / "runs"
    stage_root = run_root / "select_r0"
    _write_json(stage_root / "summary.json", {"status": "COMPLETE", "arms": {}})
    plan = generation_stage_plan(
        stage="e2",
        split="select",
        model=tmp_path / "model",
        data_dir=tmp_path / "data",
        source_cf_dir=tmp_path / "source",
        run_root=run_root,
        candidates={"candidate_a": candidate},
    )
    assert plan["status"] == "READY_GPU"
    assert plan["commands"][-1][2] == "experiments.olmo_recovery_20260912.summarize_range_generation"


def test_e5_is_blocked_instead_of_faking_a_matched_launcher():
    result = e5_plan(None, "candidate")
    assert result["status"] == "BLOCKED_MATCHED_LINEAGE"
    assert result["commands"] == []
