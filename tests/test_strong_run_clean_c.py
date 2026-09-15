import json
from pathlib import Path
import fcntl

import numpy as np
import pytest

from experiments.iclr2027_strong_evidence_20260915 import run_clean_c as runner


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_default_is_plan_only_and_schedules_only_new_batch1_c(capsys):
    runner.main([])
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["gpu_started"] is False
    assert plan["reuse_arms"] == ["tailspline", "mrpro"]
    assert plan["new_arm_only"] == "dose_control_c"
    assert plan["batch_size"] == 1
    assert plan["gpu_lock"] == "/tmp/hybrid-rope-gpu0.lock"
    assert plan["gpu_lock_mode"] == "exclusive_nonblocking_execute_only"
    assert [(job["length"], job["shard"], job["expected_rows"]) for job in plan["jobs"]] == [
        (16384, "nonqa11", 550),
        (16384, "qa2", 100),
        (32768, "full13", 2600),
    ]
    command = plan["table_command"]
    assert command[command.index("--method") + 1] == "tailspline_dose_control"
    assert "run_tailspline_llama_s4_matched_dose_c.sh" not in " ".join(command)


def test_only_frozen_clean_lengths_and_counts_are_accepted():
    assert runner.parse_lengths(["32768,16384"]) == (32768, 16384)
    assert runner.parse_lengths([["16384", "32768"]]) == (16384, 32768)
    assert runner.parse_rows_per_task(["32768:200", "16384:50"], (32768, 16384)) == {
        32768: 200, 16384: 50,
    }
    with pytest.raises(ValueError, match="supports"):
        runner.parse_lengths(["8192"])
    with pytest.raises(ValueError, match="frozen clean contracts"):
        runner.parse_rows_per_task(["16384:51"], (16384,))


def test_panel_and_completed_arm_must_be_exactly_source_order_paired(tmp_path):
    panel = tmp_path / "panel.jsonl"
    source = [
        {"task": "qa_1", "length_cap": 16384, "prompt_sha256": "a"},
        {"task": "qa_2", "length_cap": 16384, "prompt_sha256": "b"},
    ]
    write_jsonl(panel, source)
    job = runner.Job(
        length=16384, shard="qa2", tasks=("qa_1", "qa_2"), rows_per_task=1,
        panel=panel, tailspline_run=tmp_path / "t", mrpro_run=tmp_path / "p",
        control_run=tmp_path / "c",
    )
    _, prompts = runner.validate_panel(job)
    assert prompts == ["a", "b"]

    values = np.geomspace(1.0, 1e-4, 64).astype(np.float32)
    receipt = tmp_path / "table.json"
    write_json(receipt, {
        "table_sha256_float32": runner.tensor_sha256(values),
        "table": {"values_float32": values.tolist(), "gain": 1.1},
    })
    run = tmp_path / "run"
    write_json(run / "status.json", {"status": "COMPLETE", "rows": 2, "lm_rows": 0})
    write_json(run / "contract.json", {
        "generation_length_caps": [16384], "batch_size": 1,
        "static_table": {"values_float32": values.tolist(), "gain": 1.1},
    })
    write_jsonl(run / "generations.jsonl", [
        {"prompt_sha256": "b", "ruler_official_score": 1},
        {"prompt_sha256": "a", "ruler_official_score": 1},
    ])
    with pytest.raises(ValueError, match="source order"):
        runner.validate_complete_run(run, job, receipt=receipt, expected_prompts=prompts)


def test_runtime_comparison_rejects_batch_or_prefill_drift():
    reference = {
        "generation_length_caps": [32768], "limit_per_cell": 0,
        "prefill_chunk_size": 8192, "generation_prefill_strategy": "dynamic_cache_lower_right_v1",
        "batch_size": 1, "runtime_versions": {"model_dtype": "bfloat16"},
    }
    runner.validate_runtime_match(reference, dict(reference), label="same")
    changed = dict(reference, batch_size=2)
    with pytest.raises(ValueError, match="runtime contract"):
        runner.validate_runtime_match(reference, changed, label="batch2")


def test_report_contract_contains_all_three_arms_and_c_minus_p(tmp_path):
    jobs = runner.build_jobs(
        data_root=tmp_path / "data", out=tmp_path / "new",
        lengths=(16384, 32768), rows_per_task={16384: 50, 32768: 200},
    )
    args = runner.build_parser().parse_args([
        "--python", "/runtime/python", "--data-root", str(tmp_path / "data"),
    ])
    primary, secondary = runner.report_commands(args, jobs, tmp_path / "new" / "reports")
    assert primary.count("--source") == 9
    assert primary[primary.index("--candidate") + 1] == "tailspline"
    assert [primary[index + 1] for index, value in enumerate(primary) if value == "--baseline"] == [
        "dose_control_c", "mrpro",
    ]
    assert secondary[secondary.index("--candidate") + 1] == "dose_control_c"
    assert secondary[secondary.index("--baseline") + 1] == "mrpro"
    assert "--length" in primary and "16384" in primary and "32768" in primary


def test_execute_lock_conflict_fails_before_execution(monkeypatch, tmp_path):
    lock = tmp_path / "gpu0.lock"
    monkeypatch.setattr(runner, "GPU_LOCK_PATH", lock)
    called = False

    def forbidden(_args):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_execute_under_lock", forbidden)
    with lock.open("a+") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already owned"):
            runner.execute(runner.build_parser().parse_args([]))
    assert called is False
