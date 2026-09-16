from argparse import Namespace
import json
from pathlib import Path

import pytest

from experiments.iclr2027_strong_evidence_20260915 import llama_s16_confirmation as subject


def panel(block):
    return [{"row_id": f"{task}_{i}", "task": task, "length_cap": subject.LENGTH,
             "input_tokens": 130000, "prompt_sha256": f"{block}:{task}:{i}", "references": ["x"]}
            for task in subject.TASKS for i in range(10)]


def test_plan_has_four_disjoint_fixed_blocks_and_no_execution(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["confirmation", "--out", str(tmp_path)])
    monkeypatch.setattr(subject.subprocess, "run", lambda *a, **kw: pytest.fail("plan must not run"))
    subject.main()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PLAN_ONLY"
    assert output["new_generations"] == 1040 and output["new_lm_rows"] == 0
    ranges = [set(range(block["qa_offset"], block["qa_offset"] + 10)) for block in subject.BLOCKS]
    assert len(set.union(*ranges)) == 40
    assert not set.union(*ranges).intersection(range(5800, 5810))
    assert max(set.union(*ranges)) < 5928
    seeds = [{block["seed"] + i * 100 for i in range(13)} for block in subject.BLOCKS]
    assert len(set.union(*seeds)) == 52
    assert not set.union(*seeds).intersection(range(20262001, 20262014))
    assert all("--execute" not in command for command in output["commands_cpu_prepare"])
    assert not list(tmp_path.iterdir())


def test_same_row_ids_across_blocks_are_allowed_but_prompt_overlap_is_not():
    panels = {name: panel(name) for name in ("gate", "block1", "block2", "block3", "block4")}
    subject.validate_disjoint(panels)
    panels["block4"][0]["prompt_sha256"] = panels["gate"][0]["prompt_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        subject.validate_disjoint(panels)


def test_generation_reuses_runtime_and_skips_lm():
    args = Namespace(python="python", model=Path("/model"), gate=Path("/gate"), out=Path("/out"))
    ready = {"generation_runtime": {"prefill_chunk_size": 0, "batch_size": 2,
                                   "generation_order": "ascending_shape_sorted_v1"},
             "lm_prefill_chunk_size": 65536}
    command = subject.generation_command(args, {"block": 1, "inputs": "block1/inputs.jsonl"}, "tailspline", ready)
    assert "--skip-lm" in command and "--execute" not in command
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--lm-prefill-chunk-size") + 1] == "65536"
    assert "--longest-first" not in command
    assert command[command.index("--static-table-json") + 1] == "/gate/tables/tailspline.json"


def test_report_keeps_independent40_separate_from_observed_gate(monkeypatch, tmp_path):
    args = Namespace(gate=tmp_path / "gate", out=tmp_path / "confirm")
    runtime = {key: None for key in subject.RUNTIME_KEYS}
    ready = {"blocks": list(subject.BLOCKS), "generation_runtime": runtime,
             "tables": {arm: {"values_float32": [1.0], "gain": 1.2} for arm in subject.ARMS}}
    panels = {name: panel(name) for name in ("gate", "block1", "block2", "block3", "block4")}
    for name, rows in panels.items():
        for arm in subject.ARMS:
            path = (args.gate if name == "gate" else args.out / name) / "runs" / arm
            path.mkdir(parents=True)
            (path / "status.json").write_text(json.dumps({"status": "COMPLETE", "rows": 130,
                                                        "lm_rows": 10 if name == "gate" else 0}))
            (path / "contract.json").write_text(json.dumps({**runtime, "static_table": ready["tables"][arm]}))
            score = (1.0 if name == "gate" else 0.2) if arm == "tailspline" else 0.0
            (path / "generations.jsonl").write_text("".join(
                json.dumps({**row, "ruler_official_score": score}) + "\n" for row in rows))
    original = subject.paired_summary
    monkeypatch.setattr(subject, "paired_summary", lambda cells: original(cells, draws=20))
    result = subject.report(args, ready, panels)
    assert result["confirm40_primary"]["rows_per_arm"] == 520
    assert result["confirm40_primary"]["delta"] == pytest.approx(0.2)
    assert result["gate10_plus_confirm40_cumulative50"]["rows_per_arm"] == 650
    assert result["gate10_plus_confirm40_cumulative50"]["delta"] == pytest.approx(0.36)
    assert result["adaptive_stopping"] is False
    # Missing or mismatched rows cannot be silently intersected away.
    path = args.out / "block4/runs/mrpro/generations.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="coverage"):
        subject.report(args, ready, panels)


def test_paired_bootstrap_preserves_common_row_noise():
    scores = {(task, str(i)): float(i % 2) for task in subject.TASKS for i in range(10)}
    result = subject.paired_summary({arm: scores for arm in subject.ARMS}, draws=50)
    assert result["delta"] == 0 and result["ci95"] == [0, 0]


def test_negative_block_scores_do_not_stop_later_blocks(monkeypatch, tmp_path):
    ready = {"blocks": [{**block, "inputs": f"block{block['block']}/inputs.jsonl"}
                        for block in subject.BLOCKS],
             "generation_runtime": {"prefill_chunk_size": 0, "batch_size": 2,
                                    "generation_order": "ascending_shape_sorted_v1"},
             "lm_prefill_chunk_size": 0}
    panels = {f"block{i}": panel(i) for i in range(1, 5)}
    monkeypatch.setattr("sys.argv", ["confirmation", "--out", str(tmp_path), "--execute"])
    monkeypatch.setenv("GPU_LOCK_PATH", str(tmp_path / "test.lock"))
    monkeypatch.setattr(subject, "validate_ready", lambda args: (ready, panels))
    monkeypatch.setattr(subject, "load_scores", lambda *a, **kw: {})
    monkeypatch.setattr(subject, "paired_summary", lambda cells: {"delta": -1.0})
    monkeypatch.setattr(subject, "report", lambda *a: {"status": "TEST_COMPLETE"})
    commands = []
    monkeypatch.setattr(subject.subprocess, "run", lambda command, **kw: commands.append(command))
    subject.main()
    assert len(commands) == 8
    assert all("--execute" in command for command in commands)
    assert (tmp_path / "block4/report.json").is_file()
