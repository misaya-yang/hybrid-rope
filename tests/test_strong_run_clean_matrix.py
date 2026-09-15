from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.iclr2027_strong_evidence_20260915 import run_clean_matrix as runner
from experiments.fixed_rope_three_interfaces_20260913 import tables


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, *, lengths=(8192, 16384), include_native=False):
    model = tmp_path / "model"
    write_json(model / "config.json", {
        "model_type": "olmo2", "hidden_size": 128, "num_attention_heads": 1,
        "rope_theta": 500000.0, "max_position_embeddings": 4096,
    })
    data = tmp_path / "data"
    panels = {}
    for length in lengths:
        path = data / "panels" / str(length) / "inputs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for task_index, task in enumerate(runner.RULER_TASKS):
            prompt = [task_index + 1, length // 1024]
            rows.append({
                "row_id": f"{length}_{task}", "task": task, "length_cap": length,
                "prompt_ids": prompt, "prompt_sha256": runner.prompt_sha256(prompt),
                "input_tokens": len(prompt), "references": ["answer"], "max_new_tokens": 8,
                "source_order_index": 0, "selection_mode": "source-order",
                "irrelevant_padding_tokens": 0,
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        child = path.parent / "manifest.json"
        write_json(child, {
            "status": "COMPLETE", "length_cap": length, "rows": 13,
            "inputs_sha256": file_sha(path), "model_id": "tiny_olmo", "scale": 4.0,
            "rows_per_task": 1, "tasks": list(runner.RULER_TASKS),
            "selection_mode": "source-order", "content_padding": False,
        })
        panels[str(length)] = {
            "inputs": str(path.relative_to(data)), "manifest": str(child.relative_to(data)),
            "rows": 13, "inputs_sha256": file_sha(path), "manifest_sha256": file_sha(child),
        }
    write_json(data / "manifest.json", {
        "status": "COMPLETE", "model_id": "tiny_olmo", "scale": 4.0,
        "lengths": list(lengths), "rows_per_task": 1, "selection_mode": "source-order",
        "content_padding": False, "panels": panels,
        "model_identity": {"config_sha256": file_sha(model / "config.json"), "native_length": 4096},
    })
    lm = tmp_path / "lm_manifest.json"
    write_json(lm, {})
    fake_python = tmp_path / "python"
    fake_python.write_text("executable placeholder\n")
    argv = [
        "--model", str(model), "--model-id", "tiny_olmo", "--data-root", str(data),
        "--out", str(tmp_path / "out"), "--scale", "4", "--lengths",
        *(str(value) for value in lengths), "--rows-per-task", "1",
        "--data-manifest", str(lm), "--python", str(fake_python),
    ]
    if include_native:
        argv.append("--include-native")
    args = runner.parse_args(argv)
    return args


def install_fake_expected_tables(monkeypatch) -> None:
    values = np.geomspace(1.0, 1e-4, 64).astype(np.float32)
    monkeypatch.setattr(
        runner, "expected_table",
        lambda args, arm: (values.copy(), 1.0 if arm == "native" else 1.125, {}),
    )


def freeze_table(args, arm: str) -> Path:
    config = runner.read_json(args.model / "config.json")
    method, role, changes = runner.ARM_SPECS[arm]
    values, gain, _ = runner.expected_table(args, arm)
    receipt = {
        "status": runner.TABLE_FORMAT,
        "candidate_id": runner.candidate_id(args.model_id, args.scale, arm),
        "model_id": args.model_id, "role": role, "scale": args.scale,
        "model_geometry": tables.model_geometry(config), "changed_variables": changes,
        "source": f"analytic:{method}", "table_sha256_float32": tables.tensor_sha256(values),
        "table": {"values_float32": values.tolist(), "gain": gain, "construction": {}},
    }
    path = args.out / "tables" / f"{arm}.json"
    write_json(path, receipt)
    return path


def freeze_complete_arm(args, arm, panels, rows):
    table = freeze_table(args, arm)
    run = args.out / "runs" / arm
    launch = runner.launch_contract(args, arm, table, panels, rows)
    write_json(run / "launch_contract.json", launch)
    label = runner.candidate_id(args.model_id, args.scale, arm)
    kernel = {
        "arm": label, "base_arm": "Native", "unadapted": True,
        "row_ids": [runner.eval_id(path, row) for path, row in runner.expected_eval_rows(panels)],
        "generation_length_caps": sorted(args.lengths), "lm_enabled": False,
        "batch_size": 1, "prefill_chunk_size": 8192,
        "static_table": runner.read_json(table)["table"],
    }
    write_json(run / "contract.json", kernel)
    output = []
    for path, source in runner.expected_eval_rows(panels):
        output.append({
            "eval_id": runner.eval_id(path, source), "row_id": source["row_id"],
            "task": source["task"], "length_cap": source["length_cap"],
            "prompt_sha256": source["prompt_sha256"], "references": source["references"],
            "arm": label, "ruler_official_score": 1.0, "ended_eos": True, "hit_cap": False,
        })
    (run / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in output))
    write_json(run / "summary.json", {"status": "COMPLETE", "identity": kernel})
    write_json(run / "status.json", {"status": "COMPLETE", "rows": len(rows), "lm_rows": 0})


def test_default_is_read_only_plan_with_explicit_portable_contract(tmp_path, capsys):
    args = fixture(tmp_path)
    runner.main([
        "--model", str(args.model), "--model-id", args.model_id,
        "--data-root", str(args.data_root), "--out", str(args.out),
        "--scale", "4", "--lengths", "8192", "16384", "--rows-per-task", "1",
        "--data-manifest", str(args.data_manifest), "--python", str(args.python),
    ])
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == runner.PLAN_STATUS and plan["execute"] is False
    assert plan["arms"] == ["tailspline", "mrpro"] and plan["rows"] == 26
    assert not args.out.exists()
    assert plan["gpu_lock"] == "/tmp/hybrid-rope-gpu0.lock"
    assert all("--batch-size" in command and command[command.index("--batch-size") + 1] == "1"
               for command in plan["evaluations"].values())


def test_complete_arms_are_strictly_validated_and_not_repeated(tmp_path, monkeypatch):
    args = fixture(tmp_path, include_native=True)
    plan, panels, rows = runner.build_plan(args)
    install_fake_expected_tables(monkeypatch)
    for arm in plan["arms"]:
        freeze_complete_arm(args, arm, panels, rows)
    calls = []

    def fake_run(command, *, cwd, env):
        calls.append(command)
        if any("matched_generation_report" in value for value in command):
            write_json(runner.report_path(args), {
                "status": "MATCHED_GENERATION_RANGE_REPORT_V1", "candidate": "tailspline",
                "baselines": ["mrpro", "native"], "lengths": [8192, 16384],
                "tasks": sorted(runner.RULER_TASKS), "paired_prompts": 26,
                "source_files": {arm: [str(args.out / "runs" / arm / "generations.jsonl")]
                                 for arm in ("tailspline", "mrpro", "native")},
                "summaries": {arm: {"rows": 26, "rows_per_task_length": [1]}
                              for arm in ("tailspline", "mrpro", "native")},
                "contrasts": {"mrpro": {}, "native": {}},
            })
            report = runner.read_json(runner.report_path(args))
            for arm in ("tailspline", "mrpro", "native"):
                report["summaries"][arm]["by_length"] = {
                    str(length): {
                        "task_macro_official": {"tailspline": 0.8, "mrpro": 0.7, "native": 0.75}[arm],
                        "tasks": {task: {"rows": 1} for task in runner.RULER_TASKS},
                    }
                    for length in (8192, 16384)
                }
            for baseline, delta in (("mrpro", 0.1), ("native", 0.05)):
                report["contrasts"][baseline] = {
                    "delta_by_length": {str(length): delta for length in (8192, 16384)},
                    "bootstrap": {"delta_by_length_interval95": {
                        str(length): [delta - 0.02, delta + 0.02] for length in (8192, 16384)
                    }},
                }
            write_json(runner.report_path(args), report)

    monkeypatch.setattr(runner, "_run", fake_run)
    result = runner.execute(args, plan, panels, rows)
    assert result["arms"] == ["tailspline", "mrpro", "native"]
    from experiments.iclr2027_strong_evidence_20260915.summarize_matrix import normalize_source
    source = runner.read_json(Path(result["matrix_source"]))
    normalized = normalize_source(source)
    assert normalized["identity"]["model_id"] == "tiny_olmo"
    assert normalized["identity"]["required_arms"] == ["tailspline", "mrpro", "native"]
    assert [cell["length_key"] for cell in normalized["cells"]] == ["2L", "4L"]
    assert normalized["cells"][1]["contrasts"]["tailspline_minus_native"]["ci95"] == pytest.approx([0.03, 0.07])
    assert len(calls) == 1 and any("matched_generation_report" in value for value in calls[0])
    runner.execute(args, plan, panels, rows)
    assert len(calls) == 1


def test_existing_matrix_source_must_match_the_validated_report(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    plan, panels, rows = runner.build_plan(args)
    install_fake_expected_tables(monkeypatch)
    for arm in plan["arms"]:
        freeze_complete_arm(args, arm, panels, rows)
    report = {
        "status": "MATCHED_GENERATION_RANGE_REPORT_V1", "candidate": "tailspline",
        "baselines": ["mrpro"], "tasks": sorted(runner.RULER_TASKS),
        "lengths": [8192, 16384], "paired_prompts": 26,
        "source_files": {arm: [str(args.out / "runs" / arm / "generations.jsonl")]
                         for arm in ("tailspline", "mrpro")},
        "summaries": {}, "contrasts": {"mrpro": {
            "delta_by_length": {"8192": 0.1, "16384": 0.1},
            "bootstrap": {"delta_by_length_interval95": {
                "8192": [0.05, 0.15], "16384": [0.05, 0.15],
            }},
        }},
    }
    for arm, score in (("tailspline", 0.8), ("mrpro", 0.7)):
        report["summaries"][arm] = {
            "rows": 26, "rows_per_task_length": [1],
            "by_length": {str(length): {
                "task_macro_official": score,
                "tasks": {task: {"rows": 1} for task in runner.RULER_TASKS},
            } for length in (8192, 16384)},
        }
    write_json(runner.report_path(args), report)
    valid = runner.matrix_source_payload(
        args, runner.read_json(args.data_root / "manifest.json"), plan["arms"], report,
    )
    invalid = json.loads(json.dumps(valid))
    invalid["experiment_id"] = "different_complete_experiment"
    write_json(args.out / "reports" / "matrix_source.json", invalid)
    monkeypatch.setattr(runner, "_run", lambda *a, **k: pytest.fail("all inputs are complete"))
    with pytest.raises(ValueError, match="different complete report"):
        runner.execute(args, plan, panels, rows)


def test_resume_rejects_prompt_or_table_drift_before_any_command(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    plan, panels, rows = runner.build_plan(args)
    install_fake_expected_tables(monkeypatch)
    freeze_complete_arm(args, "tailspline", panels, rows)
    freeze_table(args, "mrpro")
    generated = args.out / "runs" / "tailspline" / "generations.jsonl"
    saved = runner.read_jsonl(generated)
    saved[0]["prompt_sha256"] = "drift"
    generated.write_text("".join(json.dumps(row) + "\n" for row in saved))
    monkeypatch.setattr(runner, "_run", lambda *a, **k: pytest.fail("must fail before command"))
    with pytest.raises(ValueError, match="expected prompt prefix"):
        runner.execute(args, plan, panels, rows)


def test_incomplete_arms_never_form_a_matched_report(tmp_path):
    args = fixture(tmp_path)
    assert runner.report_command(args, ["tailspline"], runner.report_path(args)) is None
    command = runner.report_command(args, ["tailspline", "mrpro"], runner.report_path(args))
    assert command is not None and command.count("--source") == 2


def test_batching_and_manifest_identity_fail_closed(tmp_path):
    args = fixture(tmp_path)
    args.batch_size = 2
    with pytest.raises(ValueError, match="batch-size 1"):
        runner.build_plan(args)
    args = fixture(tmp_path / "second")
    manifest = runner.read_json(args.data_root / "manifest.json")
    manifest["model_id"] = "wrong"
    write_json(args.data_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest drift|model/scale/panel contract"):
        runner.build_plan(args)
