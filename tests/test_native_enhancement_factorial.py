import json

import numpy as np
import pytest

from experiments.native_enhancement_oral_20260915 import run_native_factorial as subject


def test_default_factorial_plan_never_imports_or_runs_a_model(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["factorial", "--root", str(tmp_path)])
    monkeypatch.setattr(subject.subprocess, "run", lambda *a, **k: pytest.fail("plan must not execute"))
    subject.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY" and result["new_generations"] == 216
    assert set(result["commands_without_execute"]) == set(subject.NEW_ARMS)
    assert all("--execute" not in command for command in result["commands_without_execute"].values())
    assert not list(tmp_path.iterdir())


def test_factorial_effects_sum_to_total_and_retain_joint_pairing():
    panel = [{"task": task} for task in ("a", "a", "b", "b")]
    scores = np.tile([0.2, 0.4, 0.3, 0.6], (4, 1))
    report = subject.summarize(panel, scores, draws=50)
    effects = report["effects"]
    assert effects["frequency_at_gain1"]["delta"] == pytest.approx(0.2)
    assert effects["gain_at_native"]["delta"] == pytest.approx(0.1)
    assert effects["interaction"]["delta"] == pytest.approx(0.1)
    assert sum(effects[name]["delta"] for name in ("frequency_at_gain1", "gain_at_native", "interaction")) == pytest.approx(effects["combined"]["delta"])
    # Common row noise must cancel in paired differences and their intervals.
    shared_noise = np.repeat([[0.1], [0.9], [0.2], [0.8]], 4, axis=1)
    equal = subject.summarize(panel, shared_noise, draws=100)
    assert all(value["delta"] == 0 and value["ci95"] == [0, 0] for value in equal["effects"].values())


def test_factorial_rejects_missing_or_mismatched_inputs():
    row = {"row_id": "r", "task": "a", "length_cap": 32768,
           "prompt_sha256": "hash", "input_tokens": 32000, "references": ["answer"]}
    key = subject.panel_key(row)
    cells = {arm: {key: {**row, "ruler_official_score": 0.5}} for arm in subject.CELLS}
    assert subject.paired_cell_scores([row], cells).shape == (1, 4)
    cells["candidate"][key]["references"] = ["other"]
    with pytest.raises(ValueError, match="identity differs"):
        subject.paired_cell_scores([row], cells)
    cells["candidate"] = {}
    with pytest.raises(ValueError, match="every prepared prompt"):
        subject.paired_cell_scores([row], cells)
