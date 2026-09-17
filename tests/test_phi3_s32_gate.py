import pytest

from experiments.phi3_s32_20260917.run_gate import TASKS, task_equal_score
from experiments.phi3_s32_20260917.queue import model_ready


def rows(score=0.8):
    return [
        {"task": task, "ruler_official_score": score}
        for task in TASKS for _ in range(10)
    ]


def test_task_equal_gate_score():
    score, by_task = task_equal_score(rows(0.81))
    assert score == pytest.approx(0.81)
    assert set(by_task) == set(TASKS)


def test_gate_rejects_incomplete_task_coverage():
    with pytest.raises(ValueError, match="Full-13"):
        task_equal_score(rows()[:-1])


def test_model_ready_checks_complete_indexed_safetensor_keys(tmp_path):
    import torch
    from safetensors.torch import save_file

    save_file({"x": torch.ones(3)}, tmp_path / "a.safetensors")
    save_file({"y": torch.ones(2)}, tmp_path / "b.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":20},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}}'
    )
    assert model_ready(tmp_path)
    save_file({"z": torch.ones(2)}, tmp_path / "b.safetensors")
    assert not model_ready(tmp_path)
