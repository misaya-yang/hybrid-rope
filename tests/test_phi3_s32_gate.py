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


def test_model_ready_requires_exact_indexed_total(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a" * 3)
    (tmp_path / "b.safetensors").write_bytes(b"b" * 2)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":5},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}}'
    )
    assert model_ready(tmp_path)
    (tmp_path / "b.safetensors").write_bytes(b"b")
    assert not model_ready(tmp_path)
