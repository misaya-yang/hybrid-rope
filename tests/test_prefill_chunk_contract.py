"""CPU checks for prefill chunk benchmark argument handling."""

import pytest

from experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks import (
    parse_chunks,
)
from experiments.iclr2027_three_track_sprint_20260915 import benchmark_generation_batch
from experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch import (
    parse_batch_sizes,
    select_candidate_groups,
)


def test_prefill_chunk_list_is_positive_and_unique():
    assert parse_chunks('0,8192,16384,32768') == [0, 8192, 16384, 32768]
    with pytest.raises(ValueError, match='unique nonnegative'):
        parse_chunks('8192,8192')
    with pytest.raises(ValueError, match='unique nonnegative'):
        parse_chunks('-1,8192')


def test_batch_canary_uses_largest_then_longest_exact_shape_group():
    rows = [
        {"row_id": "a", "length_cap": 131072, "prompt_ids": [1] * 10, "max_new_tokens": 8},
        {"row_id": "b", "length_cap": 131072, "prompt_ids": [2] * 10, "max_new_tokens": 8},
        {"row_id": "c", "length_cap": 131072, "prompt_ids": [3] * 20, "max_new_tokens": 8},
        {"row_id": "d", "length_cap": 131072, "prompt_ids": [4] * 20, "max_new_tokens": 8},
        {"row_id": "e", "length_cap": 131072, "prompt_ids": [5] * 20, "max_new_tokens": 8},
        {"row_id": "f", "length_cap": 131072, "prompt_ids": [6] * 20, "max_new_tokens": 8},
    ]
    groups = select_candidate_groups(
        rows, length=131072, batch_sizes=[2, 4],
    )
    assert [row["row_id"] for row in groups[2]["throughput"]] == ["c", "d"]
    assert [row["row_id"] for row in groups[4]["throughput"]] == ["c", "d", "e", "f"]
    assert parse_batch_sizes("2,4,8") == [2, 4, 8]
    with pytest.raises(ValueError, match="no exact-length batch group"):
        select_candidate_groups(rows[:1], length=131072, batch_sizes=[2, 4])


def test_large_batch_candidate_stresses_longer_partial_bucket():
    rows = [
        {"row_id": f"long-{i}", "length_cap": 131072, "prompt_ids": [1] * 200,
         "max_new_tokens": 8}
        for i in range(4)
    ] + [
        {"row_id": f"short-{i}", "length_cap": 131072, "prompt_ids": [2] * 60,
         "max_new_tokens": 8}
        for i in range(8)
    ]
    selected = select_candidate_groups(rows, length=131072, batch_sizes=[8])[8]
    assert len(selected["throughput"]) == 8
    assert selected["throughput"][0]["row_id"].startswith("short-")
    assert len(selected["stress"]) == 4
    assert selected["stress"][0]["row_id"].startswith("long-")


def test_batch_canary_without_a_group_writes_batch1_fallback(tmp_path, monkeypatch):
    panel = tmp_path / "panel.jsonl"
    panel.write_text('{"row_id":"one","length_cap":131072,"prompt_ids":[1],"max_new_tokens":8}\n')
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", [
        "benchmark_generation_batch", "--model", str(tmp_path / "model"),
        "--table", str(tmp_path / "table.json"), "--panel", str(panel),
        "--length", "131072", "--batch-sizes", "2,4", "--out", str(out),
    ])
    benchmark_generation_batch.main()
    report = __import__("json").loads(out.read_text())
    assert report["recommended_batch_size"] == 1
    assert "not_applicable_reason" in report
