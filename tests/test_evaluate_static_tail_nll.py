import json
import math

import pytest
from types import SimpleNamespace

from experiments.olmo_recovery_20260912.evaluate_static_tail_nll import (
    load_table,
    native_table_for_config,
    summarize,
)


def test_load_table_and_summarize(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({
        "table": {
            "values_float32": [1.0 / (index + 1) for index in range(64)],
            "gain": 1.2,
        }
    }))
    assert load_table(path)["gain"] == 1.2
    rows = [
        {"length": length, "nll": value}
        for length, values in ((4096, (2.0, 4.0)), (8192, (1.0, 3.0)))
        for value in values
    ]
    result = summarize(rows, [4096, 8192], 2)
    assert result["4096"]["mean_tail_nll"] == 3.0
    assert result["8192"]["tail_ppl"] == pytest.approx(math.exp(2.0))


def test_load_table_rejects_non_decreasing(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"values_float32": [1.0] * 64, "gain": 1.0}))
    with pytest.raises(ValueError, match="invalid frozen"):
        load_table(path)


def test_builds_native_table_from_qwen_config():
    result = native_table_for_config(SimpleNamespace(
        hidden_size=1536,
        num_attention_heads=12,
        head_dim=None,
        rope_theta=1_000_000.0,
    ))
    assert len(result["values_float32"]) == 64
    assert result["gain"] == 1.0
    assert result["construction"]["method"] == "identity"
