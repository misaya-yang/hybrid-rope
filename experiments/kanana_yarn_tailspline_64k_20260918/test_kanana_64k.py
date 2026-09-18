from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.kanana_yarn_tailspline_64k_20260918.prepare import (
    PILOT_TASKS,
    freeze_subpanels,
)
from experiments.kanana_yarn_tailspline_64k_20260918.report import build_report
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config


def _generation(path: Path, *, arm: str, delta: float = 0.0) -> None:
    rows = []
    for task in PILOT_TASKS:
        for index in range(10):
            score = 0.5 + (delta if arm == "tailspline" else 0.0)
            rows.append({
                "prompt_sha256": f"{task}-{index}", "task": task,
                "length_cap": 65536, "ruler_official_score": score,
                "ended_eos": True, "hit_cap": False, "empty": False,
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_kanana_native_loader_geometry_is_explicit_static_only():
    config = SimpleNamespace(
        model_type="llama", hidden_size=4096, head_dim=128,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        max_position_embeddings=32768, rope_theta=None,
        rope_parameters={"rope_theta": 8_000_000.0, "rope_type": "default"},
    )
    table = table_for_config(config, "Native")
    assert len(table["values_float32"]) == 64
    assert table["gain"] == 1.0


def test_pilot_gate_expands_only_when_difference_is_small(tmp_path: Path):
    yarn = tmp_path / "yarn.jsonl"
    tailspline = tmp_path / "tailspline.jsonl"
    _generation(yarn, arm="official_yarn")
    _generation(tailspline, arm="tailspline", delta=0.05)
    report = build_report(
        tailspline_paths=[tailspline], yarn_paths=[yarn],
        mode="pilot", expand_threshold=0.10,
    )
    assert report["gate"]["decision"] == "EXPAND_FULL13"
    _generation(tailspline, arm="tailspline", delta=0.20)
    report = build_report(
        tailspline_paths=[tailspline], yarn_paths=[yarn],
        mode="pilot", expand_threshold=0.10,
    )
    assert report["gate"]["decision"] == "STOP_DIRECTIONALLY_LARGE_PILOT"


def test_subpanels_partition_full13(tmp_path: Path):
    full = tmp_path / "full.jsonl"
    rows = []
    for task in TASKS:
        for index in range(10):
            rows.append({
                "row_id": f"{task}-{index}", "task": task,
                "length_cap": 65536, "prompt_ids": [1, 2, 3],
                "max_new_tokens": 4, "selection_mode": "source-order",
                "selection_uses_model_outputs": False,
                "irrelevant_padding_tokens": 0,
            })
    full.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = freeze_subpanels(full, tmp_path / "root")
    assert result["pilot2"]["rows"] == 20
    assert result["rest11"]["rows"] == 110
