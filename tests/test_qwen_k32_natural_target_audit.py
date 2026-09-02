"""CPU-only packed-target boundary audit tests."""

from __future__ import annotations

import json

import pytest

from scripts.data import audit_qwen_k32_packed_natural_targets as audit
from scripts.data import prepare_qwen_k32_natural_nll as builder


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.setattr(builder, "GRID", (8, 16))
    monkeypatch.setattr(builder, "DOCUMENTS", 2)
    monkeypatch.setattr(builder, "TARGET_TOKENS", 2)
    monkeypatch.setattr(audit, "GRID", (8, 16))
    monkeypatch.setattr(audit, "DOCUMENTS", 2)
    monkeypatch.setattr(audit, "TARGET_TOKENS", 2)


def write_data(root, tails):
    root.mkdir()
    rows = []
    for index, tail in enumerate(tails):
        long_ids = [10 + index] * (16 - len(tail)) + tail
        for length in (8, 16):
            ids = long_ids[-length:]
            rows.append(
                {
                    "sample_id": f"qwen-k32-natural-{index:03d}",
                    "length": length,
                    "input_ids": ids,
                    "target_start": length - 2,
                    "target_tokens": 2,
                    "prompt_ids_sha256": builder.ids_hash(ids),
                    "target_ids_sha256": builder.ids_hash(ids[-2:]),
                }
            )
    rows_path = root / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "model_evaluation_status": "NOT_RUN",
        "model_outcomes_read": False,
        "grid": [8, 16],
        "natural_streams": 2,
        "target_tokens": 2,
        "tokenizer_eos_token_id": 2,
        "file": {"path": "rows.jsonl", "sha256": builder.sha256_file(rows_path)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_boundary_safe_exact_suffix(contract, tmp_path):
    root = tmp_path / "safe"
    write_data(root, [[20, 21, 22, 23], [30, 31, 32, 33]])
    report = audit.audit(root)
    assert report["status"] == audit.STATUS
    assert report["safe_streams"] == 2
    assert report["unsafe_streams"] == []


def test_eos_inside_target_context_fails(contract, tmp_path):
    root = tmp_path / "unsafe"
    write_data(root, [[2, 21, 22, 23], [30, 31, 32, 33]])
    report = audit.audit(root)
    assert report["status"].endswith("BOUNDARY_UNSAFE")
    assert report["unsafe_streams"] == ["qwen-k32-natural-000"]


def test_suffix_corruption_fails_closed(contract, tmp_path):
    root = tmp_path / "corrupt"
    write_data(root, [[20, 21, 22, 23], [30, 31, 32, 33]])
    rows_path = root / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["input_ids"][0] += 1
    rows[0]["prompt_ids_sha256"] = builder.ids_hash(rows[0]["input_ids"])
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["file"]["sha256"] = builder.sha256_file(rows_path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = audit.audit(root)
    assert report["status"].endswith("BOUNDARY_UNSAFE")
