from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT, tables
from experiments.iclr2027_strong_evidence_20260915 import pro6000_128k_preflight as subject
from experiments.iclr2027_strong_evidence_20260915 import run_clean_matrix


TABLE_VALUES = np.geomspace(1.0, 1e-6, 64).astype(np.float32)
TABLE_GAIN = 1.138629436111989


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(model: Path, root: Path, arm: str) -> None:
    config = json.loads((model / "config.json").read_text())
    method = arm
    role = "candidate" if arm == "tailspline" else "baseline"
    values, gain = TABLE_VALUES.copy(), TABLE_GAIN
    receipt = {
        "status": TABLE_FORMAT,
        "candidate_id": f"strong_qwen25_3b_s4_{arm}",
        "model_id": "qwen25_3b",
        "role": role,
        "scale": 4.0,
        "model_geometry": tables.model_geometry(config),
        "changed_variables": ["internal_frequency_allocation"],
        "source": f"analytic:{method}",
        "table_sha256_float32": tables.tensor_sha256(values),
        "table": {"values_float32": values.tolist(), "gain": gain, "construction": {}},
    }
    write_json(root / "tables" / f"{arm}.json", receipt)


def qwen_fixture(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "qwen"
    write_json(model / "config.json", {
        "model_type": "qwen2", "hidden_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 1, "num_key_value_heads": 1,
        "rope_theta": 1_000_000.0, "max_position_embeddings": 32768,
    })
    root = tmp_path / "qwen_root"
    panels = {}
    for length in subject.QWEN_LENGTHS:
        panel = root / "assets" / "panels" / str(length) / "inputs.jsonl"
        panel.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in subject.TASKS:
            for index in range(subject.QWEN_ROWS_PER_TASK):
                rows.append({
                    "row_id": f"{length}_{task}_{index}", "task": task,
                    "length_cap": length, "prompt_ids": [index + 1, 2],
                    "input_tokens": 2, "max_new_tokens": 8,
                    "selection_mode": "source-order",
                    "selection_uses_model_outputs": False,
                    "irrelevant_padding_tokens": 0,
                })
        panel.write_text("".join(json.dumps(row) + "\n" for row in rows))
        child = panel.parent / "manifest.json"
        write_json(child, {
            "status": "COMPLETE", "rows": len(rows), "length_cap": length,
            "tasks": list(subject.TASKS), "selection_mode": "source-order",
            "content_padding": False,
        })
        panels[str(length)] = {
            "inputs": str(panel.relative_to(root / "assets")),
            "manifest": str(child.relative_to(root / "assets")),
            "rows": len(rows), "inputs_sha256": sha(panel),
            "manifest_sha256": sha(child),
        }
    write_json(root / "assets" / "manifest.json", {
        "status": "COMPLETE", "model_id": "qwen25_3b", "scale": 4.0,
        "lengths": list(subject.QWEN_LENGTHS),
        "rows_per_task": subject.QWEN_ROWS_PER_TASK,
        "seed": subject.QWEN_SEED, "qa_offset": subject.QWEN_QA_OFFSET,
        "selection_mode": "source-order", "selection_uses_model_outputs": False,
        "content_padding": False,
        "rows": len(subject.TASKS) * len(subject.QWEN_LENGTHS) * subject.QWEN_ROWS_PER_TASK,
        "model_identity": {"config_sha256": sha(model / "config.json")},
        "panels": panels,
    })
    for arm in ("tailspline", "mrpro"):
        write_table(model, root, arm)
    return root, model


def llama_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "llama_root"
    panel = root / "assets" / "full13" / "inputs.jsonl"
    panel.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in subject.TASKS:
        for index in range(10):
            rows.append({
                "task": task, "length_cap": 131072, "prompt_ids": [index + 1],
                "max_new_tokens": 8,
            })
    panel.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(root / "assets" / "full13" / "manifest.json", {
        "rows": 130, "inputs_sha256": sha(panel),
    })
    lm = root / "assets" / "ppl10" / "lm.npy"
    lm.parent.mkdir(parents=True, exist_ok=True)
    np.save(lm, np.zeros((1, 2), dtype=np.int64), allow_pickle=False)
    write_json(root / "assets" / "ppl10" / "manifest.json", {
        "documents": 10, "lengths": [131072], "lm_array_sha256": sha(lm),
    })
    table_hashes = {}
    for arm in ("tailspline", "mrpro"):
        table_hashes[arm] = f"hash-{arm}"
        write_json(root / "tables" / f"{arm}.json", {
            "table_sha256_float32": table_hashes[arm],
        })
    write_json(root / "assets" / "ready.json", {
        "status": "TAILSPLINE_LLAMA_S16_128K_ASSETS_READY_V1",
        "ruler_rows": 130, "ppl_documents": 10, "length": 131072,
        "band": [18, 35], "gpu_execution": False,
        "inputs_sha256": sha(panel), "lm_array_sha256": sha(lm),
        "table_sha256": table_hashes,
    })
    return root


def install_expected_tables(monkeypatch) -> None:
    monkeypatch.setattr(
        run_clean_matrix, "expected_table",
        lambda args, arm: (TABLE_VALUES.copy(), TABLE_GAIN, {}),
    )


def test_cpu_preflight_accepts_exact_frozen_assets(tmp_path, monkeypatch):
    install_expected_tables(monkeypatch)
    qwen_root, model = qwen_fixture(tmp_path)
    llama_root = llama_fixture(tmp_path)
    qwen = subject.validate_qwen_assets(qwen_root, model)
    llama = subject.validate_llama_assets(llama_root)
    assert qwen["rows_per_arm"] == 1300
    assert qwen["panels"]["131072"]["rows"] == 650
    assert llama["rows_per_arm"] == 130 and llama["lm_documents_per_arm"] == 10


def test_cpu_preflight_rejects_padding_and_table_drift(tmp_path, monkeypatch):
    install_expected_tables(monkeypatch)
    qwen_root, model = qwen_fixture(tmp_path)
    panel = qwen_root / "assets" / "panels" / "131072" / "inputs.jsonl"
    rows = panel.read_text().splitlines()
    first = json.loads(rows[0]); first["irrelevant_padding_tokens"] = 1
    rows[0] = json.dumps(first); panel.write_text("\n".join(rows) + "\n")
    manifest = json.loads((qwen_root / "assets" / "manifest.json").read_text())
    manifest["panels"]["131072"]["inputs_sha256"] = sha(panel)
    write_json(qwen_root / "assets" / "manifest.json", manifest)
    with pytest.raises(ValueError, match="clean contract"):
        subject.validate_qwen_assets(qwen_root, model)

    qwen_root, model = qwen_fixture(tmp_path / "second")
    table = qwen_root / "tables" / "tailspline.json"
    value = json.loads(table.read_text()); value["table"]["gain"] += 0.01
    write_json(table, value)
    with pytest.raises(ValueError, match="table drift"):
        subject.validate_qwen_assets(qwen_root, model)


def test_version_parsing_matches_blackwell_floor():
    assert subject.version_pair("2.8.0+cu128") == (2, 8)
    assert subject.cuda_pair("12.8") == (12, 8)
    assert subject.cuda_pair(None) == (0, 0)
