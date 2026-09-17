from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.ca_ncp_native_20260917.build_alignment import solve, tables_only
from experiments.ca_ncp_native_20260917.io_utils import atomic_json, file_sha256
from experiments.ca_ncp_native_20260917.report import main as report_main
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from experiments.native_contrastive_proximal_20260915.tables import build_ncp_arrays
from scripts.experiments.cross_audit.tables import native_table


def fake_model(root: Path) -> Path:
    root.mkdir()
    atomic_json(root / "config.json", {
        "model_type": "olmo2",
        "architectures": ["Olmo2ForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4096,
        "rope_theta": 500000,
    })
    (root / "tokenizer.json").write_text("{}\n")
    return root


def construction(tmp_path: Path) -> Path:
    model = fake_model(tmp_path / "model")
    native = native_table(128, 500_000).astype(np.float32)
    ncp = build_ncp_arrays(native, native_length=4096)["candidate"]
    ncp_path = tmp_path / "ncp.json"
    atomic_json(ncp_path, {"values_float32": ncp.tolist(), "gain": 1.0})
    out = tmp_path / "construction"
    receipt = tables_only(model, ncp_path, out)
    assert receipt["construction"]["carrier_slot_zero_based"] == 35
    return out


def test_tables_only_exact_receipts(tmp_path):
    out = construction(tmp_path)
    method = json.loads((out / "METHOD_RECEIPT.json").read_text())
    assert method["ncp_table_sha256_float32"] == "54b9dd1f73aafc69f7bb5ed1b7b49d49128002371cb378d03ca1fd1d108e0cb7"
    assert method["carrier_table_sha256_float32"] == "10547fcaf8d8d6f0bc3935a839f0bd81734e0fa18855603cda3686c351aa06c7"


def test_alignment_solve_uses_fit_not_report(tmp_path):
    method_dir = construction(tmp_path)
    stats = tmp_path / "stats"
    (stats / "moments").mkdir(parents=True)
    rows = []
    dimension = 31
    for index, role in enumerate(("fit", "fit", "report", "report")):
        matrix = np.zeros((2, 16, dimension, dimension), dtype=np.complex128)
        matrix[..., 4, 4] = 5 if role == "fit" else -3
        bins = np.repeat(matrix[:, :, None], 5, axis=2)
        path = stats / "moments" / f"{index}.npz"
        np.savez_compressed(path, moment_real=matrix.real, moment_imag=matrix.imag,
                            bin_real=bins.real, bin_imag=bins.imag,
                            bin_counts=np.ones(5, dtype=np.int64))
        rows.append({"role": role, "moment_file": str(path.relative_to(stats)),
                     "moment_file_sha256": file_sha256(path)})
    atomic_json(stats / "STATISTICS_RECEIPT.json", {
        "status": "STATISTICS_COMPLETE",
        "method_id": "CA_NCP_NATIVE_V1_1",
        "method_receipt_sha256": file_sha256(method_dir / "METHOD_RECEIPT.json"),
        "distance_bins": ["a", "b", "c", "d", "e"],
        "documents": rows,
    })
    out = tmp_path / "alignment"
    receipt = solve(stats, method_dir, out)
    assert receipt["nonidentity_planes"] == 32
    assert receipt["plane_receipts"][0]["objective"] == 5
    assert receipt["plane_receipts"][0]["report_objective"] == -3
    with np.load(out / "alignment.npz", allow_pickle=False) as payload:
        assert payload["a"].shape == (2, 16)
        assert payload["active_indices"].tolist() == list(range(32, 63))


def test_run_plan_only_does_not_need_assets(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "experiments.ca_ncp_native_20260917.run",
        "--model", str(tmp_path / "missing-model"), "--root", str(tmp_path / "missing-root"),
    ], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["status"] == "PLAN_ONLY"
    assert payload["gpu_execution"] is False
    assert payload["expected_new_generations"] == 674
    assert payload["parallel_workers"] == 3


def _run(root: Path, arm: str, scores: dict[str, float], table: dict, alignment_sha: str | None):
    directory = root / arm
    directory.mkdir(parents=True)
    rows = []
    for task in TASKS:
        for index in range(10):
            row_id = f"{task}_{index}"
            rows.append({
                "row_id": row_id, "task": task, "prompt_sha256": f"p_{row_id}",
                "references": ["12345"], "input_tokens": 100, "max_new_tokens": 8,
                "ruler_official_score": scores[task], "output_text": "12345",
                "generated_ids": [1, 2], "ended_eos": True, "hit_cap": False, "empty": False,
            })
    (directory / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(directory / "status.json", {"status": "COMPLETE", "rows": 130, "lm_rows": 0})
    ca = {"label": arm, "sha256": alignment_sha} if alignment_sha else None
    atomic_json(directory / "contract.json", {"ca_ncp_alignment": ca})
    atomic_json(directory / "summary.json", {
        "table": table,
        "ca_ncp_alignment": ({"alignment_sha256": alignment_sha} if alignment_sha else None),
    })


def test_five_arm_report_exact_contrasts(tmp_path, monkeypatch):
    method_dir = construction(tmp_path)
    alignment_dir = tmp_path / "alignment"
    alignment_dir.mkdir()
    atomic_json(alignment_dir / "ALIGNMENT_RECEIPT.json", {
        "method_receipt_sha256": file_sha256(method_dir / "METHOD_RECEIPT.json"),
        "statistics_receipt_sha256": "stats",
        "alignment_sha256": "align",
    })
    assets = tmp_path / "assets"
    assets.mkdir()
    atomic_json(assets / "manifest.json", {
        "contract": "CA_NCP_NATIVE_FULL13_X10_REUSE_V1_1", "rows": 130,
        "panel": {"inputs_sha256": "panel"},
    })
    tables = {
        "N0": json.loads((method_dir / "native.json").read_text()),
        "C0": json.loads((method_dir / "ncp.json").read_text()),
        "P0": json.loads((method_dir / "carrier_ncp.json").read_text()),
        "N1": json.loads((method_dir / "native.json").read_text()),
        "P1": json.loads((method_dir / "carrier_ncp.json").read_text()),
    }
    runs = tmp_path / "runs"
    for arm, score in {"N0": 0.5, "C0": 0.52, "P0": 0.53, "N1": 0.54, "P1": 0.60}.items():
        _run(runs, arm, {task: score for task in TASKS}, tables[arm], "align" if arm in ("N1", "P1") else None)
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "report.py", "--assets", str(assets), "--run-root", str(runs),
        "--construction", str(method_dir), "--alignment", str(alignment_dir),
        "--out", str(out), "--draws", "100",
    ])
    report_main()
    report = json.loads(out.read_text())
    assert report["formal_contrasts"]["P1_minus_N0"] == pytest.approx(0.1)
    assert report["formal_contrasts"]["interaction"] == pytest.approx(0.03)
    assert report["formal_contrasts"]["relative_10pct_margin"] == pytest.approx(0.05)
