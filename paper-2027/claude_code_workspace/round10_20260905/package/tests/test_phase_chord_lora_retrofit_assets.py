"""CPU tests for phase-chord data, loss, and frozen frequency assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090.frequency_assets import (
    EXPECTED,
    float32_sha256,
    load_frequency_assets,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090.prepare_phase_chord_data import (
    _rows_from_receipt,
    build_split,
    write_split_streaming,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090.source_effect_loss import (
    gather_target_position_logits,
    source_effect_loss,
)


def _fixture() -> tuple[np.ndarray, list[dict[str, object]]]:
    tokens = np.arange(40 * 4096, dtype=np.uint32).reshape(40, 4096)
    rows = [
        {"row": i, "split": "train" if i < 20 else "validation", "document_id": f"doc-{i}"}
        for i in range(40)
    ]
    return tokens, rows


def test_production_geometry_and_split_isolation() -> None:
    tokens, rows = _fixture()
    result = build_split(tokens, rows, split="train", length=8192, distractor_count=6, distractor_block=1192)
    arrays = result["arrays"]
    assert arrays["correct_input_ids"].shape == (80, 8192)
    assert arrays["swapped_input_ids"].shape == (80, 8192)
    assert arrays["short_correct_input_ids"].shape == (80, 1040)
    assert arrays["short_swapped_input_ids"].shape == (80, 1040)
    assert np.array_equal(
        arrays["short_correct_input_ids"][:, 1024:],
        arrays["correct_input_ids"][:, -16:],
    )
    assert len(result["provenance"]) == 80
    assert result["manifest"]["target_position"] == [8176, 8192]
    source_starts = {row["source_position"][0] for row in result["provenance"]}
    assert len(source_starts) > 1
    assert min(source_starts) == 0
    assert max(source_starts) > 4_000
    for row in result["provenance"]:
        assert row["source_split"] == "train"
        assert row["source_document_id"] != row["other_source_document_id"]
        assert all(doc.startswith("doc-") and int(doc[4:]) < 20 for doc in row["distractor_document_ids"])
        assert row["same_positions_correct_swapped"] is True
        assert row["source_position"][0] >= 0
        assert row["source_position"][1] <= 8192 - 16


def test_validation_16k_uses_only_validation_rows() -> None:
    tokens, rows = _fixture()
    result = build_split(tokens, rows, split="validation", length=16384, distractor_count=16, distractor_block=959)
    assert result["arrays"]["correct_input_ids"].shape[1] == 16384
    assert len(result["provenance"]) == 80
    assert all(row["source_split"] == "validation" for row in result["provenance"])
    assert all(int(doc[4:]) >= 20 for row in result["provenance"] for doc in row["distractor_document_ids"])


def test_streaming_writer_is_atomic_and_matches_geometry(tmp_path: Path) -> None:
    tokens, rows = _fixture()
    output = tmp_path / "train8k"
    manifest = write_split_streaming(
        tokens,
        rows,
        split="train",
        length=8192,
        distractor_count=6,
        distractor_block=1192,
        output=output,
        source_hashes={"fixture": "0" * 64},
    )
    assert manifest["examples"] == 80
    assert not (tmp_path / "train8k.incomplete").exists()
    assert np.load(output / "correct_input_ids.npy", mmap_mode="r").shape == (80, 8192)
    assert np.load(output / "short_correct_input_ids.npy", mmap_mode="r").shape == (80, 1040)


def test_real_token_receipt_shape_uses_documents_not_integer_rows() -> None:
    documents = [
        {"parquet_row": index, "source_tokens": 5000, "text_sha256": f"sha-{index}"}
        for index in range(256)
    ]
    rows = _rows_from_receipt({"rows": 256, "documents": documents}, 256)
    assert rows[127]["split"] == "train"
    assert rows[128]["split"] == "validation"
    assert rows[-1]["document_id"] == "sha-255"


def test_source_effect_uses_teacher_positive_gate_and_p_minus_one() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions = torch.tensor([[2, 3, 4]])
    targets = torch.tensor([[3, 4, 5]])
    logits = torch.zeros(1, 5, 7)
    gathered = gather_target_position_logits(logits, ids, positions, targets)
    assert gathered.shape == (1, 3, 7)
    teacher_correct = torch.zeros(1, 3, 7)
    teacher_swapped = torch.zeros_like(teacher_correct)
    for index, token in enumerate(targets[0].tolist()):
        teacher_correct[0, index, token] = 3.0
        teacher_swapped[0, index, token] = 1.0
    student_correct = teacher_correct.clone().requires_grad_()
    student_swapped = teacher_swapped.clone().requires_grad_()
    loss, metrics = source_effect_loss(
        teacher_correct=teacher_correct,
        teacher_swapped=teacher_swapped,
        student_correct=student_correct,
        student_swapped=student_swapped,
        target_tokens=targets,
        shift_contract={
            key: (ids, positions)
            for key in ("teacher_correct", "teacher_swapped", "student_correct", "student_swapped")
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["teacher_selected_rate"] == 1.0
    assert student_correct.grad is not None
    direct_teacher = torch.full((1, 3), 2.0)
    direct_zero = torch.zeros_like(direct_teacher)
    direct_student = direct_teacher.clone().requires_grad_()
    direct_loss, direct_metrics = source_effect_loss(
        teacher_correct=direct_teacher,
        teacher_swapped=direct_zero,
        student_correct=direct_student,
        student_swapped=direct_zero,
        target_tokens=targets,
        values_are_logits=False,
    )
    direct_loss.backward()
    assert direct_metrics["teacher_selected_rate"] == 1.0
    assert direct_student.grad is not None


def test_frequency_loader_rejects_hash_or_reconstruction_drift(tmp_path: Path) -> None:
    native = np.linspace(1.0, 0.01, 64, dtype=np.float32)
    candidates = {
        "anchored_evq_cosh_tau_2": native.tolist(),
        "phase_chord_olmo_r0_lambda_0p1": native.tolist(),
    }
    payload = {"native": {"inv_freq": native.tolist()}, "candidates": {name: {"inv_freq": values} for name, values in candidates.items()}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Native frequency hash drift"):
        load_frequency_assets(path)
    assert len(EXPECTED["Native"]) == 64
    assert len(float32_sha256(native)) == 64
