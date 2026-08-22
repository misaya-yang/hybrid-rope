"""CPU tests for explicit phase AdaRoPE stage contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.objectives import (
    ComponentGateConfig,
    Stage0ObjectiveConfig,
    document_component_gate,
    selected_hidden_lm_head,
    select_tournament_winner,
    stage0_answer_margin_loss,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.receipts import (
    no_cuda_preflight,
    require_dual_authorization,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope import (
    PairView,
    _load_target,
    public_interfaces,
    stage_contract,
    validate_parent_for_stage,
)


def test_public_interfaces_and_contracts_are_bound() -> None:
    interfaces = public_interfaces()
    assert set(interfaces) == {"data", "attention"}
    assert stage_contract("stage0")["steps"] == 300
    assert stage_contract("stage1", arm="native_scale")["warmup_steps"] == 20
    assert stage_contract("stage2", arm="phase_chord")["steps"] == 250
    assert stage_contract("stage2", arm="native_scale")["arm"] == "native_scale"


def test_stage_transitions_fail_closed() -> None:
    validate_parent_for_stage("stage0", None)
    with pytest.raises(ValueError):
        validate_parent_for_stage("stage1", None)
    with pytest.raises(ValueError):
        validate_parent_for_stage("stage2", {"stage": "stage0"}, {"status": "STOP"})
    gate = {"status": "PASS", "exact_answer_terminal_eos": 0.90, "terminal_eos_rate": 1.0, "source_follow_positive_fraction": 0.90}
    validate_parent_for_stage("stage2", {"stage": "stage1"}, {"status": "PASS"}, arm="lora_only_null")


def test_component_gate_is_document_level_and_excludes_final_validation() -> None:
    rows = []
    for document in ("a", "b"):
        for offset in range(2):
            rows.append({
                "document_id": document,
                "split": "selection",
                "candidate": {"nll": 1.0, "source_effect": 2.0},
                "parent": {"nll": 1.05, "source_effect": 1.0},
                "offset": offset,
            })
    rows.append({
        "document_id": "final-only",
        "split": "final_validation",
        "candidate": {"nll": 100.0, "source_effect": -100.0},
        "parent": {"nll": 0.0, "source_effect": 0.0},
    })
    result = document_component_gate(rows, candidate_key="candidate", config=ComponentGateConfig())
    assert result["status"] == "PASS"
    assert result["selection_documents"] == 2
    assert result["final_validation_used_for_selection"] is False


def test_component_gate_accepts_selection_without_reading_final_metrics() -> None:
    rows = [
        {"document_id": "a", "split": "component_gate", "candidate": {"nll": 1.0, "source_effect": 2.0}, "parent": {"nll": 1.05, "source_effect": 1.0}},
        {"document_id": "b", "split": "component_gate", "candidate": {"nll": 1.0, "source_effect": 2.0}, "parent": {"nll": 1.05, "source_effect": 1.0}},
    ]
    result = document_component_gate(rows, candidate_key="candidate", final_manifest_sha256="final-sha", final_documents=128)
    assert result["status"] == "PASS"
    assert result["final_validation_documents"] == 128
    assert result["final_manifest_sha256"] == "final-sha"


def test_tournament_selects_at_most_one_candidate() -> None:
    rows = [{"document_id": f"{i:02d}", "split": "component_gate", "candidate": {"nll": 1.0, "source_effect": 2.0, "first_token_gold_rank": 1.0}, "parent": {"nll": 1.01, "source_effect": 1.0, "first_token_gold_rank": 2.0}} for i in range(32)]
    def payload(arm: str) -> dict:
        return {"status": "COMPLETE", "examples": 32, "per_document_rows": rows, "retention_4k": {"candidate_minus_native": 0.0}, "comparison_arm": "lora_only_null", "comparison_receipt_sha256": "a" * 64, "parent_receipt_sha256": (arm[0] * 64), "data_manifest_sha256": "d" * 64, "data_root_manifest_sha256": "r" * 64}
    result = select_tournament_winner({arm: payload(arm) for arm in ("native_scale", "context_stretch_exp_negative", "phase_chord")})
    assert result["selected_arm"] == "native_scale"
    assert sum(row["qualified"] for row in result["candidates"]) == 3
    with pytest.raises(ValueError, match="exactly"):
        select_tournament_winner({"native_scale": payload("native_scale")})


def test_stage0_objective_has_ce_margin_and_eos() -> None:
    logits = torch.zeros(1, 3, 8, requires_grad=True)
    answer = torch.tensor([[2, 3, 4]])
    alternate = torch.tensor([[1, 1, 1]])
    loss, metrics = stage0_answer_margin_loss(
        logits=logits,
        answer_tokens=answer,
        alternate_tokens=alternate,
        eos_mask=torch.tensor([[False, False, True]]),
        config=Stage0ObjectiveConfig(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["eos_tokens"] == 1.0
    assert logits.grad is not None


def test_gpu_requires_two_authorization_factors() -> None:
    assert no_cuda_preflight()["cuda_initialized"] is False
    with pytest.raises(PermissionError):
        require_dual_authorization(cli_authorize=True, environment={})
    with pytest.raises(PermissionError):
        require_dual_authorization(cli_authorize=False, environment={"PHASE_ADAROPE_GPU_AUTHORIZED": "1"})


def test_pair_view_binds_real_identifiable_data_shape(tmp_path: Path) -> None:
    root = tmp_path / "component_gate16k"
    root.mkdir()
    ids = np.arange(2 * 2 * 12, dtype=np.uint32).reshape(2, 2, 12)
    positions = np.asarray([[8, 9], [8, 9]], dtype=np.int32)
    tokens = np.stack([np.take_along_axis(ids[:, 0], positions, axis=1), np.take_along_axis(ids[:, 1], positions, axis=1)], axis=1)
    labels = np.full(ids.shape, -100, dtype=np.int32)
    labels[:, :, 8:10] = tokens
    np.save(root / "input_ids.npy", ids)
    np.save(root / "labels.npy", labels)
    np.save(root / "target_positions.npy", positions)
    np.save(root / "target_token_ids.npy", tokens)
    files = {}
    import hashlib
    for name in ("input_ids.npy", "labels.npy", "target_positions.npy", "target_token_ids.npy"):
        path = root / name
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    child = {"status": "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2", "shape": [2, 2, 12], "files": files}
    (root / "manifest.json").write_text(json.dumps(child))
    child_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"pair_views": {root.name: {"manifest_sha256": child_sha}}}))
    view = PairView.load(root)
    assert view.correct.shape == (2, 12)
    assert np.array_equal(view.positions_by_row, positions)
    assert np.array_equal(view.target_tokens[:, 0], ids[:, 0, 8:10])
    (root / "labels.npy").write_bytes((root / "labels.npy").read_bytes() + b"drift")
    with pytest.raises(ValueError, match="manifest-bound file drift"):
        PairView.load(root)


def test_pair_view_accepts_training_only_prompt_stop_sentinel(tmp_path: Path) -> None:
    root = tmp_path / "train4k"
    root.mkdir()
    ids = np.arange(2 * 2 * 12, dtype=np.uint32).reshape(2, 2, 12)
    positions = np.asarray([[8, 9], [8, 9]], dtype=np.int32)
    tokens = np.stack(
        [np.take_along_axis(ids[:, 0], positions, axis=1), np.take_along_axis(ids[:, 1], positions, axis=1)],
        axis=1,
    )
    labels = np.full(ids.shape, -100, dtype=np.int32)
    labels[:, :, 8:10] = tokens
    arrays = {
        "input_ids.npy": ids,
        "labels.npy": labels,
        "target_positions.npy": positions,
        "target_token_ids.npy": tokens,
        "generation_prompt_stops.npy": np.full((2,), -1, dtype=np.int32),
    }
    files = {}
    import hashlib
    for name, value in arrays.items():
        path = root / name
        np.save(path, value)
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "status": "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2",
        "shape": [2, 2, 12],
        "strict_autoregressive_capability": False,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    child_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"pair_views": {root.name: {"manifest_sha256": child_sha}}})
    )
    view = PairView.load(root)
    assert view.generation_prompt_stops is None


def test_selected_hidden_checks_p_minus_one_and_input_tokens() -> None:
    hidden = torch.arange(1 * 5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    input_ids = torch.tensor([[4, 5, 6, 7, 8]])
    positions = torch.tensor([[2, 4]])
    tokens = torch.tensor([[6, 8]])
    head = torch.nn.Linear(3, 9, bias=False)
    selected = selected_hidden_lm_head(hidden, positions, head, input_ids=input_ids, target_tokens=tokens)
    assert selected.shape == (1, 2, 9)
    with pytest.raises(ValueError):
        selected_hidden_lm_head(hidden, positions, head, input_ids=input_ids, target_tokens=torch.tensor([[5, 8]]))


def test_target_loader_requires_manifest_bound_hash(tmp_path: Path) -> None:
    table = np.logspace(0.0, -3.0, 64).astype("<f4")
    digest = __import__("hashlib").sha256(table.tobytes()).hexdigest()
    manifest = tmp_path / "target_manifest.json"
    manifest.write_text(json.dumps({"candidates": {"phase": {"inv_freq": table.tolist(), "inv_freq_float32_sha256": digest}}}))
    loaded = _load_target(manifest, "phase", torch)
    assert loaded.shape == (64,)
    with pytest.raises(ValueError):
        _load_target(manifest, "missing", torch)
