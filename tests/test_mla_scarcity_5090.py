from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import rebuttal.rebuttal_0723.mla_scarcity_5090.run_experiment as run_module
from rebuttal.rebuttal_0723.mla_scarcity_5090.analyze_schedules import (
    build_diagnostics,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.prepare import (
    choose_disjoint_anchor_endpoints,
    sha256_file,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.protocol import (
    ARMS,
    FREQUENCY_PAIRS,
    SEEDS,
    SPEC,
    learning_rate_for_step,
    schedule_phi,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.run_experiment import (
    _runtime_operator,
    build_model,
    checkpoint_storage_budget,
    cleanup_checkpoints,
    code_fingerprint,
    gate,
    run_preflight,
    summarize,
    summarize_yarn,
    trainable_state_sha256,
    validate_compile_cache,
)
from scripts.core_text_phases.run_evq_sweep import apply_rope


def test_fixed_architecture_changes_only_active_frequency_budget() -> None:
    parameter_counts = set()
    initialization_hashes = set()
    for pairs in FREQUENCY_PAIRS:
        model = build_model(pairs, "native_geo", 42)
        parameter_counts.add(sum(p.numel() for p in model.parameters()))
        initialization_hashes.add(trainable_state_sha256(model))
        assert model.blocks[0].attn.d_rope == 64
        assert model.blocks[0].attn.d_nope == 0
        for arm in ARMS:
            inv, metadata = training_inv_freq(arm, pairs)
            assert inv.shape == (SPEC.rotary_pair_capacity,)
            assert torch.all(torch.diff(inv[:pairs]) < 0)
            assert torch.equal(
                inv[pairs:],
                torch.zeros(SPEC.rotary_pair_capacity - pairs),
            )
            assert metadata["inactive_identity_pairs"] == 32 - pairs
    assert len(parameter_counts) == 1
    assert len(initialization_hashes) == 1


def test_inactive_pairs_are_exact_identity_rotations() -> None:
    model = build_model(8, "native_geo", 42)
    rope = model.blocks[0].attn.rope
    cos, sin = rope(7)
    inactive = list(range(8, 32)) + list(range(40, 64))
    assert torch.equal(cos[:, inactive], torch.ones_like(cos[:, inactive]))
    assert torch.equal(sin[:, inactive], torch.zeros_like(sin[:, inactive]))
    value = torch.randn(1, 2, 7, 64)
    rotated = apply_rope(
        value, cos[None, None], sin[None, None]
    )
    assert torch.equal(rotated[..., inactive], value[..., inactive])


def test_real_model_forward_backward_is_finite() -> None:
    model = build_model(8, "evq_cosh", 42)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits = model(tokens[:, :-1])
    assert logits.shape == (1, 3, SPEC.vocab_size)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.emb.weight.grad).all()
    assert torch.isfinite(
        model.blocks[0].attn.k_rope_proj.weight.grad
    ).all()


def test_range_control_matches_evq_endpoints_not_interior() -> None:
    for pairs in FREQUENCY_PAIRS:
        evq, _ = schedule_phi("evq_cosh", pairs)
        control, _ = schedule_phi("range_matched_uniform", pairs)
        assert torch.equal(evq[[0, -1]], control[[0, -1]])
        assert not torch.equal(evq[1:-1], control[1:-1])


def test_model_free_proxy_predeclares_non_monotonic_risk() -> None:
    diagnostic = build_diagnostics()
    proxy = diagnostic["scarcity_proxy"]
    assert diagnostic["status"] == "MODEL_FREE_DIAGNOSTIC_ONLY"
    assert proxy["one_to_two_x"]["k8_range_minus_evq_rms"] > 0
    assert proxy["one_to_two_x"]["k8_minus_k32"] > 0
    assert proxy["two_to_four_x"]["k8_range_minus_evq_rms"] < 0
    assert "not an LM loss" in diagnostic["definition"]


def test_runtime_operator_preserves_inactive_identity_pairs() -> None:
    base, _ = training_inv_freq("native_geo", 8, dtype=torch.float64)
    raw, raw_mscale, _ = _runtime_operator(
        base,
        frequency_pairs=8,
        arm="native_geo",
        length=8192,
        operator="raw",
    )
    yarn, yarn_mscale, metadata = _runtime_operator(
        base,
        frequency_pairs=8,
        arm="native_geo",
        length=8192,
        operator="yarn_full",
    )
    assert torch.equal(raw, base)
    assert raw_mscale == 1.0
    assert torch.all(yarn[:8] > 0)
    assert torch.equal(yarn[8:], torch.zeros(24, dtype=torch.float64))
    assert yarn_mscale > 1.0
    assert metadata["mode"] == "official_yarn_native"


def test_protocol_token_budget_and_lr_endpoints() -> None:
    assert SPEC.train_tokens <= SPEC.requested_train_tokens
    assert (
        SPEC.requested_train_tokens - SPEC.train_tokens
        < SPEC.tokens_per_optimizer_step
    )
    assert SPEC.checkpoint_steps["100m"] < SPEC.checkpoint_steps["200m"]
    assert SPEC.checkpoint_steps["200m"] < SPEC.checkpoint_steps["300m"]
    assert learning_rate_for_step(0) > 0
    assert learning_rate_for_step(SPEC.optimizer_steps - 1) == pytest.approx(
        SPEC.min_learning_rate
    )


def test_compile_cache_must_be_registered_and_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "work"
    cache = work_dir / "torchinductor_cache"
    cache.mkdir(parents=True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    result = validate_compile_cache(work_dir)
    assert result["path"] == str(cache.resolve())
    assert result["shared_across_runs"] is True
    monkeypatch.setenv(
        "TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "wrong-cache")
    )
    with pytest.raises(RuntimeError, match="registered shared path"):
        validate_compile_cache(work_dir)


def test_checkpoint_storage_budget_fits_free_space_floor() -> None:
    model = build_model(32, "native_geo", 42)
    count = sum(parameter.numel() for parameter in model.parameters())
    budget = checkpoint_storage_budget(count)
    assert budget["maximum_simultaneous_checkpoints"] == 15
    assert budget["confirmatory_terminal_checkpoint_count"] == 0
    assert budget["checkpoint_peak_upper_bound_bytes"] < 4 * 2**30
    assert budget["reserve_after_checkpoint_peak_bytes"] > 4 * 2**30


def test_preflight_receipt_includes_storage_and_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "work"
    cache = work_dir / "torchinductor_cache"
    cache.mkdir(parents=True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    diagnostic_path = work_dir / "schedule_diagnostics.json"
    diagnostic_path.write_text(json.dumps(build_diagnostics()))
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text("{}")
    manifest = {
        "train": {
            "sha256": "train",
            "token_prefix_sha256": "prefix",
        },
        "validation": {"sha256": "validation"},
        "selection_anchors": {"sha256": "selection"},
        "test_anchors": {"sha256": "test"},
    }
    monkeypatch.setattr(
        run_module, "_load_manifest", lambda *args, **kwargs: manifest
    )
    args = argparse.Namespace(
        data_manifest=manifest_path,
        work_dir=work_dir,
        full_hash_check=True,
        prefix_hash_check=True,
        verify_full_initialization=True,
    )
    result = run_preflight(args)
    assert result["status"] == "READY"
    assert result["schedule_diagnostic"]["status"] == (
        "MODEL_FREE_DIAGNOSTIC_ONLY"
    )
    assert result["storage"]["budget"][
        "maximum_simultaneous_checkpoints"
    ] == 15
    assert result["compile_cache"]["shared_across_runs"] is True


def test_disjoint_anchor_generation_is_deterministic() -> None:
    kwargs = {
        "validation_tokens": 2_000_000,
        "count": 48,
        "max_length": 32_768,
        "seed": 123,
    }
    first = choose_disjoint_anchor_endpoints(**kwargs)
    second = choose_disjoint_anchor_endpoints(**kwargs)
    assert np.array_equal(first, second)
    assert np.all(np.diff(first) >= 32_768)


def _write_eval(
    work_dir: Path,
    *,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
    operator: str,
    nll: float,
    checkpoint_sha: str = "synthetic",
) -> Path:
    run_dir = work_dir / "runs" / f"k{pairs}" / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"eval_{split}_{stage}_{operator}.json"
    summary = {
        str(length): {
            "tail_nll_mean": nll,
            "full_nll_mean": nll,
        }
        for length in SPEC.eval_lengths
    }
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "frequency_pairs": pairs,
                "arm": arm,
                "seed": seed,
                "stage": stage,
                "split": split,
                "operator": operator,
                "protocol_sha256": SPEC.fingerprint(),
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_code_sha256": code_fingerprint(),
                "training_code_sha256": code_fingerprint(),
                "summary": summary,
            }
        )
    )
    return path


def _populate_raw_effects(
    work_dir: Path, *, split: str, seeds: tuple[int, ...]
) -> None:
    values = {
        8: {
            "native_geo": 5.00,
            "range_matched_uniform": 4.90,
            "evq_cosh": 4.70,
        },
        32: {
            "native_geo": 5.00,
            "range_matched_uniform": 4.90,
            "evq_cosh": 4.85,
        },
    }
    for stage in ("200m", "300m"):
        for seed in seeds:
            for pairs in FREQUENCY_PAIRS:
                for arm in ARMS:
                    _write_eval(
                        work_dir,
                        pairs=pairs,
                        arm=arm,
                        seed=seed,
                        split=split,
                        stage=stage,
                        operator="raw",
                        nll=values[pairs][arm],
                    )


def test_seed42_gate_and_three_seed_summary(tmp_path: Path) -> None:
    _populate_raw_effects(tmp_path, split="selection", seeds=(42,))
    gate_result = gate(tmp_path)
    assert gate_result["status"] == "PASS"
    assert gate_result["test_split_read"] is False

    _populate_raw_effects(tmp_path, split="test", seeds=SEEDS)
    result = summarize(tmp_path)
    assert result["claim_gate"] == "SUPPORTS_SCARCITY_CLAIM"
    assert all(result["criteria"].values())
    assert result["primary_300m_2x"]["scarcity_interaction"]["mean"] == (
        pytest.approx(0.15)
    )


def test_yarn_secondary_summary_keeps_identity_boundary(tmp_path: Path) -> None:
    (tmp_path / "summary_mla_scarcity.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_sha256": SPEC.fingerprint(),
                "code_sha256": code_fingerprint(),
            }
        )
    )
    for seed in SEEDS:
        for pairs in FREQUENCY_PAIRS:
            for arm, raw_nll, yarn_nll in (
                ("native_geo", 5.0, 4.5),
                ("evq_cosh", 4.7, 4.1),
            ):
                _write_eval(
                    tmp_path,
                    pairs=pairs,
                    arm=arm,
                    seed=seed,
                    split="test",
                    stage="300m",
                    operator="raw",
                    nll=raw_nll,
                )
                _write_eval(
                    tmp_path,
                    pairs=pairs,
                    arm=arm,
                    seed=seed,
                    split="test",
                    stage="300m",
                    operator="yarn_full",
                    nll=yarn_nll,
                )
    result = summarize_yarn(tmp_path)
    assert result["status"] == "PASS"
    assert "not official-YaRN parity" in result["interpretation_boundary"]
    assert result["aggregates"][0][
        "diagnostic_increment_over_raw"
    ]["mean"] == pytest.approx(0.1)


def test_cleanup_requires_evaluation_proof_and_retains_json(
    tmp_path: Path,
) -> None:
    pairs, arm, seed = 8, "native_geo", 42
    run_dir = tmp_path / "runs" / "k8" / arm / "seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "train_result.json").write_text(
        json.dumps(
            {"status": "PASS", "protocol_sha256": SPEC.fingerprint()}
        )
    )
    for stage in ("200m", "300m"):
        checkpoint = run_dir / f"checkpoint_{stage}.pt"
        checkpoint.write_bytes((stage * 100).encode())
        checkpoint_sha = sha256_file(checkpoint)
        (run_dir / f"checkpoint_{stage}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "checkpoint_label": stage,
                    "checkpoint_sha256": checkpoint_sha,
                    "protocol_sha256": SPEC.fingerprint(),
                    "code_sha256": code_fingerprint(),
                }
            )
        )
        _write_eval(
            tmp_path,
            pairs=pairs,
            arm=arm,
            seed=seed,
            split="test",
            stage=stage,
            operator="raw",
            nll=4.0,
            checkpoint_sha=checkpoint_sha,
        )
    args = argparse.Namespace(
        frequency_pairs=pairs,
        arm=arm,
        seed=seed,
        work_dir=tmp_path,
        proof_split="test",
        operator="raw",
        proof_stages=("200m", "300m"),
        drop_unclaimed_100m=False,
    )
    result = cleanup_checkpoints(args)
    assert result["status"] == "PASS"
    assert not list(run_dir.glob("checkpoint_*.pt"))
    assert len(list(run_dir.glob("eval_*.json"))) == 2


def test_cleanup_resumes_after_interrupted_delete(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "k8" / "native_geo" / "seed42"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint_200m.pt"
    checkpoint.write_bytes(b"checkpoint")
    incomplete = run_dir / "stale.incomplete"
    incomplete.write_bytes(b"partial")
    receipt = run_dir / "cleanup_test_raw.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "DELETING",
                "checkpoint_targets": [
                    {
                        "path": str(checkpoint),
                        "sha256": sha256_file(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                    }
                ],
                "incomplete_targets": [
                    {
                        "path": str(incomplete),
                        "bytes": incomplete.stat().st_size,
                    }
                ],
            }
        )
    )
    args = argparse.Namespace(
        frequency_pairs=8,
        arm="native_geo",
        seed=42,
        work_dir=tmp_path,
        proof_split="test",
        operator="raw",
        proof_stages=("200m",),
        drop_unclaimed_100m=False,
    )
    result = cleanup_checkpoints(args)
    assert result["status"] == "PASS"
    assert result["cleanup_resumed"] is True
    assert not checkpoint.exists()
    assert not incomplete.exists()


def test_code_fingerprint_is_complete() -> None:
    assert len(code_fingerprint()) == 64
