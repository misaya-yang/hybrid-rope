from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import rebuttal.rebuttal_0723.reviewer27be_shape_base.prepare as prepare_module
import rebuttal.rebuttal_0723.reviewer27be_shape_base.protocol as protocol_module
import rebuttal.rebuttal_0723.reviewer27be_shape_base.run_experiment as run_module
from rebuttal.rebuttal_0723.geo_rope_contract import (
    HISTORICAL_PAPER_GEO_SHA256_FLOAT32,
    std_geo_inv_freq,
)
from rebuttal.rebuttal_0723.reviewer27be_shape_base.prepare import (
    build_manifest,
    choose_disjoint_anchors,
    sha256_file,
    validate_manifest,
)
from rebuttal.rebuttal_0723.reviewer27be_shape_base.protocol import (
    SHAPE_CORE_ARMS,
    SHAPE_REAL_ARMS,
    SHAPE_TAU_ARMS,
    ExperimentSpec,
    SPECS,
    arms_for_suite,
    estimate_parameter_count,
    learning_rate_for_step,
    schedule_phi,
    seeds_for_arm,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.reviewer27be_shape_base.run_experiment import (
    _load_checkpoint,
    _model_inv_freq,
    _save_checkpoint,
    build_model,
    code_fingerprint,
    deterministic_row_order,
    meta_parameter_count,
    tensor_sha256,
)


def test_protocol_budgets_and_parameter_count() -> None:
    shape = SPECS["shape_l128"]
    heldout = SPECS["heldout_b1m_d128"]
    assert shape.optimizer_steps == 457
    assert shape.train_tokens == 14_974_976
    assert shape.global_batch_size == 256
    assert shape.grad_accum_steps == 2
    assert heldout.optimizer_steps == 6_103
    assert heldout.train_tokens == 49_995_776
    assert heldout.head_dim == 128
    assert estimate_parameter_count(shape) == 151_898_880
    assert estimate_parameter_count(heldout) == 151_898_880


@pytest.mark.parametrize("suite", tuple(SPECS))
def test_registered_schedules_are_finite_monotone(suite: str) -> None:
    spec = SPECS[suite]
    for arm in arms_for_suite(suite):
        inv, metadata = training_inv_freq(suite, arm)
        assert inv.shape == (spec.head_dim // 2,)
        assert torch.isfinite(inv).all()
        assert torch.all(torch.diff(inv) < 0)
        assert metadata["family"]
        assert meta_parameter_count(suite, arm) == 151_898_880


def test_shape_controls_match_registered_invariants() -> None:
    target, _ = schedule_phi("shape_l128", "evq_rule")
    uniform, _ = schedule_phi("shape_l128", "uniform_span_matched")
    target_rms = torch.sqrt(torch.mean((target - uniform).square()))
    for arm in ("power_matched", "exp_matched"):
        value, metadata = schedule_phi("shape_l128", arm)
        assert value[0] == pytest.approx(float(target[0]), abs=1e-14)
        assert value[-1] == pytest.approx(float(target[-1]), abs=1e-14)
        achieved = torch.sqrt(torch.mean((value - uniform).square()))
        assert achieved == pytest.approx(float(target_rms), abs=1e-12)
        assert metadata["absolute_match_error"] < 1e-12
        assert not torch.allclose(value, target, atol=1e-8, rtol=1e-8)


def test_real_rope_shapes_match_native_span_and_deformation() -> None:
    geo, _ = schedule_phi("shape_l128", "std_geo")
    evq, _ = schedule_phi("shape_l128", "native_evq_span_rule")
    target_rms = torch.sqrt(torch.mean((evq - geo).square()))
    for arm in SHAPE_REAL_ARMS:
        value, metadata = schedule_phi("shape_l128", arm)
        assert value[0] == pytest.approx(float(geo[0]), abs=1e-14)
        assert value[-1] == pytest.approx(float(geo[-1]), abs=1e-14)
        achieved = torch.sqrt(torch.mean((value - geo).square()))
        assert achieved == pytest.approx(float(target_rms), abs=1e-12)
        assert metadata["grid"] == "native Std-RoPE endpoint/span matched"
        assert torch.all(torch.diff(value) > 0)


def test_tau_scan_and_seed_scope_are_separate() -> None:
    assert set(SHAPE_CORE_ARMS).issubset(arms_for_suite("shape_l128"))
    assert set(SHAPE_TAU_ARMS).issubset(arms_for_suite("shape_l128"))
    assert seeds_for_arm("shape_l128", "evq_tau4") == (42,)
    assert seeds_for_arm("shape_l128", "power_matched") == (42, 137, 256)
    assert seeds_for_arm("shape_l128", "std_geo") == (42, 137, 256)
    for arm in SHAPE_REAL_ARMS:
        assert seeds_for_arm("shape_l128", arm) == (42, 137, 256)
    assert seeds_for_arm("heldout_b1m_d128", "paper_geo") == (42, 137, 256)


def test_named_geo_definitions_and_historical_hash() -> None:
    std, std_meta = training_inv_freq("shape_l128", "std_geo")
    paper, paper_meta = training_inv_freq("shape_l128", "paper_geo")
    assert std_meta["method_identity"] == "Std-Geo"
    assert paper_meta["method_identity"] == "Paper-Geo"
    assert tensor_sha256(paper) == HISTORICAL_PAPER_GEO_SHA256_FLOAT32
    assert float(std[0]) == 1.0
    assert float(paper[0]) < float(std[0])
    ratio = paper / std
    assert torch.allclose(
        ratio,
        torch.full_like(ratio, ratio[0]),
        atol=1e-7,
        rtol=1e-7,
    )


def test_initialization_order_and_lr_are_deterministic() -> None:
    first = deterministic_row_order(1_000, 42)
    second = deterministic_row_order(1_000, 42)
    other = deterministic_row_order(1_000, 137)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    for spec in SPECS.values():
        values = [
            learning_rate_for_step(step, spec)
            for step in range(spec.optimizer_steps)
        ]
        assert values[0] > 0
        assert max(values) <= spec.learning_rate
        assert values[-1] == pytest.approx(spec.min_learning_rate)


def test_disjoint_anchor_generation() -> None:
    anchors = choose_disjoint_anchors(
        2_000_000, count=48, max_length=16_384, seed=123
    )
    assert anchors.dtype == np.int64
    assert len(anchors) == 48
    assert np.all(np.diff(anchors) >= 16_384)
    assert np.array_equal(
        anchors,
        choose_disjoint_anchors(
            2_000_000, count=48, max_length=16_384, seed=123
        ),
    )


def test_manifest_derivation_and_validation(tmp_path: Path) -> None:
    # Temporarily shrink protocol demands while exercising all receipt logic.
    # build_manifest itself uses SPECS, so allocate the real 50M-token prefix
    # sparsely: NPY creation writes ~400 MB and is too large for a unit test.
    # Instead validate a hand-built minimal manifest's rejection path here and
    # leave the full derivation to the server CPU preflight.
    train = tmp_path / "train.npy"
    validation = tmp_path / "validation.npy"
    np.save(train, np.arange(1_024, dtype=np.int64))
    np.save(validation, np.arange(32_768, dtype=np.int64))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {"path": str(train)},
                "validation": {"path": str(validation)},
            }
        )
    )
    with pytest.raises(ValueError, match="required"):
        build_manifest(source, tmp_path / "derived")
    assert sha256_file(train)


def test_full_small_manifest_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_specs = {
        "shape_l128": ExperimentSpec(
            name="shape_l128",
            train_length=8,
            requested_train_tokens=512,
            global_batch_size=8,
            micro_batch_size=4,
            head_dim=8,
            num_heads=2,
            rope_base=500_000.0,
            eval_lengths=(8, 16),
            eval_tail_tokens=4,
            hidden_size=16,
            num_layers=1,
            intermediate_size=32,
            vocab_size=256,
            selection_anchor_count=2,
            test_anchor_count=4,
        ),
        "heldout_b1m_d128": ExperimentSpec(
            name="heldout_b1m_d128",
            train_length=16,
            requested_train_tokens=1_024,
            global_batch_size=8,
            micro_batch_size=4,
            head_dim=8,
            num_heads=2,
            rope_base=1_000_000.0,
            eval_lengths=(16,),
            eval_tail_tokens=4,
            hidden_size=16,
            num_layers=1,
            intermediate_size=32,
            vocab_size=256,
            selection_anchor_count=2,
            test_anchor_count=4,
        ),
    }
    monkeypatch.setattr(prepare_module, "SPECS", tiny_specs)
    train = tmp_path / "train.npy"
    validation = tmp_path / "validation.npy"
    np.save(train, np.arange(2_048, dtype=np.int64) % 256)
    np.save(validation, np.arange(2_000, dtype=np.int64) % 256)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {"path": str(train)},
                "validation": {"path": str(validation)},
            }
        )
    )
    manifest = prepare_module.build_manifest(source, tmp_path / "derived")
    prepare_module.validate_manifest(manifest, check_hashes=True)
    assert manifest["selection_anchors"]["count"] == 2
    assert manifest["test_anchors"]["count"] == 4
    assert Path(manifest["train"]["path"]).is_file()
    assert Path(manifest["validation"]["path"]).is_file()


def test_validate_manifest_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        validate_manifest({"schema_version": 999}, check_hashes=False)


def test_checkpoint_roundtrip_loads_saved_inv_freq_not_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = ExperimentSpec(
        name="shape_l128",
        train_length=8,
        requested_train_tokens=512,
        global_batch_size=8,
        micro_batch_size=4,
        head_dim=8,
        num_heads=2,
        rope_base=500_000.0,
        eval_lengths=(8, 16),
        eval_tail_tokens=4,
        hidden_size=16,
        num_layers=2,
        intermediate_size=32,
        vocab_size=256,
        selection_anchor_count=2,
        test_anchor_count=4,
    )
    tiny_specs = {"shape_l128": tiny}
    monkeypatch.setattr(protocol_module, "SPECS", tiny_specs)
    monkeypatch.setattr(run_module, "SPECS", tiny_specs)

    suite, arm, seed = "shape_l128", "paper_geo", 42
    run_dir = tmp_path / "runs" / suite / arm / f"seed{seed}"
    run_dir.mkdir(parents=True)
    model = build_model(suite, arm, seed)
    expected = _model_inv_freq(model)
    metadata = {
        "suite": suite,
        "arm": arm,
        "seed": seed,
        "protocol_sha256": tiny.fingerprint(),
        "code_sha256": code_fingerprint(),
        "inv_freq_sha256": tensor_sha256(expected),
    }
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint_sha256 = _save_checkpoint(checkpoint, model, metadata)
    (run_dir / "train_result.json").write_text(
        json.dumps(
            {
                **metadata,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    )
    np.save(run_dir / "inv_freq.npy", expected.numpy())

    def wrong_constructor(
        selected_suite: str, selected_arm: str, selected_seed: int
    ):
        del selected_suite, selected_arm, selected_seed
        return run_module.GPT(
            tiny.model_config(),
            std_geo_inv_freq(tiny.head_dim, tiny.rope_base),
        )

    monkeypatch.setattr(run_module, "build_model", wrong_constructor)
    loaded, _ = _load_checkpoint(checkpoint, suite, arm, seed)
    assert torch.equal(_model_inv_freq(loaded), expected)
