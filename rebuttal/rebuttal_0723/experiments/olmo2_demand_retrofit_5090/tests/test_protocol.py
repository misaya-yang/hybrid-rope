from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.dry_run import (
    run_dry_run,
)
from rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.gates import (
    evaluate_4k_gate,
    evaluate_8k_gate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.protocol import (
    ContractError,
    NegativeMorphReproduction,
    QKOnlyProtocol,
    build_protocol_manifest,
)
from rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.slow_residual import (
    bitwise_short_route_gate,
    build_residual_shape,
    route_for_budget,
    select_slow_pairs,
)
from rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.table_adapter import (
    ConditionalCandidateRequired,
    ConditionalTableAdapter,
    TableCollisionError,
)


def _geometric_table(step: float) -> list[float]:
    return [math.exp(-step * index) for index in range(64)]


def test_default_contract_is_disabled_and_seed_matrix_is_planned() -> None:
    protocol = QKOnlyProtocol()
    protocol.validate()
    assert protocol.status == "PROPOSED_NOT_RUN"
    rows = protocol.seed_matrix()
    assert len(rows) == 6
    assert {row["arm"] for row in rows} == {"native", "evq"}
    assert all(row["status"] == "PLANNED_NOT_RUN" for row in rows)
    manifest = build_protocol_manifest(source_manifest={"sources": []})
    assert manifest["execution"]["training_authorized"] is False
    assert manifest["negative_morph_reproduction"]["enabled"] is False
    assert manifest["protected_table"]["enabled"] is False
    assert manifest["slow_residual"]["enabled"] is False


def test_historical_morph_cannot_be_enabled() -> None:
    with pytest.raises(ContractError):
        replace(NegativeMorphReproduction(), enabled=True).validate()


def test_protected_table_rejects_unsafe_same_index_collision() -> None:
    native = _geometric_table(0.2)
    evq = _geometric_table(0.1)
    adapter = ConditionalTableAdapter(
        max_protected_pairs=16,
        gap_fraction_of_native_spacing=0.2,
    )
    with pytest.raises(ConditionalCandidateRequired) as error:
        adapter.apply(
            native_inv_freq=native,
            evq_inv_freq=evq,
            protected_pairs=(1,),
        )
    assert error.value.report.collision_pairs
    assert error.value.report.passed is False


def test_conditional_candidate_collision_is_rejected() -> None:
    native = _geometric_table(0.2)
    candidate = _geometric_table(0.1)
    candidate[2] = native[2]
    adapter = ConditionalTableAdapter()
    with pytest.raises(TableCollisionError) as error:
        adapter.apply(
            native_inv_freq=native,
            evq_inv_freq=_geometric_table(0.1),
            protected_pairs=(2,),
            conditional_candidate=candidate,
        )
    assert error.value.report.passed is False


def test_protected_table_requires_exact_native_retention_and_accepts_candidate() -> None:
    native = _geometric_table(0.2)
    candidate = _geometric_table(0.2)
    adapter = ConditionalTableAdapter(
        max_protected_pairs=16,
        gap_fraction_of_native_spacing=0.2,
    )
    result = adapter.apply(
        native_inv_freq=native,
        evq_inv_freq=_geometric_table(0.1),
        protected_pairs=(2, 7),
        conditional_candidate=candidate,
    )
    assert result.mode == "offline_conditional_candidate"
    assert result.protected_pairs == (2, 7)
    assert result.collision.passed is True
    assert result.table[2] == native[2]
    assert result.table[7] == native[7]


def test_slow_residual_is_low_dimensional_and_has_strict_route_boundary() -> None:
    inv_freq = [math.exp(-0.205 * index) for index in range(64)]
    pairs = select_slow_pairs(
        inv_freq,
        minimum_wavelength_tokens=500_000.0,
        max_pairs=16,
    )
    shape = build_residual_shape(pairs)
    assert len(pairs) <= 16
    assert shape.residual_head_dim <= 32
    assert shape.augmented_cache_head_dim == 128 + shape.residual_head_dim
    assert route_for_budget(4_096) == "native"
    assert route_for_budget(4_097) == "augmented"


def test_short_route_gate_is_byte_exact() -> None:
    assert bitwise_short_route_gate(
        b"native-output",
        b"native-output",
        route="native",
    )["passed"] is True
    assert bitwise_short_route_gate(
        b"native-output",
        b"changed-output",
        route="native",
    )["passed"] is False


def _gate_metrics(
    *,
    f1: float,
    ruler: float,
    nll: float,
    strict: float,
    retention: bool = True,
) -> dict[str, object]:
    return {
        "two_wiki_token_f1": f1,
        "ruler_macro": ruler,
        "natural_nll": nll,
        "strict_autoregressive_score": strict,
        "strict_autoregressive_evaluation": True,
        "independent_retention_pass": retention,
        "ruler_family_scores": {"niah_single_1": 1.0, "fwe": 0.5},
    }


def test_4k_gate_blocks_native_positive_family_collapse() -> None:
    native = _gate_metrics(f1=25.0, ruler=70.0, nll=3.0, strict=0.8)
    candidate = _gate_metrics(f1=24.0, ruler=69.0, nll=3.05, strict=0.7)
    candidate["ruler_family_scores"] = {"niah_single_1": 0.0, "fwe": 0.5}
    result = evaluate_4k_gate(native=native, candidate=candidate)
    assert result.passed is False
    assert any("collapsed to zero" in reason for reason in result.reasons)


def test_8k_gate_runs_only_after_passing_4k() -> None:
    native = _gate_metrics(f1=25.0, ruler=70.0, nll=3.0, strict=0.8)
    candidate = _gate_metrics(f1=24.0, ruler=69.0, nll=3.05, strict=0.9)
    candidate["ruler_family_scores"] = {"niah_single_1": 1.0, "fwe": 0.5}
    four_k = evaluate_4k_gate(native=native, candidate=candidate)
    eight_k = evaluate_8k_gate(
        four_k=four_k,
        native=native,
        candidate=candidate,
    )
    assert four_k.passed is True
    assert eight_k.passed is True


def test_8k_gate_blocks_after_failed_4k() -> None:
    native = _gate_metrics(f1=25.0, ruler=70.0, nll=3.0, strict=0.8)
    candidate = _gate_metrics(f1=24.0, ruler=69.0, nll=3.05, strict=0.9)
    candidate["ruler_family_scores"] = {"niah_single_1": 0.0, "fwe": 0.5}
    four_k = evaluate_4k_gate(native=native, candidate=candidate)
    eight_k = evaluate_8k_gate(
        four_k=four_k,
        native=native,
        candidate=candidate,
    )
    assert four_k.passed is False
    assert eight_k.passed is False
    assert any("4K gate failed" in reason for reason in eight_k.reasons)


def test_dry_run_has_no_external_side_effect_flags(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    config_path = package_root / "config.json"
    source_manifest_path = package_root / "source_manifest.json"
    receipt = run_dry_run(
        config_path=config_path,
        source_manifest_path=source_manifest_path,
    )
    assert receipt["execution_proof"]["training_attempted"] is False
    assert receipt["execution_proof"]["download_attempted"] is False
    assert receipt["execution_proof"]["model_loaded"] is False
    assert receipt["execution_proof"]["optimizer_created"] is False
    assert receipt["method_id"].endswith("r4prime_v1")
    assert len(receipt["config_sha256"]) == 64
    assert len(receipt["source_manifest_sha256"]) == 64
    # The test does not write a receipt unless explicitly asked.
    assert not list(tmp_path.iterdir())
