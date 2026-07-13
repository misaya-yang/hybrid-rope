from __future__ import annotations

import math

import pytest

from rebuttal.evq_seed42_retrieval_repair.protocol import (
    decide_gate,
    extract_first_passkey,
    get_stage,
    registered_factor_for_length,
    score_text_answer,
    segment_contract,
)


def _passing_summary(stage: str) -> dict[str, object]:
    return {
        "pair_consistency": 0.80 if stage == "r8" else 0.50,
        "source_removal_positive_fraction": 0.75,
        "passkey_containment": 0.50,
        "temporal_delta_nll": 0.20,
        "finite": True,
        "task_types": ["kv", "update"],
    }


def test_stage_contract_has_fixed_physical_token_budget() -> None:
    r8 = get_stage("r8")
    r16 = get_stage("r16")

    assert (r8.seq_len, r8.accumulation, r8.factor) == (8192, 4, 1.0)
    assert (r16.seq_len, r16.accumulation, r16.factor) == (16384, 2, 2.0)
    assert r8.tokens_per_segment == r16.tokens_per_segment == 1_048_576
    assert segment_contract("r8", 2)["optimizer_state"] == "fresh"
    assert segment_contract("r8", 1)["row_range"] == [0, 128]
    assert segment_contract("r8", 2)["row_range"] == [128, 256]


def test_stage_contract_rejects_unknown_stage_and_segment() -> None:
    with pytest.raises(ValueError, match="expected r8 or r16"):
        get_stage("r32")
    with pytest.raises(ValueError, match="segment must be 1 or 2"):
        segment_contract("r8", 3)


def test_registered_factor_is_tied_to_context_length() -> None:
    assert registered_factor_for_length(8192) == 1.0
    assert registered_factor_for_length(16384) == 2.0
    assert registered_factor_for_length(32768) == 4.0
    with pytest.raises(ValueError, match="registered context length"):
        registered_factor_for_length(4096)


def test_text_metrics_separate_strict_extracted_containment_and_eos() -> None:
    score = score_text_answer("The key is 12345678. Extra.", "12345678", False)

    assert score == {
        "strict_exact": False,
        "first_value_exact": True,
        "gold_containment": True,
        "extracted_value": "12345678",
        "eos_terminated": False,
    }


def test_passkey_extraction_does_not_match_inside_longer_number() -> None:
    assert extract_first_passkey("x 12345678 y 87654321") == "12345678"
    assert extract_first_passkey("9123456780") is None
    score = score_text_answer("9123456780", "12345678", True)
    assert score["gold_containment"] is False
    assert score["first_value_exact"] is False


def test_r8_gate_passes_only_when_all_registered_checks_pass() -> None:
    summary = _passing_summary("r8")
    gate = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)

    assert gate["status"] == "pass"
    assert gate["failed_checks"] == []

    summary["passkey_containment"] = 0.49
    failed = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)
    assert failed["status"] == "rescue_allowed"
    assert "passkey_containment" in failed["failed_checks"]

    no_gain = decide_gate("r8", summary, {"pair_consistency": 0.75}, segment=1)
    assert no_gain["status"] == "stop"


def test_rescue_requires_ten_point_gain_and_is_only_available_after_segment_one() -> None:
    summary = _passing_summary("r16")
    summary.update(
        pair_consistency=0.20,
        source_removal_positive_fraction=0.50,
        passkey_containment=0.20,
        temporal_delta_nll=0.10,
    )
    parent = {"pair_consistency": 0.10}

    assert decide_gate("r16", summary, parent, segment=1)["status"] == "rescue_allowed"
    assert decide_gate("r16", summary, parent, segment=2)["status"] == "stop"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temporal_delta_nll", 0.2000001),
        ("finite", False),
        ("task_types", ["kv"]),
        ("pair_consistency", math.nan),
    ],
)
def test_gate_fails_closed_on_guardrail_or_finite_errors(field: str, value: object) -> None:
    summary = _passing_summary("r8")
    summary[field] = value

    gate = decide_gate("r8", summary, {"pair_consistency": 0.0}, segment=1)

    assert gate["status"] == "stop"
    assert field in gate["failed_checks"] or "finite" in gate["failed_checks"]
