from __future__ import annotations

import math

import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    analyze_n100_gap_structure as audit,
)


def test_wilson_interval_matches_frozen_21_of_50_cell() -> None:
    lower, upper = audit.wilson_interval(21, 50)

    assert lower == pytest.approx(0.2937500335471198)
    assert upper == pytest.approx(0.5576655823142176)


def test_fisher_exact_matches_frozen_seed_a_gap_split() -> None:
    result = audit.fisher_exact_two_sided(48, 2, 21, 29)

    assert result["odds_ratio"] == pytest.approx(33.142857142857146)
    assert result["two_sided_p"] == pytest.approx(
        2.5590184929430485e-09
    )


def test_exact_mcnemar_matches_seed_discordance() -> None:
    assert audit.exact_mcnemar(6, 4) == pytest.approx(0.75390625)


@pytest.mark.parametrize(
    ("prediction", "official", "expected"),
    (
        ("1234567.", 1.0, "strict_exact"),
        ("12345678.", 1.0, "official_only_wrong_first_number"),
        ("7654321.", 0.0, "wrong_first_number"),
        ("no numeric output", 0.0, "no_number"),
    ),
)
def test_prediction_classification(
    prediction: str,
    official: float,
    expected: str,
) -> None:
    row = {
        "references": ["1234567"],
        "prediction": prediction,
        "official_string_match": official,
    }

    assert audit.classify_prediction(row) == expected


def test_alignment_rejects_row_identity_drift() -> None:
    base = {
        "local_index": 0,
        "row_sha256": "a",
        "references": ["1234567"],
        "source_row_index": 1,
        "source_token_position_answer": 100,
        "input_tokens": 8000,
    }
    drifted = dict(base)
    drifted["row_sha256"] = "b"
    arms = {
        "native": [dict(base) for _ in range(100)],
        "evq_seed_a": [dict(base) for _ in range(100)],
        "evq_seed_b": [dict(base) for _ in range(99)] + [drifted],
    }

    with pytest.raises(ValueError, match="row mismatch"):
        audit.validate_alignment(arms)


def test_hypergeometric_probabilities_are_normalized() -> None:
    row_one = 50
    row_two = 50
    column_one = 69
    lower = max(0, column_one - row_two)
    upper = min(row_one, column_one)
    total = sum(
        audit.hypergeom_probability(
            candidate,
            row_one,
            row_two,
            column_one,
        )
        for candidate in range(lower, upper + 1)
    )

    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)
