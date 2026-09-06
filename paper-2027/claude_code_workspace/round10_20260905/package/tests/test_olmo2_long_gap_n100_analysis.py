import math

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.analyze_long_gap_n100 import (
    exact_mcnemar,
    gap_band,
    paired_summary,
    wilson_interval,
)


def _row(correct: bool) -> dict:
    return {
        "first_number_exact": float(correct),
    }


def test_exact_mcnemar_extreme_one_sided_pairing() -> None:
    assert math.isclose(exact_mcnemar(49, 0), 2.0 / (2**49))
    assert math.isclose(exact_mcnemar(48, 0), 2.0 / (2**48))


def test_seed_pairing_counts_and_agreement() -> None:
    rows_a = [_row(True)] * 39 + [_row(True)] * 10
    rows_a += [_row(False)] * 9 + [_row(False)] * 42
    rows_b = [_row(True)] * 39 + [_row(False)] * 10
    rows_b += [_row(True)] * 9 + [_row(False)] * 42
    summary = paired_summary(rows_a, rows_b)
    assert summary["both_correct"] == 39
    assert summary["arm_a_only"] == 10
    assert summary["arm_b_only"] == 9
    assert summary["both_wrong"] == 42
    assert summary["agreement_rate"] == 0.81


def test_wilson_and_registered_gap_bands() -> None:
    lower, upper = wilson_interval(49, 100)
    assert 0.39 < lower < 0.40
    assert 0.58 < upper < 0.59
    assert gap_band(3934) == "3934-4095"
    assert gap_band(4096) == "4096-5119"
    assert gap_band(6144) == "6144-7167"
    assert gap_band(8013) == "7168+"
