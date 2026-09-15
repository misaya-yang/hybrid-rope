"""CPU contracts for the clone-ready Llama S=16 128K gate."""

import pytest

from experiments.iclr2027_three_track_sprint_20260915.llama_s16_128k_report import (
    TASKS,
    ppl_summary,
    task_summary,
)
from experiments.iclr2027_three_track_sprint_20260915.prepare_llama_s16_ppl10 import (
    select_eligible,
)
from experiments.olmo_recovery_20260912.recovery_v2_eval import resolve_lm_lengths


def test_explicit_128k_lm_length_is_allowed_when_manifest_prepared_it():
    assert resolve_lm_lengths([131072], {"lengths": [131072]}) == (131072,)


def test_explicit_unprepared_lm_length_is_rejected():
    with pytest.raises(ValueError, match="absent"):
        resolve_lm_lengths([131072], {"lengths": [8192, 16384, 32768]})


def test_proofpile_selection_is_deterministic_and_requires_ten():
    assert select_eligible(list(range(12))) == select_eligible(list(range(12)))
    assert len(select_eligible(list(range(12)))) == 10
    with pytest.raises(ValueError, match="need 10"):
        select_eligible(list(range(9)))


def test_s16_report_uses_task_equal_macro_and_token_weighted_nll():
    mapping = {
        f"{task}-{index}": {"task": task, "ruler_official_score": 1.0}
        for task in TASKS for index in range(10)
    }
    assert task_summary(mapping)["macro"] == 1.0
    ppl = ppl_summary([
        {"whole_loss_sum": 20.0, "whole_target_count": 10},
        {"whole_loss_sum": 10.0, "whole_target_count": 10},
    ])
    assert ppl["whole_nll"] == 1.5
