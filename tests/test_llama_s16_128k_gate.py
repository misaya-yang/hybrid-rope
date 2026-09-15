"""CPU contracts for the clone-ready Llama S=16 128K gate."""

import pytest

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
