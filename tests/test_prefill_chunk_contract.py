"""CPU checks for prefill chunk benchmark argument handling."""

import pytest

from experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks import (
    parse_chunks,
)
from experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch import (
    select_pair,
)


def test_prefill_chunk_list_is_positive_and_unique():
    assert parse_chunks('0,8192,16384,32768') == [0, 8192, 16384, 32768]
    with pytest.raises(ValueError, match='unique nonnegative'):
        parse_chunks('8192,8192')
    with pytest.raises(ValueError, match='unique nonnegative'):
        parse_chunks('-1,8192')


def test_batch_canary_uses_longest_exact_shape_pair():
    rows = [
        {"row_id": "a", "length_cap": 131072, "prompt_ids": [1] * 10, "max_new_tokens": 8},
        {"row_id": "b", "length_cap": 131072, "prompt_ids": [2] * 10, "max_new_tokens": 8},
        {"row_id": "c", "length_cap": 131072, "prompt_ids": [3] * 20, "max_new_tokens": 8},
        {"row_id": "d", "length_cap": 131072, "prompt_ids": [4] * 20, "max_new_tokens": 8},
        {"row_id": "e", "length_cap": 131072, "prompt_ids": [5] * 20, "max_new_tokens": 16},
    ]
    assert [row["row_id"] for row in select_pair(rows, length=131072)] == ["c", "d"]
    with pytest.raises(ValueError, match="no exact-length pair"):
        select_pair(rows[:1], length=131072)
