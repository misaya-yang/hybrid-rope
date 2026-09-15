"""CPU checks for prefill chunk benchmark argument handling."""

import pytest

from experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks import (
    parse_chunks,
)


def test_prefill_chunk_list_is_positive_and_unique():
    assert parse_chunks('8192,16384,32768') == [8192, 16384, 32768]
    with pytest.raises(ValueError, match='unique positive'):
        parse_chunks('8192,8192')
    with pytest.raises(ValueError, match='unique positive'):
        parse_chunks('0,8192')
