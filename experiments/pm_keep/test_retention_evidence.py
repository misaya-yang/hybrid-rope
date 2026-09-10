import pytest
import torch

from experiments.pm_keep.retention_evidence import inspect_keep_sets, record_spans, token_span


def test_boundary_tokens_overlap_but_zero_width_tokens_do_not():
    offsets = [(0, 0), (0, 3), (3, 6), (6, 10)]
    assert token_span(offsets, 2, 7) == [1, 2, 3]
    with pytest.raises(ValueError, match="fully covered"):
        token_span(offsets, 9, 12)


def test_target_occurrence_and_key_value_are_separate():
    text = "Record key=abc; value=first.\nRecord key=abc; value=second."
    row = {"prefix_text": text, "prefix_ids": list(range(len(text))),
        "query": {"keys": ["abc"], "ordinals": [2]},
        "records": [{"key": "abc", "value": "first"}, {"key": "abc", "value": "second"}]}
    spans = record_spans(row, [(i, i + 1) for i in range(len(text))])
    assert [s["target"] for s in spans] == [False, True]
    value = spans[1]["value_tokens"]
    chosen = torch.tensor([value, value])
    result = inspect_keep_sets({"P": [chosen]}, spans, len(text))["P"][0]["records"]
    assert result[1]["value_tokens"]["all_retained_per_kv_head"] == [True, True]
    assert result[1]["key_tokens"]["retained_per_kv_head"] == [0, 0]
    assert result[0]["value_tokens"]["retained_per_kv_head"] == [0, 0]
    full = torch.arange(len(text))[None]
    result = inspect_keep_sets({"F": [full]}, spans, len(text))["F"][0]["records"]
    assert all(r["record"]["all_retained_per_kv_head"] == [True] for r in result)


def test_refuse_invalid_or_repeated_cache_slots():
    for selected in [torch.tensor([[1, 1]]), torch.tensor([[1, 0]]), torch.tensor([[0, 9]])]:
        with pytest.raises(ValueError):
            inspect_keep_sets({"P": [selected]}, [], 5)
