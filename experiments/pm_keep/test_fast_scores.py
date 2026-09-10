"""CPU execution-equivalence tests, not pretrained-model quality evidence."""
import pytest
import torch

from experiments.pm_keep.fast_scores import attention_mean_batched, execution_plan
from experiments.pm_keep.ops import attention_mean, select_fixed_budget


@pytest.mark.parametrize("heads,kv_heads,samples,length,dim,batch,amplitude", [
    (6, 2, 17, 53, 16, 13, 1.0),  # partial batch crosses Q-head boundaries
    (8, 1, 7, 61, 8, 256, 1.0),   # MQA, effective batch smaller than requested
    (4, 4, 11, 43, 8, 7, 1.0),    # ordinary MHA
    (4, 2, 19, 59, 8, 9, 15.0),   # sharp logits, stable global normalization
])
def test_reference_scores_budget_and_inputs(heads, kv_heads, samples, length, dim, batch, amplitude):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(183)
    queries = torch.randn(heads, samples, dim, generator=generator) * amplitude
    keys = torch.randn(kv_heads, length, dim, generator=generator) * amplitude
    q_before, k_before = queries.clone(), keys.clone()
    reference = attention_mean(queries, keys, query_chunk_size=5, key_chunk_size=11)
    actual = attention_mean_batched(queries, keys, query_batch_size=batch)
    torch.testing.assert_close(actual, reference, atol=1e-7, rtol=2e-5)
    torch.testing.assert_close(actual.sum(-1), torch.ones(kv_heads), atol=2e-7, rtol=0)
    assert torch.equal(queries, q_before) and torch.equal(keys, k_before)
    reference_keep = select_fixed_budget(reference, length//3, sink_tokens=1, recent_tokens=2)
    actual_keep = select_fixed_budget(actual, length//3, sink_tokens=1, recent_tokens=2)
    assert torch.equal(reference_keep, actual_keep)


def test_noncontiguous_bfloat16_and_outer_autocast_still_fp32():
    generator = torch.Generator().manual_seed(25)
    queries = torch.randn(6, 15, 24, generator=generator).bfloat16()[..., ::2]
    keys = torch.randn(2, 47, 24, generator=generator).bfloat16()[..., ::2]
    reference = attention_mean(queries, keys, attention_scale=0.17, query_chunk_size=4, key_chunk_size=7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = attention_mean_batched(queries, keys, attention_scale=0.17, query_batch_size=8)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-7)


def test_protected_keys_remain_in_softmax_denominator():
    queries = torch.ones(4, 5, 2)
    keys = torch.zeros(2, 13, 2)
    keys[:, 0] = 9.0
    actual = attention_mean_batched(queries, keys, query_batch_size=3)
    assert (actual[:, 0] > 0.9999).all()
    assert (actual[:, 1:].sum(-1) < 0.0001).all()
    kept = select_fixed_budget(actual, 5, sink_tokens=1, recent_tokens=2)
    assert all({0, 11, 12}.issubset(set(row.tolist())) for row in kept)


def test_plan_reports_workspace_and_no_kv_replication():
    plan = execution_plan((16, 256, 128), (2, 11589, 128), 256)
    assert plan["queries_per_kv_head"] == 2048
    assert plan["query_batches"] == 8
    assert plan["logits_tile_bytes"] == 2*256*11589*4
    assert not plan["key_heads_repeated"]
    assert plan["temporary_tensor_bytes_upper_estimate"] > 2*plan["logits_tile_bytes"]


def test_invalid_grouping_or_batch_rejected():
    with pytest.raises(ValueError):
        execution_plan((5, 7, 8), (2, 13, 8))
    with pytest.raises(ValueError):
        attention_mean_batched(torch.randn(4, 7, 8), torch.randn(2, 13, 8), query_batch_size=0)
