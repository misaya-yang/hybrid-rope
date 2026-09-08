"""CPU adapter contracts only; the real Transformers wrapper still needs parity."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.native_sparse_position import envelope_native as en


class FakeMeanBase:
    def __init__(self, layer_ids=(0,), block=64, local=128, topk=2):
        self.block, self.local, self.topk = block, local, topk
        self.layers = {i: SimpleNamespace(layer_idx=i, scaling=.25, head_dim=8) for i in layer_ids}
        self.mode, self.pos, self.calls = "prefix", 0, 0

    def reset(self, mode, prefix_length):
        self.mode, self.pos, self.calls = mode, prefix_length, 0

    def interface(self, module, q, k, v, mask, **kwargs):
        reps = q.shape[1] // k.shape[1]
        out = F.scaled_dot_product_attention(q, k.repeat_interleave(reps, 1),
            v.repeat_interleave(reps, 1), dropout_p=0., is_causal=False, scale=module.scaling)
        return out.transpose(1, 2).contiguous(), None


class CPUOracle(en.EnvelopeMixin, FakeMeanBase):
    pass


def fixture(layer_ids=(0,), prefix_length=512, local=128, topk=2):
    generator = torch.Generator().manual_seed(123)
    keys = torch.randn(1, 2, prefix_length, 8, generator=generator)
    centers = (torch.arange(prefix_length) // 64).float() * 10
    keys += centers[None, None, :, None]
    values = torch.randn(keys.shape, generator=generator)
    q = torch.ones(1, 4, 1, 8)
    q[:, 1::2] *= -1
    oracle = CPUOracle(layer_ids=layer_ids, local=local, topk=topk)
    for layer in layer_ids:
        oracle.interface(oracle.layers[layer], q, keys, values, None)
    return oracle, q, keys, values


def continue_inputs(q, k, v, steps=1):
    return torch.cat((k, torch.zeros(*k.shape[:2], steps, k.shape[-1])), 2), \
           torch.cat((v, torch.ones(*v.shape[:2], steps, v.shape[-1])), 2)


@pytest.mark.parametrize("mode", en.SELECTORS)
def test_cached_selection_and_original_reader_without_full_key_float(monkeypatch, mode):
    oracle, q, keys, values = fixture()
    assert oracle.prefix_post[0].data_ptr() == keys.data_ptr()
    info = en.build(oracle, 4)
    assert info["prefix_post_is_reference"]
    oracle.reset(mode, 512)
    k, v = continue_inputs(q, keys, values)
    original_float = torch.Tensor.float
    def guarded_float(tensor, *args, **kwargs):
        if tensor.numel() >= keys.numel():
            raise AssertionError("Continuation attempted to cast all keys")
        return original_float(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "float", guarded_float)
    for name in ("build_pair_envelope", "build_quest"):
        monkeypatch.setattr(en, name, lambda *a, **kw: pytest.fail("Rebuilt cache during continuation"))
    out, _ = oracle.interface(oracle.layers[0], q, k, v, None)
    # Strong per-head center offsets make selected blocks known independently.
    chosen = torch.tensor([[5, 4], [1, 2], [5, 4], [1, 2]])
    remote = (chosen[..., None]*64 + torch.arange(64)).reshape(4, -1)
    mandatory = torch.cat((torch.arange(64), torch.arange(385, 513)))
    ids = torch.cat((remote, mandatory[None].expand(4, -1)), -1).sort(-1).values
    heads = torch.arange(4)//2
    wanted = F.scaled_dot_product_attention(q, k[0, heads[:, None], ids][None],
        v[0, heads[:, None], ids][None], dropout_p=0., is_causal=False, scale=.25)
    torch.testing.assert_close(out, wanted.transpose(1, 2).contiguous(), rtol=0, atol=0)
    assert oracle.calls == 1
    assert oracle.diagnostic_stats["calls"] == 0
    assert not oracle.diagnostic_records


def test_all_key_gather_parity_without_build_and_dense_mode():
    oracle, q, keys, values = fixture(prefix_length=128, local=2048)
    k, v = continue_inputs(q, keys, values)
    outputs = {}
    for mode in ("Dense",) + en.SELECTORS:
        oracle.reset(mode, 128)
        outputs[mode] = oracle.interface(oracle.layers[0], q, k, v, None)[0]
    for mode in en.SELECTORS:
        torch.testing.assert_close(outputs[mode], outputs["Dense"], rtol=0, atol=0)


def test_split32_upper_bound_and_physical_byte_accounting():
    oracle, q, keys, _ = fixture()
    info = oracle.build(K=4)
    oracle.reset("QuestSplit32", 512)
    scores = oracle._cached_scores(0, q[0, :, 0]*.25, "QuestSplit32")
    truth = torch.einsum("hd,hbtd->hbt", q[0, :, 0]*.25,
        keys[0].reshape(2, 8, 64, 8).repeat_interleave(2, 0)).amax(-1)
    assert bool((scores >= truth).all())
    methods = info["layers"][0]["methods"]
    assert methods["Quest"]["bytes_per_physical_kv_block"] == 2*8*4
    assert methods["QuestSplit32"]["bytes_per_physical_kv_block"] == 4*8*4
    assert methods["PairEnvelope"]["total_tensor_bytes"] == methods["RandomPair"]["total_tensor_bytes"]
    assert info["total_resident_descriptor_bytes"] == sum(x["total_tensor_bytes"] for x in methods.values())


def test_diagnostic_position_set_layer_sampling_costs_and_no_selection_change():
    oracle, q, keys, values = fixture(layer_ids=(0, 18, 35))
    en.build(oracle, K=4)
    oracle.reset("RoPEMean", 512)
    assert oracle.diagnostic_layers == [0, 18, 35]
    callbacks = []
    oracle.configure_diagnostics([1, 5], question_last_position=514,
        query_positions=[512, 513], compare_modes=en.SELECTORS,
        callback=callbacks.append, top_competitors=2, row_id="fixture")
    for position in (512, 513, 514):
        k, v = continue_inputs(q, keys, values, steps=position-511)
        oracle.pos = position
        for layer in oracle.layers:
            oracle.interface(oracle.layers[layer], q, k, v, None)
    assert len(callbacks) == 9
    assert oracle.diagnostic_stats["calls"] == 9
    assert oracle.diagnostic_stats["key_bytes_read"] == 9*keys.numel()*keys.element_size()
    assert oracle.diagnostic_stats["fp32_key_materialization_bytes"] == 0
    record = callbacks[0]
    assert "queries_scaled" not in record
    assert record["position"] == 512
    assert set(record["comparisons"]) == set(en.SELECTORS)
    source = record["comparisons"]["Quest"][0]["sources"][1]
    assert source["block"] == 5 and source["selected_remote"]
    assert source["upper_bound_inflation"] >= 0
    assert source["rank_min"] == 1 and source["fully_selected"]
    assert len(oracle.pop_diagnostics()) == 9
    assert not oracle.pop_diagnostics()
    oracle.configure_diagnostics(None)
    oracle.interface(oracle.layers[0], q, k, v, None)
    assert len(callbacks) == 9


def test_reject_stale_prefix_unindexed_remote_blocks_and_excess_diagnostics():
    oracle, q, keys, values = fixture()
    oracle.reset("Quest", 512)
    k, v = continue_inputs(q, keys, values)
    with pytest.raises(RuntimeError, match="built"):
        oracle.interface(oracle.layers[0], q, k, v, None)
    en.build(oracle, K=4)
    with pytest.raises(ValueError, match="five"):
        oracle.configure_diagnostics([1], query_positions=range(6))
    oracle.pos = 767
    long_k, long_v = continue_inputs(q, keys, values, steps=256)
    with pytest.raises(RuntimeError, match="Post-prefix"):
        oracle.interface(oracle.layers[0], q, long_k, long_v, None)
    oracle.mode = "prefix"
    oracle.interface(oracle.layers[0], q, keys, values, None)
    assert not oracle.selector_caches
    oracle.reset("Quest", 512)
    with pytest.raises(RuntimeError, match="built"):
        oracle.interface(oracle.layers[0], q, k, v, None)


def test_method_build_timing_synchronization_and_static_gqa_byte_bounds(monkeypatch):
    oracle, q, keys, _ = fixture()
    sync_calls = []
    monkeypatch.setattr(en, "_sync", lambda tensor: sync_calls.append(tensor.device.type))
    info = oracle.build(K=4)
    assert len(sync_calls) >= 2*len(en.SELECTORS)
    totals = info["per_method_build_seconds"]
    assert set(totals) == set(en.SELECTORS)
    assert all(seconds >= 0 for seconds in totals.values())
    assert info["seconds"] >= sum(totals.values())
    methods = info["layers"][0]["methods"]
    for mode in en.SELECTORS:
        assert methods[mode]["build_seconds"] == totals[mode]
    # H=4, KV=2, G=2; m=2, mandatory=64+128=192; five eligible blocks.
    mean = methods["RoPEMean"]["static_reads_at_first_question_token"]
    token_bytes = 8*(4+4)
    assert mean["selected_kv_bytes_query_head_references"] == 4*(192+2*64)*token_bytes
    assert mean["selected_kv_bytes_unique_gqa_union_lower_bound"] == 2*(192+2*64)*token_bytes
    assert mean["selected_kv_bytes_unique_gqa_union_upper_bound"] == 2*(192+4*64)*token_bytes
    assert not mean["selected_union_exact"]
    assert mean["descriptor_bytes_per_score_query_head_references"] == 4*8*8*4
    assert mean["descriptor_bytes_per_score_unique_union"] == 2*8*8*4
    pair_info = methods["PairEnvelope"]
    pair = pair_info["static_reads_at_first_question_token"]
    assert pair["descriptor_bytes_per_score_query_head_references"] == pair_info["descriptor_bytes"]*2 + pair_info["shared_pair_index_bytes"]*4
    assert pair["descriptor_bytes_per_score_unique_union"] == pair_info["total_tensor_bytes"]
    oracle.reset("PairEnvelope", 512)
    k, v = continue_inputs(q, keys, keys)
    count_before = len(sync_calls)
    oracle.interface(oracle.layers[0], q, k, v, None)
    assert len(sync_calls) == count_before  # No profiler/synchronization added to scoring.
