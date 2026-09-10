"""Tiny real HF Qwen2 and exact masked-read tests; no trained-model claim."""
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.pm_keep.balanced_queries import BalancedConfig, BalancedValueSession
from experiments.pm_keep.future_query_probe import (
    evaluate_native_query, evaluate_trace, freeze_prefix_choices, full_free_generation_trace,
)


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(917)
    cfg = Qwen2Config(vocab_size=101, hidden_size=48, intermediate_size=80,
        num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


def session(model):
    cfg = BalancedConfig(samples_per_head=8, horizon=8, sink_tokens=1,
        recent_tokens=2, keep_fraction=.5, query_chunk_size=4, key_chunk_size=7)
    # K fixture is intentionally just a shape-correct prefix-only scorer. It
    # does not claim to test the separately verified author KeyDiff source.
    return BalancedValueSession(model, [1, 7, 7, 7, 8, 9, 9, 10, 11, 12, 13, 14], cfg).prefill(
        {"K": lambda data: data.keys[0].float().norm(dim=-1)})


@torch.inference_mode()
def test_native_mask_gqa_new_keys_and_future_exclusion():
    torch.manual_seed(80)
    q = torch.randn(6, 8)
    k, v = torch.randn(2, 9, 8), torch.randn(2, 9, 8)
    prefix = 5
    keep = {"S": torch.tensor([[0, 2, 4], [1, 3, 4]]),
            "F": torch.arange(prefix).expand(2, -1)}
    result, full = evaluate_native_query(q, k, v, 6, prefix, keep, attention_scale=8**-.5)
    assert result["visible_key_count"] == 7
    assert result["new_visible_key_count"] == 2
    assert result["future_key_count_masked"] == 2
    assert max(result["arms"]["F"]["attention_output_l2_error_per_query_head"]) == 0
    repeated_k, repeated_v = k.repeat_interleave(3, 0), v.repeat_interleave(3, 0)
    expected_p = (q[:, None] @ repeated_k[:, :7].transpose(-1, -2) * 8**-.5).softmax(-1)
    expected_full = (expected_p @ repeated_v[:, :7]).squeeze(1)
    torch.testing.assert_close(full, expected_full, rtol=1e-6, atol=1e-7)
    assert all(x < 1 for x in result["full_prefix_mass_per_query_head"])
    # New keys remain even when their positions were not in the prefix set.
    for h in range(6):
        indices = keep["S"][h // 3].tolist() + [5, 6]
        prob = (q[h] @ repeated_k[h, indices].T * 8**-.5).softmax(-1)
        expected_error = (prob @ repeated_v[h, indices] - expected_full[h]).norm()
        assert result["arms"]["S"]["attention_output_l2_error_per_query_head"][h] == pytest.approx(float(expected_error), abs=3e-7)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 7:] = 1e8
    changed_v[:, 7:] = -1e8
    after, after_full = evaluate_native_query(q, changed_k, changed_v, 6, prefix, keep, attention_scale=8**-.5)
    assert result == after
    torch.testing.assert_close(full, after_full, rtol=0, atol=0)
    with pytest.raises(ValueError, match="original prefix"):
        evaluate_native_query(q, k, v, 6, prefix, {"bad": torch.tensor([[0, 5], [0, 5]])}, attention_scale=8**-.5)


@torch.inference_mode()
def test_tiny_native_queries_match_actual_attention_output_and_free_tokens():
    model = tiny()
    run = session(model)
    keep, proxy = freeze_prefix_choices(run)
    expected = run.branch("F").consume([21, 22, 23]).generate(4, [])
    captured_outputs = {i: {} for i in range(2)}
    current_positions = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def before(module, args, kwargs, index=index):
            current_positions[index] = int(kwargs["cache_position"].item())
        def before_output(module, args, index=index):
            pos = current_positions[index]
            if pos in (14, 15):
                captured_outputs[index][pos] = args[0][0, 0].reshape(6, 8).clone()
        handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(before_output))
    try:
        full, queries, receipt = full_free_generation_trace(run, [21, 22, 23], 4, [])
    finally:
        for h in handles:
            h.remove()
    assert receipt["free_generation"]["generated_ids"] == expected["generated_ids"]
    assert receipt["captured_positions"] == [14, 15]
    assert receipt["final_full_cache_length"] == 18  # prefix12 + suffix3 + 3 ingested generated tokens
    assert full.cache.get_seq_length() == 18
    assert run.cache.get_seq_length() == 12
    for index, by_position in queries.items():
        assert set(by_position) == {14, 15}
        for pos, q in by_position.items():
            result, dense_output = evaluate_native_query(q, full.cache.layers[index].keys[0],
                    full.cache.layers[index].values[0], pos, 12,
                    {arm: sets[index] for arm, sets in keep.items()}, attention_scale=8**-.5)
            torch.testing.assert_close(dense_output, captured_outputs[index][pos], rtol=2e-6, atol=2e-7)
            assert result["arms"]["F"]["mean_output_l2_error"] == 0
    report = evaluate_trace(run, full, queries, keep)
    assert len(report["per_layer_query"]) == 4
    assert all(r["F_max_output_l2_error"] == 0 for r in report["per_position_summary"].values())
    assert proxy["selected_before_question"]


@torch.inference_mode()
def test_choices_and_prefix_do_not_change_with_future_questions():
    model = tiny()
    first = session(model)
    keep, proxy = freeze_prefix_choices(first)
    before = [(layer.keys.clone(), layer.values.clone()) for layer in first.cache.layers]
    full_free_generation_trace(first, [55, 56, 57], 4, [])
    again, same_proxy = freeze_prefix_choices(first)
    assert proxy == same_proxy
    for old, layer in zip(before, first.cache.layers):
        torch.testing.assert_close(old[0], layer.keys, rtol=0, atol=0)
        torch.testing.assert_close(old[1], layer.values, rtol=0, atol=0)
    second = session(model)
    other, other_proxy = freeze_prefix_choices(second)
    full_free_generation_trace(second, [90, 80, 70, 60], 4, [])
    assert other_proxy == proxy
    for arm in keep:
        for a, b, c in zip(keep[arm], again[arm], other[arm]):
            assert torch.equal(a, b) and torch.equal(a, c)


@torch.inference_mode()
def test_generation_cap_does_not_force_extra_first_answer_query():
    run = session(tiny())
    freeze_prefix_choices(run)
    full, queries, receipt = full_free_generation_trace(run, [21, 22], 1, [])
    assert receipt["captured_positions"] == [13]
    assert receipt["uncaptured_positions"] == [14]
    assert len(receipt["free_generation"]["generated_ids"]) == 1
    assert full.cache.get_seq_length() == 14
    assert all(set(layer) == {13} for layer in queries.values())
