"""Saved-alpha reconstruction and native GQA/value objective checks."""
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.pm_keep.replay_weighted_probe import native_prefix, replay_query, verify_identity
from experiments.pm_keep.run import digest


@torch.inference_mode()
def test_alpha_reconstruction_value_weights_gqa_and_full_retention():
    torch.manual_seed(221)
    q = torch.randn(6, 4, dtype=torch.float64)
    k, v = torch.randn(2, 9, 4, dtype=torch.float64), torch.randn(2, 9, 4, dtype=torch.float64)
    length = 5
    # Two different KV groups and per-Q-head alpha must not be averaged early.
    full_p = (q[:, None] @ k.repeat_interleave(3, 0).transpose(-1, -2) / 2).squeeze(1).softmax(-1)
    alpha = full_p[:, :length].sum(-1)
    sets = {"P": torch.tensor([[0, 2, 4], [0, 1, 3]]), "C": torch.tensor([[1, 3, 4], [1, 2, 4]]),
            "U": torch.tensor([[1, 2, 4], [0, 2, 3]]), "K": torch.tensor([[0, 1, 4], [0, 3, 4]]),
            "F": torch.arange(length).expand(2, -1)}
    audit = replay_query(q, k[:, :length], v[:, :length], alpha, sets, attention_scale=.5)
    norms = v[:, :length].norm(dim=-1).repeat_interleave(3, 0)
    for arm, indices in sets.items():
        repeated = indices.repeat_interleave(3, 0)
        expected_mass = full_p[:, :length].gather(-1, repeated).sum(-1)
        expected = (full_p[:, :length] * norms).gather(-1, repeated).sum(-1)
        torch.testing.assert_close(torch.tensor(audit["arms"][arm]["reconstructed_prefix_mass_per_query_head"], dtype=torch.float64), expected_mass, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.tensor(audit["arms"][arm]["real_mass_times_raw_vnorm_per_query_head"], dtype=torch.float64), expected, rtol=1e-12, atol=1e-12)
        assert audit["arms"][arm]["mean_real_mass_times_raw_vnorm"] == pytest.approx(float(expected.mean()), abs=1e-12)
    assert max(abs(x) for x in audit["arms"]["F"]["missed_weighted_prefix_mass_per_query_head"]) == 0
    zero = replay_query(q, k[:, :length], v[:, :length], torch.zeros(6), sets, attention_scale=.5)
    assert all(x == 0 for x in zero["arms"]["F"]["real_mass_times_raw_vnorm_per_query_head"])


@torch.inference_mode()
def test_native_prefix_uses_one_forward_no_scorer_or_generation():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    c = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2)
    c._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(c).eval()
    ids = [1, 4, 5, 6, 7, 8]
    calls = []
    h = model.register_forward_hook(lambda *args: calls.append(1))
    cache = native_prefix(model, ids)
    h.remove()
    assert len(calls) == 1
    assert all(layer.get_seq_length() == len(ids) for layer in cache.layers)
    expected = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1).past_key_values
    for a, b in zip(cache.layers, expected.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_fixed_identity_input_and_prefix_boundary():
    identity = {"config_sha256": "config", "tokenizer_sha256": "tok", "weights": {"w": 1}}
    row = {"row_id": "x", "prefix_ids": [1, 2], "suffix_ids": [3], "prompt_ids": [1, 2, 3]}
    contract = {"model": identity, "inputs": {"x": {"prompt_sha256": digest(row["prompt_ids"])}}}
    trace = {"prefix_choices": {"prefix_length": 2}}
    verify_identity(contract, identity, row, trace)
    changed = copy.deepcopy(row)
    changed["prompt_ids"] = [1, 2, 4]
    with pytest.raises(ValueError, match="input SHA"):
        verify_identity(contract, identity, changed, trace)
    changed_model = dict(identity, config_sha256="other")
    with pytest.raises(ValueError, match="model identity"):
        verify_identity(contract, changed_model, row, trace)
