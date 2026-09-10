"""CPU tests exercise actual Transformers Qwen2 projection/cache/attention."""
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from experiments.pm_keep.adapter import AdapterConfig, PrefillSession, cache_nbytes


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    config = Qwen2Config(
        vocab_size=101, hidden_size=48, intermediate_size=80,
        num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def config(**changes):
    defaults = dict(samples_per_head=7, horizon=8, sink_tokens=1,
                    recent_tokens=2, keep_fraction=0.5, query_chunk_size=3,
                    key_chunk_size=7)
    defaults.update(changes)
    return AdapterConfig(**defaults)


def prefix():
    return torch.tensor([[1, 9, 27, 6, 22, 5, 33, 17, 8, 4, 51, 7]])


@torch.inference_mode()
def test_keep_all_original_logits_greedy_and_positions(model):
    ids = prefix()
    suffix = torch.tensor([[13, 26, 4]])
    raw_prefix = model(ids, use_cache=True, logits_to_keep=1)
    session = PrefillSession(model, ids, config(keep_fraction=1.0)).prefill()
    torch.testing.assert_close(session.last_logits, raw_prefix.logits[:, -1], rtol=0, atol=0)
    for original, hooked in zip(raw_prefix.past_key_values.layers, session.cache.layers):
        torch.testing.assert_close(original.keys, hooked.keys, rtol=0, atol=0)
        torch.testing.assert_close(original.values, hooked.values, rtol=0, atol=0)

    branch = session.branch("P")
    raw = raw_prefix
    for token in suffix[0]:
        raw = model(token.view(1, 1), past_key_values=raw.past_key_values,
                    use_cache=True, logits_to_keep=1)
        actual = branch.step(int(token))
        torch.testing.assert_close(actual, raw.logits[:, -1], rtol=1e-6, atol=1e-7)
    full = model(torch.cat((ids, suffix), dim=1), use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(branch.last_logits, full.logits[:, -1], rtol=1e-5, atol=1e-7)

    expected = []
    for i in range(5):
        token = int(raw.logits[0, -1].argmax())
        expected.append(token)
        if token == 2 or i == 4:
            break
        raw = model(torch.tensor([[token]]), past_key_values=raw.past_key_values,
                    use_cache=True, logits_to_keep=1)
    actual = branch.generate(5, [2])
    assert actual["generated_ids"] == expected
    positions = actual["logical_positions"]
    assert positions == list(range(ids.shape[1], ids.shape[1] + len(positions)))
    assert actual["physical_cache_lengths"] == [p + 1 for p in positions]


@torch.inference_mode()
def test_native_rope_and_post_projection_query_capture(model):
    calls = []

    def scorer(data):
        module = data.attention_module
        # A direct projection of the actual post-layernorm attention input must
        # equal the captured query, including each trained projection bias.
        q = module.q_proj(data.hidden_states).view(1, data.prefix_length, 6, 8).transpose(1, 2)
        take = data.sample_indices[None, :, :, None].expand(1, 6, -1, 8)
        torch.testing.assert_close(data.query_samples, q.gather(2, take), rtol=0, atol=0)
        assert data.keys.shape == (1, 2, data.prefix_length, 8)
        assert module.rotary_emb is model.model.rotary_emb
        calls.append(data.layer_idx)
        return data.keys[0].float().norm(dim=-1)

    # Ensure the bias-sensitive check cannot accidentally test only zero bias.
    for layer in model.model.layers:
        layer.self_attn.q_proj.bias.fill_(0.25)
    reference = model(prefix(), use_cache=True, logits_to_keep=1)
    session = PrefillSession(model, prefix(), config()).prefill({"E": scorer})
    torch.testing.assert_close(session.last_logits, reference.logits[:, -1], rtol=0, atol=0)
    assert calls == [0, 1]
    for index, query in enumerate(session.query_samples):
        positions = session.plans[index].future_positions
        # Native shared rotary accepts a batch-like first dimension here.
        cos, sin = model.model.rotary_emb(query[0], positions)
        q = query[0]
        a, b = q.chunk(2, dim=-1)
        expected = q * cos + torch.cat((-b, a), dim=-1) * sin
        torch.testing.assert_close(session.rope.apply(q, positions), expected, rtol=1e-6, atol=1e-6)
    assert all(not hasattr(layer.self_attn, "rotary_emb") for layer in model.model.layers)


@torch.inference_mode()
def test_head_specific_original_gather_and_same_keep_set_path(model):
    session = PrefillSession(model, prefix(), config()).prefill()
    keep = [torch.tensor([[0, 2, 5, 8, 10, 11], [0, 3, 4, 7, 10, 11]]) for _ in session.cache.layers]
    baseline = session.branch("baseline-explicit", keep)
    candidate = session.branch("P", copy.deepcopy(keep))
    for original, compacted, selected in zip(session.cache.layers, baseline.cache.layers, keep):
        assert compacted.keys.shape == (1, 2, 6, 8)
        for head in range(2):
            torch.testing.assert_close(compacted.keys[0, head], original.keys[0, head, selected[head]], rtol=0, atol=0)
            torch.testing.assert_close(compacted.values[0, head], original.values[0, head, selected[head]], rtol=0, atol=0)
        assert compacted.keys.data_ptr() != original.keys.data_ptr()
    assert cache_nbytes(baseline.cache) == cache_nbytes(session.cache) // 2
    for token in (13, 21, 9):
        torch.testing.assert_close(baseline.step(token), candidate.step(token), rtol=0, atol=0)
    assert baseline.generate(4, [2])["generated_ids"] == candidate.generate(4, [2])["generated_ids"]
    assert all(layer.get_seq_length() == 12 for layer in session.cache.layers)
    # Starting another arm must reuse only the frozen prefix, never generated KV.
    fresh = session.branch("F")
    assert fresh.logical_position == 12
    assert fresh.cache.get_seq_length() == 12


@torch.inference_mode()
def test_new_keys_in_denominator_gqa_and_original_logical_position(model):
    session = PrefillSession(model, prefix(), config()).prefill()
    branch = session.branch("P")
    q_outputs, actual_weights = [], []
    attention = model.model.layers[0].self_attn
    handle_q = attention.q_proj.register_forward_hook(lambda m, a, output: q_outputs.append(output.clone()))
    handle_a = attention.register_forward_hook(lambda m, a, output: actual_weights.append(output[1].clone()))
    try:
        branch.step(47)
        branch.step(39)
    finally:
        handle_q.remove()
        handle_a.remove()
    assert branch.logical_positions == [12, 13]
    assert branch.physical_cache_lengths == [7, 8]
    keys = branch.cache.layers[0].keys
    q = q_outputs[-1].view(1, 1, 6, 8).transpose(1, 2)
    cos, sin = model.model.rotary_emb(q, torch.tensor([[13]]))
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    repeated = repeat_kv(keys, 3)
    expected = torch.softmax(q @ repeated.transpose(-1, -2) * attention.scaling,
                             dim=-1, dtype=torch.float32)
    torch.testing.assert_close(actual_weights[-1], expected, rtol=1e-6, atol=1e-7)
    assert expected.shape == (1, 6, 1, 8)
    assert (expected[..., -2:] > 0).all()  # both question keys enter the denominator
    assert (expected[..., :6].sum(-1) < 1).all()


@torch.inference_mode()
def test_questions_cannot_leak_into_scores_or_change_other_branches(model):
    first = PrefillSession(model, prefix(), config()).prefill()
    expected_scores = [s.clone() for s in first.score("P")]
    first_keep = first.keep_indices("P")
    a = first.branch("P").consume([9, 10, 11])
    a.generate(3, [2])
    # A totally different future question is supplied only after a fresh
    # prefix-only selection; the selector interface never receives it.
    second = PrefillSession(model, prefix(), config()).prefill()
    second_keep = second.keep_indices("P")
    second.branch("P").consume([88, 81, 5, 23]).generate(3, [2])
    for before, after, left, right in zip(expected_scores, second.score("P"), first_keep, second_keep):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert torch.equal(left, right)
    assert first.cache.get_seq_length() == second.cache.get_seq_length() == 12


@torch.inference_mode()
def test_horizon_one_controls_equal_on_real_prefix(model):
    session = PrefillSession(model, prefix(), config(horizon=1)).prefill()
    for p, c, u in zip(session.score("P"), session.score("C"), session.score("U")):
        torch.testing.assert_close(p, c, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(p, u, rtol=1e-6, atol=1e-7)
    for p, c in zip(session.keep_indices("P"), session.keep_indices("C")):
        assert torch.equal(p, c)


@torch.inference_mode()
def test_recent_prefix_fallback_uses_same_frozen_samples_for_controls(model):
    session = PrefillSession(model, prefix(), config(
        query_policy="recent_prefix", recent_query_window=3,
    )).prefill()
    for plan in session.plans:
        assert (plan.query_indices >= 9).all()
        assert (plan.query_indices < 12).all()
    samples = [q.clone() for q in session.query_samples]
    for arm in ("P", "C", "U"):
        session.score(arm)
        for old, now in zip(samples, session.query_samples):
            torch.testing.assert_close(old, now, rtol=0, atol=0)
    assert session.branch("P").consume([2, 3]).generate(2, [2])["generated_ids"]


def test_reject_padding_budget_and_unsupported_cache_contract(model):
    with pytest.raises(ValueError, match="one unpadded"):
        PrefillSession(model, prefix().expand(2, -1), config())
    with pytest.raises(ValueError, match="protected"):
        PrefillSession(model, prefix(), config(recent_tokens=9))
    with pytest.raises(ValueError, match="native position"):
        PrefillSession(model, prefix(), config(horizon=256))
    model.config.layer_types[1] = "sliding_attention"
    with pytest.raises(ValueError, match="sliding/recurrent"):
        PrefillSession(model, prefix(), config())


@torch.inference_mode()
def test_failed_callback_restores_model_hooks_and_rotary_binding(model):
    def fail(data):
        raise RuntimeError("injected author scorer failure")

    with pytest.raises(RuntimeError, match="injected"):
        PrefillSession(model, prefix(), config()).prefill({"E": fail})
    for layer in model.model.layers:
        assert not layer.self_attn._forward_hooks
        assert not layer.self_attn.q_proj._forward_hooks
        assert not hasattr(layer.self_attn, "rotary_emb")
    PrefillSession(model, prefix(), config()).prefill()
