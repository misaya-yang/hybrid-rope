import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .adapter import PrefillSession
from .key_novel_queries import KeyNovelConfig, KeyNovelSession, novelty_indices, novelty_weights
from .ops import make_sampling_plan


def test_geometric_measure_deemphasizes_repeated_direction_and_handles_zero_mean():
    keys = torch.tensor([[[1., 0.], [1., 0.], [1., 0.], [0., 1.]]])
    weights = novelty_weights(keys, 0)
    assert weights[0, 3] > weights[0, 0]
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=torch.float64))
    symmetric = torch.tensor([[[1., 0.], [-1., 0.]]])
    torch.testing.assert_close(novelty_weights(symmetric, 0), torch.full((1, 2), .5, dtype=torch.float64))
    zeros = torch.zeros(2, 3, 4)
    assert torch.isfinite(novelty_weights(zeros, 1)).all()


def test_same_native_prefix_and_no_future_indices_or_position_changes():
    torch.manual_seed(31)
    keys = torch.randn(2, 16, 8)
    plan = make_sampling_plan(16, 6, 11, horizon=8, seed=1, layer_idx=2, sink_tokens=2)
    first = novelty_indices(keys, plan, sink_tokens=2)
    assert torch.equal(first, novelty_indices(keys, plan, sink_tokens=2))
    assert first.shape == plan.query_indices.shape
    assert int(first.min()) >= 2 and int(first.max()) < 16


@torch.inference_mode()
def test_tiny_full_path_remains_exact_while_proxy_changes():
    torch.set_num_threads(1)
    torch.manual_seed(213)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    cfg = KeyNovelConfig(samples_per_head=8, horizon=8, sink_tokens=1, recent_tokens=1,
                         keep_fraction=.5, query_chunk_size=4, key_chunk_size=7)
    prefix = [4, 5, 6, 4, 5, 6, 7, 8, 4, 5, 6, 9]
    original = PrefillSession(model, prefix, cfg).prefill()
    updated = KeyNovelSession(model, prefix, cfg).prefill()
    torch.testing.assert_close(original.last_logits, updated.last_logits, rtol=0, atol=0)
    for first, second in zip(original.cache.layers, updated.cache.layers):
        torch.testing.assert_close(first.keys, second.keys, rtol=0, atol=0)
        torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)
    for old, new in zip(original.plans, updated.plans):
        assert torch.equal(old.future_positions, new.future_positions)
    first = original.branch("F").consume([11, 12]).generate(4, {2})
    second = updated.branch("F").consume([11, 12]).generate(4, {2})
    assert first["generated_ids"] == second["generated_ids"]
    assert updated.keep_indices("P")[0].shape == (2, 6)
    assert updated.cache.get_seq_length() == len(prefix)
    assert "_key_novel_sampling" not in updated.timings["score_seconds"]
