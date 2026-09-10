"""Pinned author scores, first-projection origin, native reader and branch checks."""
from pathlib import Path
import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .adapter import AdapterConfig, PrefillSession
from .baselines import keydiff_prefix_scores, load_author_class
from .canonical_keydiff import CanonicalKeyDiffSession, ROW_IDS

SOURCE = Path(os.environ.get("PM_KEEP_KVPRESS_ROOT", "/tmp/hybrid-kvpress-20260909"))
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason="pinned author source absent")


@torch.inference_mode()
def test_native_first_K_author_pre_post_and_F_path_remain_exact():
    torch.set_num_threads(1)
    torch.manual_seed(810)
    cfg = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2, attention_dropout=0.0)
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).eval()
    settings = AdapterConfig(samples_per_head=1, horizon=1, sink_tokens=1, recent_tokens=2, keep_fraction=.5)
    prefix = [1, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    original = PrefillSession(model, prefix, settings).prefill()
    pre_reference, post_reference, originals = {}, {}, {}
    capture_handles = []
    for i, layer in enumerate(model.model.layers):
        def capture(module, args, output, i=i):
            if i not in originals:
                originals[i] = output.clone()
        capture_handles.append(layer.self_attn.k_proj.register_forward_hook(capture))
    author = load_author_class("KeyDiffPress")(compression_ratio=0.0)
    def external(data):
        pre = originals[data.layer_idx].view(1, len(prefix), 2, 8).transpose(1, 2)
        pre_reference[data.layer_idx] = author.score(data.attention_module, data.hidden_states, pre, data.values, None, {})[0].float()
        post_reference[data.layer_idx] = keydiff_prefix_scores(data).float()
        # This later call must never replace the first native projection.
        data.attention_module.k_proj(data.hidden_states + 999)
        return post_reference[data.layer_idx]
    try:
        changed = CanonicalKeyDiffSession(model, prefix, settings).prefill({"external_K": external})
    finally:
        for h in capture_handles:
            h.remove()
    for i in range(2):
        torch.testing.assert_close(changed.scores["K_pre"][i], pre_reference[i], rtol=0, atol=0)
        torch.testing.assert_close(changed.scores["K_post"][i], post_reference[i], rtol=0, atol=0)
        torch.testing.assert_close(changed.cache.layers[i].keys, original.cache.layers[i].keys, rtol=0, atol=0)
        torch.testing.assert_close(changed.cache.layers[i].values, original.cache.layers[i].values, rtol=0, atol=0)
    torch.testing.assert_close(changed.last_logits, original.last_logits, rtol=0, atol=0)
    assert changed.projection_receipt["captured_layers"] == [0, 1]
    left = changed.branch("F").consume([40, 41]).generate(4, [])
    right = original.branch("F").consume([40, 41]).generate(4, [])
    assert left["generated_ids"] == right["generated_ids"]
    chosen = changed.keep_indices("K_pre")
    before = [x.clone() for x in chosen]
    actual = changed.branch("K_pre").consume([50, 51]).generate(4, [])
    assert len(actual["generated_ids"]) == 4
    assert actual["initial_prefix_kv_bytes"] == left["initial_prefix_kv_bytes"] // 2
    assert changed.cache.get_seq_length() == len(prefix)
    assert all(torch.equal(a, b) for a, b in zip(before, changed.keep_indices("K_pre")))
    assert all(not layer.self_attn.k_proj._forward_hooks for layer in model.model.layers)


def test_fixed_three_cell_DEV_panel_has_no_other_inputs():
    assert len(ROW_IDS) == len(set(ROW_IDS)) == 24
    assert all("_dev_" in x for x in ROW_IDS)
    for i in range(8):
        assert sum(f"_{i:03d}_" in x for x in ROW_IDS) == 3
