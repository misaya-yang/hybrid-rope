"""Real tiny Qwen + pinned author's reconstruction path, without future labels."""
from pathlib import Path
import os

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from .adapter import AdapterConfig, PrefillSession
from .baselines import load_author_class
from .kvzip_reconstruction import prepare_author, reconstruction_session, template_frame


SOURCE = Path(os.environ.get("PM_KEEP_KVPRESS_ROOT", "/tmp/hybrid-kvpress-20260909"))
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason="pinned KVPress source absent")


class TinyTokenizer:
    chat_template = None
    def encode(self, text, return_tensors="pt", add_special_tokens=False):
        assert return_tensors == "pt" and not add_special_tokens
        return torch.tensor([[3 + ord(c) % 80 for c in text]], dtype=torch.long)


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(124)
    cfg = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=2, attention_dropout=0.0)
    cfg._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(cfg).eval()


@torch.inference_mode()
def test_author_score_path_crops_temporary_cache_and_real_gather_generates(monkeypatch):
    model, tok = tiny(), TinyTokenizer()
    prefix = [1, 4, 5, 6, 7, 8, 9, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    cfg = AdapterConfig(samples_per_head=1, horizon=1, sink_tokens=1, recent_tokens=2, keep_fraction=.5)
    original = PrefillSession(model, prefix, cfg).prefill()
    author = load_author_class("KVzipPress", SOURCE)
    actual_score = author.score_kvzip
    calls = []
    def track(self, module, hidden, keys, values, attentions, kwargs):
        calls.append((module.layer_idx, self.start_idx, self.end_idx, kwargs["cache_position"].clone()))
        return actual_score(self, module, hidden, keys, values, attentions, kwargs)
    monkeypatch.setattr(author, "score_kvzip", track)
    run, report = reconstruction_session(model, tok, prefix, keep_fraction=.5, chunk_size=8,
                                          sink_tokens=1, recent_tokens=2, source_root=SOURCE)
    assert len(calls) == 6  # actual author score, two layers * three chunks
    assert report["author_score_calls"] == 6
    assert report["source_copy_tokens_total"] == len(prefix)
    assert report["reconstruction_input_tokens_total"] > len(prefix)
    assert [(c["original_context_start"], c["original_context_end_exclusive"]) for c in report["chunks"]] == [(0, 8), (8, 16), (16, 20)]
    for chunk in report["chunks"]:
        assert chunk["temporary_cache_lengths_after_crop"] == [len(prefix)] * 2
        assert chunk["logical_query_start"] == len(prefix)
    for _, _, _, positions in calls:
        assert int(positions[0]) == len(prefix)
        assert torch.equal(positions, torch.arange(len(prefix), len(prefix) + len(positions)))
    torch.testing.assert_close(original.last_logits, run.last_logits, rtol=0, atol=0)
    for native, preserved in zip(original.cache.layers, run.cache.layers):
        torch.testing.assert_close(native.keys, preserved.keys, rtol=0, atol=0)
        torch.testing.assert_close(native.values, preserved.values, rtol=0, atol=0)
    assert all(torch.isfinite(s).all() and bool((s > 0).any()) for s in run.scores["R"])
    selected = run.keep_indices("R")
    assert all(x.shape == (2, 10) for x in selected)
    branch = run.branch("R", selected)
    assert branch.cache.get_seq_length() == 10
    assert branch.cache.layers[0].keys.data_ptr() != run.cache.layers[0].keys.data_ptr()
    generated = branch.consume([35, 36]).generate(4, [])
    assert len(generated["generated_ids"]) == 4
    assert generated["logical_positions"] == list(range(20, 25))
    assert run.cache.get_seq_length() == 20
    assert all(not hasattr(layer.self_attn, "masked_key_indices") for layer in model.model.layers)
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    # The final Full branch remains exactly the ordinary native path.
    actual_full = run.branch("F").consume([35, 36]).generate(4, [])
    expected_full = original.branch("F").consume([35, 36]).generate(4, [])
    assert actual_full["generated_ids"] == expected_full["generated_ids"]


@torch.inference_mode()
def test_no_future_question_or_label_can_change_frozen_score():
    model, tok = tiny(), TinyTokenizer()
    prefix = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run, _ = reconstruction_session(model, tok, prefix, keep_fraction=.5, chunk_size=7,
                                     sink_tokens=1, recent_tokens=2, source_root=SOURCE)
    before = [x.clone() for x in run.scores["R"]]
    indices = [x.clone() for x in run.keep_indices("R")]
    run.branch("R").consume([60, 61]).generate(3, [])
    run.branch("R").consume([90, 91, 92]).generate(3, [])
    for old, now, left, right in zip(before, run.scores["R"], indices, run.keep_indices("R")):
        torch.testing.assert_close(old, now, rtol=0, atol=0)
        assert torch.equal(left, right)
    assert run.cache.get_seq_length() == len(prefix)


@torch.inference_mode()
def test_native_window_overflow_is_rejected_without_scoring(monkeypatch):
    model = tiny()
    model.config.max_position_embeddings = 30
    author = load_author_class("KVzipPress", SOURCE)
    called = []
    monkeypatch.setattr(author, "score_kvzip", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="native window"):
        reconstruction_session(model, TinyTokenizer(), list(range(1, 21)), keep_fraction=.5,
                               chunk_size=8, sink_tokens=1, recent_tokens=2, source_root=SOURCE)
    assert not called


def test_actual_local_qwen_chat_frame_matches_context_first_prefix():
    root = Path("results/pm_keep_20260909/source/tokenizer")
    if not root.is_dir():
        pytest.skip("local tokenizer absent")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    header, suffix = template_frame(tokenizer)
    text = tokenizer.apply_chat_template([{"role": "user", "content": "Read and remember the complete context below.\n\nCONTEXT\nknown fact"}],
                                        tokenize=False, add_generation_prompt=True)
    full = tokenizer.encode(text, add_special_tokens=False)
    assert full[:header.shape[-1]] == header[0].tolist()
    assert suffix.shape[-1] > 0
