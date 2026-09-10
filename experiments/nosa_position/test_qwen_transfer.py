"""CPU QKV proofs; optional official-HF hybrid tests never require a GPU.

Enable the latter with RUN_QWEN_HF_TESTS=1 in the existing Transformers 5.15
environment. They construct tiny random models, not task-quality evidence.
"""

import os
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from .qwen_transfer import (QwenTransferAttention, _unmaterialized_causal_mask,
                            configure_qwen_transfer, query_only_topk, retokenize_row)
from .runtime import AttentionSettings, SelectionContext


def settings(**updates):
    values = dict(kernel_size=4, kernel_stride=2, block_size=8, topk=4,
                  init_blocks=1, local_blocks=1, select_blocks=0, attention_query_chunk_size=3)
    values.update(updates)
    return AttentionSettings(**values)


class AttentionInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(310)
        torch.set_num_threads(1)
        self.module = SimpleNamespace(layer_idx=3, training=False)

    def reference(self, q, k, v, scale):
        n, t = k.shape[2], q.shape[2]
        pos = torch.arange(n-t, n)
        mask = torch.arange(n)[None] <= pos[:, None]
        return F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                                               attn_mask=mask[None, None], scale=scale).transpose(1, 2)

    def test_full_support_matches_sdpa_gqa_nondefault_scaling_and_layout(self):
        q, k, v = torch.randn(1, 4, 11, 8), torch.randn(1, 2, 11, 8), torch.randn(1, 2, 11, 8)
        interface = QwenTransferAttention("pc2", settings=settings(topk=8), rotary_dim=4)
        output, weights = interface(self.module, q, k, v, scaling=.17, position_ids=torch.arange(11)[None])
        torch.testing.assert_close(output, self.reference(q, k, v, .17), atol=1e-6, rtol=1e-5)
        self.assertEqual(output.shape, (1, 11, 4, 8))
        self.assertEqual(output.dtype, q.dtype)
        self.assertIsNone(weights)

    def test_cached_query_positions_come_from_complete_prefix(self):
        traces = []
        q, k, v = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 19, 8), torch.randn(1, 2, 19, 8)
        interface = QwenTransferAttention("pc2", settings=settings(topk=8), rotary_dim=4,
                                         trace_callback=lambda ctx, _: traces.append(ctx.query_positions.tolist()))
        output, _ = interface(self.module, q, k, v, scaling=.31, position_ids=torch.tensor([[18]]))
        torch.testing.assert_close(output, self.reference(q, k, v, .31), atol=1e-6, rtol=1e-5)
        self.assertEqual(traces, [[18]])
        with self.assertRaisesRegex(ValueError, "contiguous text positions"):
            interface(self.module, q, k, v, scaling=.31, position_ids=torch.tensor([[0]]))

    def test_partial_rope_qkv_are_not_mutated_or_rotated_again(self):
        q, k, v = torch.randn(1, 4, 7, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)
        before = [x.clone() for x in (q, k, v)]
        trace = []
        interface = QwenTransferAttention("pc2", settings=settings(), rotary_dim=4,
                                         trace_callback=lambda ctx, _: trace.append(ctx))
        interface(self.module, q, k, v, scaling=8**-.5)
        for actual, expected in zip((q, k, v), before):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(trace[0].q, q[0]))
        self.assertTrue(torch.equal(trace[0].k, k[0]))
        self.assertTrue(torch.equal(trace[0].k[..., 4:], k[0, ..., 4:]))

    def test_sparse_full_incremental_interfaces_agree_with_cached_summaries(self):
        q, k, v = torch.randn(1, 4, 51, 8), torch.randn(1, 2, 51, 8), torch.randn(1, 2, 51, 8)
        for mode in ("compressed_mean", "pc2", "cobs_rank1", "split2"):
            full = QwenTransferAttention(mode, settings=settings(), rotary_dim=4)
            expected, _ = full(self.module, q, k, v, scaling=.25)
            incremental = QwenTransferAttention(mode, settings=settings(), rotary_dim=4)
            parts = []
            for i in range(51):
                out, _ = incremental(self.module, q[:, :, i:i+1], k[:, :, :i+1], v[:, :, :i+1],
                                     scaling=.25, position_ids=torch.tensor([[i]]))
                parts.append(out)
            torch.testing.assert_close(torch.cat(parts, 1), expected, atol=2e-6, rtol=2e-5)

    def test_query_only_budget_has_no_cis_tie_fill(self):
        pos = torch.tensor([79])
        context = SelectionContext(torch.zeros(4, 1, 8), torch.zeros(2, 80, 8), torch.zeros(2, 80, 8),
                                   torch.zeros(2, 80), pos, 3, settings(topk=5))
        scores = torch.arange(10).float().expand(2, 1, -1)
        picked = query_only_topk(context, scores)
        self.assertEqual(picked[0, 0].tolist(), [0, 6, 7, 8, 9])

    def test_padding_training_and_insufficient_anchor_budget_fail(self):
        q, k, v = torch.randn(1, 4, 2, 8), torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)
        interface = QwenTransferAttention("pc2", settings=settings(), rotary_dim=4)
        with self.assertRaisesRegex(ValueError, "padding"):
            interface(self.module, q, k, v, torch.zeros(1, 12), scaling=.25)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            interface(SimpleNamespace(layer_idx=3, training=True), q, k, v, scaling=.25)
        with self.assertRaisesRegex(ValueError, "padded masks"):
            _unmaterialized_causal_mask(batch_size=1, attention_mask=torch.tensor([[1, 0]]))
        context = SelectionContext(q[0, :, :1], k[0], v[0], torch.zeros(2, 12), torch.tensor([11]), 3, settings(topk=1))
        with self.assertRaisesRegex(ValueError, "required initial/local"):
            query_only_topk(context, torch.zeros(2, 1, 2))


class RetokenizationTests(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            return "<QWEN>" + messages[-1]["content"]
        def encode(self, text, **kwargs):
            self.encode_kwargs = kwargs
            return list(text.encode())

    def test_uses_qwen_template_and_replaces_foreign_token_ids(self):
        tok = self.Tokenizer()
        row = retokenize_row({"row_id": "a", "prompt_text": "hello", "prompt_ids": [999], "max_new_tokens": 2}, tok)
        self.assertEqual(row["prompt_ids"], list(b"<QWEN>hello"))
        self.assertFalse(tok.kwargs["enable_thinking"])
        self.assertFalse(tok.encode_kwargs["add_special_tokens"])

    def test_foreign_rendered_prompt_or_overflow_is_not_silently_reused(self):
        tok = self.Tokenizer()
        with self.assertRaisesRegex(ValueError, "NOSA"):
            retokenize_row({"prompt": "<user>NOSA chat", "prompt_ids": [1, 2]}, tok)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            retokenize_row({"prompt_text": "long", "length_cap": 5}, tok)


@unittest.skipUnless(os.environ.get("RUN_QWEN_HF_TESTS") == "1", "requires the isolated Transformers 5.15 runtime")
class OfficialHybridIntegrationTests(unittest.TestCase):
    def setUp(self):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        torch.manual_seed(99)
        torch.set_num_threads(2)
        cfg = Qwen3_5TextConfig(vocab_size=43, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, layer_types=["linear_attention"]*3 + ["full_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=4,
            linear_value_head_dim=8, linear_conv_kernel_dim=4,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                             "mrope_section": [1, 1, 0], "mrope_interleaved": True})
        cfg._attn_implementation = "sdpa"
        self.model = Qwen3_5ForCausalLM(cfg).cpu().eval()
        self.ids = torch.randint(4, 43, (1, 19))

    def test_official_gdn_norm_rope_gate_and_weights_are_preserved(self):
        with torch.inference_mode():
            native = self.model(self.ids, use_cache=True).logits
            parameters = {n: p.clone() for n, p in self.model.named_parameters()}
            controller, report = configure_qwen_transfer(self.model, "pc2", settings=settings(topk=64))
            converted = self.model(self.ids, use_cache=True).logits
        self.assertEqual(report["full_attention_indices"], [3])
        self.assertEqual(report["unchanged_gdn_layers"], 3)
        self.assertEqual(report["rotary_dim"], 4)
        self.assertEqual(controller.full_layer_calls, 1)
        torch.testing.assert_close(converted, native, atol=2e-6, rtol=2e-5)
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]))
        configure_qwen_transfer(self.model, "native_full")
        with torch.inference_mode():
            restored = self.model(self.ids, use_cache=True).logits
        torch.testing.assert_close(restored, native)

    def test_official_hybrid_cache_chunked_and_full_logits_agree(self):
        configure_qwen_transfer(self.model, "pc2", settings=settings(topk=64))
        with torch.inference_mode():
            full = self.model(self.ids, use_cache=True).logits
            first = self.model(self.ids[:, :7], use_cache=True)
            second = self.model(self.ids[:, 7:18], use_cache=True, past_key_values=first.past_key_values)
            last = self.model(self.ids[:, 18:], use_cache=True, past_key_values=second.past_key_values)
        actual = torch.cat((first.logits, second.logits, last.logits), 1)
        torch.testing.assert_close(actual, full, atol=5e-6, rtol=5e-5)


if __name__ == "__main__":
    unittest.main()
