"""CPU proofs of the NOSA reference operator; these are not CUDA parity tests."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from experiments.nosa_position.runtime import (
    AttentionSettings, NosaReferenceForCausalLM, SelectionContext, apply_rope,
    cis_scores, dense_causal_attention, mandatory_blocks, native_scores,
    native_select, rope_parameters, selected_causal_attention, select_with_scores,
)


def tiny_config(**updates):
    cfg = dict(vocab_size=41, hidden_size=32, intermediate_size=48,
               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               num_hidden_layers=2, rms_norm_eps=1e-6, rope_theta=10000.0,
               eos_token_id=[2, 3], tie_word_embeddings=False)
    cfg.update(updates)
    return cfg


def tiny_settings(**updates):
    cfg = dict(kernel_size=4, kernel_stride=2, block_size=8,
               init_blocks=1, local_blocks=1, select_blocks=1, topk=4,
               attention_query_chunk_size=3)
    cfg.update(updates)
    return AttentionSettings(**cfg)


def context(length=43, positions=None, heads=4, kv_heads=2, dim=8):
    torch.manual_seed(104)
    pos = torch.arange(length) if positions is None else torch.as_tensor(positions)
    return SelectionContext(torch.randn(heads, len(pos), dim), torch.randn(kv_heads, length, dim),
                            torch.randn(kv_heads, length, dim), torch.randn(kv_heads, length),
                            pos, 0, tiny_settings())


def manual_attention(ctx, selected=None):
    """Deliberately scalar loops over heads/queries, independent of SDPA layout."""
    out = torch.empty_like(ctx.q)
    group = ctx.q.shape[0] // ctx.k.shape[0]
    for h in range(ctx.q.shape[0]):
        kh = h // group
        for qi, position in enumerate(ctx.query_positions.tolist()):
            ids = list(range(position + 1))
            if selected is not None:
                blocks = set(selected[kh, qi].tolist()) - {-1}
                ids = [p for p in ids if p // ctx.settings.block_size in blocks]
            logits = ctx.k[kh, ids].float() @ ctx.q[h, qi].float() / math.sqrt(ctx.q.shape[-1])
            weights = torch.softmax(logits + ctx.cis[kh, ids].float(), 0)
            out[h, qi] = weights @ ctx.v[kh, ids]
    return out


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selected_attention_matches_manual_gqa_and_cis(self):
        ctx = context(43, [32, 37, 42])
        selected = native_select(ctx)
        torch.testing.assert_close(selected_causal_attention(ctx, selected), manual_attention(ctx, selected), atol=1e-6, rtol=1e-5)

    def test_dense_attention_matches_manual(self):
        ctx = context(13)
        torch.testing.assert_close(dense_causal_attention(ctx), manual_attention(ctx), atol=1e-6, rtol=1e-5)

    def test_ratio_identity_preserves_learned_cis(self):
        ctx = context(11, [10], heads=2, kv_heads=2)
        plain_logits = torch.einsum("hqd,hnd->hqn", ctx.q, ctx.k) / math.sqrt(ctx.q.shape[-1])
        plain_p = plain_logits.softmax(-1)
        e_cis = ctx.cis.exp()
        numerator = torch.einsum("hqn,hnd->hqd", plain_p, ctx.v * e_cis[..., None])
        denominator = torch.einsum("hqn,hn->hq", plain_p, e_cis)[..., None]
        torch.testing.assert_close(dense_causal_attention(ctx), numerator / denominator, atol=1e-6, rtol=1e-5)
        zero_bias = replace(ctx, cis=torch.zeros_like(ctx.cis))
        self.assertGreater((dense_causal_attention(ctx) - dense_causal_attention(zero_bias)).abs().max().item(), 0.05)

    def test_attention_never_reads_future_inside_selected_block(self):
        ctx = context(43, [32])
        selected = native_select(ctx)
        altered = replace(ctx, k=ctx.k.clone(), v=ctx.v.clone(), cis=ctx.cis.clone())
        altered.k[:, 33:] = 1000
        altered.v[:, 33:] = -1000
        altered.cis[:, 33:] = 1000
        torch.testing.assert_close(selected_causal_attention(ctx, selected), selected_causal_attention(altered, selected))

    def test_selector_never_reads_future_windows(self):
        ctx = context(91, [32, 39, 47])
        for qi, position in enumerate(ctx.query_positions.tolist()):
            single = replace(ctx, q=ctx.q[:, qi:qi+1], query_positions=ctx.query_positions[qi:qi+1])
            altered = replace(single, k=single.k.clone(), cis=single.cis.clone())
            altered.k[:, position+1:] = 9999
            altered.cis[:, position+1:] = 9999
            torch.testing.assert_close(native_select(single), native_select(altered))
            a, b, _ = native_scores(single)
            x, y, _ = native_scores(altered)
            torch.testing.assert_close(a, x)
            torch.testing.assert_close(b, y)

    def test_native_stage1_and_maxpool_independent_formula(self):
        ctx = context(73, [72])
        qk, cis, mandatory = native_scores(ctx)
        s = ctx.settings
        windows = [list(range(i, i + s.kernel_size)) for i in range(0, 73-s.kernel_size+1, s.kernel_stride)]
        compressed = torch.stack([ctx.k[:, ids].mean(1) for ids in windows], 1)
        group = ctx.q.shape[0] // ctx.k.shape[0]
        probability = []
        for kh in range(ctx.k.shape[0]):
            by_head = []
            for h in range(kh*group, (kh+1)*group):
                by_head.append(torch.softmax(compressed[kh] @ ctx.q[h, 0] / math.sqrt(ctx.q.shape[-1]), -1))
            probability.append(torch.stack(by_head).sum(0)*2)
        probability = torch.stack(probability)
        cc = torch.stack([ctx.cis[:, ids].mean(1) for ids in windows], 1)
        for block in range(qk.shape[-1]):
            if mandatory[0, block]:
                self.assertTrue(torch.isposinf(qk[:, 0, block]).all())
                continue
            ratio = s.block_size // s.kernel_stride
            ids = [i for i in range(ratio*block-1, ratio*block+ratio) if 0 <= i < len(windows)]
            torch.testing.assert_close(qk[:, 0, block], probability[:, ids].amax(-1))
            torch.testing.assert_close(cis[:, 0, block], cc[:, ids].amax(-1))

    def test_official_local_includes_current_and_16_previous(self):
        ctx = replace(context(64*40, [64*30]), settings=AttentionSettings())
        expected = [0] + list(range(14, 31))
        self.assertEqual(mandatory_blocks(ctx)[0].nonzero().flatten().tolist(), expected)

    def test_shared_quota_matches_native_and_cis_has_no_q_dependence(self):
        ctx = context(89, [88])
        qk, cis, _ = native_scores(ctx)
        self.assertTrue(torch.equal(native_select(ctx), select_with_scores(ctx, qk, cis)))
        self.assertTrue(torch.equal(native_select(ctx), select_with_scores(ctx, qk)))
        torch.testing.assert_close(cis_scores(ctx), cis_scores(replace(ctx, q=ctx.q*500)))

    def test_bad_selector_duplicate_and_empty_support_are_rejected(self):
        ctx = context(43, [42])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selected_causal_attention(ctx, torch.zeros(2, 1, 2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "no causal keys"):
            selected_causal_attention(ctx, torch.full((2, 1, 1), -1, dtype=torch.long))

    def test_one_token_prefix_is_finite(self):
        ctx = context(1)
        selected = native_select(ctx)
        actual = selected_causal_attention(ctx, selected)
        expected = ctx.v.repeat_interleave(2, dim=0)
        torch.testing.assert_close(actual, expected)

    def test_rope_relative_dot_and_fixed_longrope(self):
        config = tiny_config(rope_scaling=dict(rope_type="longrope", short_factor=[2]*4,
                                             long_factor=[2]*4, attention_factor=1.0))
        model = NosaReferenceForCausalLM(config)
        inv, amp = rope_parameters(model.config, "cpu")
        q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
        dot1 = (apply_rope(q, torch.tensor([700]), inv, amp)*apply_rope(k, torch.tensor([200]), inv, amp)).sum()
        dot2 = (apply_rope(q, torch.tensor([500]), inv, amp)*k).sum()
        torch.testing.assert_close(dot1, dot2, atol=2e-5, rtol=2e-5)
        inv0 = 10000 ** (-torch.arange(0, 8, 2).float()/8)
        torch.testing.assert_close(inv, inv0/2)


class CompleteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(53)
        self.model = NosaReferenceForCausalLM(tiny_config(), settings=tiny_settings()).eval()
        # Ensure cache and logit comparisons exercise nonzero learned key bias.
        with torch.no_grad():
            for layer in self.model.model.layers:
                layer.self_attn.A.copy_(torch.tensor([0.7, -0.4]))
        self.ids = torch.randint(4, 41, (1, 51))

    def test_full_incremental_and_chunked_logits_agree(self):
        with torch.inference_mode():
            full = self.model(self.ids)
            cache = None
            pieces = []
            for idx in range(self.ids.shape[1]):
                output = self.model(self.ids[:, idx:idx+1], past_key_values=cache)
                cache = output.past_key_values
                pieces.append(output.logits)
            incremental = torch.cat(pieces, 1)
            chunked = self.model.prefill(self.ids, chunk_size=7)
        torch.testing.assert_close(full.logits, incremental, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(full.logits[:, -1:], chunked.logits, atol=2e-6, rtol=2e-5)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache[0].k.shape[1], 51)
        torch.testing.assert_close(full.past_key_values[1].cis, cache[1].cis, atol=1e-6, rtol=1e-5)

    def test_dense_full_and_incremental_logits_agree(self):
        self.model.settings = replace(self.model.settings, dense=True)
        with torch.inference_mode():
            full = self.model(self.ids)
            chunked = self.model.prefill(self.ids, chunk_size=9)
        torch.testing.assert_close(full.logits[:, -1:], chunked.logits, atol=2e-6, rtol=2e-5)

    def test_future_tokens_do_not_change_prefix_logits(self):
        changed = self.ids.clone()
        changed[:, 19:] = 17
        with torch.inference_mode():
            first = self.model(self.ids).logits[:, :19]
            second = self.model(changed).logits[:, :19]
        torch.testing.assert_close(first, second)

    def test_external_selector_and_trace_are_real_model_hooks(self):
        calls, traces = [], []
        def selector(ctx):
            calls.append((ctx.layer_idx, ctx.query_positions.tolist()))
            return native_select(ctx)
        self.model.selector = selector
        self.model.trace_callback = lambda ctx, blocks: traces.append((ctx.layer_idx, tuple(blocks.shape)))
        with torch.inference_mode():
            result = self.model(self.ids[:, :7], use_cache=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(traces), 2)
        self.assertIsNone(result.past_key_values)
        self.assertTrue(torch.isfinite(result.logits).all())

    def test_greedy_tokens_match_uncached_full_recompute(self):
        prefix = self.ids[:, :37]
        with torch.inference_mode():
            generated = self.model.greedy_generate(prefix, max_new_tokens=4, eos_token_id=[], chunk_size=7)
            expected = prefix
            for _ in range(4):
                token = self.model(expected).logits[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), 1)
        self.assertTrue(torch.equal(generated, expected))

    def test_eos_stops_on_actual_raw_token(self):
        with torch.inference_mode():
            token = self.model(self.ids[:, :9]).logits[:, -1].argmax(-1).item()
            result = self.model.greedy_generate(self.ids[:, :9], max_new_tokens=6, eos_token_id=[token])
        self.assertEqual(result.shape[1], 10)
        self.assertEqual(result[0, -1].item(), token)

    def test_checkpoint_roundtrip_loads_every_tensor(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"config.json").write_text(json.dumps(tiny_config()))
            save_file(self.model.state_dict(), str(path/"model.safetensors"))
            restored = NosaReferenceForCausalLM.from_pretrained(path, settings=tiny_settings())
            with torch.inference_mode():
                expected = self.model(self.ids[:, :11]).logits
                actual = restored(self.ids[:, :11]).logits
            torch.testing.assert_close(actual, expected)
            self.assertEqual(restored.load_report["loaded_tensors"], len(self.model.state_dict()))
            self.assertEqual(restored.load_report["layers"], 2)

    def test_loader_rejects_missing_pretrained_cis_parameter(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"config.json").write_text(json.dumps(tiny_config()))
            state = self.model.state_dict()
            del state["model.layers.0.self_attn.A"]
            save_file(state, str(path/"model.safetensors"))
            with self.assertRaisesRegex(ValueError, "missing checkpoint tensors"):
                NosaReferenceForCausalLM.from_pretrained(path)

    def test_pytorch_bin_roundtrip_uses_restricted_loader(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"config.json").write_text(json.dumps(tiny_config()))
            torch.save(self.model.state_dict(), path/"pytorch_model.bin")
            with patch("torch.load", wraps=torch.load) as reader:
                restored = NosaReferenceForCausalLM.from_pretrained(path, settings=tiny_settings())
            self.assertTrue(all(call.kwargs["weights_only"] is True for call in reader.call_args_list))
            with torch.inference_mode():
                expected = self.model(self.ids[:, :11]).logits
                actual = restored(self.ids[:, :11]).logits
            torch.testing.assert_close(actual, expected)
            self.assertEqual(restored.load_report["loaded_tensors"], len(self.model.state_dict()))

    def test_pytorch_bin_missing_cis_and_wrong_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"config.json").write_text(json.dumps(tiny_config()))
            state = self.model.state_dict()
            del state["model.layers.0.self_attn.delta.weight"]
            torch.save(state, path/"pytorch_model.bin")
            with self.assertRaisesRegex(ValueError, "missing checkpoint tensors"):
                NosaReferenceForCausalLM.from_pretrained(path)
            state = self.model.state_dict()
            state["model.layers.0.self_attn.A"] = torch.ones(11)
            torch.save(state, path/"pytorch_model.bin")
            with self.assertRaisesRegex(ValueError, "checkpoint shape mismatch"):
                NosaReferenceForCausalLM.from_pretrained(path)

    def test_all_28_layers_participate_in_logits(self):
        model = NosaReferenceForCausalLM(tiny_config(num_hidden_layers=28), settings=tiny_settings()).eval()
        observed = []
        model.trace_callback = lambda ctx, _: observed.append(ctx.layer_idx)
        with torch.inference_mode():
            result = model(self.ids[:, :3])
        self.assertEqual(observed, list(range(28)))
        self.assertTrue(torch.isfinite(result.logits).all())


if __name__ == "__main__":
    unittest.main()
