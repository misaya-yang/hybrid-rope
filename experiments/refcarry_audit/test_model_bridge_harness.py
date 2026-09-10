"""Random tiny CPU transformers test integration only, never model capability."""
import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.refcarry_audit.model_bridge_harness import ModelBridgeHarness


class ModelBridgeChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(903)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, use_sliding_window=False)
        config._attn_implementation = 'sdpa'
        self.model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        self.ids = torch.tensor([[3, 5, 7, 9, 2, 12, 21, 31, 6, 8, 1, 17, 19]])
        with torch.no_grad():
            self.cache = self.model(self.ids, use_cache=True).past_key_values

    def step(self):
        return self.model(torch.tensor([[11]]), past_key_values=copy.deepcopy(self.cache),
                          use_cache=True, logits_to_keep=1).logits

    def test_zero_adapter_is_bitwise_native_and_close_restores(self):
        with torch.no_grad():expected = self.step()
        harness = ModelBridgeHarness(self.model, 0, 2, self.model.model.rotary_emb.inv_freq)
        try:
            for mode in ('native', 'arithmetic', 'geometric', 'point', 'current'):
                harness.mode = mode
                with torch.no_grad():got = self.step()
                self.assertTrue(torch.equal(expected, got), mode)
            self.assertEqual(harness.calls, {'writer': 5, 'reader': 5})
        finally:
            harness.close()
        with torch.no_grad():got = self.step()
        self.assertTrue(torch.equal(expected, got))
        self.assertEqual(self.model.config._attn_implementation, 'sdpa')

    def test_adapter_can_receive_gradient_through_real_decoder(self):
        harness = ModelBridgeHarness(self.model, 0, 2, self.model.model.rotary_emb.inv_freq)
        try:
            harness.mode = 'arithmetic'
            logits = self.step()
            loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor([22]))
            loss.backward()
            gradients = [b.projection.weight.grad for b in harness.bridge.biases]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(g.norm().item() for g in gradients), 0.)
            self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters()
                                if 'reference_bias_adapter' not in n))
        finally:
            harness.close()

    def test_selected_support_baseline_matches_zero_adapter(self):
        def support(module, q, k, position):
            return torch.tensor([0, 5, position])[None].expand(q.shape[1], -1)
        harness = ModelBridgeHarness(self.model, 0, 2, self.model.model.rotary_emb.inv_freq, support)
        try:
            with torch.no_grad():
                expected = self.step()
                harness.mode = 'arithmetic'
                got = self.step()
            self.assertTrue(torch.equal(expected, got))
        finally:
            harness.close()


if __name__ == '__main__':
    unittest.main()
