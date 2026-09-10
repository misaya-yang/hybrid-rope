"""CPU equivalence only; real-model parity and speed remain unmeasured."""
from dataclasses import replace
import math
import unittest
import torch

from .cached_full_covariance import CachedFullCovarianceSelector, build_packed, packed_logmass
from .full_covariance_probe import FullCovarianceSelector, full_covariance_logmass
from .runtime import NosaReferenceForCausalLM
from .test_runtime import context, tiny_config, tiny_settings


class CachedCovarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fp64_packed_matches_direct_projection_moments(self):
        torch.manual_seed(31)
        k = torch.randn(2, 3, 8, 6, dtype=torch.float64)
        cis = torch.randn(2, 3, 8, dtype=torch.float64)
        q = torch.randn(2, 2, 5, 6, dtype=torch.float64) / math.sqrt(6)
        summary = build_packed(k, cis)
        actual = packed_logmass(q, summary)
        logits = torch.einsum("hgqd,hbtd->hgqbt", q, k)
        w = cis.softmax(-1)[:, None, None]
        mean = (w * logits).sum(-1)
        expected = cis.logsumexp(-1)[:, None, None] + mean + .5 * (w * (logits-mean[..., None]).square()).sum(-1)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(summary.coefficients.shape[-1], 1 + 6 + 6*7//2)
        self.assertEqual(summary.nbytes, summary.coefficients.numel() * 8)

    def test_causal_partial_and_future_contract(self):
        for position in (5, 8, 15, 23):
            ctx = context(29, [position])
            actual = CachedFullCovarianceSelector().logmass(ctx)
            torch.testing.assert_close(actual, full_covariance_logmass(ctx), atol=2e-6, rtol=2e-6)
            changed = replace(ctx, k=ctx.k.clone(), cis=ctx.cis.clone())
            changed.k[:, position+1:] = 9999
            changed.cis[:, position+1:] = 9999
            torch.testing.assert_close(actual, CachedFullCovarianceSelector().logmass(changed))

    def test_cached_append_and_selected_support_parity(self):
        ctx = context(89, [88])
        selector = CachedFullCovarianceSelector()
        short = replace(ctx, k=ctx.k[:, :65], v=ctx.v[:, :65], cis=ctx.cis[:, :65], query_positions=torch.tensor([64]))
        selector.logmass(short)
        actual = selector(ctx)
        self.assertTrue(torch.equal(actual, FullCovarianceSelector()(ctx)))
        self.assertEqual(selector.cache[0].blocks, 89//8)
        self.assertEqual(selector.metrics["max_metadata_bytes"], selector.cache[0].nbytes)

    def test_tiny_complete_generation_and_logits_match_direct(self):
        torch.manual_seed(37)
        model = NosaReferenceForCausalLM(tiny_config(), settings=tiny_settings(), selector=FullCovarianceSelector()).eval()
        ids = torch.randint(4, 41, (1, 49))
        with torch.inference_mode():
            expected_logits = model.prefill(ids, chunk_size=7).logits
            expected = model.greedy_generate(ids, max_new_tokens=3, eos_token_id=[], chunk_size=7)
            model.selector = CachedFullCovarianceSelector()
            actual_logits = model.prefill(ids, chunk_size=7).logits
            actual = model.greedy_generate(ids, max_new_tokens=3, eos_token_id=[], chunk_size=7)
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
