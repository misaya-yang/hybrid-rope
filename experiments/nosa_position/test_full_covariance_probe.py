"""Small CPU identity/causality/counterexample tests for the diagnostic bridge."""
import contextlib
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import run as base
from .exact_probe import ExactBlockSelector
from .full_covariance_probe import (FullCovarianceSelector, _run_existing,
                                   diagnose_query_state, full_covariance_logmass, main)
from .runtime import (AttentionSettings, NosaReferenceForCausalLM, SelectionContext,
                      mandatory_blocks, select_with_scores)
from .selector_controls import build_summary, summary_logmass
from .test_runtime import context, tiny_config, tiny_settings


def scalar_blocks(first_scores, second_score):
    size, dim = len(first_scores), 4
    keys = torch.zeros(1, 2 * size + 1, dim)
    keys[0, :size, 0] = torch.tensor(first_scores)
    keys[0, size:2 * size, 0] = second_score
    q = torch.tensor([math.sqrt(dim), 0., 0., 0.]).view(1, 1, dim)
    settings = AttentionSettings(kernel_size=32, kernel_stride=16, block_size=size,
        init_blocks=0, local_blocks=0, select_blocks=2, topk=2, attention_query_chunk_size=1)
    return SelectionContext(q, keys, torch.zeros_like(keys), torch.zeros(1, keys.shape[1]),
                            torch.tensor([2 * size]), 0, settings)


class MathematicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projected_score_variance_equals_full_covariance(self):
        ctx = context(25, [16, 23, 24], dim=8)
        ctx = replace(ctx, q=ctx.q.double(), k=ctx.k.double(), cis=ctx.cis.double())
        actual = full_covariance_logmass(ctx)
        x, b = ctx.k[:, :8], ctx.cis[:, :8]
        w = b.softmax(-1)
        mu = (w[..., None] * x).sum(1)
        centered = x - mu[:, None]
        covariance = torch.einsum("ht,hti,htj->hij", w, centered, centered)
        q = ctx.q.reshape(2, 2, 3, 8) / math.sqrt(8)
        expected = b.logsumexp(-1)[:, None, None] + torch.einsum("hgqi,hi->hgq", q, mu)
        expected += .5 * torch.einsum("hgqi,hij,hgqj->hgq", q, covariance, q)
        torch.testing.assert_close(actual[..., 0], expected, atol=1e-12, rtol=1e-12)

    def test_low_rank_cross_pair_cancellation_is_isolated(self):
        ctx = scalar_blocks([2., -2.] * 32, .5)
        ctx.k[0, :64, 1] = -ctx.k[0, :64, 0]
        ctx.q[0, 0, 1] = 2.
        diagnostic = diagnose_query_state(ctx)
        pair, full, exact = [diagnostic["logmass"][name] for name in ("pair", "full_covariance", "exact")]
        self.assertAlmostEqual(float(full[..., 0]), math.log(64), places=5)
        self.assertAlmostEqual(float(pair[..., 0] - full[..., 0]), 4., places=5)
        torch.testing.assert_close(full[..., :2], exact[..., :2], atol=1e-6, rtol=1e-6)
        keys = ctx.k[:, :128].reshape(1, 2, 64, 4)
        cis = ctx.cis[:, :128].reshape(1, 2, 64)
        low_rank = summary_logmass(ctx.q.reshape(1, 1, 1, 4) / 2,
                                  build_summary(keys, cis, "cobs_rank1"), "cobs_rank1")
        torch.testing.assert_close(low_rank, full[..., :2], atol=2e-6, rtol=2e-6)

    def test_rare_key_failure_survives_complete_covariance(self):
        ctx = scalar_blocks([10.] + [0.] * 63, 1.)
        before = [tensor.clone() for tensor in (ctx.q, ctx.k, ctx.cis)]
        diagnostic = diagnose_query_state(ctx)
        pair, full, exact = [diagnostic["logmass"][name] for name in ("pair", "full_covariance", "exact")]
        torch.testing.assert_close(pair[..., :2], full[..., :2], atol=1e-6, rtol=1e-6)
        self.assertLess(float(full[..., 0]), float(full[..., 1]))
        self.assertGreater(float(exact[..., 0]), float(exact[..., 1]))
        self.assertGreater(float(diagnostic["higher_cumulant_error"][..., 0]), 4.)
        self.assertLess(diagnostic["metrics"]["decomposition_residual_max"], 1e-5)
        for tensor, copy in zip((ctx.q, ctx.k, ctx.cis), before):
            torch.testing.assert_close(tensor, copy, atol=0, rtol=0)

    def test_current_block_is_exact_and_future_keys_are_invisible(self):
        for position in (5, 8, 15, 23):
            ctx = context(29, [position])
            actual = full_covariance_logmass(ctx)
            changed = replace(ctx, k=ctx.k.clone(), cis=ctx.cis.clone())
            changed.k[:, position + 1:] = 9999
            changed.cis[:, position + 1:] = 9999
            torch.testing.assert_close(actual, full_covariance_logmass(changed))
            exact = ExactBlockSelector("exact_mass").logmass(ctx)
            current = position // ctx.settings.block_size
            torch.testing.assert_close(actual[..., current], exact[..., current], atol=1e-6, rtol=1e-6)
            self.assertTrue(bool(torch.isneginf(actual[..., current + 1:]).all()))

    def test_same_gqa_normalization_quota_and_mandatory_contract(self):
        ctx = context(89, [75, 80, 88])
        selector = FullCovarianceSelector()
        scores = full_covariance_logmass(ctx)
        expected = select_with_scores(ctx, scores.softmax(-1).sum(1))
        actual = selector(ctx)
        self.assertTrue(torch.equal(actual, expected))
        protected = mandatory_blocks(ctx)
        for h in range(actual.shape[0]):
            for i in range(actual.shape[1]):
                kept = set(actual[h, i].tolist()) - {-1}
                self.assertTrue(set(protected[i].nonzero().flatten().tolist()).issubset(kept))
                self.assertLessEqual(len(kept), ctx.settings.topk)
        self.assertGreater(selector.metrics["full_covariance_raw_key_scores"], 0)
        self.assertEqual(selector.metrics["max_metadata_bytes"], 0)
        selector.reset()
        self.assertEqual(selector.metrics["full_covariance_raw_key_scores"], 0)
        self.assertTrue(torch.equal(selector(ctx), expected))

    def test_tiny_complete_model_generation_executes_new_selector(self):
        torch.manual_seed(113)
        selector = FullCovarianceSelector()
        model = NosaReferenceForCausalLM(tiny_config(), settings=tiny_settings(), selector=selector).eval()
        ids = torch.randint(4, 41, (1, 49))
        output = model.greedy_generate(ids, max_new_tokens=3, eos_token_id=[], chunk_size=7)
        self.assertEqual(output.shape, (1, 52))
        self.assertGreater(selector.metrics["full_covariance_raw_key_scores"], 0)


class EntryTests(unittest.TestCase):
    def test_registration_is_process_local_and_restored(self):
        originals = base.MODES, base.BlockSummarySelector, base.source_hashes
        def inspect():
            self.assertIn("full_covariance", base.MODES)
            self.assertIsInstance(base.BlockSummarySelector("full_covariance"), FullCovarianceSelector)
            self.assertIn("exact_probe.py", base.source_hashes())
            self.assertIn("full_covariance_probe.py", base.source_hashes())
        with patch.object(base, "main", side_effect=inspect):
            _run_existing(["--output", "/unused"])
        self.assertEqual(base.MODES, originals[0])
        self.assertIs(base.BlockSummarySelector, originals[1])
        self.assertIs(base.source_hashes, originals[2])

    def test_dry_run_loads_nothing_and_stop_prevents_execute(self):
        with patch.object(base, "main") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["--selectors", "full_covariance", "--output", "/unused"])
            self.assertEqual(result["status"], "DRY_RUN")
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "STOP").touch()
            with patch.object(base, "main") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "global STOP"):
                        main(["--execute", "--queue-root", directory, "--output", "/unused"])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
