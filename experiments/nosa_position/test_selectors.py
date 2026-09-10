"""Mathematical and causal tests, not task-quality evidence."""
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from .runtime import AttentionSettings, SelectionContext, apply_rope
from .selector_controls import BlockSummarySelector, build_summary, summary_logmass


class SummaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)

    def test_tilt_exactly_incorporates_nonconstant_bias(self):
        k = torch.randn(2, 3, 8, 4, dtype=torch.float64)
        b = torch.randn(2, 3, 8, dtype=torch.float64) * 3
        q = torch.randn(2, 2, 5, 4, dtype=torch.float64)
        # Each singleton subblock recovers exact logmass, with no extra CIS.
        x = k.reshape(2, 6, 4, 4)
        z = b.reshape(2, 6, 4)
        summary = build_summary(x, z, "split4")
        actual = summary_logmass(q, summary, "split4")
        exact = torch.logsumexp(torch.einsum("hgqd,hbtd->hgqbt", q, x) + z[:, None, None], -1)
        torch.testing.assert_close(actual, exact, rtol=1e-12, atol=1e-12)

    def test_split_jensen_bounds_with_nonuniform_cis(self):
        k = torch.randn(1, 5, 16, 6, dtype=torch.float64)
        b = torch.randn(1, 5, 16, dtype=torch.float64)
        q = torch.randn(1, 3, 4, 6, dtype=torch.float64) / 3
        means = summary_logmass(q, build_summary(k, b, "weighted_mean"), "weighted_mean")
        split2 = summary_logmass(q, build_summary(k, b, "split2"), "split2")
        split4 = summary_logmass(q, build_summary(k, b, "split4"), "split4")
        exact = torch.logsumexp(torch.einsum("hgqd,hbtd->hgqbt", q, k) + b[:, None, None], -1)
        self.assertTrue(bool((means <= split2 + 1e-12).all()))
        self.assertTrue(bool((split2 <= split4 + 1e-12).all()))
        self.assertTrue(bool((split4 <= exact + 1e-12).all()))

    def test_pc2_rotation_equivariance_not_diagonal_approximation(self):
        k = torch.randn(1, 4, 12, 8, dtype=torch.float64)
        b = torch.randn(1, 4, 12, dtype=torch.float64)
        q = torch.randn(1, 2, 3, 8, dtype=torch.float64)
        phase = torch.tensor([.13, 1.7, 2.9, -.4], dtype=torch.float64)
        def rotate(x):
            a, z = x[..., :4], x[..., 4:]
            return torch.cat((a * phase.cos() - z * phase.sin(), z * phase.cos() + a * phase.sin()), -1)
        before = summary_logmass(q, build_summary(k, b, "pc2"), "pc2")
        after = summary_logmass(rotate(q), build_summary(rotate(k), b, "pc2"), "pc2")
        torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)

    def test_cross_pair_correlation_counterexample(self):
        # In split-half basis the real axes of the two pairs are dims 0,1.
        # Their contributions cancel exactly; PC2 discards that correlation.
        k = torch.tensor([[[[2., -2., 0., 0.], [-2., 2., 0., 0.]]]], dtype=torch.float64)
        q = torch.tensor([[[[1., 1., 0., 0.]]]], dtype=torch.float64)
        b = torch.zeros(1, 1, 2, dtype=torch.float64)
        pc2 = summary_logmass(q, build_summary(k, b, "pc2"), "pc2")
        pca = summary_logmass(q, build_summary(k, b, "cobs_rank1"), "cobs_rank1")
        self.assertAlmostEqual(pc2.item(), math.log(2) + 4., places=10)
        self.assertAlmostEqual(pca.item(), math.log(2), places=10)

    def test_pc2_is_neither_upper_nor_lower_mass_bound(self):
        q = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
        b = torch.zeros(1, 1, 64, dtype=torch.float64)
        for scores, expected_sign in [([4.] * 32 + [-4.] * 32, 1), ([10.] + [-10 / 63] * 63, -1)]:
            k = torch.stack((torch.tensor(scores, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)), -1)[None, None]
            predicted = summary_logmass(q, build_summary(k, b, "pc2"), "pc2").item()
            exact = torch.logsumexp(k[0, 0, :, 0], 0).item()
            self.assertGreater(expected_sign * (predicted - exact), 1)

    def test_metadata_byte_accounting_has_no_unused_split_mean(self):
        k, b = torch.randn(2, 5, 64, 128), torch.randn(2, 5, 64)
        pc2 = build_summary(k, b, "pc2")
        rank2 = build_summary(k, b, "cobs_rank2")
        split2 = build_summary(k, b, "split2")
        self.assertEqual(pc2.nbytes(), 2 * 5 * (128 + 192 + 1) * 4)
        self.assertEqual(rank2.nbytes(), 2 * 5 * (128 + 256 + 1) * 4)
        self.assertEqual(split2.nbytes(), 2 * 5 * 2 * (128 + 1) * 4)
        for mode in ("split2", "split4", "quest"):
            summary = build_summary(k, b, mode)
            self.assertEqual(summary.mean.untyped_storage().nbytes(), 0)
            self.assertEqual(summary.log_weight.untyped_storage().nbytes(), 0)
            self.assertEqual(summary.nbytes(), sum(t.untyped_storage().nbytes() for t in summary.extra.values()))

    def test_metadata_counts_retained_view_storage(self):
        k, b = torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8)
        summary = build_summary(k, b, "weighted_mean")
        before = summary.nbytes()
        summary.mean = summary.mean[..., :0]
        self.assertEqual(summary.nbytes(), before)

    def test_pc2_unweighted_stats_and_partial_normalizers_ignore_cis(self):
        s = AttentionSettings(kernel_size=8, kernel_stride=4, block_size=8, local_blocks=1, select_blocks=1, topk=4)
        q, k = torch.randn(4, 5, 8), torch.randn(2, 64, 8)
        v, cis = torch.randn_like(k), torch.randn(2, 64) * 5
        positions = torch.arange(41, 46)
        context = SelectionContext(q, k, v, cis, positions, 0, s)
        zero = replace(context, cis=torch.zeros_like(cis))
        unweighted = BlockSummarySelector("pc2_unweighted")
        reference = BlockSummarySelector("pc2")
        torch.testing.assert_close(unweighted.logmass(context), reference.logmass(zero))
        changed = replace(context, cis=cis + torch.randn_like(cis) * 100)
        torch.testing.assert_close(unweighted.logmass(context), BlockSummarySelector("pc2_unweighted").logmass(changed))
        # Public build_summary/summary_logmass also accept the named control.
        x, z = k.reshape(2, 8, 8, 8), cis.reshape(2, 8, 8)
        query = q.reshape(2, 2, 5, 8) / math.sqrt(8)
        named = summary_logmass(query, build_summary(x, z, "pc2_unweighted"), "pc2_unweighted")
        direct = summary_logmass(query, build_summary(x, torch.zeros_like(z), "pc2"), "pc2")
        torch.testing.assert_close(named, direct)

    def test_pc2_unweighted_preserves_real_cis_quota_context(self):
        s = AttentionSettings(kernel_size=8, kernel_stride=4, block_size=8, local_blocks=1, select_blocks=1, topk=4)
        q, k, cis = torch.randn(4, 2, 8), torch.randn(2, 64, 8), torch.randn(2, 64)
        context = SelectionContext(q, k, k, cis, torch.tensor([62, 63]), 0, s)
        sentinel = torch.zeros(2, 2, 4, dtype=torch.long)
        with patch("experiments.nosa_position.selector_controls.select_with_scores", return_value=sentinel) as union:
            selected = BlockSummarySelector("pc2_unweighted")(context)
        self.assertIs(selected, sentinel)
        self.assertIs(union.call_args.args[0], context)
        torch.testing.assert_close(union.call_args.args[0].cis, cis)

    def test_pc2_unweighted_still_routes_the_real_cis_only_slot(self):
        s = AttentionSettings(kernel_size=8, kernel_stride=4, block_size=8, local_blocks=1, select_blocks=1, topk=4)
        q, k = torch.zeros(4, 1, 8), torch.zeros(2, 64, 8)
        for preferred in (2, 4):
            cis = torch.zeros(2, 64)
            cis[:, preferred*8:(preferred+1)*8] = 100
            context = SelectionContext(q, k, k, cis, torch.tensor([63]), 0, s)
            selected = BlockSummarySelector("pc2_unweighted")(context)
            self.assertEqual(selected[0, 0].tolist(), [0, preferred, 6, 7])

    def test_selector_logmass_causality_and_incremental_summary(self):
        s = AttentionSettings(kernel_size=8, kernel_stride=4, block_size=8, local_blocks=1, select_blocks=1, topk=4)
        q = torch.randn(4, 16, 8)
        k, v, cis = torch.randn(2, 64, 8), torch.randn(2, 64, 8), torch.randn(2, 64)
        positions = torch.arange(32, 48)
        context = SelectionContext(q, k, v, cis, positions, 0, s)
        future_k, future_cis = k.clone(), cis.clone()
        future_k[:, 48:] *= 100
        future_cis[:, 48:] += 100
        other = SelectionContext(q, future_k, v, future_cis, positions, 0, s)
        for mode in ("weighted_mean", "pc2", "pc2_unweighted", "cobs_rank2", "split2", "quest", "exact_mass"):
            a = BlockSummarySelector(mode)
            b = BlockSummarySelector(mode)
            torch.testing.assert_close(a.logmass(context), b.logmass(other))
            selected = a(context)
            self.assertTrue(bool(((selected < 0) | (selected <= positions[None, :, None] // 8)).all()))
            # Smaller-prefix construction then growth must give same summary.
            partial = SelectionContext(q[:, :8], k[:, :40], v[:, :40], cis[:, :40], positions[:8], 0, s)
            c = BlockSummarySelector(mode)
            c.logmass(partial)
            torch.testing.assert_close(a.logmass(context), c.logmass(context))

    def test_exact_mass_matches_direct_causal_token_softmax(self):
        s = AttentionSettings()
        q, k, v, cis = torch.randn(4, 3, 8), torch.randn(2, 150, 8), torch.randn(2, 150, 8), torch.randn(2, 150)
        positions = torch.tensor([100, 101, 102])
        context = SelectionContext(q, k, v, cis, positions, 0, s)
        block_logmass = BlockSummarySelector("exact_mass").logmass(context)
        scores = torch.einsum("hgqd,htd->hgqt", q.reshape(2, 2, 3, 8) / math.sqrt(8), k)
        scores += cis[:, None, None]
        scores.masked_fill_(torch.arange(150)[None, None, None] > positions[None, None, :, None], -torch.inf)
        probabilities = scores.softmax(-1)
        expected = torch.stack([probabilities[..., :64].sum(-1), probabilities[..., 64:128].sum(-1), probabilities[..., 128:].sum(-1)], -1)
        torch.testing.assert_close(block_logmass.softmax(-1), expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
