"""CPU operator checks and retained *failure* examples; not model evidence."""
import math
import unittest

import torch

from .ops import (NativeRope, attention_mean, floor_keep_budget,
                  make_sampling_plan, pm_keep_scores, sample_prefix_queries,
                  select_fixed_budget)


class TestSamplingAndBudget(unittest.TestCase):
    def test_shared_sampling_is_prefix_only_and_head_layer_seeded(self):
        a = make_sampling_plan(47, 4, samples_per_head=32, horizon=512)
        b = make_sampling_plan(47, 4, samples_per_head=32, horizon=512)
        self.assertTrue(torch.equal(a.query_indices, b.query_indices))
        self.assertTrue(torch.equal(a.future_positions, b.future_positions))
        self.assertTrue(bool(((a.query_indices >= 4) & (a.query_indices < 47)).all()))
        self.assertTrue(bool(((a.future_positions >= 47) & (a.future_positions < 559)).all()))
        self.assertFalse(torch.equal(a.query_indices[0], a.query_indices[1]))
        other = make_sampling_plan(47, 4, samples_per_head=32, horizon=128, layer_idx=1)
        self.assertFalse(torch.equal(a.query_indices, other.query_indices))
        h128 = make_sampling_plan(47, 4, samples_per_head=32, horizon=128)
        self.assertTrue(torch.equal(a.query_indices, h128.query_indices))
        recent = make_sampling_plan(47, 4, samples_per_head=32, query_start=40)
        self.assertTrue(bool((recent.query_indices >= 40).all()))
        full = torch.arange(4 * 47 * 2).reshape(4, 47, 2).float()
        sampled = sample_prefix_queries(full, a)
        for head in range(4):
            torch.testing.assert_close(sampled[head], full[head, a.query_indices[head]])

    def test_integer_budget_protection_unique_sorted_and_ties(self):
        self.assertEqual(floor_keep_budget(8195), 2048)
        scores = torch.zeros(2, 10)
        indices = select_fixed_budget(scores, 6, sink_tokens=2, recent_tokens=2)
        torch.testing.assert_close(indices, torch.tensor([[0, 1, 2, 3, 8, 9]]).expand(2, -1))
        self.assertTrue(bool((indices[:, 1:] > indices[:, :-1]).all()))
        with self.assertRaisesRegex(ValueError, "protected union"):
            select_fixed_budget(scores, 3, sink_tokens=2, recent_tokens=2)
        all_indices = select_fixed_budget(scores, 10)
        torch.testing.assert_close(all_indices, torch.arange(10)[None].expand(2, -1))
        # Overlapping protected regions consume their union, not their sum.
        overlap = select_fixed_budget(torch.zeros(1, 5), 5, sink_tokens=3, recent_tokens=3)
        torch.testing.assert_close(overlap, torch.arange(5)[None])
        with self.assertRaises(ValueError):
            select_fixed_budget(scores, 2.5, sink_tokens=0, recent_tokens=0)


class TestNativeRope(unittest.TestCase):
    def test_partial_and_layout_match_explicit_planes(self):
        x = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        for layout, pairs in (("split_half", ((1, 3), (2, 4))),
                              ("interleaved", ((1, 2), (3, 4)))):
            rope = NativeRope(torch.tensor([.3, .7]), rotary_start=1, layout=layout)
            expected = x.clone()
            for (a, b), omega in zip(pairs, (.3, .7)):
                c, s = math.cos(3 * omega), math.sin(3 * omega)
                expected[:, a] = c * x[:, a] - s * x[:, b]
                expected[:, b] = s * x[:, a] + c * x[:, b]
            actual = rope.apply(x, torch.tensor([3]))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1))
        identity = NativeRope(torch.empty(0), rotary_dim=0)
        torch.testing.assert_close(identity.apply(x, torch.tensor([99])), x)

    def test_analytic_mean_uses_entire_horizon_and_correct_limits(self):
        freq = torch.tensor([0., 1., .017, 2 * math.pi], dtype=torch.float64)
        rope = NativeRope(freq)
        for horizon in (2, 128, 512):
            c, s, _ = rope.average_phases(11, horizon)
            theta = torch.arange(11, 11 + horizon, dtype=torch.float64)[:, None] * freq
            torch.testing.assert_close(c.double(), theta.cos().mean(0), atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(s.double(), theta.sin().mean(0), atol=1e-7, rtol=1e-6)

    def test_unit_zero_modulus_has_documented_midpoint_fallback(self):
        rope = NativeRope(torch.tensor([math.pi], dtype=torch.float64), layout="interleaved")
        c, s, count = rope.average_phases(0, 2, unit_modulus=True)
        self.assertEqual(count, 1)
        # Frequency remainder can choose +/-pi; both integer-position supports
        # coincide, but midpoint fallback must use the *native* +pi convention.
        torch.testing.assert_close(c.square() + s.square(), torch.ones_like(c))
        torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-7, rtol=0)
        torch.testing.assert_close(s, torch.ones_like(s))

    def test_reject_unsupported_amplitude_and_noninteger_positions(self):
        with self.assertRaises(ValueError):
            NativeRope(torch.tensor([1.]), amplitude=1.1)
        with self.assertRaises(ValueError):
            NativeRope(torch.tensor([1.]), rotary_dim=1)
        with self.assertRaises(ValueError):
            NativeRope(torch.tensor([1.])).apply(torch.ones(1, 2), torch.tensor([1.5]))


class TestScorer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_tiled_attention_matches_full_softmax_and_gqa_mean(self):
        q, k = torch.randn(4, 7, 6), torch.randn(2, 13, 6)
        actual = attention_mean(q, k, attention_scale=.7, query_chunk_size=3, key_chunk_size=4)
        expected = torch.stack([
            torch.stack([(q[h] @ k[g].T * .7).softmax(-1).mean(0)
                         for h in range(g * 2, (g + 1) * 2)]).mean(0)
            for g in range(2)])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2))

    def test_protected_keys_stay_in_softmax_denominator(self):
        q = torch.tensor([[[1., 0.], [1., 0.]]])
        k = torch.tensor([[[100., 0.], [0., 0.], [0., 0.]]])
        scores = attention_mean(q, k, attention_scale=1., key_chunk_size=1)
        self.assertGreater(float(scores[0, 0]), .999999)
        self.assertLess(float(scores[0, 1:].sum()), 1e-20)
        indices = select_fixed_budget(scores, 2, sink_tokens=1, recent_tokens=0)
        torch.testing.assert_close(indices, torch.tensor([[0, 1]]))

    def test_outer_autocast_does_not_change_fp32_score_contract(self):
        q, k = torch.randn(2, 5, 4), torch.randn(1, 7, 4)
        expected = attention_mean(q, k)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = attention_mean(q, k)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_h1_three_arms_identical_and_same_samples_not_mutated(self):
        q, k = torch.randn(4, 9, 8), torch.randn(2, 17, 8)
        original_q, original_k = q.clone(), k.clone()
        rope = NativeRope(torch.tensor([1., .1, .01, .001]))
        positions = torch.full((4, 9), 17, dtype=torch.long)
        outputs = [pm_keep_scores(q, k, rope, arm=arm, horizon=1,
                    future_positions=positions, query_chunk_size=3, key_chunk_size=4)
                   for arm in ("pm", "collapse", "unit")]
        for result in outputs[1:]:
            torch.testing.assert_close(result.scores, outputs[0].scores, atol=0, rtol=0)
        torch.testing.assert_close(q, original_q, atol=0, rtol=0)
        torch.testing.assert_close(k, original_k, atol=0, rtol=0)
        self.assertEqual(outputs[0].metrics["max_logits_tile_elements"], 12)
        self.assertFalse(outputs[0].metrics["value_norm_weighting"])

    def test_common_origin_shift_preserves_three_scores(self):
        q, k = torch.randn(4, 7, 6), torch.randn(2, 13, 6)
        rope = NativeRope(torch.tensor([.6, .13]), rotary_start=1)
        positions = make_sampling_plan(13, 4, samples_per_head=7, horizon=5).future_positions
        shifted_k = rope.apply(k, torch.tensor(17))
        for arm in ("pm", "collapse", "unit"):
            a = pm_keep_scores(q, k, rope, arm=arm, horizon=5, future_start=13,
                               future_positions=positions).scores
            b = pm_keep_scores(q, shifted_k, rope, arm=arm, horizon=5, future_start=30,
                               future_positions=positions + 17).scores
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_sampling_position_is_shared_across_all_frequency_pairs(self):
        q, k = torch.randn(2, 3, 4), torch.randn(1, 9, 4)
        rope = NativeRope(torch.tensor([1., .1]))
        with self.assertRaisesRegex(ValueError, "not per-frequency"):
            pm_keep_scores(q, k, rope, future_positions=torch.ones(2, 3, 2, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "outside"):
            pm_keep_scores(q, k, rope, future_positions=torch.zeros(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "follow the complete"):
            pm_keep_scores(q, k, rope, arm="collapse", future_start=0)


class TestRetainedCounterexamples(unittest.TestCase):
    @staticmethod
    def scenario():
        rope = NativeRope(torch.tensor([1.]), rotary_dim=2, layout="interleaved")
        u = torch.tensor([math.cos(1), -math.sin(1), 1., 0.])
        q = rope.apply(u.repeat(2, 1), torch.tensor([0, 2]))
        k = torch.tensor([[0., 8 / math.sin(1), 0., 0.],
                          [0., -8 / math.sin(1), 0., 0.], [0., 0., 2., 0.]])
        return rope, u, q, k

    def test_nonlinear_ranking_reversal_and_unit_control_failure(self):
        rope, u, q, k = self.scenario()
        p = attention_mean(q[None], k[None])
        c = attention_mean(q.mean(0).repeat(2, 1)[None], k[None])
        unit = rope.apply(u[None], torch.tensor([1]))
        un = attention_mean(unit[None], k[None])
        torch.testing.assert_close(p[0], torch.tensor([.4762947, .4762947, .0474107]), atol=2e-6, rtol=0)
        torch.testing.assert_close(c[0], torch.tensor([.2119416, .2119416, .5761169]), atol=2e-6, rtol=0)
        torch.testing.assert_close(un, c, atol=2e-6, rtol=0)
        keep_p = select_fixed_budget(p, 2, sink_tokens=0, recent_tokens=0)
        keep_c = select_fixed_budget(c, 2, sink_tokens=0, recent_tokens=0)
        torch.testing.assert_close(keep_p, torch.tensor([[0, 1]]))
        self.assertIn(2, keep_c[0].tolist())

    def test_proxy_reversal_is_retained_as_a_known_pm_failure(self):
        _, _, q, k = self.scenario()
        p = attention_mean(q[None], k[None])
        c = attention_mean(q.mean(0).repeat(2, 1)[None], k[None])
        ip = select_fixed_budget(p, 2, sink_tokens=0, recent_tokens=0)
        ic = select_fixed_budget(c, 2, sink_tokens=0, recent_tokens=0)
        real = attention_mean(torch.tensor([[[0., 0., 8., 0.]]]), k[None])
        retained_p, retained_c = real.gather(1, ip).sum(), real.gather(1, ic).sum()
        self.assertLess(float(retained_p), .00068)
        self.assertGreater(float(retained_c), .9996)

    def test_symmetric_two_key_null_case(self):
        _, _, q, k = self.scenario()
        p = attention_mean(q[None], k[:2][None])
        c = attention_mean(q.mean(0).repeat(2, 1)[None], k[:2][None])
        torch.testing.assert_close(p, torch.tensor([[.5, .5]]))
        torch.testing.assert_close(p, c)

    def test_covariance_loss_identity_and_psd_for_product_proxy(self):
        torch.manual_seed(51)
        u = torch.randn(5, 4)
        rope = NativeRope(torch.tensor([.7]), rotary_dim=2, layout="interleaved")
        matrices = torch.stack([rope.apply(torch.eye(4), torch.tensor(t)).T for t in (0, 1, 2, 3)])
        mean_r = matrices.mean(0)
        mixed = torch.einsum("tij,mj->tmi", matrices, u).flatten(0, 1)
        collapsed = u @ mean_r.T
        cov = lambda x: (x - x.mean(0)).T @ (x - x.mean(0)) / x.shape[0]
        delta = cov(mixed) - cov(collapsed)
        second = u.T @ u / u.shape[0]
        centered = matrices - mean_r
        expected = torch.einsum("tij,jk,tlk->il", centered, second, centered) / len(matrices)
        torch.testing.assert_close(delta, expected, atol=3e-7, rtol=3e-6)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(delta).min()), -3e-7)


if __name__ == "__main__":
    unittest.main()
