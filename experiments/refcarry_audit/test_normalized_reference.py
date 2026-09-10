"""CPU semantic checks for the candidate read composition, not model evidence."""
import unittest

import torch

from experiments.refcarry_audit.normalized_reference import (
    ReferenceBias, References, factored_bias, normalized_mixture,
    position_features, relative_features, retain_references,
)


class ReadCompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(709)
        self.dtype = torch.float64

    def test_independent_normalizers_preserve_alternatives(self):
        branches = torch.tensor([[10., -10., 0.], [-10., 10., 0.]], dtype=self.dtype)
        masses = torch.tensor([.5, .5], dtype=self.dtype)
        values = torch.eye(3, dtype=self.dtype)
        _, mixture = normalized_mixture(torch.zeros(3, dtype=self.dtype), branches, values, masses)
        moment_read = (masses @ branches).softmax(-1)
        self.assertLess(mixture[2].item(), 5e-5)
        self.assertAlmostEqual(moment_read[2].item(), 1/3)
        torch.testing.assert_close(mixture, masses @ branches.softmax(-1))

    def test_omitted_reference_mass_is_not_renormalized(self):
        p0 = torch.tensor([1., -1.], dtype=self.dtype)
        pr = torch.tensor([[-1., 1.]], dtype=self.dtype)
        _, got = normalized_mixture(p0, pr, torch.eye(2, dtype=self.dtype), torch.tensor([.2], dtype=self.dtype))
        torch.testing.assert_close(got, .8*p0.softmax(-1)+.2*pr[0].softmax(-1))

    def test_sink_is_in_each_normalizer(self):
        native = torch.tensor([0., 1.], dtype=self.dtype)
        refs = torch.tensor([[4., -2.], [-3., 2.]], dtype=self.dtype)
        masses = torch.tensor([.25, .5], dtype=self.dtype)
        values = torch.tensor([[2., 3.], [-1., 4.]], dtype=self.dtype)
        got, probabilities = normalized_mixture(native, refs, values, masses, sink_logit=3.)
        all_logits = torch.cat((native[None], refs), 0)
        all_logits = torch.cat((all_logits, torch.full((3, 1), 3., dtype=self.dtype)), 1)
        expected = torch.tensor([.25, .25, .5], dtype=self.dtype) @ all_logits.softmax(-1)
        torch.testing.assert_close(probabilities, expected[:2])
        torch.testing.assert_close(got, expected[:2] @ values)
        self.assertLess(probabilities.sum().item(), 1.)

    def test_retention_ties_ignore_sparse_storage_order(self):
        p = torch.tensor([17, 3, 11, 5])
        w = torch.tensor([.2, .3, .3, .1], dtype=self.dtype)
        r = retain_references(p, w, 2, 20)
        self.assertEqual(r.positions.tolist(), [3, 11])
        self.assertAlmostEqual(r.residual_mass.item(), .4)
        order = torch.tensor([3, 2, 0, 1])
        r2 = retain_references(p[order], w[order], 2, 20)
        torch.testing.assert_close(r.positions, r2.positions)
        torch.testing.assert_close(r.masses, r2.masses)

    def test_future_and_duplicate_references_rejected(self):
        for positions in (torch.tensor([2, 21]), torch.tensor([2, 2])):
            with self.assertRaises(ValueError):
                retain_references(positions, torch.tensor([.4, .4]), 1, 20)

    def test_fixed_sparse_support(self):
        native = torch.tensor([1., -torch.inf, 0.], dtype=self.dtype)
        with self.assertRaises(ValueError):
            normalized_mixture(native, torch.zeros((1, 3), dtype=self.dtype),
                               torch.eye(3, dtype=self.dtype), torch.ones(1, dtype=self.dtype))
        branch = torch.tensor([[-2., -torch.inf, 3.]], dtype=self.dtype)
        _, got = normalized_mixture(native, branch, torch.eye(3, dtype=self.dtype),
                                    torch.tensor([.5], dtype=self.dtype))
        self.assertEqual(got[1].item(), 0.)

    def test_relative_features_translation_invariance(self):
        p = torch.tensor([0, 3, 12, 101])
        a = torch.tensor([2, 7])
        freq = torch.tensor([.3, .07], dtype=self.dtype)
        torch.testing.assert_close(relative_features(p, a, freq), relative_features(p+8192, a+8192, freq))

    def test_factored_bias_matches_direct_and_features_append(self):
        positions = torch.tensor([0, 3, 12, 101, 139])
        anchors = torch.tensor([2, 7, 51])
        freq = torch.tensor([.3, .07, .003], dtype=self.dtype)
        coefficients = torch.randn(6, dtype=self.dtype)
        features = position_features(positions, freq)
        direct = relative_features(positions, anchors, freq) @ coefficients
        factored = factored_bias(coefficients, features, anchors, freq)
        torch.testing.assert_close(factored, direct)
        self.assertTrue(torch.equal(position_features(positions[:3], freq), features[:3]))

    def test_large_origin_does_not_erase_small_relative_offsets(self):
        freq = 1/1000000**(torch.arange(0, 64, 2).float()/64)
        keys = torch.arange(999960, 1000001)
        anchors = torch.tensor([999951, 999999])
        u = torch.randn(64, generator=torch.Generator().manual_seed(71))
        expected = relative_features(keys, anchors, freq.double()) @ u.double()
        got = factored_bias(u, position_features(keys, freq), anchors, freq)
        torch.testing.assert_close(got.double(), expected, rtol=1e-5, atol=1e-5)

    def test_zero_initialization_native_and_trainable(self):
        module = ReferenceBias(7, torch.tensor([.3, .07], dtype=self.dtype))
        h = torch.randn(7, dtype=self.dtype)
        values = torch.randn(5, 3, dtype=self.dtype)
        native = torch.randn(5, dtype=self.dtype)
        refs = References(torch.tensor([2, 8]), torch.tensor([.2, .6], dtype=self.dtype), torch.tensor(.2))
        out, prob = module(h, native, values, torch.tensor([0, 2, 5, 9, 12]), refs)
        self.assertTrue(torch.equal(prob, native.softmax(-1)))
        torch.testing.assert_close(out, native.softmax(-1) @ values)
        out.square().sum().backward()
        grad = module.projection.weight.grad
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(grad.norm().item(), 1e-8)

    def test_branch_logit_offsets_do_not_reweight_origins(self):
        native = torch.randn(7, dtype=self.dtype)
        branches = torch.randn(3, 7, dtype=self.dtype)
        mass = torch.tensor([.1, .2, .4], dtype=self.dtype)
        values = torch.randn(7, 2, dtype=self.dtype)
        a, _ = normalized_mixture(native, branches, values, mass)
        b, _ = normalized_mixture(native+7, branches+torch.tensor([[19.], [-7.], [3.]]), values, mass)
        torch.testing.assert_close(a, b)

    def test_truncation_total_variation_bound(self):
        for _ in range(40):
            native = torch.randn(11, dtype=self.dtype)
            logits = torch.randn(5, 11, dtype=self.dtype)*3
            mu = torch.rand(5, dtype=self.dtype)
            mu /= mu.sum()
            full = mu @ logits.softmax(-1)
            ids = mu.argsort(descending=True)[:2]
            _, approximate = normalized_mixture(native, logits[ids], torch.eye(11, dtype=self.dtype), mu[ids])
            epsilon = 1-mu[ids].sum()
            tv = .5*(full-approximate).abs().sum()
            self.assertLessEqual(tv.item(), epsilon.item()+1e-12)


if __name__ == "__main__":
    unittest.main()
