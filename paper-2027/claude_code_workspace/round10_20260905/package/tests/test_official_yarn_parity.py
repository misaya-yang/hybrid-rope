#!/usr/bin/env python3
"""Parity gates for official YaRN equations vs pinned yarn@995db5b formulas."""

from __future__ import annotations

import math
import unittest

import torch

from scripts.lib.rope.official_yarn import (
    find_correction_range,
    get_mscale,
    native_endpoint_inv_freq,
    official_yarn_on_inv_freq,
    official_yarn_on_native_grid,
    parity_vs_official_source,
    repo_fixed_ramp_inv_freq,
    shared_index_yarn_control_on_inv_freq,
    virtual_dim_from_inv_freq,
)
from scripts.lib.rope.schedules import evq_cosh_inv_freq


class TestOfficialYarnParity(unittest.TestCase):
    def test_parity_primary1_hparams(self):
        r = parity_vs_official_source(
            head_dim=64,
            base=500000.0,
            scale=8.0,
            original_max_position_embeddings=2048,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        self.assertTrue(r["parity_ok"], r)
        # Audit note: at d=64,b=500k,L=2048 official transition ≈ channels 5–15
        self.assertEqual(r["low"], 5)
        self.assertEqual(r["high"], 15)
        self.assertAlmostEqual(r["mscale"], 1.0 + 0.1 * math.log(8.0), places=12)

    def test_mscale_formula(self):
        self.assertEqual(get_mscale(1.0), 1.0)
        self.assertAlmostEqual(get_mscale(8.0), 1.0 + 0.1 * math.log(8.0), places=12)

    def test_virtual_dim_recovers_native_indices(self):
        inv = native_endpoint_inv_freq(64, 500000.0)
        j = virtual_dim_from_inv_freq(inv, head_dim=64, base=500000.0)
        target = torch.arange(32, dtype=torch.float64)
        self.assertTrue(torch.allclose(j, target, atol=1e-6, rtol=1e-6))

    def test_invfreq_path_matches_native_path(self):
        inv_n, m_n, _ = official_yarn_on_native_grid(
            head_dim=64, base=500000.0, scale=8.0
        )
        inv_t, m_t, meta = official_yarn_on_inv_freq(
            native_endpoint_inv_freq(64, 500000.0),
            head_dim=64,
            base=500000.0,
            scale=8.0,
        )
        self.assertTrue(torch.allclose(inv_n, inv_t, atol=1e-9, rtol=1e-7))
        self.assertEqual(m_n, m_t)
        self.assertTrue(meta["near_native_grid"])
        self.assertEqual(meta["mode"], "official_yarn_native")

    def test_native_path_matches_representative_official_float32_execution(self):
        head_dim = 128
        base = 500000.0
        scale = 2.0
        low, high = find_correction_range(32.0, 1.0, head_dim, base, 8192)
        pos_freqs = base ** (
            torch.arange(0, head_dim, 2).float() / float(head_dim)
        )
        inv_extra = 1.0 / pos_freqs
        inv_inter = 1.0 / (scale * pos_freqs)
        ramp = torch.clamp(
            (torch.arange(head_dim // 2).float() - float(low))
            / (float(high) - float(low)),
            0.0,
            1.0,
        )
        mask = 1.0 - ramp
        expected = inv_inter * (1.0 - mask) + inv_extra * mask

        actual, _, _ = official_yarn_on_native_grid(
            head_dim=head_dim,
            base=base,
            scale=scale,
            original_max_position_embeddings=8192,
        )

        self.assertLessEqual(float((actual.float() - expected).abs().max()), 1e-7)

    def test_evq_is_yarn_derived_not_native(self):
        inv = evq_cosh_inv_freq(head_dim=64, tau=1.5, base=500000.0, midpoint=True)
        _, _, meta = official_yarn_on_inv_freq(
            inv, head_dim=64, base=500000.0, scale=8.0
        )
        self.assertFalse(meta["near_native_grid"])
        self.assertEqual(meta["mode"], "yarn_derived_virtual_dim")
        self.assertIn("YaRN-derived", meta["label"])

    def test_midpoint_geo_is_yarn_derived(self):
        inv = evq_cosh_inv_freq(head_dim=64, tau=0.0, base=500000.0, midpoint=True)
        native = native_endpoint_inv_freq(64, 500000.0)
        self.assertFalse(torch.allclose(inv, native, atol=1e-6, rtol=1e-5))
        _, _, meta = official_yarn_on_inv_freq(
            inv, head_dim=64, base=500000.0, scale=8.0
        )
        self.assertEqual(meta["mode"], "yarn_derived_virtual_dim")

    def test_repo_fixed_ramp_differs_from_official(self):
        inv = native_endpoint_inv_freq(64, 500000.0)
        off, _, _ = official_yarn_on_native_grid(head_dim=64, base=500000.0, scale=8.0)
        repo, m_repo, meta = repo_fixed_ramp_inv_freq(inv, scale=8.0)
        self.assertEqual(m_repo, 1.0)
        self.assertEqual(meta["mode"], "repo_fixed_ramp")
        self.assertFalse(torch.allclose(off, repo, atol=1e-5, rtol=1e-4))
        # Official transition ~5-15; repo uses ~20%-90% of 32 = channels 6-28
        self.assertEqual(meta["channel_start"], 6)
        self.assertEqual(meta["channel_end"], 28)

    def test_shared_index_control_is_exactly_official_on_native(self):
        for head_dim in (16, 64):
            for scale in (2.0, 4.0, 8.0):
                native = native_endpoint_inv_freq(head_dim, 500000.0)
                expected, expected_mscale, expected_meta = (
                    official_yarn_on_native_grid(
                        head_dim=head_dim,
                        base=500000.0,
                        scale=scale,
                        original_max_position_embeddings=4096,
                    )
                )
                actual, actual_mscale, actual_meta = (
                    shared_index_yarn_control_on_inv_freq(
                        native,
                        head_dim=head_dim,
                        base=500000.0,
                        scale=scale,
                        original_max_position_embeddings=4096,
                    )
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual_mscale, expected_mscale)
                self.assertEqual(actual_meta["low"], expected_meta["low"])
                self.assertEqual(actual_meta["high"], expected_meta["high"])
                self.assertTrue(actual_meta["official_on_input"])
                self.assertEqual(
                    actual_meta["mode"], "official_yarn_native"
                )

    def test_shared_index_control_uses_identical_mask_on_evq(self):
        head_dim = 16
        base = 500000.0
        scale = 8.0
        native = native_endpoint_inv_freq(head_dim, base)
        evq = evq_cosh_inv_freq(
            head_dim=head_dim,
            tau=1.414,
            base=base,
            midpoint=True,
        ).to(torch.float64)
        native_out, native_mscale, _ = (
            shared_index_yarn_control_on_inv_freq(
                native,
                head_dim=head_dim,
                base=base,
                scale=scale,
                original_max_position_embeddings=4096,
            )
        )
        evq_out, evq_mscale, meta = (
            shared_index_yarn_control_on_inv_freq(
                evq,
                head_dim=head_dim,
                base=base,
                scale=scale,
                original_max_position_embeddings=4096,
            )
        )
        native_weight = (native_out - native / scale) / (
            native - native / scale
        )
        evq_weight = (evq_out - evq / scale) / (evq - evq / scale)
        self.assertTrue(
            torch.allclose(
                native_weight, evq_weight, atol=1e-12, rtol=1e-12
            )
        )
        self.assertEqual(native_mscale, evq_mscale)
        self.assertFalse(meta["official_on_input"])
        self.assertTrue(meta["shared_index_mask"])
        self.assertEqual(
            meta["mode"], "shared_index_yarn_component_control"
        )
        self.assertIn("NOT official YaRN", meta["label"])

    def test_shared_index_control_validates_table_length(self):
        with self.assertRaisesRegex(ValueError, "expected 8"):
            shared_index_yarn_control_on_inv_freq(
                torch.ones(7),
                head_dim=16,
                base=500000.0,
                scale=8.0,
            )

    def test_shared_index_control_does_not_infer_native_from_tolerance(self):
        native = native_endpoint_inv_freq(16, 500000.0)
        almost_native = native.clone()
        almost_native[0] += 1e-10
        _, _, meta = shared_index_yarn_control_on_inv_freq(
            almost_native,
            head_dim=16,
            base=500000.0,
            scale=8.0,
            original_max_position_embeddings=4096,
        )
        self.assertFalse(meta["official_on_input"])
        self.assertEqual(
            meta["mode"], "shared_index_yarn_component_control"
        )

    def test_shared_index_control_scale_one_is_identity(self):
        evq = evq_cosh_inv_freq(
            head_dim=16,
            tau=1.414,
            base=500000.0,
            midpoint=True,
        )
        actual, mscale, meta = shared_index_yarn_control_on_inv_freq(
            evq,
            head_dim=16,
            base=500000.0,
            scale=1.0,
            original_max_position_embeddings=4096,
        )
        self.assertTrue(torch.equal(actual, evq.to(torch.float64)))
        self.assertEqual(mscale, 1.0)
        self.assertEqual(meta["mode"], "identity")
        self.assertFalse(meta["official_on_input"])

    def test_correction_range_clamps_like_official(self):
        low, high = find_correction_range(32, 1, 64, 500000.0, 2048)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 63)


if __name__ == "__main__":
    unittest.main()
