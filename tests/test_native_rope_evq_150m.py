#!/usr/bin/env python3
"""Regression gates for the native-RoPE versus endpoint-EVQ 150M pilot."""

from __future__ import annotations

import math
import random
import unittest

import torch

from experiments.native_rope_evq_150m.protocol import (
    LEGACY_PASSKEY_HASH_MULTIPLIER,
    SPEC,
    estimate_parameter_count,
    get_arm_inv_freq,
    legacy_passkey_indices,
)
from scripts.lib.rope.schedules import geometric_inv_freq


class TestNativeRopeEvq150MProtocol(unittest.TestCase):
    def test_model_and_training_counts_are_exact(self):
        self.assertEqual(estimate_parameter_count(), 151_898_880)
        self.assertEqual(SPEC.train_rows, 244_140)
        self.assertEqual(SPEC.train_tokens, 499_998_720)
        self.assertEqual(SPEC.batch_size, 60)
        self.assertEqual(SPEC.optimizer_steps, 4_069)
        self.assertEqual(SPEC.seq_len, 2_048)

    def test_legacy_passkey_selection_matches_previous_dataset(self):
        indices = legacy_passkey_indices(SPEC.train_rows)
        expected = tuple(
            i
            for i in range(SPEC.train_rows)
            if random.Random(i * LEGACY_PASSKEY_HASH_MULTIPLIER + 1).random()
            < SPEC.passkey_mix_ratio
        )
        self.assertEqual(indices, expected)
        self.assertEqual(len(indices), 4_926)
        self.assertEqual(len(indices) * SPEC.seq_len, 10_088_448)

    def test_native_arm_is_standard_endpoint_rope(self):
        actual = get_arm_inv_freq("native_rope")
        expected = geometric_inv_freq(
            head_dim=SPEC.head_dim,
            base=SPEC.rope_base,
            dtype=torch.float64,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(float(actual[0]), 1.0)

    def test_endpoint_evq_is_finite_distinct_and_tau_zero_recovers_native(self):
        evq = get_arm_inv_freq("endpoint_evq_tau1p5")
        native = get_arm_inv_freq("native_rope")
        evq_zero = get_arm_inv_freq("endpoint_evq_tau1p5", tau_override=0.0)
        self.assertTrue(torch.isfinite(evq).all())
        self.assertFalse(torch.allclose(evq, native, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(evq_zero, native, atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(SPEC.evq_tau, 1.5)
        self.assertAlmostEqual(
            SPEC.head_dim / math.sqrt(SPEC.seq_len), math.sqrt(2.0), places=12
        )

    def test_unknown_arm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            get_arm_inv_freq("geo")


if __name__ == "__main__":
    unittest.main()
