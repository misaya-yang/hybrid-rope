#!/usr/bin/env python3
"""Regression tests for the EVQ-Cosh x FMRoPE combination arm."""

from __future__ import annotations

import hashlib
import unittest

import torch

from rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.protocol import (
    ARMS,
    ARM_CONDITIONS,
    SPEC,
    estimate_parameter_count,
    runtime_frequency,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment import (
    build_model,
    meta_parameter_count,
    trainable_state_sha256,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256.run_experiment import (
    build_model as build_parent_model,
)
from rebuttal.rebuttal_0723.experiments.geo_rope_contract import (
    EVQ_COSH,
    build_training_inv_freq,
)


ARM = "evq_cosh_tau4_fmrope_base256"


class TestComboProtocol(unittest.TestCase):
    def test_registered_arm_and_model_size(self):
        self.assertEqual(ARMS, (ARM,))
        self.assertEqual(
            ARM_CONDITIONS[ARM],
            ("fixed_train_base", "target_matched_base"),
        )
        self.assertEqual(estimate_parameter_count(), 151_898_880)
        self.assertEqual(meta_parameter_count(ARM), 151_898_880)

    def test_training_frequency_is_evq_shape_at_base_256(self):
        expected = build_training_inv_freq(
            EVQ_COSH,
            head_dim=SPEC.head_dim,
            base=SPEC.fmrope_train_base,
            tau=SPEC.evq_tau,
        )
        actual = training_inv_freq(ARM)
        self.assertTrue(torch.equal(actual, expected))

    def test_target_retarget_preserves_cosh_shape(self):
        train = training_inv_freq(ARM)
        for length in SPEC.eval_lengths:
            fixed, fixed_scale, fixed_meta = runtime_frequency(
                ARM, "fixed_train_base", length
            )
            target, target_scale, target_meta = runtime_frequency(
                ARM, "target_matched_base", length
            )
            expected = build_training_inv_freq(
                EVQ_COSH,
                head_dim=SPEC.head_dim,
                base=float(length),
                tau=SPEC.evq_tau,
            )
            self.assertTrue(torch.equal(fixed, train))
            self.assertTrue(torch.equal(target, expected))
            self.assertEqual(fixed_scale, 1.0)
            self.assertEqual(target_scale, 1.0)
            self.assertEqual(target_meta["inference_base"], float(length))
            self.assertEqual(fixed_meta["inference_base"], 256.0)
        in_domain, _, _ = runtime_frequency(
            ARM, "target_matched_base", SPEC.train_length
        )
        self.assertTrue(torch.equal(in_domain, train))

    def test_trainable_initialization_matches_parent_arms(self):
        combo = build_model(ARM, seed=SPEC.seed)
        parent_hashes = {
            trainable_state_sha256(
                build_parent_model(parent_arm, seed=SPEC.seed)
            )
            for parent_arm in (
                "paper_geo_base500k",
                "fmrope_base256",
                "evq_cosh_tau4_paper_grid_base500k",
            )
        }
        self.assertEqual(len(parent_hashes), 1)
        self.assertEqual(
            trainable_state_sha256(combo),
            next(iter(parent_hashes)),
        )

    def test_frequency_receipt_is_stable(self):
        digest = hashlib.sha256(
            training_inv_freq(ARM).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
