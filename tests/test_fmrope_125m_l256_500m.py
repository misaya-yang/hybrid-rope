#!/usr/bin/env python3
"""Registration tests for the 500M-token FMRoPE / EVQ run."""

from __future__ import annotations

import unittest

import torch

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    ARMS,
    SPEC,
    anchored_cosh_inv_freq,
    estimate_parameter_count,
    runtime_frequency,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import (
    meta_parameter_count,
    summarize_records,
)
from rebuttal.rebuttal_0723.experiments.geo_rope_contract import (
    EVQ_COSH,
    PAPER_GEO,
    build_training_inv_freq,
    std_geo_inv_freq,
)


class TestProtocol(unittest.TestCase):
    def test_budget_and_parameter_count(self):
        self.assertEqual(estimate_parameter_count(), 151_898_880)
        self.assertEqual(SPEC.optimizer_steps, 7_629)
        self.assertEqual(SPEC.train_rows, 1_953_024)
        self.assertEqual(SPEC.train_tokens, 499_974_144)
        self.assertEqual(SPEC.prediction_tokens, 498_021_120)
        self.assertEqual(SPEC.micro_steps, 30_516)
        self.assertEqual(SPEC.warmup_steps, 762)
        for arm in ARMS:
            self.assertEqual(meta_parameter_count(arm), 151_898_880)

    def test_registered_training_schedules(self):
        self.assertTrue(
            torch.equal(
                training_inv_freq("paper_geo_base500k"),
                build_training_inv_freq(
                    PAPER_GEO, head_dim=64, base=500_000.0
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                training_inv_freq("fmrope_base256"),
                std_geo_inv_freq(64, 256.0),
            )
        )
        self.assertTrue(
            torch.equal(
                training_inv_freq(
                    "evq_cosh_tau4_paper_grid_base500k"
                ),
                build_training_inv_freq(
                    EVQ_COSH,
                    head_dim=64,
                    base=500_000.0,
                    tau=4.0,
                ),
            )
        )

    def test_fmrope_target_retarget(self):
        for length in SPEC.eval_lengths:
            target, scale, meta = runtime_frequency(
                "fmrope_base256", "target_matched_base", length
            )
            self.assertTrue(
                torch.equal(target, std_geo_inv_freq(64, float(length)))
            )
            self.assertEqual(scale, 1.0)
            self.assertEqual(meta["inference_base"], float(length))

    def test_anchored_cosh_preserves_fmrope_range(self):
        arm = "anchored_cosh_tau4_fmrope_range"
        train = training_inv_freq(arm)
        geo = std_geo_inv_freq(SPEC.head_dim, SPEC.fmrope_train_base)
        self.assertEqual(float(train[0]), float(geo[0]))
        self.assertEqual(float(train[-1]), float(geo[-1]))
        self.assertFalse(torch.equal(train, geo))
        self.assertTrue(torch.all(torch.diff(train) < 0))
        for length in SPEC.eval_lengths:
            target, scale, meta = runtime_frequency(
                arm, "target_matched_range", length
            )
            expected = anchored_cosh_inv_freq(
                SPEC.head_dim, float(length), SPEC.evq_tau
            )
            target_geo = std_geo_inv_freq(SPEC.head_dim, float(length))
            self.assertTrue(torch.equal(target, expected))
            self.assertEqual(float(target[0]), float(target_geo[0]))
            self.assertEqual(float(target[-1]), float(target_geo[-1]))
            self.assertEqual(scale, 1.0)
            self.assertEqual(meta["inference_base"], float(length))
        fixed, _, _ = runtime_frequency(
            arm, "fixed_train_range", SPEC.train_length
        )
        target, _, _ = runtime_frequency(
            arm, "target_matched_range", SPEC.train_length
        )
        self.assertTrue(torch.equal(fixed, target))

    def test_single_arm_summary_does_not_require_old_checkpoints(self):
        arm = "anchored_cosh_tau4_fmrope_range"
        records = []
        for condition in ("fixed_train_range", "target_matched_range"):
            for length in SPEC.eval_lengths:
                records.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "length": length,
                        "anchor": 10_000,
                        "full_nll": 4.0,
                        "tail_nll": 4.0,
                        "tail_target_sha256": "0" * 64,
                    }
                )
        summary, markdown = summarize_records(records, arms=(arm,))
        self.assertEqual(set(summary["aggregate"]), {arm})
        self.assertFalse(summary["paired"])
        self.assertIn("Range-anchored Cosh", markdown)


if __name__ == "__main__":
    unittest.main()
