#!/usr/bin/env python3
"""Registration tests for the 500M-token FMRoPE / EVQ run."""

from __future__ import annotations

import unittest

import torch

from rebuttal.rebuttal_0723.fmrope_125m_l256_500m.protocol import (
    ARMS,
    SPEC,
    estimate_parameter_count,
    runtime_frequency,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.fmrope_125m_l256_500m.run_experiment import (
    meta_parameter_count,
)
from rebuttal.rebuttal_0723.geo_rope_contract import (
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


if __name__ == "__main__":
    unittest.main()
