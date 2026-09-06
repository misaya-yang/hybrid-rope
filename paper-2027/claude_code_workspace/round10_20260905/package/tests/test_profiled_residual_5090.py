#!/usr/bin/env python3
"""Focused CPU checks for the profiled residual schedule."""

from __future__ import annotations

import unittest

import torch

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import training_inv_freq
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.protocol import (
    BASE_MULTIPLIERS,
    EVAL_LENGTHS,
    LAMBDA_RATIOS,
    candidate_grid,
    candidate_inv_freq,
    decomposition_report,
    frequency_components,
)
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.run_experiment import (
    select_candidates,
)


class TestProfiledResidual(unittest.TestCase):
    def test_orthogonal_decomposition_and_norm_match(self):
        components = frequency_components()
        q_full = components["q_full"]
        q_affine = components["q_affine"]
        residual = components["cosh_residual"]
        self.assertTrue(torch.allclose(q_full, q_affine + residual, atol=1e-12, rtol=0))
        coordinate = torch.linspace(0.0, 1.0, residual.numel(), dtype=torch.float64)
        self.assertAlmostEqual(float(residual.sum()), 0.0, places=10)
        self.assertAlmostEqual(float(torch.dot(coordinate, residual)), 0.0, places=10)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(residual)),
            float(torch.linalg.vector_norm(components["band_residual"])),
            places=10,
        )

    def test_profile_base_recovers_fmrope_training_table(self):
        actual = candidate_inv_freq(
            family="base", length=256, base_multiplier=1.0, lambda_ratio=0.0
        )
        self.assertTrue(torch.equal(actual, training_inv_freq("fmrope_base256")))

    def test_lambda_zero_is_nested_base(self):
        for length in EVAL_LENGTHS:
            for multiplier in BASE_MULTIPLIERS:
                base = candidate_inv_freq(
                    family="base",
                    length=length,
                    base_multiplier=multiplier,
                    lambda_ratio=0.0,
                )
                for family in ("cosh_residual", "band_residual"):
                    nested = candidate_inv_freq(
                        family=family,
                        length=length,
                        base_multiplier=multiplier,
                        lambda_ratio=0.0,
                    )
                    self.assertTrue(torch.equal(base, nested))

    def test_grid_has_each_lambda_with_reprofiled_base(self):
        rows = candidate_grid(8_192)
        for family in ("cosh_residual", "band_residual"):
            for ratio in LAMBDA_RATIOS:
                candidates = [
                    row
                    for row in rows
                    if row["family"] == family and row["lambda_ratio"] == ratio
                ]
                self.assertGreaterEqual(len(candidates), 1)

    def test_selection_profiles_base_per_lambda(self):
        records = []
        for length in EVAL_LENGTHS:
            for family in ("base", "raw_evq"):
                records.append(
                    {
                        "arm": "fmrope_base256",
                        "length": length,
                        "family": family,
                        "base_multiplier": 1.0,
                        "lambda_ratio": 0.0 if family == "base" else 1.0,
                        "strictly_monotonic": True,
                        "inv_freq_sha256": "a" * 64,
                        "mean_tail_nll": 2.0,
                    }
                )
            for family in ("cosh_residual", "band_residual"):
                for index, ratio in enumerate(LAMBDA_RATIOS):
                    records.append(
                        {
                            "arm": "fmrope_base256",
                            "length": length,
                            "family": family,
                            "base_multiplier": BASE_MULTIPLIERS[index % len(BASE_MULTIPLIERS)],
                            "lambda_ratio": ratio,
                            "strictly_monotonic": True,
                            "inv_freq_sha256": "b" * 64,
                            "mean_tail_nll": 1.0 + abs(ratio - 0.5),
                        }
                    )
        result = select_candidates(records)
        selected = result["selected"]["fmrope_base256"]["8192"]
        self.assertEqual(selected["cosh_residual"]["lambda_ratio"], 0.5)
        self.assertEqual(
            len(result["per_lambda_reprofiled"]["fmrope_base256"]["8192"]["cosh_residual"]),
            len(LAMBDA_RATIOS),
        )

    def test_decomposition_report_is_complete(self):
        report = decomposition_report()
        self.assertEqual(set(report["tables"]), {"paper_geo", "affine_evq", "full_evq"})
        for table in report["tables"].values():
            self.assertEqual(len(table["channels"]), 32)
            self.assertEqual(set(map(int, table["by_length"])), set(EVAL_LENGTHS))


if __name__ == "__main__":
    unittest.main()
