#!/usr/bin/env python3
"""Registration tests for the 500M-token FMRoPE / EVQ run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    aggregate_exact_range,
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
        expected_parameters = {
            "151m": 151_898_880,
            "350m": 350_112_000,
        }[SPEC.model_tier]
        self.assertEqual(estimate_parameter_count(), expected_parameters)
        expected_budget = {
            "151m": (7_629, 1_953_024, 499_974_144, 498_021_120, 30_516, 762),
            "350m": (15_258, 3_906_048, 999_948_288, 996_042_240, 61_032, 1_525),
        }[SPEC.model_tier]
        self.assertEqual(
            (
                SPEC.optimizer_steps,
                SPEC.train_rows,
                SPEC.train_tokens,
                SPEC.prediction_tokens,
                SPEC.micro_steps,
                SPEC.warmup_steps,
            ),
            expected_budget,
        )
        for arm in ARMS:
            self.assertEqual(meta_parameter_count(arm), expected_parameters)

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

    def test_three_seed_exact_range_aggregate_uses_training_seeds(self):
        comparisons = {
            "anchored_cosh_fixed_minus_fmrope_fixed": -0.2,
            "anchored_cosh_target_minus_fmrope_target": 0.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for seed in (42, 137, 256):
                paired = {
                    name: {
                        str(length): {
                            "mean_left_minus_right_tail_nll": value
                        }
                        for length in SPEC.eval_lengths
                    }
                    for name, value in comparisons.items()
                }
                path = root / f"seed_{seed}.json"
                path.write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "data_manifest_sha256": "0" * 64,
                            "eval_lengths": list(SPEC.eval_lengths),
                            "summary": {"paired": paired},
                        }
                    )
                )
                inputs.append(path)
            output = aggregate_exact_range(
                SimpleNamespace(
                    inputs=inputs,
                    output_dir=root / "aggregate",
                )
            )
            self.assertEqual(
                output["decision"],
                "THREE_SEED_SHAPE_EFFECT_WITHOUT_TARGET_SYNERGY",
            )
            self.assertEqual(output["seeds"], [42, 137, 256])


if __name__ == "__main__":
    unittest.main()
