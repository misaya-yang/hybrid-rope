#!/usr/bin/env python3
"""CPU tests for the automatic three-arm temporal evaluation summary."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from experiments.lora_evq_v2.eval_temporal_holdout_three_arm import (
    build_three_arm_comparisons,
    temporal_arm_contract,
)


def _metric(nll: float) -> dict:
    return {"nll": nll, "ppl": math.exp(nll), "nll_sum": nll * 10, "scored_tokens": 10}


def _arm(name: str, values: tuple[float, float]) -> dict:
    prefixes = {"8K": _metric(values[0]), "16K": _metric(values[1]), "32K": _metric(values[1])}
    return {
        "arm": name,
        "domain_macro": {"prefixes": prefixes, "buckets": {}},
        "domains": {
            "domain_a": {
                "prefixes": prefixes,
                "buckets": {},
                "packs": [
                    {"pack_index": 0, "prefixes": prefixes, "buckets": {}},
                    {"pack_index": 1, "prefixes": prefixes, "buckets": {}},
                ],
            }
        },
    }


class ThreeArmSummaryTests(unittest.TestCase):
    def test_temporal_arm_contract_is_seed_parameterized(self):
        contract = temporal_arm_contract(geo_seed=42, evq_seed=43)
        self.assertEqual(contract["geo_lora"]["adapter"], "geo_longalpaca_s42")
        self.assertEqual(
            contract["evq_lora"]["adapter"],
            "evq_longalpaca_tau1414_s43",
        )
        with self.assertRaisesRegex(ValueError, "42, 43, or 44"):
            temporal_arm_contract(geo_seed=42, evq_seed=45)

    def test_reports_geo_adaptation_and_evq_incremental_nll(self):
        arms = {
            "geo_base": _arm("geo_base", (2.0, 4.0)),
            "geo_lora": _arm("geo_lora", (1.9, 3.7)),
            "evq_lora": _arm("evq_lora", (1.95, 3.4)),
        }

        summary = build_three_arm_comparisons(
            arms,
            geo_base="geo_base",
            geo_lora="geo_lora",
            evq_lora="evq_lora",
        )

        macro_16k = summary["domain_macro"]["16K"]
        self.assertAlmostEqual(macro_16k["delta_nll_geo_lora_minus_geo"], -0.3)
        self.assertAlmostEqual(macro_16k["delta_nll_evq_lora_minus_geo_lora"], -0.3)
        self.assertAlmostEqual(macro_16k["delta_nll_evq_lora_minus_geo"], -0.6)
        self.assertEqual(len(summary["domains"]["domain_a"]["packs"]), 2)
        self.assertAlmostEqual(
            summary["domains"]["domain_a"]["packs"][0]["prefixes"]["8K"]
            ["delta_nll_evq_lora_minus_geo_lora"],
            0.05,
        )

    def test_rejects_incomplete_three_arm_inputs(self):
        arms = {
            "geo_base": _arm("geo_base", (2.0, 4.0)),
            "geo_lora": _arm("geo_lora", (1.9, 3.7)),
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_three_arm_comparisons(
                arms,
                geo_base="geo_base",
                geo_lora="geo_lora",
                evq_lora="evq_lora",
            )

    def test_launcher_fail_closes_artifacts_lock_and_output(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/2026-07/06_lora_temporal_three_arm_eval.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "EVQ_GEO_LONGALPACA_ADAPTER",
            "EVQ_EVQ_LONGALPACA_ADAPTER",
            "collection_manifest.json",
            "refusing to overwrite evaluation",
            "flock -n 9",
            "eval_temporal_holdout_three_arm.py",
            "EVQ_LONGALPACA_EXPECTED_SEED",
            "EVQ_GEO_LONGALPACA_EXPECTED_SEED",
            "EVQ_EVQ_LONGALPACA_EXPECTED_SEED",
            "--expected_geo_seed",
            "--expected_evq_seed",
        ):
            self.assertIn(required, launcher)


if __name__ == "__main__":
    unittest.main()
