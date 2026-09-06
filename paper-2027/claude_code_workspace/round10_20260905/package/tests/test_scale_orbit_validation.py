import unittest

import numpy as np

from scripts.analysis.scale_orbit_validation import (
    analyse_table,
    float32_boundary_count,
    float32_power_orbit_classes,
    minimum_q_chain,
    paired_bootstrap,
    same_support_geometric,
    ulp_jittered_chain,
    validate_same_support,
)


class ScaleOrbitValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.native = np.power(
            np.float32(500_000.0),
            -np.arange(64, dtype=np.float32) / np.float32(64.0),
        ).astype("<f4")

    def test_exact_chain_and_ulp_discontinuity_control(self) -> None:
        chain, receipt = minimum_q_chain(self.native, 4.0)
        jitter = ulp_jittered_chain(chain, 4.0)
        validate_same_support(chain, self.native, 4.0)
        validate_same_support(jitter, self.native, 4.0)
        self.assertEqual(float32_power_orbit_classes(chain, 4.0), 6)
        self.assertEqual(float32_boundary_count(chain, 4.0), 6)
        self.assertEqual(float32_power_orbit_classes(jitter, 4.0), 64)
        self.assertLess(
            16384 * np.max(np.abs(chain.astype(np.float64) - jitter.astype(np.float64))),
            0.01,
        )

    def test_metrics_keep_exact_and_approximate_objects_separate(self) -> None:
        chain, receipt = minimum_q_chain(self.native, 4.0)
        jitter = ulp_jittered_chain(chain, 4.0)
        chain_metrics = analyse_table(
            chain, factor=4.0, horizon=4096, levels=1, tau_grid=(0.0, 1e-5)
        )
        jitter_metrics = analyse_table(
            jitter, factor=4.0, horizon=4096, levels=1, tau_grid=(0.0, 1e-5)
        )
        self.assertLess(
            chain_metrics["realized_float32_power_orbit_classes"],
            jitter_metrics["realized_float32_power_orbit_classes"],
        )
        self.assertEqual(
            chain_metrics["approximate_one_step_matching"][1]["matched_channels"],
            jitter_metrics["approximate_one_step_matching"][1]["matched_channels"],
        )
        self.assertAlmostEqual(
            chain_metrics["continuous_gram"]["epsilon_lower_bound"],
            jitter_metrics["continuous_gram"]["epsilon_lower_bound"],
            places=4,
        )

    def test_same_support_geometric_pins_endpoints(self) -> None:
        slow = np.float32(self.native[-1] / np.float32(4.0))
        table = same_support_geometric(self.native, slow)
        validate_same_support(table, self.native, 4.0)
        self.assertEqual(table[0], self.native[0])
        self.assertEqual(table[-1], slow)

    def test_paired_bootstrap_uses_equal_task_weight(self) -> None:
        left = [
            {"task": "a", "row": 0, "score": 1.0},
            {"task": "b", "row": 0, "score": 0.0},
            {"task": "b", "row": 1, "score": 0.0},
            {"task": "b", "row": 2, "score": 0.0},
        ]
        right = [{**row, "score": 0.0} for row in left]
        result = paired_bootstrap(
            left,
            right,
            key_fields=("task", "row"),
            value_field="score",
            higher_is_better=True,
            seed=1,
            replicates=10,
        )
        self.assertEqual(result["strata"], 2)
        self.assertEqual(result["mean_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
