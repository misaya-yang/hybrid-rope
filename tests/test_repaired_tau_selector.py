import math
import unittest

import numpy as np

from rebuttal.rebuttal_0723.experiments.repaired_tau_selector import (
    collision_risk,
    distance_features,
    evq_phi,
    pair_distance_distribution,
)


class RepairedTauSelectorTests(unittest.TestCase):
    def test_finite_grid_features_and_risk(self):
        phi = evq_phi(8, 2.0)
        self.assertTrue(np.all(np.diff(phi) > 0.0))
        self.assertTrue(np.allclose(evq_phi(8, 0.0), (np.arange(8) + 0.5) / 8))
        self.assertTrue(
            np.allclose(
                evq_phi(8, 0.0, midpoint=False),
                np.arange(8) / 8,
            )
        )
        features = distance_features(500_000.0, 8, 2.0, 63)
        self.assertTrue(np.allclose(np.sum(features * features, axis=1), 1.0))
        distribution = pair_distance_distribution(32, 63)
        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        result = collision_risk(
            base=500_000.0,
            num_pairs=8,
            train_length=32,
            target_lengths=[64],
            target_weights=[1.0],
            tau=2.0,
        )
        self.assertTrue(math.isfinite(result["risk"]))
        self.assertEqual(result["risk"], max(result[k] for k in ("train", "target", "cross")))


if __name__ == "__main__":
    unittest.main()
