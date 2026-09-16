import unittest
import numpy as np

from experiments.native_contrastive_proximal_20260915.tables import reference_fourier
from experiments.native_enhancement_oral_20260915.predictions import fixed_distance_risk, freeze
from experiments.native_enhancement_oral_20260915.prepare import prepare_group
from tests.test_native_enhancement_prepare import CharacterTokenizer


class PredictionTests(unittest.TestCase):
    def test_direct_angular_loss_at_fixed_distance(self):
        a0, coefficients = reference_fourier()
        phases = np.array([0, .1, 3.14, 13, 1024.5])
        negative = np.cos(2 * np.pi * np.arange(1024) / 1024)
        direct = np.logaddexp(0, negative[None, :] - np.cos(phases[:, None])).mean(axis=1)
        self.assertTrue(np.allclose(fixed_distance_risk(phases, a0, coefficients), direct, atol=1e-12))

    def test_identity_produces_no_risk_change(self):
        rows = prepare_group(CharacterTokenizer(), "native_binding", 1024, 0)
        table = [1, .1, .01, .001]
        result = freeze(rows, table, table)
        self.assertEqual(result["prediction_rows"], 4)
        self.assertTrue(all(r["reference_risk_reduction"] == 0 for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
