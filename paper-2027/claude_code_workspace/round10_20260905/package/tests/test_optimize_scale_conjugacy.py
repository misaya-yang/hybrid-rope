import math
import sys
import unittest

import numpy as np

from scripts.analysis.optimize_scale_conjugacy import (
    exact_block_permutation_error,
    exact_identity_error,
    sampled_positions,
)


class OptimizeScaleConjugacyTest(unittest.TestCase):
    def test_cpu_helpers_do_not_import_torch(self) -> None:
        positions = sampled_positions(16, 17, 7)
        self.assertEqual(positions[0], -16)
        self.assertEqual(positions[-1], 16)
        self.assertIn(0, positions)
        self.assertNotIn("torch", sys.modules)

    def test_exact_baselines(self) -> None:
        error = exact_identity_error((0.1,), 2.0, 1, 1.0)
        self.assertAlmostEqual(error, 2 * math.sin(0.05))
        frequencies = np.asarray((1.0, 2.0))
        permutation = exact_block_permutation_error(frequencies, 2.0, 1, 0.25)
        self.assertLessEqual(
            permutation["error"], exact_identity_error(frequencies, 2.0, 1, 0.25)
        )


if __name__ == "__main__":
    unittest.main()
