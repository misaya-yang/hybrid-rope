import unittest
import numpy as np

from experiments.native_enhancement_oral_20260915.phase_control import reflection


class ReflectionTests(unittest.TestCase):
    def test_componentwise_phase_magnitude_and_rotation_norm(self):
        native = np.array([1, .5, .125, .01], dtype=np.float32)
        candidate = np.array([1, .48, .11, .01], dtype=np.float32)
        result = reflection(native, candidate, native_length=4096)
        reverse = np.array(result["values_float32"])
        for distance in (0, 1, 100, 4095):
            a = 2 * np.abs(np.sin(distance * (candidate.astype(float) - native) / 2))
            b = 2 * np.abs(np.sin(distance * (reverse - native) / 2))
            self.assertTrue(np.allclose(a, b, atol=2e-4))
        self.assertTrue(result["audit"]["endpoints_bit_exact"])

    def test_invalid_reflection_rejected_without_clipping(self):
        with self.assertRaises(ValueError):
            reflection([1, .9, .8, .1], [1, .5, .4, .1], native_length=4096)
        with self.assertRaises(ValueError):
            reflection([1, .5, .2, .1], [1, .4, .15, .05], native_length=4096)


if __name__ == "__main__":
    unittest.main()
