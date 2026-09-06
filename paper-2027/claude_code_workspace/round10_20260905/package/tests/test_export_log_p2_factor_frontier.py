import numpy as np

from scripts.analysis.export_log_p2_factor_frontier import derive_movement, realize


def test_reference_movement_and_scale_composition() -> None:
    native = np.geomspace(1.0, 1e-4, 8).astype(np.float32)
    movement = np.linspace(0.0, 1.0, 8)
    reference = realize(native, movement, 4.0)
    recovered = derive_movement(native, reference, 4.0)
    np.testing.assert_allclose(recovered, movement, atol=2e-7)
    direct = realize(native, recovered, 8.0)
    composed = np.ascontiguousarray(reference * np.power(2.0, -recovered), dtype="<f4")
    np.testing.assert_array_equal(direct, composed)
