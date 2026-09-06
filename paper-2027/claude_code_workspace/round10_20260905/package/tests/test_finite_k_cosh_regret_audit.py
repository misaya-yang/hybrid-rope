import unittest

from scripts.analysis.finite_k_cosh_regret_audit import (
    asymptotic_constant,
    audit_tau,
    build_report,
    cosh_quantile,
    histogram_components,
)


class FiniteKCoshRegretAuditTests(unittest.TestCase):
    def test_tau_zero_recovers_uniform_histogram(self) -> None:
        self.assertEqual(cosh_quantile(0.25, 0.0), 0.25)
        density_l2, volterra_l2 = histogram_components(16, 0.0)
        self.assertAlmostEqual(density_l2, 1.0, places=13)
        self.assertAlmostEqual(volterra_l2, 1.0 / 3.0, places=13)
        self.assertEqual(asymptotic_constant(1.0, 0.0), 0.0)

    def test_scaled_regret_converges_to_the_analytic_constant(self) -> None:
        audit = audit_tau(tau=4.0, pairs=(16, 32, 64, 128), alpha=1.0)
        errors = [
            abs(float(row["relative_constant_error"]))
            for row in audit["rows"]
        ]
        self.assertTrue(all(right < left for left, right in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 3e-4)
        self.assertAlmostEqual(float(audit["tail_log_log_slope"]), -2.0, delta=0.01)

    def test_default_grid_passes_scoped_gate(self) -> None:
        report = build_report(
            taus=(0.5, 1.0, 2.0, 4.0),
            pairs=(8, 16, 32, 64, 128),
            alpha=1.0,
            max_relative_error=0.01,
            max_slope_error=0.05,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertIn("not r2", report["scope"])


if __name__ == "__main__":
    unittest.main()
