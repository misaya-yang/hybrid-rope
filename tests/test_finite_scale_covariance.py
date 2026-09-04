import math
import unittest
from fractions import Fraction

import numpy as np

from scripts.analysis.finite_scale_covariance import (
    arbitrary_mixing_l2_report,
    anti_alias_corollary_lower_bound,
    anti_alias_margin,
    approximate_alias_report,
    best_coherent_matching,
    continuous_scaling_trilemma_report,
    dimension_span_requirements,
    exact_orbit_report,
    fourier_orbit_rank_report,
    montgomery_vaughan_separation_bound,
    operator_error_to_log_tau,
    orbit_growth_coherence_lower_bound,
    packed_survivors,
    phase_character_sup_error,
    sharp_periodic_phases,
    uniform_log_lattice_report,
    uniform_log_multi_generator_report,
    verify_packing_formula,
)


class FiniteScaleCovarianceTest(unittest.TestCase):
    def test_exact_scaling_trilemma_and_negative_one_exception(self) -> None:
        report = continuous_scaling_trilemma_report(6, 2.0)
        self.assertLessEqual(report["nilpotent_similarity_residual"], 1e-12)
        self.assertEqual(report["nilpotent_index"], 6)
        self.assertGreater(
            report["nilpotent_exponential_operator_norm"]["16.0"],
            report["nilpotent_exponential_operator_norm"]["1.0"],
        )
        self.assertLessEqual(report["negative_one_nontrivial_residual"], 1e-12)

    def test_theorem3_bounds_and_equality_characterization(self) -> None:
        chain = exact_orbit_report(range(6), 1, 3)
        holes = exact_orbit_report((0, 1, 3, 4), 1, 2)
        self.assertTrue(chain["first_bound_equality"])
        self.assertTrue(chain["all_residue_classes_are_consecutive"])
        self.assertFalse(holes["first_bound_equality"])
        self.assertFalse(holes["all_residue_classes_are_consecutive"])
        self.assertTrue(holes["global_bound_holds"])
        self.assertTrue(holes["two_sided_bound_holds"])

    def test_theorem3_multiple_residue_classes(self) -> None:
        report = exact_orbit_report(
            (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)),
            1,
            2,
        )
        self.assertEqual(report["residue_classes"], 2)
        self.assertEqual(report["boundary_count"], 2)
        self.assertEqual(report["orbit_modes"], 8)

    def test_theorem4_packing_formula_is_exhaustively_sharp(self) -> None:
        result = verify_packing_formula(max_total=10)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cases"], result["packed_construction_equality_cases"])
        self.assertEqual(packed_survivors(14, 5, 2), 8)

    def test_theorem4_coherent_matching_respects_bound(self) -> None:
        points = np.arange(7, dtype=np.float64)
        report = best_coherent_matching(points, alpha=1.0, tau=0.0, levels=3)
        self.assertEqual(report["best_survivors"], 4)
        self.assertEqual(report["theorem_survivor_bound"], 4)
        self.assertAlmostEqual(report["leakage"], 3 / 7)

    def test_theorem4_two_sided_matching_replaces_n_by_two_n(self) -> None:
        report = best_coherent_matching(
            np.arange(7, dtype=np.float64),
            alpha=1.0,
            tau=0.0,
            levels=2,
            two_sided=True,
        )
        self.assertEqual(report["best_survivors"], 3)
        self.assertEqual(report["theorem_survivor_bound"], 3)
        self.assertEqual(report["effective_chain_steps"], 4)

    def test_dimension_span_inverse(self) -> None:
        result = dimension_span_requirements(3, 0.2, math.log(4), 0.1)
        self.assertEqual(result["minimum_dimension"], 15)
        self.assertAlmostEqual(result["minimum_span"], (math.log(4) - 0.1) * 14)
        two_sided = dimension_span_requirements(
            3, 0.2, math.log(4), 0.1, directions=2
        )
        self.assertEqual(two_sided["minimum_dimension"], 30)

    def test_operator_error_to_tau_contains_realized_log_error(self) -> None:
        scale, omega, horizon, mismatch = 4.0, 0.2, 64.0, 0.01
        target = scale * omega * math.exp(mismatch)
        epsilon = phase_character_sup_error(target, scale * omega, horizon)
        bound = operator_error_to_log_tau(epsilon, horizon, scale, omega)
        self.assertGreaterEqual(bound["log_mismatch_bound"] + 1e-12, mismatch)

    def test_theorem5_real_extension_doubles_base_dimension(self) -> None:
        frequencies = np.array((0.1, 0.2, 0.35))
        complex_report = fourier_orbit_rank_report(
            frequencies, 2.0, 2, 64, real=False, discrete=False
        )
        real_report = fourier_orbit_rank_report(
            frequencies, 2.0, 2, 64, real=True, discrete=False
        )
        self.assertEqual(complex_report["base_dimension"], 3)
        self.assertEqual(real_report["base_dimension"], 6)
        self.assertGreaterEqual(real_report["ky_fan"]["epsilon_lower_bound"], 0.0)

    def test_theorem5_discrete_alias_collapses_orbit_modes(self) -> None:
        report = fourier_orbit_rank_report(
            (2 * math.pi / 3,), 4.0, 2, 8, real=False, discrete=True
        )
        self.assertLessEqual(report["fixed_native_subspace_max_squared_residual"], 1e-10)

    def test_theorem5_combines_orbit_growth_and_coherence(self) -> None:
        report = orbit_growth_coherence_lower_bound(12, 3, 4, 0.0)
        self.assertEqual(report["orbit_modes_lower_bound"], 24)
        self.assertAlmostEqual(report["epsilon_squared_lower_bound"], 0.5)

    def test_theorem5_dimension_free_separation_bound(self) -> None:
        modes = np.arange(1.0, 17.0)
        report = montgomery_vaughan_separation_bound(modes, math.pi, 4)
        self.assertAlmostEqual(report["minimum_separation"], 1.0)
        self.assertAlmostEqual(report["lambda_max_upper_bound"], 2.5)
        self.assertAlmostEqual(report["epsilon_squared_lower_bound"], 0.375)
        gram_report = fourier_orbit_rank_report(
            np.arange(1.0, 5.0), 5.0, 1, math.pi, real=False, discrete=False
        )
        self.assertIsNotNone(gram_report["montgomery_vaughan_separation_bound"])

    def test_theorem5_arbitrary_invertible_mixing_l2_chain(self) -> None:
        report = arbitrary_mixing_l2_report(
            np.exp(-np.linspace(0.1, 2.7, 6)),
            2.0,
            2,
            64,
            discrete=False,
            seed=7,
        )
        self.assertGreaterEqual(report["minimum_selected_vs_projection_slack"], -1e-9)
        self.assertGreaterEqual(
            report["fixed_native_optimal_mean_squared_residual"] + 1e-9,
            report["ky_fan_epsilon_squared_lower_bound"],
        )

    def test_theorem6_exact_sharp_period(self) -> None:
        dimension, scale = 5, 2
        phases = sharp_periodic_phases(dimension, scale)
        report = approximate_alias_report(phases, scale)
        self.assertLessEqual(report["scaling_defect"], 1e-12)
        self.assertEqual(report["common_period"], scale**dimension - 1)
        self.assertLessEqual(report["actual_alias_error"], 1e-12)

    def test_theorem6_approximate_alias_and_anti_alias_corollary(self) -> None:
        phases = sharp_periodic_phases(4, 2) + np.array((1e-4, -2e-4, 1e-4, 0.0))
        report = approximate_alias_report(phases, 2)
        self.assertLessEqual(
            report["actual_alias_error"], report["alias_error_upper_bound"] + 1e-10
        )
        horizon = 2**4 - 1
        gamma = anti_alias_margin(phases, horizon)
        self.assertGreaterEqual(
            report["scaling_defect"] + 1e-10,
            anti_alias_corollary_lower_bound(gamma, 4, 2),
        )

    def test_uniform_log_commensurate_and_incommensurate(self) -> None:
        exact = uniform_log_lattice_report(16, 0.25, 0.5, 3)
        approximate = uniform_log_lattice_report(16, 0.25, 0.55, 3)
        self.assertTrue(exact["commensurate"])
        self.assertEqual(exact["exact_survivors"], 10)
        self.assertEqual(exact["exact_orbit_modes"], 22)
        self.assertFalse(approximate["commensurate"])
        self.assertAlmostEqual(approximate["level_log_mismatch"][-1], 0.15)

    def test_uniform_log_disjoint_shift_saturates_orbit_growth(self) -> None:
        report = uniform_log_lattice_report(4, 1.0, 6.0, 2)
        self.assertTrue(report["commensurate"])
        self.assertEqual(report["exact_survivors"], 0)
        self.assertEqual(report["exact_orbit_modes"], 12)
        self.assertEqual(report["overlap_regime"], "disjoint_shifted_intervals")

    def test_uniform_log_multi_generator_commensurability(self) -> None:
        exact = uniform_log_multi_generator_report(16, 0.25, (0.5, 0.75), 3)
        impossible = uniform_log_multi_generator_report(
            16, 0.25, (0.5, math.sqrt(2.0)), 3
        )
        self.assertTrue(exact["all_generators_are_exact_lattice_shifts"])
        self.assertEqual(exact["joint_orbit_modes"], 25)
        self.assertFalse(impossible["all_generators_are_exact_lattice_shifts"])
        self.assertIsNone(impossible["joint_orbit_modes"])


if __name__ == "__main__":
    unittest.main()
