from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.verify_prefix_phase_handoff import (
    boundary_slope_cross_error,
    causal_attention_cached,
    causal_attention_reference,
    deterministic_check,
    gqa_attention_chunked_cache,
    gqa_attention_reference,
    prefix_handoff_score,
    rephase_cached_key,
    rotary_score,
    rotate_split_half,
    split_chunks_at_boundary,
    toy_causal_stack,
)


class PrefixPhaseHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.native = np.asarray([1.0, 0.3], dtype=np.float64)
        self.long = np.asarray([1.0, 0.075], dtype=np.float64)
        self.query = np.asarray([0.7, -0.2, 0.4, 1.1], dtype=np.float64)
        self.key = np.asarray([-0.3, 0.9, 0.2, -0.6], dtype=np.float64)

    def test_rotation_preserves_vector_norm(self) -> None:
        observed = rotate_split_half(
            self.query, position=19, frequencies=self.long
        )
        self.assertAlmostEqual(
            float(observed @ observed),
            float(self.query @ self.query),
            places=12,
        )

    def test_in_place_rephase_equals_direct_long_rotation(self) -> None:
        raw_key = np.asarray([-0.4, 0.8, 0.3, -1.2], dtype=np.float64)
        native_rotated = rotate_split_half(
            raw_key,
            position=37,
            frequencies=self.native,
            scaling=0.85,
        )
        observed = rephase_cached_key(
            native_rotated,
            position=37,
            native_frequencies=self.native,
            long_frequencies=self.long,
            native_scaling=0.85,
            long_scaling=1.15,
        )
        expected = rotate_split_half(
            raw_key,
            position=37,
            frequencies=self.long,
            scaling=1.15,
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)

    def test_prefix_query_is_exact_native_branch(self) -> None:
        expected = rotary_score(
            self.query,
            self.key,
            query_position=7,
            key_position=2,
            frequencies=self.native,
        )
        observed = prefix_handoff_score(
            self.query,
            self.key,
            query_position=7,
            key_position=2,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=8,
        )
        self.assertEqual(observed, expected)

    def test_suffix_query_rephases_prefix_key_in_long_frame(self) -> None:
        expected = rotary_score(
            self.query,
            self.key,
            query_position=13,
            key_position=2,
            frequencies=self.long,
        )
        observed = prefix_handoff_score(
            self.query,
            self.key,
            query_position=13,
            key_position=2,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=8,
        )
        self.assertEqual(observed, expected)

    def test_boundary_slope_error_has_closed_form(self) -> None:
        observed = boundary_slope_cross_error(
            0.3,
            0.075,
            query_position=13,
            key_position=2,
            boundary=8,
        )
        expected = (0.3 - 0.075) * (8 - 2)
        self.assertAlmostEqual(observed, expected, places=12)
        self.assertGreater(abs(observed), 0.0)

    def test_deterministic_receipt_passes(self) -> None:
        receipt = deterministic_check()
        self.assertEqual(receipt["status"], "CPU_ALGEBRA_PASS")
        self.assertEqual(receipt["prefix_native_score_error"], 0.0)
        self.assertEqual(receipt["suffix_stationary_long_score_error"], 0.0)
        self.assertLessEqual(receipt["cached_vs_direct_handoff_max_abs_error"], 1e-12)
        self.assertEqual(receipt["prefix_attention_vs_native_max_abs_error"], 0.0)

    def test_cached_attention_matches_direct_handoff_and_native_prefix(self) -> None:
        rng = np.random.default_rng(7)
        queries = rng.standard_normal((10, 4))
        keys = rng.standard_normal((10, 4))
        values = rng.standard_normal((10, 3))
        direct = causal_attention_reference(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=6,
            native_scaling=0.9,
            long_scaling=1.1,
        )
        cached = causal_attention_cached(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=6,
            native_scaling=0.9,
            long_scaling=1.1,
        )
        native = causal_attention_reference(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.native,
            boundary=6,
            native_scaling=0.9,
            long_scaling=0.9,
        )
        np.testing.assert_allclose(cached, direct, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(cached[:6], native[:6])

    def test_multilayer_cache_matches_direct_and_preserves_prefix(self) -> None:
        rng = np.random.default_rng(11)
        hidden = rng.standard_normal((10, 4))
        layers = [
            tuple(rng.standard_normal((4, 4)) / 2 for _ in range(4))
            for _ in range(2)
        ]
        direct = toy_causal_stack(
            hidden,
            layers,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=6,
            cached=False,
        )
        cached = toy_causal_stack(
            hidden,
            layers,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=6,
            cached=True,
        )
        native = toy_causal_stack(
            hidden,
            layers,
            native_frequencies=self.native,
            long_frequencies=self.native,
            boundary=6,
            cached=False,
        )
        np.testing.assert_allclose(cached, direct, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(cached[:6], native[:6])

    def test_chunked_gqa_matches_direct_and_preserves_prefix(self) -> None:
        rng = np.random.default_rng(17)
        queries = rng.standard_normal((12, 4, 4))
        keys = rng.standard_normal((12, 2, 4))
        values = rng.standard_normal((12, 2, 3))
        direct = gqa_attention_reference(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=7,
            native_scaling=0.9,
            long_scaling=1.1,
        )
        chunked = gqa_attention_chunked_cache(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=7,
            chunk_sizes=(3, 6, 3),
            native_scaling=0.9,
            long_scaling=1.1,
        )
        native = gqa_attention_reference(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.native,
            boundary=7,
            native_scaling=0.9,
            long_scaling=0.9,
        )
        np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(chunked[:7], native[:7])

    def test_chunked_mqa_matches_direct(self) -> None:
        rng = np.random.default_rng(23)
        queries = rng.standard_normal((9, 4, 4))
        keys = rng.standard_normal((9, 1, 4))
        values = rng.standard_normal((9, 1, 2))
        direct = gqa_attention_reference(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=5,
        )
        chunked = gqa_attention_chunked_cache(
            queries,
            keys,
            values,
            native_frequencies=self.native,
            long_frequencies=self.long,
            boundary=5,
            chunk_sizes=(5, 1, 3),
        )
        np.testing.assert_allclose(chunked, direct, rtol=0.0, atol=1e-12)

    def test_gqa_rejects_nondivisible_head_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            gqa_attention_reference(
                np.zeros((4, 3, 4)),
                np.zeros((4, 2, 4)),
                np.zeros((4, 2, 2)),
                native_frequencies=self.native,
                long_frequencies=self.long,
                boundary=2,
            )

    def test_chunk_schedule_is_split_exactly_at_handoff(self) -> None:
        self.assertEqual(
            split_chunks_at_boundary((4, 9, 5), boundary=11),
            (4, 7, 2, 5),
        )
        self.assertEqual(
            split_chunks_at_boundary((11, 1, 6), boundary=11),
            (11, 1, 6),
        )


if __name__ == "__main__":
    unittest.main()
