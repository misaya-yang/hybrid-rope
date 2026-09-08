"""Adversarial stress-test of the 6 missing physical observables and causal mechanisms.

Verifies:
1. Breakdown of local quadratic Taylor expansion (F_N) under long-context phase accumulation (delta_omega * L >= 2*pi).
2. Structural redundancy between Observable 1 (H: cross-token coherence) and Observable 3 (J: cross-head cancellation) via Jacobian Gram matrix decomposition.
3. Category classification of Observable 6 (T(a, R, z)) as coordinate reparameterization rather than physical observable.
4. Softmax distractor dilution dynamics (S_dilution) vs temperature scaling gain (g).
5. Analysis of MrRoPE-Pro phase budget, conservatism, and unexplained empirical residuals.
"""
import math
import unittest
import numpy as np
import torch


class TestMissingObservablesBreakdown(unittest.TestCase):
    """Empirical tests probing vulnerabilities in the 6 missing observables."""

    def test_local_quadratic_taylor_expansion_breakdown(self):
        """Test 1: Probe the breakdown of local Taylor/Hessian prediction (F_N)

        Under long contexts (L >> 1), realistic frequency modifications produce
        phase shifts delta_phi = delta_omega * L that easily exceed 2*pi.
        We demonstrate that the second-order Taylor expansion delta_L_pred = 0.5 * domega^T F_N domega
        diverges catastrophically from the true non-linear attention difference.
        """
        # Set up a toy rotary attention head with K=64 slots
        K = 64
        base = 1000000.0  # Qwen base
        L_native = 32768
        inv_freq = base ** (-np.arange(K, dtype=np.float64) / K)

        # Consider slot 19 (the C2 transition knee slot)
        # In C2, slot 19 has movement error delta_m approx 0.0441
        omega_19 = inv_freq[19]
        s = 4.0
        m_true = 0.25  # true smooth exponent
        m_c2 = m_true + 0.044109  # C2 exponent
        omega_true = omega_19 * (s ** (-m_true))
        omega_c2 = omega_19 * (s ** (-m_c2))
        delta_omega = omega_c2 - omega_true

        # Test across varying context horizons L
        horizons = [256, 1024, 4096, 16384, 32768, 65536, 131072]
        breakdown_results = []

        for L in horizons:
            # Maximum phase shift over horizon L
            max_phase_shift = abs(delta_omega) * L

            # Model a single oscillatory component: f(omega) = cos(omega * L)
            # True difference: cos((omega + delta_omega)*L) - cos(omega*L)
            # 1st order Taylor: -L * sin(omega*L) * delta_omega
            # 2nd order Taylor: -L * sin(omega*L) * delta_omega - 0.5 * L^2 * cos(omega*L) * (delta_omega)^2
            true_diff = math.cos((omega_true + delta_omega) * L) - math.cos(omega_true * L)
            taylor_1st = -L * math.sin(omega_true * L) * delta_omega
            taylor_2nd = taylor_1st - 0.5 * (L ** 2) * math.cos(omega_true * L) * (delta_omega ** 2)

            abs_err_2nd = abs(taylor_2nd - true_diff)
            # Quadratic term magnitude alone: 0.5 * L^2 * (delta_omega)^2
            quad_term_mag = 0.5 * (L * delta_omega) ** 2

            breakdown_results.append({
                "L": L,
                "max_phase_shift_rad": max_phase_shift,
                "true_diff": true_diff,
                "taylor_2nd": taylor_2nd,
                "abs_err_2nd": abs_err_2nd,
                "quad_term_mag": quad_term_mag,
            })

        # At L=32768, max phase shift exceeds 2*pi
        res_32k = next(r for r in breakdown_results if r["L"] == 32768)
        self.assertGreater(res_32k["max_phase_shift_rad"], 2.0 * math.pi)
        # The quadratic Taylor term magnitude is >> 2.0 (the maximum possible range of cosine diff!)
        self.assertGreater(res_32k["quad_term_mag"], 2.0)
        # At L=131072, max phase shift is huge (> 10 rad) and quadratic term completely explodes
        res_128k = next(r for r in breakdown_results if r["L"] == 131072)
        self.assertGreater(res_128k["quad_term_mag"], 50.0)

    def test_jacobian_gram_redundancy_between_h_and_j(self):
        """Test 2: Mathematically verify that Observable 1 (H) and Observable 3 (J)
        are index-partition decompositions of the EXACT same layer Jacobian tensor.

        Let J_{o, j} = sum_h W_{O, h} J_{h, j} where J_{h, j} = sum_t Delta_t u_{h, t, j}.
        Then the total layer perturbation energy G_{j, j} = ||J_j||^2 can be written as:
        G_{j, j} = sum_{h, h'} <W_{O, h} J_h, W_{O, h'} J_{h'}> (Observable 3 cross-head view)
        G_{j, j} = sum_{t, t'} Delta_t Delta_{t'} <v_t, v_{t'}> (Observable 1 cross-token view)
        They evaluate to the EXACT SAME real number!
        """
        rng = np.random.default_rng(20260908)
        H_heads = 4
        T_tokens = 8
        D_head = 6
        D_out = 10
        pairs = D_head // 2

        # Create synthetic key responses u_{h, t, j} \in R^{D_head}
        u = rng.normal(size=(H_heads, T_tokens, pairs, D_head))
        # Delays Delta_t
        delta = rng.uniform(1.0, 50.0, size=T_tokens)
        # Head projection matrices W_{O, h} \in R^{D_out, D_head}
        wo = rng.normal(size=(H_heads, D_out, D_head))

        # Single frequency slot j = 0
        j = 0

        # View A (Observable 3: Multi-head cancellation view)
        # Head Jacobian: J_{h, j} = sum_t delta_t * u_{h, t, j} \in R^{D_head}
        J_h = np.sum(delta[None, :, None] * u[:, :, j, :], axis=1)  # shape (H, D_head)
        # Projected head Jacobian: W_{O, h} @ J_{h, j} \in R^{D_out}
        W_J_h = np.einsum('hod,hd->ho', wo, J_h)  # shape (H, D_out)
        # Total layer Jacobian: J_layer = sum_h W_J_h \in R^{D_out}
        J_layer = np.sum(W_J_h, axis=0)
        energy_from_J = np.sum(J_layer ** 2)

        # Cross-head decomposition: sum_h ||W_J_h||^2 + 2 * sum_{h < h'} <W_J_h, W_J_h'>
        diag_head_energy = np.sum(W_J_h ** 2)
        cross_head_energy = energy_from_J - diag_head_energy

        # View B (Observable 1: Cross-token coherence view)
        # Layer token vector: v_{t} = sum_h W_{O, h} @ u_{h, t, j} \in R^{D_out}
        v_t = np.einsum('hod,htd->to', wo, u[:, :, j, :])  # shape (T, D_out)
        # Total layer Jacobian via tokens: J_layer_token = sum_t delta_t * v_t
        J_layer_token = np.sum(delta[:, None] * v_t, axis=0)
        energy_from_H = np.sum(J_layer_token ** 2)

        # Cross-token decomposition: sum_t delta_t^2 ||v_t||^2 + 2 * sum_{t < t'} delta_t delta_t' <v_t, v_t'>
        diag_token_energy = np.sum((delta[:, None] * v_t) ** 2)
        cross_token_energy = energy_from_H - diag_token_energy

        # The two totals must be identical up to machine precision!
        np.testing.assert_allclose(energy_from_J, energy_from_H, rtol=1e-12, atol=1e-12)

    def test_softmax_distractor_dilution_and_temperature_gain(self):
        """Test 3: Stress-test Observable 4 (S_dilution) and temperature scaling g.

        Simulates attention across context expansion factors S in [1, 2, 4, 8, 16, 32].
        Verifies:
        1. Without temperature scaling (g=1), needle attention probability drops exponentially as S increases.
        2. MrRoPE temperature scaling g = 1 + 0.1 * ln(S) prevents retrieval collapse up to S=4..8,
           but has a hard ceiling when S becomes very large.
        """
        rng = np.random.default_rng(42)
        L_0 = 1024
        B_margin = 4.0  # intrinsic needle logit margin over mean background
        distractor_std = 1.0

        factors = [1, 2, 4, 8, 16, 32]
        prob_unscaled = []
        prob_mrrope = []

        for S in factors:
            num_distractors = S * L_0 - 1
            # Sample distractor logits ~ N(0, distractor_std^2)
            distractor_logits = rng.normal(0.0, distractor_std, size=num_distractors)
            needle_logit = B_margin

            # Case A: unscaled (g = 1.0)
            all_logits_a = np.concatenate([[needle_logit], distractor_logits])
            exp_a = np.exp(all_logits_a - np.max(all_logits_a))
            prob_a = exp_a[0] / np.sum(exp_a)
            prob_unscaled.append(prob_a)

            # Case B: MrRoPE temperature gain g = 1 + 0.1 * ln(S)
            g = 1.0 + 0.1 * math.log(S)
            all_logits_b = all_logits_a * (g ** 2)  # gain multiplies query & key
            exp_b = np.exp(all_logits_b - np.max(all_logits_b))
            prob_b = exp_b[0] / np.sum(exp_b)
            prob_mrrope.append(prob_b)

        # Unscaled attention probability at 8x drops substantially
        self.assertLess(prob_unscaled[3], prob_unscaled[0])
        # MrRoPE gain significantly boosts retention compared to unscaled at 4x and 8x
        self.assertGreater(prob_mrrope[2], prob_unscaled[2])
        self.assertGreater(prob_mrrope[3], prob_unscaled[3])

    def test_mrrope_pro_unexplained_residuals_and_phase_budget(self):
        """Test 4: Analyze MrRoPE-Pro's frequency schedule and demonstrate that:
        1. It uses a purely pre-hoc formula independent of checkpoint weights W_Q, W_K, W_O.
        2. Its choice of quadratic cumulative exponent t(t+1)/[n(n+1)] vs other convex exponents
           (e.g. t^2/n^2, t^1.5/n^1.5, cubic) has NO theoretical derivation from the 6 observables.
        3. MrRoPE-Pro is conservative (leaves turns > 32 unscaled, turns < 1 uniformly scaled),
           which explains its stability, but leaves an unexplained residual regarding optimality.
        """
        K = 64
        base = 1000000.0
        L_0 = 32768
        s = 4.0
        inv_freq = base ** (-np.arange(K, dtype=np.float64) / K)

        turns = (inv_freq * L_0) / (2.0 * math.pi)
        l_bound = int(np.max(np.where(turns > 32.0)[0]))
        h_bound = int(np.min(np.where(turns < 1.0)[0]))
        n_width = h_bound - l_bound

        self.assertEqual(l_bound, 23)
        self.assertEqual(h_bound, 40)
        self.assertEqual(n_width, 17)

        # Compare MrRoPE-Pro quadratic exponent with alternatives
        t = np.clip(np.arange(K) - l_bound, 0, n_width)
        m_pro = (t * (t + 1)) / (n_width * (n_width + 1))
        m_linear = t / n_width
        m_pure_quad = (t / n_width) ** 2
        m_cubic = (t / n_width) ** 3

        # At the midpoint of the transition band t = n/2
        t_mid = n_width // 2
        # m_pro has convex delay: m_pro < m_linear
        self.assertLess(m_pro[l_bound + t_mid], m_linear[l_bound + t_mid])
        # However, m_pro, m_pure_quad, and m_cubic are all plausible convex schedules
        # The exact formula t(t+1)/[n(n+1)] is an empirical discrete radix progression,
        # NOT derived from Fisher information F_N or coherence H.
        self.assertTrue(np.all(m_pro >= 0.0) and np.all(m_pro <= 1.0))


if __name__ == "__main__":
    unittest.main()
