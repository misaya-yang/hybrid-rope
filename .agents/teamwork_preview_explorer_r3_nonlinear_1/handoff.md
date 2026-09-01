# Handoff Report — explorer_r3_nonlinear_1

- **Focus:** R3 — Mathematical & Physical Impact of Non-Linear $z \to f(z)$ vs Linear Scaling $cz$
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r3_nonlinear_1`
- **Date:** 2026-09-01 (UTC)
- **Handoff Type:** Hard (Task Complete)

---

## 1. Observation

1. **Causal Decomposition & Exponent Parameterization:**
   - In `paper-2027/research/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md` (§2, lines 140–144):
     $$x_k = -\ln\omega_k = a + R z_k, \qquad z_0=0,\ z_{K-1}=1.$$
   - In `paper-2027/research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md` (§2, lines 61–66):
     "In the geometric family, $z_k=k/(K-1)$, so fixing $(a,R)$ fixes the whole table. The exact-range intervention changes only $z$."
   - In `paper-2027/research/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md` (§1.2, lines 78–82):
     $$\theta^{-e_k} = (\theta^c)^{-e_k/c}.$$

2. **Continuous Variational Formulation & Closed-Form Optimum:**
   - In `paper-2027/appendix/a1_proofs.tex` (§A.1, lines 306–341):
     $$K_{\mathrm{app}}(\phi,\psi) = \alpha\delta(\phi-\psi) + \beta\min(\phi,\psi).$$
     Euler-Lagrange ODE: $\rho''(\phi) - \tau^2\rho(\phi) = 0$ with $\tau = \sqrt{\beta/\alpha}$, boundary conditions $\rho'(0) = -\tau^2, \rho'(1) = 0$, leading to:
     $$\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}.$$
     Quantile function:
     $$\phi_k(\tau) = 1 - \frac{1}{\tau}\operatorname{arcsinh}\left((1-u_k)\sinh\tau\right), \qquad u_k = \frac{k+1/2}{K}.$$

3. **Spectral Collapse & Low-Frequency Geometry:**
   - In `paper-2027/sections/03_theory.tex` (Proposition 1, lines 82–93) & `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` (§3, lines 166–180):
     As $\omega L \to 0$, $V_\omega \to \operatorname{span}\{1, \Delta\}$; under softmax centering, $V_\omega \to \operatorname{span}\{\Delta - \mathbb{E}_p\Delta, \Delta^2 - \mathbb{E}_p\Delta^2\}$.
     For $\omega L \le 1$, 23 pairs in a 64-pair table occupy 46 nominal dimensions but have block-whitened stable rank $r_2 = 2.00$.

4. **Falsified Arcsine Route (O5):**
   - In `paper-2027/research/three_completions/optimization_notes.md` (O5, lines 147–165):
     Free density numerical optimization under $\chi^2$-stiffness constraint yields a monotone decreasing density with fast-end spike and flat tail:
     `[3.01, 1.57, 1.14, 1.17, 1.07, 0.92, 0.82, 0.80, 0.80, 0.77, 0.71, 0.67, 0.65, 0.62, 0.65, 0.63]`.
     Arcsine U-shape is strictly falsified because the leading collision kernel $\min(\phi, \psi)^2$ is not the Green's function for $\partial_\phi^4$ ($\partial_\phi^4(K\rho) = -6\rho' - 2\phi\rho'' \neq C\rho$).

5. **Falsified Naive Collision/Logdet Minimization (Fourier Comb Collapse):**
   - In `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` (§4.2–4.3) and `INDEX.md` (§3.4, lines 153–154):
     Minimizing collision or maximizing static effective rank over window $[0, L]$ collapses frequencies into an integer Fourier comb $\omega_k = 2\pi a_k / L$, resulting in exact aliasing $\mathcal{R}(\Delta + L) = \mathcal{R}(\Delta)$ that destroys out-of-distribution extrapolation.

---

## 2. Logic Chain

1. **Step 1 (Linear Scaling $\iff$ Base Change Isomorphism):**
   - From Observation 1: $\omega(z) = b^z$. Linear transformation $z \mapsto cz$ yields $\omega(cz) = b^{cz} = (b^c)^z = \omega_{b^c}(z)$.
   - The log-frequency interval ratio $\frac{c z_2 - c z_1}{c z_4 - c z_3} = \frac{z_2 - z_1}{z_4 - z_3}$ is invariant.
   - Therefore, linear scaling $cz$ is algebraically and geometrically identical to a scalar base change $b \to b^c$. It modifies only the support span $R = c R_{\text{orig}}$ and leaves the normalized interior allocation $z_k = (x_k - a)/R$ completely unchanged.

2. **Step 2 (Non-Linear Warping as an Orthogonal Axis):**
   - Under fixed support $f(0)=0, f(1)=1$, the support parameters $(a, R)$ remain strictly pinned.
   - Non-linear warping $f(z) \neq cz$ modifies the continuous derivative $f'(\phi) = d\phi/du$, which transforms the physical spectral density $\rho_f(\omega) = \frac{1}{\omega \ln b \cdot |f'(f^{-1}(-\log_b \omega))|}$.
   - For concave log-frequency maps ($f''(\phi) < 0$, such as EVQ-Cosh), density is reduced at the slow boundary ($\rho_\tau(1) = \tau/\sinh\tau < 1$) and amplified at the fast boundary ($\rho_\tau(0) = \tau\coth\tau > 1$).

3. **Step 3 (Mitigating Low-Frequency Collapse):**
   - From Observation 3: As $\omega L \to 0$, rotary channels collapse into redundant 2D subspaces.
   - EVQ-Cosh's single-crossing theorem ($\phi_c \le 1 - 1/\sqrt{3}$) proves that mass is systematically shifted out of the collapse regime into the active resolution bands.

4. **Step 4 (Wave-Packet Dispersion Dynamics):**
   - RoPE attention logits form wave-packet superpositions $\Psi(\Delta) = \int A(k) e^{i(\omega(k)\Delta + \psi(k))} dk$.
   - Geometric RoPE exhibits exponential group velocity dispersion $\text{GVD} = (\frac{\ln b}{K})^2 \omega(k)$, producing extreme phase velocity disparity ($b:1$).
   - Non-linear warping modulates $v_g(k) = -\frac{\ln b}{K} f'(k/K)\omega(k)$, shaping the local coherence length $L_{\text{coh}}(k) \sim \frac{2\pi}{|v_g(k)|\Delta k}$ to maintain constructive phase distinguishability across $[L_{\text{train}}, L_{\text{target}}]$.

5. **Step 5 (Variational Uniqueness and Strict Boundaries):**
   - From Observation 2: The quadratic surrogate $\mathcal{C}_{\text{app}}[\rho] = \frac{\alpha}{2}\int\rho^2 + \frac{\beta}{2}\iint \rho\rho\min$ is strictly convex due to the PSD property of $\min(\phi, \psi)$.
   - Its unique Euler-Lagrange solution is $\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$.
   - However, this uniqueness holds *strictly* for the surrogate $\mathcal{C}_{\text{app}}$, not as a global optimum of full-RoPE geometry or trained language model loss.

6. **Step 6 (Falsified Route Mechanics):**
   - Arcsine fails because the leading non-linear kernel $\min^2$ does not map to a classical local Riesz energy.
   - Naive static rank optimization fails because exact pairwise orthogonality creates a periodic Fourier comb with exact aliasing $\mathcal{R}(\Delta + L) = \mathcal{R}(\Delta)$.

---

## 3. Caveats

1. **No Direct LM Loss Prediction from Static Geometry:** Static effective rank $r_2(\Gamma)$ and collision $\bar{c}$ measure basis geometry, not trained transformer loss. Content-dependent Q/K weights co-adapt to the frequency basis during training.
2. **Surrogate Scope:** EVQ-Cosh is a constructive closed-form instance derived from the convex surrogate $\mathcal{C}_{\text{app}}$, not a universal upper bound on all possible frequency tables.
3. **Finite $\tau$ Convention:** The parameter $\tau = \max(d_{\text{head}}/\sqrt{L}, 1.4)$ is an empirical operating prior, not a universal physical constant.

---

## 4. Conclusion

1. Linear scaling $f(z) = cz$ and scalar base change $b \to b^c$ are algebraically identical transformations that operate strictly on the support span $R$, preserving relative logarithmic density and leaving interior allocation $z$ unchanged.
2. Non-linear warping $f(z) \neq cz$ under fixed support constitutes a separate, causally identifiable design axis that reshapes the continuous spectral channel density $\rho_f(\omega)$ and wave-packet dispersion relations $v_g(k)$.
3. EVQ-Cosh provides a mathematically rigorous, closed-form reallocation that thins the redundant low-frequency collapse regime ($\omega L \ll 1$) while preserving multi-scale non-harmonic coverage, avoiding the Fourier comb aliasing trap.
4. The full research report has been compiled and saved to `.agents/teamwork_preview_explorer_r3_nonlinear_1/report.md`.

---

## 5. Verification Method

To independently verify the derivations and theorems presented in this report:

1. **Inspect Report Artifacts:**
   - Review `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r3_nonlinear_1/report.md`.
2. **Verify Mathematical Proofs in Appendix:**
   - Inspect `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/paper-2027/appendix/a1_proofs.tex` (lines 300–460) for the ODE derivation, Green kernel PSD proof, single-crossing lemma, and surrogate self-consistency theorem.
3. **Verify Falsification Receipts:**
   - Inspect `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/paper-2027/research/three_completions/optimization_notes.md` (O5) for the 16-bin numerical optimization data disproving the Arcsine conjecture.
   - Inspect `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` (§4.2–4.3) for the Fourier comb exact aliasing counterexample.
4. **Invalidation Conditions:**
   - The linear scaling isomorphism would be invalidated if an affine scaling $cz$ could be shown to alter normalized interior coordinates $z_k = (x_k - a)/R$ with $(a, R)$ held fixed.
   - The EVQ-Cosh derivation would be invalidated if the kernel $\min(\phi, \psi)$ were shown to fail positive semi-definiteness on $L^2([0, 1])$.

