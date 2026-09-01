# Handoff Report: Information-Theoretic Foundations of RoPE Exponent Spectrum $z = -2i/d$

- **Agent**: `explorer_r1_infotheory_1`
- **Date**: 2026-09-01
- **Focus**: R1 Axis A — Information-Theoretic Foundations of $z = -2i/d$, Continuous Spectral Density $\rho(\omega) = \frac{1}{\omega \ln b}$, Multi-Resolution Wavelet Framing, Octave Channel Capacity, Phase Entropy, and Riemann-Lebesgue Locality Prior.
- **Report Location**: `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r1_infotheory_1/report.md`

---

## 1. Observation

1. **Repository Rules and Claim Ceilings (`AGENTS.md`, lines 58–70, 114–140)**:
   - Full-RoPE geometry represents static, phase-invariant positional-basis redundancy/effective dimension; it is not an LM-quality or extrapolation predictor.
   - Low-frequency collapse: slow bands are redundant in the stated metric ($V_\omega \to \operatorname{span}\{1, \Delta\}$; softmax metric $\to \operatorname{span}\{\Delta - \bar\Delta, \Delta^2 - \bar{\Delta^2}\}$); not dead, unused, or freely reclaimable.
   - EVQ-Cosh is a closed-form, zero-learned-parameter construction and controlled intervention on the allocation axis, unique only for the stated convex surrogate functional $\mathcal{J}[\rho]$.
   - Fixed transplant obstruction: exact fixed invertible Q/K compensation is obstructed for unequal frequency multisets ($A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta) \implies \Omega' = \Omega$).
2. **Theory Architecture and Spectral Baseline (`paper-2027/research/ICLR2027_THEORY_ARCHITECTURE.md`, lines 40–127)**:
   - Relative-position attention logit: $\ell(\Delta) = \sum_i [C_i \cos(\omega_i \Delta) + D_i \sin(\omega_i \Delta)] = \sum_i A_i \cos(\omega_i \Delta + \psi_i)$.
   - Stable effective rank identity: $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$, where $\bar{c}$ is the mean whitened cross-channel redundancy.
   - Low-frequency redundancy: for $K=64, L=4096, b=500000$, the 24 slowest pairs ($\omega L \le 1$) yield $r_2 \approx 2.0002$, a $95.83\%$ dimension redundancy loss.
3. **Discrete to Continuous Spectrum Formulation**:
   - Discrete frequency: $\omega_i = b^{-2i/d} = b^{z_i}$ with $z_i = -2i/d \in [-1 + 2/d, 0]$ for $i \in \{0, \dots, d/2-1\}$.
   - Continuous parameterization: normalized channel index $u = 2i/d \in [0, 1]$, exponent profile $z(u) = -u$, continuous frequency $\omega(u) = b^{-u} = \exp(-u \ln b) \in [b^{-1}, 1]$.

---

## 2. Logic Chain

1. **Step 1 (Discrete Scale Homogeneity $\to z_i = -2i/d$)**:
   - *Premise (Observation 3)*: Enforcing boundary conditions ($\omega_0 = 1, \omega_{K-1} \approx b^{-1}$) and scale-invariance / constant ratio across adjacent channels ($\omega_i / \omega_{i+1} = r = \text{const}$) yields $r^K = b \implies r = b^{1/K} = b^{2/d}$.
   - *Deduction*: $\omega_i = r^{-i} = b^{-2i/d} \iff z_i = -2i/d$. This corresponds to an equispaced grid in logarithmic frequency space $x_i = -\ln \omega_i = i \frac{\ln b}{K}$.
2. **Step 2 (Continuous Limit $\to \rho(\omega) = \frac{1}{\omega \ln b}$)**:
   - *Premise (Observation 3)*: For continuous $u \in [0, 1]$ with uniform channel distribution $p_U(u) = 1$, $\Omega = b^{-U}$.
   - *Deduction*: The CDF is $F_\Omega(\omega) = 1 + \frac{\ln \omega}{\ln b}$. The PDF is $\rho(\omega) = \left|\frac{du}{d\omega}\right| = \frac{1}{\omega \ln b}$ on $[b^{-1}, 1]$.
   - *Property*: Scale invariance $\rho(\lambda \omega) = \lambda^{-1}\rho(\omega)$ uniquely characterizes $\rho(\omega) \propto 1/\omega$, corresponding to the Haar measure on $(\mathbb{R}^+, \times)$ and Jeffreys uninformative prior.
3. **Step 3 (Multi-Resolution Analysis & Wavelet Framing)**:
   - *Premise (Observation 2)*: Each rotary channel is a 2D harmonic subspace $V_{\omega_i} = \operatorname{span}\{\cos(\omega_i \Delta), \sin(\omega_i \Delta)\}$ at spatial scale $a_i = 1/\omega_i = b^{2i/d}$.
   - *Deduction*: RoPE forms a dyadic multi-resolution filter bank decomposing position into hierarchical detail bands. High frequencies resolve micro-scale token transitions (syntax, n-grams); low frequencies provide macro-scale monotonic position ordering.
   - *Degeneracy*: When context window $L \ll T_{\text{slow}}$, the slow channels fail to complete oscillations, collapsing to $V_\omega \to \operatorname{span}\{1, \Delta\}$ and softmax centered subspace $\operatorname{span}\{\Delta - \bar\Delta, \Delta^2 - \bar{\Delta^2}\}$, yielding severe stable rank reduction ($r_2 \to 2$).
4. **Step 4 (Information Capacity and Differential Phase Entropy)**:
   - *Deduction A*: Channel capacity per octave is equipartitioned: $K_{\text{octave}} = \frac{d \ln 2}{2 \ln b} = \text{const} \implies \mathcal{C}_{\text{octave}} = \text{const}$.
   - *Deduction B*: Differential phase entropy $H(\Theta_i)$ for wrapped phase $\Theta_i = (\omega_i \Delta) \pmod{2\pi}$ diverges across scales:
     - Fast channels ($\omega_i L \gg 2\pi$): $H(\Theta_i) \to \ln(2\pi) \approx 1.838$ nats (maximal uniform entropy).
     - Slow channels ($\omega_i L \ll 2\pi$): $H(\Theta_i) = H(\Delta) + \ln \omega_i = \ln(\omega_i L)$, yielding an entropy deficit $\Delta H_i = \ln\left(\frac{2\pi}{\omega_i L}\right)$.
5. **Step 5 (Expected Attention Kernel & Riemann-Lebesgue Locality)**:
   - *Deduction*: The continuous expected attention kernel is given by the exact Cosine Integral formula $\bar{K}(\Delta) = \int_{b^{-1}}^1 \cos(\omega \Delta) \rho(\omega) d\omega = \frac{\operatorname{Ci}(\Delta) - \operatorname{Ci}(\Delta/b)}{\ln b}$.
   - *Asymptotic Regimes*:
     - $\Delta \to 0$: $\bar{K}(0) = 1$ (maximal constructive interference).
     - $1 \ll \Delta \ll b$: $\bar{K}(\Delta) \approx 1 - \frac{\ln \Delta}{\ln b}$ (natural logarithmic locality decay).
     - $\Delta \gg b$: $O(1/\Delta) \to 0$ (Riemann-Lebesgue destructive interference).
   - *Extrapolation Consequence*: Discrete finite-$K$ sampling creates quasi-periodic Poincaré revivals and phase aliasing for $\Delta > L_{\text{train}}$, causing attention noise and softmax entropy collapse unless weights and spectrum are properly adapted.

---

## 3. Caveats

- **Static vs. Task Performance**: All derived spectral densities, stable ranks, and differential phase entropies represent static geometric properties of the positional basis. In accordance with `AGENTS.md` and repository 50M 2x2 factorial findings, static geometric optimality does not imply lower task loss or better extrapolation in isolation without learned Q/K co-adaptation.
- **Continuous Approximation**: The integral kernel $\bar{K}(\Delta) = \frac{\operatorname{Ci}(\Delta) - \operatorname{Ci}(\Delta/b)}{\ln b}$ is the exact continuum limit ($K \to \infty$). Real transformers operate with finite $K \in \{32, 64\}$, which introduces discrete residual oscillations around the continuous logarithmic decay curve.
- **Isotropy Assumption**: The analytical form of $\bar{K}(\Delta)$ assumes an isotropic query/key prior $\mathbb{E}[C_i] = 1, \mathbb{E}[D_i] = 0$. Trained transformers learn non-uniform content amplitudes $A_i$ and layer/head-specific phase allocations.

---

## 4. Conclusion

The RoPE exponent allocation $z_i = -2i/d$ is the unique uniform discretization of the continuous scale-invariant Haar density $\rho(\omega) = \frac{1}{\omega \ln b}$. This density endows attention with an innate logarithmic positional locality prior ($\bar{K}(\Delta) \approx 1 - \frac{\ln \Delta}{\ln b}$) via Riemann-Lebesgue destructive phase interference. However, over finite context windows $L \ll b$, the uniform logarithmic allocation assigns substantial channel budget to slow frequencies that suffer from severe phase entropy deficit ($\Delta H_i = \ln(\frac{2\pi}{\omega_i L})$) and 2D subspace collapse ($V_\omega \to \operatorname{span}\{1, \Delta\}$). Non-linear spectrum allocations (such as EVQ-Cosh) systematically redistribute this redundant low-frequency budget toward richer intermediate bands.

---

## 5. Verification Method

To independently verify all mathematical derivations and numerical properties:

1. **Analytical Formula Check**:
   - Verify that $\int_{b^{-1}}^1 \frac{1}{\omega \ln b} d\omega = 1$.
   - Verify differentiation: $\frac{d}{d\Delta} \left[ \frac{\operatorname{Ci}(\Delta) - \operatorname{Ci}(\Delta/b)}{\ln b} \right] = \frac{1}{\ln b} \left[ \frac{\cos \Delta}{\Delta} - \frac{\cos(\Delta/b)}{\Delta} \right]$.
2. **Python Numerical Verification**:
   ```python
   import numpy as np
   from scipy.special import ci

   b = 10000.0
   L = 4096.0
   # Verify Continuous Kernel vs. Monte Carlo / Numerical Integration
   delta_vals = np.logspace(0, 4, 100)
   k_analytical = (ci(delta_vals) - ci(delta_vals / b)) / np.log(b)
   k_approx = 1.0 - np.log(delta_vals) / np.log(b)  # for 1 << delta << b
   assert np.allclose(k_analytical[10:60], k_approx[10:60], atol=0.1)
   ```
3. **Repository File Inspection**:
   - Inspect `paper-2027/research/ICLR2027_THEORY_ARCHITECTURE.md` §1–§3 for repository theorems on subspace geometry, stable rank $r_2$, and low-frequency collapse.
   - Inspect `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` for exact canonical redundancy derivations.
