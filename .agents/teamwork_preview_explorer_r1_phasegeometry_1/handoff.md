# Handoff Report — explorer_r1_phasegeometry_1

- **Role:** Explorer Subagent (R1 Axis B: Phase Code $\Phi(\Delta)$ Geometry, Gram Matrix & Extrapolation Breakdown)
- **Target Report:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r1_phasegeometry_1/report.md`
- **Date:** 2026-09-01

---

## 1. Observation

1. **Phase Code Definition & 2D Subspace Structure:**
   - In `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` lines 71–80:
     $$f_\omega(\Delta) = C\cos(\omega\Delta) + D\sin(\omega\Delta), \qquad V_\omega = \operatorname{span}\{\cos(\omega\Delta), \sin(\omega\Delta)\}.$$
     The cosine-only kernel $K_{\cos}(\omega, \nu) = \mathbb{E}_D[\cos(\omega\Delta)\cos(\nu\Delta)]$ observes only single content phase $D=0$ and is not invariant to intra-pair phase rotation.
2. **Full-Subspace Gram & Exact Stable Rank Theorem:**
   - In `scripts/analysis/full_rope_collision_audit.py` lines 37–49 (`gram_blocks`), lines 57–68 (`whitened_blocks`), and lines 78–107 (`schedule_metrics`), the block-whitened cross-Gram $Q_{jk} = S_j^{-1/2} H_{jk} S_k^{-1/2}$ produces canonical correlations $\sigma_1, \sigma_2$, with collision metric $c_{jk} = \frac{1}{2}\|Q_{jk}\|_F^2 = \frac{\sigma_1^2 + \sigma_2^2}{2}$.
   - In `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` lines 142–160:
     $$\operatorname{tr}(R) = 2K, \quad \operatorname{tr}(R^2) = 2K[1 + (K-1)\bar{c}], \quad r_2(R) = \frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)} = \frac{2K}{1 + (K-1)\bar{c}}.$$
3. **Low-Frequency Spectral Collapse & Softmax Centering:**
   - In `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` lines 171–193 & 218–244:
     Under standard $L_2([0, L])$, as $\omega \to 0$, $V_\omega \to \operatorname{span}\{1, \Delta\}$ with leading deficit $2 - \|Q_{xy}\|_F^2 = \frac{19}{12600}(x^2 - y^2)^2 + O(\epsilon^6)$.
     Under attention softmax categorical Fisher $F_{\text{sm}} = \operatorname{diag}(p) - pp^\top$ ($F_{\text{sm}}\mathbf{1} = 0$), the constant mode is eliminated, yielding centered quotient convergence:
     $$\frac{\overline{\sin(\omega\Delta)}}{\omega} \to \Delta - \mathbb{E}_p\Delta, \quad -\frac{2\overline{\cos(\omega\Delta)}}{\omega^2} \to \Delta^2 - \mathbb{E}_p\Delta^2 \implies V_\omega^{\text{sm}} \to \operatorname{span}\{\Delta - \mathbb{E}_p\Delta, \Delta^2 - \mathbb{E}_p\Delta^2\}.$$
   - At $L = 4096, b = 500\,000, K = 64$, 23 slow pairs ($\omega L \le 1$, 46 nominal dimensions) collapse to $r_2 \approx 2.00013$ and raw entropy rank $1.079$ ($>95.65\%$ stable dimensionality loss).
4. **Falsified Routes & Counterexamples:**
   - In `INDEX.md` lines 151–165 (§3.4):
     - #1: Cosine-only collision kernel selects inverted rank ($C_{\cos}(A) < C_{\cos}(B)$ but $r_2(A) < r_2(B)$).
     - #2: Minimizing collision yields Fourier harmonic comb $\omega_k = 2\pi k / L$, which achieves $r_2 = 2K$ on $[0, L]$ but exhibits exact catastrophic aliasing $\Phi(\Delta + L) = \Phi(\Delta)$ out-of-distribution.
     - Length ranking inversion across $L, 2L, 4L$ ($C_L(A) < C_L(B)$ but $C_{2L}(A) > C_{2L}(B)$).

---

## 2. Logic Chain

1. **Torus Embedding Structure:** (from Observation 1)
   $\Phi(\Delta) = [\cos(\omega_0 \Delta), \sin(\omega_0 \Delta), \dots, \cos(\omega_{K-1} \Delta), \sin(\omega_{K-1} \Delta)]^\top \in \mathbb{R}^{2K}$ is a map $\Phi: \mathbb{R} \to \mathbb{T}^K \subset \mathbb{R}^{2K}$. Each channel $j$ has constant norm $\|x_j(\Delta)\|_2 = 1$, and total phase norm is $\|\Phi(\Delta)\|_2 = \sqrt{K}$.
2. **Stationary Gram Kernel and Euclidean Metric:** (from Observation 1 & 2)
   The pointwise inner product $\langle \Phi(\Delta), \Phi(\Delta') \rangle = \sum_{j=0}^{K-1} \cos(\omega_j(\Delta - \Delta')) = G(\delta)$ is strictly shift-invariant. The Euclidean distance is $D^2(\delta) = 2K - 2G(\delta) = 4\sum_{j=0}^{K-1} \sin^2(\omega_j \delta / 2)$, with near-field quadratic curvature $D^2(\delta) \approx \Omega_{\text{tot}}^2 \delta^2$.
3. **Exact Stable Rank Invariant:** (from Observation 2)
   Forming the block-whitened correlation matrix $R$ over a prior measure $p(\Delta)$ gives $K$ diagonal blocks $I_2$ ($\operatorname{tr}(R) = 2K$) and off-diagonal blocks $Q_{jk}$ ($\|Q_{jk}\|_F^2 = 2c_{jk}$). Summing all entries gives $\operatorname{tr}(R^2) = 2K[1 + (K-1)\bar{c}]$, proving the exact identity $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$.
4. **Mechanism of Dual-End Extrapolation Breakdown:** (from Observations 3 & 4)
   - *High-frequency aliasing:* Fast bands ($\omega_j \gg 2\pi/L$) wrap around $S^1$ multiple times in-window ($N_j \gg 1$). At $\Delta > L_{\text{train}}$, they produce pseudo-random phase noise with variance $\sim K_{\text{fast}}/2$.
   - *Low-frequency turnover:* Slow bands ($\omega_j L_{\text{train}} \ll 1$) span $\approx 2$ effective dimensions during training (polynomial $L_2$ collapse and centered quadratic softmax collapse). At $\Delta > L_{\text{train}}$, $\omega_j \Delta$ exits the linear regime into sinusoidal turnover ($\omega_j \Delta \sim \pi/2, \pi$), turning a trained monotonic distance coordinate into a decreasing function.
5. **Decoupling Static Geometry from Extrapolation & Claim Ceiling:** (from Observation 4 & `AGENTS.md`)
   Static basis geometry describes representation capacity on $[0, L_{\text{train}}]$. Maximizing static rank leads to pathological harmonic combs with exact periodic aliasing $\Phi(\Delta + L) = \Phi(\Delta)$. Therefore, static geometry is a descriptive basis invariant, NOT an LM-quality or extrapolation predictor.

---

## 3. Caveats

1. **Continuous vs. Discrete Distance Measure:**
   The analytical Gram integrals use continuous uniform distribution $\Delta \in [0, L]$. On discrete token grids $\Delta \in \{0, 1, \dots, L-1\}$, differences are $O(1/L)$ and vanish for large $L$, but can create slight finite-grid shifts at small $L$.
2. **Attention Softmax Weights:**
   The softmax quotient limit uses the categorical Fisher $F_{\text{sm}} = \operatorname{diag}(p) - pp^\top$. Real attention weights $p$ are token- and context-dependent; the 2D centered polynomial limit holds for any non-degenerate attention distribution supported on $\ge 3$ distinct points.
3. **Static Basis vs. Dynamic Model Execution:**
   Static analysis does not model the learned projection matrices $W_q, W_k$, MLP layers, or LayerNorm. Dynamic LM performance is mediated by table $\times$ weight co-adaptation (as established in the 50M $2\times 2$ study).

---

## 4. Conclusion

1. The RoPE phase code $\Phi(\Delta) \in \mathbb{R}^{2K}$ is a multi-frequency linear winding flow on the flat torus $\mathbb{T}^K$. Its pointwise Gram kernel $G(\delta) = \sum_{j=0}^{K-1} \cos(\omega_j \delta)$ and Euclidean distance $D^2(\delta) = 4\sum_{j=0}^{K-1} \sin^2(\omega_j \delta / 2)$ provide shift-invariant positional metric structure.
2. The block-whitened correlation matrix $R$ satisfies the exact stable rank identity $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$, quantifying basis redundancy across 2D frequency subspaces.
3. Original geometric RoPE fails under extrapolation ($\Delta > L_{\text{train}}$) due to a dual-end geometric failure:
   - High-frequency phase aliasing generates pseudo-random background noise.
   - Low-frequency spectral collapse ($>95\%$ dimensionality loss in-window) leads to non-linear sinusoidal turnover and distance coordinate inversion out-of-window.
4. Static collision minimization or rank maximization is NOT a valid extrapolation objective due to exact harmonic aliasing and domain ranking inversions.

---

## 5. Verification Method

To independently verify the theoretical derivations, stable rank identity, and counterexamples:

1. **Verify Full-RoPE Collision & Stable Rank Identity (CPU-only):**
   ```bash
   python3 scripts/analysis/full_rope_collision_audit.py
   ```
   *Expected Output:* Confirms exact match between direct stable rank $\operatorname{tr}(R)^2 / \operatorname{tr}(R^2)$ and analytical formula $2K / (1 + (K-1)\bar{c})$ (tolerance $< 2\times 10^{-10}$), and reproduces the three counterexamples.
2. **Verify Analytical Integrals & Beat Identities (CPU-only):**
   ```bash
   python3 scripts/analysis/verify_signed_lag_kway_gap.py
   ```
3. **Verify Third-Axis Search and Decomposition:**
   ```bash
   python3 scripts/analysis/third_axis_ceiling.py --decompose
   ```
4. **Inspect Generated Report:**
   ```bash
   cat .agents/teamwork_preview_explorer_r1_phasegeometry_1/report.md
   ```

---
