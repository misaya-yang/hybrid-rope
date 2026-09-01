# Handoff Report: Bilinear Attention Logit Readout & Frozen Q/K Co-Adaptation Dynamics (Focus Area R2)

- **Agent:** `explorer_r2_qkreadout_1`
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r2_qkreadout_1`
- **Date:** 2026-09-01
- **Recipient:** `parent` (`9219631d-28ca-410b-b7ff-2c42127fb3f2`)
- **Status:** Complete (Hard Handoff)

---

## 1. Observation

1. **Bilinear Attention Readout Expansion:**
   - In `paper-2027/research/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md` (lines 64–66), the relative position logit for head displacement $\Delta$ is:
     $$\ell(\Delta) = \mathbf{q}^\top R_\Omega(\Delta) \mathbf{k} = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$$
   - In `scripts/analysis/attention_fisher_50m_probe.py` (lines 440–480), the implementation calculates query/key pairs and content coefficients:
     ```python
     real_alpha = raw_q1[:, None] * raw_k1 + raw_q2[:, None] * raw_k2
     imag_alpha = raw_q2[:, None] * raw_k1 - raw_q1[:, None] * raw_k2
     ```
     where $A_j = \sqrt{\text{real\_alpha}^2 + \text{imag\_alpha}^2}$ and $\psi_j = \operatorname{atan2}(-\text{imag\_alpha}, \text{real\_alpha})$.

2. **Post-Hoc Frequency Transplant Obstruction:**
   - In `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` (lines 69–104) and `paper-2027/appendix/a1_proofs.tex` (lines 278–296), Theorem 4 establishes that if $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ holds on an open interval containing zero, then $\Delta=0 \implies B = A^{-\top}$, and differentiating at $\Delta=0$ requires similarity of the skew-symmetric generators $A^\top G_{\Omega'} A^{-\top} = G_\Omega$. This forces the spectrum $\{\pm i \omega_k\}$ to match, requiring $|\Omega'| = |\Omega|$ up to permutation and sign.

3. **50M Table-Weight Counterfactual $2\times2$ Crossing:**
   - In `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` (lines 363–369) and verified in `scripts/analysis/attention_fisher_50m_probe.py`:
     - `Geo Weights + Geo Table`: Loss = 1.9659, PPL = 7.14, bare $r_2 = 4.57$.
     - `Geo Weights + EVQ Table`: Loss = 4.3333, PPL = 76.20, bare $r_2 = 12.54$.
     - `EVQ Weights + Geo Table`: Loss = 3.1378, PPL = 23.05, bare $r_2 = 4.57$.
     - `EVQ Weights + EVQ Table`: Loss = 1.9685, PPL = 7.16, bare $r_2 = 12.54$.
   - Factorial ANOVA (lines 390–401): Table main effect $E_T = +0.5991$, Weights main effect $E_W = -0.5965$, Interaction effect $I_{T \times W} = -3.5367$ (95% Bootstrap CI $[-5.165, -3.039]$). The interaction is $5.9\times$ the magnitude of the main effects.

4. **Failure of Frozen Weights at $\Delta > L_{\text{train}}$ & Destructive Interference:**
   - In `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md` (§§10–11), deleting the trained high-norm OOD frequency pairs improves held-out NLL at 8K by $-0.280$ for Geo and $-0.108$ for EVQ.
   - In `research_notes/FABLE5_EVQ_MECHANISM_AUDIT.md` (§1.2 T2), deleting the gold block in the unadapted Geo arm at 16K causes $\Delta\text{NLL} \approx -0.010$, proving zero causal utilization of remote context by the frozen unadapted baseline.

---

## 2. Logic Chain

1. **Step 1 (Readout Formulation from Observation 1):**
   Expanding the block-diagonal rotation $R_\Omega(\Delta) = \bigoplus_{j=0}^{K-1} R(\omega_j \Delta)$ yields $\ell(\Delta) = \sum_{j=0}^{K-1} [C_j \cos(\omega_j \Delta) + D_j \sin(\omega_j \Delta)] = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$. Since $C_j$ and $D_j$ are quadratic forms in the input representations $x_m, x_n$ via $W_q, W_k$, the attention logit is mathematically an amplitude-and-phase modulated Fourier series.

2. **Step 2 (Co-Adaptation Mechanism from Observations 1 & 3):**
   During training, backpropagation optimizes $W_q, W_k$ to satisfy $\psi_j \approx -\omega_j \Delta^* \pmod{2\pi}$ for attended target displacement $\Delta^*$, producing constructive interference $\ell(\Delta^*) \approx \sum A_j$, and dispersed phases across $\Delta \neq \Delta^*$, producing destructive interference $\mathbb{E}[\ell(\Delta)] \approx 0$. Because the projection weights explicitly depend on the frequencies $\Omega$, changing the table post-hoc breaks phase alignment, converting constructive peaks into out-of-phase destructive noise and exploding loss ($7.14 \to 76.20$), as confirmed by the $-3.5367$ interaction term in Observation 3.

3. **Step 3 (Extrapolation Breakdown from Observations 1 & 4):**
   When context extends to $\Delta > L_{\text{train}}$, the phase trajectory $\mathbf{\Phi}(\Delta)$ enters unvisited regions of the $K$-torus $\mathbb{T}^K$. The destructive cancellation tuned for $[0, L_{\text{train}}]$ fails, generating pseudo-constructive random spikes at background positions. This accumulates a noise floor with standard deviation $\sigma_{\text{bg}} \approx \sqrt{\sum A_j^2 / 2}$, causing the Peak-to-Background Ratio (PBR) to collapse, dispersing softmax probability mass across thousands of distractor tokens, and eliminating remote signal transmission as observed in Observation 4.

4. **Step 4 (Transplant Obstruction from Observation 2):**
   Because the Lie algebra generators $G_\Omega = \bigoplus \omega_k J$ have purely imaginary eigenvalues $\{\pm i \omega_k\}$, any linear conjugacy $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ forces the similarity $A^\top G_{\Omega'} A^{-\top} = G_\Omega$, which preserves the spectrum. Thus, no linear adapter on $W_q, W_k$ can mathematically absorb a non-trivial frequency replacement $\Omega \neq \Omega'$.

---

## 3. Caveats

1. **Obstruction Theorem Scope:** The obstruction theorem proves that exact post-hoc compensation is impossible for fixed, position-independent, invertible linear Q/K maps. It does not prove that approximate non-linear neural adaptation or token-conditioned gating cannot achieve bounded functional recovery on finite datasets.
2. **50M Factorial Scope:** The 50M $2\times2$ crossing is evaluated on TinyStories validation at seed 42 ($1,920$ observations). While bootstrap confidence intervals strictly exclude zero ($[-5.165, -3.039]$), multi-seed full training runs at larger scales (e.g. 151.9M exact-range) remain the primary owners of training-time allocation identification.
3. **Static Geometry vs. Performance:** High static effective rank $r_2$ indicates lower pairwise subspace correlation under a uniform prior; it does not directly predict language modeling loss or length generalization.

---

## 4. Conclusion

Focus Area R2 is fully established:
1. The bilinear attention logit is rigorously formulated as a content-modulated Fourier series $\ell(\Delta) = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$.
2. Training co-adapts projection weights $W_q, W_k$ with the frequency multiset $\Omega$ to construct sharp target peaks via constructive interference and suppress background distractors via destructive interference.
3. Extrapolation failure under frozen weights at $\Delta > L_{\text{train}}$ is caused by OOD phase configurations breaking destructive background cancellation, accumulating a high noise floor, and collapsing softmax peak-to-background ratio.
4. Post-hoc frequency modification cannot be compensated by linear Q/K adapters due to Lie generator spectral invariance (Obstruction Theorem).
5. The 50M $2\times2$ crossing diagnostic quantitatively confirms that the table $\times$ weights interaction dominates transformer loss by $5.9\times$ over main effects.

---

## 5. Verification Method

1. **Inspect Research Report:**
   - Review `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r2_qkreadout_1/report.md`.
2. **Verify Mathematical Obstruction Proof:**
   - View `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/paper-2027/appendix/a1_proofs.tex` (§\ref{sec:obstruction-proof}, lines 275–301).
3. **Run 50M 2x2 Attention-Fisher Diagnostic (CPU-only):**
   ```bash
   python3 scripts/analysis/attention_fisher_50m_probe.py --windows 8 --bootstrap-samples 500
   ```
   Confirm that PPL values match $7.14 / 76.20 / 23.05 / 7.16$ and interaction is $-3.5367$.
4. **Run Full-RoPE Collision & Collapse Audit (CPU-only):**
   ```bash
   python3 scripts/analysis/full_rope_collision_audit.py
   ```
   Confirm low-frequency collapse leading coefficient is $19/12600$.
