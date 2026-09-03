# Handoff Report: Mathematical Red Team (R2) Audit of RoPE Retrofit Theory

- **Author:** Mathematical Red Team (R2)
- **Role:** Reviewer, Critic, Specialist
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_critic_r2_mathredteam_1/`
- **Date:** 2026-09-03
- **Task:** Adversarial mathematical red team review of theorems, bounds, compatibility arguments, and asymptotic claims in zero-training RoPE retrofit research.

---

## 1. Observation

`[OBSERVED]` We examined the theoretical corpus across the repository, focusing on the active manuscript (`paper-2027/sections/03_theory.tex`, `paper-2027/appendix/a1_proofs.tex`), foundational memos (`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`), and recent retrofit theory notes (`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`, `MAXENT_DILATION_ALLOCATION_20260901.md`, `TARGET_FREE_PHASE_ISOTROPY_ALLOCATION_THEORY_20260824.md`).

Key verbatim observations:
1. `[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 60–69) and `appendix/a1_proofs.tex` (lines 38–47):
   Theorem 1 claims: $\operatorname{tr}\Gamma = 2K$, $\operatorname{tr}(\Gamma^2) = 2K[1+(K-1)\bar{c}]$, so $r_2(\Gamma) = \frac{2K}{1+(K-1)\bar{c}}$.
2. `[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 82–93) and `appendix/a1_proofs.tex` (lines 53–97):
   Proposition 1 claims: $2 - \|Q_{x,y}\|_F^2 = \frac{19}{12600}(x^2 - y^2)^2 + O(\epsilon^6)$, and centered limit is $\operatorname{span}\{\Delta - \mathbb{E}\Delta, \Delta^2 - \mathbb{E}\Delta^2\}$.
3. `[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 121–132) and `appendix/a1_proofs.tex` (lines 304–350):
   Theorem 2 derives $\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$ as the unique minimizer of $\mathcal{C}_{\mathrm{app}}[\rho] = \frac{\alpha}{2}\int \rho^2 + \frac{\beta}{2}\iint \rho(\phi)\rho(\psi)\min(\phi,\psi)$.
   In `appendix/a1_proofs.tex` (line 571), the transport bound is $W_1(\mu_{K,\tau}, \mu_{\rho_\tau}) \le \frac{\sinh\tau}{4K\tau}$.
4. `[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 183–199) and `appendix/a1_proofs.tex` (lines 278–296):
   Theorem 3 claims: $A^\top \mathcal{R}_{\Omega'}(\Delta) B = \mathcal{R}_\Omega(\Delta)$ on an interval forces $\Omega'$ and $\Omega$ to have the same frequency multiset, up to sign and permutation.
   In contrast, `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` documents that permuting frequency slots while preserving the exact multiset causes Qwen 64K RULER to collapse from $0.7000$ to $0.0000$, and OLMo 1x PG-19 NLL to jump from $3.10423$ to $6.86493$.
5. `[OBSERVED]` In `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`:
   - Theorem T3 states: $|s'(\Delta) - s(\Delta)| \le \sum_k |c_k| \min(2, |\omega'_k - \omega_k|\Delta)$, and $\|p' - p\|_1 \le e^{2\epsilon} - 1$.
   - Theorem T8 claims: "Multiplying a head's scores by $g > 0$ preserves every per-decision argmax... gain cannot alter which key is attended, only the concentration of the mixture."
6. `[OBSERVED]` In `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`:
   - Pillar 1 proves that $\max_{d,\gamma} \gamma - \frac{\lambda}{2}\|d\|^2$ s.t. $g_j^\top d + \gamma \le 0$ has closed form $d^* = -\bar{g}^*/\lambda$, and $\gamma^* > 0 \iff 0 \notin \operatorname{conv}\{g_j\}$.
   - Fact F in `FIRST_PRINCIPLES...` records that the 18-sample 64-D behavioral-gradient route failed its unopened holdout and has exited the main line.
   - Pillar 3 labels the "bimodal basin along dilation" as a model-dependent hypothesis based on assumptions A1–A3.

---

## 2. Logic Chain

1. `[DERIVED]` From Observation 1: Theorem 1 is an algebraic identity on block matrices with $I_2$ diagonal blocks. Because $\operatorname{tr}(I_2) = 2$ and $\|\Gamma_{ij}\|_F^2 = 2 c_{ij}$, the result $r_2(\Gamma) = \frac{2K}{1+(K-1)\bar{c}}$ is an algebraic tautology. It characterizes the unweighted positional basis under a specified prior $\mu$. In Table 1 of the paper, installing EVQ under Geo weights increases $r_2$ from 4.57 to 12.54 while increasing PPL from 7.14 to 76.20. Therefore, static effective rank has zero predictive or causal relationship with trained model performance.
2. `[DERIVED]` From Observation 3: In Theorem 2, the kernel $\min(\phi,\psi)$ is the Green's function for $-\frac{d^2}{d\phi^2}$ on $[0,1]$ with mixed boundary conditions. The Cosh solution $\rho_\tau$ is the eigenfunction of this specific differential operator. In real RoPE, the cross-subspace correlation is oscillatory ($\frac{\sin((\omega-\nu)L)}{(\omega-\nu)L}$). Furthermore, the discrete transport bound $W_1 \le \frac{\sinh\tau}{4K\tau}$ grows exponentially with $\tau$; at $\tau=6, K=32$, $W_1 \le 0.26$, meaning the continuous-to-discrete transport guarantees are vacuous for moderate and large $\tau$.
3. `[DERIVED]` From Observation 4: Theorem 3 proves that if position-independent linear maps $A, B$ can compensate for frequency changes, the frequency multisets must match up to permutation and sign. If $\Omega' = \pi(\Omega)$ is a permutation of $\Omega$, there exists an orthogonal permutation matrix $P$ such that $P^\top \mathcal{R}_{\pi(\Omega)}(\Delta) P = \mathcal{R}_\Omega(\Delta)$. Thus Theorem 3 algebraically permits slot permutations. However, in frozen models, $A = B = I$. The projection matrices $W_q, W_k$ bind specific semantic content to specific rotary slot indices. Therefore, Theorem 3's necessary condition is dangerously misleading if taken to imply that multiset preservation suffices for post-hoc retrofitting.
4. `[DERIVED]` From Observation 5: In Theorem T3, when $\Delta = 16{,}384$ and $S=4$, $|\omega'_k - \omega_k|\Delta \ge 2$ across most channels. For typical head norms, $\epsilon = \sup |s' - s| \ge 20$. The resulting softmax bound $\|p' - p\|_1 \le e^{2\epsilon} - 1 \ge e^{40} - 1 \approx 2.3 \times 10^{17}$. Since $\|p' - p\|_1 \le 2$ trivially for any two distributions, the bound is mathematically vacuous under real extrapolation.
5. `[DERIVED]` From Observation 5 (Theorem T8): We constructed a minimal 2-layer counterexample where Layer 1 attention logits are scaled by gain $g$. The output $z(g) = \sum p_j(g) v_j$ feeds into Layer 2 queries and keys. For $g < 0$, Key 2 wins at Layer 2; for $g > 0$, Key 1 wins at Layer 2. Thus, while gain preserves the argmax of a single isolated softmax, it alters hidden state mixtures and reverses downstream argmax rankings in multi-layer Transformers.
6. `[DERIVED]` From Observation 6: Pillar 1's common-direction theorem is mathematically sound as an application of QP duality (Désidéri's MGDA). However, it is an infinitesimal first-order property. When applied to 64 frequency dimensions on only 18 calibration samples, $0 \notin \operatorname{conv}\{g_j\}$ holds trivially due to linear independence ($18 < 64$), but the resulting direction severely overfits and fails on holdout evaluations. Furthermore, Pillar 3's "basin barrier" is an empirical hypothesis constructed by assuming a step function for long capability and a smooth quadratic bowl for native retention, not an architectural theorem.

---

## 3. Caveats

1. `[UNKNOWN]` We did not execute GPU training or inference to empirically measure the exact Lipschitz constant of the full multi-layer Transformer map under frequency perturbations, as GPU execution is strictly prohibited by our constraints.
2. `[HYPOTHESIS]` The multi-layer gain counterexample demonstrates that gain *can* flip downstream argmax rankings; the empirical prevalence and magnitude of this effect in 32-layer production LLMs remains unmeasured.
3. `[OBSERVED]` No caveats regarding the mathematical proofs in Section 03 and Appendix A1: all relevant linear algebra, calculus of variations, and matrix derivations were independently verified by symbolic and algebraic checks.

---

## 4. Conclusion

1. `[DERIVED]` **Strict Epistemic Demarcation:**
   - Theorems 1, 2, 3, 4 and Pillar 1 are mathematically valid deductive statements within their closed idealized definitions (matrix traces, 1D Laplacian Green's functionals, Lie generator similarity, and QP duality).
   - However, none of these theorems govern or guarantee the behavior of a deep, non-linear, multi-layer Transformer under finite context extrapolation.
2. `[DERIVED]` **Actionable Theoretical Guidance:**
   - Re-frame Theorem 3 to emphasize that frozen weights enforce an even stricter rigidity ($A=B=I$), explaining why same-multiset permutations collapse.
   - Retract or qualify claims that attention gain cannot change attended keys, explicitly limiting that claim to single-layer isolated attention.
   - Treat the "basin barrier", "waterbed effect", and "operating rule $\tau = d/\sqrt{L}$" as descriptive empirical regularities or conventions, not architectural theorems.
   - Discontinue underdetermined gradient-based common-direction searches ($K > J$) without strong structural regularizers.

---

## 5. Verification Method

To independently verify all mathematical findings in this report:

1. **Verify Matrix Budget Identity (Theorem 1):**
   Inspect `paper-2027/appendix/a1_proofs.tex` lines 38–47. Verify by direct expansion that for any $2K \times 2K$ symmetric matrix $\Gamma$ partitioned into $K \times K$ blocks of size $2 \times 2$ with $\Gamma_{ii} = I_2$, $\operatorname{tr}(\Gamma) = 2K$ and $\operatorname{tr}(\Gamma^2) = 2K + \sum_{i\ne j} \|\Gamma_{ij}\|_F^2$.
2. **Verify Multi-Layer Gain Argmax Inversion Counterexample:**
   Compute the 2-layer attention equations in `analysis.md` §3 (Topic 7) for $g = -1$ and $g = 2$. Confirm that $s_1^{(2)}(g) = \frac{e^g - 2}{e^g + 1}$ and $s_2^{(2)} = -0.5$. Evaluate:
   - For $g = -1$: $s_1^{(2)} \approx -1.193 < -0.5 = s_2^{(2)} \implies \arg\max = 2$.
   - For $g = 2$: $s_1^{(2)} \approx +0.642 > -0.5 = s_2^{(2)} \implies \arg\max = 1$.
   Confirms argmax ranking inversion.
3. **Verify Vacuity of Softmax Error Bound:**
   Evaluate $\|p' - p\|_1 \le e^{2\epsilon} - 1$ for $\epsilon \ge 1$. Observe that for $\epsilon = 1$, $e^2 - 1 \approx 6.389 > 2.0$, exceeding the maximal $L_1$ distance between any two probability distributions.
4. **Invalidation Conditions:**
   - This audit would be invalidated if an exact algebraic proof were provided showing that multi-layer Transformer representations are invariant to hidden-state mixture shifts induced by scalar gain.
   - This audit would be invalidated if Theorem 3's conditions were shown to strictly forbid orthogonal permutation matrices $P$.
