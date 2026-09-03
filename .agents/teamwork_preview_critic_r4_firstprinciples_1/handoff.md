# Handoff Report — First-Principles Theorist (R4)

**Role:** First-Principles Theorist (R4)  
**Date:** 2026-09-02  
**Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_critic_r4_firstprinciples_1`  
**Target Artifacts:** `analysis.md`, `handoff.md`  
**Handoff Type:** Hard (Task complete)

---

## 1. Observation
1. **RoPE Attention Logit Formula & Complex Factorization:**
   Direct expansion of $s_{ij}(\Delta) = q_i^\top R(\Delta) k_j$ over $K = d/2$ 2D orthogonal rotation blocks yields $s_{ij}(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k e^{i \omega_k \Delta}] = \sum_{k=0}^{K-1} A_k \cos(\omega_k \Delta + \psi_k)$, where $c_k = (q_{i, 2k} - i q_{i, 2k+1})(k_{j, 2k} + i k_{j, 2k+1})$.
2. **50M 2x2 Factorial Crossing (Table $\times$ Weights Interaction):**
   In `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5.2–5.3, lines 363–401:
   - Self-consistent cells: `GG` loss $1.9659$ (PPL $7.14$), `EE` loss $1.9685$ (PPL $7.16$).
   - Mismatched cells: `GE` loss $4.3333$ (PPL $76.20$), `EG` loss $3.1378$ (PPL $23.05$).
   - Factorial effects: Table main effect $E_T = +0.5991$, Weights main effect $E_W = -0.5965$, Interaction effect $I_{T \times W} = -3.5367$ (95% CI $[-5.165, -3.039]$). The interaction is $5.9\times$ larger in magnitude than either main effect.
3. **Same-Multiset Permutation Collapse:**
   In `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 and `INDEX.md` §0 (lines 25–27):
   Permuting the 64 rotary slot assignments while preserving the exact frequency multiset collapses frozen models: OLMo 1x PG-19 NLL explodes from $3.10423 \to 6.86493$, and Qwen 64K core-4 score drops from $0.7000 \to 0.0000$.
4. **Post-Hoc Frequency Transplant Obstruction:**
   In `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` lines 65–109:
   Universal exact linear compensation $A^\top R(\Omega'\Delta) B = R(\Omega\Delta)$ requires $A^\top B = I$ and generator similarity, forcing the absolute frequency multisets to coincide: $\{|\omega'_k|\} = \{|\omega_k|\}$.
5. **Divergent Task Tolerance under Native-Only Retrofits:**
   In `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.D and §H (lines 35–43, 764–779):
   - `log_s4` achieves 1x PG-19 retention of $0.875302$ and RULER-16K score of $0.49859$, but drops on HotpotQA 16K F1 to $0.153$.
   - Official `YaRN-4` achieves HotpotQA 16K F1 of $0.500$, but fails retention gate at $0.6588$ and drops on RULER-16K to $0.1056$.
   - Both are Native-derived constructions; their inverted task rankings confirm non-identifiability from Native metrics.

---

## 2. Logic Chain
1. **From Observation 1 (Score Decomposition):**
   The attention score is a linear combination of harmonic carriers with content-dependent complex coefficients $c_k$. At $\Delta = 0$, $s_{ij}(0) = \sum_k \operatorname{Re}[c_k] = q_i^\top k_j$ is invariant under any frequency modification. However, for any $\Delta > 0$, the logit deviation is bounded tightly by $|s'(\Delta) - s(\Delta)| \le \sum_k |c_k| \min(2, |\omega'_k - \omega_k|\Delta)$.
2. **From Observation 2 & 3 (Co-adaptation and Slot Ordering):**
   Because $W_q^{(k)}, W_k^{(k)}$ are trained under carrier $\omega_k$, the network learns a coupled joint representation where phase shifts $\omega_k \Delta$ constructively synthesize attention peaks for salient contexts. Scrambling the slot-frequency assignment destroys this interference pattern, proving that the learned positional object is the ordered sequence of tuples $((W_q^{(k)}, W_k^{(k)}), \omega_k)_{k=0}^{K-1}$, not the unordered spectrum.
3. **From Observation 4 & Theorem 1 (Transplant Rigidity):**
   Any attempt to alter frequencies $\Omega \to \Omega'$ cannot be undone by a frozen, position-independent linear transformation $A, B$ because Lie generator similarity requires identical spectrum eigenvalues $\{|\omega'_k|\} = \{|\omega_k|\}$. Thus, frequency reallocation is structurally incompatible with fixed linear compensation.
4. **From Theorem 2 & Observation 5 (Off-Arc Torus Exposure & Non-Identifiability):**
   Uniform scaling ($\rho_k \equiv 1/S$) is the unique family where the deployed phase trajectory $\gamma'([0, SL])$ remains inside the native manifold $\gamma([0, L])$ on $\mathbb{T}^K$. Any non-uniform scaling ($\rho_k \ne \rho_l$) breaks collinearity, generating phase tuples off the native manifold. Because the frozen network's downstream feed-forward and attention circuits never saw off-arc phase combinations during pretraining, its behavior on them cannot be predicted by any Native-range metric ($L \le L_{\text{train}}$).
5. **From Conditioning Theorem 4:**
   The error amplification factor between native window $L$ and horizon $SL$ is exactly $S$. Consequently, zero-training extrapolation is an inherently ill-posed problem whose error conditioning degrades linearly with scale factor $S$.

---

## 3. Caveats
1. The analysis addresses post-hoc frequency modifications on frozen checkpoints without subsequent parameter training. It does not preclude approximate re-learning or parameter recovery if lightweight LoRA or full fine-tuning with long-context data is permitted.
2. The score-norm bounds provide necessary conditions and worst-case envelopes. The realized functional damage on specific downstream benchmarks depends on individual task circuit margins (e.g. margin between relevant keys and distractors), which vary by architecture and task.
3. Multiset permutation collapse has been empirically verified at scale on OLMo and Qwen checkpoints; slight non-monotonic order crossings (such as pairs 1 and 18 in Qwen construction transfer) can survive, indicating that strict monotonicity is a sufficient rather than strictly necessary condition.

---

## 4. Conclusion
1. **What is controlled:** Altering $\Omega \to \Omega'$ controls the phase velocity vector on $\mathbb{T}^K$, per-channel effective distance metrics $\Delta_k^{\mathrm{eff}} = \rho_k \Delta$, and the spectral waterbed boundary $r_k(S) = S^{1-m_k}$ allocating novelty versus blur.
2. **What is disrupted:** Altering $\Omega \to \Omega'$ inevitably disrupts:
   - In-window constructive interference of high-frequency channels (flipping signs within $\Delta \approx 4$ tokens if compressed).
   - Joint phase coherence on the torus $\mathbb{T}^K$ (off-arc exposure for all non-uniform retrofits).
   - Linear compensability (proven strictly impossible by Transplant Rigidity).
   - Downstream softmax decisiveness and decision boundary margins.
3. **Core Dilemma:** Native retention demands $\rho_k \approx 1$ ($m_k \approx 0$), whereas long de-aliasing demands $\rho_k \le 1/S$ ($m_k \ge 1$). A single static zero-training table cannot resolve this structural trade-off across diverse task families without trade-offs.

---

## 5. Verification Method
1. **Mathematical Verification:**
   - Review Theorem 1 proof via Lie algebra generator similarity at $\Delta = 0$.
   - Review Theorem 2 proof via Euclidean arc length and collinearity on $\mathbb{T}^K$.
   - Review Theorem 3 compatibility modulus and Theorem 4 conditioning factor $S$.
2. **Repository File Inspection:**
   - Inspect `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5.2–5.3 for 50M 2x2 factorial numbers.
   - Inspect `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` for transplant obstruction proof.
   - Inspect `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 for same-multiset permutation numbers.
   - Inspect `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.D and §I for `log_s4` vs `YaRN-4` empirical divergence.
3. **Invalidation Condition:**
   - The theory would be invalidated if an invertible, position-independent linear map $A, B$ is shown to preserve $R(\Omega\Delta)$ for $\Omega' \ne \Omega$ across all $\Delta$, or if a non-uniform static table is shown to remain strictly on the 1D native phase manifold on $\mathbb{T}^K$.
