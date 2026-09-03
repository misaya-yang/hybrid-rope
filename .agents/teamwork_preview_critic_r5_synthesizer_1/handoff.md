# Handoff Report — Judge / Synthesizer (R5)

- **Role:** Judge / Synthesizer (R5)
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_critic_r5_synthesizer_1`
- **Date:** 2026-09-03
- **Full Report Path:** `.agents/teamwork_preview_critic_r5_synthesizer_1/synthesis.md`
- **Handoff Type:** Hard (Audit Complete)

---

## 1. Observation

1. `[OBSERVED]` `INDEX.md` lines 16–17 establishes the repository lifecycle: `PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`.
2. `[OBSERVED]` `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` lines 35–60 demonstrates that interior exponent allocation $z$ is a causal training variable across 3 paired seeds with consistent OOD direction at fixed support $(a, R)$, whereas target-matching support reverses the ordering.
3. `[OBSERVED]` `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` lines 67–99 proves that for invertible linear maps $A, B$, $A^\top R(\Omega'\Delta) B = R(\Omega\Delta)$ requires generator similarity, forcing multiset equality $\{|\omega'_k|\} = \{|\omega_k|\}$.
4. `[OBSERVED]` `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 185–192, 271–280 confirms that permuting rotary slot assignments while preserving the exact frequency multiset collapses OLMo 1x PG-19 NLL from $3.104234$ to $6.864926$ ($\Delta = +3.760692$) and collapses Qwen 64K core-4 RULER from $0.7000$ to $0.0000$.
5. `[OBSERVED]` `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 120–143 confirms that static log-s4 table passes the 1x double gate on OLMo (1x PG-19 PPL retention $0.875302 \ge 0.875$; downstream retention $0.915103 \ge 0.875$; 16K RULER $0.49859$), but continuing to $s=8$ collapses `single_key_3` to $0.00 / 0.20 / 0.25$ at 4K/8K/16K (lines 113–118).
6. `[OBSERVED]` `paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md` lines 58, 141 shows two-parameter $C2$ clipped-affine achieves movement MAE $0.001223$, yet fails Native retention on OLMo ($0.870971 < 0.875$) and Qwen ($0.868902 < 0.875$).
7. `[OBSERVED]` `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` lines 364–368 shows that in a 50M 2x2 factorial crossing, the table $\times$ weights interaction on LM loss is $-3.5367$ (95% CI $[-5.165, -3.039]$), $5.9\times$ larger than table main effect ($+0.5991$) and weights main effect ($-0.5965$), causing PPL to degrade from $7.14$ to $76.20$ when tables are swapped.
8. `[OBSERVED]` `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` lines 18–33, 128–178 shows that 512 free headwise gain parameters lower training loss from $3.5612$ to $2.3105$, but drop HotpotQA F1 from $0.24237$ to $0.19439$ and crash EOS terminations from $178/200$ to $108/200$. The basin barrier is demonstrated: log-start achieves HotpotQA F1 $0.25223$ but fails Native retention ($0.77138 < 0.875$); Native-start achieves Native retention $1.0462$ but fails HotpotQA ($0.02560$, 8/200 EOS).
9. `[OBSERVED]` `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` lines 147–205 reports Qwen K32 packed natural 64K NLL improves by $-0.1241$ ($[-0.1479, -0.1029]$) and far source-use improves by $+1.438$ ($[0.720, 2.230]$), yet 30-row natural QA macro difference is $-0.03055$ ($[-0.1168, +0.0486]$, spans zero), and post-hoc rescues fail ($+0.00079$ contrast decoding; $+0.0067$ reranking).
10. `[OBSERVED]` `INDEX.md` line 114 and `paper-2027/HANDOFF.md` lines 67–70 confirm that zero raw JSON/JSONL artifact files exist in the repository for the 2026-09-02 Qwen evaluations due to remote host decommissioning before file transfer.

---

## 2. Logic Chain

1. `[DERIVED]` From Observation 3 (Transplant Rigidity) and Observation 7 (50M Table $\times$ Weights Crossing): Transformer query and key projection matrices $W_q, W_k$ are co-adapted to their training spectrum; post-hoc linear adapters cannot mathematically compensate for a changed frequency table.
2. `[DERIVED]` From Observation 4 (Same-Multiset Permutation Collapse): RoPE frequency allocation cannot be treated as an unordered spectral density $\pi(\omega)$. Checkpoint weights possess rigid pairings between specific 2D rotary coordinate blocks and specific frequency slots.
3. `[DERIVED]` From Observation 6 ($C2$ Retention Failure): The Native retention gate ($0.875$) is a knife-edge threshold that is unpredicted by geometric curve distance in frequency space (MAE $0.001223$ fails gate).
4. `[DERIVED]` From Observation 5 (S=4 Pass vs S=8 Collapse) and R4 Theorem 2 & 5 (Torus Off-Arc Escape): Non-uniform scaling $\omega'_k = \omega_k s^{-m_k}$ ($m_k \not\equiv \text{const}$) breaks collinearity on the phase torus $\mathbb{T}^K$. For $\Delta > L$, the deployed trajectory is completely disjoint from the native training manifold, querying unadapted circuits on un-trained phase configurations.
5. `[DERIVED]` From Observation 8 (Free Gain Sharpening Collapse) and R2 Counterexample 2 (Multi-Layer Gain Argmax Reversal): Scalar attention gain acts as a softmax sharpening tool that alters multi-layer hidden-state mixtures. While it slashes teacher-forced cross-entropy loss by rewarding overconfident short keys, it suppresses autoregressive EOS emission and destroys generation.
6. `[DERIVED]` From Observation 8 (Bimodal Basin Barrier): Native-start optimization suffers from gradient starvation on aliased frequencies, while log-start breaks Native retention.
7. `[DERIVED]` From Observation 9 (Likelihood vs. Margin Gap): Canonical NLL improves smoothly over long context, injecting $O(0.05\text{--}0.1)$ nats per token. However, greedy generation is piecewise-constant and requires positive margins against distractors with $O(1)$ margins; small likelihood shifts fail to prevent greedy decoding drift.
8. `[DERIVED]` Therefore, from Steps 1–7, no single static zero-training RoPE table can simultaneously preserve Native retention and solve natural multi-hop QA on mature checkpoints. The failure is a fundamental structural constraint, not a hyperparameter or search artifact.
9. `[DERIVED]` From Observation 10: All 2026-09-02 Qwen numbers are internal decision evidence only and are branded `UNSUPPORTED BY REPOSITORY EVIDENCE` for manuscript claims.

---

## 3. Caveats

1. `[OBSERVED]` The 2026-09-02 Qwen evaluation numbers lack local raw JSON/JSONL evaluation artifacts due to remote host decommissioning. They are strictly held as internal decision evidence.
2. `[OBSERVED]` HotpotQA Fact D (38 Native-short-correct rows) and Gain Sweep Fact E are user-supplied session facts without dedicated raw result JSON files.
3. `[HYPOTHESIS]` While R2's multi-layer gain counterexample proves algebraically that gain can reverse downstream argmax rankings, the exact empirical prevalence of this reversal across all 32 layers of production LLMs remains unmeasured.
4. `[DERIVED]` "No caveats" does not apply; these empirical data gaps and structural boundaries are load-bearing.

---

## 4. Conclusion

1. **Definitive Architectural Ruling:** The ambition to solve the zero-training RoPE retrofit dilemma via a single static frequency table that simultaneously passes Native in-window retention ($\ge 0.875$) and enables long-context natural multi-hop QA is **structurally obstructed and empirically falsified**.
2. **Established Engineering Frontier:** Non-linear interior exponent allocation $z$ is a legitimate causal training-time variable. Normalized-index static tables provide robust engineering improvements for synthetic retrieval (RULER) and natural long NLL up to $S \le 4$.
3. **Current Policy Confirmation:** Repository policy in `INDEX.md` line 16 (`GPU_METHOD_DEVELOPMENT_STOPPED`) is scientifically sound and must be strictly maintained. No further GPU compute should be authorized for zero-training single-table search.

---

## 5. Verification Method

1. **Verify Clean Git Status:**
   ```bash
   git status --porcelain
   ```
   Must show zero modifications outside `.agents/`.
2. **Verify Key File Citations and Metrics:**
   - `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`: verify slot permutation collapse ($3.1042 \to 6.8649$ NLL, $0.7000 \to 0.0000$ RULER).
   - `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`: verify basin barrier (log-start retention $0.7714$, Native-start HotpotQA F1 $0.02560$).
   - `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`: verify 50M table $\times$ weights interaction ($-3.5367$).
3. **Verify Absence of 2026-09-02 Raw Artifacts:**
   ```bash
   find paper-2027/research/attention-aware-retrofit/evidence/ -name "*20260902*"
   ```
   Returns 0 files, confirming the data gap.
4. **Invalidation Conditions:**
   The findings of this audit would be invalidated if an author demonstrates a single static frequency table that simultaneously passes Native retention ($\ge 0.875$), achieves $\ge 0.25$ LongBench HotpotQA F1 at $4\times$ extrapolation, and cleanly terminates EOS without request-length routing.
