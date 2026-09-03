# Handoff Report — Evidence Archivist (R1)

- **Role:** Evidence Archivist (R1)
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r1_archivist_1`
- **Date:** 2026-09-03T02:43:00Z
- **Full Report Path:** `.agents/teamwork_preview_explorer_r1_archivist_1/analysis.md`
- **Handoff Type:** Hard (Task complete)

---

## 1. Observation

1. [OBSERVED] `INDEX.md` lines 16--33 establishes the cold-start snapshot: `PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`.
2. [OBSERVED] `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` lines 35--60 confirms that at fixed sampled support, moving 30 interior frequencies produces consistent OOD tail NLL effects across 3/3 training seeds, whereas target-matching support reverses the ordering.
3. [OBSERVED] `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` lines 67--99 proves that for any invertible linear maps $A, B$ such that $A^\top R(\omega'\Delta)B = R(\omega\Delta)$, similarity forces trace equality $2\cos(\omega'\Delta) = 2\cos(\omega\Delta)$, establishing that exact linear compensation requires $\{|\omega'_k|\} = \{|\omega_k|\}$.
4. [OBSERVED] `paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md` lines 39--56 documents the falsification of candidate CPU selectors: $D^*$ rank correlation with RULER is $-0.550$, coverage residual correlation is $-0.250$, and `one_turn_floor_s2` scores $0.0000$ on 8K RULER macro despite optimal $D^*=0.0192$.
5. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` lines 121--163 confirms causal efficacy of $z$ in frozen checkpoints (OLMo 16K RULER: geometric 0.0056 vs derived 0.6047; Qwen 64K: geometric 0.5775 vs derived 0.6650), but derived allocation does not separate from its nearest discrete ramp (OLMo diff $-0.0056$ $[-0.0464, +0.0345]$; Qwen diff $+0.0250$ $[-0.0525, +0.1000]$).
6. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` lines 53--72 reports that stateless continuous-boundary-slope operators score $0.0000$ on core-4 RULER at 8K and 16K due to relative phase destruction across $L_{\text{native}}$.
7. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md` lines 27--35 reports that zero-parameter static tables fail 1x tail NLL (anchored EVQ-Cosh regresses $+3.9780$; protected-band Cosh regresses $+0.6922$).
8. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md` lines 42--59 shows direct 62-DOF calibration on 2 documents failed due to held-out row 2 regressing $+0.08229$ 2x NLL (gate $\le +0.05$).
9. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 44--56, 120--143 confirms log law $\omega'_k = \omega_k s^{-m_k}$ eliminates arithmetic drift (RMS drift up to 0.03200 at $s=8$; max 0.24559 at $k=21$), passes 1x PG-19 PPL retention on OLMo (0.875302), and passes 5-task downstream retention (0.915103).
10. [OBSERVED] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 185--192, 271--280 demonstrates that slot permutation of the identical frequency multiset collapses OLMo 1x PG-19 NLL from 3.104234 to 6.864926 ($\Delta = +3.760692$) and drops Qwen 64K core-4 RULER from 0.7000 to 0.0000.
11. [OBSERVED] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 113--118 confirms that continuing the frozen profile to $s=8$ hits a ceiling where single-key-3 collapses to 0.00/0.20/0.25 at 4K/8K/16K.
12. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md` lines 58--78, 141--145 shows two-parameter C2 clipped-affine law achieves movement MAE 0.001223, but fails registered Native retention on OLMo (PPL retention 0.870971 $< 0.875$) and Qwen (32K macro retention 0.868902 $< 0.875$).
13. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md` lines 24--47 shows that correcting operating reference length from config 8192 to empirical 4096 recovers Gemma K128 16K RULER from 0.0000 to 0.7250 (physical) and 0.7950 (index).
14. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md` lines 8--13, 50--66 shows normalized index beats physical-x at 16K RULER (.790000 vs .728125, $\Delta = +0.061875$ $[0.028109, 0.096250]$).
15. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md` lines 8--18 shows normalized index retains higher Native capability at K32 (.923243 vs .859459) and beats YaRN-s2 at 64K ($\Delta = +0.064375$ $[0.027500, 0.102516]$).
16. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md` lines 49--68 confirms normalized index beats YaRN-s2 on full RULER-13 at 64K ($\Delta = +0.060897$ $[0.027627, 0.095835]$), but YaRN remains superior on variable tracking (.360 vs .280) and QA rows (.200 vs .150).
17. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` lines 147--205 reports that on Qwen K32, 64K packed-natural NLL improves by $-0.1241$ ($[-0.1479, -0.1029]$, 32/32 streams) and far source-use improves by $+1.438$ ($[0.720, 2.230]$), but 30-row natural QA macro difference is $-0.03055$ ($[-0.1168, +0.0486]$, spans zero), and readout rescues fail (contrast decoding $+0.00079$, reranking $+0.0067$). Remote raw JSON/JSONL artifacts are not recovered.
18. [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` lines 18--33, 128--178 shows that learned headwise clocks improve HotpotQA F1 from 0.21169 to 0.25223, while free gain is a shortcut (loss drops $3.56 \to 2.31$, but Hotpot F1 drops $.242 \to .194$ and EOS termination collapses $178 \to 108$). The basin barrier is confirmed: log-start fails Native retention (0.77138), while Native-start achieves 1.0462 retention but fails HotpotQA (F1 0.02560).
19. [OBSERVED] `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` lines 20--56 establishes the provenance split (§0.1) and records that the 18-sample 64-D behavioral gradient route failed holdout (Fact F). It proves theorems T1--T9.

---

## 2. Logic Chain

1. [DERIVED] From Observation 3 (Transplant Rigidity Theorem), linear Q/K adapters cannot compensate for frequency changes under frozen weights; the bilinear form is permanently altered.
2. [DERIVED] From Observation 10 (Slot Permutation Collapse), RoPE frequencies cannot be treated as an unordered continuous spectral density $\pi(\omega)$. Mature checkpoints have learned rigid, ordered couplings between specific head/subspace projections and individual frequency slots.
3. [DERIVED] From Observations 4, 7, 8, and 12, all CPU-only static scalar selectors, zero-parameter single tables, underdetermined calibrations, and continuous metric fits fail operational gates because they ignore the checkpoint's internal functional distance and table-weight co-adaptation.
4. [DERIVED] From Observations 5, 9, 10, 13, 14, 15, and 16, single static log-frequency tables ($\omega'_k = \omega_k s^{-m_k}$) using normalized index transport provide robust, reproducible gains in long NLL, synthetic RULER, and source likelihood, while maintaining 1x retention on OLMo and K32.
5. [DERIVED] From Observations 11, 17, 18, and 19, this success hits a fundamental barrier:
   - At $s=8$, novelty volume super-doubles ($2 + 1/(4^{1-m_k}-1)$), causing retrieval failure.
   - On natural QA, far-source evidence successfully enters canonical logits ($\Delta = +1.438$), but greedy generation is piecewise constant and requires the correct token to win argmax margins along the self-induced decoding path.
   - Free attention gain cannot fix ranking errors; it merely sharpens logits, causing generation degradation and EOS termination collapse.
   - The optimization landscape is bimodal: log-start models achieve long QA but sacrifice Native retention, while Native-start models preserve retention but cannot bootstrap long reasoning.
6. [DERIVED] Therefore, no single static frequency table under frozen weights has been demonstrated to jointly solve Native retention and natural generation QA across multiple horizons.

---

## 3. Caveats

1. [OBSERVED] The 2026-09-02 Qwen natural NLL, QA, and table $\times$ gain numbers (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`) lack recovered remote raw JSON/JSONL evaluation logs. While parent commit hashes are bound, these numbers remain internal decision evidence only and cannot be cited in reviewer-facing manuscript text without artifact recovery.
2. [OBSERVED] Facts D (HotpotQA 38 Native-short-correct samples) and E (Gain sweep) in `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` are governing session facts that lack dedicated raw receipt files.
3. [DERIVED] No investigation of non-frozen weight retraining, full parameter fine-tuning, or KV cache compression architectures was conducted, as these lie strictly outside the zero-training mature retrofit charter.
4. [DERIVED] "No caveats" does not apply; the above empirical boundaries are load-bearing.

---

## 4. Conclusion

1. **Defensible Core Finding:** At fixed sampled support, interior exponent allocation $z$ is causally efficacious in both training and frozen mature checkpoints. The ordered rotary subspace-frequency coupling is the governing structural object. Normalized-index log-frequency scaling ($\omega'_k = \omega_k s^{-m_k}$) is the best-supported tested engineering rule for long NLL and synthetic RULER extension up to $s=4$.
2. **Defensible Negative Verdict:** Single-table zero-training retrofit does NOT solve the joint Native retention and natural generation QA objective. The likelihood-to-winner barrier, task-dependent radius, and bimodal basin barrier prevent a single static table from simultaneously guaranteeing in-window integrity and multi-hop autoregressive reasoning.
3. **Scientific Posture:** Current repository evidence does NOT justify claiming SOTA, continuous basin optimality, universal transport laws, or a solved zero-training retrofit method. Method-development GPU compute is correctly stopped.

---

## 5. Verification Method

1. **File Integrity Verification:**
   - Verify `git status --porcelain` remains completely clean across the repository root (0 modifications to source, config, or papers).
   - Inspect `.agents/teamwork_preview_explorer_r1_archivist_1/analysis.md` for full detailed tables and derivation chains.
2. **Receipt & Audit Reproduction:**
   - Inspect `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` to verify the slot permutation collapse numbers (OLMo 1x PG-19 NLL 6.864926, Qwen 64K RULER 0.0000).
   - Inspect `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` to verify the basin barrier numbers (Log-start retention 0.77138, Native-start Hotpot F1 0.02560).
   - Run CPU unit test suite if authorized: `python -m pytest tests/` on a work machine environment.
3. **Invalidation Conditions:**
   - Any discovery of a single static table that simultaneously achieves $\ge 0.875$ Native PG-19 retention, $\ge 0.25$ LongBench HotpotQA/2Wiki F1, and successful EOS stopping without request-length routing.
