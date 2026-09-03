# Zero-Training RoPE Retrofit: Complete Evidence Archive and Causal Audit

- **Role:** Evidence Archivist (R1)
- **Working Directory:** `.agents/teamwork_preview_explorer_r1_archivist_1`
- **Scope:** Mature checkpoint zero-training RoPE retrofit across all repository indices, experimental results, preflights, analysis, foundations, and rebuttal records.
- **Strict Formatting Contract:** Every substantive assertion is explicitly tagged with `[OBSERVED]`, `[DERIVED]`, `[HYPOTHESIS]`, or `[UNKNOWN]`. Every `[OBSERVED]` assertion explicitly cites the exact repository-relative file path, dates/commit tags, and specific numbers.
- **Repository Lifecycle State:** `[OBSERVED]` `INDEX.md` line 16 defines the governing state: `PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`.

---

## 1. Executive Summary & Audit Scope

[OBSERVED] `INDEX.md` lines 19--33 and `paper-2027/research/history/TIMELINE.md` lines 158--172 establish the boundaries of completed research:
1. [OBSERVED] At fixed sampled support $(a, R)$, interior exponent allocation $z$ is a causal training-time variable (`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`).
2. [OBSERVED] Static full-RoPE geometry identifies phase-invariant redundancy/effective dimension, but does not predict LM quality or extrapolation ranking (`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`).
3. [OBSERVED] Mature frozen checkpoints depend critically on the ordered pairing between learned rotary subspaces and frequency dilations; permuting slot assignments while preserving the exact frequency multiset causes catastrophic collapse (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
4. [OBSERVED] Static pure-$z$ interventions reliably improve long NLL, synthetic RULER/NIAH tasks, and far-source answer likelihood, but fail to convert reliably into natural autoregressive QA generation (`paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).
5. [OBSERVED] No tested arm jointly passes strict Native in-window retention ($\ge 0.875$) and long-context generation QA (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`).
6. [OBSERVED] Normalized pair index is the best-supported tested cross-$K$ transport coordinate, but is an empirical engineering rule rather than a canonical physical law (`paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md`, `paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md`).

[DERIVED] The core technical barrier in mature checkpoint zero-training retrofit is that changing the RoPE frequency table under frozen weights queries the model on novel joint phase configurations outside the Native training manifold ($\Delta > L_{\text{train}}$). While linear loss (NLL) and local attention matching improve smoothly, autoregressive generation requires the correct token to maintain an argmax margin at every visited step. Because the model's tolerance to off-arc phase configurations is task-dependent and inaccessible from Native-window data alone, all scalar-functional selectors and single global tables either sacrifice Native performance or fail long-context reasoning.

---

## 2. Reconstructed Chronological Timeline & Causal Chain

### Phase 0: Precursors, LoRA Adaptation & Transplant Obstruction (July 2026)

- **2026-07-24 — Collision-Risk Tau Selector Fails Historical Validation:**
  - [OBSERVED] `rebuttal/rebuttal_0723/theory_results/TRAINING_FREE_TAU_SELECTOR_20260724.md` lines 159--160: A training-free $\tau^\star$ selector based on distinct-offset squared Gram collision $R(\tau) = \max\{\mathcal C(\Pi_{\text{train}}, \Pi_{\text{train}}), \mathcal C(\Pi_{\text{target}}, \Pi_{\text{target}}), \mathcal C(\Pi_{\text{train}}, \Pi_{\text{target}})\}$ selected $\tau^\star \in [13.131, 13.344]$ for all 9 configurations across $K \in \{16, 32, 64\}$ and $L \in \{256, 512, 1024\}$, collapsing to the edge of the historical search grid and failing prospective adoption.
- **2026-07-26 — Exact Transplant Obstruction Theorem:**
  - [OBSERVED] `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` lines 67--99: Proved that for any fixed, content-independent, invertible linear $Q/K$ reparameterizations $A, B$ such that $A^\top R(\omega'\Delta)B = R(\omega\Delta)$ for all $\Delta$, $\Delta=0$ forces $B = A^{-\top}$, requiring matrix similarity $A^\top R(\omega'\Delta)A^{-\top} = R(\omega\Delta)$. Trace preservation forces $\cos(\omega'\Delta) = \cos(\omega\Delta)$, proving exact position-independent linear compensation requires multiset equality up to sign: $\{|\omega'_k|\} = \{|\omega_k|\}$.
  - [DERIVED] Replacing RoPE frequencies post-hoc under frozen weights cannot be undone by linear adapters; any non-trivial frequency replacement permanently alters the relative-position bilinear form.
- **2026-07-26 to 2026-07-31 — EVQ-LoRA Task Adaptation vs Held-Out Collapse:**
  - [OBSERVED] `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` lines 16--21: Under matched 4K Q/K/V/O LoRA, EVQ reached 69/100 and 67/100 on trained 8K RULER `niah_single_1`, while Native reached 0/100. However, on held-out 4K tasks, untouched Native scored 55% UUID retrieval and 25% variable tracking, whereas EVQ scored 0% on both immediately after injection, after 20M natural tokens, and after routing.
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` lines 34--36: Seven 4K-only LoRA schemes without RULER/NIAH supervision all failed 8K RULER (best screen 0.0750); the strongest one-token natural-retrieval arm achieved first-token top-1 of only 0.0391 (mean answer NLL 3.7357). Sparse Native-teacher KL also failed (`rebuttal/rebuttal_0723/theory_results/OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md`).

### Phase 1: Early Static Selectors Falsified & Budgeted Retrofit (2026-08-21 to 2026-08-22)

- **2026-08-21 — Attention-Distance Direct Mapping Fails:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md` lines 12--25: Mapping empirical attention distances directly to RoPE frequencies without accounting for phase interference failed to produce viable extrapolation.
- **2026-08-22 — Decisive Falsification of Three CPU Candidate Axes:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md` lines 39--56: Evaluated candidate CPU metrics on OLMo-2-1B against RULER macro:
    - $D^*$ (unrepairable in-window logit energy): Spearman rank correlation with RULER was $-0.550$ (worse in-window damage correlated with *better* RULER!).
    - Coverage residual: Spearman rank correlation was $-0.250$.
    - Phase risk (mean unseen phase per channel): Spearman rank correlation was $+0.000$.
    - Decisive counterexample: `one_turn_floor_s2`, designed to satisfy the hypothesis that "channels wrapping once in training are safe", had near-optimal $D^*=0.0192$ and lowest coverage residual (0.2410), yet scored **0.0000** on RULER macro at 8K (identical to Native).
- **2026-08-22 — Emergence of Length-Conditioned Budgeted Retrofit:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md` lines 22--24, 52--59: Defined uniqueness-budgeted movement $m_k = (1 - \text{normalized\_uniqueness}_k)^2$, arithmetic table $\omega'_k = \omega_k(1-m_k) + (\omega_k/s)m_k$, and length-dependent amplitude $a(s) = 1 + 0.1\log(s)$.
  - [OBSERVED] `LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md` lines 42--44: Exponent $p=2$ in $m_k = (1 - u_k)^p$ was chosen over $p=1$ because at 8K RULER it scored 0.5825 vs 0.5525 ($p=1$), while at 16K they scored 0.4000 ($p=2$) vs 0.4025 ($p=1$).
  - [OBSERVED] `LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md` lines 52--59: Budgeted table + matched amplitude reached 0.5825 at 8K and 0.4000 at 16K on core-4 RULER (Native scored 0.0000 at both; amplitude-only scored 0.0000 at both; frequency-only scored 0.4000 at 8K and 0.1150 at 16K).

### Phase 2: Same-Support Causal Isolation & Binary Session Routing (2026-08-23)

- **2026-08-23 — Fixed-Support Exponent Causal Identification in Frozen Models:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` lines 121--163:
    - OLMo at 16K RULER (unseen 9 tasks): same-support geometric control scored 0.0056, whereas derived allocation scored 0.6047 ($\Delta = +0.5992$ $[+0.5488, +0.6480]$); nearest movement-profile ramp scored 0.6104 ($\Delta = -0.0056$ $[-0.0464, +0.0345]$ vs derived, unresolved); official YaRN scored 0.0794.
    - Qwen-1.5B at 64K core-4 RULER: same-support geometric scored 0.5775, derived allocation scored 0.6650 ($\Delta = +0.0875$ $[+0.0025, +0.1750]$); nearest ramp scored 0.6400 ($\Delta = +0.0250$ $[-0.0525, +0.1000]$ vs derived, unresolved); Native scored 0.5450; official YaRN scored 0.6025.
  - [DERIVED] Changing interior exponent allocation $z$ at fixed support $(a, R)$ causally impacts frozen mature checkpoints, but the exact uniqueness profile does not statistically separate from its nearest discrete ramp.
- **2026-08-23 — Binary Session Routing Adopted, Alternatives Rejected:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` lines 27--31, 39--51:
    - Session-binary rule: use Native if $\text{tokens} \le L_{\text{native}}$, else use frozen $s4$ table. Scored 0.7175/0.4075 on core-4 RULER at 8K/16K (vs YaRN-4 0.2225/0.0125) and 0.2666 token F1 on 200-row 16K 2Wiki (vs YaRN-4 0.2569).
    - Stateless boundary-slope operator failed: scored 0.0000 on core-4 RULER at both 8K and 16K due to relative phase destruction across $L_{\text{native}}$.
    - Smallest-covering-profile routing (Native/s2/s4) rejected: scored only 0.2473 on 2Wiki 16K (vs 0.2774 for fixed $s4$).

### Phase 3: Single-Table Obstruction, Oracle Co-adaptation & Dose Response (2026-08-24 to 2026-08-26)

- **2026-08-24 — Failure of Zero-Parameter Single Static Tables:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md` lines 27--35: Two zero-parameter Native-support analytic tables failed the 1x retention gate:
    - Anchored EVQ-Cosh ($\tau=2$): 1x tail NLL regressed $+3.9780$ (6.9492 vs Native 2.9712), while 2x NLL improved $-0.2533$ (6.8496 vs Native 7.1029).
    - Protected-band Cosh: 1x tail NLL regressed $+0.6922$ (3.6633 vs Native 2.9712), while 2x NLL improved $-0.1845$ (6.9184 vs Native 7.1029).
- **2026-08-24 — Underdetermined 62-DOF Direct-$z$ Calibration Fails:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md` lines 42--59: Directly optimizing 62 simplex DOFs on two 8K FineWeb documents failed the held-out gate because held-out row 2 regressed $+0.08229$ NLL at 2x (threshold $\le +0.05$).
- **2026-08-25 — Co-adaptive Allocation Oracle Identifies Full vs Tail Trade-off:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md` lines 21--35, 77--107: Jointly training interior allocation and rank-64 Q/K LoRA improved long-tail NLL ($-0.03866$ at 8K, $-0.08767$ at 16K) but degraded full-sequence NLL ($+0.03844$ at 8K, $+0.02190$ at 16K) and produced zero downstream gain on 200-row 2Wiki (token F1 $-0.00654$ at 4K, $+0.00129$ at 8K, $+0.00089$ at 16K).
- **2026-08-26 — Allocation Dose Response Confirms Continuous Redistribution:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md` lines 14--29, 50--67:
    - Analytic Cosh Path A failed the joint gate at all points: smallest dose $\lambda=0.02$ improved 16K tail NLL by $-0.1153$ but degraded 4K full NLL by $+0.0169$ (exceeding $+0.01$ guard). Best 16K tail point $\lambda=0.35$ had 16K tail $-0.3185$, but 4K full $+3.2769$.
    - Learned direction Path B scaled by $1, 4, 16$ improved 16K tail ($-0.0450, -0.1174, -0.1572$) with small 4K costs ($+0.0008, +0.0061, +0.5294$), but worsened long full-sequence NLL.

### Phase 4: Scale-Consistent Log Law & Decisive Slot Permutation Collapse (2026-08-27 to 2026-08-31)

- **2026-08-31 — Log-Frequency Law Replaces Arithmetic Law:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 21--56:
    - Replaced arithmetic law $\omega'_k = \omega_k((1-m_k) + m_k/s)$ with log law $\omega'_k = \omega_k s^{-m_k}$.
    - Arithmetic law suffered severe exponent drift $q_k(s) = -\log(\omega'_k/\omega_k)/\log s$, with RMS error $0.01099/0.02195/0.03200$ at $s=2/4/8$ (max error $0.24559$ at $k=21$, where $m=0.67092$ but $q$ fell to $0.42532$).
    - Log law satisfies composition $\omega(s_1 s_2) = \omega(s_1) s_2^{-m}$.
- **2026-08-31 — 1x PG-19 Retention Gate Passed on OLMo:**
  - [OBSERVED] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 120--143:
    - Selected gain $c=0.074$ ($a(s) = 1 + 0.074\log s$) on 1x PG-19 boundary only.
    - Passed 1x double gate: PG-19 PPL retention was **0.875302** (threshold $\ge 0.875$), five-task downstream macro retention was **0.915103** (threshold $\ge 0.875$).
    - Matched arithmetic table failed 1x retention: PPL retention was $0.869584 < 0.875$.
    - Natural long endpoints: 2x PG-19 NLL was 3.083278 (vs Native 7.100855, YaRN-4 3.441096); 4x PG-19 NLL was 3.081946 (vs Native 7.205538, YaRN-4 3.793878).
    - Full RULER-13: 4K/8K/16K was $0.71397 / 0.66705 / 0.49859$ (vs Native $0.71308 / 0 / 0.00385$, YaRN-4 $0.43141 / 0.24308 / 0.10564$).
- **2026-08-31 — S=8 Early Stop (Ceiling Reached):**
  - [OBSERVED] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 113--118: At $s=8$, single-key retrieval collapsed across subtasks (single-key-1: 0.95/0.95/1.00; single-key-2: 0.50/0.55/0.65; single-key-3: **0.00/0.20/0.25** at 4K/8K/16K). Registered early stop triggered; 32K RULER-13 was not opened.
- **2026-08-31 — Decisive Discovery: Multiset Slot Permutation Collapses:**
  - [OBSERVED] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` lines 185--192, 271--280:
    - On OLMo: Permuting the interior slot assignment while preserving the exact frequency multiset, endpoints, and gain caused 1x PG-19 NLL to jump from 3.104234 to **6.864926** ($\Delta\text{NLL} = +3.760692$).
    - On Qwen at 64K: Permuting interior slot assignment of the exact frequency multiset caused core-4 RULER macro to drop from 0.7000 to **0.0000**!
  - [DERIVED] Rotary frequency allocation cannot be modeled as an unordered spectral density $\pi(\omega)$; mature checkpoints have frozen ordered pairings between rotary subspaces and specific frequency bands.

### Phase 5: Low-Dim Coupling, Reference Lengths & Cross-K Confirmation (2026-09-01)

- **2026-09-01 — Two-Parameter Coupling Law (C2): Geometry vs Function:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/CPU_LOW_DIM_COUPLING_LAW_20260901.md` and `paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`:
    - Fitted clipped-affine $G_4(x) = \text{clip}((x_H - x)/(x_H - x_L), 0, 1)$ on $x = \ln(u)$ with boundaries $x_H \approx 0.7383, x_L \approx 0.3664$. Achieved OLMo movement MAE 0.001223, RMSE 0.006118.
    - On GPU: Preserved Qwen long behavior (64K core-4: 0.6775 vs 0.6725; 128K core-4: 0.5725 vs 0.5450).
    - Failed strict Native retention gates: OLMo 1x PG-19 PPL retention was $0.870971 < 0.875$ (missed by 0.0046 NLL); Qwen 32K macro retention was $0.868902 < 0.875$.
  - [DERIVED] Sub-percent movement error in geometric space does not guarantee functional equivalence in checkpoint logit space.
- **2026-09-01 — Gemma K128 Reference-Length Calibration:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md` lines 24--47:
    - Resolved historical Gemma negative: Config claimed $L_{\text{config}}=8192$, but empirical operating reference was $L_{\text{ref}}=4096$.
    - With corrected reference, static $s4$ table recovered 8K/16K RULER: physical-x scored 0.8775/0.7975/0.7250, normalized-index scored 0.9450/0.8650/0.7950, both passing 4K retention gates. Old config-reference table remained 0.0000 at 16K.
- **2026-09-01 — Coordinate Privilege Rejected at K128 and K32:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md` lines 8--13, 50--66: On independent N80 RULER at 16K, normalized-index scored **0.790000** vs physical-x **0.728125** ($\Delta = +0.061875$ $[0.028109, 0.096250]$). Physical coordinate privilege was decisively rejected.
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md` lines 8--18, 51--59: On fresh N80 at K32, 64K macro difference was unresolved ($\Delta = +0.0050$ $[-0.038125, +0.048750]$), but normalized-index was more Native-compatible at 32K (retention 0.923243 vs 0.859459). Index beat matched YaRN-s2 at 64K ($\Delta = +0.064375$ $[0.027500, 0.102516]$).
- **2026-09-01 — K32 Normalized-Index Full RULER-13 Confirmation:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md` lines 49--68, 77--96:
    - 32K macro: Native 0.547821, Index 0.559167 (retention 1.020711), YaRN-s2 0.559423.
    - 64K macro: Native 0.220513, Index 0.514551, YaRN-s2 0.453654 ($\Delta_{\text{Index}-\text{YaRN}} = +0.060897$ $[0.027627, 0.095835]$).
    - Task trade-offs: YaRN remained superior on variable tracking (.360 vs .280), SQuAD QA (.200 vs .150), and HotpotQA (.200 vs .150).

### Phase 6: Natural QA Conversion Barrier, Headwise Clocks & Theoretical Closure (2026-09-02)

- **2026-09-02 — Packed-Natural NLL vs Far-Evidence Natural QA:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` lines 147--174:
    - Packed-natural NLL on Qwen K32 (32 paired streams): 32K NLL Native 2.597666 vs Index 2.615399 (retention ~0.9824); 64K NLL Native 2.754945 vs Index 2.630842 ($\Delta = -0.1241$ $[-0.1479, -0.1029]$, all 32 streams improved).
    - Far-evidence natural QA (30-row 2Wiki/Qasper/Hotpot): Native 0.13229, Index 0.10174, YaRN 0.11197 ($\Delta_{\text{Index}-\text{Native}} = -0.03055$ $[-0.1168, +0.0486]$, unresolved). Positive QA gate failed.
  - [OBSERVED] `ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` lines 175--195:
    - Table $\times$ gain 2x2 factorial: Index table + unit gain preserved almost all long NLL gain; gain was not the cause of QA stagnation.
    - Evidence bridge: Native far source-use was $+0.066$ (spanned zero); Index far source-use was $+1.504$ (wholly positive; $\Delta = +1.438$ $[0.720, 2.230]$). Canonical answer NLL under far source improved from 5.123 (Native) to 3.695 (Index).
    - Readout rescues failed: source-contrast greedy decoding gave $+0.00079$ $[-0.0271, 0.0292]$; sequence reranking gave $+0.0067$ $[-0.0173, 0.0330]$ (oracle macro was 0.18656).
- **2026-09-02 — Headwise Factorized Clocks & The Basin Barrier:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` lines 18--33, 128--178:
    - 256 headwise scalar clocks on OLMo-2-1B: HotpotQA-200 F1 improved from 0.21169 (frozen $s4$) to 0.24237; two-axis arm (512 parameters) reached 0.25223 (vs YaRN-4 0.27644, diff $-0.02420$ $[-0.07896, 0.03064]$).
    - Free headwise gain was a teacher-forced shortcut: dropped training loss from 3.5612 to 2.3105, but HotpotQA F1 dropped from 0.24237 to 0.19439 and EOS terminations collapsed from 178/200 to 108/200.
    - The Basin Barrier:
      - Log-start arm: HotpotQA F1 0.25223, but 4K PG-19 PPL retention was **0.77138** (failed $\ge 0.875$ gate).
      - Native-start arm: 4K PG-19 PPL retention was **1.0462** (better than Native!), but HotpotQA F1 collapsed to **0.02560** (only 8/200 EOS terminations).
- **2026-09-02 — First-Principles Theoretical Closure:**
  - [OBSERVED] `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` lines 20--56, 172--386:
    - Retired route: 18-sample 64-D behavioral-gradient candidate failed unopened holdout with inconsistent 8K/16K behavior (Fact F).
    - Provenance split (§0.1): Object-level construction of $m_k$ is Native-only, but meta-level family selection ($p=2$, log vs arithmetic) was selected against long outcomes.
    - Rigorous negative theorems: T1 (PI curve uniqueness), T2 (Transplant rigidity), T3 (Compatibility modulus & weight-blind metric vacuity), T4 (Conditioning theorem with factor $S$), T5 (Non-identifiability & off-arc phase queries), T6 (Semigroup vacuity / gauge), T7 (Super-doubling of novelty volume), T8 (Gain argmax invariance & sharpening shortcut), T9 (Three layers: Representation, Compatibility, Circuit Robustness, and task-dependent radius $R_T$).

---

## 3. Verified Facts vs Unsupported Claims

### 3.1 Verified Positive Findings (Empirically Backed)

1. [OBSERVED] **Causal Efficacy of Interior Exponent Allocation:**
   - At fixed sampled support $(a, R)$, moving only interior $z$ alters trained model behaviour across all 3 seeds with consistent OOD direction (`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`).
   - In frozen checkpoints, same-support $z$ modification improves 16K RULER on OLMo from 0.0056 (geometric) to 0.6047 (derived), and 64K RULER on Qwen from 0.5775 to 0.6650 (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`).
2. [OBSERVED] **Superiority of Log-Frequency Law Over Arithmetic Law:**
   - Arithmetic interpolation has severe scale-dependent exponent drift (RMS error up to 0.03200 at $s=8$; max error 0.24559 at pair 21). Log law $\omega'_k = \omega_k s^{-m_k}$ eliminates exponent drift and satisfies composition $\omega(s_1 s_2) = \omega(s_1) s_2^{-m}$ (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
   - Log law passes 1x PG-19 PPL retention on OLMo (0.875302), whereas arithmetic fails (0.869584) (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
3. [OBSERVED] **Subspace-Frequency Ordered Coupling (Non-Exchangeability):**
   - Permuting slot assignments while preserving the exact frequency multiset causes complete collapse: OLMo 1x PG-19 NLL jumps from 3.104234 to 6.864926 ($\Delta = +3.760692$), and Qwen 64K core-4 RULER drops from 0.7000 to 0.0000 (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
4. [OBSERVED] **Normalized Index as Preferred Cross-$K$ Coordinate:**
   - On independent N80 RULER at 16K, normalized-index beats physical-x by $+0.061875$ ($[0.028109, 0.096250]$) (`K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md`).
   - On K32, normalized-index achieves higher Native retention than physical-x (.923243 vs .859459) and outperforms YaRN at 64K by $+0.064375$ (`K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md`).
   - On K32 full RULER-13, normalized-index beats YaRN at 64K by $+0.060897$ ($[0.027627, 0.095835]$) (`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md`).
5. [OBSERVED] **Natural Long-Context NLL Improvement:**
   - On Qwen K32 packed natural text, normalized-index reduces 64K NLL by $-0.1241$ ($[-0.1479, -0.1029]$), improving 32 out of 32 paired streams (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).
6. [OBSERVED] **Far-Evidence Injection into Logits:**
   - Normalized-index increases far-source utilization from $+0.066$ (Native, spanning zero) to $+1.504$ (Index, positive; $\Delta = +1.438$ $[0.720, 2.230]$) and reduces canonical answer NLL from 5.123 to 3.695 (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).

### 3.2 Verified Negative Findings (Empirically Falsified Routes)

1. [OBSERVED] **Static Scalar Selectors of Shared Tables Falsified:**
   - $D^*$ ranking sign is inverted (Spearman $-0.550$).
   - Coverage residual sign is inverted (Spearman $-0.250$).
   - Phase-risk / "wrapped channels are safe" fails decisively: `one_turn_floor_s2` scores 0.0000 on RULER (`RETROFIT_AXIS_FALSIFICATION_20260822.md`).
2. [OBSERVED] **Stateless Absolute-Boundary Slope Operators Fail:**
   - Preserving absolute slope at $L_{\text{native}}$ scores 0.0000 at 8K and 16K on core-4 RULER due to relative phase destruction (`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`, `ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`).
3. [OBSERVED] **Zero-Parameter Single Static Tables Fail Native Retention:**
   - Anchored EVQ-Cosh regresses 1x tail NLL by $+3.9780$; protected-band Cosh regresses by $+0.6922$ (`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`).
4. [OBSERVED] **Underdetermined Direct-$z$ Calibration Fails:**
   - Optimizing 62 DOFs on 2 documents overfits to document heterogeneity, failing held-out 2x NLL gate (`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`).
5. [OBSERVED] **18-Sample 64-D Behavioral Gradient Route Failed:**
   - Direction failed unopened holdout with inconsistent 8K/16K behavior and exited main line (`FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` Fact F).
6. [OBSERVED] **Readout Post-Processing Rescues Fail:**
   - Source-contrast greedy decoding ($+0.00079$) and 3-candidate sequence reranking ($+0.0067$) fail to convert far-source likelihood into greedy generation accuracy (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).
7. [OBSERVED] **Free Headwise Gain is a Teacher-Forced Shortcut:**
   - Dropped training loss from 3.5612 to 2.3105, but collapsed HotpotQA F1 from 0.24237 to 0.19439 and EOS terminations from 178/200 to 108/200 (`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`).
8. [OBSERVED] **Frozen $s=8$ Continuation Ceiling:**
   - At $s=8$, single-key-3 retrieval collapsed to 0.00/0.20/0.25 at 4K/8K/16K, triggering registered early stop (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).

### 3.3 Verified Null / Unresolved Findings

1. [OBSERVED] **Profile Detail vs Nearest Ramp:**
   - Derived uniqueness allocation and its nearest movement-profile ramp are statistically unresolved on OLMo (diff $-0.0056$ $[-0.0464, +0.0345]$) and Qwen (diff $+0.0250$ $[-0.0525, +0.1000]$) (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`).
2. [OBSERVED] **Natural Generation QA Conversion:**
   - 30-row far-evidence QA panel macro difference between normalized-index and Native is $-0.03055$ ($[-0.1168, +0.0486]$); interval spans zero. Neither positive improvement nor statistically significant degradation is established (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).
3. [OBSERVED] **Physical vs Index at K32:**
   - 64K macro difference is $+0.0050$ ($[-0.038125, +0.048750]$), completely unresolved (`K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md`).
4. [OBSERVED] **Phase Isotropy / Pair-Volume Metrics:**
   - Marked `SCREEN_UNRESOLVED` in `INDEX.md` §3.2 (`PHASE_ISOTROPY_50M_M4_RESULT_20260824.md`, `PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md`).

### 3.4 Unsupported Claims & Methodological Pitfalls

1. **UNSUPPORTED BY REPOSITORY EVIDENCE:** *Claim that 2026-09-02 Qwen packed-natural NLL, QA, and table $\times$ gain numbers are external manuscript evidence.*
   - [OBSERVED] `HANDOFF.md` §5 and `ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` §2, §8: The raw JSON/JSONL evaluation artifacts on the remote compute host were not retrieved before shutdown. While parent commit SHA-256 hashes are verified, numbers remain internal decision evidence only and cannot be promoted to reviewer-facing text.
2. **UNSUPPORTED BY REPOSITORY EVIDENCE:** *Claim that HotpotQA Fact D and Gain Sweep Fact E have raw repo owners.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §I: Facts D and E are user-supplied governing session facts; they lack recovered raw result JSON files.
3. **FALSIFIED / METHODOLOGICAL DEFECT:** *Claim that movement-space RMSE certifies functional checkpoint equivalence.*
   - [OBSERVED] Two-parameter $C2$ had movement MAE 0.001223, but failed the registered Native retention gate on OLMo ($0.870971 < 0.875$) and Qwen ($0.868902 < 0.875$) (`LOW_DIM_COUPLING_GPU_RESULT_20260901.md`).
4. **FALSIFIED / METHODOLOGICAL DEFECT:** *Claim that continuous scale-flow ODE or semigroup property provides a normative design rule or optimality certificate.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (T6): Semigroup property $F(S_1 S_2) = F(S_2) \circ F(S_1)$ forces power laws $S^{-m_k}$ as a trivial bookkeeping identity of the log parameterization. Static installations depend only on endpoint tensors, rendering flow trajectories unobservable gauge. Furthermore, exact composition holds for $s=4 \to s=8$ while behavioral performance collapses (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
5. **FALSIFIED / METHODOLOGICAL DEFECT:** *Claim that gain optimum follows a smooth convex trade-off $L(g) = e^{-g\mu} + 2\alpha\epsilon g$.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (T8): Falsified by sweep derivative non-monotonicity (convexity violated between $g=1.0$ and $1.05$). The $16\times$ collapse below $g=1$ reflects a discrete threshold event, not smooth leakage.
6. **RETRACTED / MATHEMATICAL ERROR:** *Novelty volume multiplier $2(1+2^{m-1}) \in (2,3]$.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (T7, G.5): Retracted on three grounds: formula depended on $S$, fell in $(3,4)$, and $N_k$ is strictly concave in $m$. Correct super-doubling factor is $2 + 1/(4^{1-m_k}-1) \in (2, \infty)$.
7. **RETRACTED / MATHEMATICAL ERROR:** *Mean power direction under slot merging.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (A3, G.2): Merging two slots onto the same frequency shifts mean power by $\text{Re}(c_1 \bar c_2) \in [-|c_1||c_2|, +|c_1||c_2|]$, which can increase or decrease power depending on phase alignment; earlier claim that it "can only drop" was false.
8. **CONFIRMED ARTIFACT:** *Old Qwen 128K score of 0.6175.*
   - [OBSERVED] `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` §2: Created by stride-16 sampling aliasing; valid corrected score is **0.5400**.
9. **META-LEVEL CONTAMINATION:** *Claim that log_s4 was derived strictly from Native geometry without long-task outcomes.*
   - [OBSERVED] `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.1: Object-level values ($m_k$, $G_4$, $c=0.074$) were Native-derived or 1x-selected. However, the design family itself (exponent $p=2$, log law over arithmetic) was chosen by inspecting long-range RULER outcomes (8K RULER 0.5825 vs 0.5525; 2x/4x/8x core-4).

---

## 4. Comprehensive Claim -> Evidence -> Evidence Strength Table

| # | Claim | Evidence Owner (File Path) | Commit / Date Tag | Specific Numbers / Metrics | Evidence Strength & Status |
|---|---|---|---|---|---|
| 1 | Interior exponent $z$ is a causal training variable at fixed sampled support | `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` | 2026-08-20 | 151.9M 3-seed replication; OOD tail NLL consistent across 3/3 seeds; reversing support reverses order | **Strong (Causal Gold Standard)**; raw JSON receipt hash-verified |
| 2 | Post-hoc linear Q/K compensation is obstructed for unequal multisets | `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | 2026-07-26 | $A^\top R(\omega'\Delta)A^{-\top} = R(\omega\Delta) \implies \{|\omega'_k|\} = \{|\omega_k|\}$ | **Strong (Exact Algebraic Theorem)**; verified |
| 3 | Static CPU selectors ($D^*$, coverage, phase risk) do not predict RULER | `paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md` | 2026-08-22 | Spearman: $D^* = -0.550$, cov $= -0.250$, risk $= 0.000$; `one_turn_floor_s2` RULER $= 0.0000$ | **Strong (Falsified)**; registered counterexample |
| 4 | Non-geometric $z$ allocation improves frozen mature checkpoints | `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` | 2026-08-23 | OLMo 16K RULER: geometric 0.0056 vs derived 0.6047; Qwen 64K: geometric 0.5775 vs derived 0.6650 | **Strong (Empirical Fact)**; paired bootstrap CI above zero |
| 5 | Derived profile does not statistically separate from nearest discrete ramp | `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` | 2026-08-23 | OLMo 16K diff: $-0.0056$ $[-0.0464, +0.0345]$; Qwen 64K diff: $+0.0250$ $[-0.0525, +0.1000]$ | **Strong (Null Finding)**; bounds registered |
| 6 | Stateless boundary-slope operator collapses at long context | `paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` | 2026-08-23 | Core-4 RULER macro: 0.0000 at 8K and 16K | **Strong (Falsified)**; relative phase destruction |
| 7 | Zero-parameter single static tables damage 1x Native retention | `paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md` | 2026-08-24 | 1x tail NLL: EVQ-Cosh $+3.9780$ (6.9492 vs 2.9712); protected Cosh $+0.6922$ (3.6633 vs 2.9712) | **Strong (Falsified / Negative Gate)**; stopped |
| 8 | Underdetermined 62-DOF direct-$z$ calibration fails held-out rows | `paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md` | 2026-08-24 | Held-out row 2 regressed $+0.08229$ 2x NLL (gate $\le +0.05$) | **Strong (Negative Gate)**; stopped |
| 9 | Log-frequency law eliminates arithmetic movement saturation drift | `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` | 2026-08-31 | Arithmetic RMS drift $0.01099/0.02195/0.03200$ (s=2/4/8); max error 0.24559 at k=21; log drift $= 0$ | **Strong (Empirical & Mathematical)**; audit hash-bound |
| 10 | Static log-s4 table passes 1x PG-19 and downstream gates on OLMo | `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` | 2026-08-31 | 1x PG-19 PPL retention: 0.875302 ($\ge 0.875$); 5-task retention: 0.915103 ($\ge 0.875$); RULER-13 4K/8K/16K: .714/.667/.499 | **Strong (Verified Deployment Cell)**; hash-bound |
| 11 | Multiset slot permutation causes catastrophic collapse | `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` | 2026-08-31 | OLMo 1x PG-19 NLL: $3.104 \to 6.865$ ($\Delta = +3.761$); Qwen 64K core-4: $0.7000 \to 0.0000$ | **Strong (Decisive Architectural Finding)**; SHA-256 bound |
| 12 | Low-dim C2 clipped affine fails strict Native retention gates | `paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md` | 2026-09-01 | Movement MAE 0.001223; OLMo 1x PPL retention 0.870971 ($<0.875$); Qwen 32K retention 0.868902 ($<0.875$) | **Strong (Empirical Boundary)**; hash-bound |
| 13 | Reference length calibration resolves Gemma K128 16K collapse | `paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md` | 2026-09-01 | Config 8192 corrected to operating 4096; 16K RULER recovered from 0.0000 to 0.7250 (phys) / 0.7950 (index) | **Strong (Protocol Resolution)**; hash-bound |
| 14 | Normalized index outperforms physical-$x$ at K128 and K32 | `paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md` | 2026-09-01 | K128 16K: Index .790000 vs Phys .728125 ($\Delta = +.061875$ $[.028109, .096250]$); K32 32K ret: .923 vs .859 | **Strong (Replicated Empirical Advance)**; hash-bound |
| 15 | Static pure-$z$ table improves packed natural 64K NLL | `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` | 2026-09-02 | Qwen K32: 64K NLL $2.7549 \to 2.6308$ ($\Delta = -0.1241$ $[-0.1479, -0.1029]$, 32/32 streams) | **Moderate (Internal Decision Only)**; raw JSON unrecovered |
| 16 | Far-source evidence enters canonical answer logits | `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` | 2026-09-02 | Far source-use: Native $+0.066$ vs Index $+1.504$ ($\Delta = +1.438$ $[0.720, 2.230]$); answer NLL $5.123 \to 3.695$ | **Moderate (Internal Decision Only)**; raw JSON unrecovered |
| 17 | Static pure-$z$ fails to improve natural generation QA | `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` | 2026-09-02 | 30-row QA macro: Native 0.13229 vs Index 0.10174 ($\Delta = -0.03055$ $[-0.1168, +0.0486]$, spans zero) | **Moderate (Internal Decision Only)**; raw JSON unrecovered |
| 18 | Contrast decoding & sequence reranking fail as readout rescues | `paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md` | 2026-09-02 | Source contrast decoding: $+0.00079$; sequence reranking: $+0.0067$; oracle headroom: 0.18656 | **Moderate (Internal Decision Only)**; raw JSON unrecovered |
| 19 | Free headwise gain is a teacher-forced shortcut, not QA solution | `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` | 2026-09-02 | Loss dropped $3.56 \to 2.31$, but Hotpot F1 dropped $.242 \to .194$ and EOS termination $178 \to 108$ | **Strong (Falsified Operator Class)**; raw bundle hash-bound |
| 20 | Basin barrier: Native-start and log-start occupy opposite Pareto sides | `paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` | 2026-09-02 | Log-start: Hotpot F1 .25223, 4K ret .77138 (fail); Native-start: Hotpot F1 .02560 (fail), 4K ret 1.0462 | **Strong (Structural Obstruction)**; raw bundle hash-bound |
| 21 | 18-sample 64-D behavioral gradient common-direction candidate | `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` | 2026-09-02 | Exited main line: failed unopened holdout with inconsistent 8K/16K behavior (Fact F) | **Falsified (Retired Route)**; verified |
| 22 | Continuous scale-flow ODE establishes canonical multi-scale trajectory | `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` | 2026-09-02 | Proved to be gauge / bookkeeping identity; $s=4 \to s=8$ composes algebraically but fails behaviorally | **Falsified / Theoretical Artifact**; (T6) verified |
| 23 | Smooth convex gain balance $L(g) = e^{-g\mu} + 2\alpha\epsilon g$ | `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` | 2026-09-02 | Falsified by sweep non-monotonicity (derivative violates convexity between $g=1.0$ and $1.05$) | **Falsified / Retracted**; (T8) verified |
| 24 | Novelty volume formula $2(1+2^{m-1}) \in (2,3]$ | `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` | 2026-09-02 | Correct super-doubling formula is $2 + 1/(4^{1-m_k}-1) \in (2, \infty)$ | **Mathematical Error Corrected**; (T7) verified |
| 25 | Slot merging monotonically drops mean power | `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` | 2026-09-02 | Correct shift is $\text{Re}(c_1 \bar c_2) \in [-|c_1||c_2|, +|c_1||c_2|]$, sign is content-dependent | **Mathematical Error Corrected**; (A3) verified |

---

## 5. Synthesis & Answers to Architectural Review Questions

### A. Precise Problem Definition
[DERIVED] The zero-training RoPE retrofit challenge is to construct a single, content-independent, inference-time frequency table replacement under frozen pretrained weights that simultaneously preserves Native in-window performance (PG-19 retention $\ge 0.875$) and enables long-range autoregressive generation QA ($\Delta > L_{\text{native}}$). The fundamental obstacle is that any non-trivial table modification queries the frozen attention weights on novel joint phase configurations off the Native manifold, where linear evidence transmission (NLL / logit likelihood) dissociates from autoregressive top-1 margin dominance and terminal EOS generation.

### B. Confirmed Empirical Facts
1. [OBSERVED] Moving interior exponent allocation $z$ at fixed support causally changes long-context behavior in both training and frozen checkpoints (`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`, `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`).
2. [OBSERVED] Rotary subspace-to-frequency pairing is strictly ordered and non-exchangeable; multiset permutations collapse OLMo and Qwen (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
3. [OBSERVED] Static log-s4 tables pass 1x retention on OLMo and provide massive gains on synthetic RULER and natural NLL, but hit a ceiling at $s=8$ (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`).
4. [OBSERVED] Normalized index is empirically superior to physical-$x$ across K32 and K128 architectures (`K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md`, `K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md`).
5. [OBSERVED] Natural long NLL and far-source likelihood improve, but natural generation QA remains flat and readout rescues fail (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`).
6. [OBSERVED] The optimization landscape is bimodal: Native-start models preserve Native retention but cannot bootstrap long QA; long-start models achieve long QA but fail Native retention (`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`).

### C. Historical Cognitive Traps & Misleading Abstractions
1. [DERIVED] **Unordered Spectral Density Fallacy:** Treating RoPE as an invariant continuous density $\pi(\omega)$ or optimizing static collision kernels ignored the dominant learned coupling between specific matrix blocks in $W_q, W_k$ and specific rotary slots.
2. [DERIVED] **Likelihood-to-Generation Equivalence Fallacy:** Assuming that lowering answer NLL or improving far-source logit likelihood automatically translates into greedy autoregressive generation accuracy. Generation is piecewise-constant and governed by argmax competitor margins, not mean likelihood.
3. [DERIVED] **Geometric Closeness Fallacy:** Assuming that fitting a 2-parameter curve with sub-percent RMSE (0.001223) in movement space guarantees functional equivalence at knife-edge operational gates.
4. [DERIVED] **Semigroup Flow Fallacy:** Treating scale-flow ODEs $\frac{dx}{d\tau} = v_\theta(x)$ as a predictive law. For static tables, flow trajectories are unobservable gauge, and formula-level composition does not prevent behavioral failure at $s=8$.
5. [DERIVED] **Free Gain Sharpening Shortcut:** Interpreting massive teacher-forced training loss reductions from amplitude scaling as true long-context reasoning, when it was merely a sharpening shortcut that degraded actual generation and EOS stopping.

### D. Minimal Mechanistic Explanation of Historical Failures
[DERIVED] The unified minimal explanation across all historical failures is the **Off-Arc Novel Phase Configuration Disconnect**:
1. At $\Delta \le L_{\text{native}}$, the model was trained on the 1-parameter phase trajectory $\gamma(t) = (\omega_k t) \pmod{2\pi}$.
2. Under a non-uniform retrofit table $\omega'_k = \omega_k s^{-m_k}$, the deployed trajectory $\gamma'(t)$ for $t \in (L, sL]$ forms a distinct 1-parameter subgroup whose joint phase configurations are completely disjoint from the Native trajectory (except at the origin).
3. The frozen network's downstream feedforward layers and attention heads have never been exposed to these joint phase combinations.
4. While smooth, linear projections (NLL loss) average out interference and reflect increased energy from slower frequencies, non-linear multi-hop reasoning and autoregressive decoding require sharp, coordinated vector alignments across multiple layers to maintain winner margins over distractors.
5. Because tolerance to these off-arc configurations is circuit- and task-dependent ($R_T(\theta, \Omega')$), no single static table can satisfy the competing constraints of preserving Native-window phase precision while preventing destructive interference on unseen long-context combinations.

### E. Assessment of Future Solutions
[DERIVED] **Current repository evidence does NOT support the existence of any high-confidence zero-training single-table solution.**
Every tested single-table candidate either fails Native retention, fails natural QA generation, or collapses at $s=8$. Fabricating a new parametric curve, ramp, or selector without acknowledging this barrier violates scientific truth. Any viable future direction must depart from a single global static table (e.g., exploring input-dependent routing, cheap low-rank co-adaptation, or headwise specialization).

### F. Single Highest Information-Gain Next Action
[DERIVED] **Action:** Recover and verify the missing raw JSON/JSONL artifacts for the 2026-09-02 Qwen natural NLL, QA, and table $\times$ gain experiments from external storage, if a surviving copy exists.
- **Success Criterion:** Bitwise SHA-256 matches for the reported session hashes `ba489f47...`, `6109434e...`, and `5d6f2f2e...`.
- **Stop Condition:** If recovered, promote these internal decision numbers to formal repository evidence owners; if no copy exists, freeze these numbers as permanent internal decision evidence and do NOT run unauthorized GPU reruns.
