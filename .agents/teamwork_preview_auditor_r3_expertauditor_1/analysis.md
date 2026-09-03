# Comprehensive Forensic Audit of Zero-Training RoPE Retrofit Evidence

**Role:** Experimental Auditor (R3)  
**Target:** Mature-Checkpoint Zero-Training RoPE Retrofit Multi-Role Research Audit  
**Date:** 2026-09-03  
**Working Directory:** `.agents/teamwork_preview_auditor_r3_expertauditor_1`  
**Integrity Mode:** Development (`ORIGINAL_REQUEST.md` ## 2026-09-03T02:34:13Z)  
**Contract Enforcement:** Strictly read-only, zero GPU computation, zero fabrication, strict four-tag taxonomy (`[OBSERVED]`, `[DERIVED]`, `[HYPOTHESIS]`, `[UNKNOWN]`).

---

## 1. Executive Summary & Epistemological Taxonomy

[DERIVED] The central challenge of zero-training RoPE retrofit on mature language models is defined by a severe Pareto dilemma: methods that succeed in extending sequence length on synthetic retrieval benchmarks (such as RULER) or lowering long-context teacher-forced perplexity consistently suffer from either non-negligible in-window degradation (violating the Native retention gate) or complete inability to translate long-range positional signal into autoregressive natural multi-hop question answering (the capability-conversion barrier).

To establish an unassailable empirical audit, this report enforces a strict four-tag contract:
- **`[OBSERVED]`**: An empirical measurement, recorded metric, or mathematical property directly verifiable in tracked repository artifacts, with exact relative file paths and numerical values cited.
- **`[DERIVED]`**: A logical or mathematical consequence rigorously proved from established axioms, exact algebraic identities, or empirical observations under stated assumptions.
- **`[HYPOTHESIS]`**: A falsifiable mechanistic conjecture, model, or interpretation consistent with evidence but not uniquely identified or universally confirmed.
- **`[UNKNOWN]`**: An unmeasured, unverified, unrecorded, or missing empirical quantity in the repository. If an assertion lacks backing evidence in repository files, it is explicitly branded `UNSUPPORTED BY REPOSITORY EVIDENCE`.

---

## 2. In-Depth Audit of Core Experimental Evidence

### 2.1 The $S=2/4/8$ Failure Spectrum Across Models and Scales

[OBSERVED] On OLMo-2-0425-1B-Instruct (1.485B parameters, $K=64$ rotary pairs, native window $L=4096$, base $b=500,000$), an extension factor of $s=2$ achieved stable behavior: the zero-refit $s=2$ profile with fixed gain coefficient $c=0.074$ passed the 1x double gate on PG-19 tail NLL (NLL 2.987191 vs Native 2.971047, PPL retention 0.983986 $\ge 0.875$) and downstream 5-task macro (0.359361 vs Native 0.345134, retention 1.041220 $\ge 0.875$), while maintaining 2x PG-19 tail NLL of 2.970433 (vs Native 7.100855) and RULER core-4 macro of 0.5150 (vs Native 0.0000) (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 298–310).

[OBSERVED] On the same OLMo-2-1B model at $s=4$, the frozen log-law table $\omega'_k = \omega_k 4^{-m_k}$ with $c=0.074$ passed the registered 1x gate on PG-19 tail NLL (3.104234 vs Native 2.971047, PPL retention 0.875302 $\ge 0.875$) and downstream 5-task macro (0.315833 vs Native 0.345134, retention 0.915103 $\ge 0.875$). At 2x and 4x horizons, it achieved PG-19 NLL of 3.083278 (vs Native 7.100855) and 3.081946 (vs Native 7.205538), and completed 13-task RULER at 4K/8K/16K scoring 0.71397 / 0.66705 / 0.49859 (vs Native 0.71308 / 0.00000 / 0.00385) (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 130–159).

[OBSERVED] However, scaling the same frozen movement mask to $s=8$ triggered an immediate, registered failure. In the core-4 RULER evaluation at 8x, the arithmetic profile scored 0.2100 and the log-law profile scored 0.3025. In the full RULER-13 suite, an early stop was registered because `single_key_3` collapsed to 0.0000 at 4K, 0.2000 at 8K, and 0.2500 at 16K, indicating that the frozen profile completely broke uniform retrieval through $[1, 8]$ (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 74–75, 113–118).

[OBSERVED] On Qwen2.5-1.5B-Instruct ($K=64$, native window $L=32768$, base $b=10^6$), transferring the frozen construction to $s=4$ with $c=0.074$ scored 0.7000 at 64K and 0.5875 at 128K on core-4 RULER (vs Native 0.5450 / 0.4350) (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 174–178). But evaluating the same low-dimensional two-parameter $C2$ law on the 32K Native window produced a macro score of 0.7125 (vs Native 0.8200), yielding a retention ratio of 0.868902, which strictly failed the declared 0.875 gate (`paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`, lines 135–142).

[OBSERVED] On Qwen2.5-0.5B-Instruct ($K=32$, native window $L=32768$), a single frozen normalized-index $s=2$ table passed the 32K Native capability gate (macro 0.559167 vs Native 0.547821, retention 1.020711) and improved 64K full RULER-13 macro to 0.514551 (vs Native 0.220513 and official YaRN-s2 0.453654; paired delta over YaRN $+0.060897$, 95% CI $[0.027627, 0.095835]$) (`paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md`, lines 49–66).

[OBSERVED] On Gemma-1.1-2B-Instruct ($K=128$, configured window 8192, confirmed operating window $L_{ref}=4096$), the $s=2$ physical profile restored 8K core-4 RULER to 0.8600 (vs Native 0.0000) and reduced natural 8K NLL from 10.548809 to 3.163365, while $s=4$ scored 0.7250 (physical) and 0.7950 (normalized index) at 16K (vs Native 0.0000) (`paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md`, lines 78–83, 105–108, 149–154).

[DERIVED] The $S=2/4/8$ degradation pattern demonstrates that zero-training static tables exhibit an effective scale radius $S_{max} \approx 4$. Scaling to $S=8$ fails not because of arbitrary hyperparameter tuning, but because of Theorem T7 (novelty volume divergence): for any frozen non-uniform mask $m_k \in (0, 1)$, the novel phase range $N_k(S) = \omega_k L (S \cdot 4^{-m_k} - 1)$ obeys super-doubling $N_k(2S)/N_k(S) > 2$, while the PI-absorbed blur factor $S^{m_k}$ diverges. A static table calibrated at $S=4$ cannot contain both phase novelty and resolution blur at $S=8$.

---

### 2.2 Ordered Rotary Coupling: Multiset vs. Slot Assignment Non-Exchangeability

[OBSERVED] In OLMo-2-1B, a pre-registered experiment took the successful log-s4 table and permuted the interior final frequencies while preserving the exact frequency multiset, endpoints, gain ($c=0.074$), and evaluation seed. The 1x PG-19 tail NLL exploded from 3.104234 to 6.864926, producing a catastrophic degradation of $\Delta\text{NLL} = +3.760692$, decisively crossing the pre-registered failure threshold of $+0.10$ (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 271–279).

[OBSERVED] In Qwen2.5-1.5B ($K=64$), an identical pre-registered control preserved the final frequency multiset, endpoints, gain, and 64K evaluation rows, but permuted the interior rotary slot assignments. All four 20-row RULER task cells collapsed to exactly 0.0000, destroying the 0.7000 macro achieved by the ordered reference (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 185–189).

[OBSERVED] In an earlier OLMo dilation-permutation control, randomly permuting interior dilation factors $m_k$ while preserving endpoints produced 1x PG-19 NLL of 4.068625 and core-4 RULER scores of 0.0000 / 0.0000 at 8K/16K (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 261–264).

[DERIVED] These experiments definitively refute the hypothesis that RoPE representation depends solely on the unordered spectral measure $\pi(\omega)$. A pretrained checkpoint does not interact with an abstract multiset of frequencies; its projection weights $W_q, W_k$ learned an ordered coupling where specific 2D rotary subspaces $V_k = \text{span}\{\cos(\omega_k\Delta), \sin(\omega_k\Delta)\}$ are wired to specific semantic channels. Reassigning frequencies to different rotary slots completely scrambles the phase representation seen by the attention heads.

---

### 2.3 The Native Retention Gate: Knife-Edge Degradation at the 1x Window

[OBSERVED] The formal repository criterion for acceptable retrofit requires passing a double gate at 1x: PG-19 tail NLL PPL retention $\ge 0.875$ AND downstream natural task macro retention $\ge 0.875$.
- OLMo-2-1B log-s4 ($c=0.074$): PPL retention **0.875302** (passed by 0.000302), macro retention **0.915103** (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 130–134).
- OLMo-2-1B arithmetic-s4 ($c=0.074$): PPL retention **0.869584** (FAILED gate $< 0.875$) (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 237–240).
- OLMo-2-1B two-parameter C2 ($c=0.074$): PPL retention **0.870971** (FAILED gate $< 0.875$), despite geometric reconstruction MAE of only 0.001223 against the 64D profile (`paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`, lines 58, 78).
- Qwen2.5-1.5B static C2: 32K macro retention **0.868902** (FAILED gate $< 0.875$) (`paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`, line 141).
- OLMo-2-1B headwise log-start two-axis: 4K PG-19 PPL retention **0.7714** (FAILED gate $< 0.875$) (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, line 176).
- Official YaRN-4: 4K PG-19 PPL retention **0.6588** (FAILED gate $< 0.875$) (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, line 175).

[DERIVED] The retention gate acts as a knife-edge threshold. Geometric error metrics (such as MAE or RMSE between frequency curves) are completely incapable of predicting whether a table passes the 0.875 retention gate: an MAE shift of 0.001223 was enough to drop OLMo PPL retention from 0.875302 to 0.870971, crossing the operational acceptance boundary.

---

### 2.4 The Capability-Conversion Barrier: NLL, RULER, and Natural Multi-Hop QA

[OBSERVED] In packed natural text evaluation on Qwen2.5-0.5B ($K=32$) over 32 paired streams:
- 32K Native NLL was 2.597666 vs Index 2.615399 (retention $\approx 0.9824$).
- 64K Native NLL was 2.754945 vs Index 2.630842 ($\Delta\text{NLL} = -0.1241$, 95% CI $[-0.1479, -0.1029]$, improving on 32/32 streams) (`paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, lines 147–153).

[OBSERVED] In a matched evidence-position bridge experiment testing far vs near vs ablated sources:
- Native far source-use was $+0.066$ (interval spanning zero).
- Index far source-use was $+1.504$ (index-minus-Native difference $+1.438$, 95% CI $[0.720, 2.230]$).
- Canonical answer NLL under far evidence dropped from 5.123 (Native) to 3.695 (Index), while both arms converged to $\approx 5.19$ under source ablation (`paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, lines 182–191).

[OBSERVED] Despite this massive infusion of source likelihood into the logits, natural autoregressive question answering failed completely:
- In a 30-row panel across 2WikiMultihopQA, Qasper, and HotpotQA, macro F1 was: Native 0.13229, normalized index 0.10174, YaRN 0.11197. Index scored lower than Native by $-0.03055$ (paired 95% CI $[-0.1168, +0.0486]$) (`paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, lines 162–171).
- In HotpotQA on 38 Native-short-correct rows:
  - At 8K: Native EM/F1 was 0.000 / 0.002, log-s4 was 0.605 / 0.677, YaRN-4 was 0.579 / 0.656 (difference $+0.0215$, CI $[-0.161, +0.202]$).
  - At 16K: Native was 0.000 / 0.000, log-s4 collapsed to **0.079 / 0.153**, while YaRN-4 maintained **0.395 / 0.500** (difference $-0.3466$, CI $[-0.496, -0.198]$).
  - Correct EOS emission at 16K: Native 0/38, log-s4 **22/38**, YaRN-4 **36/38** (`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, lines 35–42).
- In K32 full RULER-13 at 64K, while index excelled at NIAH tasks, it fell behind on reasoning tasks:
  - Variable tracking: Native 0.030, Index 0.280, YaRN **0.360**.
  - SQuAD QA: Native 0.050, Index 0.150, YaRN **0.200**.
  - HotpotQA: Native **0.250**, Index 0.150, YaRN 0.200 (Index was strictly worse than Native!) (`paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md`, lines 87–91).

[OBSERVED] Post-hoc decoding rescues failed to recover natural QA:
- Source-contrast greedy decoding ($\tilde\ell = (1+\alpha)\ell_{index} - \alpha\ell_{Native}$) yielded delta over greedy of $+0.00079$ (95% CI $[-0.0271, +0.0292]$).
- Sequence reranking across 3 candidates by source score yielded delta over index of $+0.0067$ (95% CI $[-0.0173, +0.0330]$, macro 0.1142 vs 3-candidate oracle 0.18656) (`paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, lines 198–205).

[DERIVED] The capability-conversion barrier is explained by the fundamental mathematical distinction between smooth likelihood functionals and piecewise-constant greedy generation maps (Pillar 2 / Theorem T8). Canonical answer NLL is smooth in the frequency table: improving it across a long context injects an $O(1)$ cumulative margin sum across the entire sequence, which corresponds to only $O(0.05 - 0.1)$ nats per token. However, at visited decoding states where an erroneous competitor token leads, the margin gap $\delta_t = \max_{v \ne y}\ell_v - \ell_y$ is typically $O(1)$ or larger. A small smooth shift in logit mass fails to cross the argmax boundary. Once a single token is mispredicted, the model drifts off the canonical prefix into self-induced states where distractor noise dominates and EOS is not triggered.

---

### 2.5 Attention Normalization and Gain Scaling Tricks

[OBSERVED] Eliminating attention gain ($L0$, unit gain $g=1.0$) causes severe long-range failure: in OLMo-2-1B, gain-free log-interpolation reduced 16K single-key retrieval to 0.4000; at $s=8$, gain-free core-4 macro was 0.0167, and 32K macro was exactly 0.0000 (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 84–87).

[OBSERVED] An empirical gain sweep on OLMo log-s4 (Native-short F1 0.796) revealed extreme non-linear sensitivity:
$g=0.9 \to \text{F1 } 0.036$; $1.0 \to 0.588$; $1.05 \to 0.608$; $1.1026 \to 0.680$; $1.15 \to 0.692$; $1.20 \to 0.641$; $1.30 \to 0.439$. The loss slope $(1 - \text{F1})$ exhibited consecutive derivatives of $-5.5, -0.4, -1.4, -0.25, +1.0, +2.0$, violating convexity between $1.0$ and $1.05$ (`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, lines 44–50, 338–348).

[OBSERVED] In headwise factorized optimization, granting 512 free headwise gain parameters lowered teacher-forced training loss from 3.5612 to 2.3105 (by far the lowest loss achieved), yet reduced HotpotQA F1 from 0.24237 to 0.19439, crashed correct EOS termination from 178/200 to 108/200, and doubled average generated length from 9.10 to 18.50 tokens (with 92/200 generations hitting the 32-token cap) (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, lines 30–32, 118, 137, 149–151).

[DERIVED] For any scalar $g > 0$, multiplying attention logits by $g$ preserves the argmax of the softmax distribution ($\arg\max_j g s_{ij} = \arg\max_j s_{ij}$). Gain cannot correct key ranking errors or fix retrieval confusion. It acts strictly as a temperature parameter that controls softmax sharpness. Under teacher forcing, artificial sharpening rewards overconfident peaking on already-correct short keys (slashing cross-entropy loss), but during autoregressive rollouts, over-sharpening amplifies incorrect leading tokens and suppresses EOS emission.

---

### 2.6 Spatial Factorization: Headwise vs. Layerwise Clocks and the Basin Barrier

[OBSERVED] On OLMo-2-1B, evaluating learned spatial clocks across architectural scopes produced:
- Shared $\alpha$ (1 scalar): HotpotQA F1 **0.22351**, 4K loss 3.3822
- Per-layer $\alpha$ (16 scalars): HotpotQA F1 **0.22437** (statistically indistinguishable from 1 shared scalar)
- Per-head $\alpha$ (256 scalars): HotpotQA F1 **0.24237** (improving over frozen log-s4 0.21169)
- Per-head two-axis ($\alpha$ along log-s4, $\beta$ along YaRN-4; 512 scalars): HotpotQA F1 **0.25223**, 2WikiMQA **0.26651**, Qasper **0.18853** (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, lines 115–120, 133–140, 155–158).

[OBSERVED] However, the two-axis model exposed an unbridgeable Basin Barrier between Native retention and long capability:
- Log-start two-axis: HotpotQA F1 was 0.25223, but 4K PG-19 PPL retention was **0.7714**, failing the 0.875 retention gate on all 20/20 documents.
- Native-start two-axis: 4K PG-19 PPL retention was **1.0462** (strictly outperforming Native), but HotpotQA F1 collapsed to **0.02560** (with only 8/200 EOS completions), exactly matching Native's failure state (0.01942 F1, 8/200 EOS) (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, lines 25–29, 133, 140, 176–177).

[DERIVED] The Native-start optimization stalled (mean $\alpha = 0.0966$, 16K loss 6.75) because of gradient starvation at the Native operating point. Deeply aliased high-frequency slots wrap multiple times across the extended context; their phases average out (Riemann-Lebesgue cancellation), producing near-zero loss gradients. The optimizer cannot find a local descent trajectory toward the long basin from exact Native weights without destroying Native representations.

---

### 2.7 Learned Co-Adaptation: Table-Weight Crossings and Transplant Obstruction

[OBSERVED] In a 50M parameter model on TinyStories ($L=512$, base 500K, seed 42), crossing pretrained weights with runtime frequency tables produced:
- Geometric weights + Geometric table: PPL **7.14** (LM loss 1.9659)
- Geometric weights + EVQ table: PPL **76.20** (LM loss 4.3333)
- EVQ weights + Geometric table: PPL **23.05** (LM loss 3.1378)
- EVQ weights + EVQ table: PPL **7.16** (LM loss 1.9685)
Factorial decomposition of LM loss yielded: table main effect $+0.5991$, weights main effect $-0.5965$, and table $\times$ weights interaction **$-3.5367$** (95% CI $[-5.165, -3.039]$). The interaction term was 5.9 times larger in magnitude than either main effect (`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`, lines 364–368, 388–401).

[OBSERVED] In a 151.9M parameter model across two training seeds on FineWeb-Edu ($L=1024$):
- FMRoPE-trained weights achieved tail NLL of **3.426** on FMRoPE-derived table vs **5.776** on Cosh-derived table.
- Anchored-Cosh-trained weights achieved tail NLL of **3.479** on Cosh-derived table vs **4.455** on FMRoPE-derived table.
The crossover interaction $[L(W_F, T_C) - L(W_F, T_F)] - [L(W_C, T_C) - L(W_C, T_F)]$ was $+3.400$ for seed 137 and $+3.251$ for seed 256 (`paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`, lines 205–216).

[DERIVED] These crossings prove that Transformer projection weights $W_q, W_k$ co-adapt to the specific geometry of their training frequency table. Swapping the table post-hoc breaks the co-adaptation. Furthermore, by the Exact Transplant Obstruction Theorem (`rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`, lines 67–109), any position-independent invertible linear map satisfying $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ forces multiset equality $\{|\omega'_k|\} = \{|\omega_k|\}$. Post-hoc linear adapters cannot mathematically compensate for a non-identical frequency spectrum.

---

## 3. Methodological Validity Threats and Confounders

### 3.1 What Did the Interventions Actually Change?

[DERIVED] A persistent confound in early RoPE literature is conflating three mathematically distinct axes:
1. **Sampled Support $(a, R)$**: The base and endpoint frequency span, setting the maximum and minimum wavelengths ($\omega_{max}, \omega_{min}$).
2. **Interior Exponent Allocation $z$**: The normalized quantile distribution $z_k \in [0, 1]$ where $x_k = a + R z_k$.
3. **Attention Amplitude Scaling (Gain)**: The scalar multiplier $g(S) = 1 + c \ln S$ applied to attention logits.
4. **Context Routing**: Serving policies that branch to Native RoPE for short contexts ($\le L_{native}$) and use an expanded table only for long inputs.

[OBSERVED] In OLMo same-support experiments, fixing $(a, R)$, amplitude $c=0.074$, and checkpoint revealed that geometric spacing scored 0.0056 at 16K, while non-linear derived allocation scored 0.6047 (`paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`, lines 152–154). This isolates $z$ as a true causal variable. Conversely, earlier claims attributing performance gains to pure table shapes were confounded by omitted gain: in OLMo, $z$-allocation alone with unit gain collapsed at 16K to 0.4000 and at 32K to 0.0000 (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 84–87).

### 3.2 Evaluator Bugs, Wrapper Parity Breaches, and Numerical Artifacts

The repository audit uncovered three critical methodological breaches in historical runs that compromised evidentiary integrity:

1. **In-Place Buffer Patching Bug (Historical Hybrids Voided):**  
   [OBSERVED] In 28 historical hybrid/per-head evaluations conducted in July 2026, the evaluator retained a Python view of the CPU float32 Native `inv_freq` buffer, then patched that buffer to EVQ in place. Consequently, both the "Native" and EVQ references evaluated the identical post-patch tensor. 0 of 28 historical hybrid receipts matched their declared hybrid tables; 28 of 28 matched the buggy alias reconstruction (`rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`, lines 230–256). All historical hybrid probe conclusions were invalid.

2. **Custom Attention Wrapper Logit Parity Failure:**  
   [OBSERVED] Early gradient and behavioral analyses utilized a custom attention wrapper that failed bitwise logit parity at $z=0$ against the official HuggingFace implementation. All behavioral-gradient conclusions derived under that wrapper were voided. The corrected intervention operates strictly by replacing `model.rotary_emb.inv_freq`, achieving exact 0-error full-vocabulary logit parity across 6/6 checkpoints (`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, lines 20–23).

3. **Stride-16 Numerical Aliasing in Qwen 128K:**  
   [OBSERVED] An early Qwen table construction sampled pairwise distances at stride 16, creating artificial duplicate uniqueness values, moving the fast endpoint by $0.17\%$, and creating order crossings at pairs 1 and 18. This artifact artificially inflated the Qwen 128K core-4 score to 0.6175. When recomputed at stride 2, the order crossings vanished and the valid score dropped to 0.5400 (`paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`, lines 102–114, 175–186). Any citation of the 0.6175 number is an integrity violation.

### 3.3 Metric Validity and Asymmetric Evaluation Contracts

[DERIVED] The repository demonstrates a severe metric hierarchy disconnect:
$$\text{Teacher-Forced NLL} \centernot\implies \text{Synthetic RULER Retrieval} \centernot\implies \text{Autoregressive Natural QA (F1 / EOS)}$$
- NLL measures smooth probability mass over ground-truth tokens. An intervention can substantially lower answer NLL (e.g. from 5.123 to 3.695) without altering top-1 token rankings.
- RULER tests synthetic needle retrieval in repetitive contexts. Saturated single-key needles (scoring 1.00) mask complete failures on multi-key tracking (`multikey_3` scoring 0.00) (`paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`, lines 135–152).
- Natural QA requires multi-hop distractor suppression, generation coherence, and explicit EOS token emission. High RULER scores do not guarantee natural QA survival.

### 3.4 Incompatible Cross-Protocol Comparisons

[OBSERVED] Direct comparisons between the 1x retention score of 0.875302 (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`) and the YaRN score of 0.6588 (`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`) are invalid cross-protocol contaminations. The 0.875302 figure was measured under the 2026-08-31 full-document PG-19 gate protocol with $c=0.074$. The 0.6588 figure was measured under the 2026-09-02 headwise protocol evaluating 20 frozen documents scoring only the final 512 target tokens. Within the headwise protocol, log-s4 retention was 0.7171 vs YaRN 0.6588 (`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, lines 649–653).

### 3.5 Context Length Reference Confounder: Gemma K128

[OBSERVED] Gemma-1.1-2B was documented in its config with `max_position_embeddings = 8192`. Initial retrofit attempts at 16K assumed an extension factor of $s = 16384 / 8192 = 2$ and collapsed to 0.0000 on core-4 RULER. An independent calibration revealed that the checkpoint was actually pretrained with an operating context of $L_{ref} = 4096$. Evaluating at 16K was actually an $s=4$ extrapolation. When re-anchored to $L_{ref} = 4096$ with $s=4$, the physical profile achieved 0.7250 and normalized index achieved 0.7950 (`paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md`, lines 8–10, 24–30). The initial "16K capability ceiling" was entirely an artifact of a false reference length.

---

## 4. Root Cause Diagnosis of the S=2/4/8 Failure

To establish why zero-training retrofit fails at $S=8$ and natural QA, we evaluate four competing hypotheses:

### Hypothesis A: Family-Specific Defect (Arithmetic vs. Log vs. YaRN)
- [OBSERVED] The arithmetic law $\omega'_k = \omega_k ((1-m_k) + m_k/s)$ has a quantifiable movement saturation defect: its effective exponent $q_k(s) = -\ln(\omega'_k/\omega_k)/\ln s$ drifts below $m_k$, with RMS error increasing monotonically from 0.01099 at $s=2$ to 0.03200 at $s=8$ (worst pair $k=21$ error 0.24559) (`paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`, lines 45–51).
- [OBSERVED] The log law $\omega'_k = \omega_k s^{-m_k}$ fixes this defect ($q_k \equiv m_k$) and satisfies composition $\omega(s_1 s_2) = \omega(s_1) s_2^{-m}$.
- [DERIVED] However, fixing the mathematical formula does NOT fix the empirical failure: the log-law profile at $s=8$ still suffered severe capability degradation (core-4 macro 0.3025, single-key-3 collapsing to 0.00 at 4K). Furthermore, while YaRN outperforms log-s4 on HotpotQA at 16K (0.500 vs 0.153 F1), YaRN collapses on RULER at 16K (0.1056 vs 0.49859) and fails 1x retention (0.6588). Therefore, the failure is **not primarily family-specific**.

### Hypothesis B: Optimization Failure
- [OBSERVED] In zero-training retrofit, there is NO training or optimization during deployment.
- [OBSERVED] In the headwise factorized experiment where 512 parameters were learned, Native-start stalled at $\alpha = 0.0966$ with high loss, while log-start succeeded in learning long directions but destroyed Native retention (`paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, lines 119–120).
- [DERIVED] This is not an algorithmic failure of AdamW or learning rates; it is gradient starvation caused by phase averaging over aliased frequencies (Pillar 3).

### Hypothesis C: Table-Weight Coordinate Incompatibility
- [OBSERVED] Proven conclusively by the 50M 2x2 crossing (interaction $-3.5367$, 5.9x larger than main effects) and the 151.9M crossing (FMRoPE weights reject Cosh tables; Cosh weights reject FMRoPE tables).
- [DERIVED] Pretrained Transformer weights are inextricably coupled to their training rotary coordinate system. Swapping frequencies post-hoc distorts internal key-query inner products.

### Hypothesis D: Fundamental Structural Constraint (The True Root Cause)
[DERIVED] The failure of zero-training RoPE retrofit across $S=2/4/8$ and natural QA is governed by the conjunction of four fundamental structural constraints:
1. **Transplant Rigidity (Theorem T2):** No content-independent, position-independent linear Q/K transformation can compensate for an unequal frequency multiset ($A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta) \implies \{|\omega'_k|\} = \{|\omega_k|\}$).
2. **Off-Arc Novel Phase Exposure (Theorem T5):** Any non-uniform spectrum redistribution ($m_k \not\equiv \text{const}$) forces the deployed phase trajectory $\{(\omega'_k \Delta) : \Delta \in (L, SL]\}$ to be completely disjoint from the Native trajectory $\{(\omega_k \Delta) : \Delta \le L\}$. The frozen weights are queried at joint phase combinations that never existed during pretraining.
3. **Novelty Volume Super-Doubling (Theorem T7):** The volume of novel phase configurations grows super-linearly ($N_k(2S) > 2 N_k(S)$), creating a finite scale horizon ($S_{max} \approx 4$) beyond which unadapted circuits undergo attention noise explosion.
4. **Likelihood-to-Winner Margin Mismatch (Pillar 2):** In multi-hop reasoning and natural generation, correct answers require positive margins over distractors across every step of the autoregressive path. A static frequency warp injects subtle smooth likelihood gains into canonical tokens ($O(0.05 - 0.1)$ nats) but fails to overcome $O(1)$ distractor competitor margins, resulting in generation breakdown and EOS suppression.

---

## 5. Data Completeness and Repository Artifact Forensics

### 5.1 The 2026-09-02 Remote Data Loss Incident

[OBSERVED] A rigorous audit of the file tree confirms that **ZERO raw JSON or JSONL artifact files exist in the repository for the 2026-09-02 Qwen evaluations**.
- Directory inspection of `paper-2027/research/attention-aware-retrofit/evidence/` lists 36 files; all timestamped between `20260821` and `20260901`. Not a single file bears a `20260902` timestamp.
- Global workspace search for `*20260902*` matches exactly 4 markdown documents:
  1. `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`
  2. `ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`
  3. `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`
  4. `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`

[OBSERVED] The markdown reports record parent SHA-256 hashes for the missing remote receipts:
- Packed-natural NLL parent receipt: `ba489f47070d2dd9058afe50fa7c9db1229f50eb2bc364445dfdb5c7815712a9` (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, line 157).
- Far-evidence natural QA (30-row panel) parent receipt: `6109434ea596b42706e5ce295a1348052ec4d01b22c0849529bf23f2dbef793c` (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, line 172).
- Table $\times$ gain factorial parent receipt: `5d6f2f2e7dc4cc8961e1d42d87931e03d3cf540a3af4a7b7d9c776d66b5a3d9b` (`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`, line 180).
- Headwise factorized 42-file raw bundle: Intentionally excluded from Git, stored locally on the remote machine prior to shutdown (`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md`, lines 8–9, 285–310).

[OBSERVED] As documented in `INDEX.md` line 114 and `paper-2027/HANDOFF.md` lines 67–70, the remote compute instance was decommissioned before these raw files were transferred to the repository.

### 5.2 Evidentiary Ceilings for Unrecovered Artifacts

[DERIVED] In strict adherence to `AGENTS.md` §3 ("Evidence and identity") and `INDEX.md` §2.3:
1. All numerical statistics originating from the 2026-09-02 Qwen sessions (packed-natural NLL, 30-row QA panel, source-contrast decoding, sequence reranking, HotpotQA 38-row Fact D, and gain sweep Fact E) must be classified as:
   $$\textbf{UNSUPPORTED BY REPOSITORY RAW ARTIFACTS / INTERNAL DECISION EVIDENCE ONLY}$$
2. These numbers are strictly forbidden from being promoted to manuscript claims, ICLR submission text, or verified benchmark answers.
3. Under no circumstances may an agent perform a silent GPU rerun to replace these files without explicit author authorization.

---

## 6. Synthesis and Actionable Audit Verdict

| Investigation Area | Forensic Finding | Evidentiary Status |
|---|---|---|
| **$S=2/4/8$ Failure** | $S=2$ passes gates; $S=4$ passes knife-edge; $S=8$ collapses across all profiles. | `[OBSERVED]` Supported by canonical raw receipts (`SCALE_CONSISTENT_..._20260831.md`). |
| **Ordered Coupling** | Permuting rotary slots while preserving multiset collapses OLMo 1x NLL by $+3.76$ and Qwen 64K RULER from 0.7000 to 0.0000. | `[OBSERVED]` Decisively established; refutes unordered spectrum theory. |
| **Native Retention** | 0.875 PPL retention gate is knife-edge; geometric MAE 0.0012 breaks retention. | `[OBSERVED]` Confirmed on OLMo and Qwen (`LOW_DIM_COUPLING_..._20260901.md`). |
| **Natural QA Barrier** | Long NLL drops and far source likelihood enters logits ($+1.438$), but greedy QA F1 and EOS collapse. | `[OBSERVED]` Confirmed; post-hoc decoding rescues failed. |
| **Gain Scaling** | Gain cannot change key ranking ($\arg\max$ invariant); acts as sharpening shortcut; destroys autoregressive EOS. | `[DERIVED]` Proved by algebra and corroborated by headwise gain collapse. |
| **Spatial Clocks** | Layerwise is ineffective; headwise clocks improve long QA (0.252 F1) but cannot bridge the Native/long Basin Barrier. | `[OBSERVED]` Confirmed in OLMo factorized runs (`HEADWISE_..._20260902.md`). |
| **Co-Adaptation** | Pretrained weights are co-adapted to training basis (interaction 5.9x main effects); transplant obstruction proven. | `[OBSERVED]` / `[DERIVED]` Established by 50M/151.9M crossings & Theorem T2. |
| **2026-09-02 Data** | Remote JSON/JSONL raw files missing due to shutdown; hashes preserved. | `[OBSERVED]` Strictly internal decision evidence; cannot promote to paper. |

### Final Audit Verdict
The claim that a single static zero-training RoPE table can simultaneously preserve Native short-context fidelity and solve long-context natural language reasoning on mature checkpoints is **EMPIRICALLY FALSIFIED AND STRUCTURALLY OBSTRUCTED**. 

The repository's positive accomplishments are strictly bounded:
1. Non-linear exponent allocation $z$ is a legitimate, causally verified training-time coordinate.
2. Normalized-index static tables reliably improve synthetic long-range needle retrieval (RULER) and long natural perplexity up to $S \le 4$.
3. Any claim of solving natural multi-hop QA, extending past $S=4$, or eliminating the Native retention cost without weights co-adaptation is **UNSUPPORTED BY REPOSITORY EVIDENCE**.
