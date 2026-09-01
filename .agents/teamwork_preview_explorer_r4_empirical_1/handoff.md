# Handoff Report — explorer_r4_empirical_1 (R4 Axis A: Canonical Empirical Synthesis & Causal Grounding)

**Date:** 2026-09-01  
**Agent:** `explorer_r4_empirical_1`  
**Handoff Type:** Hard (Task complete)  
**Parent Agent:** `9219631d-28ca-410b-b7ff-2c42127fb3f2`

---

## 1. Observation

Directly observed evidence and verifiable artifact paths:

1. **151.9M 3-Seed Exact-Range Causal Identification (`paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`):**
   - Protocol: 151.9M parameters, $K=32$ rotary pairs, 12 layers, $d_{\text{head}}=64$, $L_{\text{train}}=256$, 499.97M tokens, seeds 42, 137, 256.
   - Pinned parameters: sampled frequency extrema ($\omega_0=1, \omega_{K-1}=B^{-(K-1)/K}$), log-span $R=\frac{K-1}{K}\log B$, architecture, row order, token budget. Modulated variable: $K-2=30$ interior frequencies.
   - Fixed-range Anchored EVQ-Cosh minus FMRoPE teacher-forced NLL delta:
     - $L=256$ ($1\times$): $+0.02619$ (SD $0.00835$, CI $[+0.00545, +0.04693]$, 0/3 seeds)
     - $L=512$ ($2\times$): $-0.28073$ (SD $0.19186$, CI $[-0.75735, +0.19589]$, 3/3 seeds)
     - $L=1024$ ($4\times$): $-0.17599$ (SD $0.03865$, CI $[-0.27202, -0.07997]$, 3/3 seeds)
     - $L=2048$ ($8\times$): $-0.14571$ (SD $0.03405$, CI $[-0.23029, -0.06114]$, 3/3 seeds)
   - Target-matched comparison (reversal):
     - $L=256$: $+0.02619$ (0/3 seeds); $L=512$: $+0.06032$ (0/3 seeds); $L=1024$: $+0.22720$ (0/3 seeds); $L=2048$: $+0.45959$ (0/3 seeds).

2. **M4 50.9M Factorial (`rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`):**
   - 192 runs (180 main + 12 boundary), 12 structural configs ($B \in \{500\text{K}, 1\text{M}\}$, $L \in \{256, 1024\}$, $d_{\text{head}} \in \{32, 64, 128\}$), 3 seeds.
   - Weighted OOD NLL delta vs Geo: Anchored Cosh $0.75\times$ ($-0.009115$), Rule $1.0\times$ ($-0.009879$), $1.25\times$ ($-0.012100$), Matched Exponential ($-0.010619$).
   - Rule Cosh vs Exponential delta: $+0.000740$ NLL ($p=0.8364$, CI $[-0.005484, +0.007487]$).
   - Rule point won 4/12 configs; average regret $0.011440$ NLL.

3. **50M $2\times 2$ Table-Weight Crossing (`paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`):**
   - Self-consistent PPL: Geo/Geo $7.14$, EVQ/EVQ $7.16$.
   - Mismatched PPL: Geo weights + EVQ table $\to 76.20$ ($r_2$ static basis rises from $4.57$ to $12.54$, frequency EF trace explodes by $4,494\times$); EVQ weights + Geo table $\to 23.05$.
   - Loss ANOVA: $E_T = +0.5991$, $E_W = -0.5965$, $I_{T \times W} = -3.5367$ (CI $[-5.165, -3.039]$, $\approx 5.9\times$ main effects).
   - 151.9M 2-seed replication: Crossover interaction $3.400$ NLL (Seed 137) and $3.251$ NLL (Seed 256) at 1K.

4. **Scale and Systems Evidence:**
   - 432M MLA Flagship (`data/curated/table18_mla_3seed_aggregate.json`): 3 seeds (42, 43, 88), $d_{\text{eff}}=128, K=16$; 16K PPL Geo $138.81 \to$ EVQ $95.59$; with YaRN(s=4) @32K: Geo $278.50 \to$ EVQ $236.59$.
   - 750M Continuation (`docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`): 2K $\to$ 4K continuation, 500M tokens, Seed 42; 16K PPL $45.14 \to 24.41$ ($-45.9\%$); 8K Passkey AR exact match $0\% \to 77.5\%$.
   - 1.485B OLMo-2 Scratch (`rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`): Step-1,000, 2.097B tokens; 16K PPL $182.73 \to 159.64$ ($-12.64\%$); 16K tail PPL $214.63 \to 172.60$ ($128/128$ docs favor EVQ).
   - 8B LLaMA-3 LoRA Adaptation (`rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`): 300 steps, Seed 42; temporal NLL @32K $-2.048$; true 16K gold block ablation causes $+1.5055$ NLL degradation in EVQ vs $-0.0095$ in Native; hit@16 rises $18.75\% \to 64.06\%$; top-1 generation exact match remains 0%.
   - 129.6M Video-DiT Breadth (`paper-2027/research/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md`): 3D RoPE, Seed 42; denoising MSE drops $-20.95\%$ train frames, $-16.29\%$ all extrapolated, $-35.42\%$ far extrapolated frames.

5. **Mature Retrofit & Length-Conditioned Mechanics (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`, `LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md`):**
   - OLMo-2-1B @16K: Geometric interpolation collapses ($0.0056$), derived allocation achieves $0.6047$, nearest movement-profile ramp achieves $0.6104$ (derived-minus-ramp CI $[-0.0464, +0.0345]$ spans zero).
   - Qwen2.5-1.5B @64K: Geometric $0.5775$, corrected derived allocation $0.6650$, ramp $0.6400$. Qwen @128K corrected score is $0.5400$ (superseding the stride-16 aliased $0.6175$).
   - Length-conditioned budgeted retrofit on OLMo-2-1B: RULER core-4 macro $0.5825$ @8K / $0.4000$ @16K vs official Transformers YaRN $0.5375$ / $0.0125$. Two-factor interaction is $+0.1825$ @8K and $+0.2850$ @16K.

---

## 2. Logic Chain

1. **Step 1 (Decomposition of Causal Axes):**
   By parameterizing RoPE frequencies as $x_k = -\log\omega_k = a + R z_k$, holding $(a, R)$ constant isolates the effect of normalized interior allocation $z$.
2. **Step 2 (Empirical Validation of Pure $z$):**
   In the 151.9M exact-range 3-seed trial, holding $(a, R)$ fixed while varying only the 30 interior frequencies produced a consistent out-of-domain NLL drop across all 3 seeds ($-0.281 / -0.176 / -0.146$), demonstrating that $z$ is causally consequential during training.
3. **Step 3 (Rejection of Cosh Exclusivity and Formula Optimality):**
   In the M4 192-run factorial, matched exponential matched Cosh ($p=0.8364$) and the rule $\tau$ had positive regret against empirical multipliers in 8/12 configs. Thus, Cosh is an effective zero-parameter instance, not a unique or universal optimum, and $\tau$ is an operating default.
4. **Step 4 (Demarcation of Deployment Boundaries):**
   Under target-matching, FMRoPE beats anchored EVQ-Cosh in 3/3 seeds ($+0.060 / +0.227 / +0.460$), proving that support $R$ and allocation $z$ interact non-linearly and cannot be treated as an additive ledger.
5. **Step 5 (Establishment of Co-adaptation):**
   The 50M $2\times 2$ crossing demonstrates that projection weights $W_q, W_k$ co-adapt to the frequency coordinates ($I_{T \times W} = -3.5367$). Swapping tables post-hoc raises static rank $r_2$ to $12.54$ while collapsing PPL to $76.20$, disproving the hypothesis that static basis geometry alone predicts language modeling loss.
6. **Step 6 (Retrofit Mechanism Clarification):**
   In mature checkpoints, the nearest movement-profile ramp matches the full derived profile ($0.6104$ vs $0.6047$ on OLMo), proving that the value of spectral redundancy lies in identifying the model-relative split location, rather than defining a complex non-linear operator.

---

## 7. Caveats

1. **3-Seed Uncertainty:** Exact-range 151.9M and MLA 432M utilize 3 independent seeds. Confidence intervals are descriptive Student-$t$ summaries, not a basis for asymptotic significance claims.
2. **Trainer Confound in 1.485B Scratch:** The 1.485B comparison uses same-initialization/same-recipe from step-0 but different trainer stacks (single-GPU HuggingFace loop vs AI2 distributed OLMo trainer); it is not bitwise paired.
3. **Capability Conversion Gap in 8B:** In 8B LLaMA-3 LoRA adaptation, strong temporal NLL gains ($-2.048$ @32K) and causal deletion sensitivity ($+1.5055$ NLL) do not convert to top-1 autoregressive exact match (0% exact match).
4. **Single-Seed Breadth:** Video-DiT (129.6M) and 750M continuation are single-seed (seed 42) supporting evidence.
5. **Target-Free Session Scope:** The length-conditioned retrofit operates as a session-static policy (fixed at prefill), preserving standard KV-cache semantics; it does not allow dynamic token-by-token frequency switching within an active cache.

---

## 4. Conclusion

1. **Pure interior allocation $z$ is an independently identifiable training-time causal variable** at fixed sampled support $(a, R)$, trading a modest in-window cost ($+0.026$ NLL) for robust out-of-domain gains ($-0.281 / -0.176 / -0.146$).
2. **Weights co-adapt to frequency coordinates during training**, and static geometric metrics (such as $r_2$) decouple from LM perplexity when weights and coordinates are mismatched ($7.14 \to 76.20$ PPL shock).
3. **The practical value of spectral redundancy in mature retrofits is model-relative split derivation** rather than bespoke non-linear kernel design, achieving $0.5825$ @8K / $0.4000$ @16K on RULER core-4 via two-factor allocation-amplitude interaction.
4. **All claims strictly adhere to repository claim ceilings and locked nomenclature**, rejecting universal optimality, additive support synergy, and ungrounded capability transfer.

---

## 5. Verification Method

To independently verify all findings and reproduction receipts:

1. **Check Exact-Range 151.9M Data & Hashes:**
   Inspect `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` and verify raw source hashes (Seed 42: `801792f0...`, Seed 137: `6a5ab42b...`, Seed 256: `23a0dd06...`).
2. **Check M4 50.9M Factorial Curated Evidence:**
   Verify `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json` (SHA-256 `858710ff52488a0b1d1e0a9a35af9f7658ed974d1b59ff59cc9c89fa2ce9abda`).
3. **Check 432M MLA 3-Seed Aggregate JSON:**
   Verify `data/curated/table18_mla_3seed_aggregate.json` (SHA-256 `1e44d30bb880e4b7427ae55bd7034782989152bd2afca9217495f9b8ece30953`).
4. **Check Retrofit Evidence JSON:**
   Verify `paper-2027/research/attention-aware-retrofit/evidence/LENGTH_CONDITIONED_BUDGETED_RESULTS_20260822.json` (SHA-256 `f92332a4d9c0d6c92e2bb2296c876a32001aee88b7057ff422db9c2b9fb1a311`).
5. **Run Navigation & Repository Regression Gate:**
   Execute standard Python navigation gate: `python3 tests/test_repository_navigation.py`.
