# Auditor 5/8 — Anti-Overclaim Scan

**Auditor**: 5 of 8 (parallel deep audit)
**Working dir**: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`
**Handover read**: `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md` (evidence tiers loaded)
**Constraint**: Do NOT modify paper; do NOT lint prose. Find REAL overclaim only.

---

## Methodology

1. Case-insensitive grep across `paper/main.tex`, `paper/sections/`, `paper/appendix/`, `paper/tables/` for the following inflections:
   `validate|prove|confirm|demonstrate|establish|show|guarantee|exact|robust|universal|always|never|significant|substantial|outperform|surpass|exceed|remarkably|dramatic|clearly|definitively`.
2. For each hit, classified by:
   - **Tier**: I/II/III/Theory-conditional/Theory-derivation (per handover map).
   - **Tag**: ✅ proportional / ⚠️ borderline / ❌ overclaim.
   - **Replacement**: only proposed for ⚠️ and ❌.
3. Special attention to: Abstract, Contributions §1, Captions, Theorem statements, Checklist Q1/Q3.
4. Filtered out structural hits ("Section X shows", "Table Y shows") and the LaTeX `Proof:` environment.
5. Cross-rung ladder coherence check.

Evidence-tier reference (from handover):
- **Tier I (3-seed Primary)**: EVQ×YaRN 454M / PE-dominant 125M&454M / MLA 432M 500M-tokens.
- **Tier II (Robustness 3-seed)**: FineWeb-Edu consistency, base sweep, capability preservation.
- **Tier III (Supporting 1-2 seed)**: video DiT, LoRA-on-LLaMA-3-8B, progressive training, MLA 1B-tokens, 750M continue.
- **Theory-conditional**: under broadband surrogate / diffuse-baseline assumption.
- **Theory-derivation (semi-analytic)**: τ* law (d_head, L^{-1/2}; λ calibrated).

---

## Findings table

Format: `file:line | quoted phrase | tier | tag | suggested replacement`.

| # | file:line | Quoted phrase (trimmed) | Tier | Tag | Replacement / note |
|---|---|---|---|---|---|
| F1 | paper/main.tex:46 | "Three primary stress tests anchor our evidence … EVQ improves over geometric RoPE and a learned positional operator" | I (PE-dominant 3-seed) | ✅ | "improves over … on the tested DAPE-style protocol" — caveat already implicit in the section §5.2; abstract level acceptable. |
| F2 | paper/main.tex:46 | "EVQ alone reduces 2× extrapolation PPL by 31.1% at +1.1% in-distribution cost, **surpassing** Geo+YaRN" | I (MLA primary) | ⚠️ | Recommend "surpassing **matched-scale (s=4)** Geo+YaRN" — body text and Table 5 already qualify with `s=4`; abstract drops the qualifier. P1. |
| F3 | paper/main.tex:46 | "These results **support** training-time allocation shape … as a first-order RoPE design axis." | I aggregated | ✅ | proportional ("support" not "prove"). |
| F4 | paper/main.tex:80 | "an empirically **validated** O(1) scale" (Checklist Q1) | Theory-derivation | ⚠️ | "empirically calibrated" is more accurate — λ=1 is by convention falling inside basin (a posteriori check), not "validated" in the strong sense. P1. |
| F5 | paper/main.tex:94 | "Theorems 1-2 and Proposition 1 are **exact** conditional on the broadband surrogate" | Theory-conditional | ✅ | algebra-exact; conditional clause is in place. |
| F6 | paper/main.tex:94 | "The surrogate itself is **validated** in Appendix" | Theory-conditional | ⚠️ | "supported" or "checked functionally" — surrogate is functional (not pointwise) check; the word "validated" overlaps with frequentist "validated against held-out data". Body §3.2 already labels it "functionally" and limitations call it "main approximation". P2 (close to lint, but flagged because Q3 justification is high-visibility). |
| F7 | paper/main.tex:122 | "Primary claims are anchored in multi-seed evidence and single-seed runs are **clearly** labeled as supporting only." (Q7) | meta | ✅ | supports the labeling rather than the result; OK. |
| F8 | paper/main.tex:70 | "we do not present EVQ as a **universal guarantee** of downstream improvement" (Broader Impact) | meta | ✅ | explicit negation of universal — proportional. |
| F9 | paper/sections/01_intro.tex:7 | "deployable operating rule … used as a **robust** default inside an empirically flat basin" | Theory-derivation | ⚠️ | "robust" applied to the τ rule, not labeled as a Robustness-tier experiment. Replace with "stable" or "consistent across the tested grid". P2 (defensible since "robust" here means "low sensitivity to grid", and §3.7 already calls basin flat). |
| F10 | paper/sections/01_intro.tex:15 | "τ=d_head/√L is used as a **robust** default inside an empirically flat basin rather than a claimed global optimum." (Contributions #3) | Theory-derivation | ⚠️ | Same as F9. Contribution-list visibility means borderline. Proposed "stable / consistent default". P1 (high-visibility). |
| F11 | paper/sections/03_theory.tex:15 | "the surrogate is functional, not pointwise … 24-92% collision-score reduction across 12 configs" | Theory-conditional | ✅ | accurate; "functional" qualifier in place. |
| F12 | paper/sections/03_theory.tex:32 | "We **validate** C_app **functionally**, not pointwise" | Theory-conditional | ✅ | "functionally" qualifier in place; OK. |
| F13 | paper/sections/03_theory.tex:39 | "Theorem [**Exact** stationary allocation under the broadband surrogate]" | Theory-conditional | ✅ | algebra-exact under stated surrogate; clause "under the broadband surrogate" present. |
| F14 | paper/sections/03_theory.tex:81 | "the endpoint/midpoint choice shifts PPL by <1% across all K≥16 settings we tested" | I | ✅ | tested-grid clause in place. |
| F15 | paper/sections/03_theory.tex:113 | "numerical optimization yields an exponent within 1% PPL of L^{-0.500}" | Theory-derivation | ✅ | qualifier "within 1% PPL" present. |
| F16 | paper/sections/03_theory.tex:115 | "the formula falls **within <1% PPL** of the best observed sweep points across 27 configurations; pure geometric (τ=0) … falls 10-46% outside" | I+II+III aggregated | ✅ | numerical bound + tested grid scope present. |
| F17 | paper/sections/05_experiments.tex:23 | "EVQ+YaRN reaches 100% passkey retrieval at 8K while Geo+YaRN remains at 61%, with the advantage persisting at 12/16K" | I (3-seed) | ✅ | seeds-on-record and "matched scale" qualifier in following sentence. |
| F18 | paper/sections/05_experiments.tex:23 | "We read the result as EVQ increasing YaRN's leverage **under matched scale**; we do not claim dominance over every tuned-scale YaRN baseline." | I | ✅ | explicit anti-overclaim hedge — model citizen. |
| F19 | paper/sections/05_experiments.tex:28 | (Caption) "Their combination under matched YaRN scale **substantially outperforms** Geo+YaRN" | I (MHA 454M, 3-seed) | ✅ | numerical magnitudes in Table 2 (PPL 70.9 vs 82.9, 61% → 100%) justify "substantially"; matched-scale clause present. |
| F20 | paper/sections/05_experiments.tex:41 | "A complementary τ-sweep at L_train=256 (Appendix Fig. 10) **confirms** that the formula's prediction τ*=4.0 is favored among tested settings" | I (3-seed PE-dominant 125M & 454M) | ⚠️ | "confirms" is borderline because the τ-sweep is a check of one prediction inside the tested range, not an independent confirmation of the law structure. Replace with "is consistent with" or "supports". P2. |
| F21 | paper/sections/05_experiments.tex:51 | "EVQ reduces 2× extrapolation PPL from 138.8 to 95.6 (-31.1%) … **exceeding** Geo+YaRN at matched YaRN scale s=4 (117.9)" | I (MLA 3-seed) | ✅ | "matched s=4" qualifier present and numerical, accurate. |
| F22 | paper/sections/05_experiments.tex:53 | (1B-token §) "the YaRN composition boost **grows** from +8.6 pp at 500M … to +13.6 pp at 1B (EVQ+YaRN+FT, -2.5% vs. Geo+YaRN+FT)" | III (single-seed 1B-token) | ✅ | single-seed labeled "Preliminary single-seed evidence"; tier-III caveat in place. |
| F23 | paper/sections/05_experiments.tex:61 | "in-range PPL stays within ±1.7% while long-range improves 10-46%; capability checks (Table 3) **show** EVQ is non-destructive." | II (capability 3-seed) | ✅ | empirical "show" is OK on 3-seed in-tier evidence; range-bracketed. |
| F24 | paper/sections/05_experiments.tex:61 | "raw PPL -46% 1-seed / -13% 3-seed" | I+III | ✅ | seed-count explicit. |
| F25 | paper/appendix/a1_proofs.tex:6 | "This subsection **proves** the main-text Theorem 1" | Theory-conditional | ✅ | algebra-proof of theorem under stated surrogate; proper use. |
| F26 | paper/appendix/a1_proofs.tex:47 | "Conditional on C_app, this derivation is **exact**." | Theory-conditional | ✅ | conditional clause present. |
| F27 | paper/appendix/a1_proofs.tex:87 | "which is **exactly** the uniform quantile map" | Theory-conditional | ✅ | algebra-exact; OK. |
| F28 | paper/appendix/a1_proofs.tex:89 | "for u_0=0, one has φ_0(τ)=0 **exactly**, so the leading inverse frequency remains b^0=1" | Theory-conditional | ✅ | algebra-exact identity, OK. |
| F29 | paper/appendix/a1_proofs.tex:99 | (subsection title) "Surrogate quality: **functional validation**" | Theory-conditional | ✅ | "functional" in place. |
| F30 | paper/appendix/a1_proofs.tex:102 | "The 24-92% reduction reported below also serves as the operational forced-branch residual diagnostic" | Theory-conditional | ✅ | numerical and tied to surrogate level. |
| F31 | paper/appendix/a1_proofs.tex:107 | (Caption) "Functional surrogate validation across 12 configurations. EVQ … reduces collision and increases effective rank under the **exact** kernel K **in every tested regime**" | Theory-conditional | ✅ | algebra/exact-kernel reduction; "tested regime" qualifier. |
| F32 | paper/appendix/a1_proofs.tex:138 | "EVQ reduces collision under the exact kernel **in all 12 configurations** (-24% to -92%)" | Theory-conditional | ✅ | tested grid scope present. |
| F33 | paper/appendix/a1_proofs.tex:140 | (subsection title) "**Exact** higher-order displacement floor" | Theory-conditional | ✅ | algebra-exact Taylor series; OK. |
| F34 | paper/appendix/a1_proofs.tex:143 | "an **exact** Taylor series in 1/K that is fully closed-form (no empirical calibration)" | Theory-conditional | ✅ | algebra-exact, verified symbolically; OK. |
| F35 | paper/appendix/a1_proofs.tex:153 | "coefficients (-1/16, +1/256, +17/30720) are **exact** rationals" | Theory-conditional | ✅ | OK. |
| F36 | paper/appendix/a1_proofs.tex:202 | "three **exact** identities" | Theory-conditional | ✅ | OK. |
| F37 | paper/appendix/a1_proofs.tex:247 | "leave-one-out cross-validation gives max prediction error <0.4%" | Theory-derivation | ✅ | numerical and bounded. |
| F38 | paper/appendix/a1_proofs.tex:284 | "is **supported** across 99 trained models spanning 27 validation settings" | I+II+III aggregated | ✅ | "supported" not "validated"/"proven"; appropriate. |
| F39 | paper/appendix/a1_proofs.tex:321 | (Caption) "Scaling-law sweep over 99 trained models across 27 validation settings" | mixed (Primary + Supporting) | ✅ | "Primary claims rely only on the 3-seed primary settings" caveat present in caption. |
| F40 | paper/appendix/a1_proofs.tex:401 | "Numerical verification under this phenomenological model **shows** the transition is apparent-sharp" | Phenomenological model + III | ✅ | "phenomenological model"/"apparent-sharp" qualifier present; "we do not claim an exact first-principles derivation" follows. |
| F41 | paper/appendix/a1_proofs.tex:438 | "EVQ and the oracle search **substantially** reduce E_off relative to geometric in all tested configurations" | Theory-derivation | ✅ | numerically verified in Table 13. |
| F42 | paper/appendix/a3_supporting_results.tex:13 | (Caption) "MLA **validation** (432M, d_rope=32, 3-seed mean ± std). EVQ alone **outperforms** GEO+YaRN(s=4) at 16K" | I (3-seed) | ✅ | 3-seed primary, matched-scale stated explicitly; "validation" used in benign sense. |
| F43 | paper/appendix/a3_supporting_results.tex:29 | "Under matched YaRN scale in this setup, a training-time allocation change is thus comparable to or better than the inference-time rescaling baseline we tested, but **we do not claim superiority over every possible inference-time method**." | I | ✅ | exemplary anti-overclaim hedge. |
| F44 | paper/appendix/a3_supporting_results.tex:31 | "Training-progression analysis **shows** EVQ's advantage is visible from early training" | I | ✅ | 3-seed; OK. |
| F45 | paper/appendix/a3_supporting_results.tex:55 | (Caption, fig:attn-mechanism) "Waterbed verification in learned attention patterns (**750M**)" | III (750M single-seed) | ⚠️ | The fig label says "verification" but the underlying 750M run is single-seed. Replace "verification" → "illustration" or "single-seed visualization". P1 (high-visibility caption + verification verb on tier-III evidence). |
| F46 | paper/appendix/a3_supporting_results.tex:62 | (Caption, frequency dynamics) "750M frequency dynamics and retrieval divergence (**single seed**)" | III | ✅ | single-seed flagged in-caption. |
| F47 | paper/appendix/a3_supporting_results.tex:69 | "gold-answer NLL **separates** the two RoPE variants **clearly**" | III (single-seed QuALITY) | ⚠️ | "clearly" + single-seed = borderline. Numerical signal (-30.1%, -21.4%) is large but n=1 seed. Replace with "separates the two RoPE variants" (drop "clearly"). P1. |
| F48 | paper/appendix/a3_supporting_results.tex:103 | "EVQ **consistently** trades a small in-range cost for large long-range gains. The pattern is consistent across model scales: long-range gains are **substantial** while in-range cost remains bounded." | mixed I+III (50M-750M aggregated) | ⚠️ | "Consistently/substantial" applied across a mixed-tier aggregation including single-seed 50M, 125M, 750M rows. Replace with "in the tested grid (50M-750M, ±1.7% in-range, 10-46% long-range)" — i.e. quote the bracketed range used elsewhere. P1. |
| F49 | paper/appendix/a3_supporting_results.tex:108 | (Caption fig:waterbed-multiscale) "Waterbed **verification** across model scales … pattern is consistent from 50M to 750M **within the tested grid**, consistent with the theoretical prediction." | mixed I+III | ⚠️ | "verification" is strong; the underlying rows include single-seed 50M, 125M, 750M. Replace "verification" → "illustration" or "evidence". The "within the tested grid" qualifier is present, partially mitigating. P1 (caption visibility). |
| F50 | paper/appendix/a2_experiment_details.tex:153 | (Caption tab:dit-base1000) "Dead-channel **validation**: base=1,000 DiT head-to-head (129.6M, **seed 42**, YaRN). The sharp phase transition between τ=1.2 and τ=1.5 observed at base=10,000 **vanishes entirely**" | III (single-seed video DiT) | ❌ | Two issues: (i) "validation" verb on a single-seed (seed 42) DiT result; (ii) "vanishes entirely" is an absolute empirical claim from one seed. Replace with: "Dead-channel mechanism check: … the sharp phase transition between τ=1.2 and τ=1.5 observed at base=10,000 is no longer present in this seed". **P1**. |
| F51 | paper/appendix/a2_experiment_details.tex:194 | (Heading) "Base sweep: EVQ **robustness** across the base spectrum." | III (1-2 seed video DiT) | ⚠️ | "robustness" used in tier-III video-DiT setting, not a Robustness-tier 3-seed experiment per Table evidence_tier. Replace with "Base-sensitivity check" or "EVQ stability across the base spectrum (1-2 seed)". Body line 218 already says "**indicating robustness to the base hyperparameter within the tested grid**" which is appropriately hedged. P2. |
| F52 | paper/appendix/a2_experiment_details.tex:198 | (Caption tab:dit-base-sweep) "EVQ is **remarkably stable** across base≥500" | III | ⚠️ | "Remarkably" is emotive/superlative on tier-III evidence (1-2 seed). Replace with "EVQ is stable across base≥500 in this 1-2 seed setting (1.9× MSE range vs. GEO's 12×)". P2 (close to lint, but emotive intensifier on tier III). |
| F53 | paper/appendix/a2_experiment_details.tex:218 | "indicating **robustness** to the base hyperparameter **within the tested grid**" | III | ✅ | "within the tested grid" qualifier in place — model citizen. |
| F54 | paper/appendix/a4_supporting_experiments.tex:36 | (Caption tab:lora-8b) "EVQ trades a modest in-distribution cost for **dramatic** extrapolation gains." | III (single-seed LoRA-8B) | ⚠️ | "Dramatic" on single-seed (n=1) LoRA experiment. The numerical 8×, 19× is notable, but emotive on tier-III. Replace with "large extrapolation gains (8× at 16K, 19× at 32K)" — let the numbers speak. P1. |
| F55 | paper/appendix/a4_supporting_experiments.tex:55 | "After Stage 2, EVQ raw … **surpasses** EVQ+YaRN at Stage 1 … This is a **single-seed observation**; multi-seed replication is in progress." | III | ✅ | single-seed flagged in same paragraph; OK. |
| F56 | paper/appendix/a4_supporting_experiments.tex:57 | (Heading) "Cross-modal **confirmation**: video DiT (supporting)." | III (1-2 seed video DiT) | ❌ | "Confirmation" is a strong empirical word applied to a tier-III (1-2 seed) video DiT result. The body of this paragraph already says "supporting mechanism check rather than a standalone video contribution". The heading verb contradicts the qualified body. Replace heading "Cross-modal confirmation" → "Cross-modal mechanism check" or "Cross-modal supporting evidence". **P1** (heading visibility). |
| F57 | paper/appendix/a4_supporting_experiments.tex:64 | "Table 3 **shows** that EVQ is non-destructive when retrieval is absent from training and improves extrapolated retrieval by +10-13 pp" | II (capability 3-seed) | ✅ | 3-seed Robustness tier; appropriate "shows". |
| F58 | paper/appendix/a4_supporting_experiments.tex:9 | (Caption fig:pe-dominant) "Left: EVQ's closed-form allocation … closes more of the 128→8K extrapolation gap than the learned positional-operator baseline." | I (3-seed) | ✅ | tested-protocol caveat in body §5.2. |
| F59 | paper/appendix/a4_supporting_experiments.tex:14 | (Caption tab:pe-yarn-l256) "The predicted τ*=4.0 is the best of the tested settings, and EVQ4+YaRN **outperforms** Geo+YaRN at every extrapolation ratio." | I (3-seed PE-dominant) | ✅ | "of the tested settings" qualifier in place; per-ratio numerical claim. |
| F60 | paper/tables/table_evidence_tier.tex:6 | (Caption) "Primary claims rest only on 3-seed primary tier; robustness rows are 3-seed but not primary anchors." | meta | ✅ | exemplary tier discipline. |
| F61 | paper/tables/table1_multiscale_raw_ppl.tex:2 | (Caption) "Positive in-range deltas remain small, while long-range gains are **consistent across scales**." | mixed I+III (1, 3-seed rows) | ✅ | the "consistent across scales" framing is hedged; row-by-row seed counts visible in table. |
| F62 | paper/tables/table_lambda_cv.tex:2 | (Caption) "**supporting** surrogate-exact curvature agreement at the 2% level" | Theory-derivation | ✅ | "supporting" not "proving"; numerical bound. |

---

## Caption-specific findings (separate)

Captions are high-visibility. Re-summarizing the caption-specific tags from above:

| Caption file:line | Verb / phrase | Tier | Tag | Severity |
|---|---|---|---|---|
| `paper/sections/05_experiments.tex:28` (fig:evq-yarn) | "substantially outperforms" | I (3-seed) | ✅ | — |
| `paper/appendix/a1_proofs.tex:107` (tab:surrogate-validation) | "Functional surrogate validation" | Theory-cond. | ✅ | — |
| `paper/appendix/a1_proofs.tex:273` (fig:orthogonal) | "EVQ and YaRN correct orthogonal deficiencies" | I | ✅ | — |
| `paper/appendix/a1_proofs.tex:321` (fig:tau-sweep) | "Scaling-law sweep over 99 trained models across 27 validation settings" + Primary-only caveat | mixed I+II+III | ✅ | — |
| `paper/appendix/a3_supporting_results.tex:13` (tab:mla) | "MLA validation (432M, d_rope=32, 3-seed mean ± std)" | I | ✅ | — |
| `paper/appendix/a3_supporting_results.tex:55` (fig:attn-mechanism) | **"Waterbed verification in learned attention patterns (750M)"** | III | ⚠️ | **P1** |
| `paper/appendix/a3_supporting_results.tex:62` (frequency-dynamics) | "750M frequency dynamics and retrieval divergence (single seed)" | III | ✅ | — |
| `paper/appendix/a3_supporting_results.tex:108` (fig:waterbed-multiscale) | **"Waterbed verification across model scales … 50M to 750M within the tested grid"** | mixed | ⚠️ | **P1** |
| `paper/appendix/a2_experiment_details.tex:124` (tab:dit-h2h) | "EVQ wins all metrics across both seeds" | III (2 seeds) | ✅ | seed count explicit |
| `paper/appendix/a2_experiment_details.tex:153` (tab:dit-base1000) | **"Dead-channel validation … vanishes entirely"** (seed 42 only) | III | ❌ | **P1** |
| `paper/appendix/a2_experiment_details.tex:174` (tab:dead-channels) | "Video models waste 32-50%; the text reference (LLaMA-2) has zero dead channels." | architecture audit | ✅ | counted from configs, not trained results |
| `paper/appendix/a2_experiment_details.tex:198` (tab:dit-base-sweep) | "EVQ is **remarkably stable** across base≥500" | III | ⚠️ | P2 |
| `paper/appendix/a4_supporting_experiments.tex:9` (fig:pe-dominant) | "closes more of the 128→8K extrapolation gap" | I | ✅ | — |
| `paper/appendix/a4_supporting_experiments.tex:36` (tab:lora-8b) | **"trades a modest in-distribution cost for dramatic extrapolation gains"** | III | ⚠️ | **P1** |
| `paper/tables/table4_pe_dominant.tex:2` (tab:pe-dominant) | "closes more of the extrapolation gap than the learned positional-operator baseline" | I | ✅ | — |
| `paper/tables/table5_phase11_leverage.tex:2` (tab:phase11-leverage) | "NTK-aware scaling is not complementary at large τ" | I | ✅ | — |
| `paper/tables/table_evidence_tier.tex:6` | "Primary claims rest only on 3-seed primary tier" | meta | ✅ | exemplar |

**Net caption verdict**: 4 captions mismatch verb-strength to tier (P1). The other ~13 captions are well-disciplined.

---

## Ladder-coherence check (Abstract ↔ §1 contributions ↔ §3.7 Proposition ↔ §5 ↔ Checklist Q1)

**Rung 1 — Abstract (main.tex:46)**
- Claim: "the d_head factor and L^{-1/2} exponent are predicted under a small-τ softmax-transport model with χ² channel-load stiffness, and the remaining O(1) normalization is fixed by convention inside an empirically flat PPL basin."
- Strength: derived structure + convention-fixed scale + empirical basin.
- Match with rung 2/3: ✅ identical phrasing.

**Rung 2 — §1 Contributions list (01_intro.tex:15)**
- Claim: "structural scaling τ*∝d_head/√L with convention-dependent prefactor; τ=d_head/√L is used as a robust default inside an empirically flat basin rather than a claimed global optimum."
- Strength: identical to rung 1; explicit "not a claimed global optimum" hedge.
- Match: ✅. Lone friction: "robust default" — the word "robust" carries a slightly stronger connotation than "stable / consistent" (see F10).

**Rung 3 — §3.7 Proposition 1 (03_theory.tex:95-109)**
- Claim: small-τ expansions, then "τ*² = 45 λ Q_1(L,b)·d_head²/L, τ*∝d_head/√L. The d_head factor and L^{-1/2} exponent follow from the combined small-τ scalings; … The remaining O(1) prefactor (c_S, λ, Q_1) depends on normalization convention and is fixed empirically inside a flat basin, **not claimed as a global optimum**."
- Strength: lifts to derivation + explicit caveat. Theory-conditional under diffuse-baseline assumption (stated in proposition).
- Match: ✅. Strong, well-disciplined.

**Rung 4 — §3.7 Three epistemic tiers paragraph (03_theory.tex:113-115)**
- Claim: "the formula falls within <1% PPL of the best observed sweep points across 27 configurations; pure geometric (τ=0) … falls 10-46% outside."
- Strength: numerical + tested-grid scope.
- Match: ✅.

**Rung 5 — §5.2 PE-dominant body (05_experiments.tex:41)**
- Claim: "complementary τ-sweep at L_train=256 confirms that the formula's prediction τ*=d_head/√L=4.0 is favored among tested settings at both 125M and 454M."
- Strength: tier-I 3-seed; "favored among tested settings" qualifier.
- Match: ✅. The verb "confirms" (F20) is borderline since the τ-sweep is a check inside the prediction's domain rather than an independent confirmation; recommend softening to "is consistent with" or "supports".

**Rung 6 — Checklist Q1 (main.tex:80)**
- Claim: "(2) demonstrating that a semi-analytic rule —with derived d_head and L^{-1/2} structure and an empirically validated O(1) scale — falls within the best observed sweep basin across the tested grid of 27 configurations (primary claims rely on 3-seed anchors; supporting 1-2-seed rows extend scope), and (3) the matched-scale systems result that EVQ increases YaRN leverage under a fixed YaRN scale (we do not claim dominance over every tuned YaRN baseline)."
- Strength: tier-aware language + explicit hedging.
- Match: ⚠️ minor friction — "empirically validated O(1) scale" (F4). The O(1) scale is calibrated by setting λ=1 and confirming a posteriori that c_pred lands inside the basin. "Validated" is slightly stronger than the actual epistemic position. Replace with "empirically calibrated" or "empirically anchored".

**Rung 7 — Checklist Q3 (main.tex:94)**
- Claim: "Theorems 1-2 and Proposition 1 are exact conditional on the broadband surrogate; the surrogate itself is validated in Appendix."
- Strength: conditional clause in place.
- Match: ⚠️ same friction as F6 — "validated" overstates the functional check. Suggest "supported by a functional check (24-92% collision-score reduction)".

**Ladder verdict**: The ladder is mostly coherent. Two rungs (Q1 and Q3) use "validated" where the actual evidentiary status is "empirically calibrated / functionally checked". The contributions list (rung 2) and PE-dominant section verb "confirms" (rung 5) are also slightly stronger than warranted, but each is partially mitigated by adjacent qualifiers. **No rung claims more than its upstream support in a way that would mislead the reviewer**, but the four "validate(d)" instances and one "confirms" weaken the otherwise excellent epistemic discipline.

---

## Summary

### Severity counts

| Severity | Count |
|---|---|
| ❌ P0 (definitively unsupported) | 0 |
| ❌ P1 (strong language on tier-III / theory-conditional w/o caveat) | 9 |
| ⚠️ P2 (borderline; defensible but suggested softer wording) | 4 |
| ✅ proportional | ~50 |

### Top P1 findings (what to fix in rebuttal/camera-ready, ranked)

1. **F50** `paper/appendix/a2_experiment_details.tex:153` — Caption "Dead-channel **validation** … phase transition … **vanishes entirely**" on **single-seed** (seed 42) DiT. Two strong words on tier-III. **Fix**: "Dead-channel mechanism check … the phase transition is no longer present in this seed".
2. **F56** `paper/appendix/a4_supporting_experiments.tex:57` — Heading "Cross-modal **confirmation**: video DiT (supporting)" on 1-2 seed video DiT. Heading verb "confirmation" contradicts body's own "supporting mechanism check rather than a standalone video contribution". **Fix**: heading → "Cross-modal mechanism check" or "Cross-modal supporting evidence".
3. **F45 / F49** `paper/appendix/a3_supporting_results.tex:55, 108` — Two captions use "Waterbed **verification**" but underlying rows include single-seed 50M, 125M, 750M evidence. **Fix**: replace "verification" → "illustration" or "evidence".
4. **F54** `paper/appendix/a4_supporting_experiments.tex:36` — Caption tab:lora-8b: "**dramatic** extrapolation gains" on single-seed LoRA. **Fix**: drop "dramatic"; let numbers (8×, 19×) speak.
5. **F2** `paper/main.tex:46` — Abstract "**surpassing** Geo+YaRN" without "matched-scale" qualifier (qualifier present in body §5.4 and Table tab:mla). **Fix**: "surpassing matched-scale (s=4) Geo+YaRN".
6. **F4** `paper/main.tex:80` — Checklist Q1 "empirically **validated** O(1) scale". **Fix**: "empirically calibrated O(1) scale".
7. **F6** `paper/main.tex:94` — Checklist Q3 "the surrogate itself is **validated** in Appendix". **Fix**: "the surrogate is supported by a functional check (24-92% collision-score reduction)" — or keep "validated functionally" with the "functional" word repeated, since main text already uses that qualifier.
8. **F47** `paper/appendix/a3_supporting_results.tex:69` — Body "gold-answer NLL **separates** the two RoPE variants **clearly**" on single-seed. **Fix**: drop "clearly".
9. **F48** `paper/appendix/a3_supporting_results.tex:103` — "EVQ **consistently** trades a small in-range cost for large long-range gains. … **substantial** while in-range cost remains bounded" mixes 1- and 3-seed rows. **Fix**: replace with the bracketed empirical range "(±1.7% in-range, 10-46% long-range across the tested grid)".

### Top P2 findings (defensible; lint-adjacent)

10. **F9 / F10** `paper/sections/01_intro.tex:7, 15` — "robust default" applied to the τ rule (not a Robustness-tier experiment). Defensible since "robust" here means low-sensitivity-to-grid; basin flatness is documented. Minor.
11. **F20** `paper/sections/05_experiments.tex:41` — "**confirms** that … τ*=4.0 is favored among tested settings" — soften to "is consistent with".
12. **F51 / F52** `paper/appendix/a2_experiment_details.tex:194, 198` — "EVQ **robustness** across the base spectrum" / "**remarkably stable**" — body line 218 already softens to "within the tested grid"; caption headers can match.

### What is well-disciplined (do not change)

- §3 theorems ("exact conditional on the broadband surrogate") — algebra-exact, conditional clauses present (F5/F13/F25-F36).
- §5.2 PE-dominant body's "matched-scale" hedging (F18) and §A1 MLA caveat "we do not claim superiority over every possible inference-time method" (F43).
- Broader Impact's explicit "we do not present EVQ as a universal guarantee" (F8).
- Evidence-tier table caption "Primary claims rest only on 3-seed primary tier; robustness rows are 3-seed but not primary anchors" (F60) — exemplar.
- The 99-run sweep figure caption's "Primary claims rely only on the 3-seed primary settings; 1-2-seed rows extend the tested grid but do not anchor the main empirical claims" (F39) — exemplar.

### Bottom-line

The paper's **epistemic discipline is generally strong**: zero P0 overclaims, only nine P1 occurrences, and most of those are caption verbs ("validation", "verification", "confirmation") attached to tier-III evidence whose body text already provides the right hedge. The ladder (Abstract ↔ §1 contributions ↔ §3.7 Proposition ↔ §5 ↔ Checklist) is coherent: no rung claims more than its upstream support in a substantively misleading way. The small set of word-choice misalignments (eight captions/headings + two checklist "validated" instances + one "confirms" verb) is fully addressable in a single editing pass without affecting content.
