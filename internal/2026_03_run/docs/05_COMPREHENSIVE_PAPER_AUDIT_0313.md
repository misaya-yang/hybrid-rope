# EVQ-Cosh Paper: Comprehensive Readiness Audit
**Date**: 2026-03-12 | **Document**: mainstory.md v21
**Target Venue**: NeurIPS 2026 | **Submission Mode**: Standard (9 pages + appendix + checklist)

---

## EXECUTIVE SUMMARY

The paper is **substantially ready with identifiable gaps**. The theoretical framework is rigorous and novel. The experimental evidence spans five scales with multi-seed validation where critical. However, three category of gaps exist:

1. **THEORY**: Core derivation is complete; supporting claims need precision on boundaries
2. **EXPERIMENTS**: Main empirical line is strong; single-seed results dominate Claim 4 (progressive training)
3. **FIGURES/TABLES**: 11/13 planned figures created; two critical figures (Fig 4, Fig 7) need completion

**Recommendation**: Paper is at **strong accept / spotlight contender** readiness. Fix identified gaps before camera-ready.

---

# SECTION 1: THEORY AUDIT

## 1.1 Theoretical Claims Inventory

| Claim ID | Claim | Mathematical Status | Proof Completeness | Evidence |
|----------|-------|---------------------|-------------------|----------|
| **T1** | RoPE allocation is a variational inverse problem | **Exact Theorem** | ✅ COMPLETE | Phase-collision kernel formulation (§4.1) |
| **T2** | Phase-collision kernel admits broadband projection K ≈ αI + βA⁻¹ | **Exact Lemma + Modeling Convention** | ⚠️ PARTIAL | 24K-config sweep shows R²>0.99 in regime; boundary conditions stated; residual sources identified |
| **T3** | Solution family: φₖ(τ) = 1 - (1/τ)arcsinh((1-uₖ)sinhτ) | **Exact after T2** | ✅ COMPLETE | Euler-Lagrange ODE → CDF inversion (§4.1 steps 4-6) |
| **T4** | Geometric RoPE is τ=0 boundary case | **Exact Corollary** | ✅ COMPLETE | sinh(τ)→τ as τ→0 implies arcsinh→x, φₖ→uₖ |
| **T5** | Waterbed inequality: ∫ln E(φ) dφ ≥ ln b - ln c | **Exact Proposition under explicit assumptions** | ⚠️ STATED NOT PROVEN | The bound is stated in §4.4; formal proof deferred or incomplete in document |
| **T6** | Sub-cycle fraction reduction: Δρ = x - sinh(τx)/sinh(τ) > 0 | **Exact Proposition** | ✅ COMPLETE | Proof by strict convexity of sinh (§4.5); three corollaries proved |
| **T7** | τ* = d_head / √L scaling law | **Empirical Law / Conjecture** | ⚠️ EMPIRICAL ONLY | 99-run validation (27 configs × 3 seeds); no theoretical derivation; should be labeled "Empirical conjecture" not "scaling law theorem" |
| **T8** | Fisher information → attention utility bridge | **Asymptotic / Local Heuristic** | ⚠️ NOT RIGOROUS | Provides interpretive intuition only; acknowledged as local-curvature approximation (§4.4, footnote) |
| **T9** | Progressive training superlinearly amplifies EVQ | **Empirical Pattern** | ⚠️ EMPIRICAL | 3-stage: 34.6% → 52.0% → 81.2% (Claim 4); pattern is consistent but partially single-seed |

### 1.1.1 Mathematical Status Summary

**Rigorous Proofs (Ready for Theorem Statements)**:
- T1: Variational formulation ✅
- T3: Solution via Euler-Lagrange ✅
- T4: Geometric as τ=0 limit ✅
- T6: Sub-cycle monotone decrease ✅

**Approximations with Validation (Need Explicit Boundary Statements)**:
- T2: Broadband projection
  - **Current State**: Validated R² > 0.99 under D(Δ)∝1/Δ, base∈[8K,100K], L≥4096
  - **Issue**: Boundary of validity is stated in text but not formalized in theorem statement
  - **Fix Needed**: Write as "Lemma 2.1 (Broadband Projection): Under conditions {D, base, L}, the projection achieves R² > 0.99 in the mid-band"
  - **Impact**: This IS the key assumption; must be crystal clear

**Heuristic / Empirical (Should Not Be Called Theorems)**:
- T7: Fisher bridge — relabel as "Remark" or "Intuitive interpretation"
- T8: τ* scaling law — currently justified empirically, not derived

### 1.1.2 Missing or Informal Derivations

**GAP 1: Waterbed Inequality Proof**
- **Claim**: §4.4 states the waterbed as a "inequality"
- **Current**: Described as "Jensen equality" and "bounded cost" but full proof is not present in mainstory.md
- **Action**: Either include proof or add reference "Proof deferred to Appendix A.1"
- **Severity**: Medium — the intuition is correct but rigor is missing

**GAP 2: Distance Prior Justification**
- **Claim**: "D(Δ) ∝ 1/Δ (scale-invariant / Jeffreys prior)" (§4.2)
- **Current**: Stated without derivation of why this prior is the "right" one
- **Missing**: Why is Jeffreys prior the choice? How does one derive this from attention statistics?
- **Action**: Add 1-paragraph justification in Method section, or defer to appendix
- **Severity**: Low-Medium — does not block main narrative but leaves a gap for reviewer

**GAP 3: τ* Derivation (Not Existence)**
- **Claim**: τ* = d_head/√L
- **Current**: Validated empirically across 99 runs but never derived theoretically
- **Missing**: Is there a theoretical argument connecting channel count, context length, and optimal τ?
- **Action**: Reframe as "Empirical Scaling Law" or "Conjecture"; do NOT claim theoretical derivation
- **Severity**: High — this is currently under-scoped in language
- **Current Language (WRONG)**: "parameter-free scaling law" (sounds like theorem)
- **Correct Language**: "empirically validated scaling law" or "proposed scaling law"

**GAP 4: Fisher → Attention Signal Mapping**
- **Claim**: §4.4 uses Fisher information as interpretive bridge
- **Current**: Described as "asymptotic" but not rigorously connected
- **Missing**: Full derivation of how Hessian of log-likelihood maps to attention allocation
- **Action**: Acknowledge as "local-curvature heuristic" not "formal derivation"
- **Severity**: Medium — supporting only, not required for main result

**GAP 5: Collision Threshold c = ln(L/2π)/ln(b) Derivation**
- **Claim**: §4.5 defines "wavelength exceeds training window" threshold
- **Current**: Definition is stated; the constant 2π is not derived
- **Missing**: Where does 2π come from? Is this a standard result?
- **Action**: Add one sentence: "The factor 2π arises from the period of the oscillatory component in RoPE"
- **Severity**: Low — technical detail

---

## 1.2 Theory-Derived Predictions vs. Empirical Validation

| Prediction | Derivation | Empirical Test | Result | Status |
|-----------|-----------|------------------|--------|--------|
| Geometric is suboptimal (τ>0 exists) | T1-T4 | Phase 17, 350M, all scales | ✅ Confirmed across 5 scales | **A+** |
| τ* shallow basin (<1% gap) | T2 + T3 | Phase 16 (99-run sweep) | ✅ 8/9 configs within top-3 | **A+** |
| Low-freq expansion, high-freq compression | T3 + T6 | Phase 18 (base=10K/500K) | ✅ Collision reduction -50% | **A** |
| Waterbed cost ≤ +0.4% short-range | T5 | Multi-scale Phase tests | ✅ Confirmed +0.4% at 350M | **A** |
| Waterbed gain ≥ 13.3% at 16× extrap | T5 (corollary) | 350M 3-seed PPL@16K | ✅ Exactly -13.3% | **A+** |
| Dead zone when c → 0.9 (base=10K, L=4096) | T9 (collision analysis) | Phase X (base=10K, L=4096) | ✅ EVQ underperforms (predicted) | **A** |
| EVQ raw can replace YaRN after progressive | T3 + empirical | Phase 17b (512→1024→2048) | ✅ EVQ raw@16K (11.2) ≈ EVQ+YaRN@8K | **B+** |

---

## 1.3 Theory Writing Quality Issues

### Issue 1: Notation Inconsistency
- φ_k sometimes written as "phi_k", sometimes as subscript
- τ is introduced as a parameter but not defined algebraically until Step 2 of the derivation chain
- **Fix**: Standardize notation in preliminaries section

### Issue 2: Order of Presentation
Current order: D(Δ) → K(φ₁,φ₂) → Broadband projection → ODE → Solution → CDF inversion

**Problem**: The broadband projection (the KEY ASSUMPTION) is sandwiched in the middle
**Better order**:
1. Motivate the variational problem
2. State the broadband projection as "Key Approximation" upfront
3. Then derive solution using this approximation
4. Validate the approximation empirically

### Issue 3: Boundary Condition Clarity
- §4.1 Step 6 (CDF inversion) assumes φ ∈ [0,1] range; this is stated implicitly
- **Fix**: Add sentence: "The quantile space u_k ∈ [0,1] maps to frequency space φ_k ∈ [0,1] via the monotone increasing CDF of the ODE solution"

### Issue 4: Missing Derivation Link
- The paper jumps from "ρ* = C₁cosh(τφ) + ... " to "φ_k(τ) = ..." without explicitly stating that you're inverting the CDF
- **Fix**: Add explicit sentence: "To obtain the frequency allocation φ_k, we invert the cumulative distribution function of the density solution ρ*"

---

# SECTION 2: EXPERIMENTS AUDIT

## 2.1 Claim × Evidence Matrix

**Legend**: **A+** = Multi-seed + large scale + converging lines  |  **A** = 3-seed or converging single-seeds  |  **A-** = 1-2 seed with strong direction  |  **B+** = Single-seed with theory support  |  **B** = Preliminary

| Claim | Subclaim | Evidence Source | Sample Sizes | Rigor | Status |
|-------|----------|-----------------|---------------|-------|--------|
| **Claim 1** (Geometric is special case) | φₖ(τ→0) → geometric | Mathematical proof | — | Exact | **A+** |
| **Claim 2** (τ* scaling law) | τ* = d_head/√L is near-optimal | Phase 16 (99 runs: 27 configs × 3 seeds) | 99 independent runs | Shallow basin <1% gap | **A+** |
| **Claim 2** (Parameter-free default) | Practitioners need no hyperparameter tuning | Worst case (L=512, d=32): rank #5 of 9 configs | 9 configs tested | Rank-based assessment | **A** |
| **Claim 3** (EVQ+YaRN >> Geo+YaRN) | Same-length extrapolation: -86% avg PPL | Phase 17 (454M, L=512, seed=42) | 4 lengths (4K-32K), 1 seed | -86% is extreme; direction consistent | **A-** (single seed) |
| **Claim 3** | 48K extrapolation at PPL ≤ 3.3 | Phase 17c (454M, 512→1024→2048, seed=42) | 7 lengths (2K-48K), 1 seed | 82% vs baseline (14.22 → 2.63) | **A-** (single seed) |
| **Claim 3** | 100% passkey at all tested lengths | Passkey-mix (350M, L=2048, 6 seeds) | 40 trials (6 seeds × 2-3 attempts) | Zero variance across seeds | **A+** |
| **Claim 4** (Progressive amplifies EVQ) | Stage 1: -34.6%, Stage 2: -52.0%, Stage 3: -81.2% | Phase 17 (seed=42) + Phase 17b (seed=42) + Phase 17c (seed=42) | Three stages, 1 seed per stage | Monotonic but single-seed chain | **B+** |
| **Claim 4** | Stage 1 multi-seed confirmation | 454M, L=512→1024→2048, seeds 43/44 | PPL@4K: GEO 314.94, EVQ 262.83 (2-seed avg) | 2-seed validates direction | **A-** |
| **Claim 4** | NIAH@1K Stage 1: +26pp | Same as above (seeds 43/44) | EVQ 82% ± 0, GEO 56% ± 4pp | Zero variance on EVQ | **A-** |
| **Claim 5** (Waterbed on downstream) | NLL reversal ±4.4% at in-dist vs 2× extrap | Phase 21a (750M, 13 LongBench tasks, zero-shot) | 13 tasks | Symmetric reversal, direction consistent | **A** |
| **Claim 5** | QA tasks up to -16.8% EVQ advantage | Same (Phase 21a) | Subset of 13, retrieval-heavy | Task-type decomposition | **A** |
| **Claim 5** | GovReport generation (ROUGE) | Phase 21B (750M, 200 samples, finetuned) | 200 samples, mean ± std reported | EVQ lower variance (ROUGE-2 std -20.2%) | **B+** |
| **Claim 5** | **QuALITY accuracy** | Phase 21B (454M, n=2086, task-specific finetuned) | n=2086 test set, full gold answers | @8K: +2.2pp (p≈0.02) | **A-** |
| **Claim 5** | **QuALITY Gold Answer NLL** | Same Phase 21B | n=2086 test set | @8K: -30%, @16K: -21% (strongest signal) | **A** |
| **Claim 6** (Zero-parameter vs learnable) | EVQ 333.7 vs DAPE-style 455.3 at 64× extrap | Phase 0–3 (125M, L=128, 3-seed) | 3 seeds, extreme extrapolation | -35% with 0 extra parameters | **A** |
| **Cross-scale consistency** | 50M → 750M direction holds | Phases across 5 scales (50M, 125M, 350M, 454M, 750M) | Distributed across multiple phases | Direction consistent, gain magnitude varies | **A** |
| **Cross-base generalization** | EVQ wins at base=10K and base=500K | Phase 18 (454M, L=512, single seed) | Two base values; PPL nearly identical (192 ≈ 192) | Single-seed but both bases agree | **B+** |
| **Cross-architecture transfer** | Llama-3 / Qwen-2.5 (LoRA) | Passkey@16K: 100% vs Geo 80% | 1 seed per model | Different tokenizers/training; LoRA confound | **B** |
| **Cross-modal transfer** | Video temporal (2-seed) | 3D RoPE temporal axis: -47% at 8× extrap | 2 seeds, K_t=8 temporal channels | EVQ+YaRN synergy replicates to video | **B+** |

### 2.1.1 Sample Size Breakdown

**Multi-Seed Gold Standard (A+)**:
- Phase 16 (τ* sweep): 99 independent runs ✅
- Passkey-mix (350M): 6 seeds, 40 trials ✅
- 350M raw PPL (FineWeb-Edu): 3 seeds ✅

**Partial Multi-Seed (A or A-)**:
- Phase 17 Stage 1 (454M): 2 seeds ✅
- Phase 21a (750M, 13 tasks): 1 seed but converging task-type pattern
- Phase 21B QuALITY: n=2086 full test set ✅

**Predominantly Single-Seed (B+ = directional but needs confirmation)**:
- Phase 17c (full progressive chain): seed=42 only — Stages 2-3 pending
- Phase 18 (base generalization): 1 seed per base
- Phase 17 (same-length YaRN composition): seed=42 only
- Phase 15 (750M 2K→4K): seed=42 only

---

## 2.2 Statistical Rigor Assessment

### Significance Testing

| Experiment | Test Type | Result | p-value / CI | Assessment |
|------------|-----------|--------|--------------|------------|
| QuALITY accuracy @8K | Accuracy difference | EVQ 26.8%, Geo 24.6%, Δ=+2.2pp | p≈0.02 (2.3 standard errors) | **Significant but weak** (near capacity floor) |
| Phase 16 τ* (exact match) | Rank distribution (9 configs) | 3/9 exact, 8/9 top-3 | Non-parametric rank | **Strong** (>89% within top-3) |
| Passkey-mix (6-seed zero variance) | Variance comparison | EVQ ±0pp, GEO ±4pp | Binomial / variance ratio | **Very strong** (zero variance unusual) |
| Phase 21a waterbed reversal | Sign test across 13 tasks | +4.4% in-dist, -4.4% at 2× extrap | Symmetric, task-type consistent | **Strong** (task-by-task decomposition) |
| Phase 21B GovReport ROUGE | Mean difference + variance | ROUGE-2 std: 5.01→4.00 (-20.2%) | Reported as mean ± std | **Weak on mean, strong on variance** |

### Missing Statistical Elements

**GAP 1: Confidence Intervals on Key PPL Numbers**
- Phase 17c reports EVQ+YaRN@48K = 2.63 but gives no uncertainty
- **Fix**: If single-seed, state "n=1 checkpoint; range [lower, upper] from beam search variance over 3 runs"
- **Impact**: Readers cannot assess whether 2.63 vs 2.70 is a real difference

**GAP 2: Hypothesis Tests on Single-Seed Multi-Context Results**
- Phase 17 reports "-86% average" across 4K, 8K, 16K, 32K
- **Issue**: What is the variance across contexts? Is the improvement consistent?
- **Fix**: Add "Per-length PPL gaps: 4K -86.3%, 8K -90.2%, 16K -88.8%, 32K -79.2%; average -86%, std 4.3pp"

**GAP 3: Multiple Comparison Correction**
- Phase 21a tests 13 downstream tasks without correction
- **Issue**: With 13 independent tests, expect ~1-2 spurious p<0.05 results by chance
- **Fix**: State "We report task-level results without multiple-comparison correction; the symmetric reversal pattern (direction consistent across 11/13 tasks) is the key finding"

### 2.2.1 Confounds and Limitations

**Known Confounds**:

1. **LoRA Fine-Tuning (Cross-Architecture Results)**
   - Llama-3/Qwen-2.5 LoRA results mix PE allocation optimization with task-specific feature learning
   - **Cannot isolate** the pure PE effect
   - **Recommendation**: Label as "preliminary cross-architecture transfer evidence" in text, not main claim

2. **GovReport Finetuning at Non-Standard Length**
   - Phase 21B finetuned at L=8192 (not training length L=4K)
   - **Confound**: Improvement could come from task-specific feature specialization, not PE
   - **Mitigation**: The variance reduction (ROUGE-2 std -20.2%) is less susceptible to this confound than mean scores
   - **Recommendation**: Lead with variance result, note mean as secondary

3. **QuALITY Finetuning with Low Sample Gradient**
   - 454M model + 2000 steps × bs4 = ~400× less than FIRE's training
   - **Issue**: At capacity floor (~25% accuracy), accuracy is a coarse metric
   - **Mitigation**: Gold Answer NLL (-30%) is the true signal; accuracy is secondary
   - **Recommendation**: Frontload Gold NLL, explain accuracy saturation in text

4. **Video Temporal: Applies τ to only one axis**
   - Spatial axes remain geometric
   - **Not a clean PE isolation** but does show cross-modal principle applicability
   - **Recommendation**: Appendix only or brief discussion

---

## 2.3 Reproducibility and Seed Reporting

### Current Seed Reporting

**Well-Reported**:
- Phase 16 (99-run sweep): Exact configuration list provided
- Phase 21a (13 tasks): Zero-shot, single checkpoint seed
- Passkey-mix (6-seed): Seeds 1-6 reported (though specific values not in mainstory.md)

**Under-Reported**:
- Phase 17c (single seed=42): Should list all checkpoint paths, learning rate, optimizer state
- Phase 18 (base generalization): Seed mentioned but no explicit config list
- Phase 21B (QuALITY): Finetuning hyperparameters (lr=1e-5) reported but seed not explicit in mainstory.md

### Reproducibility Gaps

| Experiment | Code Availability | Data Availability | Hyperparameter Card | Status |
|-----------|-------------------|-------------------|---------------------|--------|
| Phase 16 (τ* sweep) | ⏳ Assumed in codebase | FineWeb-Edu (public) | Yes (§3 "Practical Recipe") | Reproducible with codebase |
| Phase 17c (flagship) | ⏳ Assumed | FineWeb-Edu | Implicit in narrative | **Needs appendix card** |
| Phase 21a (NLL) | ⏳ Assumed | LongBench (public) | Implicit (zero-shot, seed=42) | **Needs appendix card** |
| Phase 21B (QuALITY) | ⏳ Assumed | QuALITY (public) | Reported: lr=1e-5, 2000 steps | Reproducible |
| Passkey-mix | ⏳ Assumed | Synthetic (code-generated) | Reported: 10% mix | Reproducible |

**Missing Reproducibility Card Template**:
```
Appendix A.X: Hyperparameter Cards

Phase 17c (454M, 512→1024→2048):
- Base checkpoint: [URL/path]
- Optimizer: AdamW (β₁=0.9, β₂=0.95, ε=1e-8)
- Learning rate: 1e-4 (no decay)
- Batch size: 128 (256 tokens × 512 context)
- Training steps: [per stage]
- Random seed: 42 (Stages 1-3)
- RoPE config: base=500K, τ=d_head/√L_stage, d_head=64
```

---

## 2.4 Experimental Design Quality

### Strengths

1. **Controlled Ablations**: Phase 18 isolates base (10K vs 500K) while holding other vars constant
2. **Multi-Scale Chain**: 50M → 750M span is the largest in PE allocation literature
3. **Converging Evidence Lines**: PPL, NIAH, passkey, NLL, downstream QA all show consistent direction
4. **Cross-Regime Testing**: PE-dominant (Phase 0–3) + natural data (Phase 21a) + both extremes tested
5. **Theory-Predicted Negative Result**: base=10K dead zone is predicted by collision theory and observed (§6.4)

### Weaknesses

1. **Limited Downstream Complexity**: LongBench is zero-shot (length extrapolation task, not complex reasoning). QuALITY finetuning is minimal (2000 steps).
   - **Better experiment**: Finetuning to convergence (25K steps) on LongBench tasks
   - **Current state**: Acceptable as primary evidence but not definitive

2. **Single Seed on Flagship Result**: Phase 17c (EVQ+YaRN@48K = 2.63) is seed=42 only
   - **Needed**: Confirm Stage 2-3 with seeds 43-44 (in progress per mainstory.md)
   - **Timeline**: This is the #1 priority for camera-ready

3. **No Inference Overhead Measurement**: Paper claims "zero inference overhead" but doesn't measure wall-clock time
   - **Missing experiment**: Compare latency of EVQ+YaRN vs Geo+YaRN at same context length
   - **Expected result**: Should be identical (EVQ is just initialization, YaRN does the scaling)
   - **Why it matters**: Reviewers will ask "is the speedup offset by slower inference?"

4. **Limited Baseline Comparisons**: Compared against Geo and Geo+YaRN, but limited against DAPE/FIRE/CREAM at same scale
   - **Current**: Phase 0–3 (125M) vs DAPE (also 125M) shows EVQ > DAPE
   - **Missing**: 350M+ comparison with FIRE/CREAM
   - **Rationale**: FIRE/CREAM use LoRA which confounds PE with adaptation; acknowledged in limitations

5. **Passkey Mix Variance is Suspicious**: EVQ@8K shows ±0pp variance across 6 seeds (literally perfect)
   - **Question**: Is this sampling variance ceiling (passkey is binary 0/1), or is the implementation noise-free?
   - **Fix**: Report the raw breakdown: "6 seeds × N trials: 100%, 100%, 100%, 100%, 100%, 100%" (full data)
   - **Why**: Readers will question whether ±0pp is real or a reporting artifact

---

# SECTION 3: FIGURES / TABLES AUDIT

## 3.1 Figures Status

| Figure | Filename | Status | Referenced in | Quality | Issues |
|--------|----------|--------|-----------------|---------|--------|
| Fig 1 | `fig1_frequency_dynamics.pdf` | ✅ EXISTS | §1 Positioning, §5 Results | Supporting | Subplots: (a) training loss, (b) passkey dynamics, (c) PPL extrapolation |
| Fig 2 | `fig2_evq_yarn_synergy.pdf` | ✅ EXISTS | Claim 3, §5.2 | **PRIMARY** | Four-line PPL plot (Geo raw, Geo+YaRN, EVQ raw, EVQ+YaRN) across 2K–48K; **KEY RESULT** |
| Fig 3 | `fig3_pe_dominant_scaling.pdf` | ✅ EXISTS | Claim 6, §5.5 | Main empirical | Extreme extrapolation (Phase 0–3: 64× extrap, L=128→8K) + Phase 11 (32× extrap with YaRN) |
| Fig 4 | `fig4_phase17c_flagship.pdf` | ✅ EXISTS | Claim 3, §1.1 | Extended context | Phase 17c progressive training: 512→1024→2048 with PPL@2K–49K |
| Fig 5 | `fig5_downstream_qa.pdf` | ✅ EXISTS | Claim 5, §5.4 | Downstream | QuALITY Gold Answer NLL and accuracy across lengths |
| Fig 6 | `fig6_tau_formula_validation.pdf` | ✅ EXISTS | Claim 2, §5.1 | τ* scaling law | Phase 16 sweep: PPL gap vs τ/τ* across 9 configurations |
| Fig 7 | `fig7_multiscale_waterbed.pdf` | ✅ EXISTS | Claim 5, §5.4 | Waterbed theory | NLL reversal across scales (750M Geo vs EVQ at in-dist vs 2× extrap) |
| Fig 8 (Collision) | `fig_tau_sweep_collision.pdf` | ✅ EXISTS | §4.5 sub-cycle theory | Theory supporting | Collision score vs τ with analytical prediction overlay |
| Fig 9 (Freq Distribution) | `fig_tau_sweep_freq_dist.pdf` | ✅ EXISTS | §4.3 physical meaning | Theory supporting | Channel spacing: Geo (uniform log) vs EVQ (redistributed) |
| Fig 10 (Cross-Scale) | `fig_tau_sweep_cross_scale.pdf` | ✅ EXISTS | Claim 6, Cross-scale section | Scale independence | PPL improvement across 5 scales at same τ and context ratio |
| Fig 11 (τ Sweep PPL) | `fig_tau_sweep_ppl.pdf` | ✅ EXISTS | Claim 2, Shallow basin | Mechanism | PPL vs τ showing flat basin around τ* |
| **Fig 4b: Collision-Block Theory** | ❌ **MISSING** | — | §4.5 + Reviewer defense | Critical | Should show base vs collision fraction with theoretical prediction; explains why base=10K fails |
| **Fig 5b: NLL Task Decomposition** | ⚠️ NEEDS POLISH | — | Claim 5, task-type mechanism | Important | Current data in table form; should visualize QA vs non-QA divergence |

### 3.1.1 Figure Quality Assessment

**Tier 1: Ready for Submission**
- Fig 2 (EVQ+YaRN synergy): Cleanest, most impactful 4-line plot. ✅
- Fig 3 (PE-dominant): Extreme extrapolation results are visually clear. ✅
- Fig 5 (QuALITY Gold NLL): Shows the -30%/-21% pattern clearly. ✅
- Fig 6 (τ* validation): 99-run sweep results, good visual hierarchy. ✅

**Tier 2: Needs Minor Polish**
- Fig 1 (Frequency dynamics): Supporting figure, good subplots but color contrast could improve
- Fig 4 (Flagship progressive): Busy with 7 PPL traces; consider splitting into (a) raw vs (b) +YaRN
- Fig 7 (Waterbed reversal): Shows NLL but lacks task-type color coding

**Tier 3: Needs Creation or Redesign**
- ❌ **Missing: Collision-block / base dead zone figure**
  - **Should show**: x-axis = base value (8K, 10K, 50K, 100K, 500K); y-axis = collision fraction or predicted EVQ gain
  - **Overlay**: Theoretical prediction curve + empirical points (Phase 18 data + base=10K L=4096 negative result)
  - **Why critical**: Explains the one negative result (base=10K) as a feature, not a bug. **Reviewer-defense essential**.
  - **Effort**: 1-2 hours to generate from Phase 18 data + collision formula

- ⚠️ **Enhance: Fig 5 task decomposition**
  - **Current**: Table showing QA (-16.8%) vs others (+4.4% averaged)
  - **Better**: Stacked bar chart or scatter plot with task on x-axis, NLL gap on y-axis, color-coded by task type
  - **Why**: Visual clarity of "which tasks benefit most" is important for reviewer intuition

---

## 3.2 Tables Status

| Table | Content | Scope | Completeness | Status |
|-------|---------|-------|--------------|--------|
| **Table 1** | Cross-scale raw PPL consistency | 50M–750M, 2K/4K lengths | 5 scales, all reported | ✅ Ready |
| **Table 2** | EVQ+YaRN vs baselines (Claim 3) | Phase 17c: 2K–48K lengths | 7 lengths + passkey + NLL | ✅ Ready |
| **Table 3** | Capability preservation (passkey-mix) | 350M, L=2048, 6-seed | NIAH@4K, @8K, trained vs untrained | ✅ Ready |
| **Table 4** | τ* sweep results (Claim 2) | Phase 16: 9 configurations × 3 seeds | Rank distribution + worst-case gap | ✅ Ready |
| **Table 5** | Phase 11 PE-dominant (Claim 6) | 454M, L=256, 3-seed | PPL@16× extrap, YaRN leverage 10× | ✅ Ready |
| **Table 6** | Phase 18 base generalization | 454M, L=512, base=10K & 500K | PPL@512 + PPL@4K with Δ% | ⚠️ Single-seed, needs flag |
| **Table 7** | 750M dynamics (Phase 15) | 750M, 2K→4K, PPL + AR exact | Lengths 8K–16K, passkey conversion | ⚠️ Single-seed, supporting |
| **Table 8** | Phase 21a (13 downstream tasks) | 750M, 13 LongBench tasks, NLL | Geo vs EVQ at 4K + 8K | ✅ Ready (task-type decomposition) |
| **Table 9** | Phase 21B GovReport | 750M, L=8K→16K ROUGE scores | Mean ± std for ROUGE-1/2/L | ✅ Ready (variance focus) |
| **Table 10** | **Phase 21B QuALITY** | **454M, n=2086, accuracy + Gold NLL** | **@4K, @8K, @16K ± YaRN** | **✅ Ready (strongest signal)** |
| **Table 11** | Related Work Comparison | PE allocation methods | 8 methods × 5 attributes (base/params/downstream/scale) | ✅ Ready |
| **Table A1** | Phase 0–3 (DAPE comparison) | 125M, L=128, 3-seed | PPL@8K extrapolation, parameter count | ✅ Ready (appendix) |
| **Table A2** | Passkey-mix detailed (6-seed) | 350M, all 6 seeds × 2 trials | Raw counts: "100% / 100% / ..." | ⚠️ Needs explicit listing |

### 3.2.1 Table Quality Checklist

**Data Integrity**:
- ✅ QuALITY n=2086 final data incorporated (v21 correction)
- ✅ 350M 3-seed FineWeb-Edu (raw PPL) included
- ✅ Phase 16 (99-run) rank distribution clear
- ⚠️ Phase 17c single-seed flagged but not highlighted in caption
- ⚠️ Phase 18 base generalization single-seed not flagged

**Formatting**:
- ✅ Column headers are clear (PPL@4K, etc.)
- ✅ Negative values (improvements) use consistent formatting
- ⚠️ Some cells show only percentage change; raw PPL would aid intuition (consider dual reporting)
- ⚠️ Passkey percentages: EVQ reported as "100% across 6 seeds" but raw breakdown missing

**Interpretation**:
- ✅ Winners are visually highlighted (bold or shading)
- ✅ Standard deviations/ranges reported for multi-seed
- ⚠️ Some tables lack a "Δ" column for quick comparison
- ⚠️ Captions could be more explicit about statistical significance

---

## 3.3 Missing Figures / Tables for Camera-Ready

| Gap | Current State | Needed For | Effort | Priority |
|-----|---------------|-----------|--------|----------|
| **Collision-block / base dead zone** | Data exists (Phase 18 + base=10K L=4096) | Explain negative result as theory prediction | 1-2 hours | **CRITICAL** |
| **Multi-config τ sweep heatmap** | Raw data in Phase 16 | Show why τ* is robust (worst-case +1% gap) | 2-3 hours | High |
| **Training loss curves (Geo vs EVQ)** | Probably exists in logs | Illustrate that short-range loss is flat (waterbed) | 1 hour | Medium |
| **NLL task-type decomposition plot** | Data in Phase 21a table form | Visual clarity on which tasks benefit (QA vs summarization) | 1 hour | Medium |
| **Downstream scaling trend** | Results across 5 scales scattered | Unified plot showing consistency across scales | 2 hours | Medium |
| **YaRN effectiveness comparison** | Implicit in Phase 17 / 17c | Isolate "how much does YaRN help EVQ vs Geo?" subplot | 1 hour | Low |
| **Waterbed cost-benefit curve** | Stated qualitatively | Quantitative plot: τ on x-axis, short-range cost vs long-range gain on y-axis | 1-2 hours | Low |

---

## 3.4 Figure / Table Allocation to 9-Page Budget

**Current Allocation** (from NEURIPS_SUBMISSION_PLAN.md):
- Main paper body: 9 pages (title through conclusion)
- Figure 1: Supporting (frequency dynamics) — 0.5 page
- Figure 2: Primary (EVQ+YaRN synergy) — 0.7 page **← MAIN EMPIRICAL**
- Figure 3: PE-dominant regime — 0.5 page
- Tables 1-2: Main results — 1.0 page total
- Appendix: Unlimited

**Recommended Reallocation**:
1. Keep Fig 2 as primary (4-line PPL plot is the killer result)
2. Add missing "Collision-block" figure to appendix with Reviewer Defense section
3. Move Fig 1(c) (750M dynamics) to appendix
4. Inline Table 1 + Table 2 (cross-scale + YaRN synergy) — 1.2 pages
5. Inline Table 8 (13-task NLL) as small table — 0.4 pages
6. Move Tables 5-7 (PE-dominant, base generalization, 750M) to appendix
7. Total main-body figures: Fig 2 (primary) + Fig 3 (extreme extrap) + Fig 6 (τ* validation) = 1.7 pages
8. Total main-body tables: Table 1, 2, 8, 10 (QuALITY) = 1.5 pages

**Page Budget Impact**:
- Figures & tables now use 3.2 pages of the 4.05-page experiments section ✅ (acceptable)
- Appendix will have 6+ pages of supporting figures/tables ✅ (standard for NeurIPS)

---

# SECTION 4: BIGGEST GAPS FOR REVIEWERS

## 4.1 Ranking of Likely Reviewer Attacks

| Risk | Probability | Severity | Evidence Needed | Status |
|------|-------------|----------|-----------------|--------|
| **"Single seed on flagship result"** | **Very High** | **Critical** | Phase 17c confirmation with seeds 43-44 for Stages 2-3 | **In Progress** (Stage 1 done, 2-3 pending) |
| **"Waterbed cost is hidden"** | High | Medium | Phase 21a NLL reversal + GovReport ROUGE directly address this | ✅ Solved (±4.4% waterbed, costs in variance) |
| **"Why not just tune base?"** | High | Medium | Phase 18 shows EVQ nearly identical across bases; τ* acts as effective base compensator | ⚠️ Explained but weak (single-seed) |
| **"LoRA confound on cross-architecture"** | High | Low | Acknowledge LoRA as confound; present as preliminary evidence only | ✅ Acknowledged in §6.2 |
| **"Only synthetic (passkey), no real language"** | High | High | Phase 21a + 21B provide downstream NLL and QA validation; QuALITY n=2086 is the answer | ✅ Addressed (3-line downstream evidence) |
| **"Models are too small"** | Medium | Medium | Argue 750M is upper end of "from-scratch" feasibility; larger models would be post-hoc only | ⚠️ Stated but not fully defended |
| **"τ* scaling law is just empirical"** | Medium | Low | Reframe as "empirical conjecture validated on 99 runs" not "theorem" | ⚠️ Language needs tightening |
| **"base=10K result contradicts main claim"** | Medium | High | Collision-block theory predicts this negative result; treat as theory validation | ❌ **MISSING Figure 4** |
| **"Compared only to geometric, not FIRE/CREAM/DAPE"** | Medium | Medium | Phase 0–3 (125M) beats DAPE; acknowledge that FIRE/CREAM use LoRA which confounds PE | ✅ Acknowledged |
| **"Gold NLL at capacity floor is a weak metric"** | Medium | Low | Gold NLL is superior to accuracy; show the four-layer signal gradient (PPL→NLL→passkey→accuracy) | ✅ Explained in Claim 5 |
| **"Where's the inference overhead analysis?"** | Low | Medium | Expected to be zero (EVQ is init, YaRN does inference); no measurement provided | ⚠️ **MISSING experiment** |
| **"Progressive training might just be memorization"** | Low | Medium | Show that EVQ raw approaches EVQ+YaRN performance (Stage 2: 11.2 vs 16.8@16K) — model is learning allocation | ✅ Phase 17b explains |

---

## 4.2 Missing Proofs / Evidence Blocks

| Gap | Impact | Evidence Existence | Fix Effort | Priority |
|-----|--------|-------------------|-----------|----------|
| **Waterbed inequality formal proof** | Theory rigor | Stated qualitatively, not formally proved | 1-2 hours appendix | Low |
| **τ* derivation from first principles** | Theory completeness | Does not exist (empirical only) | Not possible; reframe as conjecture | Medium |
| **Distance prior D(Δ)∝1/Δ justification** | Theory motivation | Described as "Jeffreys" but not derived | 1 hour appendix | Low |
| **Collision threshold c formula derivation** | Theory completeness | Stated but 2π factor not derived | 30 mins appendix | Low |
| **Stages 2-3 multi-seed confirmation** | Evidence robustness | Stage 1 done (2-seed), Stages 2-3 pending | 1-2 weeks compute | **CRITICAL** |
| **Collision-block figure (base dead zone)** | Reviewer defense | Raw data exists, figure missing | 2-3 hours | **HIGH** |
| **Inference latency comparison** | Practical impact | Not measured | 1-2 hours experiment | Low-Medium |
| **Failure mode documentation** | Honest assessment | base=10K is documented; others? | 2-3 hours analysis | Low |

---

# SECTION 5: CAMERA-READY CHECKLIST

## 5.1 Must-Do (Blocking Submission)

- [ ] **Stages 2-3 multi-seed confirmation** (seeds 43-44) — CRITICAL for flagship result
- [ ] **Figure 4: Collision-block / base dead zone** — Reviewer defense essential
- [ ] **Tighten τ* language** — Change "scaling law" to "empirical scaling law" / "conjecture" throughout
- [ ] **Fix notation consistency** — Standardize φ_k, τ, ρ notation in preliminaries
- [ ] **Write reproducibility cards (Appendix)** — Hyperparameters for all main phases
- [ ] **Formal statement of broadband projection boundary** — Lemma 2.1 with explicit conditions
- [ ] **QuALITY data finalized** — v21 corrections incorporated ✅ (done)
- [ ] **Passkey breakdown listing** — Explicit "100% / 100% / 100% / 100% / 100% / 100%" for 6 seeds

## 5.2 Should-Do (Strong Accept Likely Without, Spotlight Needs It)

- [ ] **Stages 2-3 multi-seed** (see above — same item)
- [ ] **Collision-block figure** (see above — same item)
- [ ] **NLL task-type decomposition plot** — Visual clarity on which tasks benefit
- [ ] **Training loss curves (Geo vs EVQ)** — Illustrate waterbed in action
- [ ] **Inference latency measurement** — Confirm "zero overhead" claim
- [ ] **Phase 18 multi-seed (base generalization)** — Confirm single-seed direction

## 5.3 Nice-to-Have (Strengthens Narrative)

- [ ] **1.5B scale validation** — Larger model would cement claims
- [ ] **More downstream tasks** — SCROLLS or other long-context benchmarks
- [ ] **Training recipe for practitioners** — Step-by-step guide to EVQ integration
- [ ] **Code release** — GitHub repo with reproducible pipeline

---

# SECTION 6: GAPS BY SUBMISSION READINESS TIER

## 6.1 For Spotlight / Oral Contender

**Currently Achieved** ✅:
- Closed-form theoretical derivation with validation
- 5-scale PPL consistency
- 3-line downstream evidence (NLL + ROUGE + QA accuracy + Gold NLL)
- 6-seed passkey (100% zero variance)
- 99-run τ* validation
- Cross-architecture transfer
- Cross-modal transfer (video)

**Still Needed** ❌:
- ✅ Stages 2-3 multi-seed confirmation (Stage 1 complete; 2-3 in progress)
- ❌ Figure 4 (collision-block visual)
- ⚠️ Inference latency (nice-to-have, not blocking)
- ⚠️ Phase 18 multi-seed (nice-to-have; pattern is clear at single-seed)

**Verdict**: **On track for spotlight with Stages 2-3 multi-seed + Figure 4**

## 6.2 For Strong Accept (Poster)

**Currently Achieved** ✅:
- All of above

**Still Needed** ❌:
- ✅ Figure 4 (collision-block) — mandatory
- ⚠️ Stages 2-3 multi-seed — should-have, not blocking

**Verdict**: **Already at strong accept level; Figure 4 + language tightening → likely accept**

## 6.3 For Weak Accept / Borderline

**Currently Achieved** ✅:
- Theory is novel + rigorous
- Multi-scale PPL evidence
- Downstream validation (though limited)
- Honest about limitations

**Could Fail On** ❌:
- Single-seed flagship result (Phase 17c) — if Stages 2-3 don't confirm
- Missing collision-block explanation — if Fig 4 not included
- τ* presented as theorem — if language not tightened
- "Only passkey" narrative — if reviewers don't accept downstream as sufficient

**Verdict**: **Robust to weak accept; figure completion + multi-seed confirmation → strong accept likely**

---

# SECTION 7: FINAL SUMMARY TABLE

| Dimension | Status | Evidence | Gaps | Priority |
|-----------|--------|----------|------|----------|
| **THEORY** | Ready with caveats | Closed-form derivation validated; assumptions clearly stated | τ* is empirical not derived; waterbed proof missing (appendix) | Low (appendix) |
| **EXPERIMENTS** | Strong with gaps | 5-scale PPL (✅), 6-seed passkey (✅), 3-line downstream (✅), 99-run τ* (✅) | Stage 2-3 multi-seed pending ⚠️; Phase 18 single-seed; inference latency missing | **CRITICAL** (Stage 2-3) |
| **FIGURES** | 11/12 ready | Fig 1-3, 5-6 complete; 4 flagged figures exist; main narrative covered | Figure 4 (collision-block) missing for reviewer defense | **HIGH** |
| **TABLES** | Ready | 10+ tables covering all claims; QuALITY n=2086 final data incorporated | Table formatting minor polish; some need single-seed flags | Low |
| **WRITING** | Ready | Narrative flows; related work table clear; claims well-scoped | τ* language needs tightening ("conjecture" vs "law"); broadband projection boundary not formally stated | Low-Medium |

---

**RECOMMENDATION**: **Paper is at strong-accept readiness with spotlight-contender potential.**

**Submission timeline**:
1. **Week 1**: Complete Stages 2-3 multi-seed (in progress)
2. **Week 2**: Create Figure 4 (collision-block), tighten τ* language, write reproducibility appendix
3. **Week 3**: Final proofreading + checklist compliance
4. **Week 4**: Submit

**Risk**: If Stages 2-3 multi-seed does not confirm EVQ advantage, Claim 4 (progressive amplification) drops to "B" tier and paper is still strong-accept (on Claims 1-3, 5-6 alone), but spotlight candidacy weakens.

---

*Audit prepared: 2026-03-12 | Analyst: Claude Code | Document: mainstory.md v21*
