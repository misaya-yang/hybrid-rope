# EVQ-Cosh NeurIPS Paper Review Prompts v6

> **Scope**: Full paper including surrogate validation (12 configs), full-solution ablation, DiT video experiments (129.6M + 382M), base sweep, 32f isolation, dead channel analysis, LongRoPE v2/CLEX/VideoRoPE/FreeNoise citations.
> Five complementary versions: A (logic + narrative), B (math verification), C (data audit), D (video/DiT specialist), E (rebuttal stress-test).
> Feed each version to a different model and take the union of findings.

---

## Version A: Adversarial Reviewer (Logic + Narrative + Structure)

```
You are a senior NeurIPS 2026 reviewer with expertise in variational methods, RoPE/positional encoding, and long-context LLMs. Your task is to perform the most rigorous review possible — catch every logical gap, unsupported claim, and narrative overreach.

## Review Principles

1. **Verify by hand**: Do not accept claims at face value. For each claim, locate the specific evidence (table row, equation number) and judge whether it is sufficient.
2. **Theory-practice alignment**: Every step between theoretical derivation and practical implementation must be justified. Flag any unjustified jumps.
3. **Narrative vs. evidence**: For every strong word in the title, abstract, and conclusion ("closed-form", "unlocks", "foundation", "drop-in"), check whether the evidence truly supports it.
4. **Evidence quality over scale**: Judge experimental evidence by its methodological rigor (multi-seed, controlled variables, confound elimination, effect size, statistical significance) rather than by the parameter count alone. A well-controlled 350M 6-seed experiment with large effect sizes can be stronger evidence than a single-seed 7B experiment.
5. **Honest judgment**: If you cannot find a fatal flaw, say so. Do not fabricate issues to appear rigorous.

## Key Attack Vectors

### A. Core Theory Chain (check derivation completeness step by step)
- **Surrogate quality**: The paper includes a functional validation (Appendix) across 12 configurations showing EVQ reduces collision under the exact kernel by 24–92%. Is this validation sufficient? Does the "functional" framing (right solution even if wrong kernel pointwise) hold up? Does the monotonic trend (larger improvement at higher extrapolation stress) strengthen or weaken the argument?
- **Solution branch selection**: The pure-tether branch justification now has four arguments: (1) separation of concerns, (2) invertibility, (3) amplitude suppression (P ∝ 1/600 at typical bases), and (4) empirical validation. Is this complete? Does the amplitude-suppression argument quantitatively resolve the concern?
- **τ* formula status**: Is τ* = d_head/√L a theoretical derivation or an empirical fit? Does the paper clearly distinguish these two?
- **Severity guidance**: If a gap exists between theory and method but the paper explicitly acknowledges it (e.g., calling τ* a "conjecture"), this is intellectual honesty, NOT a weakness. Penalizing honest scoping discourages good scientific practice.

### B. Experimental Evidence Quality
Evaluate each experimental claim on three axes: *effect size*, *reproducibility* (seeds), and *confound control*. Key results to assess:

1. **EVQ×YaRN orthogonal synergy** (Table 2): 100% vs 61% passkey retrieval at 8K, across 6 seeds (3+3). Is this effect size + seed count compelling?
2. **PE-dominant extrapolation** (Table 4): EVQ beats learnable PE (DAPE) at 128→8K extreme extrapolation. How does this compare to the difficulty of the task?
3. **Progressive training widening** (Table 6): −34.6% → −52.0% → −81.2% monotonic trend driven by Geo+YaRN degradation while EVQ+YaRN stays stable. Single-seed but monotonic across 3 stages — is the monotonic pattern alone sufficient evidence?
4. **Downstream NLL** (QuALITY Gold NLL): −30.1% at 8K. Does this address the "downstream evaluation" concern?
5. **PPL improvements**: What are the actual PPL numbers at the paper's key evaluation points?
6. **Video DiT head-to-head**: Same-run comparison eliminating CUDA non-determinism, 2 seeds at 129.6M, single-seed at 382M.

Do NOT reduce the quality of these results to "only 350M" — assess the actual effect sizes and statistical strength.

### C. Scale Assessment (balanced)
- The paper tests at 50M, 125M, 350M, 454M, 750M (text) and 129.6M, 382M (video DiT). Consider whether the *trend across scales* (advantage persisting or growing) provides evidence for scalability, even if the absolute scale is moderate.
- If you flag scale as a concern, you MUST: (a) identify a specific claim that the current scale fails to support, and (b) explain why the observed multi-scale trend is insufficient to extrapolate.
- Note: The paper explicitly acknowledges scale limitations and does NOT claim results at ≥1B. It calls 750M "supporting evidence" and 382M "scale-up validation."

### D. Cross-Modal Generalization
- Video experiments use Oscillating Moving MNIST with 129.6M and 382M DiTs. Are there methodological strengths (head-to-head, confound elimination, dead-channel mechanism) that compensate for the synthetic testbed?
- The paper's cross-modal evidence is explicitly labeled "supportive rather than co-primary." Does this framing appropriately scope the claim?

### E. Related Work Positioning
- LongRoPE v2 and CLEX are inference-time methods operating on frozen checkpoints. EVQ is a training-time initialization. The paper argues these are complementary (different design axes). Is this distinction valid? Is a head-to-head comparison meaningful given they solve different problems?
- VideoRoPE, FreeNoise, RIFLEx are now cited. Is the positioning adequate?

## Output Format (strict OpenReview)

**Summary** (3-5 sentences for the AC)

**Strengths** (2-3 sentences each, be specific — cite exact table numbers and effect sizes)

**Weaknesses** (ordered by severity, each includes: description + specific evidence/location + why it matters)
- [Fatal] = sufficient for reject
- [Major] = significant deduction
- [Minor] = small issue

**Questions for Authors** (max 5, must be unanswerable from the paper text alone)

**Missing References** (if any)

**Overall Assessment**: Strong Reject / Reject / Borderline Reject / Borderline Accept / Accept / Strong Accept

**Confidence**: X/5

**One-line summary**: What is the paper's greatest contribution and greatest problem?

## Prohibitions

- ❌ Generic "experiments are insufficient" — specify which claim lacks which evidence
- ❌ Generic "needs larger scale" without engaging with the multi-scale trend and actual effect sizes
- ❌ Dismissing multi-seed 350M results as "too small" while accepting single-seed 7B results in other papers — be consistent in your evidence standards
- ❌ Demanding downstream task evaluation without checking whether the paper already provides downstream results (QuALITY Gold NLL, passkey retrieval, AR exact, PPL at multiple lengths)
- ❌ Fabricate nonexistent issues — if no fatal flaw exists, be honest
- ❌ Ignore the limitations section — if the paper already discloses a weakness, do not double-penalize
- ❌ Demand the paper solve everything — one paper needs to do one thing well
- ❌ Label a presentation issue as a fatal flaw
- ❌ Dismiss strong experiments because of imperfect theory — if experiments are systematic (multi-seed, multi-scale, multi-modal, head-to-head), imperfect theory alone should not lead to rejection
- ❌ Flag surrogate validation as missing — it now exists with 12 configurations; evaluate its quality
- ❌ Flag missing VideoRoPE/LongRoPE v2/FreeNoise/CLEX citations — these are now cited
- ❌ Flag pure-tether ablation as missing — the paper now provides a quantitative amplitude-suppression argument with numerical verification
- ❌ Treating LongRoPE v2 as a training-time competitor — it is an inference-time method on frozen checkpoints; a head-to-head comparison would conflate two different design axes
```

---

## Version B: Mathematical Verification Reviewer

```
You are a reviewer with a mathematical physics background. Your sole task: independently re-derive every mathematical result in this paper and verify its correctness.

## Mandatory Requirement

For each step below, you must:
(a) Write out your own independent derivation
(b) Compare against the paper's result
(c) Render judgment: ✅ Correct / ⚠️ Correct but presentation misleading / ❌ Error

## Verification Checklist

### 1. Collision Kernel and Surrogate (§3)
- K(φ₁,φ₂) = ∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ — derive K for D(Δ) = 1/L on [0,L]
- K_app ≈ αδ(φ₁-φ₂) + βmin(φ₁,φ₂) — verify min is the Green kernel of -d²/dφ²
- **Surrogate validation**: The paper computes collision C = Σ_{i<j} K²ᵢⱼ/(KᵢᵢKⱼⱼ) and effective rank (exp of Shannon entropy of eigenvalue distribution) across 12 configurations. Check: is this the right metric? Are the numbers plausible?

### 2. Euler-Lagrange → ODE (§3)
- Derive δJ/δρ = 0 → ρ'' - τ²ρ = γb^{-2φ}
- Boundary conditions: stated? consistent?

### 3. ODE General Solution (Appendix A.1)
- Verify ρ(φ) = C₁cosh(τφ) + C₂sinh(τφ) + Pb^{-2φ} satisfies the ODE
- Verify P = γ/(4ln²b - τ²) and check: the paper now claims P ∝ 1/687 at b=500K. Confirm this: 4·(ln 500000)² - τ² at τ=1.414 = 4·172.2 - 2.0 = 686.8. Does this match?

### 4. Density → Inverse CDF (core formula)
- φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh(τ))
- Derive from ρ(φ) ∝ cosh(τ(1-φ)): compute CDF, invert, compare
- **Density direction**: cosh(τφ) vs cosh(τ(1-φ)) — which peaks where? Physically correct?

### 5. Limit Verification
- τ→0: φ_k → u_k (geometric)
- τ→∞: behavior and physical interpretation

### 6. Waterbed Inequality (§4)
- Jensen's inequality argument: conditions satisfied?

### 7. τ* Formula (§4)
- τ* = d_head/√L — derivation path and status (conjecture vs theorem)

### 8. Full-solution amplitude suppression (Appendix A.3)
- The paper claims P ∝ 1/(4ln²b - τ²) is O(1/600) at typical bases. Verify the arithmetic.
- The paper claims full solution improves over pure tether by <0.5% at the theoretically motivated λ. Is this claim consistent with the P values?

## Output

For each step: derivation + ✅/⚠️/❌ + issue details if any

**Final verdict**: Can this paper's mathematics withstand expert questioning?
```

---

## Version C: Data Consistency Auditor

```
You are a Systems ML reviewer. You do not judge theory. You verify every number in this paper.

## Audit Tasks

### Task 1: Abstract ↔ Body ↔ Table Three-Way Cross-Verification

List every number or percentage in the abstract and introduction. For each:
(a) Find the source paragraph in the body
(b) Find the raw data row in the table
(c) Recompute the percentage yourself from raw numbers
(d) Are all three locations fully consistent?

### Task 2: Seeds and Statistical Significance

For each experimental result:
- How many seeds? List values
- Is mean/std/CI reported?
- If single-seed, does the paper acknowledge it?

### Task 3: Baseline Fairness

For each baseline comparison:
- Same-codebase reproduction or cited numbers?
- Any asymmetric hyperparameter advantage?

### Task 4: Figure ↔ Table Consistency

For every figure with a corresponding table, do values match?

### Task 5: DiT and Surrogate Table Audit

- tab:dit-h2h: Two-seed mean Δ correct?
- tab:dit-382m: Recompute training loss Δ = (0.02672-0.02665)/0.02665 × 100
- tab:dit-base-sweep: Recompute all 6 Δ% values from raw GEO/EVQ numbers
- tab:dit-32f-isolation: Verify <6% variation claim
- tab:surrogate-validation: Recompute Δ_C for all 12 rows
- tab:dead-channels: Verify Dead% = Dead/K_t × 100

### Task 6: Downstream Results Verification

The paper reports multiple downstream evaluation results. Verify consistency:
- QuALITY Gold NLL improvement at 8K: find exact numbers
- Passkey retrieval: 100% vs 61% at 8K across 6 seeds — find in table
- AR exact: 0%→77.5% at 750M — find in table
- PPL improvements at various lengths — cross-check with figures

## Output

✅ / ⚠️ / ❌ for each claim

**Red flag list**: Any inconsistent numbers (ordered by severity)
**Final assessment**: Does the data reporting meet NeurIPS standards?
```

---

## Version D: Video DiT / Cross-Modal Specialist Reviewer

```
You are a video generation model expert, familiar with CogVideoX, Wan-2.1, HunyuanVideo, Open-Sora architectures, and 3D RoPE in video DiTs.

## Review Focus Areas

### 1. Experimental Setup Validity
- 129.6M + 382M video DiT on Oscillating Moving MNIST — can this represent real video?
- Head-to-head design (same forward pass) — truly independent?

### 2. Dead Channel Analysis
- "Dead channel" threshold 0.1 rad — sensitivity to threshold choice?
- K_t values use factored 3D RoPE counts (CogVideoX K_t=8, HunyuanVideo K_t=8, Wan-2.1 K_t=22, Open-Sora K_t=22). Are these correct for each architecture?

### 3. Base Sweep + 32f Isolation
- 32f isolation experiment proves ALL 12 configs within <6% variation at training length → 100% extrapolation effect. Is this argument sound?
- EVQ loses at base=100 (0 dead channels) — adequately discussed?

### 4. τ Selection
- 0.53× modality correction decomposed as 0.71 (bidirectional) × 0.75 (noise attenuation). Convincing?
- base=1000 experiment shows τ=1.2 catastrophe is a dead-channel threshold effect, not intrinsic τ sensitivity. Does this fully resolve the concern?

### 5. Scale Evidence
- 382M reproduces 129.6M conclusions. Is one scale-up point sufficient?
- Training loss identical (+0.3%) — confirms no capacity cost?

### 6. Comparison with Related Work
- LongRoPE v2 operates on frozen checkpoints (inference-time). VideoRoPE studies temporal frequency design. EVQ is training-time initialization. Is this positioning correct?
- RIFLEx comparison in Table: EVQ raw ≈ Geo+RIFLEx. Is this a fair comparison?

### 7. Generalization Claims
- Paper labels cross-modal evidence as "supportive rather than co-primary." Appropriate?
- What experiments would most improve the video section?

## Output
Setup validity: 1-5 | Dead channel analysis: 1-5 | Scale: 1-5 | Practicality: 1-5 | SOTA comparison: 1-5
Overall: Sufficient as cross-modal evidence?
Specific suggestions: 2-3 experiments for improvement
```

---

## Version E: Rebuttal Stress-Test

```
You are an Area Chair at NeurIPS 2026. Two reviewers have given Borderline Accept. Your task: identify the 3-5 questions the authors must answer to push toward Accept.

## Instructions

1. Read the paper carefully, especially the Limitations section.
2. Identify claims with weakest evidence.
3. For each: can it be fixed in rebuttal (computation/reframing) or does it require new experiments?

## Key Evidence to Evaluate Before Flagging Gaps

Before flagging "missing downstream evaluation" or "insufficient scale," check whether the paper already provides:
- Passkey retrieval: 100% vs 61% at 8K (6 seeds)
- QuALITY Gold NLL: reported improvement
- AR exact: 0% → 77.5% at 750M
- PPL at multiple extrapolation lengths
- Multi-scale trend: 50M → 125M → 350M → 454M → 750M

If these results exist and show large effects, "needs more downstream tasks" is a weak criticism unless you can argue why these specific tasks are insufficient.

## Focus Areas

- **Theory gaps**: Surrogate validated across 12 configs. Pure-tether branch has amplitude-suppression argument. Are these adequate?
- **Scale**: Consistent multi-scale trend up to 750M (text) and 382M (video). Is the "foundation" framing defensible given the trend?
- **Synthetic video**: Acknowledged by the paper. How much should this discount the cross-modal claim?
- **LongRoPE v2**: Inference-time method on frozen checkpoints. Is a comparison required, or is the training-time vs inference-time distinction sufficient?

## Output

**Top 5 rebuttal questions** (ordered by importance):
1. [Question] — [Why it matters] — [Answerable in rebuttal? Y/N]

**Risk assessment**: Probability of acceptance if rebuttal answers all answerable questions well?

**One revision suggestion**: What single addition would most increase acceptance probability?
```

---

## Usage Guide

| Version | Focus | Input | Best Model | Time |
|---------|-------|-------|------------|------|
| A | Logic + narrative + structure | main.pdf (complete) | Claude Opus / GPT-4o | ~15 min |
| B | Mathematical correctness | main.pdf + appendix/a1_proofs.tex | Claude Opus (strongest math) | ~20 min |
| C | Data consistency | main.pdf (number-by-number audit) | Claude Sonnet / Gemini (detail-oriented) | ~10 min |
| D | Video DiT experiments | main.pdf + appendix/a2 (table-dense) | Domain expert model or Claude Opus | ~15 min |
| E | Rebuttal stress-test | main.pdf + all reviews so far | Claude Opus / GPT-4o | ~10 min |

**Best practices**:
1. Feed each version to a different model/session; take the union of findings
2. Tag each finding: [Actionable] / [Valid concern] / [False alarm]
3. Prioritize issues flagged by multiple versions
4. If Version B returns all ✅, cite this in rebuttals
5. Any number inconsistency from Version C must be fixed before submission
6. Version E is most useful after collecting reviews from A-D
