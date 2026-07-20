# Consolidated theory stance after strong-model adjudication (2026-07-20)

Date: 2026-07-20 (rebuttal deadline 2026-07-22; real reviews not yet received).
Status: `consolidated_post_adjudication`. Modifies no paper number.

This document is the AUTHORITATIVE theory stance after the independent
strong-model adjudication. It reconciles:
- `ADVERSARIAL_REVIEW_FINDINGS_20260720.md` (the adversarial findings, T1–T8 / E1–E10), and
- `STRONG_MODEL_THEORY_VERDICT_20260720.md` (fable5's per-question adjudication Q1–Q8, with independent numerics).

Where this document and the verdict conflict with `ADVERSARIAL_REVIEW_FINDINGS`,
THIS DOCUMENT SUPERSEDES. The findings file is retained as the pre-adjudication record.

Verification: the verdict's numerics script
(`rebuttal/strong_model_verdict_numerics_20260720.py`, pure NumPy) was re-run in
this workspace and reproduces every load-bearing number, including the Phase16
d-sweep (Section G) after one fix — the script originally hardcoded the
adjudicator's sandbox path for `data/curated/phase16_99run_manifest.csv`; this was
replaced with a path resolved relative to the script. No paper text was touched.

---

## 1. Adjudication summary (my findings × strong-model verdict)

| Finding | Topic | fable5 verdict | Net status |
| --- | --- | --- | --- |
| T1 / Q1 | cosh theorem is a Green-kernel tautology | 成立 | **Confirmed + sharpened**: the exact quadratic form `ρᵀKρ` itself selects a TRUNCATED density (13–15 of 32 channels zeroed), not cosh; variable-coefficient diagonal selects Bessel. The theorem is genuine math (existence/uniqueness/closed form) but has zero RoPE-discriminating content — relabel to representation/solvability result. |
| T2 / Q2 | deployed cosh is the minimizer of nothing | 成立 | **Confirmed + quantified**: `τ_surr=√(β/α)=6.24/5.70` vs deployed `1.41/1.00` (4.4×/5.7×, grows with L); `‖ρ_surr−ρ_deploy‖₁=0.90/0.93`; **no gauge connects them** (both exponents differ); the 24–92% table is computed at the DEPLOYED τ (reproduced row-by-row). |
| T3 / Q3 | d_head "structural factor" is a convention | 部分成立 | **Confirmed in derivation, premise corrected**: a is a normalization pairing (three defensible pairings give d⁰/d^{1/2}/d¹; the paper uses √d and d¹ in two tiers, unmarked). BUT a normalization-independent observable DOES exist: the Phase16 fixed-L d-sweep coarsely supports d¹ and rejects √d within its 0.75–1.5 window. |
| T4 / Q4 | balance needs U=O(τ²) linear but labeled "KL" (O(τ⁴)) | 成立 | **Confirmed + strengthened**: text still prints the KL label; numerically KL slope θ^1.84 (≈O(τ⁴)) vs linear score θ^0.99 (O(τ²)). NEW: the 1/L factor is ITSELF a proxy normalization choice — under a total-variance reading the L-dependence disappears entirely. |
| T5 / Q5 | collision sign-indefinite; "shaping reduces ∫w/ρ²" false | (a) 成立 / (b) 部分成立 | **(a) Confirmed** (collision argmin τ≈13.1, c≈4.6, far outside the PPL basin). **(b) Confirmed the sentence is false** (∫1/ρ²=1.004/1.05/1.18/1.59/11.6/113, strictly rising) **but premise corrected**: a non-circular weight DOES exist — `w=q(Lb^{−φ})` (the transport-proxy weight) — under which cosh reduces ∫w/ρ² (ratios 0.32–0.87); however generic monotone reallocations do too (linear tilt beats cosh), so it is not cosh-specific. |
| T6 / Q6 | three inconsistent "exact kernels" | 部分成立 | **Premise REFUTED**: under a uniform prior the cos-product integral IS the sum+diff sinc closed form — kernels (i)≡(ii); fit and validation use the same kernel (reproduced). Real residual = content weights (folds into Q7) + **NEW finding**: the validation table's 4 video rows are computed at τ=1.5=0.53·16/√32 with d_eff=K=16≠2K, an undisclosed d_eff/grid convention mix within the same table. |
| T7 / Q7 | Fisher forcing is the only model-coupled term, dropped | 部分成立 | **Confirmed structurally** (η_F is the only trained-model input, never measured; "controlled residual" cannot stand; the validation table is not a forcing-residual diagnostic). **My amplification number corrected**: sinh(τ)/τ = 6.82 at τ=4 (not 13.6); 1.37 at the primary τ=1.41; 186 at τ=8. |
| T8 / Q8 | 24–92% validation has no discriminating power | 成立 | **Confirmed + strengthened**: a one-parameter EXPONENTIAL density matches or exceeds EVQ on 12/12 configs; EVQ at its own τ_surr gives 97–100% (> the deployed point); collision-only argmin c≈4.6–9.6 lies far beyond. The table only shows "moving mass out of dead channels reduces an unsigned redundancy statistic." |

**The four corrections fable5 made to my question framing** (record so the findings
file is not read as gospel): Q3 (a normalization-independent observable exists),
Q5(ii) (a non-circular weight exists), Q6 (kernels (i)≡(ii); my premise wrong),
Q7 (amplification 6.82, not 13.6).

The experimental/statistical findings E1–E10 are untouched by the theory
adjudication and stand as written in `ADVERSARIAL_REVIEW_FINDINGS_20260720.md`.

---

## 2. Final three-tier survivor set (post-adjudication)

Each tier survives but one notch narrower than the submitted paper, and narrower
than the 07-11/07-13 audits:

1. **Exact tier (keep, relabeled).** Convexity of `C_app`; unique positive
   minimizer; closed-form CDF/inverse-CDF; `τ→0` geometric limit; self-consistency
   identity; waterbed inequality (allocation-divergence level); `K^{−1}/K^{−2}`
   quantization bounds. **Identity: representation/solvability results about a
   surrogate chosen for solvability; ZERO RoPE-discriminating power** (any fit of
   (α,β) returns a cosh; the exact quadratic form selects a truncated density;
   variable diagonal selects Bessel).
2. **Conditional proxy tier (keep, all conditions explicit).**
   `U_tr=(M/L)∫qρ` is linear ⇒ nonzero first-order change ⇒ `τ*²=45λQ₁M²/L`.
   Three normalization knobs must ALL be disclosed: stiffness `1/d` (the a=1
   pairing), channel-additive utility `M∝d`, and the displacement-energy
   normalization giving `1/L` (a total-variance reading removes the L-dependence).
   **Identity: a self-consistent stationarity calculation under a proxy objective
   — not a KL theorem, not a task theorem.**
3. **Empirical tier (clearer after this audit).** In the Phase16 9-config window:
   `τ=0` is never optimal (geo dominated) in 9/9; best multiple `c∈[0.75,1.5]`
   everywhere; fixed-L d-sweep coarsely supports d¹ and rejects √d in-window;
   L-exponent −0.34…−0.50. Plus the previously retained matched midpoint contrasts
   and the MLA scarce-channel directional result.

**No longer writable (further nailed down this round):** a RoPE variational
justification of cosh; any "O(1) convention" reconciliation of τ_surr vs deployed
τ; an unconditional "structural d_head factor"; the KL naming of U and "½ factors
absorbed into λ"; the 24–92% table as validation of the surrogate/shape/τ;
"controlled forcing residual" and "the validation table is a residual diagnostic";
the unweighted "shaping reduces ∫w/ρ²" claim.

---

## 3. Definitive retract / rewrite ledger

### 3A. Theory items newly sharpened by the adjudication (priority; ready-to-use English supplied by fable5)

| # | Location | Current (wrong) | Required action + replacement |
| --- | --- | --- | --- |
| A1 | `paper/tables/table_epistemic_map.tex:11` (row 2, "Theoretical core") | cosh = exact RoPE core | Relabel: "Closed-form solvability/representation result: the Green-kernel surrogate admits a unique positive minimizer in closed form for every (α,β). Because every fit of (α,β) necessarily returns a member of this cosh family, agreement of the fitted family with cosh carries no evidential weight about RoPE; all RoPE-specific support must come from tests outside the surrogate." |
| A2 | `paper/appendix/a1_proofs.tex:315-323` ("What the surrogate predicts") | understated τ_surr gap | State the disconnect: "The fitted surrogate's own optimum is τ_surr=√(β/α)≈6.2 (L=2048) and 5.7 (L=4096) at d=64, b=500K, versus the deployed τ=1.41 and 1.00 — a factor 4.4–5.7× that grows with L, with ‖ρ_surr−ρ_deploy‖₁≈0.90–0.93. The deployed allocation is therefore not the surrogate optimum, and no rescaling convention connects the two: the exponent pairs (d^{1/2}, L^{−0.11}) and (d¹, L^{−1/2}) differ in both variables. Theorem 1 supplies the shape family only." |
| A3 | `a1_proofs.tex:126-127` (table caption); `03_theory.tex:15,32` | "validates the surrogate/theorem" | Note the validation τ is the DEPLOYED value (text rows τ=d_eff/√L; video rows τ=0.53·16/√32) and remove any "validates the theorem/surrogate optimum" reading. |
| A4 | `03_theory.tex:113` | "derives the structural d_head factor and L^{−1/2} exponent" | Replace: "Within the stated proxy normalization (per-channel-extensive utility, α-normalized stiffness), the balance yields τ*∝d_head/√L; the d-exponent depends on this normalization pairing (alternative defensible pairings give d⁰ or d^{1/2}), while the L^{−1/2} exponent follows from the diffuse-softmax displacement normalization. Empirically, the fixed-L d-sweep (d∈{32,64,128}) is consistent with the d¹ pairing and inconsistent with d^{1/2} within its 0.75–1.5× search window." |
| A5 | `a1_proofs.tex:320` | "correctly captures the d_head dependence" (re √d) | Annotate the √d-vs-d¹ contradiction between the two tiers, or delete "correctly". |
| A6 | `03_theory.tex:93` | U = "per-channel post-softmax KL gain"; "½ factors from the KL Taylor expansion are absorbed into λ" | Delete the KL-Taylor clause; relabel U as the linear transport score (see A7). |
| A7 | `03_theory.tex:95-109` (Proposition narrative) | O(τ²)-vs-O(τ⁴) "KL" balance | Replace with the honest balance: "Define the diffuse transport score U_tr(ρ;L)=(M/L)∫₀¹q(Lb^{−φ})ρ(φ)dφ, the channel-sum of squared probability displacements ‖J(p₀)c_ω‖²=q(ωL)/L at the uniform baseline. U_tr is linear in ρ, so along ρ_τ=1+τ²η+O(τ⁴) it has the nonzero first-order change (M/L)Q₁τ². Balancing this against the O(τ⁴) Pearson stiffness yields τ*²=45λQ₁M²/L. The interior optimum exists only because U_tr is linear in ρ; ordinary baseline-to-perturbed KL is O(τ⁴) and yields no such point. The 1/L factor is the displacement-energy normalization of the diffuse softmax Jacobian, not a KL curvature; under the alternative total-variance normalization the L-dependence disappears, so the L^{−1/2} exponent is conditional on this normalization choice." |
| A8 | `a1_proofs.tex:626` | "a shaped ρ reduces the weighted inverse-density load ∫w/ρ²" | Replace: "For the unweighted load (w≡1), shaping strictly increases ∫1/ρ² (Jensen; equality iff ρ≡1): the values are 1.004/1.05/1.59/11.6 at τ=0.5/1/2/4. For the phase-variance weight w(φ)=q(Lb^{−φ}) — fixed by (L,b) and independent of ρ — the cosh density does reduce ∫w/ρ² below uniform at all deployed configurations (ratios 0.32–0.87), but so do generic monotone reallocations; the reduction reflects moving cells toward phase-resolving channels, not a cosh-specific property, and inherits the uniform-position-prior assumption." |
| A9 | `a1_proofs.tex:425-451` + every E_off reference | E_off reduction = "the primary driver" | Add: "unsigned redundancy diagnostic; its minimizer (τ≈13, c≈4.6) lies far outside the trained-PPL basin" (may reuse EXPERIMENT_THEORY_REVIEW §1.3(2) wording). |
| A10 | `a1_proofs.tex:124` + validation-table caption | no per-row τ disclosed | Disclose the τ used per row (text: d_eff=2K deployed rule; video: 0.53·d_head/√L with d_head=16≠2K). |
| A11 | `03_theory.tex:48`; `a1_proofs.tex:72` | "controlled but nonzero residual" | Delete "controlled". Replace: "…the forced branch is an unmeasured residual: its amplitude depends on the activation-conditioned coefficient η_F, which we do not measure; the L¹/CDF bound controls mass transport but is amplified by sinh τ/τ (1.4 at τ=1.41, 6.8 at τ=4, 186 at τ=8) under inverse-CDF inversion, so no pointwise control is claimed at deployed τ." |
| A12 | `a1_proofs.tex:122` (last sentence) | "…also serves as the operational forced-branch residual diagnostic" | Delete the sentence entirely (the table never evaluates the forced branch). |
| A13 | `table_epistemic_map.tex:12` (row 3) | "forcing branch is CDF/L¹-suppressed at typical bases" | Append "amplitude unmeasured (η_F never estimated)". |
| A14 | `a1_proofs.tex:119-158` (section) | "Surrogate quality: functional validation" | Rename (e.g. "Directional collision diagnostic at the deployed allocation"); caption: "This diagnostic is computed at the deployed τ on the same kernel and configurations used to fit (α,β). It is not shape-discriminating: a one-parameter exponential reallocation matches or exceeds these reductions on all 12 configurations, EVQ at the surrogate's own τ_surr achieves 97–100%, and the collision-only optimum lies at c≈4.6–9.6. We report it only as evidence that the deployed allocation moves in the redundancy-reducing direction, not as validation of the cosh shape, the fitted surrogate, or the deployed τ." |
| A15 | `03_theory.tex:15,32`; `a1_proofs.tex:106,117(iii)` | 24–92% as "functional validation of the surrogate/cosh" | Downgrade every such reference to "directional diagnostic (not shape- or τ-discriminating; see App. …)". |

### 3B. Previously-conceded items STILL PRINTED in the submitted PDF (disclosure catalog; from prior audits + adversarial review §4)

Because NeurIPS 2026 forbids uploading a revised PDF, each stays live in the copy
reviewers score until addressed in a response / merged AC disclosure:

- `c_coll=1.171` apparatus — `03_theory.tex:108,113`, all of `table_lambda_cv.tex`
  (1.6%/2%, CV 0.28%, λ∞=0.96, LOO, "<1% across all 27"), `a1_proofs.tex:267,271,273`
  ("±20% basin"); authors' re-optimization (argmin c≈4.7, paper point ~670× above
  the family minimum) makes the "flat basin" affirmatively false.
- Waterbed — `03_theory.tex:88` "both vanish at ρ≡1" (actually `C_app[1]=α/2+β/6≠0`)
  and the causal "thus paid for"; `a3_supporting_results.tex:110` Fig.7 caption
  "consistent with the theoretical prediction".
- "K_app is the theory's SOLE approximation" — `a1_proofs.tex:122` (contradicts the
  ~8 further approximations); premise of the epistemic-map table.
- Self-consistency theorem stated for an UNDEFINED functional `J` —
  `a1_proofs.tex:222`; the `T₁` closed form is wrong by a factor τ yet the same
  paragraph claims "verified to <1e-15" (false against the displayed formula);
  checklist still certifies the proofs.
- Bessel-positivity argument (judged unusable) still printed — `a1_proofs.tex:117`.
- YaRN "orthogonal/additive/complementary" — `main.tex:50` (abstract),
  `01_intro.tex:20`, `03_theory.tex:117`, `a1_proofs.tex:293`, `02_related.tex:9`,
  `06_limitations.tex:4`, `a3_supporting_results.tex:29`.
- "Against DAPE" / "isolating frequency structure from operator capacity" —
  `01_intro.tex:11,20`, `02_related.tex:6`, `05_experiments.tex:41`,
  `table4_pe_dominant.tex:2,14` (the row is a 32-parameter shared-frequency vector,
  not DAPE, and adds no operator capacity).
- "27 configurations / <1% PPL" and the 3/9-6/9-8/9 ranks — `03_theory.tex:117`,
  `table_lambda_cv.tex:2`, `a1_proofs.tex:271,304,334,341`, `main.tex:79` (real
  structure: 9 configs, single-seed pilots mixed with 3-seed confirmations,
  common-metric 7/9 wins 2/9 losses).
- LoRA "r≳K restores EVQ viability" — `a1_proofs.tex:405-418`; refuted at exactly
  r=K=64 by the registered-negative E7 gate; printed 8B table numbers do not match
  the tracked matched artifact.
- Primary II hyperparameters LR 6e-4 / batch 16 / "125M" —
  `a2_experiment_details.tex:25,42,45`, `table4_pe_dominant.tex:2` (actual ~151.9M,
  LR 3e-4, effective batch 64, ~1831 steps, PE LR×100).
- Compute checklist "anonymous internal A100/H100 cluster" — `main.tex:128`,
  contradicted by M4 Max (basin sweep), RTX 5090 (QA16K), `results_5090b` (Primary
  I raw), and REPRODUCE.md.
- `03_theory.tex:81` "endpoint/midpoint shifts PPL by <1% across all K≥16" —
  contradicts the rebuttal's own midpoint-confound defense (1.51× context stretch
  at K=16). Unaddressed by any prior audit.
- QuALITY "continued at 4K, hence 4K is in-distribution" — `a3_supporting_results.tex:71`
  (the direct opposite of audit E-09: direct 2K→4K finetune, no 4K continuation).
- Fig.3 auto-s32 numbers under a "fixed s=8" label also appear in a body-cited
  TABLE `tab:pe-yarn-l256` (`a4_supporting_experiments.tex:22-24`), beyond the
  figure/caption scope of audit E-08.
- Primary III pooled ± (batch-confounded; audit E-03) still printed; the 32K
  EVQ-alone reversal omitted; Primary II body prose omits the only 3-seed row
  (learnable-τ 437.9±12.2).

---

## 4. τ=d/√L — strongest honest defense (fable5 §9.2)

> τ=d_head/√L is an operating-point selector with three independent supports and
> three disclosed limitations. Supports: (i) a conditional stationarity calculation
> — linear diffuse-transport score against Pearson stiffness — yields τ*∝d/√L under
> explicitly stated normalization choices; (ii) in the 9-configuration trained sweep
> the rule's multiple lies in [0.75,1.5] everywhere, τ=0 is never optimal, and the
> fixed-L d-sweep is consistent with the d¹ pairing while rejecting d^{1/2} within
> the search window; (iii) the L-exponent is empirically −0.3…−0.5 on the tested
> grid. Limitations: (i) the exact tier does not support it — the fitted surrogate's
> own optimum is √d·L^{−0.11}, a 4.4–5.7× larger τ with near-disjoint density, and
> the exact-kernel collision optimum (c≈4.6–9.6) has no d/√L structure at all;
> (ii) both the d-factor and the 1/L factor are normalization-dependent within the
> proxy; (iii) all trained evidence lives in a ≤2× search window at ≤1024 context,
> with the L≥16K falsification test unrun.

Net: **defensible as "proxy-motivated, coarsely d-and-L-consistent empirical
default"; NOT defensible as a derived optimum at any tier.** Relative to the
07-11/07-13 audits: empirical side slightly STRONGER (the d-sweep window evidence
was previously unused); theory side WEAKER (validation table proven
non-discriminating, τ_surr disconnect quantified, the 1/L convention-dependence
exposed).

---

## 5. Proposed tightened playbook kernels (FOR AUTHOR APPROVAL — not applied)

The frozen theory kernels in `rebuttal_playbook.md` §2.5 and §3.5 currently imply
slightly more than is defensible after this adjudication. PROPOSED replacements
(do NOT freeze/send until the author approves):

- **§2.5 (τ / KL kernel) — propose replacing with the §4 paragraph above**, which
  adds the d-exponent-is-a-pairing caveat and the "exact tier does not support it"
  limitation explicitly. The current §2.5 kernel does not mention that the surrogate
  optimum is τ_surr≈6 (4.4–5.7× off) nor that the d-factor is pairing-dependent.
- **§3.5 (kernel / surrogate kernel) — propose appending**: "The 24–92% reduction
  is a directional diagnostic computed at the deployed τ on the fit kernel; it is
  not shape- or τ-discriminating (a one-parameter exponential reallocation matches
  or exceeds it on 12/12 configs, and EVQ at τ_surr reaches 97–100%). The exact
  theorem is a closed-form solvability/representation result for the surrogate, not
  a RoPE-physical selection of cosh."

The §3.1 capability kernel and the comparator-identity / remaining-contribution
kernels are unaffected by the theory adjudication.

---

## 6. Author decisions still needed

1. Approve the survivor stance in §2 and the τ defense in §4 as the single theory
   voice across all responses.
2. Approve the tightened §2.5 / §3.5 kernels in §5 (then freeze).
3. Decide the merged AC integrity-disclosure scope from the §3B catalog (which
   still-printed items to proactively disclose if reviewers do not name them).
4. Confirm no paper number/figure is changed in this window (none is changed here);
   the §3A/§3B items are manuscript-correction targets for any future revision and
   rebuttal-response content only.

## 7. Files

- This stance: `rebuttal/THEORY_STANCE_CONSOLIDATED_20260720.md`
- Adjudication: `rebuttal/STRONG_MODEL_THEORY_VERDICT_20260720.md`
- Numerics (re-runnable): `rebuttal/strong_model_verdict_numerics_20260720.py`
- Pre-adjudication findings (superseded where conflicting): `rebuttal/ADVERSARIAL_REVIEW_FINDINGS_20260720.md`
- Handoff questions: `rebuttal/CORE_THEORY_QUESTIONS_FOR_STRONG_MODEL_20260720.md`
