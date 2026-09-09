# Independent adversarial review — findings (2026-07-20)

Date: 2026-07-20 (rebuttal deadline 2026-07-22; real reviews not yet received).
Status: `internal_adversarial_review`. Modifies no paper number.

> **Superseded in part (2026-07-20).** A strong-model adjudication
> (`STRONG_MODEL_THEORY_VERDICT_20260720.md`) confirmed most theory findings but
> corrected four premises. The authoritative stance is
> `THEORY_STANCE_CONSOLIDATED_20260720.md`; where the two disagree, it wins —
> notably T6/Q6 (kernels (i)≡(ii); my "three kernels" premise was wrong),
> T7/Q7 (amplification is sinh τ/τ = 6.82 at τ=4, not 13.6), T3/Q3 (a
> normalization-independent observable DOES exist: the Phase16 d-sweep), and
> T5/Q5(b) (a non-circular weight w=q(Lb^{−φ}) DOES exist, though not cosh-specific).

Method: a 4-phase multi-agent audit (5 deep-readers → 6 adversarial lenses →
consolidation → per-finding verification) produced 49 raw findings; these were
de-duplicated and compared against the existing self-audits. Below, each item is
tagged:
- 🆕 = newly identified (not stated, or not stated this sharply, by the existing
  audits `FULL_PAPER_INTEGRITY_AUDIT_20260713.md`,
  `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`,
  `EXPERIMENT_THEORY_REVIEW_20260720.md`);
- 🔁 = already conceded by the audits but still printed in the submitted PDF or
  sharpened here.

Companion handoff: `rebuttal/CORE_THEORY_QUESTIONS_FOR_STRONG_MODEL_20260720.md`
contains the 8 self-contained theoretical questions (Q1–Q8) distilled from the
theory section below, for adjudication by a stronger model.

Authority note: this review EXTENDS, and does not override, the canonical audits.
All claims below should be read against the paper source; the highest-leverage
items were cross-checked line-by-line against `paper/sections/03_theory.tex` and
`paper/appendix/a1_proofs.tex`.

---

## 1. Bottom line

- **Theory.** The exact tier is internally consistent (cosh minimizes `C_app`),
  but there is **no surviving, un-withdrawn link between it and the deployed rule
  `τ=d/√L`** — this is the real fatal point, not a label fix like the KL order.
  The audit concedes the exponent mismatch (`√d·L^{-0.11}` fit vs `d·L^{-1/2}`
  deployed) but does not compute the actual factor (**4.4–5.7×**, densities
  near-disjoint, `L¹≈0.9`) nor state that the deployed cosh is the exact minimizer
  of nothing (T2).
- **Experiments.** Read as a single disjunction, the survivor set is closed in
  every branch by the authors' own data (E1) — a global conclusion the audits
  never state. The only real-model (8B) test is a registered negative, and the
  only confidence interval in the corpus belongs to that negative (E5).
- **Meta (largest trust risk).** NeurIPS 2026 forbids uploading a revised PDF,
  yet the submitted text still prints a whole surface of withdrawn claims
  (§4). The playbook freezes 3–4 kernels for 3–5 *triggered* concerns but has
  **no catalog of the full still-printed withdrawn surface**; every un-triggered
  item stays live in the copy reviewers score.

---

## 2. Theory vulnerabilities

**🆕 T1 — the exact theorem is a self-referential tautology.** Because
`min(φ,ψ)` is the Green kernel of `−∂²` on `[0,1]`, differentiating the EL
equation `αρ+βg+ν=0` twice FORCES `ρ''−(β/α)ρ=0`, whose positive mass-1 solution
is the cosh family for EVERY `(α,β)`. The ansatz family and the minimizer family
coincide by construction; the logical direction is cosh→surrogate (a tractable
quadratic form reverse-engineered to have a cosh minimizer). The audits concede
"surrogate-specific / tractability" but do not frame it as a tautology. → Q1.

**🆕 T2 — the deployed cosh is the exact minimizer of NOTHING (most fatal).** With
the paper's own fit `α≈1/d_rot` (a grid-spacing artifact: kernel diagonal ~1/2 ×
spacing `1/K`) and `β∼L^{-0.22}`, the theorem's minimizer is
`τ_surr=√(β/α)≈√d·L^{-0.11}` ≈ **6.2** (d=64, L=2048, b=500K) vs deployed
`τ*=d/√L≈1.41` — a factor **4.4–5.7×**, GROWING with L and maximal in the primary
L=2048–4096 regime; concentration ratio `cosh τ` ≈ 257:1 vs 2.18:1;
`‖ρ_surr−ρ_deploy‖₁≈0.90–0.93` (near-disjoint). Audit P0.4 concedes only the
EXPONENT mismatch, not the factor or the density distance, and does not state
that the deployed point is far outside the surrogate's own optimum. The 24–92%
table, if evaluated at the mild deployed τ (≈1.5, Fig.1), validates the
deployment rule, not the theorem. → Q2. This is the crux for whether "τ is
theory-motivated" survives at all.

**🆕 T3 — the "structural d_head factor" is a normalization convention.**
`τ*∼d^{(a+b)/2}·L^{−b/2}`; the proposition takes `a=1` (d-normalized stiffness)
to get `d¹`, but the UNNORMALIZED stiffness (`a=0`) gives `d^{1/2}` — matching the
surrogate tier (T2) and contradicting the proposition. Only `L^{−1/2}` (`b=1`,
from the diffuse `1/L` Jacobian) is convention-independent. → Q3.

**🆕 T4 — the scaling balance REQUIRES `U=O(τ²)` linear, but `U` is labeled "KL"
(which is `O(τ⁴)`).** The only softmax KL computed is
`D_KL(p‖p_{εg})=½ε²gᵀJ_sm g+O(ε³)=O(τ⁴)` at `ε=τ²` (zero first variation), while
`U=(d/L)[Q₀+τ²Q₁+…]` is LINEAR in `ρ_τ=1+τ²η+…` (a transport first moment). The
balance `τ*²=45λQ₁·d²/L` REQUIRES `U=O(τ²)`: if `U` were the `O(τ⁴)` KL, both
`½S_χ²` and `λU` would be `O(τ⁴)` and stationarity would fix a coefficient ratio,
not yield `τ²=d²/L`. Audit T-02 concedes the mislabel but not this structural
dependency. The body still prints "post-softmax KL gain" and "O(τ²) utility gain"
at `paper/sections/03_theory.tex:93,108`; the appendix equation `a1_proofs.tex:501`
refutes it on the page. → Q4.

**🆕 T5 — the collision mechanism is sign-indefinite, and "shaping reduces
∫w/ρ²" is false.** (a) `E_off=Σ_{i<j}K_ij²/(K_ii K_jj)` is squared mutual
coherence over ALL position pairs; minimizing it gives a near-orthogonal
spreading code (good for worst-case position discrimination), but autoregressive
attention needs a SIGNED, distance-resolved criterion (nearby cohere, distant
decorrelate). (b) For the only task-free weight `w≡1`, Jensen gives
`∫1/ρ²dφ ≥ (∫1/ρ dφ)² ≥ 1`, equality iff `ρ≡1`; the cosh density gives
`∫1/ρ² = 1.004 (τ=0.5), 1.05 (τ=1), 1.59 (τ=2), 11.6 (τ=4)` — shaping STRICTLY
INCREASES the quantization distortion. The claimed reduction needs `w`
pre-concentrated where `ρ` is large (circular). → Q5.

**🆕 T6 — three inconsistent "exact kernels".** (i) `K=∫D(Δ)cos(ω₁Δ)cos(ω₂Δ)dΔ`
(cos-product, no sum-frequency term, no content weights) used for "validation";
(ii) the sinc kernel `K=(1/2L)[sin((ω₁−ω₂)L)/(ω₁−ω₂)+sin((ω₁+ω₂)L)/(ω₁+ω₂)]`
which KEEPS the `cos((ω₁+ω₂)Δ)` sum term, used to FIT `α,β`; (iii) the true RoPE
Gram `z_ij=Re Σ_k α_{ij,k}e^{ir_ijω_k}`, content-weighted. The cosh shape is fit
from (ii) but validated against (i); neither carries the content weights the
paper's own Fisher coefficient depends on. → Q6.

**🆕 T7 — Fisher forcing is the ONLY model-dependent term, and it is dropped.**
The forcing `γb^{−2φ}` carries `η_F(φ_k)=(1/2s_att²)E[w|α_{ℓhij,k}|²]` — the only
place trained-attention statistics enter. The deployed homogeneous pure-tether
branch depends only on `τ=√(β/α)` fit to kernel geometry, hence is entirely
model-free/universal. The "controlled residual / O(1/log b)" argument uses an
L¹/CDF bound amplified by `sinh τ/τ` in the inverse-CDF (≈13.6 at τ=4), so
"mass-small" is not "pointwise-small" at the deployed τ. → Q7.

**🆕 T8 — the 24–92% "functional validation" has no discriminating independence.**
`α,β` are fit to the exact kernel `K` on the same 12 configs used for validation;
the collision score `C=Σ_{i<j}K_ij²/(K_ii K_jj)` is a NONLINEAR function of `K`,
distinct from the quadratic form `⟨ρ,K_app ρ⟩` the surrogate minimizes; the τ used
is undisclosed; and a collision-only oracle reaches comparable (~65%) reduction.
→ Q8.

**🔁 T9 — known but still printed (must enter the disclosure catalog):**
- The entire `c_coll=1.171` apparatus (`table_lambda_cv.tex`: 1.6%/2%, CV 0.28%,
  `λ∞=0.96`, LOO, "<1% across all 27"; `a1_proofs.tex:273` "±20% basin") is still
  printed at `03_theory.tex:108,113`; the authors' own re-optimization (argmin
  `c≈4.7`, paper point ~670× above the family minimum) makes the "flat basin"
  affirmatively false (`c_deployed=1` is ~79% below the true collision argmin).
- Waterbed `03_theory.tex:88` still prints "both vanish at ρ≡1" (actually
  `C_app[1]=α/2+β/6≠0`, audit T-04) and the causal "thus paid for"; the Fig.7
  caption (`a3_supporting_results.tex:110`) additionally prints "consistent with
  the theoretical prediction" (a new vehicle no audit lists).
- "K_app is the theory's SOLE approximation" (`a1_proofs.tex:122`) still printed,
  contradicting T-07's ~8 further approximations, and is the premise of the
  epistemic-map table.
- The self-consistency theorem is stated for an UNDEFINED functional `J`
  (`a1_proofs.tex:222`); the `T₁` closed form is wrong by a factor τ (acknowledged
  typo) yet the same paragraph claims "verified to <1e-15" — affirmatively false
  against the displayed formula (2.0026 vs 1.0013 at τ=0.5); the checklist still
  certifies the proofs as correct.
- The Bessel-positivity argument (`a1_proofs.tex:117`, "variable-α Bessel can
  violate ρ>0 at φ=1") is judged unusable by audit P1.2 but is still printed twice
  as reason (ii) for choosing cosh.

---

## 3. Experiment / statistics vulnerabilities

**🆕 E1 — the survivor set, read as a disjunction, is closed in every branch by
the authors' own data (most important global conclusion).** "EVQ allocation
matters" holds only if it matters in (i) abundant-channel MHA, (ii) the repo
fixed-ramp Primary I contrast, (iii) scarce-channel MLA, or (iv) the 8B real
model. The authors' own runs close all four: (i) full official YaRN erases the
MHA gap to ~0.007 NLL and the difference-in-differences favors Native, and an
official LINEAR ramp REVERSES the ordering (E4 probe); (ii) the fixed-ramp
"differential leverage" Primary I keeps is attributed by the authors' own probe to
the broad smoothstep RAMP shape, not the allocation; (iii) MLA "EVQ alone beats
the rescaler" holds only at 2× (ties at 24K, reverses at 32K); (iv) 8B is a
registered negative. The audits list the negatives one at a time and keep a
per-primary positive survivor; none states the global conjunction.

**🆕 E2 — at base=100 (all channels alive) GEO beats EVQ by ~20%**
(`a2_experiment_details.tex:240`, printed). So the cosh shape is actively harmful
in the non-pathological regime and helps only in mis-configured baselines (dead
channels from base/length mismatch; missing range scaling) — the "third axis /
generally better" thesis reduces to "please configure your baseline correctly."

**🆕 E3 — the primary metric PK (teacher-forced NLL-gap) inverts against EVQ in
the authors' own data.** In the 151.9M diagnostic, EVQ has BETTER long-length PPL
(6.5–10.6% lower) but teacher-forced PK aggregate **46.0% vs Geo 80.67%**; AR
exact match is **0% in all six cells** (including the "100%" cell). So
PK-positive and AR/QA-negative are two views of ONE artifact (EVQ changes the
teacher-forced logit geometry PK rewards), not independent positive and negative
lines.

**🆕 E4 — the entire τ* basin evidence is a 50.9M / local-wikitext / laptop (M4
Max) sweep** whose training lengths {256,512,1024} have ZERO overlap with any
primary anchor's L_train (2048/128/8192) and whose scale matches none; its
selection metric is NLL-gap passkey under which AR exact is 0 in all 198 manifest
cells. "99 trained models spanning 27 validation settings"
(`a1_proofs.tex:304`) conflates a 9-config 50M sweep with a heterogeneous
portfolio that also COUNTS the withdrawn registered-negative 8B-LoRA row and the
0.53×-modality-corrected video rows.

**🆕 E5 — inferential asymmetry: the single statistically rigorous result in the
corpus is a negative.** The only bootstrap CI is the QA16K negative gate; every
positive primary is an untested point estimate (Primary II 333.7 is single-seed
with no variance; Primary I reports std only at the saturated 8K cell `100±0` =
ceiling, WITHHOLDING the 12K/16K std — from the tracked per-seed JSON, EVQ 12K is
79±14pp with one seed at 0.64 vs Geo mean 0.59). The checklist answers "Yes" on
error bars.

**🆕 E6 — the printed 8B table hides a +47% matched in-distribution regression
that CAUSES the failure.** The printed "+30%" is against the UNADAPTED base and
uses an EVQ 8K PPL (9.63) matching no tracked run; the matched artifact gives EVQ
10.068 vs Native-LoRA 6.817 = **+47.7%**, and the QA deficit is concentrated ≤8K,
so the 8×/19× extrapolation PPL gains are purchased by destroying in-distribution
task quality on the same model. The "PASS" causal gates are measured on the same
native-pretrained→midpoint conversion the audit says is not a clean shape
contrast.

**🆕 E7 — provenance inversion.** The rebuttal's strongest replacement narrative
(three-gate mechanism chain) and the only CI-bearing result rest on unregistered,
gitignored 07-14/07-15 artifacts, while the withdrawn-but-printed tables the
rebuttal is retreating from carry curated hashes. If the AC inspects the one
rigorous statistic, it is not in the manifest.

**🆕 E8 — compute checklist ("anonymous internal A100/H100 cluster") is
contradicted by a provenance trail**: the τ* basin sweep ran on an M4 Max laptop;
the QA16K gate on a single RTX 5090; Primary I raw payload is sourced from a
directory named `results_5090b`; REPRODUCE.md directs reproduction to consumer
RTX 4090/5090 and Apple M-series. This is a materially false hardware attestation,
not a missing record — beyond the conceded Primary I/II item in audit E-10.

**🆕 E9 — the τ* rule is unfalsifiable as presented.** Phase16 counts any τ within
a 1.5× band (a 3× range) as a hit while the empirical optima are SYSTEMATICALLY
right-shifted ~1.20–1.25× (so the deployed τ is ~20% sub-optimal by the paper's
own sweep, excused by "flat basin"); the one sharp cross-modal prediction (video,
2.83) misses by ~2× (best 1.5) and is rescued post-hoc by a free integer
`m∈{1,2,4}`.

**🔁 E10 — known but still printed / needs the catalog:**
- Primary III ± is a batch-confounded pseudo-std (seed 42 batch 6 vs seeds 43/88
  batch 5; audit E-03) yet the table still prints the pooled ±; the 32K
  EVQ-alone-WORSE-than-GEO+YaRN reversal is selectively omitted (significant at
  16K, ties at 24K, overlaps at 32K; EVQ std 3–6× GEO's).
- Primary II body prose OMITS the only 3-seed row (learnable-τ 437.9±12.2); the
  learned controls ran ~1831 steps with PE LR×100 (may be undertrained), so
  "zero-param beats learned" is consistent with "learned controls failed to
  converge."
- QuALITY `a3_supporting_results.tex:71` prints "continued at 4K, hence 4K is
  in-distribution" — the DIRECT OPPOSITE of audit E-09 (direct 2K→4K finetune, no
  4K continuation), collapsing the in-distribution-vs-extrapolation contrast.
- Fig.3 auto-s32 numbers (labeled "fixed s=8") are also printed in a body-cited
  TABLE `tab:pe-yarn-l256` (audit E-08 scopes its fix to the figure and caption
  only).

---

## 4. Paper-text ↔ survivor-set contradiction surface (Attacker-F line-by-line)

Because no revised PDF can be uploaded, each of these stays live in the copy
reviewers score, and each contradicts a frozen rebuttal kernel:
- YaRN "orthogonal / additive / complementary" framing spans the ABSTRACT
  (`main.tex:50`), contributions (`01_intro.tex:20`), `03_theory.tex:117`,
  `a1_proofs.tex:293`, `02_related.tex:9`, `06_limitations.tex:4`,
  `a3_supporting_results.tex:29` — while the authors' own faithful-scaler data
  give the RANGE verdict by the paper's stated substrate-vs-range criterion
  (`05_experiments.tex:22`).
- `02_related.tex:6` "isolating the frequency-structure effect from operator
  capacity" is voided by the table's own 32-parameter shared-frequency row (no
  operator capacity is added, so nothing is isolated).
- `a1_proofs.tex:416` "LoRA with r≳K should restore EVQ viability" is refuted at
  exactly r=K=64 by the authors' own registered-negative E7 gate.
- Primary II printed hyperparameters LR 6e-4 / batch 16 / "125M"
  (`a2_experiment_details.tex:25,42,45`) — not just a scale mislabel but actively
  wrong LR (2× too high) and batch (¼ too small) that would reproduce a different
  run.
- `03_theory.tex:81` "endpoint/midpoint shifts PPL by <1% across all K≥16"
  directly contradicts the rebuttal's own midpoint-confound defense (a global
  `b^{−1/(2K)}` factor = 1.51× context stretch at K=16, "not a trivial index
  convention"). No audit addresses this line.
- "Against DAPE" head-to-head still printed across intro/related/experiments/table
  after the DAPE relabel.

Recommended action: build a single table of the full still-printed withdrawn
surface and decide which items enter a merged AC integrity disclosure. The
playbook currently lacks this catalog.

---

## 5. Recommended next steps

1. Hand the companion file `CORE_THEORY_QUESTIONS_FOR_STRONG_MODEL_20260720.md`
   (Q1–Q8) to a stronger model for adjudication; the crux is **Q2** (is the
   τ_surr-vs-deployed gap a convention or a disconnection?) and **Q4** (is the
   balance internally consistent once U is honestly defined?), since those
   underwrite the conditional tier. Then Q1/Q5.
2. Build the §4 still-printed-withdrawn table and decide the merged disclosure.
3. Two zero-cost honesty fixes: report PK and AR-exact as ONE measurement
   (including 12K/16K std, the MLA 32K reversal, and the matched +47.7%); and
   reconcile `03_theory.tex:81` ("<1%") with the midpoint-confound defense.
4. Before relying on T2/T5/T6/E4, spot-check the numbers with the existing
   `scripts/analysis/` scripts (τ_surr factor, ∫1/ρ², the manifest AR column).

## 6. Verification note

These are adversarially generated candidate findings. The "🆕 / 🔁" tags reflect
comparison against the three self-audits; the highest-leverage mathematical items
(T2, T4, T5, T6) and the AR-exact=0 fact (E4) carry precise `file:line` and
recomputable paths and should be independently re-checked (per §5.4) before any
reviewer-facing use. Nothing here modifies a paper number or the claim tiers.
