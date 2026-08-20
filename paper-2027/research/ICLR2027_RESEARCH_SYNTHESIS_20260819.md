# ICLR 2027 research synthesis: finite RoPE basis and co-adaptation

- **Status:** canonical internal decision memo; architecture implemented in the
  active manuscript and retained as its claim/evidence contract
- **Created:** 2026-08-19
- **Last synchronized with manuscript:** 2026-08-20
- **Scope:** theory architecture, evidence routing, related-work positioning,
  and acceptance-oriented writing decisions
- **Not submission prose:** all numbers must still be checked against their
  canonical owner before entering the manuscript

## 1. Decision in one paragraph

The paper should remain *RoPE Has a Spectral Budget*, but its theoretical
center should change. A RoPE frequency is a two-dimensional positional
subspace, not a cosine feature. A finite table therefore defines a spectral
coordinate system whose phase-invariant subspace geometry limits static
identifiability. Training then learns coefficients that use this system, so
the table and model weights co-adapt. EVQ-Cosh is a simple analytic table that
exposes and exploits this allocation axis; it is not the universal optimizer
of full-RoPE geometry or language-model loss.

This is a stronger ICLR paper than either “we found a better extrapolation
schedule” or “cosh minimizes our surrogate.” It joins real theory, controlled
identification, mature-model evidence, and a clear relationship to LeRoPE.

## 2. Recommended central claim

> **Even at a fixed spectral range, the interior allocation of a finite RoPE
> table is an independent training-time design variable: it changes the full
> sin/cos subspace geometry and trained behaviour, while model weights co-adapt
> to the table used during training.**

The paper-facing coordinate form is

\[
x_k=-\log\omega_k=a+Rz_k,\qquad z_0=0,\ z_{K-1}=1.
\]

Here $(a,R)$ is sampled spectral support and $z$ is the normalized interior
allocation.  In the geometric family, $z_k=k/(K-1)$, so fixing $(a,R)$ fixes
the whole table.  The exact-range intervention changes only $z$.  The anchored
and deployed Cosh tables share exactly the same normalized $z$; anchoring
changes only the support embedding.  An "effective body base", median
wavelength, or slow-channel count is a summary of $z$, not an unheld scalar
confound.

| Clause | Evidence | Status |
| --- | --- | --- |
| Each frequency is a 2D sin/cos subspace | full self/cross Gram | proved |
| Phase-invariant redundancy exactly determines stable rank \(r_2\) | canonical correlations + exact stable-rank identity | proved |
| Interior placement changes what is learned at fixed range | exact-range + M4 | completed empirical evidence |
| Weights co-adapt to the training table | exact transplant obstruction + 50M 2x2 | proved exact case + completed empirical evidence |
| A fixed table learned elsewhere can transfer value into a new training run | Fixed-LeRoPE | external primary-source evidence |

## 3. Theory that survives adversarial review

### 3.1 Full-RoPE collision

For one pair,

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

With self Gram \(S_\omega\), cross Gram \(H_{\omega\nu}\), and whitened
cross-Gram

\[
Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2},
\]

the singular values of \(Q_{\omega\nu}\) are canonical correlations. The
pair redundancy

\[
c_{\omega\nu}=\frac{\sigma_1^2+\sigma_2^2}{2}
\]

is invariant to pair-internal phase rotations and basis changes. The current
cosine-only kernel is only one block of this object.

### 3.2 Exact effective-dimension relation

For the global block-whitened Gram \(R\),

\[
r_2(R)=\frac{2K}{1+(K-1)\bar c}.
\]

This exact stable-rank identity is the cleanest theoretical bridge from
pairwise redundancy to finite positional dimension. It is a static geometry
result, not a claim about task loss or extrapolation.

### 3.3 Low-frequency collapse

As \(\omega L\to0\),

\[
V_\omega\to\operatorname{span}\{1,\Delta\}.
\]

Many slow frequencies therefore become redundant copies of nearly the same
subspace. Under the attention softmax metric, the constant direction is
removed and the centered limit is spanned by \(\Delta\) and \(\Delta^2\).
The safe interpretation is **redundant, not necessarily unused**.

### 3.4 Exact post-hoc transplant obstruction

The repository already contains a complete proof for the exact invertible
case. If fixed, position-independent maps \(A,B\) satisfy

\[
A^\top R_{\Omega'}(\Delta)B=R_\Omega(\Delta)
\]

for all \(\Delta\) in an interval, then \(\Omega\) and \(\Omega'\) have the
same frequency multiset up to sign, permutation, and the integer-position
alias. At \(\Delta=0\), \(A^\top B=I\); differentiating the resulting
similarity relation at zero preserves the generator spectrum
\(\{\pm i\omega_k\}\).

The exact theorem is paper-ready. A quantitative lower bound for approximate
compensation remains open and is not required to state the exact result.

### 3.5 What remains true about Cosh

The Cosh density is the unique minimizer of the stated convex
\(\mathcal C_{\mathrm{app}}\). Its inverse CDF is a useful closed-form family
with geometric RoPE as the \(\tau\to0\) limit. This is a construction theorem,
not a full-RoPE or LM-optimality theorem. The finite \(\tau\) convention is an
operating choice.

## 4. Empirical evidence chain

### 4.1 Pure allocation

The exact-range study fixes sampled endpoints and log span and moves only 30
interior frequencies. The seed-42 effect is
`-0.47750/-0.20499/-0.11284` NLL at `512/1K/2K`. M4 then supplies a separate
12-configuration, three-seed direction check: the pre-specified `1.25x` Cosh
and matched exponential arms beat uniform in `10/12` and `9/12`
configurations; the formula point beats uniform in `7/12`.

This owns the claim that allocation is not reducible to scalar base or range.

### 4.2 Mature scale

The scale story is a sequence, not a pooled effect estimate.  A 1.485B
same-initialisation/same-scientific-recipe trajectory crosses in favour of EVQ
at 8K/16K.  Matched 300-step LLaMA-3-8B LoRA changes 16K/32K PPL from
`108.958/991.475` to `24.068/127.911`.  Matched OLMo Q/K-only adaptation keeps
4K 2Wiki exact at `22.0/21.5%` while producing `0/17.5%` at 8K.  Separate
RULER task-adaptation runs provide the 1.485B `2.02/31.63%` at 8K and 8B
`0.295/14.03%` at 16K.  Each protocol keeps its own endpoint and seed scope.

Single seed is not an automatic reason to hide a large controlled result.
Seed scope remains exact in the internal owner. The outward manuscript need
not volunteer a generic “single-seed limitation” unless required for a
sentence to remain true; it must not claim unsupported statistical
significance or universality.

### 4.3 Co-adaptation

In the 50M frozen-weight 2x2, the self-consistent Geo/Geo and EVQ/EVQ systems
have PPL `7.14/7.16`; post-hoc Geo/EVQ and EVQ/Geo mismatches have PPL
`76.20/23.05`. The loss interaction is `-3.5367`, much larger than either main
effect. Bare static rank improves in the worst mismatch cell, directly showing
that structural geometry alone cannot predict a trained model's loss after a
table swap.

Base-only controls recover part, not all, of the historical-table gap. This is
why pure interior allocation belongs to the exact-range experiment, while the
2x2 belongs to co-adaptation and retrofit diagnosis.

## 5. Related work: LeRoPE positioning

LeRoPE is not an existential threat to this paper; it forces a more precise
novelty claim.

Verified local facts:

- LeRoPE learns one scalar per frequency band, shared across layers and heads;
- it reports stable non-geometric profiles from 52M to 2.5B;
- at 217M, retraining from the start with a table learned in an independent run
  retains 63.6% of the full LeRoPE validation-PPL gain, versus 10.4% for
  partial RoPE.

| LeRoPE owns | This paper owns |
| --- | --- |
| Learned per-band tables | Phase-invariant full-RoPE subspace geometry |
| Evidence that a learned fixed table transfers value into a new training run | Exact fixed-range interior-allocation identification |
| Joint table/weight optimization | Exact frozen-retrofit obstruction and 2x2 co-adaptation diagnosis |
| Learned/search cost | Closed-form zero-learned-parameter EVQ construction |

Do not say EVQ approximates LeRoPE, LeRoPE validates EVQ extrapolation, or one
dominates the other without a matched comparison.

## 6. What the independent Claude audits add

### Full-RoPE audit

Use its derivation of the missing sin/cos blocks, canonical correlations,
low-frequency collapse, and the existence of cosine-order and length-order
counterexamples. Do not copy its report verbatim: its current one-sided
quadrature check required a resolution/tolerance repair before passing; an
older causal example was later rejected by its own verification script; and its chosen
\(G_{\mathrm{attn}}\) interpretation is narrower than the softmax/LM-gradient
separation in the canonical report.

Its v2 also constructs exact same-parity harmonic grids
\(\omega_k=\pi a_k/L\) with full static rank when capacity permits. This is a
useful exact static-geometry result, but it reinforces rather than solves the
task problem: the construction is locked to the chosen window and should not
be promoted as a new long-context schedule during the current rewrite.

### Dependency-spectrum audit

Use it as an internal falsification result: no unmeasured distance prior should
be promoted into a universal optimal density. Its additive utility collapses
all channels to one frequency, and its candidate density rankings depend on
the chosen kernel. The checkpoint pilot measures a steep local gradient
spectrum, approximately \(r^{-2.4}\) on one small sample, but that object mixes
content, softmax probability, downstream gradients, and trained-model state.
It is not a new ICLR claim without a promoted owner.

## 7. Claims to reject

- Static collision, rank, or logdet improvement implies better extrapolation.
- Cosh is the full-RoPE or task-loss optimum.
- Slow bands are dead or freely reclaimable.
- The multi-source RULER split proves an exclusive regime-II mechanism.
- A single shared table is mathematically unable to serve both length regimes.
- R1/R2/R3 exhaust every possible design.
- \(\rho^*=\sqrt p\), \(p^{1/3}\), or another smooth density is a universal
  attention-derived solution.
- LeRoPE and EVQ are the same mechanism from opposite ends.
- The unsigned A.15 structural-curvature density $\rho\propto w^{1/3}$ predicts
  LeRoPE's learned profile.  The CPU-only oracle audit places it beyond EVQ in
  the direction away from LeRoPE; a signed, trajectory-aware LM-risk object
  would be required instead.

## 8. Implemented nine-page architecture

1. **Introduction:** finite table as a training-time spectral coordinate
   system; exact-range result; mature-scale headline.
2. **Related work:** range transport, scalar base, learned/searched tables,
   fixed analytic constructions; LeRoPE stated early and accurately.
3. **Theory:** full-RoPE canonical collision, exact stable-rank identity,
   low-frequency collapse, exact transplant obstruction.
4. **Construction:** compress \(\mathcal C_{\mathrm{app}}\), inverse-CDF Cosh,
   and the operating convention into a short subsection; proofs and detailed
   \(\tau\) derivation move to the appendix.
5. **Experiments:** exact-range/M4 first; 50M 2x2 if it fits; 1.485B and 8B
   mature evidence next.
6. **Discussion:** static identifiability versus trained use; exact versus
   approximate retrofit; range transport and LeRoPE complementarity.

Use the full nine-page allowance by replacing low-leverage material. Do not
compress a strong result merely to create empty space, and do not stack the new
theory on top of the old surrogate-heavy narrative.

## 9. Acceptance-first writing principles

- The objective is to maximize ICLR 2027 acceptance probability, subject only
  to scientific truth and avoiding experiments that cannot change the paper.
- The work is already a strong theory paper with substantial experiments. The
  writing must make that strength legible instead of sounding like an audit,
  rebuttal appendix, or AI-generated status report.
- Confident packaging and the strongest accurate interpretation are allowed.
  Fabricated data, false statements, unsupported SOTA/universal/significance
  claims, and protocol splicing are not.
- A strong single-seed controlled result is usable. Record seed scope
  internally; do not automatically weaken outward prose with a generic caveat.
- Lead with one memorable claim, the strongest theorem, and the decisive
  number. Do not give every experiment equal space.
- Use plain human language before notation. Every theorem must have one clear
  experimental or conceptual consequence.
- Internal audits are exhaustive; outward prose is selective and accurate.
  A known boundary limits the claim but is not automatically a sentence in the
  paper.

## 10. Compaction packet

### NON_NEGOTIABLE_CONSTRAINTS

- `paper-2027/` is the only active manuscript; `paper/` is immutable.
- Maximize acceptance probability without fabrication, false attribution,
  unsupported statistics, or junk experiments.
- Preserve raw owners, protocol identity, metric identity, and result hashes.
- Do not call static geometry an extrapolation or LM-quality predictor.
- Do not call EVQ-Cosh a universal optimum.
- Do not start GPU training or paid execution without explicit authorization.
- Preserve unrelated worktree changes; do not stage, commit, or push without
  explicit authorization.

### ARCHITECTURE_DECISIONS

- Central theory: full sin/cos subspaces, canonical collision, stable-rank
  identity, low-frequency collapse, exact transplant obstruction.
- Central empirical chain: exact-range/M4 -> 50M co-adaptation diagnostic ->
  1.485B/8B mature persistence.
- EVQ-Cosh: constructive instance and controlled intervention.
- LeRoPE: external learned-table and fixed-table evidence, not mechanism
  equivalence.
- Main text: result-first, human-readable, nine pages fully used.

### FILE_LEDGER

| Path | Role | State |
| --- | --- | --- |
| `paper-2027/research/README.md` | durable research index | created |
| `paper-2027/research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md` | canonical rewrite decision memo | created |
| `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` | canonical technical report | existing |
| `paper-2027/research/audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md` | independent-audit record | created |
| `paper-2027/research/audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md` | dependency-audit record | created |
| `AGENTS.md` | project objective, routing, and claim boundaries | modified |
| `paper-2027/HANDOFF.md` | sole volatile manuscript/build/next-action state | current |
| `paper-2027/README.md` | active-package entrypoint | modified |

### REJECTED_APPROACHES

- Retain cosine-only collision as the true RoPE geometry.
- Derive extrapolation claims from static collision/rank.
- Replace the paper with a new dependency-density or routed-attention method.
- Treat every internal negative as outward-facing disclosure material.
- Hide the main result behind owner/tier/status language.
- Require multi-seed execution solely for cosmetic symmetry when a decisive
  controlled result already answers the question.

### RISKY_REGIONS

- `paper-2027/sections/03_theory.tex`: keep the full-RoPE geometry primary and
  the surrogate explicitly conditional; do not restore the cosine-only theory.
- `paper-2027/appendix/a1_proofs.tex`: preserve the exact-vs-modelled boundary
  and keep removed auxiliary theory out of the main narrative.
- `paper-2027/sections/04_experiments.tex`: keep pure allocation,
  co-adaptation, and mature persistence as distinct causal layers.
- `paper-2027/sections/05_discussion.tex`: keep one theory-to-evidence chain;
  do not restore the post-hoc multi-source mechanism split.
- `results/dependency_spectrum_audit_20260819/`: ignored/volatile raw output;
  cite the durable audit record and recheck raw files before promotion.
- Page budget: the body is already nine pages; every addition needs a named
  replacement.

## 11. Next action

Keep the architecture fixed while independent AI cross-reviews are pending.
When the user supplies them, treat them as adversarial hypotheses: verify each
alleged defect against the PDF, source, theorem, and owner; rank only verified
issues by positive-reviewer score ceiling, technical-reviewer score ceiling,
human readability, and submission validity. Make replacement-level changes
only for issues that can affect acceptance. Do not launch a new experiment or
expand the story merely because an external model proposes one.
