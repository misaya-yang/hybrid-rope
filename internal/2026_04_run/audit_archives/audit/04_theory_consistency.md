# Auditor 4 — Theory consistency + numerical reproduction

**Scope.** Independent reproduction of (i) the closed-form q(x), (ii) the
Q_1(L,b) integral and the c_pred = sqrt(45·Q_1) prefactor over the 8-row
handover grid, plus consistency sweeps for Theorem 1 (body↔proof),
the d_head/d_rot/d_rope/d_eff/K convention, the S_χ² normalization, and
the τ²_* prefactor.

**Reproducibility.** All numerics done from scratch in
`audit/scripts/04_q_x_verify.py` and `audit/scripts/04_Q1_grid.py`,
using `scipy.integrate.quad` (no reuse of paper code). Python:
`/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python`
(scipy 1.17, numpy). Severity tags: P0 = numerical/algebraic
inconsistency body↔appendix; P1 = wording divergence / convention drift
/ prefactor disagreement; P2 = stylistic.

---

## 1. Methodology

For each task I (a) recomputed values from first principles in fresh
scripts, then (b) cross-checked the recomputed numbers against the
explicit numerical disclosures in `paper/sections/03_theory.tex`,
`paper/appendix/a1_proofs.tex`, `paper/tables/table_lambda_cv.tex`, and
the handover doc. The recomputation never reads existing repo code; it
re-derives q(x) from `Var_{t∼U[0,1]}[cos(xt)]` and Q_1 from
`∫_0^1 a(φ) q(L b^{-φ}) dφ`. For the body↔proof check I read
`03_theory.tex` lines 39–48 (Theorem~ref{thm:ode}) and
`a1_proofs.tex` lines 1–47 (subsection "Collision-only surrogate
stationary density") side by side. For the convention sweep I grepped
`d_{\mathrm{head}}`, `d_{\mathrm{rot}}`, `d_{\mathrm{rope}}`,
`d_{\mathrm{eff}}`, ` K=`, ` K =`, `S_{\chi^2}`, `\chi^2` across
`paper/sections/` and `paper/appendix/`.

---

## 2. q(x) verification

Script: `audit/scripts/04_q_x_verify.py`. q(x) is computed two ways:
(i) closed form `1/2 + sin(2x)/(4x) - (sin x/x)²`; (ii) splitting
`Var = E[cos²] - (E[cos])²` and integrating each via adaptive `quad`
with chunked subintervals (chunks ∝ x to handle oscillation up to
x = 10⁴).

| x | q (closed form) | q (numerical) | abs err | rel err |
|---:|---:|---:|---:|---:|
| 0.1 | 0.0000022190 | 0.0000022190 | 4.4e-16 | 2.0e-10 |
| 1   | 0.0192509384 | 0.0192509384 | 2.2e-16 | 1.2e-14 |
| 5   | 0.4360175139 | 0.4360175139 | 5.6e-17 | 1.3e-16 |
| 25  | 0.4973482243 | 0.4973482243 | 1.7e-16 | 3.3e-16 |
| 100 | 0.4977911161 | 0.4977911161 | 2.2e-16 | 4.5e-16 |
| 1000 | 0.5002318261 | 0.5002318261 | 1.6e-15 | 3.1e-15 |
| 10000 | 0.5000145487 | 0.5000145487 | 2.9e-14 | 5.8e-14 |

**Max abs err = 2.92e-14, max rel err = 2.00e-10** (the latter at x=0.1
where the closed form is itself O(1e-6) so the relative blow-up is
expected). The closed form `q(x) = 1/2 + sin(2x)/(4x) - (sin x/x)²` in
`03_theory.tex:103` and the probabilistic identity in
`a1_proofs.tex:264-268` are reproduced to machine precision: ✅.

---

## 3. Q_1(L,b) grid recompute

Script: `audit/scripts/04_Q1_grid.py`. Direct adaptive integration of
`Q_1(L,b) = ∫_0^1 a(φ) q(L b^{-φ}) dφ` with `a(φ)=(1-φ)²/2 - 1/6`
(matches `eta(phi)` in `03_theory.tex:103` and `a(phi)` in
`leff-derivation` `a1_proofs.tex:444`).

### 3.1 Handover-doc 8-row grid

| L | b | Q_1 (audit) | c_pred (audit) | Q_1 (handover) | c_pred (handover) | verdict |
|---:|---:|---:|---:|---:|---:|:---|
| 128 | 10 K | 0.031688 | 1.1941 | 0.0317 | 1.194 | ✅ matches (≤0.01%) |
| 1024 | 10 K | 0.024135 | 1.0421 | 0.0241 | 1.042 | ✅ matches |
| 4096 | 10 K | 0.014123 | 0.7972 | 0.0141 | 0.797 | ✅ matches |
| 8192 | 10 K | 0.008300 | 0.6111 | 0.0083 | 0.611 | ✅ matches |
| 128 | 500 K | 0.030090 | 1.1636 | 0.0301 | 1.164 | ✅ matches |
| 2048 | 500 K | 0.030515 | 1.1718 | 0.0305 | 1.172 | ✅ matches |
| 4096 | 500 K | 0.028782 | 1.1381 | 0.0288 | 1.138 | ✅ matches |
| 8192 | 500 K | 0.026461 | 1.0912 | 0.0265 | 1.091 | ✅ matches |

All 8 rows match the handover doc to displayed precision (4-decimal
c_pred, 4-decimal Q_1).

### 3.2 `tables/table_lambda_cv.tex` (b=500K, L∈{256,512,1024})

The table is independent of d_head (Q_1 depends only on L,b), and lists
the same Q_1 in three rows per L. Recomputed:

| L | Q_1 (audit) | Q_1 (table) | c_pred (audit) | c_pred (table) | verdict |
|---:|---:|---:|---:|---:|:---|
| 256  | 0.03145 | 0.03145 | 1.1896 | 1.190 | ✅ matches |
| 512  | 0.03192 | 0.03192 | 1.1985 | 1.199 | ✅ matches |
| 1024 | 0.03159 | 0.03159 | 1.1922 | 1.192 | ✅ matches |

`table_lambda_cv.tex` Q_1 column reproduced to 5 decimals. ✅.

### 3.3 `a1_proofs.tex:255` b-dependence claim

Body & appendix both quote: at b=500K across L∈[128,4096], c_pred ∈
[1.14, 1.20]; at b=100K, c_pred ∈ [1.06, 1.20]; at b=10K, c_pred → 0.94
at L=2048 and 0.80 at L=4096.

| Range / point | claim | recomputed | verdict |
|:---|:---|:---|:---|
| b=500K, L∈[128,4096] | [1.14, 1.20] | min=1.138 (L=4096), max=1.199 (L=512) | ✅ matches |
| b=100K, L∈[128,4096] | [1.06, 1.20] | min=1.060 (L=4096), max=1.198 (L=256) | ✅ matches |
| b=10K, L=2048 | 0.94 | 0.9364 | ✅ matches |
| b=10K, L=4096 | 0.80 | 0.7972 | ✅ matches |

### 3.4 `a1_proofs.tex:574` "a posteriori consistency" check

Claim: at b=500K, L∈{256,512,1024}, Q_1 ∈ [0.0314, 0.0319]; c_pred ≈ 1.19.

Recomputed: Q_1 ∈ [0.03145, 0.03192]; c_pred ∈ [1.1896, 1.1985] →
quoted "≈1.19" is correct. ✅.

**Verdict on §3.** Every disclosed numerical value of Q_1, c_pred, and
the b-dependence ranges is reproduced from scratch within rounding —
no numerical drift anywhere.

---

## 4. Theorem 1 body ↔ proof check

**Body** (`paper/sections/03_theory.tex:39-48`, Theorem~ref{thm:ode}
"Exact stationary allocation under the broadband surrogate"):
hypotheses α>0, β≥0; constraints ρ ∈ C²([0,1]), ρ>0, ∫ρ=1; conclusion
ρ_τ(φ) = τ cosh(τ(1-φ))/sinh τ with τ=√(β/α); β=0 case explicitly
stated.

**Proof** (`paper/appendix/a1_proofs.tex:1-47`, subsection
"Collision-only surrogate stationary density") closes the body
statement.

| Required item | Body | Proof (a1) | Match |
|:---|:---|:---|:---|
| (a) ρ_τ(φ) = τ·cosh(τ(1-φ))/sinh τ | line 43 | line 37 (eq:rho-tau-closed) | ✅ identical |
| (b) ρ'(0)=-τ², ρ'(1)=0 derivation | line 37 (refers to Appendix) | lines 30-35 (full derivation via g'(0)=∫ρ=1, g'(1)=0) | ✅ derived; mass constraint enters via BC |
| (c) β=0 limit | "at β=0 the minimizer is the τ→0 limit ρ_0≡1 (uniform/geometric)" line 45 | implicit only: ρ_τ → 1 as τ→0 follows from τ/sinh τ → 1, but the proof never substitutes β=0 explicitly | ⚠️ P1 (see finding F1) |
| (d) Regularity (ρ ∈ C²([0,1])) | C²([0,1]) — line 41 | line 47 explicitly relaxes to L²([0,1]) ⊃ C²([0,1]) for the PSD/quadratic-form argument | ⚠️ P2 (see finding F2) |

### F1 (P1) — β=0 case is implicit, not literally proven

`03_theory.tex:45` states "at β=0 the minimizer is the τ→0 limit
ρ_0 ≡ 1 (uniform/geometric)". The proof at `a1_proofs.tex:1-40`
tacitly assumes β>0 (so τ=√(β/α) is well-defined and the substitution
ρ = C_1cosh(τφ) + C_2sinh(τφ) is non-degenerate). For β=0 the ODE
collapses to ρ''=0 with ρ'(0) = ρ'(1) = 0 and ∫ρ=1 → ρ≡1; this is the
endpoint of `lim τ→0 ρ_τ = 1` (which IS true, since
τ·cosh(τ(1-φ))/sinh τ → 1 + O(τ²) as τ→0). The body claim is
mathematically sound, but the proof never writes "set β=0; the
substitution ρ=C_1+C_2φ with the BCs ρ'(0)=ρ'(1)=0 and ∫ρ=1 forces
ρ≡1." A one-line addition at `a1_proofs.tex:40` would close this gap.

Severity P1 because it does not affect any quantitative claim and the
limit is unambiguous; but a careful reader will notice the appendix
proof does not explicitly handle the β=0 case the body promises.

### F2 (P2) — Regularity widening is one-sided

Body says ρ ∈ C²([0,1]) (theorem hypothesis at line 41); the appendix
PSD paragraph at line 47 says the surrogate is well-defined on
L²([0,1])⊃C²([0,1]). The widening makes the bound stronger, not
weaker, so this is a P2 stylistic divergence — body could note "the
proof relaxes to L² without altering the conclusion."

### Other items

- **PSD/Convexity** (body line 41 "convex" + Appendix~ref{sec:proofs};
  appendix lines 42-47 with the explicit min-kernel identity
  ∫(∫_s^1 f)² ds ≥ 0): ✅ proof provided, body justifies the existence
  of a unique minimizer.
- **Existence/uniqueness of minimizer** (body claim "the unique
  constrained minimizer" line 41): ✅ supported by the PSD argument at
  line 47.
- **Pure-tether vs. forced** (body line 48 / appendix
  `subsec:fisher-ext` line 49-63): ✅ identical β=0/γ=0 framing on
  both sides.

---

## 5. Symbol convention sweep (d_head, d_rot, d_rope, d_eff, K)

§3.1 (`03_theory.tex:19`) and §3.5 Eq.(4) (`03_theory.tex:79-81`)
define:
- d_rot = #RoPE-applied dims; for MHA d_rot = d_head, for MLA d_rot < d_head;
- K = d_rot/2 = channel-pair count;
- d_eff = effective head dim used in τ = d_eff/√L; chosen as d_head for
  both MHA and MLA per the calibration disclosure.

### Survey table

| File:line | Symbol used | Local use / inferred meaning | Verdict |
|:---|:---|:---|:---|
| `03_theory.tex:19` | d_rot, K=d_rot/2, d_eff, d_head | full convention statement | reference |
| `03_theory.tex:73` | d_eff | τ=d_eff/√L | ✅ matches §3.1 |
| `03_theory.tex:79-81` | d_rot, K=d_rot/2, d_eff=d_head MLA | Eq.(4) restates convention | ✅ matches |
| `03_theory.tex:99-105` | d_head | Prop. softmax-transport (S_χ², U, τ²_*) | ✅ MHA convention; consistent with calibration choice d_eff=d_head |
| `03_theory.tex:115` | (no dim symbol; uses c_pred) | basin paragraph | ✅ |
| `a1_proofs.tex:147` | K=d_head/2 | Habitable-Zone bound | ⚠️ should be K=d_rot/2 — see F3 |
| `a1_proofs.tex:259` | d_head | τ²_* = 45·λ·Q_1·d²_head/L | ✅ matches §3.7 |
| `a1_proofs.tex:289` | α≈1/(2K) = 1/d_head | surrogate diagonal fit (MHA) | ✅ if d_rot=d_head, but a1:548 contradicts (see F4) |
| `a1_proofs.tex:312` | "via α=1/d_head" summary | scaling derivation | ✅ for MHA; loose for MLA |
| `a1_proofs.tex:386` | d_head | S_χ²(τ) closed form, normalized | ✅ matches §3.7 c_S definition |
| `a1_proofs.tex:399` | K=d_head/2 | LoRA phase-transition formula | ⚠️ MHA-specific; OK because LoRA section is LLaMA-only |
| `a1_proofs.tex:474` | d²_head/L_eff^J | trained-attention extension | ✅ matches §3.7 |
| `a1_proofs.tex:548` | "α=1/d_rot derived in §sec:tau-scaling" | Normalization convention paragraph | ⚠️ contradicts a1:289 which says α=1/d_head — F4 |
| `a1_proofs.tex:614` | K=16 ↔ d_rope=32; K=64 ↔ d_head=128 | MLA scaling | ✅ K=d_rope/2 for MLA, K=d_head/2 for MHA |
| `a3_supporting_results.tex:6` | d_rope (DeepSeek MLA notation) explicitly equated to d_rot in §sec:theory | calibration paragraph | ✅ explicit reconciliation |
| `a3_supporting_results.tex:10` | "α = 1/(2K) = 1/d_rope" | "Two distinct dimensions" paragraph | ✅ correct MLA-side statement; consistent with α=1/d_rot |
| `05_experiments.tex:51` | d_rope=32 (16 channels after pairing) | MLA scarce-channel disclosure | ✅ |

### F3 (P1) — `a1_proofs.tex:147` uses K=d_head/2 (should be d_rot/2)

Looking at line 147: "For the EVQ-Cosh mapping with K = d_head/2
channels..." — this is silently MHA-specific. For MLA we have
K = d_rope/2 ≠ d_head/2. Either the line is implicitly assuming MHA
(in which case OK but undisclosed), or it is sloppy. Recommend changing
to `K = d_rot/2` (the §3.1 convention) for full consistency. P1.

### F4 (P1) — `a1_proofs.tex:548` says "α=1/d_rot derived in §sec:tau-scaling", but `a1_proofs.tex:289` (§sec:tau-scaling) actually derives α≈1/(2K) = 1/d_head

Direct quote, a1:548: "the d_head factor enters via the surrogate
diagonal α=1/d_rot derived in §ref{sec:tau-scaling}".

But §sec:tau-scaling at a1:289 fits α via:
> `α ≈ 1/(2K) = 1/d_head, β ∼ L^{-0.22}, O(1).`

For MHA (where d_rot = d_head and K = d_rot/2 = d_head/2), 1/(2K) =
1/d_head = 1/d_rot, so α=1/d_rot and α=1/d_head agree numerically.
However the §sec:tau-scaling derivation uses uniform spacing in [0,1]
on K = d_head/2 channels (line 287: "evaluated at K = d_head/2 uniformly
spaced channel positions"), so the natural variable is d_head (= 2K)
in the MHA derivation. The line at 548 calls it d_rot to gesture at
generality, but the body of §sec:tau-scaling never derives the more
general α=1/d_rot — only α=1/(2K)=1/d_head with K=d_head/2 baked in.
For MLA where K=d_rope/2≠d_head/2, the derivation would need to be
redone with K=d_rope/2, yielding α=1/d_rope. `a3_supporting_results.tex:10`
correctly states α=1/(2K)=1/d_rope for MLA, which is consistent with
"α=1/d_rot" only if we take d_rot ≡ d_rope (which §3.1 does).

Net: the d_rot ↔ d_rope ↔ d_head identification is correct in spirit
but the §548 cross-reference to §sec:tau-scaling is misleading since
that section derives α only for the MHA case. **Suggested fix**:
either (i) rephrase §548 as "α=1/d_rot from `a3_supporting_results:10`
in the MLA case and α=1/d_head from §sec:tau-scaling in the MHA case",
or (ii) update §sec:tau-scaling to write α=1/d_rot generically.
Severity P1.

### F5 (P2) — body §3.7 line 99 puts d_head in `S_χ²(τ) = c_S τ⁴/d_head + O(τ⁶)` without restating the d_rot vs. d_head distinction

The body says "d_head" because the deployed convention sets
d_eff = d_head for both MHA and MLA. The derivation passes through
α = 1/d_rot (which under MLA = 1/d_rope). For MLA the substitution
would naïvely give c_S τ⁴/d_rope (not /d_head); the d_eff = d_head
calibration is what restores the d_head factor at the operating-point
formula. `a3_supporting_results.tex:10` calls this out explicitly
("the variational α=1/d_rope derivation alone would give a much
smaller deployment τ; we deploy d_eff=d_head=128 as a calibrated
convention"), and §3.5 line 81 references it. Body §3.7 line 99 does
NOT remind the reader of this. Severity P2 (already disclosed in
§3.1 and §A.3, just not echoed in §3.7).

---

## 6. S_χ² normalization sweep

The "Normalization convention" paragraph at `a1_proofs.tex:548` (added
in commit 2b33a59) states:

> Throughout this paper, S_χ²(τ) denotes the d_head-normalized
> stiffness S_χ²(τ) := (1/d_head) ∫₀¹ (1-ρ_τ)² / ρ_τ dφ, with leading
> expansion τ⁴/(45 d_head) + O(τ⁶) consistent with Proposition. Identities
> below that produce a bare τ⁴/45 leading coefficient apply to
> d_head·S_χ²(τ) (the un-normalized integral).

I cataloged every S_χ²/χ² use:

| Site | Form used | Normalization | Verdict |
|:---|:---|:---|:---|
| `03_theory.tex:99` | S_χ²(τ) = c_S τ⁴/d_head, c_S=1/45 | normalized (matches a1:548 declared convention) | ✅ |
| `03_theory.tex:103` | F(τ) = (1/2)S_χ²(τ) - λU(τ,L) | normalized (Dirichlet-energy convention with 1/2 prefactor) | ✅ |
| `03_theory.tex:113` | "Pearson χ² stiffness S_χ²(ρ)=Var_{ρdφ}(1/ρ)" | un-normalized name, but S_χ²(ρ)=∫(1-ρ)²/ρ ≠ τ⁴/(45 d_head) — see F6 | ⚠️ P1 |
| `a1_proofs.tex:163` | χ²(U‖P_ρ) = ∫(1-ρ)²/ρ dφ | un-normalized (no 1/d_head); used inside W(ρ), not as the stiffness in F(τ) | ✅ — different functional, distinct symbol χ² vs. S_χ² |
| `a1_proofs.tex:189` | S_χ²(ρ_τ) = sinh τ/τ² · arctan(sinh τ) - 1 = τ⁴/45 + O(τ⁶) | un-normalized integral (no 1/d_head) — denoted with the same symbol S_χ² | ⚠️ P1 — F7 |
| `a1_proofs.tex:194` | W(ρ_τ) = (α/45 + 2β/945)τ⁴ | mixed: α=1/d_rot makes α/45 = 1/(45 d_rot) | ✅ consistent (uses α explicitly) |
| `a1_proofs.tex:257` | (1/2)S_χ²(τ) = τ⁴/(90 d_head) | normalized (consistent with §548) | ✅ |
| `a1_proofs.tex:386` | S_χ²(τ) = (1/d_head)[sinh τ · arctan(sinh τ)/τ² - 1] | normalized | ✅ |
| `a1_proofs.tex:397` | S_total = S_χ²(τ) + Λ₀(1-r/K) τ²/d_head | normalized (the τ²/d_head term has explicit d_head) | ✅ |
| `a1_proofs.tex:474` | F(τ) = S_χ²(τ) - λU; τ²_* = 45 λ Q_1^{eff} d²_head/L_eff^J | normalized (no extra d_head needed) | ✅ |
| `a1_proofs.tex:488` | F(τ) = S_χ²(τ) - λU(τ,p) | normalized | ✅ |
| `a1_proofs.tex:552` | S(ρ) = ∫(1-ρ)²/ρ dφ = S_χ²(ρ) | un-normalized integral named S_χ²(ρ) (axiomatic identity) | ⚠️ P2 — F8 |
| `a1_proofs.tex:566` | λ_true = ∂_θ S_χ²(...)/∂_θ U_exact(...) | normalized (ratio invariant) | ✅ |
| `a1_proofs.tex:569` | d_head · S_χ²(τ) = sinh τ/τ²·arctan(sinh τ)-1 = τ⁴/45 | un-normalized form, explicitly multiplied by d_head | ✅ correctly disclosed |

### F6 (P1) — body §3.7 line 113 calls `S_χ²(ρ) = Var_{ρdφ}(1/ρ)` "the canonical load-variance penalty"

The expression `Var_{ρdφ}(1/ρ)` equals `∫(1-ρ)²/ρ dφ` (the
un-normalized integral, identity at `a1_proofs.tex:552`). Yet body
line 99 in the same section defines `S_χ²(τ) = τ⁴/(45 d_head) + ...`
which corresponds to (1/d_head) times the integral. The two
identifications can only both hold if the reader understands that
`S_χ²(ρ)` (a functional of ρ) and `S_χ²(τ)` (the special evaluation
at ρ_τ, normalized) are kept apart — but the body never makes this
explicit, and the appendix Normalization-convention paragraph (a1:548)
is what reconciles them, 14 pages later. Severity P1: a one-clause
parenthetical in body line 113 ("the d_head-normalized version
appearing in line 99 differs by 1/d_head from the bare integral
∫(1-ρ)²/ρ") would close this. Currently the only disambiguator is
the appendix paragraph 359 LaTeX lines downstream of its first use.

### F7 (P1) — `a1_proofs.tex:189` uses `S_χ²(ρ_τ)` for the un-normalized integral

Same symbol, two different normalizations, separated by 359 lines.
The Normalization-convention paragraph at a1:548 explicitly disclaims
this ("Identities below that produce a bare τ⁴/45 leading coefficient
apply to d_head·S_χ²(τ) (the un-normalized integral)"). The disclaimer
covers the case but appears AFTER the use at a1:189 (waterbed
section). For a reader walking the appendix front-to-back, the first
encounter at a1:189 has S_χ²(ρ_τ) = τ⁴/45, while §sec:chi2-load at
line 548 declares S_χ²(τ) = τ⁴/(45 d_head). A reader reasonably
infers the symbol is reused with two normalizations and may mis-apply
the τ⁴/45 form in §3.7. **Suggested fix**: at a1:189 add `(here the
un-normalized integral; cf. Normalization convention,
§sec:chi2-load)`. Severity P1.

### F8 (P2) — `a1_proofs.tex:552` axiomatic identity S(ρ) = S_χ²(ρ)

Same un-normalized integral aliased with the same symbol. Already
covered by a1:548 disclaimer two paragraphs above (lines 548 vs.
552). Severity P2 (resolved by adjacency).

---

## 7. τ²_* prefactor sanity check

§3.7 Proposition (`03_theory.tex:105`):
> τ²_* = 45·λ·Q_1(L,b)·d²_head / L.

Appendix derivation (`a1_proofs.tex:257-260`):
> Setting (1/2)S_χ²(τ) = τ⁴/(90 d_head), U = (d_head/L)[Q_0 + τ² Q_1],
> then ∂F/∂τ = 0 gives
> τ²_* = 45·λ·Q_1·d²_head / L.

### Algebra spot-check

F(τ) = (1/2)S_χ²(τ) - λU(τ,L) = τ⁴/(90 d_head) - λ(d_head/L)[Q_0 + τ² Q_1] + O(τ⁶)
∂F/∂(τ²) = τ²/(45 d_head) - λ(d_head/L)Q_1 = 0
→ τ²_* = 45·λ·Q_1·d²_head/L. ✅

The constant **45 = 1/c_S** holds because c_S = 1/45 and the
substitution (1/2) c_S τ⁴/d_head means stationarity of τ²/(2 c_S d_head)
- λ d_head Q_1/L = 0 in θ=τ², giving θ_* = c_S^{-1}·λ·d²_head·Q_1/L =
45·λ·d²_head·Q_1/L. ✅ Fully consistent.

### Every appearance of the 45 prefactor

| File:line | Quote | Verdict |
|:---|:---|:---|
| `03_theory.tex:99` | `c_S = 1/45` | ✅ |
| `03_theory.tex:105` | `τ²_* = 45·λ·Q_1·d²_head/L` | ✅ |
| `03_theory.tex:113` | `c_pred = √(45 Q_1) ≈ 1.19` | ✅ |
| `03_theory.tex:115` | `c_pred(L,b) = √(45·Q_1(L,b))` | ✅ |
| `a1_proofs.tex:189` | `S_χ²(ρ_τ) = τ⁴/45 + ...` (un-normalized) | ✅ if reader reads a1:548 first; see F7 |
| `a1_proofs.tex:194` | `(α/45 + 2β/945)τ⁴` | ✅ |
| `a1_proofs.tex:253-255` | basin paragraph: `√(45 Q_1) ≈ 1.19`, `c_pred(L,b)=√(45·Q_1(L,b))` | ✅ |
| `a1_proofs.tex:257-260` | derivation: `τ⁴/(90 d_head)` and `45·λ·Q_1·d²_head/L` | ✅ |
| `a1_proofs.tex:261` | `c_pred(L,b;λ) = √(45 λ Q_1)` | ✅ |
| `a1_proofs.tex:386` | `S_χ²(τ) = (1/d_head)[...]` whose Taylor at small τ is `τ⁴/(45 d_head)` | ✅ |
| `a1_proofs.tex:474` | `τ²_* = 45·λ·Q_1^{eff}·d²_head/L_eff^J` | ✅ |
| `a1_proofs.tex:488` | `τ²_* = 45 λ Q_1^J(p,b) d²_head/L_eff^J + ...` | ✅ |
| `a1_proofs.tex:530` | `‖ρ_EVQ - 1‖_2 = τ²/√45` | ✅ (consistent with τ⁴/45 leading order) |
| `a1_proofs.tex:569` | `d_head·S_χ²(τ) = ... = τ⁴/45 + ...` | ✅ |
| `a1_proofs.tex:574` | `c_pred = √(45 Q_1)` | ✅ |

**Search for rogue "5" / "60" / other prefactors**: none found.
Every single 45 appearance is consistent with c_S = 1/45 under the
declared `(1/2)S_χ²` convention. ✅.

The factor 45 has a clean derivation chain:
1. q(x) Taylor: q(x) = x⁴/45 + O(x⁶) for x ≪ 1 (mentioned at
   `a1_proofs.tex:261`, "the leading Taylor coefficient is x⁴/45,
   not x²/45").
2. S_χ²(ρ_τ) = (sinh τ · arctan(sinh τ)/τ² - 1)/d_head expanded:
   τ⁴/(45 d_head) + O(τ⁶) (a1:386,569).
3. Stationarity in θ=τ² of (1/2)S_χ² - λU yields θ_*/(45 d_head) =
   λ d_head Q_1/L → θ_* = 45·λ·Q_1·d²_head/L (a1:257-260).

All three uses of 45 trace to the same Taylor coefficient. No
algebraic inconsistency. ✅.

---

## 8. Summary

**Numerical reproduction (Tasks 1, 2)**: ✅ Every disclosed value (q(x)
closed form, the 8-row Q_1/c_pred grid, the table_lambda_cv 9-row
implicit Q_1, the §3.7/§A.10 b-dependence ranges, the §A.11 a posteriori
check) is reproduced from scratch within rounding. No P0 numerical
mismatch anywhere. Max abs error in q(x): 2.92e-14. Max relative
deviation in c_pred vs. handover: ~0.05% (rounding in handover doc's
4-decimal Q_1 column). Independent recompute confirms the paper's
single point of c_pred ≈ 1.19 and the b-dependence statements.

**Theorem 1 body↔proof (Task 3)**: ρ_τ closed form, BC derivation,
PSD argument all match. β=0 limit (F1, P1) is mathematically OK but
not literally written in the appendix proof. Regularity (F2, P2)
widening from C² to L² is one-sided and benign.

**Symbol convention (Task 4)**: §3.1 declares the convention cleanly.
Two real issues: F3 (a1:147 silently uses K=d_head/2 instead of
K=d_rot/2) and F4 (a1:548 cross-reference says α=1/d_rot is "derived
in §sec:tau-scaling" but that section actually only derives the
MHA-specific α=1/(2K)=1/d_head form — generic α=1/d_rot is asserted,
not derived). Both P1, neither alters any quantitative claim, both
fixable in one line.

**S_χ² normalization (Task 5)**: F6, F7 (both P1) flag that the same
symbol S_χ² is used for both the un-normalized integral (a1:189,
552, 569 second member) and the d_head-normalized stiffness (§3.7,
a1:386). The Normalization-convention paragraph at a1:548 covers
this, but appears AFTER the first body and appendix uses. Body
line 113 conflates `Var_{ρdφ}(1/ρ)` (un-normalized) with the §3.7
definition (normalized) without disambiguation in the body itself.
F8 (P2) is resolved by paragraph adjacency.

**τ²_* prefactor (Task 6)**: ✅ Exactly 45 everywhere; derivation chain
q(x) Taylor → S_χ² Taylor → stationarity all consistent. No rogue 5,
60, or other constants. The 45 = 1/c_S relation holds.

### Severity roll-up

| Sev | Count | Findings |
|:---:|:---:|:---|
| P0 | 0 | — |
| P1 | 4 | F1 (β=0 case implicit), F3 (K=d_rot/2 vs. K=d_head/2 at a1:147), F4 (α=1/d_rot vs. α=1/d_head cross-ref), F6+F7 (S_χ² same symbol two normalizations, disclaimer placement) |
| P2 | 3 | F2 (regularity widening), F5 (§3.7 d_head doesn't restate d_eff=d_head calibration), F8 (S_χ² axiomatic alias OK by adjacency) |

**Bottom line**: paper survives independent numerical reproduction
without any P0. The four P1 findings are all wording / cross-reference
fixes that take a single sentence each; none changes any numerical
claim or invalidates any proof. The paper as currently committed
(2b33a59) is consistent at the algebraic and numerical level.

### Suggested 1-line fixes (rebuttal-friendly, no submission risk)

1. (F1) `a1_proofs.tex:40` — append: "For β=0 the ODE collapses to
   ρ''=0 with ρ'(0)=ρ'(1)=0 and ∫ρ=1, forcing ρ≡1, which is also
   the τ→0 limit of (37)."
2. (F3) `a1_proofs.tex:147` — change "K = d_head/2" to "K = d_rot/2".
3. (F4) `a1_proofs.tex:548` — change "α=1/d_rot derived in
   §sec:tau-scaling" to "α=1/d_rot (= 1/d_head for MHA, = 1/d_rope
   for MLA; cf. §sec:tau-scaling Eq.~(289) and §sec:mla-results)".
4. (F6/F7) `a1_proofs.tex:189` — append at the end of the equation
   chain: "(un-normalized integral; cf. Normalization convention in
   §ref{sec:chi2-load})".

### Reproducibility scripts

- `audit/scripts/04_q_x_verify.py` — q(x) closed form vs. numerical
  Var, 7 x values from 0.1 to 10⁴.
- `audit/scripts/04_Q1_grid.py` — Q_1(L,b) and c_pred = √(45 Q_1)
  for the 8 (L,b) handover pairs.

Both run in <2s on M4 Max with scipy 1.17.
