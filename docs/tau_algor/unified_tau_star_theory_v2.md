# Unified tau* Theory for EVQ-Cosh: v2

**Date**: 2026-03-22  
**Authoring basis**: Claude v1 draft + Codex full-repo audit + DiT/MHA/MLA evidence cross-check  
**Goal**: keep the derivation theory-first, then use existing experiments only to validate or reject specific closures

---

## 0. Revision note

This v2 document intentionally revises the framing in `unified_tau_star_theory_v1.md`.

The main correction is methodological:

1. We should start from the theorem-level object that is already in the repo.
2. We should then derive architecture-specific approximations for the coefficients that determine `tau*`.
3. Only after that should we check those approximations against experiments.

So the right question is **not** "which formula best fits the experiments?"  
The right question is:

> Given the exact EVQ-Cosh variational result `tau* = sqrt(beta/alpha)`, what are the most defensible approximations for `alpha` and `beta` in MHA, MLA, and DiT?

This distinction matters, because several stronger statements in v1 are currently too aggressive:

- "`for any architecture, any base, any L` there exists a tau* that gives slight short-range loss and large long-range gain" is false as stated.
- DiT at `base=100` is a counterexample: GEO beats EVQ when no temporal channels are dead.
- MLA does support positive tau, but current repo evidence does **not** prove that MLA obeys the same uncorrected law as MHA.
- The DiT `0.53` factor is a useful heuristic, but the repo itself already labels it as a post-hoc rationalization, not a derivation.

So v2 aims for a stronger mathematical backbone and a stricter evidence hierarchy.

---

## 1. The theorem-level starting point

The most solid theoretical object already present in the repository is the surrogate-kernel result in:

- `paper/sections/03_theory.tex`
- `paper/appendix/a1_proofs.tex`
- `scripts/lib/rope/learnable_evq.py`

The logic is:

1. RoPE frequency allocation induces a kernel over frequency coordinates:

   \[
   K(\phi,\phi') = \int D(\Delta)\cos(\omega(\phi)\Delta)\cos(\omega(\phi')\Delta)\,d\Delta.
   \]

2. Under the broadband surrogate

   \[
   K_{\text{app}}(\phi,\phi') \approx \alpha\,\delta(\phi-\phi') + \beta\,\min(\phi,\phi'),
   \]

   the optimal continuous warp is EVQ-Cosh.

3. The single scalar that controls the optimal warp is

   \[
   \boxed{\tau^* = \sqrt{\beta/\alpha}}.
   \]

This is the key point.  
Any architecture-specific formula must be understood as an approximation to `beta/alpha`, not as an unrelated empirical guess.

That means the real problem is:

- `alpha`: how expensive is local/high-frequency distortion?
- `beta`: how valuable is additional low-frequency / long-range transport capacity?

Once we know how `alpha` and `beta` scale, `tau*` follows.

---

## 2. MHA: why `tau* ~ d_head / sqrt(L)` is still the leading law

### 2.1 First-order closure

For standard causal MHA, all Q/K dimensions carry RoPE, and the score is fully position-modulated:

\[
s_{mn} = \frac{q(m)\cdot k(n)}{\sqrt{d_{head}}}.
\]

The simplest closure consistent with both the theorem and the repo's experimental evidence is:

\[
\alpha_{\text{MHA}} \propto \frac{1}{d_{head}},
\qquad
\beta_{\text{MHA}} \propto \frac{d_{head}}{L}.
\]

Then

\[
\tau^*_{\text{MHA}} \propto \sqrt{\frac{d_{head}/L}{1/d_{head}}}
= \frac{d_{head}}{\sqrt{L}}.
\]

This is the cleanest theoretical route to the familiar law:

\[
\boxed{\tau^*_{\text{MHA}} \approx \frac{d_{head}}{\sqrt{L_{\text{train}}}}}.
\]

Interpretation:

- `alpha` shrinks as `d_head` grows because the local-resolution burden is spread across more frequency channels.
- `beta` grows with `d_head` and shrinks with `L` because long-range coverage becomes harder as the target distance scale increases, but more channels provide more transport capacity.

This closure is not a formal theorem, but it is mathematically aligned with the theorem-level `sqrt(beta/alpha)` structure and is far better motivated than treating `d_head/sqrt(L)` as a naked fit.

### 2.2 What the experiments do and do not validate

The MHA evidence in the repo strongly supports this as a **near-optimal centerline**:

- `docs/exp/2026-03-09_phase16_formula_optimality_sweep_results.md`
- `docs/exp/2026-03-04_phase11_L256_results.md`
- `docs/exp/2026-03-11_phase17c_2048_continue_results.md`
- `paper/tables/table1_multiscale_raw_ppl.tex`

What is actually supported:

- exact best in only part of the grid;
- top-2 or top-3 in most tested settings;
- empirical optima remain close to the predicted value;
- short-range cost is usually small;
- long-range gain is often very large.

What is **not** supported:

- exact optimality in every setting;
- a theorem that is independent of `base`, task, or metric.

So for MHA the right phrasing is:

> `d_head / sqrt(L_train)` is the strongest currently validated first-order closure of `sqrt(beta/alpha)` in causal text attention.

---

## 3. Base correction: a theoretically motivated small factor

Claude v1 included an important intuition that is worth keeping, but in a tighter form.

The natural base-dependent quantity is the channel density per octave:

\[
n_{\text{oct}} = \frac{K\ln 2}{\ln b},
\qquad K = d_{\text{rope}}/2.
\]

This measures how many RoPE frequencies already cover one octave in period-space under geometric allocation.

- Larger `base` means smaller `n_oct`: the geometric spectrum is sparser across octaves.
- Smaller `base` means larger `n_oct`: GEO already covers long periods more densely.

If EVQ's job is to compensate for insufficient long-range frequency density, then the amount of extra warp needed should decrease as `n_oct` increases.

The mildest theoretically reasonable correction is therefore

\[
\kappa_{\text{base}}(b; b_{ref})
\approx \sqrt{\frac{\ln b}{\ln b_{ref}}},
\]

where `b_ref = 5e5` is the text-MHA reference regime where the current main law was validated.

This gives:

- `kappa_base = 1` at `base=500K`
- `kappa_base \approx 0.835` at `base=10K`

That is exactly the direction we want:

- lower base -> GEO already denser -> smaller optimal tau
- higher base -> GEO sparser -> larger optimal tau

This factor is modest in the range `[10K, 500K]`, so it does not overturn the MHA law; it only refines it.

So the refined text-like base-aware closure is:

\[
\boxed{
\tau^*_{\text{text-like}}
\approx
\kappa_{\text{base}}(b; 5\times 10^5)
\cdot
\tau^*_{\text{structural}}
}.
\]

---

## 4. MLA: derive the correction from score dilution, not from ad hoc fitting

For a focused derivation of why the MLA correction is more plausibly linear
(`d_qk / d_rope`) than square-root (`sqrt(d_qk / d_rope)`), see
`docs/tau_algor/mla_linear_vs_sqrt_correction_v1.md`.

### 4.1 Structural difference from MHA

In MLA, only the RoPE subspace is position-dependent:

\[
s_{mn}
=
\frac{
q_{\text{nope}}\cdot k_{\text{nope}}

+ q_{\text{rope}}(m)\cdot k_{\text{rope}}(n)
}{\sqrt{d_{qk}}},
\qquad
d_{qk}=d_{\text{nope}}+d_{\text{rope}}.
\]

The position-dependent term is therefore diluted inside the full score by the positional fraction

\[
\rho = \frac{d_{\text{rope}}}{d_{qk}}.
\]

This is the core theoretical reason why MLA should not inherit the raw MHA law unchanged.

### 4.2 Competing closures

There are two natural theoretical closures for how this dilution enters `tau*`.

#### Closure A: variance-level dilution

If the positional signal enters through logit standard deviation, then one expects

\[
\kappa_{\text{dilute}} \sim \rho^{-1/2}
= \sqrt{\frac{d_{qk}}{d_{\text{rope}}}}.
\]

#### Closure B: logit-level dilution

If what matters is preserving the **mean positional modulation amplitude** inside the normalized score, then one expects

\[
\kappa_{\text{dilute}} \sim \rho^{-1}
= \frac{d_{qk}}{d_{\text{rope}}}.
\]

The repo's currently available MLA evidence favors Closure B more than Closure A:

- old MLA (`d_qk/d_rope = 2`) wants a correction close to `2x`, not `sqrt(2)`;
- Claude v1's newer MLA point (`d_qk/d_rope = 3`) is also closer to linear than square-root;
- the current `350m_mla32_500m` result is empirically strong on effectiveness, though not on exact optimality.

So the best current **candidate** is:

\[
\boxed{
\kappa_{\text{dilute, MLA}}
\approx
\frac{d_{qk}}{d_{\text{rope}}}
}.
\]

I do **not** claim this is proven.  
I do claim it is the most defensible current closure if we insist on a concrete formula.

### 4.3 Resulting MLA formula

Combining the MHA law, base correction, and MLA dilution correction gives the
first-order MLA centerline:

\[
\boxed{
\tau^*_{\text{MLA}}
\approx
\kappa_{\text{base}}(b; 5\times10^5)
\cdot
\frac{d_{qk}}{d_{\text{rope}}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}
}
\]

where `64` is the MHA reference head dimension underlying the validated text law.

This formula has three good properties:

1. It reduces exactly to MHA when `d_qk=d_rope=64`.
2. It explains why compressed-RoPE MLA should prefer larger tau than MHA.
3. It naturally predicts that smaller base should reduce the optimal tau somewhat.

However, old MLA also lives in a sparse-window regime where the target
extrapolation band is narrower than one channel spacing. In that regime there is
an additional discrete crossing floor `tau_cross(K,b,r)`: very small tau values
remain effectively GEO-like because no useful channel has yet entered the target
band. So the better practical formula is

\[
\boxed{
\tau^*_{\text{MLA}}
\approx
\max\!\left(
\tau_{\text{cross}}(K,b,r),
\kappa_{\text{base}}(b; 5\times10^5)
\cdot
\frac{d_{qk}}{d_{\text{rope}}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}
\right)
}
\]

The structural term sets the center of the good tau band; the crossing floor
sets the minimum tau that is no longer effectively identical to GEO.

### 4.4 Sanity checks

#### Old MLA (repo-validated main result)

- `d_qk = 64`
- `d_rope = 32`
- `base = 500K`
- `L_train = 8192`

Prediction:

\[
\tau^* \approx 1 \cdot 2 \cdot \frac{64}{\sqrt{8192}}
= 1.414.
\]

This exactly matches the tau used in the current 3-seed MLA experiment.

The repo's own rebuttal notes explain why the naive uncorrected value
`32 / sqrt(8192) = 0.354` is too small: it falls in the low-tau regime where
EVQ barely differs from geometric. That is precisely the behavior expected when
`tau` is below the sparse-window crossing floor.

Important caveat:

- this **does not prove** the formula, because the repo also admits that the MLA `tau=1.414` choice was not obtained from a full sweep at `L=8192`;
- it does show that the formula is internally consistent with the strongest current MLA asset.

#### Claude v1 new MLA point

Claude v1 proposes an additional MLA point:

- `d_qk = 192`
- `d_rope = 64`
- `base = 10K`
- `L_train = 4096`

Then

\[
\tau^* \approx
\sqrt{\frac{\ln 10^4}{\ln 5\times10^5}}
\cdot
3
\cdot
\frac{64}{\sqrt{4096}}
\approx 0.835 \cdot 3 \cdot 1.0
\approx 2.50.
\]

That matches the value in Claude v1 essentially exactly.

However, I have **not** independently verified the raw experiment asset behind that point in the current repo.  
So this is encouraging but should still be treated as provisional support, not final proof.

### 4.5 What this does not solve

This formula does **not** remove the MLA discrete-window problem.

When

\[
n_{\text{oct}} = \frac{K\ln 2}{\ln b}
\]

is too small, the extrapolation window contains only 0-1 usable frequencies, and the tau landscape becomes rugged because channels enter and exit the target window one at a time.

This is exactly what the MLA-specific analysis scripts are showing:

- `scripts/core_text_phases/mla_tau_optimization_v2.py`

So for MLA there are really two statements:

1. a structural first-order formula for the center of the good tau band;
2. a discrete frequency-density condition that determines whether the landscape is smooth enough for that centerline to be observable without fine sweeps.

---

## 5. DiT: same EVQ family, different control regime

DiT does not fit cleanly into the same closure as MHA/MLA.

Why:

1. bidirectional attention changes the geometry of relative positions;
2. only the temporal RoPE axis is relevant for temporal extrapolation;
3. the repo shows a strong dead-channel threshold effect that can dominate the tau landscape.

### 5.1 Theoretical branch

The clean DiT branch is:

\[
\tau^*_{\text{DiT}}
\approx
\gamma_{\text{bi}}
\cdot
\frac{K_t}{\sqrt{T_{\text{train}}}},
\]

with current evidence suggesting

\[
\gamma_{\text{bi}} \approx 0.53
\]

**only inside the current dead-channel regime**.

This is the right way to read the DiT result:

- it is not a universal bidirectional constant;
- it is a regime-specific correction for the current temporal-base setting.

### 5.2 Why the branch needs a regime condition

The repo already contains a direct counterexample to a universal positive-tau DiT law:

- at `base=100`, where all temporal channels are alive, GEO beats EVQ.

So the DiT branch should really be written as

\[
\boxed{
\tau^*_{\text{DiT}}
\approx
\mathbf{1}_{\text{dead-channel regime}}
\cdot
\gamma_{\text{bi}}
\cdot
\frac{K_t}{\sqrt{T_{\text{train}}}}
}
\]

or more smoothly,

\[
\tau^*_{\text{DiT}}
\approx
\psi\!\left(n_{\text{alive}}, D_{\min}\right)
\cdot
\gamma_{\text{bi}}
\cdot
\frac{K_t}{\sqrt{T_{\text{train}}}},
\]

where `psi` is near zero when all channels are already alive, and near one in the oversized-base dead-channel regime.

This is the central theoretical lesson from DiT:

> In bidirectional temporal extrapolation, tau is not just balancing local vs long-range transport. It is also compensating for base-induced dead channels on the extrapolated axis.

That is a different mechanism from text MHA, even though the optimal family is still EVQ-Cosh.

---

## 6. Final recommended formulas

### 6.1 Best current text-family formula

For text-style attention, the best current theory-driven approximation is:

\[
\boxed{
\tau^*_{\text{text}}
\approx
\kappa_{\text{base}}(b; 5\times10^5)
\cdot
\kappa_{\text{arch}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}
}
\]

with

\[
\kappa_{\text{base}}(b;5\times10^5)
\approx
\sqrt{\frac{\ln b}{\ln (5\times10^5)}}
\]

and

\[
\kappa_{\text{arch}}=
\begin{cases}
1, & \text{MHA} \\
\dfrac{d_{qk}}{d_{\text{rope}}}, & \text{MLA (leading candidate)}
\end{cases}
\]

This gives:

- **MHA**

  \[
  \boxed{
  \tau^*_{\text{MHA}}
  \approx
  \sqrt{\frac{\ln b}{\ln (5\times10^5)}}
  \cdot
  \frac{d_{head}}{\sqrt{L_{\text{train}}}}
  }
  \]

  which reduces to `d_head/sqrt(L)` at `base=500K`.

- **MLA**

  \[
  \boxed{
  \tau^*_{\text{MLA}}
  \approx
  \sqrt{\frac{\ln b}{\ln (5\times10^5)}}
  \cdot
  \frac{d_{qk}}{d_{\text{rope}}}
  \cdot
  \frac{64}{\sqrt{L_{\text{train}}}}
  }
  \]

This is, in my view, the best current approximate formula that is both theoretically motivated and reasonably aligned with the evidence we have.

### 6.2 Best current DiT formula

For temporal DiT extrapolation, the best current formula is branch-specific:

\[
\boxed{
\tau^*_{\text{DiT}}
\approx
\psi\!\left(\text{dead-channel severity}\right)
\cdot
0.53
\cdot
\frac{K_t}{\sqrt{T_{\text{train}}}}
}
\]

where `psi` should be understood as:

- near `0` when the temporal base is small enough that channels are already alive;
- near `1` in the current `base=10K`, `K_t=16`, `T=32` regime.

If we need a practical default for the current video setup in the repo, this reduces to:

\[
\tau^*_{\text{DiT, current}} \approx 1.5.
\]

---

## 7. Validation summary

### Strongly supported

- MHA: `tau ~ d_head/sqrt(L)` as a stable near-optimal centerline.
- MLA: positive tau is structurally more important when RoPE is compressed into a small subspace.
- DiT: positive tau is strongly beneficial in the dead-channel temporal regime.

### Supported but still conditional

- base correction as a small multiplicative factor in text-like regimes;
- linear MLA dilution correction `d_qk/d_rope` as the best current leading candidate.

### Not yet theorem-level

- universal cross-architecture single-line law with no regime conditions;
- universal DiT `0.53` constant;
- unconditional MLA formula valid for all `d_rope`, `base`, and `L`.

---

## 8. What should go into the paper

### Safe version

Use the following hierarchy:

1. **Theorem-level statement**

   Under the surrogate broadband kernel, the optimal warp is EVQ-Cosh and its strength parameter satisfies `tau* = sqrt(beta/alpha)`.

2. **Causal text closure**

   In causal text attention, a first-order closure gives `tau* ~ d_head/sqrt(L_train)`, which is strongly supported as a near-optimal default across the repo's main MHA experiments.

3. **Architecture corrections**

   MLA requires an additional structural correction because RoPE occupies only a fraction of the Q/K score; DiT requires a separate bidirectional/dead-channel branch.

### Stronger version, if we want one explicit formula

For the text family only:

\[
\boxed{
\tau^*_{\text{text}}
\approx
\sqrt{\frac{\ln b}{\ln (5\times10^5)}}
\cdot
\kappa_{\text{arch}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}},
\qquad
\kappa_{\text{arch}}=
\begin{cases}
1 & \text{MHA}\\[4pt]
\dfrac{d_{qk}}{d_{\text{rope}}} & \text{MLA}
\end{cases}
}
\]

with DiT explicitly presented as a separate branch rather than forced into the same scalar law.

That is the highest-confidence "more correct formula" I think we can responsibly write today.

---

## 9. Bottom line

The central correction to the earlier narrative is this:

> We do not need to discover tau by blind sweeping first and only then invent a formula.  
> The repo already gives the right theoretical primitive: `tau* = sqrt(beta/alpha)`.  
> The real task is to derive how `alpha` and `beta` transform under different attention geometries.

From that perspective, the current best synthesis is:

- **MHA**: `tau* ~ d_head/sqrt(L)`
- **MLA**: `tau* ~ kappa_base * (d_qk/d_rope) * 64/sqrt(L)`
- **DiT**: `tau* ~ psi(dead-channel severity) * 0.53 * K_t/sqrt(T)`

This is not a fully closed universal theorem yet.  
But it is no longer "just an empirical fit" either.  
It is a theory-led family of formulas, with each correction term tied to a concrete structural mechanism:

- surrogate kernel balance,
- positional dilution,
- frequency density,
- bidirectionality,
- dead-channel activation.

That is, in my judgment, the right level of mathematical ambition for the next paper draft.
