# MLA tau*: Why the `d_qk / d_rope` correction is more plausibly linear than square-root

**Date**: 2026-03-22  
**Status**: Focused derivation note to refine `unified_tau_star_theory_v2.md`

---

## 1. Question

For MLA, suppose we write the structural correction as

\[
\tau^*_{\text{MLA}} \approx \kappa_{\text{dilute}} \cdot \tau^*_{\text{MHA-like}}.
\]

What should `\kappa_dilute` be?

Two candidates are natural:

\[
\kappa_{\text{sqrt}} = \sqrt{\frac{d_{qk}}{d_{rope}}}
\qquad \text{vs.} \qquad
\kappa_{\text{lin}} = \frac{d_{qk}}{d_{rope}}.
\]

This note argues that:

1. `sqrt(d_qk / d_rope)` is the correct scaling if we match **random logit variance**.
2. `d_qk / d_rope` is the more relevant scaling if we match the **coherent position-dependent softmax bias** that EVQ is actually trying to reshape.

So the current best first-order MLA correction remains

\[
\boxed{
\kappa_{\text{dilute, MLA}} \approx \frac{d_{qk}}{d_{rope}}
}.
\]

---

## 2. Setup

For MLA, the attention score can be written as

\[
s_{ij}
=
\frac{
q_{\text{nope},i}\cdot k_{\text{nope},j}

+ q_{\text{rope},i}\cdot R_{\Delta_{ij}} k_{\text{rope},j}
}{\sqrt{d_{qk}}},
\qquad
d_{qk}=d_{\text{nope}}+d_{\text{rope}}.
\]

Define the RoPE fraction

\[
\rho := \frac{d_{rope}}{d_{qk}}.
\]

Relative to MHA, MLA changes two things simultaneously:

1. only a fraction `rho` of Q/K dimensions carry positional signal;
2. the full score is still normalized by the total Q/K width `sqrt(d_qk)`.

So the key question is:

> when EVQ changes the frequency layout, what quantity should be held fixed across architectures?

---

## 3. Why a square-root correction appears at first glance

If one treats the RoPE term as an unstructured random dot product, then the standard deviation of the positional logit contribution is

\[
\operatorname{std}(s_{\text{rope}})
\sim
\frac{\sqrt{d_{rope}}}{\sqrt{d_{qk}}}

= \sqrt{\rho}.
\]

This immediately suggests

\[
\kappa_{\text{sqrt}} \sim \rho^{-1/2}
= \sqrt{\frac{d_{qk}}{d_{rope}}}.
\]

This logic is internally consistent, but it answers the wrong question.

It matches the scale of **random score fluctuations**.  
EVQ is not designed to increase random score fluctuations.

EVQ is designed to change the **systematic, distance-dependent bias pattern** carried by the RoPE term.

---

## 4. Why EVQ should be matched at the softmax-bias level instead

The quantity that matters for long-context extrapolation is not the RMS size of a random positional logit.

It is the change in attention weights induced by a structured positional bias across keys.

Write one row of attention logits as

\[
\mathbf{s} = \mathbf{c} + \lambda \mathbf{p}_{\tau},
\]

where

- `c` is the content/nope-dominated baseline,
- `p_tau` is the distance-structured positional bias produced by the RoPE branch,
- `lambda` is the effective amplitude with which that positional bias enters softmax.

Then

\[
\operatorname{softmax}(\mathbf{c} + \lambda \mathbf{p}_{\tau})
=
\operatorname{softmax}(\mathbf{c})
+
\lambda\,J_{\text{softmax}}(\mathbf{c})\mathbf{p}_{\tau}
+
O(\lambda^2).
\]

So the first-order change in the attention pattern is linear in the **coherent bias amplitude** `lambda`.

This is the correct operating regime for EVQ:

- EVQ changes which distances are distinguishable,
- that changes the shape of the coherent bias `p_tau`,
- the attention update is first-order in that bias.

Therefore the right correction should preserve `lambda`, not `std(s_rope)`.

---

## 5. Why the coherent positional bias scales like `rho`, not `sqrt(rho)`

The crucial point is that the RoPE term is bilinear:

\[
q_{\text{rope},i}\cdot R_{\Delta}k_{\text{rope},j}.
\]

If the head's norm budget is split across subspaces, then the rope parts of both query and key carry only a `sqrt(rho)` fraction of the full-head norm:

\[
q_{\text{rope}} \sim \sqrt{\rho}\,\widetilde q_{\text{rope}},
\qquad
k_{\text{rope}} \sim \sqrt{\rho}\,\widetilde k_{\text{rope}}.
\]

Hence the coherent RoPE bias scales as

\[
q_{\text{rope}}^\top R_{\Delta} k_{\text{rope}}
\sim
(\sqrt{\rho})(\sqrt{\rho})
\cdot
\widetilde q_{\text{rope}}^\top R_{\Delta}\widetilde k_{\text{rope}}
=
\rho \cdot \widetilde p_{\Delta}.
\]

After the shared `1 / sqrt(d_qk)` normalization is absorbed into the reference bias, MLA behaves like

\[
\mathbf{s}_{\text{MLA}}
=
\mathbf{c}
+
\rho \,\mathbf{p}_{\tau}.
\]

That is the key step:

> the **coherent** position-dependent part loses one `sqrt(rho)` from the query side and one `sqrt(rho)` from the key side, so its first-order softmax leverage is reduced by `rho`, not by `sqrt(rho)`.

This is exactly where linear correction comes from.

---

## 6. Frequency-allocation interpretation: EVQ must compensate channel mass, not noise variance

EVQ does not directly multiply logits.

It redistributes frequency mass toward the useful extrapolation band.

Let `B` be the target log-period band corresponding to `[L_train, L_target]`, and let

\[
M_{\tau}(B)
=
\int_B \lambda_{\tau}(\phi)\,d\phi
\]

denote the amount of EVQ frequency mass placed in that band.

Then a minimal first-order model for effective extrapolation leverage is

\[
\mathcal{E}_{\text{MLA}}(\tau)
\propto
\rho \, M_{\tau}(B).
\]

Interpretation:

- `M_tau(B)` counts how much useful frequency budget EVQ allocates to the extrapolation band,
- `rho` tells us how much each such channel actually matters once mixed with the noPE branch.

To preserve the same effective extrapolation leverage as MHA, MLA therefore needs

\[
M_{\tau^*_{\text{MLA}}}(B)
\approx
\rho^{-1}
M_{\tau^*_{\text{MHA}}}(B).
\]

Now, for EVQ-Cosh in the practical operating range before severe crossing effects,
the band mass grows approximately linearly with tau:

\[
M_{\tau}(B) \approx a_B + b_B \tau.
\]

So the compensation law becomes

\[
\tau^*_{\text{MLA}}
\approx
\rho^{-1}
\tau^*_{\text{MHA-like}}
=
\frac{d_{qk}}{d_{rope}}
\tau^*_{\text{MHA-like}}.
\]

This is a much better conceptual match to what EVQ actually does.

Square-root correction would match noise amplitude.  
Linear correction matches useful band mass after softmax.

This is the **first-order structural correction**.  
It still does not explain why old MLA can have a rugged, crossing-dominated tau landscape.

---

## 6.1 Sparse-window correction: the crossing floor

Old MLA lives in a frequency-sparse regime:

- `K = d_rope / 2 = 16`
- `base = 5e5`
- the 2x band width in `phi`-space is `ln 2 / ln b`
- the corresponding density ratio is

  \[
  R_2 = \frac{K \ln 2}{\ln b} \approx 0.84 < 1.
  \]

So the target extrapolation band is narrower than one channel spacing.  
This is why old MLA behaves like a binary crossing problem rather than a smooth optimization problem.

Near `tau = 0`, EVQ-Cosh changes `phi_k` only quadratically:

\[
\phi_k(\tau)
=
u_k - c(u_k)\tau^2 + O(\tau^4),
\qquad
c(u)=\frac{u(1-u)(2-u)}{6}.
\]

Therefore a very small tau barely moves any channel at all.  
The useful MLA tau range must start above a crossing floor

\[
\tau_{\text{cross}}
\approx
\sqrt{
\frac{(u_* - \phi_c)_+}{c(u_*)}
},
\]

where

\[
\phi_c = \frac{\ln(\sqrt{r}L / 2\pi)}{\ln b}
\]

is the center of the target band `[L, rL]`, and `u_*` is the nearest geometric channel likely to enter it.

Interpretation:

- the linear `d_qk / d_rope` term tells us the first-order centerline of the good tau band;
- `tau_cross` tells us how small tau is still effectively indistinguishable from GEO because no useful crossing has happened yet.

For old MLA, `tau_cross` is only `O(0.6-0.8)` in scale.  
So it explains why the naive `32 / sqrt(8192) = 0.354` is too small, but it does **not** by itself explain `tau = 1.414`.

That remaining factor is exactly what the linear dilution correction provides.

---

## 7. Consistency check against the two MLA anchor points currently in discussion

Use the base-aware MHA-like reference

\[
\tau^*_{\text{MHA-like}}
=
\sqrt{\frac{\ln b}{\ln(5\times 10^5)}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}.
\]

### Anchor A: current repo main MLA result

- `d_qk = 64`
- `d_rope = 32`
- `base = 500K`
- `L_train = 8192`

Then

\[
\tau^*_{\text{MHA-like}} = \frac{64}{\sqrt{8192}} = 0.707.
\]

- square-root correction gives

  \[
  0.707 \cdot \sqrt{2} = 1.00
  \]

- linear correction gives

  \[
  0.707 \cdot 2 = 1.414
  \]

The current MLA experiment uses `tau = 1.414`, and this value is strongly effective in the 3-seed results.

So for the strongest verified MLA anchor in the repo, the linear correction lands exactly on the operating point, while the square-root correction under-predicts substantially.

### Anchor B: Claude v1 MLA-v2 point

Claude v1 proposes:

- `d_qk = 192`
- `d_rope = 64`
- `base = 10K`
- `L_train = 4096`

Then

\[
\tau^*_{\text{MHA-like}}
=
\sqrt{\frac{\ln 10^4}{\ln(5\times 10^5)}}
\cdot
\frac{64}{\sqrt{4096}}
\approx
0.838.
\]

- square-root correction gives

  \[
  0.838 \cdot \sqrt{3} \approx 1.45
  \]

- linear correction gives

  \[
  0.838 \cdot 3 \approx 2.51
  \]

Claude v1 reports the observed optimum as approximately `2.5`, which again strongly favors the linear branch.

Important caveat:

- I have verified the script defining that MLA-v2 setup:
  `scripts/core_text_phases/run_50m_mla_v2_tau_sweep.sh`
- I have **not** independently located the final raw report artifact for that sweep in the current workspace.

So Anchor B is encouraging, but still weaker than Anchor A.  
In the current workspace it should be treated as a provisional consistency check, not as a hard evidence anchor.

---

## 8. Why the square-root branch is still useful conceptually

The square-root correction is not nonsense. It is the right answer to a different question:

> How does the RMS magnitude of the positional logit term shrink when RoPE occupies only a fraction of Q/K dimensions?

That answer is indeed

\[
\sqrt{d_{rope}/d_{qk}}.
\]

But EVQ is not a random-feature variance booster.

It is a structured frequency allocation that changes the deterministic distance-dependent bias seen by softmax.

For that purpose, the bilinear `q_rope x k_rope` attenuation and the band-mass compensation argument both push toward the linear correction.

So the two exponents correspond to two different observables:

- `1/2`: random score scale
- `1`: coherent softmax bias / useful extrapolation leverage

The extrapolation experiments probe the second observable.

---

## 9. Resulting recommendation

The best current MLA formula is two-layered:

\[
\boxed{
\tau^*_{\text{MLA}}
\approx
\max\!\left(
\tau_{\text{cross}}(K,b,r),
\sqrt{\frac{\ln b}{\ln(5\times10^5)}}
\cdot
\frac{d_{qk}}{d_{rope}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}
\right)
}
\]

The meaning of the two terms is different:

- the second term is the **first-order structural law**;
- the first term is a **discrete sparse-window floor** that becomes active when `K` is small or `base` is large enough that channels enter the target band one at a time.

If the spectrum is dense enough, the floor is inactive and the formula reduces to the simpler linear law:

\[
\boxed{
\tau^*_{\text{MLA}}
\approx
\sqrt{\frac{\ln b}{\ln(5\times10^5)}}
\cdot
\frac{d_{qk}}{d_{rope}}
\cdot
\frac{64}{\sqrt{L_{\text{train}}}}
}
\]

with one important regime warning:

- when the frequency density per octave

  \[
  n_{\text{oct}} = \frac{(d_{rope}/2)\ln 2}{\ln b}
  \]

  is too small, channel-crossing effects make the tau landscape rugged;
- in that sparse regime the linear term should be read as the center of the good tau band, and `tau_cross` supplies the minimum useful tau scale.

---

## 10. Bottom line

If we force the problem into a one-parameter exponent family

\[
\kappa_{\text{dilute}} =
\left(\frac{d_{qk}}{d_{rope}}\right)^{\alpha},
\]

then the current theory and evidence point to:

\[
\boxed{\alpha \approx 1}
\]

for MLA's EVQ operating regime.

The reason is not "because it fits better".

The reason is:

1. EVQ acts through a coherent RoPE bias, not random score variance;
2. that coherent bias is bilinear in the rope subspaces of both Q and K;
3. after softmax linearization, the relevant first-order attention effect is linear in that bias;
4. preserving useful extrapolation leverage therefore requires compensating the `rho = d_rope/d_qk` attenuation linearly.

That is the cleanest current explanation for why MLA looks linear rather than square-root.
