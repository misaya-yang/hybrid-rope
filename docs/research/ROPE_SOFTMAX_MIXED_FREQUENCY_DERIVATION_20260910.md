# Allocation must account for mixed frequencies in softmax

2026-09-10, mathematical work in progress; no model execution or new candidate
queue. This develops a different missing mechanism from the linear positional
Gram analyses. It does not claim that harmonic generation itself is new.

## An exact calculation inside a conditional attention row

First hold the content coefficients fixed over the positions being analyzed:

\[
s(d)=\sum_{j=1}^K r_j\cos(\nu_jd-\varphi_j),\qquad r_j\ge0.
\]

This is an exact rotary logit for repeated fixed Q/K content. For arbitrary
keys, the coefficients depend on d; that case requires additional analysis
rather than silently applying a stationary Fourier model to a full prompt.

The classical [modified Bessel expansion, DLMF 10.35.2](https://dlmf.nist.gov/10.35.E2),
applied to every factor of exp(s), gives the absolutely convergent series

\[
e^{s(d)}=\sum_{n\in\mathbb Z^K}c_n e^{i(n^\top\nu)d},\qquad
c_n=\prod_j I_{n_j}(r_j)e^{-in_j\varphi_j}.
\]

Its total absolute coefficient mass is exp(sum r_j). Frequencies of the
unnormalized attention weights therefore include integer combinations of
the rotary frequencies, not only the original frequencies themselves.
Normalizing one attention row divides every position weight by the same
scalar; it does not remove their relative spatial modulation.

The simplest cross term is already visible at second order:

\[
e^{a\cos\omega d+b\cos\eta d}
=1+a\cos\omega d+b\cos\eta d
+\frac{a^2}{4}(1+\cos2\omega d)
+\frac{b^2}{4}(1+\cos2\eta d)
+\frac{ab}{2}\{\cos(\omega-\eta)d+\cos(\omega+\eta)d\}
+R_3(d),
\]

where |R3(d)| <= exp(|a|+|b|)(|a|+|b|)^3/6. Thus two individually rapid
oscillations can create a slow envelope. Its amplitude and sign depend on
content, and suppressing a mixed frequency is not automatically beneficial.

Related work already analyzes softmax/Bessel spectra in an idealized RoPE
flow: [Ye, Section 5 and Appendix B](https://arxiv.org/html/2607.24502v1).
That model studies normalized spherical token dynamics and a resonant ring;
its consensus results are not frozen-language-model accuracy guarantees.
The broader observation of harmonic generation in RoPE attention also appears
in [Ruscio and Silvestri, 2024 preprint, Section 7](https://arxiv.org/html/2410.18067v2).
Here we use the exact exponential product and retain row normalization
explicitly, rather than treating an unnormalized second-order expansion as
a normalized softmax formula.

## Why high individual frequencies do not guarantee coarse averaging

For integer positions in an interval I of length T,

\[
\left|\frac1T\sum_{d\in I}e^{i\kappa d}\right|
\le\min\left(1,\frac1{T|\sin(\kappa/2)|}\right).
\]

At a multiple of 2 pi, the right-hand side is interpreted as 1. This is the
finite geometric-series identity; it retains discrete aliasing. In the
continuous counterpart the bound is min(1,2/(T|kappa|)).

Consequently, coarse averaging of exp(s) depends on the significant
combinations kappa = n^T nu. Conditions on T nu_j individually do not suffice.
For any finite set of retained harmonics, summing |c_n| times the displayed
bound controls their residual interval-average modulation. The omitted
absolute coefficient mass supplies a separate, finite tail bound. One need
not assume that an infinite small-divisor sum converges.

For completeness, in the continuous case let the retained nonconstant fast
part have a primitive F with ||F||_infinity <= B. For a slow function g of
bounded variation on [a,b], integration by parts gives

\[
\left|\int_a^b g(d)\{e^{s_{fast}(d)}-\mu\}\,dd\right|
\le B\{ |g(a)|+|g(b)|+\operatorname{TV}(g)\},
\]

with a separate tail term if the Fourier sum was truncated. Exact resonances
belong in the mean mu. Applying the inequality to g = exp(s_slow) and to
g times a coarse value/test function controls the denominator and numerator
of a normalized attention observable. This gives a conditional mathematical
meaning to treating a high-frequency block as a rapidly averaging factor.
It also identifies the failure condition: a significant mixed mode remains
slow on the interval, even though its constituent frequencies are rapid.

## A hierarchy present in the geometric grid

For omega_j = r^j with r = b^(-1/K), the q-th consecutive difference is

\[
\kappa_{j,q}=\sum_{a=0}^q(-1)^a{q\choose a}\omega_{j+a}
=\omega_j(1-r)^q.
\]

The corresponding integer combination has l1 norm 2^q and first appears
at that order of the exponential's power series. It is not a new independent
rotary pair, and its amplitude can become very small at higher orders.

For b = 10^6, K = 64, W = 32768, the first zero-based j satisfying the
illustrative half-cycle condition |kappa_jq| W <= pi is:

| Consecutive-difference order q | Earliest exponential order | First j |
|---|---:|---:|
| 0: individual frequency | 1 | 43 |
| 1: adjacent difference | 2 | 36 |
| 2: second difference | 4 | 28 |
| 3: third difference | 8 | 21 |

The pi threshold specifies half-cycle variation, not a universal learning
cutoff. Changing that phase threshold changes the boundary. The structural
result is the hierarchy and its exact dependence on b, K and W; the listed
indices must not be presented as independently predicted optimum cutoffs.

In particular, omega_28 - 2 omega_29 + omega_30 has a period of about 70K
tokens although the three original periods are only a few thousand tokens.
With zero content phases, its leading cosine amplitude is a*b^2*c/16 for
three logit amplitudes a,b,c. The exact coefficient is obtained from the
Bessel product, so inspecting only the new long period without its amplitude
would be misleading.

## What a frequency change must preserve, and what it can break

Under a common dilation nu = omega/S, every mixed frequency satisfies
n^T nu = (n^T omega)/S. Thus the complete conditional softmax profile, not
only each linear rotary feature, is exactly retimed. For a subset of channels
scaled together, this remains true for all harmonics supported inside that
subset. Cross-subset harmonics need not follow either time coordinate.

If n defines a relevant source interaction, preserving its retiming is the
linear frequency constraint

\[
n^\top\nu=\frac{n^\top\omega}{S}.
\]

Local interactions that should keep their original physical scale instead
require n^T nu = n^T omega. These constraints can conflict on shared channels.
That conflict gives a concrete allocation problem; a scalar count of slow
pairs cannot express it. A harmonic with l1 norm one recovers the familiar
per-frequency requirement, whereas mixed n couples the middle allocation.

A finite set of specified interaction targets admits a coupled quadratic
local approximation with matrix sum_n w_n n n^T and box/order constraints on
nu. This would derive an allocation from interaction requirements, without
choosing a ramp first. However, selecting the useful interactions, their
weights and their desired scales is still necessary. Merely minimizing
all small mixed frequencies would erase potentially useful long-range
computation. Those choices are not supplied by a positional Gram matrix.

## Boundary of the present result

The calculation identifies an actual nonlinear mechanism absent from the
previous linear-feature risk criteria. It does not establish that any
particular harmonic caused the existing E1, P2, or Smooth outcomes. General
content varies with position, and earlier layers change that content when
the frequency table changes. Those are substantive modeling conditions.

The discriminating next measurement would retain native conditional content
coefficients and compare an interaction-preserving frequency intervention
with a comparably sized interaction-breaking one. A downstream score alone
would not identify the harmonic mechanism. No such new model intervention
has been queued in this mathematical session.

## CPU checks of the conditional mechanism

The executable calculation is `scripts/analysis/rope_softmax_harmonic_audit.py`,
with output `results/nongeometric_screen_20260909/planned_controls/softmax_harmonic_audit.json`.

In an 8192-position artificial row with amplitudes (1,1) and fixed phases
(.17,-.31), frequencies (.51,.5103) each complete more than 664 cycles, but
their difference advances only 2.4576 radians. Exact normalized attention
assigns 60.9581% of its mass to the first half. Replacing the second frequency
by .53 makes the difference advance 163.84 radians and gives 50.0306% to
the first half. This demonstrates the specific failure of inferring coarse
averaging from individual cycle counts. It is not a Qwen result or an
accuracy comparison.

A product Bessel expansion truncated at |n_j| <= 9 agrees with exp(s) within
3.12e-9 in both rows. Both errors are below the independently summed absolute
tail bound 3.135e-9. The actual stored Qwen native frequencies give the
28-minus-2-times-29-plus-30 period as 70285.94 tokens.

For three equal assumed logit amplitudes a, the ratio of that mixed cosine
coefficient to the torus constant coefficient is
2 I1(a)^2 I2(a) / I0(a)^3. It is .000487777 at a=.3, .0427302 at a=1,
and .294301 at a=2. These are artificial coefficient settings, not measured
activation amplitudes. The strong amplitude dependence is precisely why
the existence of a long mixed period alone is insufficient to select a rule.
