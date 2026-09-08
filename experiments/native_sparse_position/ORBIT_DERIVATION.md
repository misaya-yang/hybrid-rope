# Independent construction under the two-hour constraint

Status: a derived candidate, not a model improvement or an acceptance guarantee.
No GPU allocation is justified by the projection identities alone. A repairable
complete-answer failure and an informative same-budget intervention remain needed.

The user gave a two-hour deadline starting 2026-09-08 22:13:46 UTC. We reuse all
existing model/data assets. One independent method subagent is reasoning in
parallel. The original broad scientific objective has not been narrowed.

## What was missing from the earlier reasoning

PSR's unweighted offset metric optimizes a phase covering radius. It does not
optimize the measured post-RoPE key distribution, the relevant score margins, or
answers. Our aggregate two-prefix diagnostic also cannot reject PSR under all of
its proposed conditions. The plan specifically calls for conditional missed-block
analysis and matched causal corrections; we did not complete those by measuring
an average recall and then changing models.

Simply replacing the phase groups with another heuristic is not a remedy. A
construction should accommodate both constant raw content (a genuine rotary orbit)
and constant post-RoPE keys (the plan's counterexample to phase-span reasoning).

## A centered orbit representation of the actual key measure

Represent a native split-half rotary pair as a complex coordinate y_j, with native
frequency omega and within-block offsets t_j=j-(B-1)/2. Define

    z_j = exp(i omega t_j) - mean_l exp(i omega t_l),
    sigma^2 = mean_j |z_j|^2,
    phi_j = z_j / sigma,
    mu = mean_j y_j,
    beta = mean_j conjugate(phi_j) y_j.

For omega=0 the basis and coefficient are zero. For slow nonzero frequencies,
compute centered expm1(i omega t), avoiding catastrophic subtraction near 1.
Approximate selector keys by yhat_j=mu+phi_j beta. Nonrotary dimensions retain
their true mean. The exact reader K/V are never changed.

The basis has mean zero and mean squared modulus one. Therefore beta is the
least-squares coefficient in this centered complex orbit family, and

    mean ||y-mu||^2 - mean ||y-yhat||^2 = sum_rotary_pairs |beta|^2.

This is an exact projection identity in real arithmetic. It is not a statement
about projected query error, attention ranking, value output, or answer accuracy.

Two useful exact classes hold for every query:

1. Constant post-RoPE keys: beta=0 and the representation is exact.
2. Constant pre-RoPE content y_j=exp(i omega t_j)u: the representation is exact,
   including phase cancellation that destroys the single pooled mean.

Under a common position shift with fixed block members, mu and beta rotate with
the native pair, preserving the reconstructed selector response. The CPU tests
check these statements, partial rotation, zero/slow frequencies, and the
Pythagorean identity. They do not test model capability.

## Scoring and actual resource question

Use Fhat_b(q)=log sum_j exp(q dot yhat_bj) as a virtual-key block score. Cache one
mean vector of width D and one coefficient vector of width d_rot per KV head and
physical block. Counts/phase basis are global for fixed native frequencies. This
is 2D numbers under full RoPE and D+d_rot under partial RoPE.

The B virtual points are not B raw KV reads. Their scores can be formed by
contracting q and beta into per-frequency coefficients and multiplying by a
shared phase-basis matrix. Nevertheless this costs roughly B times a one-mean
score contraction, plus logsumexp; real latency must be measured before any
quality/cost claim. An explicit virtual-key CPU implementation is only a reference.

For residuals e_j=y_j-yhat_j,

    |F_b(q)-Fhat_b(q)| <= max_j |q dot e_j| <= ||q|| max_j ||e_j||.

The first inequality is logsumexp's Lipschitz property. A smaller Frobenius residual
does not imply a uniformly tighter directional bound or a better top-k margin.
Unlike a partition's means, this approximation is not generally a mass lower bound.

## Necessary counterarguments and comparisons

At very low frequency the normalized orbit becomes a linear positional trend.
Thus a same-byte linear-trend representation is necessary: otherwise any gain
could just be smooth content regression rather than use of native frequencies.
Mean plus diagonal covariance, low-rank covariance, matched contiguous summaries,
Quest, and PSR are relevant same-interface comparisons with actual byte accounting.

TriAttention (arXiv:2604.04921) already uses pre-RoPE concentration and a trigonometric
distance function to score token importance. The construction above instead fits
each block's actual post-RoPE measure, retains the actual current query and returns
a nonlinear block response. That distinction is not yet an exhaustive novelty
check. Prism already studies pooling attenuation; COBS already studies mass
estimation from moments. Neither basic harmonic algebra nor projection optimality
is by itself a new paper contribution.

Next decision: does an already observed, complete-answer sparse failure recover
when the missing input evidence is supplied at the identical physical KV budget?
If so, inspect the relevant missed-block margins and test whether this construction
or the independent agent's proposal repairs that measured failure. Do not launch
another model sweep on the strength of the algebra.

## Independent falsification retained

The independent method agent constructed an explicit 64-point unit-norm key block
whose centered orbit projection reduces key MSE by about 67% but creates a virtual
real-axis maximum 1.273, larger than every actual point's maximum 1. A query of
20 then ranks it above a constant 1.1 block despite the exact ranking being the
reverse. Details and numbers are in `METHOD_INDEPENDENT_20260908.md`. Thus this
candidate has not been granted GPU testing on its projection properties. The
positive-key mixture code is also prepared, not automatically scheduled; the
independent agent's rotary-pair extremal construction is the current candidate
being connected to an actual same-budget repairable failure.
