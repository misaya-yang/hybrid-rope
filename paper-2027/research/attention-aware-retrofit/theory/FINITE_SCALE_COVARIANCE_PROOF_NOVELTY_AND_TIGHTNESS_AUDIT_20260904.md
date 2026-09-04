# Finite scale covariance: proof, novelty, and tightness audit

- **Date:** 2026-09-04.
- **Status:** `PROOFS CHECKED / GENERAL REAL-ORTHOGONAL EXTENSION DERIVED /
  SEPARATION-ORDER COROLLARY DERIVED / NOVELTY NOT CERTIFIED / OPERATOR
  TIGHTNESS UNRESOLVED`.
- **Question:** Which mathematical claims in the supplied Pro Sections 5--12
  are correct, what is genuinely added by combining them, and what remains
  necessary before the result can carry a paper?
- **Protocol:** theorem-by-theorem proof reconstruction, limiting-case and
  quantifier checks, direct numerical counterchecks, and a targeted
  primary-source search current to the date above.
- **Evidence boundary:** proofs below are **Derived results** under their stated
  assumptions. Literature non-discovery is **Unresolved**, not proof of novelty.
  Existing LM outcomes are separate **Observations** and do not prove these
  theorems.
- **Artifacts:** theorem harness SHA-256
  `19222ff5f88438b812dd40d3ed58baecd9f6feccfeb43d798beef9c907980007`;
  best-`D_j` optimizer SHA-256
  `97c6f8dfff59e8e3679dd7ec3af884ba1df9cc2422f027e7023a2d6d1f96fd76`;
  final no-card optimizer preflight receipt SHA-256
  `37aebf84b11b26fa2d1c707abb0c86ae6a0e78bf156d798aaa84d2c60b497358`.
- **Correction:** the earlier preflight's statement that the general
  real-orthogonal extension was missing is superseded by Section 2 below.
  Its statement that an operator-similarity upper construction is missing
  remains current. The subsequent
  [`tightness result`](../results/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md)
  passes a nontrivial positive control but finds only identity/error-two
  solutions in 45 primary trajectories; it stops multilevel without claiming
  global impossibility.

## 1. Proof audit verdict

| Claim | Verdict | Material qualification |
| --- | --- | --- |
| Exact continuous scaling trilemma | **Derived result; classical** | state it for positive `s != 1`, or more generally `abs(s) != 1`; `s=-1` has nontrivial bounded examples |
| Theorem 3 exact orbit growth and boundary | **Derived result** | equality characterization requires `N >= 1`; `N=0` is vacuous |
| Theorem 4 coherent approximate packing | **Derived result** | applies to one repeatedly used partial injection; it is not an arbitrary-mixing operator theorem |
| Operator error to log tolerance | **Derived result** | requires `epsilon < 2` and an absolute positive-frequency anchor strong enough to keep the lower endpoint positive |
| Theorem 5 Fourier-orbit rank | **Derived result** | valid for arbitrary invertible `D_j`; no `D_j=D^j` or condition-number premise is needed for the lower bound |
| General real orthogonal extension | **Derived result** | the correct dimension is the number of distinct characters in the complexified signed spectrum, at most the real dimension |
| Montgomery--Vaughan separation corollary | **Derived result** | dimension-free `O((L delta)^-1)` slack; scaling order is sharp, constant is not claimed optimal |
| Exact discrete periodicity | **Derived result; group-theoretic structure classical** | `M <= s^d-1` and `gcd(M,s)=1`; the bound is attained |
| Theorem 6 approximate alias and anti-alias | **Derived result** | exponential-in-dimension scale is unavoidable without added separation/locality assumptions |

### Continuous exact case

Differentiating `D T_t D^{-1}=T_{st}` at zero gives

\[
DAD^{-1}=sA.
\]

The finite spectrum of `A` is invariant under multiplication by `s`. For
positive `s != 1`, every nonzero eigenvalue would generate an infinite orbit,
so every eigenvalue is zero and `A` is nilpotent. If `exp(tA)` is uniformly
bounded on an unbounded ray, every nonzero nilpotent Jordan block would produce
polynomial growth; hence `A=0`. For `s=-1`, a planar skew generator conjugated
by a reflection is a nontrivial bounded counterexample.

### Exact and coherent approximate orbit growth

Within each residue class modulo `alpha`, write the log frequencies as a finite
integer set `A`. The elementary integer sumset inequalities

\[
|A-\{0,\ldots,N\}|\ge |A|+N,
\qquad
|A+\{-N,\ldots,N\}|\ge |A|+2N
\]

give Theorem 3 after summing over residue classes. Equality for `N>=1` holds
exactly when every class is a consecutive chain.

For Theorem 4, every matching edge decreases log frequency by at least
`beta=alpha-tau>0`. A partial injection therefore decomposes into disjoint
acyclic chains of length at most

\[
C_\tau=\min\{K,\lfloor R/\beta\rfloor+1\}.
\]

A chain of length `ell` has `(ell-N)_+` one-sided survivors and
`(ell-2N)_+` two-sided survivors. Packing total mass into chains of length
`C_tau` maximizes survivors and yields the stated formulas. This also proves
the `N -> 2N` bidirectional substitution.

## 2. General real-orthogonal operator theorem

Let `T_t` be any continuous orthogonal representation of the real translation
group on `R^d`. Its generator is real skew-symmetric. After complexification
there is a unitary basis in which

\[
T_t=\operatorname{diag}(e^{i\lambda_1t},\ldots,e^{i\lambda_dt}),
\]

where the multiset is closed under sign and may include zero or repeated
frequencies. Let `Lambda` be the set of distinct base frequencies,
`r=|Lambda|<=d`, and

\[
\Omega_N=\bigcup_{j=0}^N s^j\Lambda,
\qquad M=|\Omega_N|.
\]

If for every `j` there is an arbitrary real invertible `D_j` satisfying

\[
\sup_{|t|\le L}\|D_jT_tD_j^{-1}-T_{s^jt}\|_{\rm op}\le\varepsilon,
\]

complexify `D_j` and use the same unitary eigenbasis. Every diagonal coordinate
of the conjugated operator is

\[
\sum_{\ell=1}^d
(D_j)_{k\ell}(D_j^{-1})_{\ell k}e^{i\lambda_\ell t},
\]

and therefore belongs to the fixed `r`-dimensional space

\[
V=\operatorname{span}\{e^{i\lambda t}:\lambda\in\Lambda\}.
\]

Operator norm controls each diagonal entry, so every distinct character in
`Omega_N` lies within `epsilon` of `V` in both `L-infinity` and normalized
`L2([-L,L])`. The Ky-Fan variational principle then gives

\[
\boxed{
\varepsilon^2\ge
1-\frac1M\sum_{q=1}^{r}\lambda_q(G_N)
}.
\]

Thus the Pro complex proof does extend to every continuous finite-dimensional
real orthogonal RPE. Standard RoPE is only the multiplicity-free signed-pair
special case `r=d=2K`.

There is also a strictly stronger table-specific necessary condition. With
`G_0` the base-character Gram matrix and `g_nu` its cross-Gram vector against
`f_nu`,

\[
\boxed{
\varepsilon^2\ge
\max_{\nu\in\Omega_N}
\left(1-g_\nu^*G_0^\dagger g_\nu\right).
}
\]

This is the worst fixed-native-subspace projection residual already emitted by
the numerical harness. It is stronger than the orbit-average Ky-Fan bound but
is not an LM-performance selector.

## 3. Dimension-free separation corollary

Let the distinct continuous orbit frequencies be `delta`-separated. The
weighted generalized Hilbert inequality of Montgomery and Vaughan gives a safe
constant `3 pi/2`. Applying it to the two exponential terms in the sinc Gram
quadratic form yields

\[
\|G_N-I\|_{\rm op}
\le \frac{3\pi}{2L\delta}.
\]

Consequently,

\[
\boxed{
\varepsilon^2\ge
\left[
1-\frac{r}{M}
\left(1+\frac{3\pi}{2L\delta}\right)
\right]_+.
}
\]

Unlike elementwise coherence plus Gershgorin, the slack does not grow with
`M`. The inverse-`L delta` order cannot generally be improved: for two modes,
the largest Gram eigenvalue is `1+|sinc(L delta)|`, which is of that order along
an infinite sequence of separations. No claim is made that `3 pi/2` is the best
constant. The source is Montgomery and Vaughan's
[Hilbert's Inequality](https://doi.org/10.1112/jlms/s2-8.1.73); the original
paper provides both uniform-gap and weighted forms.

## 4. Why the requested matching upper construction is still missing

The orthogonal-character example proves that `1-r/M` is sharp for the relaxed
problem of approximating `M` unit vectors by an arbitrary `r`-dimensional
subspace. It does **not** show sharpness for similarity conjugacy.

Similarity imposes coefficients

\[
c_{j,k,\ell}=(D_j)_{k\ell}(D_j^{-1})_{\ell k},
\qquad \sum_\ell c_{j,k,\ell}=1,
\]

and the theorem's operator norm is worst-coordinate rather than orbit-average.
For mutually orthogonal characters, one new character outside the Native span
already forces its `L2` distance to one, while the Ky-Fan average bound is only
`sqrt(1-r/M)`. Therefore a matching operator-norm construction cannot be
inferred from PCA sharpness and may not exist at that scale.

The GPU program must decide between two live alternatives:

1. bounded-condition non-permutation conjugacies approach the lower-bound
   scaling, supporting an operator theorem with a construction; or
2. a persistent gap remains, showing that similarity constraints require a
   stronger lower bound or an explicitly averaged error metric.

The prepared optimizer scans orthogonal and bounded-condition general real
`D_j`, reports exact identity/permutation baselines, retains every selected
matrix and hash, and labels its sampled maximum as a lower estimate of the true
continuous supremum—not a certified upper bound.

## 5. Primary-source novelty audit

| Source | What it already establishes | Relation to the present result |
| --- | --- | --- |
| [Montgomery & Vaughan, 1974](https://doi.org/10.1112/jlms/s2-8.1.73) | generalized Hilbert inequalities for separated real frequencies | owns the separation tool; that corollary alone is not novel |
| [Deep Scale-spaces, NeurIPS 2019](https://papers.neurips.cc/paper/8956-deep-scale-spaces-equivariance-over-scale.pdf) | finite scale-space truncation breaks global equivariance and creates boundary effects | directly removes novelty from the qualitative finite-boundary claim |
| [Zhu et al., JMLR 2022](https://www.jmlr.org/papers/v23/20-099.html) | quantifies scale-channel truncation error and calls the boundary leakage unavoidable | closest explicit prior art for quantitative scale-boundary leakage, but not a PE Fourier-orbit-rank theorem |
| [C*-stability of discrete groups](https://arxiv.org/abs/1808.06793) | metric-sensitive stability of almost representations, including some Baumslag--Solitar groups | adjacent mathematical literature; does not by itself give the finite-window PE bound |
| [Dutkay & Jorgensen, 2007](https://arxiv.org/abs/0704.2050) | operator-theoretic and finite-dimensional representations of the Baumslag--Solitar relation used by wavelets | removes novelty from recognizing the discrete dilation--translation group relation |
| [Positional Encodings as Group Representations](https://openreview.net/forum?id=18f4nhMJ33) | treats many PEs as group-representation features | broad representation framing is prior art |
| [Algebraic Positional Encodings](https://arxiv.org/abs/2312.16045) | maps algebraic domains to orthogonal operators | orthogonal group construction is prior art; no scale-conjugacy lower bound identified |
| [STRING](https://arxiv.org/abs/2502.02562) | characterizes separable translation-invariant orthogonal encodings and learns changes of basis/frequencies | strongly overlaps the representation class, not the finite scale-orbit obstruction |
| [GRAPE](https://arxiv.org/abs/2512.07805) | unifies orthogonal multiplicative and unipotent additive PE group actions | broad group/operator generalization is not available as novelty |
| [Jordan-RoPE](https://arxiv.org/abs/2605.04217) | constructs non-semisimple one-parameter relative representations | shows that leaving norm preservation admits polynomial features; outside the orthogonal theorem |
| [Rethinking RoPE](https://arxiv.org/abs/2504.06308) | Lie-theoretic constraints and orthogonal basis transformations for N-D RoPE | overlaps the real-orthogonal structural setup |
| [Wavelet Networks](https://arxiv.org/abs/2006.05259) and [Morlet PE](https://arxiv.org/abs/2606.01258) | scale-translation equivariance in wavelet networks; localized wavelet positional features | alternative multiscale constructions, not the claimed finite norm-preserving RPE lower bound |

The targeted searches did not identify a primary source with the exact chain

\[
\text{finite-window operator scale error}
\to \text{scaled-character Gram/Ky-Fan rank}
\to \text{dimension lower bound for relative PE}.
\]

That is a **search observation**, not a novelty certificate. A defensible claim
must say “we are not aware of” and retain the search date and scope until an
independent domain-expert or systematic bibliographic review confirms it.

## 6. Paper verdict and stop rules

The general real-orthogonal extension and dimension-free separation corollary
remove two gaps identified by Pro. The remaining load-bearing gap is operator
tightness. Therefore:

- exact obstruction, periodicity, and boundary leakage are motivation/prior-art
  synthesis, not the novelty claim;
- the narrow candidate contribution is the finite-window
  operator-error-to-resolvable-scale-orbit-rank theorem for norm-preserving
  relative PE;
- the table-specific projection residual is a stronger mathematical diagnostic
  than the Ky-Fan average, but prior experiments already forbid calling it a
  universal behavioural selector;
- if optimized bounded-condition conjugacies remain far above the theorem
  lower bound, revise the main metric or strengthen the theorem before spending
  on new LM evaluations;
- if the lower bound is vacuous on all hash-frozen RoPE tables, retain the
  theorem only as a structural result and do not center the empirical paper on
  it.

No new RoPE allocation curve, universal `z`, or Native-derived movement profile
follows from this audit.
