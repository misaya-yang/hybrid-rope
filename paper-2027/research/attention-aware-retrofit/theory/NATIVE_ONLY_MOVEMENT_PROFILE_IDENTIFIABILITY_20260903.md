# Native-only movement profiles: identifiability boundary and a canonical surrogate

- **Date:** 2026-09-03
- **Status:** `COMPLETE THEORY / DERIVED IDENTIFIABILITY BOUNDARY /
  CONDITIONAL UNIQUE SURROGATE / EXECUTED FOLLOW-UP DOES NOT DOMINATE CURRENT-P2`
- **Question:** Can an ordered, global, log-frequency, request-static movement
  profile `m` be uniquely derived from a Native checkpoint without data,
  activations, LM losses, or long-context outcomes?
- **Inputs:** Native context length `L`, ordered Native frequencies `Omega`,
  and rotary-pair count `K`. Frozen weight tensors are audited separately in
  Section 6 but do not enter the constructed profile. The target factor
  `S > 1` is declared by deployment.
- **Compute:** no training, GPU inference, paid compute, or model execution.
- **Correction/supersession:** this note supersedes no behavioural owner. It
  closes the underspecified identification question and supplies one explicit
  conditional construction. It does not rehabilitate the retracted T5 claims
  in `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`.

## 1. Decision

The literal question has no unique behavioural answer. The Native checkpoint is
the base point of a post-hoc design problem; it does not contain the deployment
utility that ranks directions away from that point.

There is, however, one parameter-free construction after a squared Euclidean
surrogate is postulated explicitly. Let `u_k` be the fraction of rotary pair
`k`'s Native-window phase feature that cannot be reconstructed from the other
pairs. Charge equally for moving this unique fraction and for failing to
convert the complementary redundant fraction into dilation. Under this
declared quadratic geometry surrogate, the unique ordered solution is

\[
\boxed{m^\dagger=\operatorname{Iso}(1-u)},
\]

with the fast and slow endpoints pinned by this owner's deployment contract.
The long table is

\[
\boxed{\omega_k'(S)=\omega_k S^{-m_k^\dagger}}.
\]

This replaces the historical, long-outcome-selected exponent `p=2` with a
declared quadratic convention. It is a **derived minimizer of that surrogate**,
not an observation that the resulting table preserves LM behaviour or solves a
downstream task. Its later behavioural test is owned separately and does not
dominate current-p2 across measured endpoints.

## 2. Why the original problem is not identifiable

Let

\[
\mathcal C_K=
\{m\in[0,1]^K: m_0=0,\ m_{K-1}=1,\ m_0\le\cdots\le m_{K-1}\}.
\]

For any `m` in this set, install

\[
\omega_k(S)=\omega_k S^{-m_k}.
\]

Every such profile is global, request-static, scale-independent, and
log-frequency. It also preserves strict frequency order because

\[
\frac{\omega_k'}{\omega_{k+1}'}
=\frac{\omega_k}{\omega_{k+1}}
S^{m_{k+1}-m_k}>1.
\]

For every `K >= 3`, the family

\[
m_k^{(p)}=\left(\frac{k}{K-1}\right)^p,\qquad p>0,
\]

contains uncountably many distinct members of `C_K`. All return the same Native
table at `S=1`. Under continuous per-slot multiplicative action, scale
composition only proves the power-law form `S^{-m_k}`; it does not choose its
generator `m`.

This is a narrow **derived non-identification result**: the stated inputs and
structural constraints do not select a unique point. It is not the former,
retracted claim that no checkpoint-only predictive functional can ever exist.

Two boundary cases make the missing objective unavoidable:

1. If one static table must preserve every Native-window rotary bilinear form
   for every query and key on a real-lag interval, the exact transplant
   obstruction forces the positive ordered table back to Native up to its
   stated sign/permutation aliases, hence `m = 0` here. For integer-only lags,
   the same conclusion additionally requires excluding `2 pi` aliases.
   Requiring a nontrivial slow endpoint then makes the feasible set empty.
2. If approximate preservation is allowed, an error metric and tolerated
   error are part of the estimand. The checkpoint does not supply them.

Likewise, two candidate functions that differ on any input can be ranked in
opposite orders by two legitimate deployment losses. Without a task utility or
a declared task-free surrogate, "behaviourally optimal" is undefined.

## 3. Native conditional uniqueness

Use the complete causal pair-count measure on Native integer lags:

\[
\mu_L(\Delta)=\frac{2(L-\Delta)}{L(L+1)},
\qquad \Delta=0,\ldots,L-1.
\]

For rotary pair `k`, define the two-column phase feature

\[
\Phi_k(\Delta)=
\begin{bmatrix}
\cos(\omega_k\Delta) & \sin(\omega_k\Delta)
\end{bmatrix}.
\]

Let `P_{-k}` be the orthogonal projector in
`L2(mu_L)` onto the columns of every other pair. Define

\[
u_k=
\frac{\|(I-P_{-k})\Phi_k\|_{\mu_L,F}^2}
     {\|\Phi_k\|_{\mu_L,F}^2}
\in[0,1].
\]

Equivalently, form the weighted phase Gram matrix `G`. If `G_kk` is pair
`k`'s `2 x 2` block, then its conditional residual Gram is the generalized
Schur complement

\[
S_k=G_{kk}-G_{k,-k}G_{-k,-k}^{\dagger}G_{-k,k},
\qquad
u_k=\frac{\operatorname{tr}S_k}{\operatorname{tr}G_{kk}}.
\]

Thus `u` is a deterministic function of `(L, Omega)` and uses no corpus,
activation, task label, or long outcome. `u_k = 1` means the Native phase
feature is conditionally unique; `u_k = 0` means it lies in the span of the
other Native features under the declared lag measure.

The mathematical definition uses the exact projector. A numerical artifact
must separately freeze precision, lag discretization, and the Moore--Penrose
rank rule: the slow-frequency Gram matrix is ill-conditioned, so a downsample
or SVD cutoff is part of implementation identity, not an innocuous detail.

## 4. The missing link and one quadratic closure

The old construction computed `u` from Native geometry but then selected

\[
m_k=(1-\widetilde u_k)^2
\]

after comparing long outcomes. The unowned step was the link `u -> m`.

Define the admissible ordered set `C_K` above and the surrogate

\[
J(m;u)=\sum_{k=0}^{K-1}
\left[u_km_k^2+(1-u_k)(1-m_k)^2\right].
\]

The terms have a direct, symmetric interpretation:

- `u_k m_k^2` charges movement of Native-unique phase energy;
- `(1-u_k)(1-m_k)^2` charges failure to convert Native-redundant phase energy
  into full position interpolation.

The pointwise loss has the complement symmetry

\[
(u,m)\longleftrightarrow(1-u,1-m)
\]

and leaves `J` unchanged. The increasing endpoint-pinned feasible set is not
itself invariant under this exchange unless slot order is also reversed.
Moreover, complement symmetry alone does not select a squared loss: absolute,
cubic, and other symmetric losses are possible. The postulate here is the
**squared Euclidean surrogate**, which both sets the exchange rate to one and
chooses a projection geometry. It is a declared convention, not a hidden
checkpoint fact.

Expanding the objective gives

\[
J(m;u)=\sum_k\left(m_k-(1-u_k)\right)^2+\text{constant}.
\]

Therefore

\[
m^\dagger
=\arg\min_{m\in\mathcal C_K}J(m;u)
=\operatorname{Proj}_{\mathcal C_K}(1-u).
\]

The objective is strongly convex and `C_K` is closed and convex, so the
minimizer exists and is unique. It is the endpoint-pinned isotonic regression
of `1-u`, computable by pool-adjacent-violators. For an interior index, the
standard min--max representation is

\[
m_k^\dagger=
\max_{1\le i\le k}\;
\min_{k\le j\le K-2}
\frac{1}{j-i+1}\sum_{t=i}^j(1-u_t),
\]

with `m_0=0` and `m_{K-1}=1`. If `1-u` is already nondecreasing, every interior
target is unchanged. The full vector reduces exactly to

\[
\boxed{m_k^\dagger=1-u_k}.
\]

only if the endpoint targets also satisfy `u_0=1` and `u_{K-1}=0`; otherwise
the two deployment pins replace those endpoint values.

The same unprojected target follows if a log-affine barycentre is separately
declared. In that interpretation, treat fraction `u_k` as retaining the Native log frequency and fraction
`1-u_k` as taking the fully interpolated log frequency:

\[
\log\omega_k'
=u_k\log\omega_k
+(1-u_k)\log(\omega_k/S)
=\log\omega_k-(1-u_k)\log S.
\]

The declared affine average in log-frequency coordinates yields a geometric
mean and is scale-equivariant. Scale equivariance alone would not exclude
other homogeneous means. The isotonic projection is the least-squares
correction when independently computed fractions violate the ordering
contract.

## 5. What the construction guarantees

For `S > 1`, the installed table satisfies:

1. **Bounds:** `omega_k/S <= omega_k' <= omega_k`.
2. **Order:** `omega_0' > ... > omega_{K-1}'`.
3. **Scale generator:** the effective log exponent is exactly `m_k^dagger`.
4. **Composition:** `omega(S1 S2) = omega(S1) S2^{-m^dagger}`.
5. **Runtime contract:** one table is installed before prefill and remains
   fixed for the entire request/KV-cache lifetime; `u` and `m` are not
   recomputed after an intermediate scale step.
6. **Scope contract:** the same table is shared by all layers and heads.

It also gives a per-pair Native operator bound:

\[
\|R(\omega_k'\Delta)-R(\omega_k\Delta)\|_2
=2\left|\sin\frac{(\omega_k'-\omega_k)\Delta}{2}\right|
\le
\min\left(2,\Delta\omega_k(1-S^{-m_k^\dagger})\right).
\]

These are geometric and operator guarantees. They do not imply NLL retention,
retrieval, QA, EOS termination, or a universal context radius.

## 6. Why checkpoint weight norms do not finish the identification

The frozen weights do define an activation-free ordered bilinear pencil. For
layer `ell`, query head `h`, its MHA/GQA key head `g(h)`, and rotary pair `k`,
let `A_lhk` and `B_lgk` be the two-row Q/K projections and let
`J_2` be the planar quarter-turn. Then

\[
C_{\ell hk}=A_{\ell hk}^{\mathsf T}B_{\ell g(h)k},\qquad
D_{\ell hk}=A_{\ell hk}^{\mathsf T}J_2B_{\ell g(h)k},
\]

and the complete pair contribution is

\[
M_{\ell h}(\Delta)=\sum_k
\left[\cos(\omega_k\Delta)C_{\ell hk}
+\sin(\omega_k\Delta)D_{\ell hk}\right].
\]

This pencil, rather than separate Q/K norms, is invariant to reciprocal Q/K
scaling and preserves ordered slot identity. It can define a checkpoint-only
worst-case Native compatibility functional. But minimizing that functional
without a long-utility constraint always admits the trivial Native profile
`m=0` as a global minimizer; null pencils or aliases can create ties.

Actual average slot use additionally requires an activation covariance and
cross-position moments. RMSNorm does not provide an isotropic covariance, and
deep-layer activations change with the installed table. Replacing the missing
distribution by isotropic hidden states is an explicit prior. Therefore a
weight-aware profile still needs a declared dilation budget or task-free long
utility; raw Q/K norms do not identify it.

This is why a nonseparable head-by-frequency field remains, at most, a working
hypothesis when routing is forbidden. It is not required by, and must not be
folded into, the canonical global surrogate above.

## 7. Relation to existing evidence

- A historical approximation to Native conditional uniqueness is present in
  `scripts/analysis/export_uniqueness_budgeted_tables.py` and described in
  `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`. That script downsamples
  the lag grid and uses a finite-precision singular-value cutoff; it is not the
  exact-projector artifact defined here.
- The historical `p=2` link was retained after reading 8K outcomes; the log
  embedding was also selected through long outcomes. Their behavioural owners
  remain valid at their exact scopes, but neither is a purely Native-identified
  family.
- A historical `p=1` arithmetic arm is not a validation of the present
  `log-frequency + exact-projector + quadratic` construction. Table embedding,
  numerical projector identity, gain, and evaluated protocol differ.
- The two-parameter `G_4` law compresses an already selected profile and misses
  a registered Native gate. Its fitted boundaries are not analytic constants.
- MaxEnt, EVQ-Cosh, physical-coordinate, and normalized-index constructions are
  unique only after their own extra preference, surrogate, or frozen inherited
  parameters are declared.

## 8. Claim boundary and next decision

### Derived results

- The original structural constraints leave an uncountable profile family.
- Universal exact Native preservation makes every nontrivial one-table profile
  infeasible under the transplant-obstruction assumptions.
- Given the exact Native conditional uniqueness and the stated squared
  Euclidean surrogate, `Iso(1-u)` is its unique ordered minimizer.
- The resulting log table has the bounds, ordering, composition, and runtime
  properties in Section 5.

### Tested working hypothesis

- `Iso(1-u)` was tested as a candidate intended to trade Native compatibility
  for long-context dilation on a frozen mature checkpoint. It improved natural
  likelihood but was materially worse on fresh core-4 at 4K/8K; see the result
  owner rather than inferring behaviour from this derivation.

### Unsupported

- behavioural optimality, a universal law, cross-checkpoint success, or any
  improvement in NLL, downstream generation, retrieval, QA, or EOS;
- that the squared complement-symmetric loss is the deployment utility encoded
  by a model;
- that a finite-precision implementation reproduces the exact-projector object
  without a frozen numerical-rank contract;
- any new gain, head selector, head-by-frequency field, threshold, sweep, GPU
  run, or manuscript claim.

The theoretical identification problem is therefore bounded and one
conventional closure is solved: **without a preference there is no unique
`m`; under the declared squared Native-geometry surrogate, the unique minimizer
is `m^dagger = Iso(1-u)`.** This is not a latent checkpoint profile. Behaviour
is not a consequence of the derivation; the executed follow-up is an
endpoint-dependent tradeoff and a negative for replacing current-p2 at its
fresh 4K/8K scope.
