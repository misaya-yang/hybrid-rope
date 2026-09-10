# Frequency allocation through joint spectral relations

Status: mathematical work in progress, CPU only. This follows the author's
instruction to derive a better allocation rule rather than continue scoring
hand-built tables. No candidate in this document has been sent to a model.

## The object that the allocation has to preserve

Write the full positional feature vector as

\[
x_\omega(d)=(\cos\omega_0d,\sin\omega_0d,\ldots,
\cos\omega_{K-1}d,\sin\omega_{K-1}d)^\top.
\]

A conditional rotary logit is a linear functional of this vector. A band may
contain directions that are poorly resolved on the source window: coefficient
combinations whose positional response is small there. This is a joint property
of the frequencies, not a statement that each participating channel can be
moved independently at no cost.

This is also why an ill-conditioned coefficient representation does not by
itself imply a bad represented function. The distinction is established in
the Fourier-extension literature; see Adcock, Huybrechs and Martin-Vaquero,
[On the numerical stability of Fourier extensions, Section 1.2](https://arxiv.org/pdf/1206.4111).
Here the relevant question is what happens to the represented function when
its frequencies are changed while its content coefficients are retained.

## An elementary cancellation calculation

Consider

\[
f(d)=\omega_2\sin(\omega_1d)-\omega_1\sin(\omega_2d).
\]

Its linear terms cancel, so

\[
f(d)=\frac{\omega_1\omega_2(\omega_2^2-\omega_1^2)}6d^3+O(d^5).
\]

If both frequencies are divided by the same expansion factor S, then
\(f_{new}(Sd)=f(d)\) exactly. If they are instead multiplied by a and b,

\[
f_{new}(d)=\omega_1\omega_2(a-b)d+O(d^3).
\]

An unequal allocation can therefore introduce a first-order response into
a combination whose source response began at third order. The coefficients
can be normalized without changing this conclusion. A large relative change
need not imply a large absolute logit change; that depends on the frequencies,
distance and content coefficients. The result identifies a mechanism to
preserve, not a model-accuracy theorem.

## A finite-window version

For a source distance distribution define

\[
G_0=\mathbb E_{d\sim p_0}x_\omega(d)x_\omega(d)^\top.
\]

Let N be the orthogonal projector onto a specified low-energy eigenspace of
\(G_0\), and P=I-N. For a deployed table define its remote exposure of these
directions by

\[
U(\nu)=\mathbb E_{d\sim p_{far}}\|Nx_\nu(d)\|^2
=\operatorname{tr}(NG_{far}(\nu)).
\]

This retains the joint sin/cos basis and the association with the original
slots. It differs from recomputing a whitened effective rank independently
for each new table. Its threshold is a modeling/numerical resolution choice;
it is not automatically the model's learning precision.

**Uniform-PI bound.** In the continuous case, take p0 uniform on [0,W] and
pfar uniform on [W,SW], S>1. Then

\[
G_{far}(\omega/S)
=\frac{1}{W-W/S}\int_{W/S}^{W}x_\omega(t)x_\omega(t)^\top dt
\preceq\frac{S}{S-1}G_0.
\]

Consequently \(U(\omega/S)\le S/(S-1)\operatorname{tr}(NG_0)\).
The proof is the change of variable t=d/S followed by positivity of the
omitted part of the source integral. This is an operator statement about
source-weak directions, not about task accuracy or whole-network states.

**Selective-PI identity.** Let H select retained channels and T=I-H select
channels compressed by S, with both coordinates of each pair selected together.
Then

\[
Nx_\nu(d)=Nx_\omega(d/S)
+NH\{x_\omega(d)-x_\omega(d/S)\}.
\]

Thus if NH=0, selective PI has exactly the same unresolved response as full
PI, while retaining the high block's original frequencies. If NH is small,
the additional term has an explicit norm bound. A meaningful transition
boundary is therefore related to the support of the joint weak subspace,
not solely to whether an individual frequency completed one turn.

For example, Minkowski gives

\[
\sqrt{U(\nu)}\le
\sqrt{U(\omega/S)}+
\sqrt{\mathbb E\|NH(x_\omega(d)-x_\omega(d/S))\|^2}.
\]

The second term can be evaluated directly. The older repository's
`rope_transport/nullband.py` uses a different, one-turn safety condition;
its claim that already-wrapped channels can be moved freely is not an
assumption of this derivation.

## Why a middle transition is a coupled allocation problem

Let M repeat a scalar m_j on the two coordinates of pair j. Under the local
variation \(\nu_j(h)=\omega_j\exp(-hm_j)\),

\[
\left.\partial_hNx_{\nu(h)}(d)\right|_{h=0}
=-NMP\,d x'_\omega(d)-NMN\,d x'_\omega(d).
\]

The first term mixes source-resolved response into source-weak directions;
the second retains the derivative of the already-weak response and must not
be silently dropped. Moreover,

\[
\tfrac12\|[M,P]\|_F^2
=\sum_{i<j}\|P_{ij}\|_F^2(m_i-m_j)^2,
\]

where Pij is a 2-by-2 block. The natural coupling is therefore a graph of
joint spectral relations. It is not uniform smoothing of adjacent frequency
indices. This is only a first-order diagnostic: a finite S=4 transformation
must be checked with the exact feature response.

## CPU comparison with the existing tables

The initial calculation uses all integer source lags 0..32767, uniform weight,
and a **direct design-matrix SVD** with relative singular-value threshold
1e-10. There are 84 retained and 44 discarded directions. The corresponding
source unresolved energy is 4.9246e-21. This is a discrete numerical diagnostic;
the continuous PI bound above is not claimed as an exact bound for these two
different quadrature conventions.

All integer target lags 32768..131071 are evaluated in chunks. No strided
lag compression or model forward pass is used. For local positional distortion,
the table reports
\(C_T=\mathbb E_{d\in\{0,\ldots,T-1\}}\|x_\nu(d)-x_\omega(d)\|^2\).
T labels relative-distance ranges, not new context-length model evaluations.

| Existing table / CPU construction | Remote U | C26 | C32768 |
|---|---:|---:|---:|
| Full PI | 4.4282e-21 | 23.9139 | 90.7246 |
| MrPro | .235928 | .000240986 | 42.5180 |
| BM | .00168015 | .000685777 | 42.9997 |
| FullLagP2 | 7.6285e-19 | .000833780 | 37.1821 |
| Smooth MrBudget | .0494352 | .000159185 | 35.3793 |
| LongBridgeSlower | .126909 | .000243575 | 42.3313 |
| LongBridgeFaster | .525238 | .000238496 | 42.7174 |
| HighGapToLong | .503264 | .493408 | 87.3346 |
| E1 s28 less | .235928 | .000230655 | 42.3012 |
| Native prefix 0..29, PI from 30 | 1.5900e-20 | .000808417 | 30.7749 |

Receipt: `results/nongeometric_screen_20260909/planned_controls/source_subspace_transport_audit.json`.
The last row is a CPU construction for understanding the allocation problem,
not a queued model candidate or a new-method claim. It uses total compression
34 versus P2's 34.1789; equal support and pair count do not imply equal total
compression. No same-budget causal comparison is claimed.

An earlier scratch computation applied 1e-10 to Gram eigenvalues. That is
**not** the same cutoff as 1e-10 on design singular values: the latter would
correspond to 1e-20 in the Gram spectrum. Its large regularized amplification
numbers are not used in this table. Direct SVD avoids that comparison error.

## What the calculation explains, and what prevents premature adoption

The rule behind P2 nearly preserves the source's weak joint relations after
extension. Arbitrary high-gap transfer does not. A common compression of the
slow block has a concrete finite-window justification; independently distributing
compression across a correlated middle block can break it. These observations
give the allocation question more structure than a count of slow channels.

However, U and unweighted local distortion are **not a sufficient selector**.
Smooth MrBudget improves both quantities relative to MrPro in this calculation,
yet its existing 128K development score is worse. E1 s28 has essentially the same
U as MrPro despite different task outputs. Those are direct limitations, not
minor caveats to hide after presenting a new optimum.

The remaining mathematical work is to retain this joint-relation constraint
while identifying the additional value/cost of the resolved directions. A
rule obtained merely by minimizing U would prefer full PI or nearly constant
features and would not solve the requested allocation problem. The new criterion
must distinguish useful long-scale computation from both unresolved activation
and loss of the learned computation inside the resolved subspace.

One tractable route is a coupled quadratic allocation around a coherent
reference, with a PSD matrix built from exact projected phase derivatives,
and an explicit local-computation cost. In the ideal separated case it reduces
to native high frequencies plus a commonly compressed slow block. The middle
transition then arises from cross-block coupling. The finite-change objective,
the cost weighting, and the relation to EVQ's density functional still need
to be derived and checked; they are not filled in by an arbitrary smoothing
penalty or a manually chosen transition width.

## Checking whether trained Q/K operator weighting closes the gap

An additional CPU calculation reads Q/K projection slices from the pinned
checkpoint, without loading or running a model. For half-split pair rows,
define the bilinear matrices

\[
A_j=q_{0j}^\top k_{0j}+q_{1j}^\top k_{1j},\qquad
B_j=q_{1j}^\top k_{0j}-q_{0j}^\top k_{1j}.
\]

Here the signed separation is key-position minus query-position. Append the
bias as the last projection column. Under independent input vectors with
unit second-moment matrices (including that last constant coordinate),
the exact operator MSE is obtained from the Frobenius Gram of these matrices.
It can be computed using only row inner products of Q and K, without forming
hidden-size-squared matrices for every frequency. A dense small-matrix identity
check precedes extraction. Actual GQA sharing is retained, attention gain is
held at one, and the stored aggregate averages 16 heads and 36 layers.

Implementation: `scripts/analysis/rope_qk_operator_gram.py`.
Derived arrays: `results/nongeometric_screen_20260909/planned_controls/qk_operator_gram.npz`.
This input distribution is an explicit mathematical model; it is not a
measurement of the checkpoint's activation distribution. The cost is also
logit-operator MSE, not softmax KL or loss of useful semantic information.

For causal lag ranges ending at 26/256/2048/32768, the mean weighted costs are:

| Table | 26 | 256 | 2048 | 32768 |
|---|---:|---:|---:|---:|
| MrPro | .000035 | .004192 | .204290 | 6.043652 |
| P2 | .000089 | .009705 | .612880 | 5.394220 |
| Smooth MrBudget | .000024 | .002630 | .174953 | 5.263579 |
| Native prefix 0..29, PI from 30 | .000085 | .009057 | .616595 | 4.795435 |

Smooth still improves this cost and U relative to MrPro, so merely replacing
uniform phase weights by projection-matrix norms does not close the explanatory
gap. The step example also does not dominate P2 at every weighted local scale.
This prevents promoting it as a generally better rule from the preceding
unweighted comparison. The next derivation has to distinguish alteration of
useful computation from undirected operator distance, or adopt a more precisely
scoped allocation objective whose relation to capability can actually be tested.


## Does weighting the unresolved response change the counterexample?

The preceding weighted table measured local operator distortion. A separate
check now also weights the **remote source-weak response** by the actual Q/K
operator Gram: U_H(nu) = tr(N H N G_far(nu)). Both the source design and target
features use negative signed separations for positive causal lags, preserving
the Q/K sine convention. The same independent unit-second-moment assumption
applies; this is still not a measurement of activation statistics.

| Source relative singular cutoff | Discarded directions | MrPro U_H | Smooth U_H | P2 U_H |
|---|---:|---:|---:|---:|
| 1e-06 | 47 | 1.12494981 | 0.749846431 | 1.83208014e-12 |
| 1e-08 | 45 | 0.427494108 | 0.172023636 | 6.88334975e-17 |
| 1e-10 | 44 | 0.359961033 | 0.144080138 | 3.93622351e-18 |

Smooth has lower weighted unresolved response than MrPro in **all 36 layers**
at each of these three cutoffs. Thus neither averaging over layers nor the
single previous numerical cutoff explains the counterexample. This closes
the proposed repair of merely weighting the same source-weak-energy criterion.
It does not imply that the joint-relation mechanism is absent, or that an
arbitrary learned-coefficient distribution is equivalent to these weights.

Reproducible CPU calculation: `scripts/analysis/rope_weighted_subspace_audit.py`.
Receipt: `results/nongeometric_screen_20260909/planned_controls/weighted_source_subspace_audit.json`.
