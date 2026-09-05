# Static Native no-harm boundary and a prefix-preserving coordinate handoff

- **Date:** 2026-09-03
- **Status:** `COMPLETE DERIVED CLASS BOUNDARY / CPU ALGEBRA PASS /
  PREFIX-HANDOFF WORKING HYPOTHESIS / NO MODEL EXECUTION`
- **Evidence labels:** Sections 2--3 are **Derived results** under their stated
  quantifiers; Section 4 is a **Construction** with CPU-checked rotary algebra;
  any claim of transformer-quality improvement is a **Working hypothesis**.
- **Exact question:** Can one non-Native, stationary RoPE table on a frozen
  checkpoint guarantee the exact Native function for every short request and
  also change the long-context geometry? If not, what is the smallest causal
  escape that keeps the prefix exactly Native and gives every later query a
  consistent long-table relative phase?
- **Inputs:** the exact transplant obstruction, the completed Native/s4 session
  policy, the 2026-09-03 Native-isotonic tradeoff, and the failed boundary-slope
  operator linked below.
- **CPU checker:**
  [`scripts/analysis/verify_prefix_phase_handoff.py`](../../../../scripts/analysis/verify_prefix_phase_handoff.py),
  SHA-256 `b4d841c26954c646bb85927e679279904a4a8d004175cec8787f6c1a5885753b`.
- **Compute:** deterministic NumPy algebra only; no model load, inference,
  training, remote access, GPU, or paid compute.

## 1. Decision

The exact one-table version of the Native--long objective is impossible under
standard stationary RoPE and universal exact-preservation quantifiers. A
non-Native table may have a small measured short cost, but it cannot guarantee
the exact Native function for all short inputs. The 2026-09-03 result is
consistent with this boundary: its Native-derived table improves natural
likelihood while losing fresh structured capability at 4K/8K. That observation
is not itself the proof.

The repository already contains one empirically supported escape: choose the
exact Native branch for a short request and one frozen long table for a long
request before prefill. The completed OLMo owner verifies exact short-route
parity and strong protocol-specific long capability. This is a practical
two-mode solution, not a single-table theorem.

There is also a strictly stronger causal construction worth separating from
both the impossible one-table objective and the completed per-request route:

> Process positions before `L_native` exactly under Native RoPE, but retain a
> standard Native key cache. At `L_native`, rephase those cached keys in place
> from the Native frame to the frozen long frame. Then rotate every new query
> and key in that same long frame.

This **prefix-preserving coordinate handoff** removes the cross-boundary phase
mismatch of the failed boundary-slope map while preserving every prefix state
by causality. Its rotary algebra is exact and CPU checked. Its transformer
behaviour is unknown: no model execution has tested whether a Native-computed
prefix is a better initial state for later long-frame queries.

## 2. One static table cannot guarantee exact Native short behaviour

### 2.1 Assumptions and quantifier

For one fixed rotary plane, let the Native and candidate frequencies be
`omega, omega' in (0, pi)`. The candidate uses one table for every position in
the request. “Exact Native” means equality of the rotary bilinear form for all
content vectors and every short-context integer relative position; it does not
mean equality on a finite evaluation sample.

### 2.2 Fixed-coordinate proposition

If Q/K coordinates remain fixed and

\[
x^\top R(\omega'\Delta)y=x^\top R(\omega\Delta)y
\quad\text{for all }x,y
\]

at `Delta=1`, then the two rotation matrices are equal. Hence
`omega'=omega mod 2 pi`; the interval `(0,pi)` removes the alias and gives
`omega'=omega`. Applying the argument to every fixed rotary plane proves that
a universally exact non-Native table does not exist in this class.

Allowing arbitrary invertible, position-independent Q/K compensation does not
open a frequency-changing escape. The existing
[`transplant obstruction`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md)
proves that the frequency multiset must still match up to sign/permutation and,
for integer positions, excluded `2 pi` aliases.

**Corollary.** Under these assumptions, exact short-function preservation and
a nontrivial long-frequency change cannot be supplied by the same static
branch. Approximate retention remains an empirical multi-objective problem; the
proposition does not say its Pareto set is empty.

## 3. A nonlinear absolute-phase boundary cannot retain stationary relative phase

Let a per-token phase map be `f(t)` and suppose every attention pair depends
only on relative distance, so

\[
f(q)-f(k)=h(q-k)\quad\text{for all integer }q,k.
\]

Setting `k=0` and subtracting the constant `f(0)` gives a function `g` with
`g(q)-g(k)=g(q-k)`. Therefore `g(a+b)=g(a)+g(b)` on the integers and
`g(t)=t g(1)`: the phase map is affine. A per-token map cannot change slope
after a boundary while retaining one stationary relative-position kernel.

For the historical continuous boundary-slope map,

\[
f(t)=\begin{cases}
\omega t,&t<L,\\
\omega L+\omega'(t-L),&t\ge L,
\end{cases}
\]

a suffix query `q>=L` and prefix key `k<L` have phase error relative to the
stationary long table

\[
[f(q)-f(k)]-\omega'(q-k)=(\omega-\omega')(L-k).
\]

This vanishes only for an unchanged frequency or a key exactly at the
boundary. It is the structural mismatch diagnosed by the completed
[`Native/s4 owner`](../results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md),
whose boundary-slope arm scored zero in its stated core-4 protocol. The algebra
explains the mismatch; it does not by itself explain the behavioural score.

## 4. Prefix-preserving coordinate handoff

Fix one already-owned long table `Omega_long` and its already-owned long
attention amplitude. Do not derive a new curve, gain, cutoff, or selector.
At every transformer layer:

1. Process prefix positions `t<L_native` through the untouched Native branch.
2. At the boundary convert each already-Native-rotated cached key in place:

   \[
   K_{\rm long}(t)=
   \frac{c_{\rm long}}{c_{\rm Native}}
   R((\Omega_{\rm long}-\Omega_{\rm Native})t)
   K_{\rm Native}(t).
   \]

   Plane rotations commute, so this is exactly
   `c_long R(Omega_long t) K_raw(t)` when the original cache used
   `c_Native R(Omega_Native t) K_raw(t)`.
3. For every suffix query `q>=L_native`, use
   `Q_long(q)=R(Omega_long q) Q_raw(q)` and attend only against long-rotated
   cached/new keys.
4. Keep the long frame fixed for the remaining request and cached decode.

### 4.1 Exact properties

- **Prefix identity.** By causal induction over positions and layers, every
  prefix hidden state and logit is the Native one, provided the implementation
  uses the same numerical kernel/order. Future suffix tokens cannot change a
  causal prefix.
- **Cross-boundary stationarity.** A suffix query and any prefix key use the
  same long frame, so their positional bilinear contains the ordinary
  `Omega_long(q-k)` relative phase. The boundary-slope error in Section 3 is
  absent.
- **Zero learned parameters.** The construction changes cache preparation and
  phase dispatch only. It reuses an existing frozen long table and gain.
- **Causal serving.** A long prompt can be processed as an exact Native prefix
  chunk followed by an in-place cache conversion and a long-frame suffix
  chunk. Access to the stored rotated keys and their absolute positions is
  required; pre-RoPE keys and a second persistent K frame are not.

The CPU checker verifies exact zero score error for the Native prefix branch
and for a suffix query against a prefix key rephased in the long frame. Direct
long rotation and in-place rephasing agree within `2.22e-16` even with unequal
positive Native/long rotary amplitudes. A tokenwise single-head cache simulation
matches the direct handoff definition within `3.33e-16`, and its prefix outputs
match the pure-Native reference exactly. A three-layer causal stack with dense
Q/K/V/O, residuals, and nonlinearities matches within `2.22e-16` and preserves
the Native prefix exactly. Chunked GQA (`4` query / `2` KV heads) and MQA
(`4` / `1`) match within `4.44e-16` and `2.78e-16`. The scheduler turns a
requested `(4,9,5)` chunking around boundary `11` into `(4,7,2,5)`, so no
attention call straddles the handoff. These are float64 model-free
rotary/attention checks, not bitwise transformer parity or quality evidence.

### 4.2 What it does not preserve

Once a query crosses the boundary, all of that query's positional interactions
use the long frame, including nearby prefix keys. The construction preserves
prefix computation, not every Native short-distance interaction inside a long
request. It also does not equal the output of running the long table from token
zero, because prefix hidden states were produced by Native layers.

### 4.3 Source-level integration audit

The personal-PC source inspection used Transformers `4.57.6`; the recorded
work-machine runtime is `5.15.1`, so this is an implementation-shape finding,
not a frozen canonical-source receipt. In the inspected OLMo2 path:

1. `Olmo2Model.forward` computes one `(cos,sin)` tuple and passes it to every
   decoder layer;
2. `Olmo2Attention.forward` rotates Q and K before calling `Cache.update`;
3. `Cache.update` returns the same stored key tensor that the current attention
   call consumes.

Therefore replacing only the shared rotary module cannot implement prefix
handoff. But a second persistent K frame is unnecessary: blockwise rotations
commute, so an already-rotated Native cache can be conjugated in place at the
boundary. The minimal integration contract is:

- split any prefill chunk exactly at `L_native` before attention;
- let prefix attention and cache use the ordinary Native K/V path;
- between the final prefix call and first suffix call, rotate every layer's
  cached K by `(Omega_long-Omega_Native)*position` and multiply by the ratio of
  long to Native rotary amplitudes;
- rotate all suffix Q/K with the existing long table and amplitude thereafter.

No attention call then mixes frames, so each prefix/suffix call can in principle
retain an ordinary Flash-compatible causal path. This has not been verified on
the canonical runtime. Persistent cache size remains the ordinary `K+V`; the
boundary adds one linear pass over stored keys. GQA/MQA rephase each stored KV
head once, not once per query head. Dynamic/static cache mutation details and
finite-precision parity remain version-specific and unverified.

## 5. Relation to completed evidence

| Object | Valid conclusion | Relation to this boundary |
| --- | --- | --- |
| [`Native/s4 session policy`](../results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | Exact Native short sessions plus protocol-specific long capability using a branch fixed before prefill | completed practical escape; entire long request uses the long table |
| [`Native-isotonic`](../results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) | Better natural likelihood/retention point estimates, worse fresh core-4 at 4K/8K | demonstrates one approximate one-table tradeoff; not the impossibility proof |
| [`Selective-31`](../results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) | exact fixed mask is negative against registered controls | different head/slot intervention; neither validates nor refutes a prefix handoff |
| failed boundary-slope arm | cross-boundary absolute-phase construction scored zero | prefix handoff removes its algebraic mismatch but has no behavioural evidence |

## 6. Decision table

| Requirement | One static table | Completed session route | Prefix handoff |
| --- | --- | --- | --- |
| exact Native for every short request | only if the table is Native | yes | yes |
| nontrivial frozen long table | conflicts with exact first column | yes | yes |
| exact Native prefix states inside a long request | no | no | derived yes |
| one stationary long phase for suffix query-to-prefix-key pairs | yes | yes | derived yes |
| ordinary unmodified KV-cache path | yes | yes | no; cache preparation changes |
| behavioural long evidence | candidate-specific | yes, OLMo scope | none |

## 7. Supported and unsupported claims

### Supported

- The universal exact-no-harm version of the one-static-table objective is
  closed under the stated standard-RoPE assumptions.
- The historical boundary-slope map has an unavoidable cross-boundary relative
  phase error when a frequency changes.
- Rephasing cached prefix keys into the same long frame as suffix queries
  removes that error while leaving causal prefix computation exactly Native at
  the algebraic level.
- Stock OLMo2 cache flow cannot realize the handoff through a rotary-module
  replacement alone; the source-level contract requires boundary splitting and
  an in-place per-layer cached-K conversion.
- The completed per-request binary policy remains the only repository-owned
  solution with behavioural evidence for exact short sessions plus long use.

### Unsupported

- that prefix handoff improves NLL, retrieval, QA, EOS, or any model output;
- bitwise transformer parity without an exact implementation receipt;
- equivalence to a full-request static long table;
- universality across checkpoints, GQA/MQA cache layouts, kernels, or tasks;
- a new frequency profile, gain, selector, sweep, manuscript claim, or GPU run.

This owner authorizes no continuation or model execution.
