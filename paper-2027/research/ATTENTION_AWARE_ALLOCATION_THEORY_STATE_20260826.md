# Attention-aware RoPE allocation: theory state

- **Date:** 2026-08-26
- **Role:** durable theory continuation owner for work after the current ICLR
  submission
- **Not:** a manuscript claim, experiment result, action queue, or compute
  authorization
- **Agenda authority:** [`../../INDEX.md`](../../INDEX.md) §6
- **Live state:** [`../HANDOFF.md`](../HANDOFF.md)

This document answers one question for the next working session: what has the
paper established, what has the later research ruled out, and which missing
bridge must be identified before another method is trained?

## 1. Established theoretical core

The current paper owns a complete first-generation result:

1. A finite RoPE table separates into sampled support and normalized interior
   allocation,
   \[
   x_k=-\log\omega_k=a+Rz_k.
   \]
2. At fixed `(a,R)`, the three-seed exact-range intervention identifies `z` as
   a consequential training-time variable.
3. Each frequency is a full two-dimensional positional subspace
   \(V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}\).
   Block whitening gives the exact Rényi-2 identity
   \[
   r_2(R)=\frac{2K}{1+(K-1)\bar c}.
   \]
4. Slow geometric pairs collapse toward a common positional subspace. This is
   a static finite-basis diagnosis, not a language-model ordering rule.
5. EVQ-Cosh is the unique solution of its stated convex surrogate and a
   zero-learned-parameter construction on the identified axis. It is not a
   universal language-model optimum.
6. Unequal frequency multisets cannot be exactly absorbed by fixed,
   position-independent invertible Q/K maps. Approximate adaptation and other
   operators remain possible.

Canonical owners are the
[`full-RoPE report`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md),
[`causal-variable grammar`](ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md),
[`three-seed exact-range result`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md),
and the
[`transplant theorem`](../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md).

## 2. Empirical constraints on a second-generation theory

Any new theory must explain all rows below without changing their estimands.

| Observation | What it establishes | What it rules out |
| --- | --- | --- |
| Frozen 50M weights/table crossing: PPL `7.14/76.20/23.05/7.16` | weights co-adapt to the installed table | static table geometry alone predicts LM quality |
| Exact-range 151.9M, three seeds: `+0.026/-0.281/-0.176/-0.146` | allocation matters during matched training | `z` is only notation for base/support |
| Target-retargeted reversal: `+0.060/+0.227/+0.460` OOD | useful allocation is conditional on selected support | allocation and support add independently |
| Frozen OLMo same-support: `0.56% -> 60.47%`; coarse ramp `61.04%` | mature behaviour remains highly sensitive to `z` | the detailed Cosh profile is uniquely responsible |
| Matched co-adaptation: 4K full `+0.00098`, 8K/16K tail `-0.0387/-0.0877`, long full NLL worse | a small learned move can preserve the short window and improve the far tail while hurting intermediate/full averages | one scalar “better table” score captures all positions |
| Registered dose curve | allocation produces a graded full/tail response; analytic Path A misses its joint guard; static `r2` misses the useful dose | static `r2` is a behavioural selector |
| Native 4K core-four `1.00/0.85/0.60/0.03` | capability varies sharply by task | comparison with independently generated 8K/16K rows identifies a position failure or model ceiling |

The mature-result owners are indexed under
[`attention-aware-retrofit/`](attention-aware-retrofit/README.md). The compact
receipts preserve hashes; they do not replace raw per-row artifacts.

## 3. The missing object

The first-generation theory studies a frequency table under a declared
separation measure. A trained transformer instead realizes a coupled object:

\[
\text{behaviour}
=\text{RoPE table}\times\text{Q/K coefficients}\times
\text{attention measure}\times\text{training adaptation}.
\]

The 50M crossing shows that the interaction term dominates a frozen swap. The
co-adaptive oracle then shows that the same allocation displacement can improve
the final tail while worsening long full-sequence NLL. A useful second-generation
theory must therefore be position-resolved and co-adaptation-aware. It cannot be
another scalar functional of one shared table under assumed content weights.

The target is deliberately target-free:

> Find a fixed or grouped allocation, chosen without `L_target`, that preserves
> in-window behaviour and improves extrapolation after a matched adaptation
> protocol, then verify that the gain transfers to capability rather than only
> teacher-forced tail NLL.

Evaluation horizons may test the frozen construction. They must not enter its
definition.

## 4. Required identification bridge

The next missing experiment is not another model scale or another RULER length.
It is a matched-content phase intervention:

- freeze checkpoint, token content, token order, causal mask, answer, decoder,
  and evaluation rows;
- compare contiguous position IDs with a virtual-gap map that increases the
  answer-to-evidence separation while preserving local order;
- cross the same position maps with Native and one frozen candidate table;
- report official task score and answer-token NLL on the same rows;
- use short-condition-success rows as a mechanism subset, labelled as such,
  rather than a population benchmark.

This 2x2 isolates `table x realized phase` on identical content. A global
position offset is invalid because relative RoPE is invariant to it; the map
must change relative separations.

### Decision readings

| Result | Reading |
| --- | --- |
| Short condition succeeds; virtual gap hurts Native; candidate restores it | position coding is a causal part of the remaining headroom |
| Short condition succeeds; both tables fail under the virtual gap | tested allocation does not repair the position failure; do not call it a model ceiling |
| Short condition already fails | row is not informative for position-versus-capability identification |
| Candidate improves task score but not answer NLL, or vice versa | endpoint mechanism differs; keep both rather than pooling |

This is a protocol design only. No GPU run is authorized by this document.

## 5. Method directions after the bridge

1. **If the matched-content bridge is positive:** test grouped per-layer
   allocation before per-head allocation. The O7 Jensen result supplies a
   theoretical reason for heterogeneity; existing code is untrained evidence
   only.
2. **If full and tail still move in opposite directions:** optimize a
   position-resolved matched-training objective, not another static table
   score. Preserve a Native-table matched-adaptation arm.
3. **Only after one 1.485B matched pair improves in-window, far-tail, and one
   capability endpoint:** run multiple seeds, then consider a second
   checkpoint.
4. **Do not reopen:** cosine-only collision, static collision/logdet selectors,
   `kappa_att` ordering, LeRoPE `w^(1/3)` oracle, direct distance mapping,
   coverage residual, phase-risk, two-document direct-`z`, or the two failed
   analytic static tables. The complete anti-repeat ledger is
   [`../../INDEX.md`](../../INDEX.md) §3.4.

## 6. Static-rank diagnostics: retained scope

[`third_axis_ceiling.py`](../../scripts/analysis/third_axis_ceiling.py) remains
the reproducible owner for the best-found static `r2` landscape. Its numbers
must always travel with measure, support, optimizer, and restart conventions.
It establishes neither support invariance nor a global or behavioural ceiling.
The detailed numerical table remains in `INDEX.md` history until separately
archived; it is not a candidate-selection rule.

## 7. What the PC must have

Git contains the manuscript, owners, compact receipts, code, tests, and this
theory state. Git does not contain checkpoints, raw GPU rows, caches, or the
prepared one-billion-token corpus. If a future decision depends on raw dose
position bins or per-example capability outputs, transfer and hash those
artifacts explicitly before analysis; do not reconstruct them from compact
means.
