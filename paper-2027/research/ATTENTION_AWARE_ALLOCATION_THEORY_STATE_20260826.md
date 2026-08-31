# Attention-aware RoPE allocation: theory state

- **Original date:** 2026-08-26
- **Lifecycle update:** 2026-08-30
- **Status:** active theory-continuation boundary for work after the current
  September manuscript cycle; experiment priority is superseded by the R1/R2
  order in `INDEX.md` §6
- **Role:** durable theory continuation owner, not a parallel experiment plan
- **Not:** a manuscript claim, experiment result, action queue, or compute
  authorization
- **Agenda authority:** [`../../INDEX.md`](../../INDEX.md) §6
- **Live state:** [`../HANDOFF.md`](../HANDOFF.md)

This document answers what the paper established, what later research ruled
out, and which mechanism bridge remains available. Its former scheduling
priority is superseded: the current order freezes a candidate contract, runs the
minimal Native-window/far-tail screen, and uses the matched-content bridge only
when a valid result is mechanism-ambiguous and the answer changes candidate
design. Leave-one-band-out remains conditional after a positive bridge.

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
| Stateless continuous-boundary target-free candidate: core-four RULER `0.0000` at both 8K and 16K | this exact boundary-slope construction is a closed negative on the tested checkpoint and harness | every target-free or absolute-position-dependent operator must fail |

The mature-result owners are indexed under
[`attention-aware-retrofit/`](attention-aware-retrofit/README.md). The compact
receipts preserve hashes; they do not replace raw per-row artifacts.
The continuous-boundary negative is owned by
[`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`](attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md)
§2 and must not be relaunched as the same candidate.

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

> Find a fixed allocation, chosen without `L_target`, that preserves or improves
> in-window behaviour and improves extrapolation on a frozen checkpoint; only
> after that zero-training gate consider grouped allocation or matched
> adaptation, and verify capability separately from teacher-forced tail NLL.

Evaluation horizons may test the frozen construction. They must not enter its
definition.

## 4. Available conditional identification bridge

When a valid R2 candidate result cannot distinguish content failure from
position/allocation failure, and that distinction changes candidate design,
the available identification bridge is a matched-content phase intervention:

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

## 5. Conditional method directions after the bridge

1. **Only if the matched-content bridge is positive:** the next mechanism
   design is the leave-one-band-out restoration in
   [`PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md`](attention-aware-retrofit/analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md)
   §9. It asks which frequency movement carries the short-window cost and the
   long-range benefit. It remains `DESIGN_NOT_EXECUTED`, needs a new preflight,
   and is not an active route before the bridge.
2. **Only after that mechanism gate warrants heterogeneous allocation:** test
   grouped per-layer allocation before per-head allocation. O7 contains a
   hypothesis that head-dependent priors may favour heterogeneous tables; it is
   not a proved Jensen result or a method-selection theorem. Existing code is
   untrained evidence only.
3. **If full and tail still move in opposite directions:** optimize a
   position-resolved matched-training objective, not another static table
   score. Preserve a Native-table matched-adaptation arm.
4. **Only after one 1.485B matched pair improves in-window, far-tail, and one
   capability endpoint:** run multiple seeds, then consider a second
   checkpoint.
5. **Do not reopen:** the tested stateless continuous-boundary target-free
   candidate, cosine-only collision, static collision/logdet selectors,
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
