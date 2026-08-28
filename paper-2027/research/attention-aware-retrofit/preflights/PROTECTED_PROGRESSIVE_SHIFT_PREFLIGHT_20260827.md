# Protected progressive frequency-shift preflight (2026-08-27)

- **Status:** `FROZEN_PROTOCOL_DESIGN_NOT_EXECUTED`
- **Direction status (2026-08-28):** the protected-ramp protection-formula
  scanning direction this preflight serves was superseded as the active
  investigation direction by the leave-one-band-out frequency-band causal
  attribution design in
  [`../analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md`](../analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md)
  §9 (`DESIGN_NOT_EXECUTED`; requires a new preflight and explicit compute
  authorization). This preflight remains `FROZEN_PROTOCOL_DESIGN_NOT_EXECUTED`.
- **Role:** separate post-submission method-development preflight; not
  manuscript evidence, a result, or compute authorization
- **Primary owner:** `INDEX.md` §6.2–§6.3 and the current zero-training
  progressive-policy owners
- **Companion bridge:**
  [`MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md)
- **Authorization:** every GPU or checkpoint-evaluation stage requires a new
  explicit author authorization. This document does not authorize a run.

This protocol tests one method only: protect the fast frequency bands of the
current successful simple progressive frequency-shift curve using fixed
YaRN-style rotation cutoffs. It is a new test of protecting the successful
simple curve. It is not a rerun of, or rebuttal against, the earlier
protected-band Cosh result recorded in
[`ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md`](ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md)
and
[`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`](../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md).
The earlier result remains its own candidate-specific owner.

The existing simple progressive curve, its implementation identity, and its
claim ceiling remain owned by
[`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](../results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md)
and the fresh natural-text confirmation in
[`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md).
Resolve the current curve and its gain from those owners at execution time;
do not copy unstable result numbers into this preflight.

## 1. Causal question and scope

The question is:

> Can the current simple progressive shift retain its tested long-range NLL
> improvement while protecting fast bands well enough to avoid its short-window
> cost, using one fixed zero-training table and no routing?

The only new method variable is the fixed protection transform applied to the
existing simple curve's shift. The checkpoint, content, rows, decoder,
attention operator, and matched long gain are held fixed in the decomposition
that compares the simple and protected curves.

This protocol does not identify a pure interior-allocation effect by itself in
Stage 1, because the Native and s4 endpoint-matched arms differ in support.
Within Stage 1, C/D/E are the controlled allocation block and D versus E is
the pure interior-$z$ protection contrast. A complete practical policy may
include support, allocation, and gain, but those variables are not to be
described as one causal effect. Amplitude/gain, adaptation, and serving/routing
remain separate estimands under the project’s
[`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md)
§2 grammar.

## 2. Exact protected-curve construction

Let the verified Native frequency tensor be `omega_k`, the model Native window
be `L_native`, and the declared extension factor be `s`. For every frequency,
compute Native-window rotations

```text
r_k = L_native * omega_k / (2 * pi)
```

Let `m_old,k` be the existing simple progressive curve's shift coefficient for
that frequency and factor `s`. Define the fixed protection interpolation:

```text
t_k       = clip((32 - r_k) / (32 - 1), 0, 1)
h_k       = t_k * t_k * (3 - 2 * t_k)       # smoothstep
m_new,k   = m_old,k * h_k
```

Therefore:

- if `r_k >= 32`, `h_k = 0` and the frequency receives no shift;
- if `r_k <= 1`, `h_k = 1` and the existing simple shift is unchanged;
- if `1 < r_k < 32`, the existing shift is smoothly attenuated;
- the cutoffs `1` and `32` are fixed protocol constants and are not tuned.

Apply `m_new,k` multiplicatively to the existing project transformation:

```text
omega_new,k = omega_k * ((1 - m_new,k) + m_new,k / s)
```

Use the project’s existing frequency transformation and serialization path.
Do not replace it with a new interpolation operator, a scalar-base rule, a
different Cosh profile, a new gain, or a learned table. Record the realized
Native, simple, protected, and endpoint-matched-uniform tensors and hashes
before reading any model metric.

The method is target-free: construction reads the Native frequency tensor,
`L_native`, and the declared `s`; it does not read `L_target`, task labels,
losses, attention statistics, Qwen results, or any Stage 1 outcome. The
rotation cutoffs are defined in Native-window coordinates and are unchanged
across all later stages.

## 3. Stage 1 — six-condition NLL decomposition

### Checkpoint and content

Use the same OLMo-2-0425-1B-Instruct checkpoint and frozen FineWeb document
rows as the current simple progressive owner. Resolve checkpoint, tokenizer,
content, row, and Native-buffer identities from the owner at execution time;
record their hashes in the manifest rather than duplicating them here.

Declare `s=4` before table construction. Freeze one common randomized row order
and use the same document/token content in every condition. The decoder,
precision, attention backend, causal mask, tokenization, sequence truncation,
and NLL mask are identical across conditions. No route changes during a
request, no learned parameters, and no weight updates are allowed.

### Conditions

| Cell | Frequency table | Gain | Scientific role |
| --- | --- | --- | --- |
| A | Native frequency | `1.0` | Native baseline and short-window reference |
| B | Native frequency | matched long gain | gain-only decomposition control |
| C | s4 endpoint-matched uniform interior table | matched long gain | support-change reference with linear interior allocation |
| D | current simple progressive table at `s=4` | matched long gain | existing successful curve |
| E | protected progressive table at `s=4` | matched long gain | new method under test |
| F | official Transformers YaRN | official YaRN gain | external deployment reference |

C, D, and E must share the checkpoint, exact endpoints, log-frequency support,
gain, rows, decoder, precision, hardware, causal mask, and every other
evaluation setting. Consequently D versus E is a pure interior-$z$ contrast:
the protection factor changes the placement of the shifted frequencies while
the endpoint support and gain are fixed. A versus B isolates the matched-gain
change on the Native table. C versus D/E provides the endpoint-matched
allocation comparison.

F is an external reference and is not a pure-$z$ control. Its official gain and
implementation identity must remain distinct from the matched gain used by
B–E. Do not state that F proves or disproves the protected curve's mechanism.

The previously observed 4K degradation motivates this test only. Link to the
canonical simple-curve owner above; do not copy an unstable historical delta or
use it as a new baseline estimate.

## 4. Position-stratified primary NLL analysis

Evaluate the same 16K document rows and report teacher-forced NLL by token
position in these predeclared bins:

```text
0-4K, 4-8K, 8-12K, 12-16K, and final 1K tail
```

The final 1K tail overlaps the 12–16K bin by design and is a named summary,
not an independent partition. Compute per-row means within each bin before
averaging across rows. The primary plot is fixed before execution: for each of
B, C, D, and E, plot NLL delta against A by position bin, with D versus E
shown separately as the allocation/protection contrast. F may appear as an
external reference line only if its tokenization and positions are exactly
comparable; otherwise retain it in a separate panel.

The primary Stage 1 outcomes are:

1. short-window NLL at 4K, with E minus A as the practical retention contrast;
2. long-range NLL at 8K and 16K, with E compared against A and against D;
3. the position-stratified NLL-delta plot and per-row data needed to reproduce
   it.

All endpoint values are NLL, so lower is better. Do not convert them into task
capability claims.

### Retention calculation

For length `L`, let `N_A(L)` be the Native/A NLL, `N_D(L)` the existing simple
progressive/D NLL, and `N_E(L)` the protected/E NLL, using the same frozen rows.
Define improvement relative to Native as positive:

```text
Delta_simple(L)    = N_A(L) - N_D(L)
Delta_protected(L) = N_A(L) - N_E(L)
Retention(L)       = Delta_protected(L) / Delta_simple(L)
```

The long-range retention gate requires `Retention(8K) >= 0.80` and
`Retention(16K) >= 0.80`, with the denominator rule below frozen before
execution. The gate is evaluated on the point estimates and accompanied by
paired row-bootstrap intervals; intervals are not silently substituted for the
registered ratio.

If `Delta_simple(L) <= 0`, the simple curve has no positive NLL improvement at
that length, so retention is **not defined** and the gate fails at `L`; do not
flip the sign or call a negative ratio retention. If
`0 < Delta_simple(L) < 0.01` NLL, the denominator is **near zero** under the
predeclared `epsilon=0.01` NLL rule; report retention as undefined/unstable and
fail the gate at `L` rather than allowing a large ratio to pass. A denominator
at least `0.01` is positive and uses the formula directly. The zero and
near-zero cases are reported as protocol outcomes, not repaired by changing
the baseline or length.

### Preregistered Stage 1 gates

Stage 1 passes only if both gates hold:

1. **Short-window retention:** `N_E(4K) - N_A(4K) <= +0.01` NLL.
2. **Long-range retention:** `Retention(8K) >= 0.80` and
   `Retention(16K) >= 0.80` under the signed/near-zero denominator rule above.

If the protected curve passes, it is a practical candidate, not yet a
capability or position-causal result. D versus E and A versus B must still be
reported so that a pass is not attributed to gain or support.

Do not retune cutoffs, `m_old`, `s`, gain, or table construction after seeing
the `s=2`/`s=4` comparison, any Qwen result, or any position-stratified plot.
Stage 1 is fixed at `s=4`; later `s=2` is generalization, not tuning.

## 5. Randomization, bounded uncertainty, and receipts

This is a frozen-checkpoint evaluation, not a training replication. There is no
training-seed uncertainty. Before any model output is generated:

- create one common row permutation with RNG seed `20260827` and apply it to
  A–F;
- use greedy decoding with the exact existing EOS, answer extraction, maximum
  generation, and normalization contract;
- use bootstrap RNG seed `20260828`, with paired row resampling stratified by
  document/task where the owner’s scorer requires it;
- record the number of bootstrap replicates before execution and use the same
  count for every cell and primary contrast.

The run manifest must include, at minimum:

- checkpoint and tokenizer revisions plus weight/tokenizer hashes;
- exact content, row, token-ID, NLL-mask, and common row-order hashes;
- Native, uniform-s4, simple-progressive, and protected-progressive frequency
  tensors with float32 hashes;
- `s=4`, `r_k` convention, fixed cutoffs `1` and `32`, smoothstep formula,
  `m_old` source revision, and all gain values;
- model/code revision, decoder/EOS settings, causal mask, precision, hardware,
  attention backend, and maximum physical sequence length;
- per-cell output paths, per-row JSONL, position-bin definitions, and the
  bootstrap seed/replicate count.

The runner must fail closed if any cell changes token content, rows, answer
mask, causal mask, decoder, gain, support, or table identity; if the protected
tensor is not derived from `m_old` by the registered formula; if attention
falls back from the required backend; or if any output is non-finite. Raw
per-row outputs and checkpoints remain outside Git; only a sanitized result
owner may later be promoted after provenance review.

## 6. Stage 2 — contingent capability and phase bridge

Stage 2 starts only if both Stage 1 NLL gates pass. Run the following in order,
without changing the protected rule, cutoffs, gain, or table after Stage 1:

1. RULER core-4 at 8K and 16K using the existing official scorer;
2. RULER unseen-nine confirmation using the existing confirmation contract;
3. the existing
   [`MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md);
4. Qasper or 2Wiki using the existing official scorer and frozen answer
   extraction contract.

RULER and 2Wiki rows that share generator/task families with adaptation data
are task-family adaptation evidence, not unseen-task transfer. Keep official
task score and answer-token NLL as separate endpoints.

For step 3, use the protected curve’s normalized interior profile only after
re-embedding it to Native support and Native amplitude as required by the
companion bridge’s pure-interior contract. Do not pass the raw s4 long table
with its changed endpoints or matched long gain into the pure-$z$ 2x2. Freeze
the re-embedded tensor hash before any bridge output. The bridge’s ideal
four-cell pattern is:

| Cell | Expected pattern |
| --- | --- |
| Native + contiguous positions | normal reference capability and answer NLL |
| Native + virtual-gap positions | capability degrades and answer NLL worsens |
| Protected table + contiguous positions | approximately preserves Native continuous performance |
| Protected table + virtual-gap positions | restores virtual-gap performance toward Native continuous performance |

“Approximately” is judged by the companion bridge’s pre-registered
non-inferiority margins, not by visual inspection. If only one endpoint follows
the ideal pattern, retain the endpoint-specific result and do not call the
joint bridge positive.

Stage 2 does not make the protected method a universal capability improvement.
It tests one checkpoint, one rule, and the declared task contracts.

## 7. Stage 3 — unchanged-cutoff generalization

Stage 3 starts only after the OLMo rule is settled by Stage 1 and the contingent
capability sequence has been completed or stopped by its gates. Reuse exactly
the same protection cutoffs, smoothstep, `m_old` implementation, gain policy,
decoder, and analysis rules without tuning:

| Model | Extension factor | Evaluation |
| --- | ---: | --- |
| OLMo | `s=2` | 8K |
| OLMo | `s=4` | 16K |
| Qwen Native 32K | `s=2` | 64K |
| Qwen Native 32K | `s=4` | 128K |

The Qwen rows are unverified future evaluations. They are not an existing
claim, and the corrected Qwen 128K evidence in the mature owner must not be
reused as a result for this protected-progressive method. Transfer to Qwen is
allowed only after the OLMo rule is frozen and only with the same cutoffs; no
Qwen-specific retuning is permitted.

## 8. Stopping rules and interpretation

Stop before execution if the existing simple curve or its `m_old` coefficients
cannot be resolved reproducibly, if the protected tensor changes endpoints or
gain in the D/E comparison, or if the content/answer boundaries cannot support
the position bins and exact answer mask.

Invalidate the run if any manifest, decoder, backend, table, gain, or row
identity drifts across cells, if the first real batch is non-finite, or if raw
per-row receipts are missing. Do not repair an invalid run by rerunning only
one condition.

After Stage 1, stop and classify the result as follows:

| Result | Interpretation and next action |
| --- | --- |
| Both NLL gates pass | Establishes a practical zero-training, no-routing candidate under the tested OLMo contract; proceed to Stage 2 only. |
| Short-window improves but long-range retention fails | Suggests middle-band dependence or over-protection; consider layerwise protection or routing only through a new preflight. Do not claim the single table solves the joint objective. |
| Short-window loss continues | Inspect the A–C decomposition. If support or gain explains the loss, stop further single-table shape sweeps; do not retune cutoffs. |
| D and E differ, but A–C shows the dominant change is gain/support | Attribute the result to the relevant component; do not call it a pure protection/allocation effect. |
| Stage 2 task score and NLL disagree | Preserve both endpoint results; do not replace capability with NLL or vice versa. |
| Virtual-gap Native does not degrade | The phase bridge is not identified; do not claim the protected table restores a position failure. |
| Virtual-gap Native degrades but protected does not recover | The protected rule does not repair the tested phase failure; stop this candidate and do not call it a model ceiling. |

A positive Stage 1 or Stage 2 result does not establish compatibility or
performance outside the tested checkpoint, rows, lengths, endpoints, and
hardware. It does not prove that the cutoffs are optimal, that the table is
universally better, or that routing/adaptation/gain are irrelevant. A negative
result is candidate-specific and does not invalidate the completed fixed-
support training or mature pure-$z$ owners.

## 9. Provenance and linked protocols

- Existing simple progressive policy and its bounded claim:
  [`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](../results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md)
- Fresh natural-text confirmation and allocation/routing separation:
  [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md)
- Prior protected-band Cosh candidate, explicitly not this method:
  [`ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md`](ZERO_PARAMETER_PROTECTED_BAND_PREFLIGHT_20260824.md)
- Matched-content phase 2x2 bridge invoked only in Stage 2:
  [`MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md`](MATCHED_CONTENT_PHASE_2X2_BRIDGE_PREFLIGHT_20260827.md)
- Current agenda and method-entry order:
  [`INDEX.md`](../../../../INDEX.md) §6.2–§6.3

This document records a separate protocol only. It contains no experiment
result, launch receipt, or authorization to use GPU, remote, or paid compute.
