# Matched-content phase 2x2 bridge preflight (2026-08-27)

- **Status:** `FROZEN_PROTOCOL_DESIGN_NOT_EXECUTED`
- **Role:** post-deadline method-development preflight; not manuscript evidence,
  a result, or compute authorization
- **Scheduling gate:** research execution is ineligible before the 2026-09-25
  manuscript deadline. After that date this remains only a design until a new
  author decision confirms that the bridge still has priority.
- **Decision owner:** `INDEX.md` §6.2 and
  [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](../../ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
  §4
- **Primary implementation target:** the mature OLMo checkpoint and task
  contract already owned by
  [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md)
- **Authorization:** none. Every GPU or checkpoint evaluation stage requires a
  new explicit author authorization after the scheduling gate. This document
  does not authorize a run.

This preflight is the sole next identification bridge, not another model-scale
sweep. It tests whether a controlled change in realized relative phase creates
the failure that the candidate table is supposed to repair, on identical
content and with no training, gain, or serving-policy intervention.

The former protected-progressive companion route is historical and superseded;
it cannot invoke this bridge as a stage. A positive completed bridge may later
admit a separately preregistered leave-one-band-out mechanism design, but no
such design is part of this protocol.

## 1. Causal question and estimand

The question is:

> On one frozen mature checkpoint and identical content, does increasing the
> evidence-to-query relative phase separation harm Native, and does a
> pre-frozen interior-allocation table reduce that same harm?

The primary estimand is the table-by-position-map interaction. For a metric
$Y$, define

\[
  I_Y = [Y(\mathrm{candidate},\mathrm{gap})-Y(\mathrm{candidate},\mathrm{contiguous})]
       -[Y(\mathrm{Native},\mathrm{gap})-Y(\mathrm{Native},\mathrm{contiguous})].
\]

For official task score, larger is better and a positive $I_Y$ means that the
candidate recovers gap damage. For answer-token NLL, smaller is better and a
negative $I_Y$ means recovery. The Native gap contrast must also be reported;
an interaction without a measurable Native phase perturbation is not a
position-failure identification.

This is a frozen-checkpoint estimand. It does not estimate the effect of
training a model with the candidate table, the effect of attention amplitude,
the effect of LoRA adaptation, or the effect of a Native/long serving route.
Those remain separate owners under the causal grammar in
[`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md)
§2.

## 2. Pre-frozen table and position map

### 2.1 Checkpoint and candidate table

Use the OLMo-2-0425-1B-Instruct checkpoint selected by the same-support owner.
Resolve its checkpoint, tokenizer, task rows, and Native frequency buffer from
that owner at execution time and record their hashes in the run manifest; do
not copy identities or numbers into this preflight.

The candidate is the existing model-relative **derived allocation shape** from
the same-support owner, affine-embedded to the checkpoint's verified Native
support:

1. obtain the owner’s normalized interior profile
   $z^{\mathrm{derived}}$;
2. set $x_k^{\mathrm{candidate}}=a_{\mathrm{Native}}+
   R_{\mathrm{Native}}z_k^{\mathrm{derived}}$;
3. recover $\omega_k^{\mathrm{candidate}}=\exp(-x_k^{\mathrm{candidate}})$;
4. serialize the contiguous float32 frequency tensor and freeze its hash before
   content materialization or model scoring.

The candidate must satisfy, bitwise in the serialized float32 receipt,

- the same pair count as Native;
- the same fast and slow endpoints and log-frequency span as Native;
- a strictly ordered positive frequency vector;
- a different interior $z$ from Native;
- the Native rotary amplitude, precision, attention backend, and all model
  weights unchanged.

If the owner’s profile cannot be reproduced at Native support without changing
an endpoint, amplitude, or operator, stop before evaluation and return to the
protocol owner. Do not substitute anchored EVQ-Cosh, a factor-four long table,
the session policy, a YaRN-style gain, or a newly searched table.

### 2.2 Content and virtual-gap map

Use one pre-frozen row manifest from the owner’s held-out 2WikiMultiHopQA task
contract, including the exact token IDs, evidence/context boundaries, question,
gold answer, and answer mask. The primary test rows and a separate calibration
split are frozen before any four-cell output is read. The task score is the
official scorer’s token F1; exact match may be retained as a secondary output,
but it is not pooled with token F1.

For every row, the contiguous map is the ordinary position sequence

```text
0, 1, 2, ..., T-1
```

The virtual-gap map uses the same token IDs and the same causal mask, leaves
the context/evidence positions unchanged, and adds one model-relative Native
window $g=L_{\mathrm{native}}$ to every position in the question-and-answer
suffix:

```text
context:       0, 1, ..., c-1
question/answer: c+g, c+g+1, ..., c+g+(T-c)-1
```

The map must preserve local order, create no duplicate or decreasing position
IDs, and remain inside the checkpoint’s verified supported position range. A
global offset applied to every token is forbidden because relative RoPE would
make it an uninformative translation. The gap changes only relative phase
between evidence and query/answer.

No candidate construction uses an evaluation target length. The declared gap
is an evaluation intervention and is fixed before scoring; it must not be
searched or changed after inspecting an outcome.

## 3. Four cells

All four cells run the same frozen checkpoint, token IDs, rows, answer mask,
batching, precision, causal mask, decoder, and row order.

| Cell | Frequency table | Position IDs | Role |
| --- | --- | --- | --- |
| N-C | Native | contiguous | frozen short-condition reference |
| N-G | Native | virtual gap (`g = L_native`) | tests Native phase sensitivity |
| C-C | pre-frozen Native-support derived candidate | contiguous | tests table effect without a phase gap |
| C-G | pre-frozen Native-support derived candidate | virtual gap | tests candidate recovery under the same phase perturbation |

The table intervention is pure interior allocation: no support movement, rotary
amplitude/gain change, operator replacement, adaptation, or routing. The
position intervention is a phase-map change: no content, answer, causal-mask,
decoder, or token-index change. The candidate is installed for the complete
forward pass; there is no Native/long session decision and no mid-request route
change.

## 4. Primary outcomes and analysis

### Primary outcomes

1. **Official task score:** official 2Wiki token F1, computed per row with the
   existing answer normalization and scorer. Report the four cell means and
   paired row-level contrasts.
2. **Answer-token NLL:** teacher-forced NLL masked to the gold answer tokens
   plus the registered terminal EOS convention, averaged per answer token per
   row. Report the four cell means and paired row-level contrasts.

These endpoints answer different questions. Do not convert NLL into a task
score, and do not replace a missing task effect with an NLL effect.

For each endpoint, report:

- N-C, N-G, C-C, and C-G per-row values;
- Native gap effect $N\text{-}G-N\text{-}C$;
- candidate gap effect $C\text{-}G-C\text{-}C$;
- the interaction $I_Y$ above;
- paired bootstrap intervals over rows, stratified by task/template as
  applicable, with the training seed absent because weights are frozen.

The row bootstrap conditions on one checkpoint, one task manifest, and one
position-map contract. It is not checkpoint-population, model-population, or
training-seed uncertainty.

### Preregistered validity and success gates

The calibration split is used only to verify that the prompt builder, answer
mask, and official scorer operate on the intended short condition. The primary
test rows are never selected using any of the four primary outcomes.

The run is **valid** only if all preflight receipts pass and the Native
contiguous calibration cell produces finite logits, finite answer NLL, and
officially scoreable answers. A valid run is **positive for the full bridge**
only when all of the following hold on the primary test rows:

1. N-G is worse than N-C on official task score and answer-token NLL in the
   expected directions, with the paired 95% interval excluding zero for both;
2. the candidate is non-inferior to Native in the contiguous condition, using
   margins frozen before execution: at most 2 percentage points of token F1
   loss and at most 0.02 NLL per answer token increase;
3. C-G recovers the Native gap damage on both primary endpoints: $I_Y>0$
   for token F1 and $I_Y<0$ for answer NLL, with the paired 95% interval
   excluding zero for each interaction.

The margins are decision thresholds, not uncertainty intervals. No threshold,
row subset, gap size, scorer, or answer mask may be changed after viewing
results.

## 5. Randomization, seeds, and receipts

There is no training randomness: the checkpoint and weights are frozen, and
decoding is greedy. Before any model output is generated:

- generate one common row permutation with RNG seed `20260827` and apply it to
  all four cells;
- use bootstrap RNG seed `20260828` for the analysis receipt;
- record the framework/model-code revision, tokenizer revision, checkpoint hash,
  Native and candidate float32 table hashes, candidate construction receipt,
  token-ID and answer-mask hashes, row-manifest and row-order hashes, position
  map hashes, causal-mask configuration, precision, SDPA backend, decoder/EOS
  configuration, and output schema version.

Each cell must write per-row JSONL before aggregation. The sanitized result
owner must preserve the raw per-row files outside Git and commit only the
reviewer-safe receipt required by the evidence route. A plan, launch log,
checkpoint inventory, or aggregate without per-row provenance is not a result.

The future runner should expose a no-GPU contract mode and fail closed on:

- token, answer, row-order, or prompt-text drift across cells;
- a non-identical causal mask or decoder configuration;
- candidate endpoint/span mismatch or non-monotone frequencies;
- duplicate/decreasing/out-of-range virtual positions;
- attention-backend fallback, non-finite logits/NLL, or missing raw rows;
- any amplitude, LoRA, adapter, session-route, or target-length input.

The current repository has reusable building blocks in
`scripts/analysis/rope_transport/same_support_controls.py`,
`scripts/data_prep/target_free_context_builder.py`, and the mature OLMo
evaluation owners, but no claim should be made that this 2x2 runner exists or
that this protocol has been executed until a separate implementation and CPU
contract review are complete.

## 6. Stopping rules

Stop before GPU execution if the candidate cannot meet the pure-interior table
contract, the row manifest lacks stable evidence/query/answer boundaries, or
the maximum virtual position is unsupported.

Stop the run immediately and invalidate it if the first real inference batch is
non-finite, the backend falls back, the cell manifests diverge, or any receipt
is missing. Do not repair an invalid run by editing rows or rerunning only one
cell.

After a valid full 2x2, stop. Do not tune the candidate, gap, gain, route,
adapter, task subset, or decoder after inspecting the result. The verdict is
based on the preregistered gates above.

## 7. Negative-result interpretation and post-result gates

| Observed result | Interpretation and next action |
| --- | --- |
| N-G does not degrade Native | No tested Native position failure was identified; stop this bridge and do not attribute later gains to phase recovery. |
| N-G degrades, C-G does not recover, and C-C is non-inferior | The tested allocation does not repair this phase failure; do not call it a model ceiling or launch another static-table search. Revisit the position-map hypothesis only with a new preflight. |
| N-G and C-G both degrade similarly | The gap effect is not rescued by the candidate; retain it as position sensitivity without candidate-specific allocation attribution. |
| C-C violates the in-window non-inferiority gate | The candidate does not meet the joint objective; do not promote it as a target-free mature retrofit. |
| Task score passes but answer NLL does not, or vice versa | Keep the endpoint-specific result; the bridge is partial, not jointly established. |
| Both primary endpoints pass all gates | Position coding is a causal part of the remaining headroom under this contract. Stop and write the owner first. Only then may a separately preregistered leave-one-band-out mechanism design be considered; grouped allocation or adaptation remains later and independently gated. |

A positive bridge does not prove that the derived profile is optimal, that all
checkpoints share the effect, or that amplitude, adaptation, or routing are
irrelevant. A negative bridge does not invalidate the completed fixed-support
training identification or frozen mature pure-$z$ owners.

## 8. Provenance and decision links

- Current protocol boundary and decision order:
  [`INDEX.md`](../../../../INDEX.md) §6.2–§6.3
- Missing identification bridge and decision readings:
  [`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](../../ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
  §3–§5
- Pure allocation and mature-checkpoint claim ceilings:
  [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md)
- Training-stage fixed-support identification:
  [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](../../EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
- Causal-variable definitions for support, allocation, gain, adaptation, and
  routing:
  [`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md)

This document records a protocol only. It contains no experiment result,
launch receipt, or authorization to use GPU, remote, or paid compute.
