# K32 independent crossing confirmation (2026-09-01)

## Decision

**Status: `COMPLETE / CROSSING_UNRESOLVED / INDEX_FAVORED_VS_YARN /
P3_ENTRANCE_FAILED`.**

The independent 80-row-per-task confirmation does **not** reproduce the old
physical-x long advantage over normalized-index.  At 64K the difference is
only `+.0050`, with the preregistered two-length-corrected interval spanning
zero.  Both fixed profiles improve 64K macro by about `.22` over Native, while
normalized-index is materially more Native-compatible at 32K.

A separately preregistered official-equation YaRN-s2 completion arm on the
same rows scores exactly the same 32K macro as normalized-index, but is lower
at 64K by `.064375`; the paired 95% interval is wholly positive for index.
This makes normalized-index the current engineering representative for a new
untouched breadth confirmation. It is not yet a SOTA or universal-law result.

This weakens the strongest physical-coordinate claim.  It does not erase the
useful frozen-profile result: both static tables recover substantial 64K
capability.  The registered Native-Q/K mechanism branch required a confirmed
physical/index crossing, so it is not opened.

Owners:

- [Preregistration](../../preflights/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_PREFLIGHT_20260901.md)
- [Hash-bound receipt](../../evidence/K32_PAIRED_CROSSING_CONFIRMATION_RECEIPT_20260901.json)
- [Reproducer](../../../../../scripts/analysis/summarize_k32_crossing_confirmation.py)
- [Matched YaRN completion receipt](../../evidence/K32_FRESH_YARN_COMPLETION_RECEIPT_20260901.json)

## Frozen protocol

The exact Qwen2.5-0.5B-Instruct K32 checkpoint, Native tensor, frozen
physical-x and normalized-index s2 tensors, gains, greedy decoder, scorer, and
request-wide static-table lifetime are unchanged from the completed P2 panel.
The only new information is seed `202609026` with 80 fresh rows per task at
32K and 64K.  The four tasks are single-key, numeric multi-key, UUID multi-key,
and variable tracking.  No old pilot rows are pooled.

Each arm completes 640 generations, for 1,920 terminal rows total.  Raw result,
example, run-manifest, checkpoint, tokenizer, input-cell, Native tensor, active
tensor, table-file, factor, and attention-scaling identities are verified by
the receipt.  Complete generated token IDs, decoded predictions, references,
generation budgets, EOS metadata, and official task scores are retained.

## Complete results

Task-vector order is single / mk2 / mk3 / VT.

| Profile | 32K task vector | 32K macro | 64K task vector | 64K macro | 32K Native retention |
| --- | --- | ---: | --- | ---: | ---: |
| Native | `1/.7375/.1125/.4625` | `.578125` | `.80/.0625/0/.12` | `.245625` | `1.000000` |
| physical-x | `1/.3875/.075/.525` | `.496875` | `1/.5125/.10/.2525` | `.466250` | `.859459` |
| normalized-index | `1/.5625/.075/.4975` | `.533750` | `1/.4875/.0625/.295` | `.461250` | `.923243` |
| official YaRN-s2 | `1/.525/.125/.485` | `.533750` | `1/.2625/.0375/.2875` | `.396875` | `.923243` |

The `.875` Native point gate fails physical-x and passes normalized-index and
YaRN. This is a RULER capability ratio, not a Qwen natural-PPL gate.

## Paired inference

The primary estimand is physical-x minus normalized-index at both registered
lengths.  Ten thousand task-stratified paired bootstrap replicates use seed
`202609027`; Bonferroni over the two prespecified lengths gives 97.5% marginal
intervals.

| Contrast | 32K delta [registered interval] | 64K delta [registered interval] |
| --- | --- | --- |
| physical-x − normalized-index | `-.036875 [-.077500,.002508]` | `+.005000 [-.038125,.048750]` |

Neither endpoint identifies an ordering.  In particular, the historical 64K
point difference `+.0675` is not replicated.

The prespecified descriptive comparisons against Native are:

| Contrast | 32K delta [95% CI] | 64K delta [95% CI] |
| --- | --- | --- |
| physical-x − Native | `-.081250 [-.125625,-.038125]` | `+.220625 [.174375,.267500]` |
| normalized-index − Native | `-.044375 [-.088125,-.002500]` | `+.215625 [.171875,.260016]` |

Thus the robust update is not a physical/index long ranking.  It is that the
two profiles produce essentially the same long improvement on this panel,
while normalized-index pays the smaller measured Native cost.

## Matched official-YaRN completion

The YaRN arm was frozen after the three candidate arms completed, so this is a
same-row baseline completion rather than an untouched method-selection
holdout. It uses the installed-HF official-equation factor-two tensor and
published amplitude, with no search. The original physical/index crossing
decision remains unchanged.

| Contrast | 32K delta [95% CI] | 64K delta [95% CI] |
| --- | --- | --- |
| normalized-index − YaRN | `.000000 [-.042500,.041875]` | `+.064375 [.027500,.102516]` |

The 64K interval is wholly above zero under the preregistered paired
task-stratified bootstrap. The gain is concentrated in numeric multi-key
retrieval (`+.225`); single-key is tied, UUID multi-key is `+.025`, and VT is
`+.0075`. Therefore this panel supports a bounded index-over-YaRN core-4 result,
not uniform per-task dominance.

## Consequence for the research sequence

The conditional
[Native-Q/K finite-KL protocol](../../preflights/coupling-transfer/NATIVE_QK_FINITE_KL_PREFLIGHT_20260901.md)
required `CONFIRMED_CROSSING`.  The observed decision is `UNRESOLVED`, so no
model forward, KL result, compatibility predictor, selector, or P4 method is
authorized by that preregistration.  Its already-prepared CPU input receipt is
preserved as unused infrastructure rather than relabeled evidence.

The later independent K128 16K owner repeats the index advantage, and the
separately frozen new-seed full-RULER owner confirms normalized-index against
Native and YaRN without reusing these development rows. Those outcomes have
their own canonical reports; broad parameter search remains outside this
owner.

## Claim ceiling

This is one checkpoint, one new data seed, four fixed RULER tasks and one
decoder/scorer contract. It supports useful K32 static s2 extension, rejects a
strong claim from the old physical/index point difference, and identifies a
same-row index advantage over YaRN at 64K. It does not prove index universality,
K causality, natural-text quality, arbitrary-scale consistency, or SOTA.
