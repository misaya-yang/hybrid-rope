# K128 coordinate-ranking confirmation (2026-09-01)

## Decision

**Status: `COMPLETE / INDEX_ADVANTAGE_REPLICATED /
PHYSICAL_COORDINATE_PRIVILEGE_REJECTED`.**

On an independent seed with 80 rows per task at 16K, frozen
normalized-index scores `.790000` and frozen physical-x scores `.728125`.
The paired index-minus-physical difference is `+.061875`, with 95% interval
`[.028109,.096250]`. The earlier K128 sample-level index tilt therefore
replicates without changing the reference length, scale, table parameters,
gain, decoder, scorer, or task set.

This is decisive against the strong claim that the current physical `x`
coordinate is the privileged cross-K long-context transport. It does not prove
that normalized index is a universal physical law. It selects the already
frozen normalized-index construction as the current engineering representative
for a new untouched breadth confirmation.

Owners:

- [Preregistration](../../preflights/coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_PREFLIGHT_20260901.md)
- [Hash-bound receipt](../../evidence/K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json)
- [Reproducer](../../../../../scripts/analysis/summarize_gemma_k128_coordinate_confirmation.py)

## Frozen protocol

The exact Gemma-1.1-2B-Instruct redistribution, independently confirmed 4K
operating reference, s4 target, physical/index tables, and amplitude
`1.102585782722872` are unchanged from the completed reference-corrected owner.
The new RULER seed is `202609028`; only 16K is evaluated, with 80 rows for each
of single-key, numeric multi-key, UUID multi-key, and variable tracking.

Native and YaRN are not rerun because the prior panel already establishes a
non-collapsed endpoint. The new experiment estimates only the paired coordinate
contrast. Old N20 rows are not pooled.

Each arm completes 320 terminal generations. The receipt verifies checkpoint,
Native/table tensors, table files, gain, tokenizer, data manifest, task-cell,
runner/runtime and raw artifact hashes. It also retains full decoded
predictions, generated token IDs and EOS metadata, and recomputes every official
row score from the complete prediction.

## Complete result

Task-vector order is single / mk2 / mk3 / VT.

| Frozen profile | Task vector | 16K macro |
| --- | --- | ---: |
| physical-x | `1/.8875/.20/.8250` | `.728125` |
| normalized-index | `1/.8875/.40/.8725` | `.790000` |

The task-level index-minus-physical deltas are `0/0/+.20/+.0475`. Thus the
aggregate advantage is not uniform across tasks; it is driven mainly by UUID
multi-key retrieval with a smaller VT contribution.

Ten thousand paired bootstrap replicates use seed `202609029`, resampling rows
within each fixed task and averaging the four task means equally:

```text
index - physical = +0.061875
95% CI            = [0.028109, 0.096250]
decision          = ABOVE_ZERO
```

No observation is dropped, no old row is pooled, and no optional stopping or
normal approximation is used. The interval is conditional on this checkpoint,
task set, data seed, decoder and scorer; it is not checkpoint or training-seed
uncertainty.

## Cross-K update

Taken only at the claim level supported by each owner:

1. **K32:** fresh N80 does not separate physical and index at 64K, but index
   preserves more 32K capability; index also exceeds matched official YaRN at
   64K on the same core-4 rows.
2. **K64:** the two transports collapse to the same construction, so K64 cannot
   rank coordinates.
3. **K128:** the independent N80 result resolves in favor of index.

The consistent engineering decision is therefore to stop treating physical
`x` as privileged and carry one frozen normalized-index table into the next
new-seed breadth confirmation. This is a simplification, not a new curve,
selector, residual, or K-dependent law.

## Claim ceiling

This owner identifies a relative ordering for two frozen tables on one K128
checkpoint and one 16K core-4 protocol. It does not establish K causality,
normalized-index universality, natural-text likelihood, continuous-range
guarantees, strict full-string/EOS task success, or SOTA. Those require the
separately frozen full-RULER and natural-text confirmations; no parameter search
is authorized by this positive result.
