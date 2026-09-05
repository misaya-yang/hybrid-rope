# K32 normalized-index full-RULER confirmation — preregistration

**Status:** `EXECUTED / RESULT_OWNED`

The terminal result is owned by
[`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901`](../../results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md).

## Hypothesis

The independent 80-row core-4 development confirmation no longer supports a
physical-x long advantage.  Frozen normalized-index preserved 32K capability
better than physical-x and was statistically unresolved from it at 64K.  The
bounded confirmatory question is therefore whether this already-frozen
normalized-index table remains useful across the complete 13-task RULER suite
and is competitive with deterministic YaRN factor two.

This experiment does not refit the two coupling boundaries and does not add a
new profile.  It is a confirmation of an outcome-selected engineering
representative on newly generated rows, not evidence that normalized index is
the physical coordinate or that K has no effect.

## Frozen protocol

- checkpoint: Qwen2.5-0.5B-Instruct K32 with weight-set SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
- Native context: 32768; target: 65536; scale: exactly 2;
- RULER commit: `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`;
- unseen data seed: `202609027`;
- all 13 official synthetic tasks, lengths 32768 and 65536, 20 rows per
  task-length cell;
- arms, fixed before data generation:
  1. Native;
  2. normalized-index tensor SHA-256
     `8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f`,
     amplitude `1.0512928913614359`;
  3. official-equation YaRN factor-two tensor SHA-256
     `d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59`,
     amplitude `1.0693147180559945`.

No physical-x arm, additional baseline, scale sweep, gain sweep, boundary
change, selector, or task-specific choice is admitted.

## Confounds and controls

Normalized-index was selected after seeing the prior core-4 development panel;
the new seed is therefore the confirmation boundary.  Do not inspect or report
partial arm outcomes.  Complete all three arms on identical rows before
comparison.  This remains one checkpoint and one task family; it cannot prove
cross-checkpoint universality, K causality, natural-text likelihood, or SOTA.

## Entrance gate and execution order

The current 80-row official-YaRN completion must terminate with finite,
non-zero 64K capability and all identity checks passing.  The registered K128
coordinate confirmation must also finish and be attributed first; its outcome
cannot change this protocol's candidate, seed, tasks, or decision rule.  Only
after both conditions are recorded may the new seed be prepared and the three
arms run in the frozen order Native, normalized-index, YaRN.  An identity
failure or collapse of both extension arms makes the screen unresolved and
does not authorize tuning.

## Decision rule

Use official task scores, equal-weight macro across all 13 tasks, and a paired
task-stratified bootstrap that resamples rows within each task while preserving
all arm/length pairings.

1. Native compatibility gate: normalized-index 32K macro retention relative to
   Native must be at least `0.875`.
2. Extension resolver: the 95% interval for normalized-index minus Native at
   64K must lie entirely above zero.
3. Baseline contrast: if the 95% interval for normalized-index minus YaRN at
   64K lies above zero, record `CLEAR_ADVANCE`; if it contains zero after gates
   1–2 pass, record `COMPETITIVE_UNRESOLVED`; if it lies below zero or gate 1
   fails, record `BASELINE_LOSS`.

Report per-task rows and the 32K YaRN contrast regardless of category.  No
result authorizes a fourth arm or parameter search.
