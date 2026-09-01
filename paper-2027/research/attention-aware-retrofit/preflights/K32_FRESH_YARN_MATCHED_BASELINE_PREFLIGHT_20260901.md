# K32 fresh YaRN matched-baseline completion — preregistration

**Status:** `FROZEN_FOR_EXECUTION`

## Question and hypothesis

The independently seeded K32 confirmation has completed for Native, frozen
physical-x, and frozen normalized-index.  It did not reproduce a physical-x
long-context advantage: at 64K the physical-minus-index macro difference was
`+0.0050` with an interval spanning zero, while normalized-index retained more
Native capability.  The next bounded question is whether that fixed
normalized-index profile is competitive with the already-exported
deterministic YaRN factor-two baseline on exactly the same rows.

This is a matched baseline completion, not a new profile search and not an
independent method-selection holdout.

## Confound and claim ceiling

Outcomes for the first three arms on these rows are already known.  Therefore
the added YaRN arm can support a paired same-row comparison, but the resulting
four-arm panel cannot be called an untouched final-method holdout.  The core-4
suite is also not full RULER, natural QA, a K-causal intervention, or a SOTA
claim.

## Frozen construction

- checkpoint: the same hash-bound Qwen2.5-0.5B-Instruct K32 checkpoint used by
  the completed confirmation;
- data: seed `202609026`, 80 rows per task and length, core-4 tasks, lengths
  32K and 64K;
- comparator: installed-Hugging-Face official-equation YaRN static tensor,
  factor `2`, `beta_fast=32`, `beta_slow=1`, original/reference length 32768;
- attention amplitude: `1 + 0.1 log(2) = 1.0693147180559945`;
- expected table tensor SHA-256:
  `d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59`;
- no change to the frozen coupling tables, no gain or boundary search, and no
  additional candidate.

## Entrance gate and stop condition

Execution is allowed only because the three-arm confirmation is terminal and
the same YaRN construction previously resolved non-zero 64K capability.  Run
exactly one YaRN arm over both lengths and stop.  Any identity mismatch,
non-finite output, or near-zero positive-control collapse makes the comparison
invalid; it does not authorize tuning.

## Decision rule

The primary contrast is normalized-index minus YaRN at 64K, paired by the same
320 task rows and bootstrapped at the row level within task.  A 95% interval
entirely above zero favors normalized-index on this core-4 endpoint; entirely
below zero favors YaRN; an interval containing zero is unresolved.  Report the
32K contrast and Native retention separately.  Do not combine lengths into a
selector or promote a winner from this development-completion panel.
