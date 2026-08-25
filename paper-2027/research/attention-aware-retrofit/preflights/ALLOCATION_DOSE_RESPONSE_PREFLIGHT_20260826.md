# Fixed-support allocation dose response on a mature checkpoint

- **Date:** 2026-08-26
- **Status:** preregistration; code complete, CPU build verified, **no GPU run**
- **Role:** submission experiment for reviewer objections R2 and R4, plus one
  method-development test (E3)
- **Code:** `scripts/analysis/allocation_dose_grid.py`,
  `scripts/eval/eval_allocation_dose_grid.py`,
  `scripts/analysis/compare_learned_allocation_directions.py`,
  `scripts/eval/run_allocation_dose_grid_5090.sh`

## Why this is not closed route #12

`INDEX.md` §3.4 closes eleven *selectors*: static scalar functionals proposed to
pick a table. This study proposes no selector. The dose grid is fixed by
construction between two objects the manuscript already owns, the static
quantities are computed by the tracked owner
`scripts/analysis/third_axis_ceiling.py`, and they are written into
`prediction.json` **before** any language-model number is read. The behavioural
sweep can therefore falsify the theory rather than be fitted by it. A selector
claims a table is good; this stakes the theory on a curve that an independent
measurement can take away.

## What is missing today

The manuscript owns `z` at the two ends of the axis and nothing between them.
Anchored EVQ-Cosh at Native support has been measured only at full strength,
where it is catastrophic in-window (`+3.9780` 1x tail NLL,
`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`). Native is the other end. The
2026-08-25 co-adaptive oracle landed at `max|dz| = 0.00130`, which is `1/321`
of the anchored-Cosh displacement. No point in between has ever been evaluated,
so no owner can say whether the in-window cost rises before, with, or after the
long-range gain. R2 is open for exactly this reason.

## E1 — dose response (submission)

**Intervention.** Frozen released OLMo-2-0425-1B-Instruct. Weights bitwise
unchanged, attention amplitude `1.0`, no routing, no learned parameter, no
target length. Only the shared rotary buffer changes. Both sampled endpoints
stay bitwise Native in float32, so every arm is a pure interior-`z` move.

- Path A (theory): `z = (1-lambda) z_geo + lambda z_cosh(tau=4)`, `lambda` in
  `{0, .02, .05, .1, .2, .35, .5, .7, 1}`; `tau=2` at `{.1, .35, .7}` as a
  shape control.
- Path B (empirical): gap logits scaled by `lambda` along the direction the
  oracle learned, `lambda` in `{1, 4, 16, 64}`; `lambda=1` reproduces that
  table bitwise.

**Data and endpoints.** The frozen 128-document fresh-FineWeb long holdout at
`1x/2x/4x`, which is the same protocol and the same rows as the session-s4
owner, so results land beside its published Native row `2.7538 / 7.0023 /
7.2703`. Endpoints per document: declared final-1024-token tail NLL, dense
whole-sequence NLL, and per-1024-position-bin NLL. Per-row values are written
for a 20,000-resample paired document bootstrap against the bitwise-Native arm.

**Registered predictions** (frozen in `prediction.json` by the build stage):

- P1 `r2` under the causal measure increases monotonically in `lambda`.
- P2 in-window 4K NLL increases monotonically in `lambda`.
- P3 16K tail NLL has an interior minimum at some `lambda > 0`.
- P4 that minimising `lambda` lies closer to `argmax r2` than to `0`.

**Decision rule.** A dose effect is established when some `lambda > 0` improves
16K tail NLL with a paired bootstrap interval excluding zero, while dense 4K
NLL degrades by at most `+0.01` with its interval excluding `+0.05`.

**Stop conditions.**

1. If 16K tail NLL is monotone non-decreasing across all of Path A, the
   zero-training fixed-support route buys no target-free long-range gain on
   this checkpoint. Report the negative and stop; do not sweep `tau`, bands, or
   amplitude.
2. If the effect of `learnedB_lam1` does not grow through `lam4` and `lam16`,
   the oracle table's long-range effect is not a dose effect and must not carry
   a mechanism claim.
3. The screen stage (32 documents) runs first. If no arm reaches half the
   registered tail effect at 32 documents, the full stage is not launched.

## E2 — capability confirmation (submission)

Only after E1 selects an operating point, and only for the Native arm, that
point, and its two neighbours. Uses the existing harnesses unchanged:
`target_free_ruler_smoke.py` (core-4 RULER at 8K/16K, 20 rows per task) and
`target_free_formal_eval.py` (LongBench QA at `1x/2x/4x`, which supplies the
in-window capability guard that RULER's `{2L,4L}` restriction cannot).

NLL is not capability. A dose effect that does not survive here is reported as
an NLL-only effect.

## E3 — is the learned direction reproducible? (method development)

**Hypothesis it can falsify.** That the oracle's learned table is a directed
solution rather than drift.

The observed cosine between the learned displacement and the anchored-Cosh
direction is `+0.8541` (`tau=2`: `+0.9328`; interior sign agreement `54/62`).
That looked decisive and is not. Because `z` is a cumulative sum of softmax
gaps with both endpoints pinned, undirected gap-logit noise produces smooth
single-humped displacements that are already strongly correlated with any fixed
smooth reference. The simulated pinned-walk null in
`compare_learned_allocation_directions.py` gives cosine-against-EVQ p99
`0.9647` and pairwise p99 `0.8883` at `K=64`. **The single-run alignment is
inside that null and proves nothing in either direction.**

The test is therefore two or three repeats of the completed dense-natural
recovery protocol differing only in data order. Reproducible direction requires
every pairwise cosine to exceed the null pairwise p99; theory alignment
requires every EVQ cosine to exceed the null reference p99; cosines inside the
null establish drift. All three outcomes are informative and the run is short
(the oracle trained in 433 s).

## Compute and receipts

Single RTX 5090, BF16, flash-only SDPA with the math and memory-efficient
backends explicitly disabled. Build stage is CPU and needs no authorization.
Every GPU stage requires `--authorize` **and**
`ALLOCATION_DOSE_GRID_GPU_AUTHORIZED=YES`. Estimated GPU time: screen ~35 min,
full ~2.5 h, RULER ~40 min for four tables.

Receipts: table float32 and file hashes, manifest hash, evaluation-row file
hash, script hashes, environment/SDPA receipt, peak memory, runtime, per-row
JSONL, and the prediction file frozen before the first language-model number.

## Claim ceiling

Frozen zero-training interventions on one released checkpoint under one
document set. Row bootstraps condition on that checkpoint and those documents;
they are not checkpoint-population or training-seed uncertainty. Nothing here
changes the three-seed fixed-support identification owner, and a positive
result is mature-checkpoint dose evidence, not a new method claim.
