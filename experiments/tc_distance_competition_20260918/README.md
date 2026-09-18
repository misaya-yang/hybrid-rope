# T-C distance x competition behavior experiment

This experiment tests a narrow prediction suggested by the completed equal-
displacement TailSpline (T) versus control C results:

> T's relative benefit should increase when the required binding distance is
> longer and competing key-value records are present.

It does not fit a new frequency table.  Both arms use the existing Llama-3-8B
S=4 TailSpline and exact equal-displacement C tables, with the same checkpoint,
gain, decoder and input in every paired cell.

## Stage 1: existing-output behavior audit

`audit_existing_multikey.py` reads every paired T/C multikey output from the
completed clean 16K and 32K panels.  It decodes the frozen prompt IDs, recovers
the full key-value map and classifies each response as correct, a value bound to
another key, other, ambiguous or empty.  All rows remain in the denominator.

The GPU stage is qualified only if prompt mapping coverage is at least 99%, the
32K aggregate has more confirmed wrong-binding repairs than damages, both
multikey-2 and multikey-3 are non-negative, and at least one is positive.  This
gate is fixed before reading the audit result.

### Stage 1 V2 result

The CPU-only recomputation retains all 750 paired rows and reaches 100% prompt
mapping coverage.  The V1 parser had excluded 33 valid rows merely because two
non-target keys sometimes shared the same randomly generated value; target
keys, target values and references were still unambiguous.  V2 keeps those rows
and records them explicitly in
[`reports/existing_multikey_audit_v2.json`](reports/existing_multikey_audit_v2.json).

At 16K, confirmed wrong-binding repairs minus damages are `+2` (`3-1`).  At
32K they are `+14` (`39-25`): multikey-1 is `-6`, multikey-2 is `+13`, and
multikey-3 is `+7`.  All preregistered Stage 1 gate conditions therefore pass.
The 32K net correctness advantage is 52 paired rows (81 T-unique-correct versus
29 C-unique-correct); only 14 of that net is directly attributable to confirmed
wrong-binding correction.  The remaining net transitions are primarily
`other -> correct` and `ambiguous -> correct`, so reduced wrong binding is a
real but partial behavioral account rather than the whole T-C gain.

The required official-score reconciliation is in
[`reports/stage1_official_reconciliation_v1.json`](reports/stage1_official_reconciliation_v1.json).
All 750 recorded official scores reproduce exactly.  At 32K, audit versus
official T-C differences are `+0.5pp` versus `-3.5pp` for multikey-1,
`+10.5pp` versus `+10.5pp` for multikey-2, and `+15.0pp` versus `+14.5pp`
for multikey-3.  The apparent nine-row net discrepancy is therefore fully
localized: eight net rows in multikey-1 and one in multikey-3.  The 23
individual scorer disagreements are retained as official substring matches
that the stricter behavior classifier calls `ambiguous` or `other`.

## Stage 2 V1 diagnostic

If qualified, use 64 base samples and cross two distance conditions with two
competition conditions for both T and C.  The primary statistic is the
difference-in-differences interaction

`[(T-C)_far,strong - (T-C)_near,strong] - [(T-C)_far,weak - (T-C)_near,weak]`.

The base sample, answer format and within-block order remain fixed across the
four conditions. Strong competition uses 512 fixed distractor records; the
weak cell replaces them with equal-length neutral filler. Competition content
therefore differs between weak and strong cells, but T/C always share identical
token IDs within a cell. A common
position-ID offset parity check must pass before model execution.

This is an end-to-end behavioral intervention, not a head-level mediation test.
Large raw rows remain on the experiment server; Git retains code, the frozen
contract and compact reports.

The already completed Stage 2 server run is diagnostic-only: it began before
the V1 audit had finished, did not enforce the failed V1 gate, and has no
position-ID offset parity receipt.  The V2 Stage 1 qualification does not
retroactively validate that run.  No replacement GPU execution is implied by
the CPU recomputation.

## Stage 2 confirmatory V2

The author subsequently approved one corrected confirmation run.  Its fresh
64-base-sample panel is frozen before model execution in
[`reports/confirmatory_preregistration_v2.json`](reports/confirmatory_preregistration_v2.json).
The confirmation seed is `2026091802`; its 256-row input-file receipt is
`22d28a1ee1d01ce70acead0de30d95a7147c839ccdb8742e7d08dd3fbfddda76`
and its row-identity receipt is
`d1d3b8608d99a39bfb569781d1ab8749aefe952d88b0a5a59352035ce0b8b81c`.
These samples are disjoint from both the V1 diagnostic panel and the separate
four-base-sample parity panel.

Near and far rows within each context cell now have exactly identical token
IDs and causal order.  A suffix-only position-ID gap changes the target-to-query
distance from exactly `8,463` to `30,595`; every position remains in `[0,32767]`.
The two context conditions are named honestly: 512 structured compact
key-value distractors versus token-count-matched neutral context.  T/C share
token and position IDs within every cell, so their table file is the only arm
difference.

Before confirmation, each table must independently pass an exact generated-token
parity check under a common `+17` position offset and a first-step full-logit
maximum absolute difference tolerance of `0.02`.  Runtime contracts retain each
row's input-token and position-ID hashes plus the installed table-file hash.
The first TailSpline parity attempt exposed BF16 absolute-phase-origin drift:
identical-position repeats were exact, while a common offset changed some
outputs.  The failed receipt is retained in
[`reports/parity_tailspline_failed_before_origin_canonicalization_v2.json`](reports/parity_tailspline_failed_before_origin_canonicalization_v2.json).
The runtime now subtracts the first prompt position from all prompt and cached
decode position IDs before RoPE.  This canonicalizes the mathematically
irrelevant global origin; it leaves the frozen confirmation rows unchanged
because they already start at zero and preserves every declared relative gap.
The primary interaction remains the strict behavior-classifier DiD; official
RULER substring score is secondary, wrong-binding transitions are secondary,
and all uncertainty resamples the 64 base samples rather than 512 generations.
