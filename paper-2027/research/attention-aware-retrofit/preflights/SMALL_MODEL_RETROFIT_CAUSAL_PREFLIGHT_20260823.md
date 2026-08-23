# 151.9M weights-by-runtime-table retrofit audit

- **Registered:** 2026-08-23 before GPU execution
- **Status:** planned inference-only evaluation
- **Purpose:** extend the existing weights-by-table co-adaptation diagnosis to
  the raw-hash-receipted 151.9M exact-range checkpoints; supporting mechanism
  only, not a new paper pillar

## Question

Two training seeds each contain weights trained with the same sampled support
but different interiors: FMRoPE and anchored EVQ-Cosh.  At fixed evaluation
rows, install both training tables and both derived long tables across both
weight sets.  This separates runtime-table effects from weights-table
co-adaptation more directly than evaluating each checkpoint only with its own
table.

This does not replace the existing three-seed training-time estimate or the
50M 2x2 crossing.  Its value is a cheap scale/recipe replication and a bridge
to the new frozen-checkpoint controls.

## Frozen protocol

- Seeds: `137`, `256`; checkpoint pairs share the registered initialization,
  row order, optimizer, token budget, validation and anchor identities within
  seed.
- Checkpoints:
  - seed 137 FMRoPE `ae065557...b1cf9d`, anchored Cosh `b9922460...405669`;
  - seed 256 FMRoPE `841724fa...3e802a`, anchored Cosh `1387f28f...f72c09`.
- Validation manifest SHA-256:
  `2c4b1c0ec6993a4065a666dd04c26f1c3439e812de25e49cbf5d21b106ab9433`.
- 32 paired FineWeb-Edu validation anchors; lengths `512/1024`; primary
  endpoint is final-128-token NLL.
- Runtime tables:
  - FMRoPE training table and anchored-Cosh training table, each with amplitude 1;
  - the same two tables with only the factor-four amplitude;
  - factor-four uniqueness and nearest-ramp tables derived separately from
    each training table;
  - one geometric factor-four table shared by both because their endpoints are
    identical.
- No task labels, passkey examples, or evaluation losses select a table.
- Evaluator SHA-256:
  `64e510d2ae663363bf1e51b45f6e8fbeb33ab8f440c7a888dc382849eb639953`.

Passkey is intentionally excluded: these exact-range checkpoints were trained
on FineWeb-Edu without a passkey-supervision contract.  A passkey NLL gap would
be a weak probe and AR exact is expected to floor; it cannot support capability.

## Readout

For each length, report the complete `weights x runtime-table` paired NLL
matrix by training seed.  The co-adaptation contrast is the difference of
table-swap effects between FMRoPE-trained and EVQ-trained weights.  Anchors are
paired observations inside a seed; the training seed remains the replication
unit.

## Stop and claim boundary

- If both weight sets rank runtime tables identically with negligible
  interaction, this adds no new co-adaptation evidence; record it and stop.
- If cross-assignment penalties reverse with the trained table, it replicates
  weights-table co-adaptation at 151.9M but does not prove the new uniqueness
  profile optimal.
- If the derived tables help both substrates, report a model-relative retrofit
  direction; do not merge it with the mature-checkpoint RULER estimand.
