# K128 physical-vs-index coordinate confirmation — preregistration

**Status:** `FROZEN_PENDING_EXECUTION`

## Hypothesis and purpose

The old Gemma K128 core-4 panel favored normalized-index over physical-x by
`0.070` macro at 16K, but it used only 20 rows per task and its multiplicity
sensitivity interval crossed zero.  The fresh K32 N80 panel found essentially
no long difference between the same two frozen constructions.  A single new
K128 endpoint is therefore the highest-information test of whether the apparent
K128 coordinate ordering is reproducible.

This is not a table search.  Both tensors, the reference length, scale,
boundaries, and gain remain byte-for-byte frozen.

## Frozen protocol

- checkpoint: exact Gemma-1.1-2B-Instruct artifact with weight-set SHA-256
  `584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933`;
- operational reference length: 4096; endpoint: 16384; scale: exactly 4;
- new RULER seed: `202609028`;
- tasks: `niah_single_1`, `niah_multikey_2`, `niah_multikey_3`, `vt`;
- 80 new rows per task, only the 16384-token endpoint;
- physical-x tensor SHA-256
  `be5c2b3b4ce01d7fe6020cb01d9041e10aad93b9cdb03e989e64b8fa17561423`;
- normalized-index tensor SHA-256
  `1b908f90aebccc006521b3840b5662c217caa040d173c974527d33c7ea9e9849`;
- both use frozen attention amplitude `1.102585782722872`;
- identical static-table loading, tokenizer, prompts, greedy decoding, scorer,
  and row order.

No Native/YaRN rerun, other length, new profile, gain, boundary, scale, or
checkpoint is admitted.  Their existing positive controls establish that this
checkpoint and endpoint are resolvable; they are not needed to estimate the
coordinate contrast.

## Confound and entrance gate

The old N20 result motivated this confirmation, so it is development evidence
only and must not be pooled with the new rows.  Generate the new seed only
after this protocol is frozen.  Both arms must complete all 320 rows with
matching identities and finite task scores.  A shared collapse or identity
failure makes the test invalid and does not authorize tuning.

## Decision rule

Primary estimand: normalized-index minus physical-x 16K macro, using equal task
weights and a paired task-stratified row bootstrap with 10,000 replicates and a
95% interval.

- interval entirely above zero: reproduce an index advantage at K128 and reject
  the strong claim that physical-x is a privileged cross-K long coordinate;
- interval entirely below zero: reject the prior K128 index ordering;
- interval containing zero: coordinate ranking remains unresolved at both K32
  and K128; close the physical-vs-index superiority branch rather than adding
  samples, tasks, or parameters.

This experiment can identify only a K128 16K ordering for two frozen tables. It
cannot establish K causality, universal transport, natural-text quality, or
SOTA.
