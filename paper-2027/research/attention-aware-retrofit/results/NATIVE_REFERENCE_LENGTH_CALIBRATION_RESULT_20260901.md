# Native reference-length identification (2026-09-01)

## Passport and current decision

- **Status:** `P0_V1_ABSTAIN / SINGLE_CODE_REFERENCE_INDEPENDENTLY_CONFIRMED`
- **Intervention:** none. Exact Gemma-1.1 checkpoint, Native frequencies,
  original config length 8192, amplitude one, zero model updates.
- **Question:** can independent Native measurements identify an operating
  reference length before any new frozen-coupling holdout?
- **Protocol:**
  [`COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901`](../preflights/COUPLING_NEGATIVE_ATTRIBUTION_PREFLIGHT_20260901.md).
- **Prior transport evidence:**
  [`FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901`](FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md).

The first P0 calibration does **not** identify a joint `L_ref`. Fresh natural
continuation independently reproduces severe Native degradation at 8192,
but the new two-code capability instrument fails its own short-context
competence floor. Its exact formatting errors cannot be silently forgiven.
The v1 verdict remains abstention. The one registered single-code replacement
subsequently selected 4096 and passed independent confirmation (§5), without
reusing any final long holdout or changing `G`.

## 1. Training length is not the inferred quantity

The [official Gemma technical report](https://arxiv.org/html/2403.08295v4)
states a training context of 8192 in its architecture section. Thus the family
cannot simply be described as trained at 4K. The measured object here is the
Native operating range of the exact instruction artifact under specified
likelihood and generation contracts, not historical training exposure.

Checkpoint identity is `unsloth/gemma-1.1-2b-it`, revision
`619e546640669a280738627cc623f4bd74a7b069`, weight SHA-256
`584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933`.
This is the same public redistribution used in the prior owner; no unverified
claim of canonical Google tensor equivalence is added.

## 2. Completed implementation controls

The runtime is Transformers `5.15.1`, PyTorch `2.8.0+cu128`, with the resolved
Gemma activation `GELUTanh` and `GemmaTextScaledWordEmbedding`.

On six fixed prior canaries—two tasks at 4096/8192/16384—stock-HF SDPA plus
`generate` and the research Flash backend plus handwritten decoding have:

- exactly equal prefill and one forced cached-decode logit tensors;
- exactly equal complete generated token IDs;
- unchanged Native frequency tensor throughout.

These comparisons retain the original 8192 config value. They reproduce
Native success at the two 4K canaries and failure at the two 8K and two 16K
canaries; the old convenience runner's expanded config capacity is therefore
not necessary for that failure. They do not establish correctness of every
possible shared dependency. Cached-versus-uncached execution is a separate
numerical comparison, not claimed bitwise equal.

The direct phase audit uses int64 positions `0..8191` and the same runtime
Native tensor. Research TF32/high and IEEE-FP32 modes produce identical phase
hashes, zero adjacent-position aliases, and maximum FP64-reference phase
error `0.000244140625`. Cos/sin compared after the same BF16 output conversion
differ by at most `0.00390625`. No whole-model precision intervention or
quadratic long-context attention was introduced.

Raw control identities:

| Artifact | SHA-256 |
| --- | --- |
| parity results | `80f8a3f5093d2b7cf539b6d28a953b6c8b1e1dd480d171b672af9fee89a1e06a` |
| parity raw rows | `6c4bce0cbda04f5151c9e835af7d965f3bfeb158dd1d1e8f159bcd5a7a1cba23` |
| phase precision audit | `f70bf81689244f67595cf5bf1267d3cb59d338a9be73d4a9d019a7c99f7b71ef` |

Code owners are
[`audit_rope_runtime_parity.py`](../../../../scripts/eval/audit_rope_runtime_parity.py)
and [`audit_native_rope_precision.py`](../../../../scripts/analysis/audit_native_rope_precision.py).
These prior canaries never enter fresh P0 selection.

## 3. First Native-only calibration

The data manifest was frozen before model evaluation:
`f29569ea8576e4c527ad9cb9dbe24ef545c01cf4447271e1b172e774ceb8cee4`.
It contains 32 calibration and 64 untouched confirmation natural documents,
plus 64/128 independent capability blueprints. Each natural document has the
same final 256 target tokens at all four lengths. Capability keeps eight
records and two queried codes fixed while changing only filler length.

Natural input documents are newly materialized from a fixed source-row range.
Calibration/confirmation text hashes are disjoint. The supplied older
exclusion row files expose no `source_text_sha256` fields, so that exclusion
check contributes **zero** historical document hashes; it must not be
described as exhaustive historical deduplication. The synthetic capability
blueprints are independent of both those documents and prior/final RULER rows.

| Endpoint | 1024 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: |
| Native mean continuation NLL, 32 documents | 3.440426 | 3.319199 | 3.229847 | 11.427202 |
| paired NLL change from 1024 | 0 | -0.121228 | -0.210579 | +7.986776 |
| two-code exact + EOS, 64 blueprints | 47/64 | 42/64 | 21/64 | 0/64 |

The natural 8192-minus-1024 difference has paired-document bootstrap 95%
interval `[6.916190, 9.095847]`. This is a large endpoint degradation,
not a `.01--.03` ranking difference. Natural text's contiguous passing grid
ends at 4096 under the preregistered `.875` PPL-retention margin.

The capability compact control is only `45/64=.703125`; its 1024 control is
`47/64=.734375`. Both miss the `.75` point-estimate competence floor. Its
formal point frontier is 2048, but the failed instrument gate prevents using
that number as a reliable behavioral reference. The confidence bounds also
leave the 2048 retention unresolved.

The frozen CPU decision is therefore `ABSTAIN`, with
`BASELINE_COMPETENCE_FAILED`, `COMPACT_COMPETENCE_FAILED`, and
`INCOMPATIBLE_FAMILY_FRONTIERS`. No confirmation model outcomes or new long
profiles were opened from this decision.

Runtime: 448 rows, 1,477,755 input tokens, 4,137 generated tokens, 92.652
seconds, peak reserved memory 7,780,433,920 bytes. A live sample recorded
100% GPU utilization; this is not asserted as a run-wide average. Raw rows
SHA-256: `b30f423744032437bac0405f12f54a66692526bb888f2f07e2d8e8934e0aa231`.

## 4. Specific measurement failure and bounded repair

Inspection of the original failed raw continuations finds both requested
six-digit numbers in the correct order in 18 of 19 compact failures and 16
of 17 failures at 1024. Typical outputs omit the final period or use two
periods instead of the requested comma-space separator. One compact failure
also returns a key rather than the second code.

This is a **failure classification, not a replacement success metric**.
The exact scores above remain unchanged; ordered-number extraction must never
select a length or promote the old probe. Errors at 4096 also include genuine
wrong-code retrieval, so short formatting errors do not explain all length
degradation.

The only registered repair uses fresh single-code blueprints with seven
distractor records, unchanged Native model and length grid, complete-code
exact plus EOS, and the same independent-confirmation rules. It removes list
formatting but also reduces multi-target load; any accepted reference will be
explicitly specific to natural continuation plus single-code retrieval.
If the new compact/1024 instrument fails, no additional prompt repair is
registered. Existing natural calibration repeats are not new independent data.

## 5. Single-code calibration and independent confirmation

The replacement compact/1024 controls are each `64/64`; its calibration
1024/2048/4096/8192 exact counts are `64/64/63/0` out of 64. The natural and
capability families therefore independently propose 4096. This decision was
frozen before any confirmation model outcome.

On the separate 64-document, 128-blueprint confirmation split:

| Endpoint | 1024 | 4096 | 8192 |
| --- | ---: | ---: | ---: |
| mean natural suffix NLL | 3.427443 | 3.196320 | 11.158831 |
| single-code exact + EOS | 128/128 | 127/128 | 0/128 |

All four preregistered primary gates pass. The conservative 4096 capability
retention lower bound is `.945268`, above `.875`; the 8192 upper bound is
`.040446`. Natural 4096-minus-1024 NLL has paired-bootstrap 95% interval
`[-.282463,-.184386]`, while 8192-minus-1024 is `[7.088818,8.442138]`.
The next grid point fails both families as a secondary boundary check.

The compact machine-path-free confirmed reference is
[`GEMMA_NATIVE_REFERENCE_CONFIRMED_20260901.json`](../evidence/GEMMA_NATIVE_REFERENCE_CONFIRMED_20260901.json).
The confirmed **operating reference** is 4096 for these two measurement
families; this does not relabel training history or establish all-task Native
capacity. P1 may now compute `s=target/4096` with frozen law parameters.

A separate two-document FP32 attention reference also retains the 8K NLL
failure. Across the four document/length cells, the largest mean-NLL
disagreement with Flash is `.012370`, below the predeclared `.05` diagnostic
bound. This was explicit 128-query chunked reference computation, not a
silent full-matrix attention fallback.

## 6. K-identification guardrail for the next stage

The existing normalized-index control is not a fixed OLMo raw-index curve.
[`export_frozen_coupling_transport.py`](../../../../scripts/analysis/export_frozen_coupling_transport.py)
constructs counterfactual K64 samples at the **target checkpoint's** `b,L`,
then interpolates by normalized index. Consequently, when the target itself
has K64, physical and index movement are algebraically identical, apart from
runtime/formula rounding that must not be treated as a substantive contrast.

Thus Qwen K32/K64 is useful family triangulation, but it cannot establish a
trend in `physical-minus-index` versus K: the K64 zero contrast is imposed by
construction. Repeating both K64 arms would not buy a second coordinate test.
Nor may the index definition be changed after seeing the K32 results to make
that contrast non-degenerate. Future paired K32 replication can strengthen
the existing non-degenerate Pareto evidence, but must not be mislabeled K
causality or a measured multi-K trend. P2 remains conditional on P0 and its
own Native/YaRN resolver gates.

## Claim ceiling

The current data establish a large Native natural-text degradation between
the tested 4K and 8K endpoints on this exact checkpoint. They do not identify
`L_train=4K`, a task-independent scalar `L_ref`, causal K dependence, the
failure of frozen `G`, or a successful replacement table. V1 remains abstained;
the single-code replacement independently confirms a protocol-specific 4096
operating reference only.
