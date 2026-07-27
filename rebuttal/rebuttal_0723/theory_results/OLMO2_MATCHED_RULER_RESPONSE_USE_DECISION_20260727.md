# OLMo-2 Matched RULER: Reviewer-Use Decision

Date: 2026-07-27
Audience: authors and rebuttal editors
Status: evidence verified; reviewer-facing selection requires deliberate use

## Decision

The matched OLMo-2 Native/EVQ result is scientifically valid, but the complete
three-length comparison is not automatically the strongest reviewer-facing
presentation.

- To answer whether physical-4K adaptation can produce real autoregressive
  capability at 8K/16K, use the EVQ absolute result:
  `37.51% / 21.29% / 6.13%` at 4K/8K/16K.
- To claim that EVQ changes length transfer relative to a matched Native
  control, use the full matched result and retain the 4K cost:
  Native/EVQ is `82.16%/37.51%` at 4K, `0.08%/21.29%` at 8K, and
  `0%/6.13%` at 16K.
- Do not present only the matched 8K/16K advantages while omitting the matched
  4K reversal. If the 4K tradeoff is too costly for the response objective,
  omit the matched comparison rather than truncate it.

This is a response-selection decision, not a change to the experimental
result.

## What was already saved

The supervised EVQ result predates the matched Native completion. Its
standalone owner is
`OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`.

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Legacy EVQ two-seed mean | 24.89% | 15.19% | 4.08% |
| Supervised EVQ continuation | **37.51%** | **21.29%** | **6.13%** |

Thus the 13-family physical-4K continuation improved the EVQ arm at all three
reported lengths. The 4K EVQ score was not lost, reconstructed, or introduced
by the later Native run.

The supervised EVQ arm is bound to:

- final adapter SHA-256
  `b18f6cfee8aad7d9004938beec0a075834cc05c69c6433aaed1943c7a603a653`;
- evaluation result SHA-256
  `9e96ba9134aad63ff77af42ae2678068af7658c290e1716ba73cb80773f9d17f`;
- 780-example prediction stream SHA-256
  `bf51a499ffdaeafeb03f7afa95c705dac4cf437a05485264fe65fe59ce3ce660`.

## What the later matched control added

The later run gave the Native parent the same fixed 736-row continuation,
physical-4K training limit, 276 optimizer steps, optimizer/LR semantics, seed,
and 13-task evaluation protocol. Its owner is
`OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md`.

| Arm | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Native-LoRA | **82.16%** | 0.08% | 0% |
| EVQ-LoRA | 37.51% | **21.29%** | **6.13%** |

Native also ends with lower continuation-validation NLL (`0.5020` versus
EVQ's `0.9881`). The large 4K score difference is therefore consistent with
the training receipt rather than an aggregation or transcription error.

The defensible interpretation is a tradeoff: Native learns the supervised
in-window distribution more strongly, whereas EVQ retains substantially more
capability at 2x and 4x length. This does not establish unseen-task transfer,
pure Cosh attribution, or universal EVQ superiority.

## Why the numbers entered the response drafts

The newly completed matched control was synchronized into the response package
because it directly strengthened attribution beyond the earlier EVQ-only
result. That synchronization correctly preserved the material 4K boundary,
but it treated evidence completeness as a reason for broad reviewer-facing
promotion. Those are separate decisions.

Current draft state:

- `paste/REVIEWER_zWsa.md`, `paste/REVIEWER_Dz6s.md`, and
  `paste/AC_PUBLIC.md` state the full 4K/8K/16K matched comparison.
- `paste/AC_CONFIDENTIAL.md` states the matched 8K/16K OLMo result but does not
  state the 4K `82.16%/37.51%` pair.

No response file is changed by this decision note.

## Recommended response routing

### Scale and stronger-evaluation question

Lead with the absolute EVQ capability result after physical-4K task-family
adaptation. It directly establishes measurable 8K/16K autoregressive RULER
capability without making the large in-window Native advantage the headline.

Recommended wording:

> With every backward pass capped at physical 4K, the 1.485B EVQ model reaches
> 37.51%, 21.29%, and 6.13% official macro across the complete 13-task RULER
> matrix at 4K, 8K, and 16K. Training and evaluation rows are disjoint, while
> generator families are shared; we therefore describe this as
> task-family-adapted length transfer rather than unseen-task transfer.

### Matched-attribution question

Use the matched table only when the response needs to establish that the
frequency substrate changes the length-transfer profile under the same
continuation. Keep the 4K result adjacent.

Recommended wording:

> Under the same physical-4K continuation, Native/EVQ official macro is
> 82.16%/37.51% at 4K, 0.08%/21.29% at 8K, and 0%/6.13% at 16K. Native learns
> the in-window task distribution more strongly, while EVQ preserves
> substantially more capability beyond the training length.

## Final guardrail

The supervised EVQ absolute result and the matched Native/EVQ attribution
result answer different questions. Do not automatically propagate every
completed matched control into every reviewer response. Select the narrowest
result that answers the actual concern; when the matched comparison is used,
retain its in-window reversal.
