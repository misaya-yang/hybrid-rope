# `paper-2027/research/` — internal research layer

This directory stores theory, canonical result owners, mature-checkpoint
research, audits, historical ledgers, and frozen plans. It does not own a
second agenda or handoff.

## Cold start

1. [`../../AGENTS.md`](../../AGENTS.md) — rules and claim ceilings.
2. [`../../INDEX.md`](../../INDEX.md) — current theory/evidence routing, closed
   routes, directory ownership, and durable agenda.
3. [`../HANDOFF.md`](../HANDOFF.md) — live Git/PDF/machine state and immediate
   actions.
4. [`history/TIMELINE.md`](history/TIMELINE.md) — chronological background only
   when needed.

## Folder contract

| Folder | Content | Authority boundary |
| --- | --- | --- |
| [`foundations/`](foundations/) | durable full-RoPE, causal-variable, construction, and historical theory documents | theory/notation; facts still defer owners |
| [`evidence/`](evidence/) | compact paper-level result owners outside mature retrofit | numerical owner at stated scope |
| [`attention-aware-retrofit/`](attention-aware-retrofit/) | mature-checkpoint results, receipts, analyses, preflights, and theory | subfolder roles are strict |
| [`audits/`](audits/) | theory/evidence/manuscript/compliance audits | cannot upgrade a claim |
| [`history/`](history/) | chronological ledgers | non-authoritative summary |
| [`archive/`](archive/) | retired plans, simulated reviews, process logs | frozen input; never a queue |
| [`external-reviews/`](external-reviews/) | untrusted external-model review snapshots | historical analysis only |
| [`three_completions/`](three_completions/) | historical optimization/proof workspace | bounded theory archive |

## Paper-level canonical owners

| Question | Owner | Maximum role |
| --- | --- | --- |
| Does interior allocation matter at fixed support during training? | [`evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) + [JSON](evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json) | primary causal identification; retain seed/support scope |
| What does full sin/cos geometry establish? | [`foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | static redundancy/effective dimension and co-adaptation diagnosis; not LM ranking |
| How are support and allocation separated? | [`foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) | notation/intervention grammar, not a number owner |
| What is the bounded EVQ-Cosh claim architecture? | [`foundations/ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](foundations/ICLR2027_RESEARCH_SYNTHESIS_20260819.md) | historical implementation/claim synthesis; current TeX owns wording |
| What is the exact frozen transplant boundary? | [`OLMO2_POSTHOC...`](../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) | exact position-independent invertible compensation only |
| What cross-modal evidence exists? | [`evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md`](evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) | one matched seed-42 supporting comparison |

Mature-checkpoint questions route through
[`attention-aware-retrofit/README.md`](attention-aware-retrofit/README.md), but
the definitive current map remains `INDEX.md` §2.3.

## Evidence discipline

- Pure-`z` attribution holds support/base, operator, checkpoint/training
  contract, gain, routing, data/rows, decoder, and endpoint fixed.
- NLL/PPL, teacher-forced retrieval, strict generation, RULER/NIAH, QA,
  causal source use, adaptation, and transfer are separate tiers.
- Training seeds, row bootstraps, tasks, configurations, and single matched
  trajectories are different uncertainty units.
- A preflight records what was frozen before execution; it is not a result.
- External reviews, archives, filenames, and compact receipts never supersede
  a canonical/raw owner.

## Placement and stop rules

New paper-level foundations go in `foundations/`; compact paper-level result
owners go in `evidence/`; mature retrofit work goes in the matching
`attention-aware-retrofit/` subfolder. Retired plans/reviews go in a dated
`archive/` folder. Any new owner must be routed in `INDEX.md` in the same
change.

No future compute is queued by this directory. Current authorization and next
actions live only in [`../HANDOFF.md`](../HANDOFF.md).
