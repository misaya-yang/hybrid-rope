# Mature-checkpoint retrofit research index

This directory is the durable internal layer for attention-aware allocation and
frozen-checkpoint retrofit. It is not manuscript prose. Start here instead of
opening the newest dated file.

The cross-project conceptual grammar is
[`../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md).
It distinguishes the realised frequency tensor from the causal variables used
to analyse it, and prevents training-time allocation, frozen-table effects,
attention gain, and serving policy from being merged into one estimand.

## Current decision

Two owners are live:

1. [`results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md)
   owns the causal paper-upgrade case study. At fixed support and amplitude,
   interior exponent allocation `z` changes frozen OLMo and Qwen behaviour; a
   nearest movement-profile ramp matches the detailed derived profile, so this is not
   a new interpolation-family claim. Its 151.9M two-seed crossing supports
   weights/table co-adaptation.
2. [`results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md)
   owns the practical zero-training policy: exact Native inside the model's own
   window, one deployment-frozen long profile beyond it, fixed before prefill
   for the entire KV-cache lifetime.

The second owner is the complete practical intervention: deterministic long
frequency tensor, fixed long attention amplitude, and session route. The first
owner and the frequency-by-gain 2x2 answer its component questions under
separate matched protocols; they are not one pooled factorial. Do not describe
the complete policy as a pure `z` experiment, and do not describe the controls
as separate modules that must be deployed in sequence.

The same-support result is routed as a compact frozen-checkpoint corollary in
the body and Appendix F.2; the complete session policy owns the separate Qasper
appendix endpoint. No GPU experiment is queued, and another table, gain, rank,
or RULER sweep is explicitly stopped.

The later direct-`z` calibration pilot is a completed negative method gate, not
a third live owner. It improved mean held-out 2x tail NLL but violated its
per-row robustness gate; see
[`results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md).
Its PG-19, RULER, LoRA, and full-task queue was not run.
This branch-local stop does not supersede or block the completed zero-training
Native/s4 policy above.

A subsequent new-shard natural-NLL study separates the practical policy from
its mechanisms; see
[`results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md).
The bundled Native/s4 policy persists on fresh FineWeb-Edu rows, while matched
fixed-support controls attribute the 4x effect to interior allocation, show
that a nearest ramp matches the derived profile, and identify exact Native
session routing as the in-window retention mechanism. The failed analytic
single-static-table gate is recorded separately and does not replace the
session-policy owner.

## Read order

1. The conceptual foundation linked above.
2. The two live owners above.
3. [`evidence/README.md`](evidence/README.md) for machine-path-free receipts.
4. [`results/README.md`](results/README.md) for completed results and their
   claim ceilings.
5. [`analysis/README.md`](analysis/README.md) for mechanism analyses and
   falsified design axes.
6. [`preflights/README.md`](preflights/README.md) only when reconstructing a
   protocol or checking what was registered before execution.
7. [`theory/README.md`](theory/README.md) for the historical method agenda.

## Directory contract

| Directory | Contains | Evidentiary status |
| --- | --- | --- |
| `results/` | completed experiment owners and decision reports | usable only within each owner's claim ceiling |
| `evidence/` | compact JSON receipts and hashes | navigation receipts; raw outputs remain external |
| `analysis/` | CPU/mechanism analyses and falsifications | internal support or negative evidence |
| `preflights/` | preregistrations, revoked plans, and launch contracts | never a result |
| `theory/` | dated agendas and exploratory derivations | historical reasoning, superseded by live owners where stated |

## Reviewer-facing causal ladder

| Question | Primary owner |
| --- | --- |
| Does interior allocation matter during training at fixed support? | `../EXACT_RANGE_151M_3SEED_RESULT_20260820.md` |
| Do weights learn the installed coordinate system? | `../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` plus the 151.9M crossing in the live same-support owner |
| Does `z` still matter in a frozen mature checkpoint? | live same-support owner |
| Is the detailed uniqueness curve required? | live same-support owner; current answer is no |
| Does the frequency table interact with attention amplitude? | `results/JOINT_MECHANISM_REPORT_20260822.md`; current answer is yes in the tested cells |
| Can deployment preserve the Native short path without training? | live session-binary owner |

Keep these estimands separate. The complete replacement may combine variables
that its controls isolate. RULER is task-family adaptation, row bootstraps
condition on one checkpoint/task set, and a frozen-checkpoint intervention is
not a from-training estimate.

## Code routes

- deterministic same-support tables:
  `scripts/analysis/rope_transport/same_support_controls.py`;
- mature RULER evaluator: `scripts/eval/target_free_ruler_smoke.py`;
- 151.9M crossing evaluator:
  `scripts/eval/evaluate_151m_same_support_retrofit.py`;
- session policy and cache-safe RoPE: `scripts/lib/rope/`;
- focused regression tests: `tests/test_same_support_rope_controls.py`,
  `tests/test_length_conditioned_budgeted_rope.py`,
  `tests/test_target_free_rope.py`, and
  `tests/test_target_free_context_builder.py`.

External-model reviews are indexed under
[`../external-reviews/README.md`](../external-reviews/README.md). They may
identify defects or suggest controls, but never supersede these owners.
