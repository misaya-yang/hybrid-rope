# `scripts/` — implementation and tooling map

Code is not evidence and no script is an automatic action queue. Before using a
runner, resolve the current owner and authorization through [`../INDEX.md`](../INDEX.md)
and [`../paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md).

## Directory roles

| Path | Role | Status |
| --- | --- | --- |
| [`lib/rope/`](lib/rope/) | reusable frequency tables, injection, fixed-support and target-free primitives | implementation authority |
| [`analysis/`](analysis/) | CPU diagnostics, summarizers, geometry/table builders | reusable; output needs an owner |
| [`experiments/scale_transport/`](experiments/scale_transport/) | frozen-table evaluation, cache response and visibility-replay preparation | executed and unqualified paths are distinguished by their indexed protocols |
| [`experiments/cross_audit/`](experiments/cross_audit/) | input/contract validation, model evaluation, training and bounded job supervision | reuse the phase contract and existing receipts |
| [`data/`](data/) | current data builders and receipts | active utilities |
| [`data_prep/`](data_prep/) | NeurIPS/rebuttal-era data preparation | historical/supporting |
| [`eval/`](eval/) | mature-checkpoint evaluators and launch wrappers | use only with a live frozen protocol |
| [`core_text_phases/`](core_text_phases/) | February–March from-scratch phase chain | completed historical runners |
| [`text_eval/`](text_eval/) | continued-pretraining/text evaluation | historical/supporting |
| [`video_temporal/`](video_temporal/) | Video-DiT/temporal supporting code | historical/supporting |
| [`figures/`](figures/) | NeurIPS-era figures | historical manuscript assets |
| [`../paper-2027/figs/`](../paper-2027/figs/) | active ICLR figure generators | active manuscript assets |
| [`supporting_eval/`](supporting_eval/) | shared supporting metrics | endpoint identity must be confirmed |
| [`2026-04/`](2026-04/), [`2026-07/`](2026-07/) | dated launch/analysis helpers | archive; not current queue |
| [`mac_train/`](mac_train/), [`m4_max_36gb/`](m4_max_36gb/) | M4 historical training/diagnostics | archive |

## Current source-of-truth routes

| Need | Entry |
| --- | --- |
| construct or inspect RoPE tables | `scripts/lib/rope/` |
| reproduce static full-RoPE diagnostics | `scripts/analysis/full_rope_collision_audit.py` |
| reproduce finite-`K` surrogate audit | `scripts/analysis/finite_k_cosh_regret_audit.py` |
| summarize/reference current frozen transport | the specific `scripts/analysis/*` or `scripts/eval/*` file linked by the result owner |
| prepare current long-text inputs | `scripts/data/` |
| build the anonymous supplement | `scripts/package_supplement.py --profile iclr2027` from repository root |

The complete Figure/Table→code→data map is historical infrastructure under
[`../docs/overview/PAPER_CLAIMS_MAP.md`](../docs/overview/PAPER_CLAIMS_MAP.md).
Current ICLR claim routing is in `INDEX.md`, not in this README.

## Safety and maintenance

- Follow the phase authorization and budget recorded in HANDOFF and root AGENTS.
  A phase can authorize necessary preparation, runs and ordinary fixes; a script
  or old plan alone never authorizes a launch. Preserve the same cumulative
  budget across retries and follow explicit pause/closeout instructions.
- Before paid GPU work, read
  [`../docs/overview/RTX5090_BLACKWELL_PROFILE.md`](../docs/overview/RTX5090_BLACKWELL_PROFILE.md)
  and freeze code/config/data/checkpoint/table/output identities plus stop and
  shutdown plans.
- Put reusable diagnostics in `scripts/analysis/`, data builders in
  `scripts/data/`, and standalone evaluators in `scripts/eval/`. Extend an
  existing `scripts/experiments/` family for its related protocol rather than
  creating a duplicate runner solely to match a folder label.
- Do not add another phase runner merely because an old preflight exists.
- Historical experiment reports now live under
  `docs/exp/YYYY-MM/YYYY-MM-DD_slug.md`.
- A passing unit test proves only the code path it executed; it does not prove a
  model result, scientific claim, or cross-environment release.
