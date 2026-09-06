# RoPE Has a Spectral Budget

This project isolates an under-studied coordinate by separating sampled support and interior
allocation: `x_k = -log(omega_k) = a + R z_k`. Fixed-support interventions identify
`z`; target-aware support retargeting demonstrates their interaction. Exact
sin/cos geometry explains the finite positional budget. EVQ-Cosh is one analytic
construction, distinct from the mature frozen derived/coarse/log-p2 tables.

The active manuscript is `paper-2027/`. Completed frozen,
adaptation, and from-training studies keep their own estimands. The current
paper does not depend on prospective 8x/32x success.

**Repository layout (2026-09-06 slim).** The working branch `main_0726_09_06`
keeps only the active manuscript and its live apparatus: `paper-2027/`,
`docs/`, `scripts/`, `tests/`, this README, `INDEX.md`, `AGENTS.md`, and the
supplement zip at `paper-2027/rope-spectral-budget-iclr2027-supplement.zip`.
All historical trees (`paper/`, `rebuttal/`, `data/`, top-level `results/`,
`internal/`, `experiments/`, `research_notes/`, `analysis/`, …) are archived
unchanged on branch `main_0726` — read them there with `git show
main_0726:<path>`; nothing scientific was deleted.

## Start here — GPT-6-led research sessions

1. Read `AGENTS.md`, this README, and [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md).
2. Use [`INDEX.md`](INDEX.md) to open the exact question's current owner only.
3. For the next experiment, read the [first-principles contract](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md)
   and [staged work-machine protocol](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md).
4. For paper changes, read the outcome-dependent edit plan in
   [`REVISION_BRIEF.md`](paper-2027/REVISION_BRIEF.md), then current TeX and its owner.

Do not read the timeline, theory tree, failed plans, or external-model reviews
wholesale. They are searchable history, not an inherited research queue. A
model's confidence and a previous run's completion are not assay validation.

## The two active questions

| Route | Quantified target | Next decisive evidence |
| --- | --- | --- |
| Z: zero training | One global static table/gain; separate Native PPL and task retention >=.88; maximize measured physical reach from 4x toward 8x | Independent source twins + nearby oracle + deleted-source control, complete generation/EOS, and natural retention |
| F: light adaptation | Qualified Native-compact natural tasks at physical <=16K; same Native limits; untouched farther capability | Fixed N/Z/Y all-linear r16, lawful-world full-trajectory supervision and actual-deployment Native teacher constraint |

Table factor is not useful context length. NLL, attention scores and operator
bounds are diagnostics, never capability selectors. The full-p2 s4/c=.074
incumbent passes historical .875 retention but is marginal at strict .88.
The old mixed C2-s2/p2-s4 fit is not a valid same-path ceiling. The latest fixed-witness candidates failed the declared joint Native gates;
do not reopen their scale/gain/curve search. The supplied GPT-6 Pro
dossier changed the adaptation design; the exact review is in the theory owner §8.

## Next execution entrypoint

Read the [prepared round, protocol §10](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md#10-开机后的固定比较轮次--2026-09-05-准备版), then:

```bash
python3 scripts/experiments/matched_transfer_round.py template
```

The next comparison is N_compact, then fixed-recipe Qwen Z/Y, using the archived
N128 training engine. Preparation verifies assets and matched exposures on the
work machine; GPU execution is a separate explicit action. No resume, prefix
change or test opening is part of this round. Read the [execution report](paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md)
for completed results: the OLMo frozen candidates failed their Native gates;
N128 passed aggregate Native confirmation with a format/indexing regression.
The original staged driver remains historical; its default step64 is not this
round's uninterrupted step128 contract. Local checks do not establish GPU readiness.

## Evidence hierarchy

The paper's strongest causal owners remain the three-seed fixed-support study
and the matched-support frozen study. A successful new single-table result is
a separate deployment increment; do not relabel the earlier arithmetic/routed
results. The fully frozen zero-training, matched low-rank adaptation, and
from-training/co-adaptation routes retain their separate estimands.
Submission milestones remain 2026-09-17, 2026-09-18 and 2026-09-25, with live policy checks.
New research and manuscript verification proceed separately.

Rules live in `AGENTS.md`, scientific routing in `INDEX.md`, live state only in
`paper-2027/HANDOFF.md`. The low-configuration personal PC may implement and run
light CPU checks; canonical GPU, packaging and release validation stay on the
work machine. Do not install or recreate the work-machine environment here.
