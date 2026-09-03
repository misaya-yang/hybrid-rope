# RoPE Has a Spectral Budget

Research code, historical evidence, and the active ICLR 2027 submission for
finite RoPE allocation, full sin/cos geometry, and training co-adaptation.

## Cold start for an AI

Read exactly these files in order:

1. [`AGENTS.md`](AGENTS.md) — rules, claim ceilings, terminology, compute and
   Git safety.
2. [`INDEX.md`](INDEX.md) — the only durable theory/evidence/code map, closed
   routes, directory ownership, and research agenda.
3. [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) — current Git/PDF/machine
   state, validation receipts, known issues, and immediate actions.

If historical context is needed, then read
[`paper-2027/research/history/TIMELINE.md`](paper-2027/research/history/TIMELINE.md).
The timeline is a ledger, not a fourth authority. On conflict:
**rules > index > state > historical summary**.

## Current snapshot

`PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED /
NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`

- Fixed-support interior allocation is causally active in the 151.9M
  three-seed training study.
- Frozen mature checkpoints are sensitive to the ordered pairing between
  rotary subspaces and frequency/dilation; an unordered spectrum is
  insufficient.
- Static pure-`z` interventions improve several long NLL and capability
  endpoints, but the tested Native-retention/natural-generation-QA joint
  objective remains unsolved.
- No new GPU run, OpenReview upload, commit, or push is implied by repository
  state. Check the handoff for live authorization.

## Repository layout

| Path | What it contains | Status |
| --- | --- | --- |
| [`paper-2027/`](paper-2027/) | active ICLR manuscript, build, submission state | active |
| [`paper-2027/NARRATIVE_GUIDE.md`](paper-2027/NARRATIVE_GUIDE.md) | reviewer-facing story and author doctrine | active manuscript guidance |
| [`paper-2027/REVISION_BRIEF.md`](paper-2027/REVISION_BRIEF.md) | bounded revision contract | active manuscript guidance |
| [`paper-2027/research/foundations/`](paper-2027/research/foundations/) | durable theory and causal-variable documents | canonical internal theory |
| [`paper-2027/research/evidence/`](paper-2027/research/evidence/) | paper-level result owners outside mature retrofit | canonical evidence |
| [`paper-2027/research/attention-aware-retrofit/`](paper-2027/research/attention-aware-retrofit/) | mature-checkpoint results, receipts, analyses, historical preflights, theory | canonical programme archive |
| [`paper-2027/research/history/`](paper-2027/research/history/) | chronological summaries | non-authoritative ledger |
| [`paper-2027/research/archive/`](paper-2027/research/archive/) | retired plans, simulated reviews, process logs | frozen history |
| [`analysis/`](analysis/) | historical full-RoPE audit bundle and its raw static outputs | reproduction archive; current claims route elsewhere |
| [`docs/`](docs/) | NeurIPS-era provenance, historical reports, and superseded theory | historical/infrastructure layer |
| [`docs/exp/`](docs/exp/) | NeurIPS-era experiment reports grouped by month | historical reports |
| [`experiments/`](experiments/) | standalone supporting model/protocol packages | code; no result by itself |
| [`rebuttal/rebuttal_0723/`](rebuttal/rebuttal_0723/) | NeurIPS review/rebuttal and July mature-model owners | historical evidence layer |
| [`data/curated/`](data/curated/) | small portable machine-readable evidence | tracked provenance |
| [`scripts/`](scripts/) | reusable implementation, diagnostics, data/eval tools | code; not evidence by itself |
| [`paper_experiments/`](paper_experiments/) | integrity-checked browsing view of historical paper code | generated workspace; not an owner |
| [`research_notes/`](research_notes/) | legacy exploratory bundles | non-authoritative history |
| [`nonuniform-alloc/`](nonuniform-alloc/) | protected legacy allocation study | closed historical branch; do not treat as queue |
| [`falsification_benchmark/`](falsification_benchmark/) | 16-episode blind theory benchmark | completed repository tool |
| [`paper/`](paper/) | NeurIPS 2026 baseline | immutable; never compile or edit |
| `internal/`, `results/`, local caches | private/raw/archive layers | not navigation or automatic evidence |

The complete ownership table and placement rules are in `INDEX.md`.

## Build and validation

Machine profiles are intentionally distinct:

- **Work machine:** owns the canonical Conda `aidemo` environment,
  PyTorch/pytest validation, supplement packaging, and final release checks.
- **Low-configuration personal PC:** reading, documentation, planning,
  lightweight standard-library checks, and local LaTeX/Tectonic iteration.
  `aidemo` is not expected here; record them as skipped when work-machine
  checks are unavailable rather than recreating the environment.

Canonical work-machine checks:

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_repository_navigation.py tests/test_rope_core.py -q
```

```bash
cd paper-2027 && ./compile.sh
```

```bash
conda run --no-capture-output -n aidemo \
  python scripts/package_supplement.py --profile iclr2027
```

Run packaging from the repository root. `compile.sh` proves format/build health,
not scientific evidence. Never compile `paper/` and never archive the repository
root as a supplement.

## Non-negotiable operating rules

- A plan, script, filename, checkpoint inventory, or launch log is not a
  result. Follow `INDEX.md` to the canonical owner.
- Keep NLL/PPL, teacher-forced retrieval, strict generation, RULER/NIAH, QA,
  adaptation, and transfer as separate evidence tiers.
- Preserve unrelated work. Do not pull, switch, stage, commit, push, reset,
  stash, or start GPU/paid compute without explicit authorization.
- Current submission validity gates are in
  [`paper-2027/SUBMISSION_CHECKLIST.md`](paper-2027/SUBMISSION_CHECKLIST.md);
  their live pass/fail state is only in the handoff.
