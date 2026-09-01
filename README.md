# RoPE Has a Spectral Budget

Research code and the active ICLR 2027 submission package for finite RoPE
spectral allocation, full sin/cos subspace geometry, and training
co-adaptation.

The current paper's central claim is:

> Even at fixed sampled spectral support, the normalized interior allocation
> of a finite RoPE table is a separately identifiable training-time variable.
> It changes full sin/cos subspace geometry and trained behaviour, while model
> weights co-adapt to the table used during training.

## Start here

Three files, one authority each. Read them in this order.

1. [`AGENTS.md`](AGENTS.md) — **rules**: objective, claim ceilings, locked
   nomenclature, evidence identity, compute authorization, Git discipline.
2. [`INDEX.md`](INDEX.md) — **index**: theory, evidence owners, code, directory
   ownership, falsified routes, research agenda, two-machine workflow.
3. [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) — **state**: current
   manuscript, hashes, validation receipts, Git and machine status, author
   actions.

On conflict, rules beat index beats state. Everything else in this repository
is reachable from `INDEX.md`; do not add a fourth navigation authority.

`AGENTS.md`, `INDEX.md`, and `paper-2027/HANDOFF.md` are internal repository
navigation and are intentionally absent from the anonymous supplement.

The active September 2026 iteration is scoped by
[`paper-2027/REVISION_BRIEF.md`](paper-2027/REVISION_BRIEF.md); the handoff is
the only place that records its current step, receipts, and deadlines. Closed
August plans, author-verdict ledgers, alternating model-review logs, and
external-model reviews are audit history, not current state or action queues.

[`paper-2027/NARRATIVE_GUIDE.md`](paper-2027/NARRATIVE_GUIDE.md) owns the
non-negotiable author doctrine, target/non-target propositions, and
future-session stop rules. Together they guard against reverting to an
arbitrary-$z$, caveat-first, method-race, or over-restrained paper.

## Workspace

- `paper-2027/` is the only active manuscript.
- `paper/main.pdf` is the immutable NeurIPS 2026 baseline: never edit, compile,
  move, or regenerate it.
- `rebuttal/rebuttal_0723/README.md` is a historical review and evidence
  archive, not the current action queue.
- The current branch, upstream, divergence, and worktree state live only in the
  handoff and must be verified before any mutation.

`INDEX.md` §5 owns the full directory table and the historical review/provenance
entrypoints. Stable rules do not duplicate those paths.

## Build and validate

### Machine profiles

- **Work machine:** owns the canonical Conda `aidemo` environment and is the
  default place for Python/PyTorch/pytest, packaging, and final cross-environment
  release validation.
- **Low-configuration personal PC:** intended for repository reading,
  documentation, planning, local LaTeX/Tectonic compilation, visual PDF
  iteration, and lightweight static or standard-library checks. `aidemo` is not
  expected on this machine. Do not install or reproduce the work-machine
  environment, run model compute, or substitute the local build for the final
  work-machine packaging/cross-environment receipt.

The Conda commands below are canonical work-machine invocations. On the personal
PC, if those checks are unavailable, record them as skipped instead of treating
missing Conda as a repository failure; `paper-2027/compile.sh` may run on either
machine.

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_repository_navigation.py tests/test_rope_core.py -q
```

Expand to the tests owned by the changed path; the handoff records the latest
validated scope rather than implying that one command certifies the repository.

The retired W0/F1 success-first tournament has no runnable command in this
README. Its code and preflight remain historical provenance only. The latest
two-parameter static-`z` GPU result is mixed and has no active rescue candidate;
current evidence and next-action status live in `INDEX.md` and the handoff.

```bash
cd paper-2027 && ./compile.sh
```

```bash
conda run --no-capture-output -n aidemo python scripts/package_supplement.py --profile iclr2027
```

Run packaging from the repository root. `compile.sh` verifies format and build
health only, not scientific evidence. Never compile `paper/` and never zip the
repository root. Exact current hashes and receipts live in the handoff.

## Ground rules

- Every paper number resolves to a canonical owner with method, protocol,
  endpoint, and seed identity. Plans, scripts, filenames, launch logs, and
  external-model reports are not completed evidence.
- AI cross-reviews are adversarial inputs. Verify alleged defects against the
  PDF, source, proof, and owner before revising.
- No GPU or paid experiment is authorized unless the user explicitly approves
  that exact run.
- Preserve unrelated worktree changes. Do not stage, commit, push, reset, or
  switch branches without explicit authorization.

Stable policy, anonymity, dual-submission, and release gates are in
[`paper-2027/SUBMISSION_CHECKLIST.md`](paper-2027/SUBMISSION_CHECKLIST.md).
Current author actions and their live status are recorded only in
[`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md).
