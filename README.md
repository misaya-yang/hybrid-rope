# RoPE Has a Spectral Budget

Research code and the active ICLR 2027 submission package for finite RoPE
spectral allocation, full sin/cos subspace geometry, and training
co-adaptation.

## Objective

The first principle and highest priority is to maximize the probability of
ICLR 2027 acceptance, subject to scientific truth and submission validity.
Work that cannot change a reviewer decision or protect validity is out of
scope.

The current paper's central claim is:

> Even at fixed sampled spectral support, the normalized interior allocation
> of a finite RoPE table is an independent training-time variable. It changes
> full sin/cos subspace geometry and trained behaviour, while model weights
> co-adapt to the table used during training.

EVQ-Cosh is one closed-form, zero-learned-parameter construction on this axis,
not a universal optimum.

## Start here

1. [`AGENTS.md`](AGENTS.md) — stable objective, claim boundaries, safety, and
   verification rules.
2. [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md) — current manuscript,
   worktree, validation, and next-action state.
3. [`paper-2027/README.md`](paper-2027/README.md) — active ICLR package,
   format, build, and source layout.
4. [`paper-2027/research/README.md`](paper-2027/research/README.md) —
   canonical theory/evidence routing.
5. [`REPO_MAP.md`](REPO_MAP.md) — directory ownership and source-of-truth map.

`AGENTS.md`, `paper-2027/HANDOFF.md`, and `REPO_MAP.md` are internal
repository navigation and are intentionally absent from the anonymous
supplement.

`paper-2027/` is the only active manuscript. `paper/` is the immutable
NeurIPS 2026 baseline. `rebuttal/rebuttal_0723/` is a historical review and
evidence archive, not the current action queue.

## Current scientific chain

| Layer | Main owner |
| --- | --- |
| Pure interior-allocation identification | 151.9M exact-range control + 50.9M M4 factorial |
| Static finite-basis theory | full sin/cos Gram, canonical correlations, stable-rank identity, low-frequency collapse |
| Training co-adaptation | exact transplant obstruction + 50M weight/table crossing |
| Mature-scale persistence | 1.485B from-initialization and OLMo/LLaMA adaptation protocols |
| Constructive instance | fixed analytic EVQ-Cosh table |
| Range composition | same fixed-scale `YaRN-style` transform on Geo and EVQ substrates |

Protocol-specific endpoints remain separate: NLL/PPL, teacher-forced passkey
NLL-gap, strict autoregressive exact match, 2Wiki, RULER, and causal source-use
are not interchangeable.

## Build and validate

The repository's Python/PyTorch validation environment is Conda `aidemo`.

```bash
conda run --no-capture-output -n aidemo python -m pytest \
  tests/test_rope_core.py \
  tests/test_rebuttal_protocol_regressions.py \
  tests/test_rebuttal_evidence_bundle.py \
  tests/test_fmrope_125m_l256_500m.py \
  tests/test_frequency_adaptation_8b.py \
  tests/test_olmo2_1b_evq.py -q
```

Build the active paper:

```bash
cd paper-2027
./compile.sh
```

Build the curated anonymous supplement from the repository root:

```bash
conda run --no-capture-output -n aidemo \
  python scripts/package_supplement.py --profile iclr2027
```

Do not zip the repository and do not compile `paper/`. Exact current hashes
and validation receipts live in [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md).

## Repository layout

```text
paper-2027/                 active ICLR 2027 manuscript and internal handoff
  research/                 canonical theory, audits, and evidence routing
paper/                      immutable NeurIPS 2026 submission baseline
rebuttal/rebuttal_0723/     historical NeurIPS review and evidence owners
scripts/lib/rope/           canonical RoPE schedule implementation
scripts/analysis/           reusable CPU diagnostics
scripts/core_text_phases/   main small/mid-scale experiment chain
scripts/supporting_eval/    capability and supporting evaluators
experiments/                standalone model-scale experiment packages
data/curated/               tracked, sanitized machine-readable evidence
docs/overview/              historical submission provenance and reproduction
paper_experiments/          manifest-driven view of canonical experiment code
tests/                      implementation, protocol, evidence, and package gates
results/ and internal/      local/historical layers; not automatic paper evidence
```

## Evidence and external reviews

- Every paper number must resolve to a canonical owner with method, protocol,
  endpoint, and seed identity.
- Plans, scripts, filenames, and external-model reports are not completed
  evidence.
- AI cross-reviews are adversarial inputs. Verify alleged defects against the
  PDF, source, proof, and owner before revising.
- No GPU or paid experiment is authorized unless the user explicitly approves
  that exact run.
- Preserve unrelated worktree changes and do not stage, commit, push, reset, or
  switch branches without explicit authorization.

## Submission readiness

See [`paper-2027/SUBMISSION_CHECKLIST.md`](paper-2027/SUBMISSION_CHECKLIST.md)
for remaining author actions, policy checks, anonymity, and dual-submission
handling. Current work status is intentionally not duplicated here.
