# NeurIPS 2026 → ICLR 2027 contribution distinctness

> **Role.** This internal document is the durable owner of the scientific
> contribution boundary between the immutable NeurIPS 2026 baseline in
> `../paper/` and the active ICLR 2027 manuscript in this directory. It is not a
> manuscript-status file, PDF-hash receipt, section locator, review-response
> ledger, experiment plan, or action queue. Live state belongs only in
> [`HANDOFF.md`](HANDOFF.md).

This document is excluded from the anonymous paper and supplement.

## 1. Durable contribution boundary

The two submissions share the EVQ-Cosh research lineage, but the active ICLR
paper changes the central scientific question, theory, causal evidence, and
evidence architecture.

| Dimension | NeurIPS 2026 baseline | ICLR 2027 manuscript |
| --- | --- | --- |
| Organising claim | EVQ-Cosh construction and its operating rule | Interior allocation `z` is a separately identifiable finite-RoPE design coordinate |
| Parameterisation | Frequency-allocation method on the original formulation | Explicit decomposition \(x_k=a+Rz_k\), separating support from interior allocation |
| Theory | Convex Cosh surrogate and operating-point analysis | Exact full-sin/cos subspace geometry, spectral-budget identity, slow-frequency collapse, co-adaptation boundary, and exact frozen-transplant obstruction; EVQ-Cosh remains a bounded construction |
| Causal identification | Earlier EVQ/range-composition and mechanism studies | Raw-hash-receipted three-seed fixed-support training intervention plus mature same-support frozen pure-`z` controls |
| Empirical structure | Method-centred experiments and older learned/range comparisons | Protocol-separated fully frozen, matched-adaptation, and from-training consequences, with architecture/scale/modality evidence in their actual roles |
| Construction claim | EVQ-Cosh as the main proposed answer | EVQ-Cosh as a closed-form zero-learned-parameter construction unique only for its stated surrogate |
| Reviewer memory | A variational allocation method for RoPE | A finite RoPE table is determined by sampled support and interior allocation; allocation changes its effective positional basis and behaviour |

The ICLR manuscript is therefore a scientific continuation with shared method
ancestry, not a relabelling of the earlier manuscript. Its independent
contribution is the controlled coordinate, exact full-basis account, causal
identification, and lifecycle intervention evidence.

## 2. Retained material and honest overlap

Distinctness does not require deleting valid shared foundations. The ICLR paper
may retain EVQ-Cosh definitions, supporting experiments, citations, and proof
components when they remain scientifically necessary. Their role must be stated
accurately:

- EVQ-Cosh is a construction on the newly isolated coordinate, not the sole
  owner of the coordinate claim.
- Earlier experiments may support construction persistence or breadth but do
  not replace the fixed-support causal owners.
- Shared technical text, equations, or artifacts must follow the venue's current
  citation and dual-submission rules.
- The immutable `../paper/` baseline is never edited, compiled, regenerated, or
  presented as the active ICLR submission.

## 3. Durable method-identity corrections

Two historical identity failures must not return:

1. The repository's learned inverse-frequency row is a layer-shared learned
   table with 32 learned frequency parameters. It is not DAPE and cannot isolate
   allocation shape from parameterisation or optimisation effort.
2. The repository fixed-index smooth-ramp scaler is `YaRN-style`, not an exact
   reproduction of official YaRN. The run-specific MLA wavelength-blend
   operator is a third, separate object. Standard Transformers YaRN, the
   YaRN-style operator, and the MLA operator must never be conflated.

The locked nomenclature and current claim ceilings live in
[`../AGENTS.md`](../AGENTS.md). Canonical evidence owners and current roles live
in [`../INDEX.md`](../INDEX.md); this document does not duplicate their numbers.

## 4. Conditional citation and submission branch

The final ICLR submission must follow the live venue policies and the actual
NeurIPS decision state.

- If the outcome requires citation of the earlier work, cite it in the permitted
  third-person form and state the contribution boundary in §1 accurately.
- If the outcome does not require a manuscript change, do not add speculative
  acceptance/rejection language.
- Recheck title, abstract, author roster, public-status language, and the final
  PDF against the current policy immediately before upload.

The stable release gate is in
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md). The actual outcome,
decision, applied edit, and validation receipt belong only in the handoff.

## 5. Historical snapshot

The earlier 2026-08-19 document contained a reviewer-by-reviewer response table,
old section locators, a current-at-the-time PDF hash comparison, a numerical
owner ledger, audit corrections, and a “next round” experiment list. Those
materials are preserved in Git for provenance but are retired as active inputs:

```bash
git show 79aa932:paper-2027/CHANGES_FROM_NEURIPS2026.md
```

Do not revive an old experiment, unresolved item, or reviewer framing from that
snapshot. New submission work starts from the current manuscript under
[`REVISION_BRIEF.md`](REVISION_BRIEF.md); post-submission research is routed only
through `INDEX.md` §5.
