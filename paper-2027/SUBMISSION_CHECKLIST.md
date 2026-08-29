# ICLR 2027 submission gate specification

> **Role.** This file defines stable release gates. It deliberately contains no
> completion checkboxes, current hashes, current page receipt, or pass/fail state.
> Record all live results, blockers, uploaded-state receipts, and author actions
> only in [`HANDOFF.md`](HANDOFF.md).
>
> Canonical build, Python/PyTorch/pytest, packaging, and release checks run on
> the work machine with its `aidemo` environment. The low-configuration
> personal PC does not own those gates.

## Official sources and dates

Open these official sources at each relevant gate; the handoff records the
latest live-check date and outcome:

- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- [official ICLR 2027 style files](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)

Submission milestones:

- **2026-09-17:** internal title, abstract, author-roster, and author-metadata
  freeze;
- **2026-09-18, 11:59 PM AoE:** official abstract deadline;
- **2026-09-25:** official full-paper deadline.

Policies and platform requirements can change. Reopen the official pages and
recheck the live OpenReview form immediately before the 9/17 freeze, the 9/18
abstract submission, and the 9/25 full-paper submission. A dated check in this
file is not proof of current compliance.

## Gate A — internal abstract and author-metadata freeze, 2026-09-17

The freeze is ready only when all of the following are true:

- The title and abstract match the current manuscript's scientific claim,
  evidence scope, terminology, and canonical owners.
- The abstract contains no unsupported result, new protocol interpretation,
  hidden submission experiment, or stale language from an archived review plan.
- The author roster is final for the abstract deadline; author order and any
  later permitted metadata change follow the live official policy.
- Every author has the required OpenReview profile and satisfies the current
  profile-completeness, reciprocal-reviewing, submission-count, and author-limit
  rules.
- Any required reviewer registration or official exemption is resolved rather
  than assumed.
- The AI-use statement and author confirmations remain literally accurate under
  the current ICLR policy.
- The dual-submission condition and the possible NeurIPS-decision branch are
  understood by the authors.
- The candidate OpenReview title and abstract exactly match the internally
  frozen text.

The live author confirmations and unresolved items are recorded in the handoff,
not by editing this gate specification.

## Gate B — official abstract submission, 2026-09-18

Before 11:59 PM AoE:

- Submit the frozen title, abstract, author roster, and required metadata through
  the official OpenReview venue.
- Confirm the platform accepted the submission and did not alter math, Unicode,
  whitespace, or author metadata materially.
- Compare the saved OpenReview record with the frozen source.
- Record the submission identifier, timestamp, and exact frozen state only in
  the handoff; do not place identifying submission data in the anonymous paper
  or supplement.

After the abstract deadline, treat the roster and other frozen fields according
to the live official policy. Do not infer that a field remains editable merely
because the interface exposes it.

## Gate C — manuscript format and build

The full-paper candidate must satisfy the current official format and the local
mechanical gates:

- Use the official `iclr2027_conference` style and leave `\iclrfinalcopy`
  disabled for anonymous submission.
- Keep the main text within the official nine-page limit; bibliography,
  appendices, and exempt required statements must appear in the allowed order.
- Include the required AI-use statement and the selected ethics and
  reproducibility statements in policy-compliant form.
- Produce anonymous US-Letter output with no undefined citations/references,
  disallowed overfull boxes, Type-3 fonts, unembedded fonts, or oversized PDF.
- Build from a clean temporary state using the documented repository command;
  record source/PDF hashes and the exact receipt in the handoff.
- Visually inspect every page at final size, including figures, tables,
  footnotes, references, statements, and appendix transitions.

Compilation establishes format and layout health only. It does not validate a
scientific claim or prove that the uploaded artifact matches the local file.

## Gate D — scientific and provenance consistency

The final source/PDF passes only when:

- Every reviewer-facing number, table cell, figure value, seed count,
  uncertainty unit, and protocol interpretation resolves to its canonical owner.
- Fixed-support identification, the 50M factorial, table-by-weights crossing,
  mature frozen intervention, matched adaptation, scale studies, and video-DiT
  evidence remain separate protocols with their actual causal roles.
- Likelihood, perplexity, strict autoregressive exact, RULER, Qasper, 2Wiki, and
  causal source-use endpoints are not translated into one another.
- The exact-range claim uses the raw-hash-receipted three-training-seed owner;
  evaluation rows or anchors are not counted as independent seeds.
- The 1.485B from-initialisation comparison retains its
  same-initialisation/same-scientific-recipe scope; the 8B result remains matched
  adaptation, not pretraining-scale evidence.
- Static full-RoPE geometry remains a positional-basis diagnosis, not an
  LM-quality predictor.
- EVQ-Cosh, anchored EVQ-Cosh, frozen derived/coarse allocations, `Geo`,
  `Native`, FMRoPE, YaRN-style, and the MLA wavelength-blend operator retain
  their locked identities.
- The transplant theorem, finite-`tau` rule, LeRoPE relationship, and all other
  claims remain within the ceilings in `AGENTS.md`.
- One final author review checks every number against the named owner after
  layout is frozen.

## Gate E — anonymity and anonymous supplement

The paper and release package pass only when:

- No author names, affiliations, acknowledgements, private machine paths, host
  names, credentials, identifying repository links, checkpoints, caches, or
  ignored raw outputs leak into reviewer-visible artifacts.
- Internal planning, handoff, review-log, external-review, and archived verdict
  documents remain excluded.
- The anonymous package contains the source, code, configuration, evaluator
  contracts, and machine-readable evidence required by its curated profile.
- Every supplement evidence-map entry routes to a genuine owner or receipt, not
  back to manuscript prose as if the paper were its own evidence.
- The curated packager, leak scan, archive-integrity check, isolated paper build,
  and allowlisted tests complete successfully under the required environment.
- The immutable `../paper/` baseline remains unchanged and is never compiled or
  packaged as the active submission.

## Gate F — dual-submission and NeurIPS-decision branch

Before the full-paper upload:

- Recheck the current ICLR and NeurIPS policies against the actual decision and
  submission state; do not rely only on the August timing audit.
- If the NeurIPS outcome requires a citation or contribution-boundary update,
  cite the work in the permitted third-person form and state the old/new
  contribution boundary accurately.
- If no update is required, do not add speculative acceptance/rejection language
  to the manuscript.
- Use [`CHANGES_FROM_NEURIPS2026.md`](CHANGES_FROM_NEURIPS2026.md) as the durable
  distinctness owner, not as a record of the current decision state.

The actual outcome, author decision, and applied branch belong only in the
handoff.

## Gate G — final upload, 2026-09-25

The submission is complete only when:

- The final title and abstract match between OpenReview and the uploaded PDF.
- The OpenReview AI-use disclosure matches the final mandatory manuscript
  statement and the authors' confirmed research workflow.
- The PDF, source package if requested, and anonymous supplement are the exact
  locally verified artifacts.
- The uploaded files are downloaded from the platform and inspected again for
  corruption, wrong version, metadata drift, anonymity leaks, and rendering
  changes.
- All author-visible warnings and required platform fields are resolved.
- Final local and platform receipts are recorded in the handoff, with passed,
  failed, skipped, and unverified checks distinguished.

This file remains unchanged when a gate passes. Only the handoff records that
live fact.
