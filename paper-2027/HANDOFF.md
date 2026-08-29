# ICLR 2027 active handoff

- **Updated:** 2026-08-29
- **Target:** ICLR 2027
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Live branch / upstream at cycle start:** `main_0726` / `origin/main_0726`
- **HEAD at cycle start:** `8bfcd3bdfe6b53335400d8be2503d124fbbc1e98`
  (`docs: post-cycle documentation cleanup and routing refresh`)
- **Observed divergence before local edits:** `0 / 0`
- **Publication authorization:** on 2026-08-29 the author explicitly authorized
  one scoped commit and ordinary push of this September documentation reset.
  Resolve the final local/remote SHA live after publication.
- **Current machine:** low-configuration personal PC. It is a
  documentation/planning host, not the work machine; Conda `aidemo` is not
  expected here.

## 1. Current status

The author opened a new September iteration on 2026-08-29. The current
`paper-2027/` manuscript is the only starting point. August revision plans,
author-verdict ledgers, alternating model-review logs, and external-model review
bundles are archived audit inputs and do not carry tasks into this cycle.

This first task is documentation governance only: remove stale routes and
duplicated rules, replace the closed August brief with the September scope,
separate current evidence owners from historical manifests, and leave one
research decision order. No manuscript TeX, training, GPU evaluation, paid
compute, upload, commit, or push is authorised by this task.

The stable scope is [`REVISION_BRIEF.md`](REVISION_BRIEF.md). The manuscript
strategy is [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md). This file is the only
live state and action queue.

## 2. Submission milestones

Official ICLR sources were rechecked on 2026-08-29:

- [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)

| Milestone | Role | Current state |
| --- | --- | --- |
| **2026-09-17** | Internal title, abstract, author-roster, and author-metadata freeze | pending |
| **2026-09-18, 11:59 PM AoE** | Official abstract deadline | pending; recheck live policy and OpenReview form |
| **2026-09-25 AoE** | Official full-paper deadline | pending; recheck live policy and OpenReview form |

The 9/17 date is an internal safety freeze, not the venue deadline. Stable gate
definitions are in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md); their
live pass/fail state belongs here.

## 3. Current action queue

| Order | Action | State / exit condition |
| --- | --- | --- |
| 1 | September documentation reset | completed locally; scoped commit/push authorized; lightweight PC checks passed, canonical work-machine validation remains skipped |
| 2 | Current-PDF/source audit | pending; start from current TeX and last-built PDF, then identify only score-, credibility-, comprehension-, or validity-changing deltas |
| 3 | 9/17 abstract and metadata freeze | pending; owner-audited title/abstract and author-policy confirmations |
| 4 | 9/25 full-paper closure | pending; owner-by-owner scientific audit, build, visual review, anonymous supplement, upload/readback |

No old A/R item is pending by default. A new revision item enters this queue
only after it passes the admission fields in `REVISION_BRIEF.md` §7.

## 4. Manuscript contract

Title: *RoPE Has a Spectral Budget*.

The stable reviewer memory, reader path, causal contract, and body-allocation
rules live only in [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md). The current local
delta is documentation governance only; no manuscript TeX has changed in this
task. [`../INDEX.md`](../INDEX.md) §3 and
[`research/README.md`](research/README.md) route evidence.

## 5. Source, PDF, and release baseline

The last manuscript-changing checkpoint is
`93d7eac941d45ed6877bcc17a5f73443f53dfdfb` (`paper: complete ICLR 2027
revision cycle`). HEAD `8bfcd3b` adds documentation-only routing cleanup.

Last-built active PDF:

- `paper-2027/main.pdf`
- SHA-256: `ade3beb59eefca91b339ef4d6731cdda19fa8eb40f384d3eab60fd9c4df06f01`
- size: `757522` bytes
- historical receipt: 9 body / 28 total pages, references beginning on page 10

Immutable baseline:

- `paper/main.pdf`
- SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`
- size: `1393787` bytes

The committed source is one token ahead of the last-built PDF: after the last
compile, `sections/04_experiments.tex` restored a missing backslash on the
`\evq` macro in a caption. Therefore the historical PDF/build/supplement hashes
are not a current-source build receipt. Resolve this drift only in the later
authorised build stage; do not describe the last-built PDF as source-synchronised.

Earlier `43/43`, `181/181`, and packaged `144/144` test counts remain historical
receipts until rerun. They prove neither the current documentation tree nor the
future submission artifact.

## 6. Documentation reset receipt

Changed scope is documentation and its navigation contract only. The reset:

- removes branch and current-state facts from stable rules;
- replaces August `REVISION_BRIEF` v2 with September v3;
- freezes the old author verdict and Codex/Claude review log as history;
- removes external-model outputs from cold start, evidence routing, and current
  priority;
- rewrites `INDEX.md` §6 around the September paper and a single post-submission
  research gate;
- classifies preflights and theory notes as current owner, design-only,
  superseded, closed negative, or historical provenance;
- marks NeurIPS-era provenance/code manifests as historical/partial views, not
  current ICLR authority.

Local low-configuration-PC validation:

- **Passed:** `git diff --check`;
- **Passed:** local-relative-link check across all 40 modified Markdown files;
- **Passed:** added-line scan for private absolute paths, private-key markers,
  and common secret assignments;
- **Passed (fallback only):** 20/20 pure standard-library `unittest` navigation
  and paper-experiment-workspace checks with bytecode writes disabled;
- **Skipped by machine profile:** canonical Conda `aidemo` pytest, paper build,
  supplement packaging, isolated package tests, and visual PDF inspection;
- **Unverified:** current-source/PDF synchronisation, current supplement hash,
  full scientific owner audit, and final submission compliance.

The fallback tests are useful PC diagnostics but do not replace a work-machine
`aidemo` receipt. No manuscript TeX or reviewer-facing numerical owner changed
in this task.

## 7. Post-submission research boundary

No submission experiment is in the active queue. The durable post-submission
research order lives only in [`../INDEX.md`](../INDEX.md) §6. The next routed
design is currently `FROZEN_PROTOCOL_DESIGN_NOT_EXECUTED`, has no dedicated
runner/readiness receipt, and has no GPU authorization. No later research branch
is active on this PC or in this manuscript cycle.

## 8. Author actions

Before 9/17:

- freeze the author roster and confirm current OpenReview profiles;
- confirm author-count/submission quotas and reciprocal-review eligibility;
- confirm that the AI-use statement remains literally complete under the live
  policy;
- confirm the exact title and abstract to enter in OpenReview.

Before 9/25:

- decide the NeurIPS-outcome citation/distinctness branch using
  [`CHANGES_FROM_NEURIPS2026.md`](CHANGES_FROM_NEURIPS2026.md);
- perform the final owner-by-owner number review after layout freezes;
- approve the final uploaded PDF/supplement and downloaded-platform readback.

## 9. Authorization and Git boundary

Current authorization covers this documentation reset, its lightweight
static/standard-library validation on the personal PC, one scoped Git commit,
and one ordinary push to the existing upstream branch. Canonical
Python/PyTorch/pytest, final build, and packaging remain skipped here and belong
on the work machine's `aidemo` environment. Authorization does not cover
installing that environment here, manuscript TeX edits, training, GPU
inference/evaluation, paid compute, other remote mutation, upload, force-push,
branch operations, or history rewriting.

Before any later authorised Git publication, recheck branch, upstream,
divergence, worktree, staged scope, sensitive content, and immutable `paper/`.
Record the resulting local/remote SHA only after a successful ordinary push.
