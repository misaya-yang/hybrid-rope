# ICLR 2027 active handoff

- **Updated:** 2026-08-29
- **Target:** ICLR 2027
- **Active manuscript:** paper-2027/
- **Immutable NeurIPS baseline:** paper/
- **Branch / upstream:** main_0726 / origin/main_0726
- **Published baseline before this iteration:**
  40bde33a47e3a43d7c9afaf6a57b0f4fd9f68234
  (checkpoint: prepare September paper iteration)
- **Published manuscript/package checkpoint:**
  09876d5 (`checkpoint: upgrade ICLR paper and freeze narrative`)
- **Current published HEAD before this framing repair:**
  8bfaecf219e37003c272144ce62a6cc183c3ec41
- **Verified divergence before edits:** 0 / 0
- **Git publication state:** the paper-identity/framing repair is local and
  uncommitted; no commit or push was authorised in this turn.

## 1. Current result

The evidence-complete September manuscript is **locally complete**. The latest
framing repair makes the scientific identity explicit before any benchmark or
method identity:

1. $x_k=-\log\omega_k=a+Rz_k$ decomposes a finite table into sampled support
   $(a,R)$ and interior allocation $z$;
2. the paired fixed-support intervention identifies $z$, while target-aware
   support retargeting reverses the ordering and identifies support as the
   distinct interacting coordinate;
3. exact phase-invariant geometry exposes the finite spectral budget and
   slow-end redundancy;
4. EVQ-Cosh is one analytic construction on this object;
5. frozen, adapted, and from-training evidence establishes distinct
   behavioural consequences without pooling estimands.

Fixed-support FMRoPE versus anchored EVQ-Cosh owns the clean allocation
identification. The target-aware reversal owns the support--allocation
interaction. FMRoPE is therefore a causal control, not a competitor to
handicap. LeRoPE is attributed convergent evidence that learned allocation can
improve in-window behaviour; it is not a matched comparator or mechanism
validation.

The frozen model-relative tables are not EVQ-Cosh. EVQ-Cosh remains the
closed-form training/adaptation construction. The bundled Native/long session
policy remains distinct from the pure-$z$ frozen comparison.

The stable scope is [REVISION_BRIEF.md](REVISION_BRIEF.md), and the sole
narrative contract is [NARRATIVE_GUIDE.md](NARRATIVE_GUIDE.md).
[../INDEX.md](../INDEX.md) and [research/README.md](research/README.md)
route every numerical owner.

The durable post-submission agenda remains [`../INDEX.md`](../INDEX.md) §6.
This file is the only
live state and action queue.

## 2. Authorization and machine boundary

Current authorization covers manuscript, appendix, figure, navigation
documents, supplement allowlist closure, local LaTeX compilation, lightweight
static tests, visual PDF review, curated supplement packaging, and a scoped
ordinary commit/push of this complete change set.

It does **not** authorize new experiments, training, GPU inference/evaluation,
paid compute, OpenReview upload, unrelated remote mutation, branch operations,
force-push, or history rewriting.

The current low-configuration personal PC remains a documentation/planning host
and is also a valid LaTeX/Tectonic and visual PDF host. It may run
lightweight static or standard-library checks.
The work machine remains the canonical host for Conda aidemo
Python/PyTorch/pytest, the curated supplement build, isolated package build,
and final cross-environment release receipt. `aidemo` is not expected here and
must not be recreated merely to duplicate the work machine.

## 3. Submission milestones

| Milestone | Role | State |
| --- | --- | --- |
| **2026-09-17** | Internal title, abstract, author-roster, and metadata freeze | pending author confirmation |
| **2026-09-18, 11:59 PM AoE** | Official abstract deadline | pending live-policy/OpenReview recheck |
| **2026-09-25 AoE** | Official full-paper deadline | pending final release validation and upload |

The 9/17 date is an internal safety freeze. Stable submission gates are in
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

## 4. Current manuscript contract

Title: *RoPE Has a Spectral Budget*.

The reviewer path is:

1. a finite RoPE table has sampled support $(a,R)$ and an independent interior
   allocation $z$;
2. paired fixed-support training identifies allocation cleanly, while
   target-aware retargeting reverses the ordering and identifies support as the
   second interacting coordinate;
3. exact full-sin/cos geometry characterises the finite basis's
   Renyi-2 effective dimension, including the representative
   46-nominal-dimension to $r_2=2.00$ slow block;
4. a convex surrogate yields the closed-form EVQ-Cosh construction;
5. frozen, adaptation, and from-training results establish distinct practical
   consequences without pooling their estimands.

Claim boundaries remain those in [../AGENTS.md](../AGENTS.md). In particular,
the finite multiplier grid is not a continuous basin or optimum; static
geometry is not an LM-quality predictor; mature studies keep their protocol
identities; and the 1.485B from-initialisation comparison is same
initialisation/same scientific recipe, not bitwise paired execution.

## 5. Current PDF and immutable baseline

Active PDF:

- path: paper-2027/main.pdf
- SHA-256:
  de458e2701a1a5aed85f9381b2205ffdac3e80d3aad4588a8545dfc863a4cd04
- size: 552716 bytes
- body: 9 pages
- references begin: page 10
- total: 31 pages
- page size: US Letter
- undefined references/citations: 0
- worst overfull box: 0 pt
- PDF author metadata: none
- Type 3 fonts: 0
- unembedded fonts: 0

Figure 1:

- path: paper-2027/figs/fig_evidence_overview.pdf
- SHA-256:
  97f6e72a109250cc27742d8765e1c9e053e0925edc885e74fe2a5e4ec049fae7
- size: 72764 bytes

Immutable baseline:

- path: paper/main.pdf
- SHA-256:
  fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772
- size: 1393787 bytes
- status: unchanged

## 6. Validation receipts

Passed locally on the personal PC:

- paper-2027/compile.sh after the paper-identity repair: all ICLR format gates
  passed;
- title/abstract/Figure 1/Contributions reviewer smoke test: the first recovered
  identity is the finite-table support--allocation decomposition and spectral
  budget, not a frequency-reallocation method;
- Figure 1 owner assertions and vector-PDF generation;
- final visual review of the abstract/intro, Figure 1, zero-training table,
  body closure, the assumption-bound derivation, and the expanded appendix
  evidence map;
- 21/21 standard-library navigation/workspace tests;
- supplement static-theory allowlist-closure regression: 1/1;
- curated ICLR supplement packaging, leak scan, archive inspection, and
  isolated 9-body/31-total-page compilation;
- git diff --check;
- source/PDF synchronization at the hashes above;
- immutable paper/main.pdf hash unchanged.

Skipped on this PC:

- canonical aidemo Python/PyTorch/pytest suites;
- final cross-environment release receipt.

Compilation certifies format and source/PDF health, not scientific truth.
Scientific quantities were checked against their canonical owners during the
rewrite; the final author number read-through remains a separate release gate.

## 7. Supplement state

The tracked root archive
rope-spectral-budget-iclr2027-supplement.zip was rebuilt from the pre-repair
allowlist and manuscript source on 2026-08-29:

- SHA-256:
  6ea9fd61c12e9ed7d04926c03693a78e968644f00a2043f04dd96e60585b561b
- size: 992809 bytes
- status: its prior leak scan and isolated build passed, but the archive is no
  longer source-synchronised after this framing repair; rebuild it before the
  final work-machine cross-environment receipt and OpenReview upload

The curated allowlist now includes the minimal reviewer-facing static-theory
reproduction closure:

- analysis/full_rope_audit/verify_core.py
- analysis/full_rope_audit/verify_small_models.py
- scripts/analysis/third_axis_ceiling.py

The whole audit directory is intentionally excluded because it contains
non-reviewer historical material. The local package contains both new body
tables and the minimal static-theory reproduction closure. Re-run the same
curated packager and cross-environment checks on the work machine before final
OpenReview upload.

## 8. Remaining action queue

| Order | Action | Exit condition |
| --- | --- | --- |
| 1 | Current framing-repair disposition | author reviews the local PDF; commit/push only under explicit authorisation |
| 2 | Rebuild curated supplement | package contains the repaired source and passes leak/isolated-build checks |
| 3 | Work-machine validation | canonical aidemo changed-path tests and supplement cross-build pass or are explicitly dispositioned |
| 4 | 9/17 abstract/metadata freeze | title, abstract, author roster, profiles, quotas, and AI statement confirmed |
| 5 | 9/25 full-paper freeze | owner-by-owner number read-through, final PDF/supplement approval, upload and downloaded-platform readback |

The two new body table files,
tables/table_zero_training_system.tex and
tables/table_allocation_routes.tex, are required manuscript sources and are
tracked in checkpoint `09876d5`; a clean-checkout build must never omit them.

## 9. Research boundary

No submission experiment is active, and the current manuscript is frozen
against non-essential narrative redesign. Further from-scratch scaling is closed:
the completed 1.485B same-initialisation/same-scientific-recipe comparison is
the current from-training ceiling. No larger from-initialisation run or new
seed program is planned or authorised.

The author-ordered post-submission priority is frozen-checkpoint zero-training
allocation optimization first and matched LoRA work second. The first target
is a single allocation that itself improves both Native-window and long-range
capability, rather than preserving the former only through routing.
Matched-content phase 2x2 or band attribution is used only when the diagnostic
would change the candidate design; it is not the automatic first task. These
are research targets, not claims in the current manuscript, and require
separate design, readiness, and compute authorization. Durable ordering lives
in [`../INDEX.md`](../INDEX.md) §6.

## 10. Author actions

Before 9/17:

- freeze the author roster and confirm current OpenReview profiles;
- confirm author-count/submission quotas and reciprocal-review eligibility;
- confirm that the AI-use statement remains literally complete;
- approve the exact title and abstract entered in OpenReview.

Before 9/25:

- decide the NeurIPS-outcome citation/distinctness branch using
  [CHANGES_FROM_NEURIPS2026.md](CHANGES_FROM_NEURIPS2026.md);
- complete the final owner-by-owner number review after layout freezes;
- approve the uploaded PDF/supplement and downloaded-platform readback.

Before any authorised Git publication, recheck branch, upstream, divergence,
worktree, staged scope, sensitive content, and immutable paper/. Record
local/remote SHAs only after a successful ordinary push.
