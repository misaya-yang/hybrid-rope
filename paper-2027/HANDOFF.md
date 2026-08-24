# ICLR 2027 active handoff

- **Updated:** 2026-08-24
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Published content commit:** `79aa93218154a959afc979476c4de004316f16bf`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** manuscript, owner repairs, reviewer-facing wording pass, PDF, and
  anonymous supplement are validated, committed, and pushed. This is Git
  publication only; no OpenReview upload, GPU run, or submission is implied.
- **Internal only:** exclude this file from the anonymous supplement.

## 1. Cold-start order

1. [`../AGENTS.md`](../AGENTS.md) — stable scientific, safety, naming, compute,
   Git, and delivery rules.
2. This file — current manuscript, validation, worktree, and next actions.
3. [`main.pdf`](main.pdf) and the corresponding `sections/`, `appendix/`, and
   `tables/` sources — reviewer-visible truth.
4. [`research/README.md`](research/README.md) — sole claim/evidence router.
5. The canonical owner named there before changing any claim or number.

`CHANGES_FROM_NEURIPS2026.md` is a historical migration snapshot, not current
state. External reviews, audits, historical handoffs, scripts, preflights, and
filenames may locate a question; they never supersede the current PDF/source or
canonical owner.

## 2. Current paper contract

The paper tells one story:

1. geometric RoPE spends a finite set of rotary pairs redundantly at the slow
   end;
2. `x_k=-log(omega_k)=a+Rz_k` separates sampled support `(a,R)` from interior
   allocation `z`, and a three-seed fixed-support intervention identifies `z`;
3. full sin/cos geometry, the exact effective-rank identity, the frozen
   transplant obstruction, and the 50M weights-by-table crossing explain the
   budget and co-adaptation;
4. EVQ-Cosh is one closed-form, zero-learned-parameter construction on that
   axis, with the Cosh family exact only for the stated convex surrogate and
   its direction checked under the exact cosine-feature kernel;
5. 432M MLA, 454M range composition, 750M continuation, and 1.485B
   from-initialisation studies supply architecture, training-stage, and scale
   consequences; 1.485B and 8B adaptation supply capability evidence.

Evidence roles remain distinct:

- 151.9M exact-range and M4 own pure fixed-support allocation identification;
- the 50M crossing owns weights/table co-adaptation;
- 432M MLA is the scarce-channel systems flagship;
- 454M owns substrate-dependent leverage of the same `YaRN-style` operator;
- 750M and 1.485B own full-parameter persistence/crossover;
- frozen OLMo/Qwen controls are a compact fixed-support corollary, not a second
  method or proof of profile-detail uniqueness;
- 1.485B/8B adaptation, QA, RULER, probability, and causal source use retain
  their protocol-specific endpoint identities.

The latest wording pass removed repetitive self-disqualification without
weakening these boundaries. Negative constructions are judged semantically:
strong distinctions such as “not a disguised base change” remain, while
duplicated disclaimers are stated positively or removed. Theorem and protocol
scope stays beside the claim it governs; no synthetic limitations inventory or
lexical `not` ban is used.

Use the locked names in `AGENTS.md`: `Geo`, `Native`, `FMRoPE`, `anchored
EVQ-Cosh`, `YaRN-style`, cited `YaRN`, and `MLA wavelength-blend operator` are
not interchangeable.

## 3. Current paper and package receipt

- Title: *RoPE Has a Spectral Budget*.
- Main text ends on page 9; total PDF length is 31 US-Letter pages.
- `paper-2027/main.pdf`
  - SHA-256: `e6fa28feeebed65b7e47ade034d3ea9ea2b1e768d9ebc7e66e44529bbc71f564`
  - size: `699603` bytes
- `rope-spectral-budget-iclr2027-supplement.zip`
  - SHA-256: `63144b104c727dcd45f1230eeeb782a7a44173177cdad08791ffbed9ea774fd9`
  - size: `862842` bytes
- Immutable `paper/main.pdf`
  - SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

The AI-use statement has author-confirmed factual coverage. Do not shorten or
cosmetically rewrite it without renewed author confirmation and a current venue
policy check.

## 4. Latest validation

The 2026-08-24 wording/package pass established:

- `./paper-2027/compile.sh`: body page 9, 31 total pages, zero undefined
  references/citations, `0pt` worst overfull box, anonymous, Letter, no Type-3
  or unembedded fonts;
- curated supplement: ZIP integrity clean and isolated paper rebuild passed;
- isolated packaged CPU suite: `144/144` tests passed in Conda `aidemo`;
- repository navigation, downstream-helper, and same-support focused suite:
  `22/22` tests passed;
- pages affected by the wording pass (2, 6--9, 14, 18, 23, and 29--30) were
  visually inspected;
- `git diff --check` passed;
- `paper/` remained unchanged at the hash above.

These checks establish build/package health, not scientific acceptance,
OpenReview upload, or publication.

## 5. Workspace and ownership

The published set spans the manuscript, research owners/receipts, evaluator
guard/tests, packager, curated supplement, and documentation cleanup. Preserve
unrelated edits and inspect branch, upstream, and worktree live before any
future Git operation; this file intentionally avoids a self-invalidating current
HEAD field.

The 2026-08-24 documentation cleanup removed the unreferenced tracked
`DOCUMENT_TEXT_MAP.md`, which duplicated the LaTeX manuscript, and marked
`CHANGES_FROM_NEURIPS2026.md` as a historical snapshot. Ignored LaTeX scratch
files, `.DS_Store`, and the unreferenced stray page image were moved to the
recoverable sibling archive `../hybrid-rope-cleanup-20260824/`. The stale,
unreferenced root `REAL_CONTEXT_TARGET_FREE_PREFLIGHT_20260822.md` was removed;
its historical content remains recoverable from Git.

Durable placement:

| Material | Owner |
| --- | --- |
| Current mutable state | this file only |
| Claim/evidence routing | `research/README.md` |
| Central paper-facing owners | research root files named by the router |
| Mature retrofit results and receipts | `research/attention-aware-retrofit/` |
| Internal theory/manuscript audits | `research/audits/` |
| External-model reviews | `research/external-reviews/` |
| Reviewer-facing source/package | `paper-2027/` and curated packager output |

Raw checkpoints, GPU rows, caches, server details, and private paths remain
outside the repository. Plans, scripts, commands, and preflights are not
results.

## 6. Next actions

Highest-leverage author actions:

1. Read the final nine-page PDF as a fresh ICLR reviewer, especially the
   abstract, Figure 1, pages 6--9, and the transition into Discussion.
2. Recheck live ICLR policy, deadlines, dual-submission state, and author
   profile requirements immediately before submission.
3. Confirm OpenReview title and abstract exactly match the final PDF.
4. Keep future Git publication approval-gated and repeat the scoped
   staging/leak review.

Stop list:

- no new table, gain, beta, rank, step-count, or RULER sweeps for the current
  submission;
- no revival of revoked source-selection or CE-only far-pass protocols;
- no promotion of the old aliased Qwen `0.6175` result;
- no merging of exact-range, co-adaptation, frozen retrofit, and mature
  capability into one causal estimand;
- no deletion of correct appendix theory to reduce total PDF length;
- no edit, compile, move, or regeneration of `paper/`;
- no GPU work, commit, push, branch operation, or upload without explicit user
  authorization.

## 7. Known open state

- The current repair/manuscript/package set was published to `main_0726`.
- Raw GPU artifacts remain external; compact tracked receipts own only their
  stated hashes and metrics.
- No promoted scientific result is known incomplete. Remaining work is author
  review, live submission-policy verification, and explicitly authorized Git
  publication.
