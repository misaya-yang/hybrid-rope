# ICLR 2027 active handoff

- **Updated:** 2026-08-28
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Manuscript/provenance checkpoint:** `001a900702c50a301c1970fe23b8f7aaa610732e`
  (`paper: center ICLR narrative on allocation interventions`); use `git rev-parse
  HEAD` for the receipt-only commit that carries this handoff line.
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** the reviewer-facing narrative is converged around one paper
  identity: fixed-support interior allocation `z` is a causal RoPE design
  coordinate. Figure 1 carries definition, geometric consequence, and causal
  identification. Figure 2 makes the mature frozen pure-`z` intervention
  visually immediate, and Figure 3 separates the 8B adaptation route's remote
  routing and source-deletion endpoints. The empirical section poses three
  parallel intervention questions for fully frozen, low-rank-adapted, and
  from-training models.
  Theory explains the empirical thesis, and EVQ-Cosh remains the closed-form
  construction on the coordinate. The target-retargeted ordering reversal is
  stated once in the body and fully documented in Appendix C. The checkpoint
  above remains the latest published manuscript; the current worktree adds a
  uncommitted provenance/citation patch plus a reviewer-facing narrative pass.
  The latter strengthens active claim language and puts decisive long-context
  results before their retained near-window costs without changing any value,
  protocol, or evidence role. The body now uses the ninth-page budget for the
  cross-configuration allocation-family result and for explicit interpretation
  of the scarce-channel, 750M, 1.485B, and video-DiT evidence. References begin
  on page 10. No upload, GPU work, commit, or push is
  authorised by this pass.
- **Internal only:** exclude this file, the narrative guide, revision plans, and
  the Codex/Claude review log from the anonymous supplement.
- **Concurrent routing edits:** during the 2026-08-27 narrative pass, separate
  uncommitted changes appeared in `INDEX.md` and several `research/` routing
  documents that promote `REVISION_BRIEF.md` as an execution input. This pass
  did not create or alter those changes. Under `AGENTS.md`, the Qwen bundle
  remains external review input and this handoff remains the sole current
  action queue until the author reconciles that routing explicitly.

## 1. Cold start and authority

From the repository root, inspect before changing Git state:

```bash
git status --short --branch
git rev-list --left-right --count HEAD...origin/main_0726
git log -1 --oneline --decorate
```

Do not pull, clean, reset, stash, checkout, commit, or push over the current
dirty worktree without explicit author direction.

Read in this order:

1. [`../AGENTS.md`](../AGENTS.md) -- rules and claim ceilings.
2. [`../INDEX.md`](../INDEX.md) -- canonical theory, evidence, and owner routing.
3. [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md) -- mandatory manuscript strategy
   and revision guardrails; it is not a numerical owner.
4. [`research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`](research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md)
   -- append-only alternating-review journal and author corrections.
5. This file -- live manuscript state and validation receipt.
6. [`main.pdf`](main.pdf) and `sections/` -- reviewer-visible truth after the
   latest clean build.
7. The canonical owner routed by [`research/README.md`](research/README.md)
   before changing a fact, number, protocol identity, theorem, or claim ceiling.

Build and test commands live in [`../README.md`](../README.md), under "Build and
validate." Never compile or modify `paper/`.

## 2. Manuscript contract

Title: *RoPE Has a Spectral Budget*.

Reviewer memory:

> A finite RoPE table is not exhausted by base or range. At fixed sampled
> support, its interior allocation `z` selects the effective positional basis
> realised by a finite rotary budget and has large, controlled behavioural
> consequences.

Reader path:

1. write `x_k = a + R z_k` and expose support, span, and allocation as distinct
   finite-table coordinates;
2. show the spectral-budget phenomenon with full sin/cos geometry and the
   `46 nominal dimensions -> r2=2.00` hook;
3. identify `z` at fixed support with the paired 151.9M three-seed training
   intervention;
4. derive EVQ-Cosh as a closed-form construction for its stated convex surrogate;
5. present consequences in the author-specified order: zero-training mature
   checkpoint, matched low-rank adaptation, then from-training/co-adaptation;
6. close on allocation as a broader RoPE design coordinate, not an
   extrapolation-only recipe.

Locked scientific identities:

- `FMRoPE` is published related work and the paper-faithful fixed-support
  training control. The body states once that support retargeting reverses the
  tested ordering; Appendix C owns all numbers, per-seed detail, and protocol
  interpretation. This is a support--allocation result, not a method rivalry.
- `anchored EVQ-Cosh` changes only interior allocation at the FMRoPE extrema and
  log-span.
- `Geo` is a geometric training baseline; `Native` is an unmodified pretrained
  checkpoint.
- The zero-training YaRN comparator is the verified Hugging Face Transformers
  implementation at factor four.
- The mature zero-training uniform/coarse/derived allocations follow Eq.~(2),
  are three values of the same fixed-support `z` coordinate, and are **not**
  EVQ-Cosh.
- EVQ-Cosh is unique only for its stated convex surrogate. Static effective
  rank diagnoses table geometry; it does not rank trained LM quality.

## 3. Evidence order and claim ownership

| Reviewer-facing role | Headline result | Owner |
| --- | --- | --- |
| mature pure-`z` consequence | OLMo 16K RULER `0.56% -> 60.47%`; coarse label-free control `61.04%`; Qwen 64K `57.75% -> 66.50%`, all at matched support | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| zero-training deployment breadth | 4K FineWeb-Edu `+0.1236` NLL; PG-19 and RULER-13 at 8/16K; full-200 Qasper and 2Wiki task F1 | [`attention-aware-retrofit/README.md`](research/attention-aware-retrofit/README.md) and routed owners |
| matched adaptation | 1.485B task-family transfer; 8B remote-source deletion changes NLL by `+1.5055` in the EVQ arm | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md) and [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) |
| fixed-support training identification | `+0.026/-0.281/-0.176/-0.146` NLL; every OOD length favours the reallocation in `3/3` seeds | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| full-pair static geometry | 23 slow pairs / 46 nominal dimensions / `r2=2.00` under the stated prior | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819`](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| architecture / scale / modality breadth | 432M MLA three-seed, 750M continuation, 1.485B early-training crossover, and one matched video-DiT comparison | [`research/README.md`](research/README.md) and its routed owners |

Keep likelihood and capability endpoints separate. Do not translate NLL/PPL
changes into percentage capability claims. Do not pool uncertainty units across
training seeds, evaluation rows, deterministic frozen interventions, or single
matched trajectories.

## 4. Current source and PDF receipt

- Main body: 9 pages; references begin on page 10 after the required statements.
- Total PDF: 28 US-Letter pages.
- `paper-2027/main.pdf`
  - SHA-256: `ade3beb59eefca91b339ef4d6731cdda19fa8eb40f384d3eab60fd9c4df06f01`
  - final observed size: `757522` bytes
- Immutable `paper/main.pdf`
  - SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

The active PDF, source, and curated supplement include the uncommitted patch
described in the status block. The manuscript checkpoint above remains the
latest Git-published version.

Latest `./compile.sh` validation:

- 9 body / 28 total pages; References starts on page 10;
- zero undefined references or citations;
- `0pt` worst overfull box;
- anonymous US-Letter output with `\iclrfinalcopy` disabled;
- no Type-3 or unembedded fonts;
- body pages 2--3 and 7--9, reference page 10, and appendix pages 26--28
  visually inspected after the current patches;
- current navigation and ICLR-supplement contract checks pass `43/43` under
  Conda `aidemo`; the earlier code/provenance suite remains `181/181`, with no
  code changed in this narrative pass;
- curated supplement rebuilt, ZIP SHA-256
  `a270d8f9a85184b0e6be57bf16e9054a7258cc3567302f9e6ee6319ee0e52e08`;
- isolated supplement compile passes at 9 body / 28 total pages, and its
  allowlisted test suite passes `144/144` under Conda `aidemo`;
- `git diff --check` passes;
- immutable `paper/main.pdf` hash unchanged.

This receipt proves only current build/layout health. It does not import the
older package's test counts, prove acceptance, or imply an OpenReview upload.

## 5. Current edit state

Implemented:

- Abstract and Introduction now state the construction actively: optimising
  the convex allocation surrogate yields the closed-form EVQ-Cosh table, with
  uniqueness scoped once to that surrogate. The abstract retains one
  performance-number group and the Contributions remain three auditable claims.
- Section 4 now leads each route with its decision-relevant long-context or
  causal result. All near-window values remain in the same paragraph, but
  follow the result they contextualise; Discussion owns the cross-protocol
  crossover interpretation. Supporting single-seed identities remain in
  appendix protocol/table captions rather than becoming body warnings.
- Appendix E now defines the label-free model-relative ramp projection,
  including the OLMo/Qwen transition-pair bounds, and records that the
  nine-task OLMo set was evaluated only after core-four method selection was
  frozen. The body adds only an appendix pointer; it does not expand the
  protocol into a second results narrative. The anonymous curated receipt
  carries the same split and task-selection provenance.
- The mature frozen-table displacement equation and model-specific construction
  now live where they are used: Section 4.1 gives the fixed-support map and
  Appendix E.2 owns the residual-energy and coarse-ramp details. Theory keeps a
  one-sentence interface before proceeding directly to the full sin/cos basis.
- The former body Table 2 is now Figure 2, a compact OLMo/Qwen double panel.
  Grey reference rows and the shaded matched-support pure-`z` block remain
  visually and causally distinct; exact values and evaluation-row intervals
  remain in the caption and Appendix E.2.
- Figure 3 gives the other high-value empirical intervention that prose alone
  obscured: the matched 8B LoRA model routes more strongly to the remote source,
  and position-preserving deletion of that source changes its answer NLL. The
  two panels share one protocol and do not pool metrics or uncertainty with the
  frozen or from-training routes.
- Appendix A cites Gray and Neuhoff's quantisation survey for the
  Bennett-integral high-resolution expansion. The 750M single-seed identity
  remains in Appendix Table 6 rather than being repeated in the body; the
  three-seed causal anchor remains foregrounded.
- The abstract is 164 source words and carries one numerical result group: the
  mature fixed-support RULER intervention. It opens with the field-level
  base/range/phase-transport framing and the allocation coordinate that this
  framing leaves implicit. Geometry, replication, construction, and lifecycle
  breadth are stated without a second results ledger.
- Abstract and Introduction lead with the third coordinate and the mature
  same-support pure-`z` result. The 30-second path is now explicit: isolate
  `z`, show the spectral-budget consequence, identify pure-`z` behaviour,
  construct a table, and test the coordinate at frozen, adapted, and
  from-training stages.
- Figure 1 carries the coordinate definition, exact geometry, and three-seed
  causal identification; its caption makes those evidence roles explicit.
- The forced page break after Introduction has been removed; Section 2 now
  begins on page 2 instead of leaving a large blank region.
- The forced page break before the bibliography is also removed. The required
  statements retain their full text, the body-end label falls on page 9, and
  the References heading follows them on page 10 instead of creating a
  one-line statement page and delaying references to page 11.
- Identification has one body job: the 151.9M fixed-support causal owner, plus a
  compact 50.9M configuration/shape breadth paragraph. It now explains why the
  Cosh strengths and deformation-matched exponential support an allocation
  family rather than a single privileged curve, while keeping the 151.9M
  intervention as the effect-size owner. One additional sentence
  reports that support retargeting reverses the tested ordering, establishing
  support and allocation as distinct but interacting design coordinates.
- Theory opens from the empirical result and uses the full sin/cos positional
  object, the exact spectral-budget identity, slow collapse, co-adaptation,
  transplant obstruction, and a bounded
  Cosh surrogate. The frozen derived profile and coarse label-free control now
  live in Section 4.1 and Appendix E.2, while Theory keeps only their interface
  to the same `z` coordinate. Theory begins on page 3 and
  closes on page 5 by stating the full basis-allocation loop explicitly.
- The empirical section is titled around the allocation coordinate and poses
  three parallel questions: whether `z` changes a fully frozen model, whether
  pretrained representations exploit it with limited adaptation, and whether
  weights learn it through from-training co-adaptation. Mature likelihood and
  task capability remain separate. Its from-training route now states why the
  432M MLA result is a scarce-budget test, distinguishes 750M capability
  conversion from 1.485B natural-text scale persistence, and uses the video-DiT
  in-window result to expose allocation beyond autoregressive extrapolation.
- The body now includes a compact mature-checkpoint result figure. It separates
  Native/official Transformers YaRN references from the matched-support pure-`z`
  block, carries the confirmation-only OLMo protocol, and appears after its
  explanatory paragraph. The co-adaptation table likewise stays inside its own
  subsection rather than floating ahead of the spectral-budget theorem.
- Related Work is a 171-word late-body section after Experiments. It groups
  range/phase methods rather than teaching their implementation names, retains
  the nearest citations, and leaves the Identification-to-Theory transition
  uninterrupted.
- Discussion unifies the three routes as interventions on one coordinate at
  different model-building stages. It retains extrapolation as the sharpest
  present identification setting and LeRoPE as the complementary in-window
  direction toward joint native/long allocation.
- The old repository-defined 454M/125M range-composition material is absent
  from compiled inputs. Its source evidence remains in the repository but must
  not be relabelled as standard YaRN or restored without a new author decision.
- The ICLR supplement allowlist also excludes the retired 454M/125M figures,
  generator, orphaned appendix/table source, and five obsolete curated
  artifacts. The curated ZIP was rebuilt after the revision; SHA-256
  `a270d8f9a85184b0e6be57bf16e9054a7258cc3567302f9e6ee6319ee0e52e08`.
- The video-DiT breadth result is now the raw-backed seed-42 head-to-head only;
  the unreceipted second-seed and base-1000 rows are absent from reviewer-facing
  source. Its tracked JSON and canonical owner are routed by `INDEX.md` and the
  supplement README.
- The 1.485B early-training crossover figure remains once in Appendix E; it is
  not a body scale claim and must not be duplicated.
- The exact identity now explains the title directly: fixed
  `tr(Gamma)=2K` conserves nominal rotary dimension while allocation changes its
  distribution across positional directions.
- Discussion adds the practical reporting consequence that finite RoPE tables
  require both sampled support and interior allocation to define a matched
  design.
- A `FloatBarrier` after the Appendix E 1.485B figure prevents the following
  frozen-checkpoint subsection from overtaking the float. Page 26 now starts
  that subsection after the complete figure; page 28 carries the final MLA
  table and interpretation. Its remaining tail space is the natural end of the
  scientific appendix, not a misplaced float or page-filling target.

No page filler, cross-protocol lifecycle ledger, new experiment, or new theorem
is needed merely to reach nine pages. Add material only when it closes a real
scientific or reviewer-path gap.

## 6. Alternating Codex / Claude Code review

Both reviewers use
[`research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`](research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md).

- Read the entire shared log before editing.
- Append, never overwrite, using `Codex:` or `Claude Code:`.
- Record observation, evidence/owner, decision, changed files, verification,
  and remaining question.
- Adopt the other reviewer's point only after checking the current manuscript
  and canonical owner. Explain disagreements in the same log.
- After every manuscript change, compile and visually inspect affected pages.
- Preserve the user's manuscript strategy and the documented Codex failure
  corrections in the shared log.

## 7. Research continuation and stop conditions

The research agenda is not state; it lives in [`../INDEX.md`](../INDEX.md) §6.
This section records only the current stop boundary.

The generic zero-parameter target-free operator was implemented and evaluated,
but the tested continuous-boundary version did not produce a positive RULER
result. It is future research, not current manuscript evidence. Other
post-submission theory and experiment routes remain governed by `INDEX.md` and
their owners.

Do not start training, GPU evaluation, a new baseline, or a new target-free
search without explicit author authorisation. Do not commit, push, upload, or
alter Git history. Stop and return to the owner if a proposed edit changes a
number, merges protocols, promotes unfinished evidence, or requires a stronger
claim than the current owner supports.

## 8. Git boundary

Before any future authorised commit/push:

- preserve `paper/`;
- stage explicit paths, never `git add -A`;
- exclude credentials, machine paths, checkpoints, raw rows, caches, and
  unrequested build products;
- run cached-diff and sensitive-scope checks;
- ordinary-push only, then compare local and remote SHA.
