# ICLR 2027 active handoff

- **Updated:** 2026-08-26
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Repository HEAD:** `de6aee99a3dce04051f87a2a8d9a2caa84fe0bd8`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** an uncommitted whole-paper acceptance-first revision is active and
  locally validated. It foregrounds the third finite-table allocation
  coordinate, leads the empirical section with the mature zero-training
  pure-`z` intervention, separates likelihood from capability evidence, and
  removes the old repository-defined 454M/125M range-composition line from
  reviewer-facing inputs. No commit, push, upload, or GPU work is authorised.
- **Internal only:** exclude this file, the narrative guide, revision plans, and
  the Codex/Claude review log from the anonymous supplement.

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
4. derive EVQ-Cosh as one closed-form witness for its stated convex surrogate;
5. present consequences in the author-specified order: zero-training mature
   checkpoint, matched low-rank adaptation, then from-training/co-adaptation;
6. close on allocation as a broader RoPE design coordinate, not an
   extrapolation-only recipe.

Locked scientific identities:

- `FMRoPE` is published related work and the paper-faithful fixed-support
  training control. The target-matched policy remains in Appendix C as protocol
  context and must not become the body narrative.
- `anchored EVQ-Cosh` changes only interior allocation at the FMRoPE extrema and
  log-span.
- `Geo` is a geometric training baseline; `Native` is an unmodified pretrained
  checkpoint.
- The zero-training YaRN comparator is the verified Hugging Face Transformers
  implementation at factor four.
- The mature zero-training geometric/derived/coarse-ramp profiles follow the
  movement-profile construction and are **not** EVQ-Cosh.
- EVQ-Cosh is unique only for its stated convex surrogate. Static effective
  rank diagnoses table geometry; it does not rank trained LM quality.

## 3. Evidence order and claim ownership

| Reviewer-facing role | Headline result | Owner |
| --- | --- | --- |
| mature pure-`z` consequence | OLMo 16K RULER `0.56% -> 60.47%`; coarse ramp `61.04%`; Qwen 64K `57.75% -> 66.50%`, all at matched support | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| zero-training deployment breadth | 4K FineWeb-Edu `+0.1236` NLL; PG-19 and RULER-13 at 8/16K; full-200 Qasper and 2Wiki task F1 | [`attention-aware-retrofit/README.md`](research/attention-aware-retrofit/README.md) and routed owners |
| matched adaptation | 1.485B task-family transfer; 8B remote-source deletion changes NLL by `+1.5055` in the EVQ arm | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md) and [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) |
| fixed-support training identification | `+0.026/-0.281/-0.176/-0.146` NLL; every OOD length favours the reallocation in `3/3` seeds | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| full-pair static geometry | 23 slow pairs / 46 nominal dimensions / `r2=2.00` under the stated prior | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819`](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| architecture / scale / modality breadth | 432M MLA three-seed, 750M continuation, 1.485B early-training crossover, and two-seed video DiT | [`research/README.md`](research/README.md) and its routed owners |

Keep likelihood and capability endpoints separate. Do not translate NLL/PPL
changes into percentage capability claims. Do not pool uncertainty units across
training seeds, evaluation rows, deterministic frozen interventions, or single
matched trajectories.

## 4. Current source and PDF receipt

- Main body: 8 pages; references begin after the required statements.
- Total PDF: 28 US-Letter pages.
- `paper-2027/main.pdf`
  - SHA-256: intentionally not frozen during alternating review; both reviewers
    rebuild and the embedded build timestamp changes the bytes. Recompute after
    the final pre-freeze build.
  - latest observed size: `434101` bytes
- Immutable `paper/main.pdf`
  - SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

The active PDF and source are uncommitted working-tree artifacts. Do not confuse
this receipt with the older published checkpoint or curated supplement receipt.

Latest `./compile.sh` validation:

- 8 body / 28 total pages;
- zero undefined references or citations;
- `0pt` worst overfull box;
- anonymous US-Letter output with `\iclrfinalcopy` disabled;
- no Type-3 or unembedded fonts;
- full body pages 1--8 visually inspected;
- latest affected appendix pages 20 and 26 visually inspected after removal of
  stale 125M/454M labels and promotion of the mature result table;
- `git diff --check` passes;
- immutable `paper/main.pdf` hash unchanged.

This receipt proves only current build/layout health. It does not import the
older package's test counts, prove acceptance, or imply an OpenReview upload.

## 5. Current edit state

Implemented:

- Abstract and Introduction lead with the third coordinate and the mature
  same-support pure-`z` result.
- Figure 1 carries allocation, exact geometry, and three-seed causal
  identification; do not restore the old multi-protocol montage.
- The forced page break after Introduction has been removed; Section 2 now
  begins on page 2 instead of leaving a large blank region.
- Identification has one body job: the 151.9M fixed-support causal owner, plus a
  compact 50.9M configuration/shape breadth sentence.
- Theory uses the full sin/cos positional object, the exact spectral-budget
  identity, slow collapse, co-adaptation, transplant obstruction, and a bounded
  Cosh surrogate. The movement-profile equation specifies mature derived/ramp
  allocation separately.
- Experiments are ordered zero training, matched adaptation, from training.
  Mature likelihood and task capability are presented in separate paragraphs.
- The body now includes one compact mature-checkpoint result table. It separates
  Native/official Transformers YaRN references from the matched-support pure-`z`
  block and carries the confirmation-only OLMo protocol.
- Related Work now cites Jet-Long as bifocal range transport and distinguishes
  learned use of a supplied geometric grid from changing interior allocation.
- Discussion presents LeRoPE positively as a complementary learned route and
  keeps `z` broader than extrapolation.
- The old repository-defined 454M/125M range-composition material is absent
  from compiled inputs. Its source evidence remains in the repository but must
  not be relabelled as standard YaRN or restored without a new author decision.
- The ICLR supplement allowlist also excludes the retired 454M/125M figures,
  generator, orphaned appendix/table source, and five obsolete curated
  artifacts. The existing ZIP has not been rebuilt during this revision.
- The 1.485B early-training crossover figure remains once in Appendix E/F; it is
  not a body scale claim and must not be duplicated.

No page filler, duplicate lifecycle figure/table, new experiment, or new theorem
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
