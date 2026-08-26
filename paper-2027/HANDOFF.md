# ICLR 2027 active handoff

- **Updated:** 2026-08-26
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Published manuscript checkpoint:** `06ad6262c1b3cacecc9f8f9b0d2c764c884eed26`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** reviewer-path narrative, citation/novelty, and appendix-evidence
  passes are implemented and locally validated. The paper now cites the direct
  support/frequency neighbours, adds exact protocol provenance, and includes
  appendix figures for the three-seed exact-range control, the 1.485B
  full-parameter crossover, and range composition. Figure 1 now visualizes the
  allocation coordinate, the exact slow-block spectrum, and its fixed-support
  training identification rather than a multi-protocol result montage. No new
  experiment or OpenReview upload is implied.
- **Internal only:** exclude this file from the anonymous supplement.

## 1. PC cold start

From the repository root:

```bash
git fetch origin main_0726
git status --short --branch
git rev-list --left-right --count HEAD...origin/main_0726
git log --oneline -5
```

Do not pull over a dirty worktree. If the PC checkout is clean and only behind,
run `git pull --ff-only origin main_0726`; otherwise inspect ownership before
changing Git state. After the update, `git rev-parse HEAD` must contain the
published manuscript checkpoint above or a documented descendant.

Read in this order:

1. [`../AGENTS.md`](../AGENTS.md) — rules and claim ceilings.
2. [`../INDEX.md`](../INDEX.md) §2, §3.4, §6, §7 — theory, closed routes,
   agenda, and cross-machine boundaries.
3. This file — current manuscript, validation, and author actions.
4. [`main.pdf`](main.pdf) and `sections/` — reviewer-visible truth after a
   clean local build.
5. [`research/ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md`](research/ICLR2027_NARRATIVE_OPTIMIZATION_PLAN_20260826.md)
   — current 9-page narrative edit order for Codex.
6. [`research/ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md`](research/ICLR2027_CITATION_NOVELTY_AUDIT_20260826.md)
   — related-work / novelty Codex patch; do not restore the NeurIPS PE zoo.
7. [`research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
   — post-submission theory continuation.
8. The canonical numerical owner routed by
   [`research/README.md`](research/README.md) before changing a claim.

Build and test commands live only in [`../README.md`](../README.md)
“Build and validate.” Python/PyTorch/pytest use Conda `aidemo`. Never compile
`paper/`.

## 2. Final manuscript contract

Title: *RoPE Has a Spectral Budget*.

Reviewer path:

1. finite RoPE has sampled support and interior allocation;
2. Section 2 identifies allocation at fixed support with three training seeds;
3. Related Work positions that axis;
4. full-pair geometry supplies the exact effective-dimension account and slow
   collapse;
5. EVQ-Cosh is one analytic, zero-learned-parameter witness;
6. architecture, scale, frozen-checkpoint, and capability studies establish
   relevance without replacing the identification owner.

Locked identities:

- `FMRoPE` is the published rule, instantiated at fixed training support in
  the identification experiment; the target-retargeted policy is reported
  separately.
- `anchored EVQ-Cosh` changes only interior allocation at the same FMRoPE
  extrema and log-span.
- `Geo` is a geometric training baseline; `Native` is an unmodified
  pretrained checkpoint.
- `YaRN-style` is the repository fixed-index operator, not official YaRN.
- EVQ-Cosh is unique only for its stated convex surrogate.

Headline evidence:

| Role | Result | Owner |
| --- | --- | --- |
| causal identification | `+0.026/-0.281/-0.176/-0.146`, every OOD length `3/3` seeds | [exact-range owner](research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) |
| full-pair static geometry | 23 slow pairs / 46 nominal dimensions / `r2=2.00` under the stated prior | [full-RoPE owner](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| scarce-channel relevance | 432M MLA, `K=16`, 16K PPL `138.8 -> 95.6`, three seeds | [curated owner](../data/curated/table18_mla_3seed_aggregate.json) |
| mature fixed-support consequence | OLMo 16K RULER `0.56% -> 60.47%`; coarse ramp `61.04%` | [same-support owner](research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| pretraining-scale ceiling | full-parameter evidence through 1.485B; 8B is adaptation/capability evidence | [research router](research/README.md) |

## 3. Current paper and package receipt

- Main text: 9 pages.
- Total PDF: 29 US-Letter pages.
- `paper-2027/main.pdf`
  - current local SHA-256: `019bcac763be8f2b41e38299617e7a927badf25e3ef865a6d76c614419bc7cbb`
  - size: `696391` bytes
- `rope-spectral-budget-iclr2027-supplement.zip`
  - current local SHA-256: `890920524d849dd688ee551077f0bb72522353ea9b135436ece6f588ce3bd2e8`
  - size: `912678` bytes
- Immutable `paper/main.pdf`
  - SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

Commit `06ad626` publishes the generated PDF, BBL, and curated supplement ZIP
with their source so the PC starts from the exact reviewed artifact. This Git
publication is not an OpenReview submission.

## 4. Latest validation

The bounded manuscript-optimization revision passed:

- active and isolated-package builds: 9 body / 29 total pages;
- zero undefined references or citations;
- `0pt` worst overfull box;
- anonymous Letter output, no Type-3 or unembedded fonts;
- repository navigation, RoPE, exact-range, same-support, and package/evidence
  tests in Conda `aidemo`: `186/186`;
- signed-lag / gap / $k$-way CPU identity checks: `11/11`;
- isolated supplement CPU suite: `144/144`;
- isolated supplement build with the same 9/29 page and format receipt;
- ZIP integrity and immutable-`paper/` SHA-256 check;
- visual review of all 29 pages, including Section 2 on page 3, Theory on page
  4, experiments/discussion on pages 7--9, the five appendix figures, and the
  final two-page composition layout with no orphan table or float-only blank.

The ICLR 2027 Author Guidelines were checked live on 2026-08-26; the unchecked
author-roster, quota, and reciprocal-reviewing gates are recorded in
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md).

These receipts prove build/package health and the tested code paths. They do not
prove acceptance, policy currency, OpenReview state, or unmeasured claims.

## 5. Core PC continuation plan

### Objective

Raise the reviewer score ceiling by making the existing science resolve to one
judgment: RoPE support does not determine how a finite head allocates its
frequency budget. No new experiment is needed for this pass.

The 30-second, 3-minute, and full-paper readings should all recover the same
chain: fixed-support identification, exact full-pair geometry, one analytic
construction, then scale and capability consequences.

### Locked decisions

- Keep the title and section order. Section 2 remains on page 3 and Theory
  begins on page 4.
- Keep Figure 1 as the allocation / collapse / fixed-support identification
  figure. Do not restore the old multi-protocol montage or add frozen OLMo to
  it.
- Keep the abstract near its current length and retain only the central
  `46 nominal dimensions -> 2.00 effective dimensions` numerical hook. Do not
  add the exact-range NLL vector or frozen-checkpoint score ledger.
- Freeze Related Work unless a citation or technical distinction is factually
  wrong. The current classifier and direct-neighbour citations are sufficient.
- Preserve every theorem, claim owner, completed scale result, and sound
  appendix proof. The appendix budget is a ceiling, not a target.

### Execution order

1. **Section 2 framing — highest priority.** In
   [`sections/02_identification.tex`](sections/02_identification.tex), replace
   the two negative question headlines with positive scientific questions:
   allocation is identifiable at fixed support; support and allocation
   interact; the coordinate remains visible after pretraining. Retain the
   target-matched `+0.060/+0.227/+0.460`, `0/3` reversal in the paragraph. Do
   not hide or weaken it.
2. **One flagship scale visual.** Move the existing
   [`figs/fig_olmo_scale_crossover.pdf`](figs/fig_olmo_scale_crossover.pdf)
   figure environment and label from Appendix F into the 1.485B paragraph of
   [`sections/04_experiments.tex`](sections/04_experiments.tex). Keep detailed
   protocol text in Appendix F and do not duplicate the figure.
3. **Pay the page cost by removing repetition.** Shorten
   [`sections/05_discussion.tex`](sections/05_discussion.tex), primarily its
   repeated MLA, frozen, scale, and related-work recap. Preserve the synthesis:
   geometry diagnoses the finite basis, training binds weights to that basis,
   and allocation is a controllable design coordinate. Do not delete a claim
   or number merely to fit the figure.
4. **Final prose polish.** Reserve `consequential` and “not a disguised base
   change” for one decisive use each. Remove repeated defensive formulations,
   but keep the nearest material scope beside the governed claim. Figure 1 may
   receive a shorter poster-readable title; its data and three-panel identity
   stay fixed.

### Scientific gates

Before accepting any rewrite, verify all of the following:

- exact-range remains the sole fixed-support training identification owner;
- target-aware FMRoPE still wins its separate protocol;
- frozen OLMo `derived` and `coarse ramp` are not called EVQ-Cosh;
- static effective rank is not presented as an LM-quality predictor;
- 1.485B is the full-parameter scale ceiling and 8B remains LoRA/adaptation;
- RULER/2Wiki remain task-family adaptation, not unseen-task transfer;
- EVQ-Cosh uniqueness remains conditional on the stated convex surrogate;
- LeRoPE remains compatible evidence, not validation or a matched comparator.

### Acceptance criteria

- body remains 9 pages; Section 2 page 3; Theory starts no later than page 4;
- the 1.485B crossover is visible in Section 4 and absent as a duplicate float
  from Appendix F;
- undefined references/citations `0`; worst overfull at target `0pt`, hard
  ceiling `5pt`;
- US Letter, anonymous, `\iclrfinalcopy` disabled, no Type-3 or unembedded
  fonts;
- visual read of body pages 1--9 confirms Figure 1 is one scientific argument,
  the new scale figure is legible, and no float creates an orphan or large
  blank region;
- curated supplement rebuilds, passes its isolated CPU suite, and contains the
  same source and figure identities;
- `paper/` SHA-256 remains
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.

### Explicit non-goals and stop conditions

Do not add citations, experiments, seeds, scales, appendix filler, priority
claims, new theory, or a second overview figure in this pass. Do not modify
`paper/`, `main.tex`, venue style files, ethics, reproducibility, or AI-use
statements. Stop and return to the numerical owner if a proposed wording needs
a stronger claim, merges protocols, or changes a displayed result. GPU work,
Git history changes, commit/push, and OpenReview upload each require fresh
explicit authorization.

## 6. Research continuation

The research agenda is not state. Its authority is [`../INDEX.md`](../INDEX.md)
§6; the theoretical reasoning is in the
[theory state](research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md).

Current missing bridge: a matched-content `table x virtual-gap position map`
2x2 that keeps tokens, answer, decoder, and rows fixed. Different-length RULER
rows cannot answer position failure versus model capability.

Do not start another shared-table static score, 50M candidate verdict, extra
oracle shell/step/seed/LR sweep, larger checkpoint, or GPU run from this
handoff. Any compute requires a new preflight and explicit user authorization.

## 7. What Git does not contain

| Artifact | Git status | Consequence on the PC |
| --- | --- | --- |
| manuscript source, owners, compact receipts, code, tests | tracked | available after verified sync |
| checkpoints and adapters | external | locate or transfer before evaluation |
| raw GPU rows and per-example outputs | external/ignored | compact means are insufficient for new per-position analysis |
| caches and token arrays | external/ignored | regenerate only from a pinned owner |
| prepared one-billion-token FineWeb-Edu corpus | previously machine-local; portability unverified | do not assume it exists on the PC or a new server |

A compact receipt is provenance, not the raw artifact. Missing local raw files
never mean the experiment did not run.

## 8. Author actions

Submission:

1. Confirm OpenReview title and abstract exactly match the PDF.
2. Recheck current ICLR policy, deadlines, anonymity, author roster/quotas,
   profiles, and dual-submission state immediately before upload.
3. Upload only the curated paper/package, never a repository-root archive.

Research after submission:

1. Read the theory state and the complete closed-route ledger in INDEX §3.4.
2. Write the matched-content phase preflight without selecting on long-context
   results.
3. Request explicit GPU authorization only after code/config/data/table/output
   identities and a stop rule are frozen.
4. Require a 1.485B matched in-window + far-tail + capability gate before
   multi-seed or second-checkpoint expansion.

## 9. Git boundary

At handoff, always report source publication and local build artifacts
separately. Before any future commit/push:

- preserve `paper/`;
- stage an explicit scope, never `git add -A`;
- exclude credentials, machine paths, checkpoints, raw rows, caches, and
  unrequested build products;
- run `git diff --cached --check` and the relevant `aidemo` tests;
- push `main_0726`, then compare local and remote SHA.

Historical experiment decisions remain reachable from INDEX §3. They are not
duplicated here.
