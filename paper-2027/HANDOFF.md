# ICLR 2027 active handoff

- **Updated:** 2026-08-24
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** manuscript/package validation and the authorised 2026-08-24/25
  retrofit experiment reports are committed and pushed. No OpenReview upload or
  submission is implied. Verify live Git state instead of copying a commit SHA
  into this mutable handoff.
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

### 2026-08-24/25 closed GPU experiment ledger

These experiments are complete. Read the linked owner before proposing another
run; a failed gate is a stop decision, not an unfinished queue.

Cross-experiment interpretation and the problem-2 research route are owned by
[`research/attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md`](research/attention-aware-retrofit/analysis/POST_GPU_REFLECTION_AND_PROBLEM2_ROADMAP_20260824.md).
Read it before treating session routing, phase-chord, a collision proxy, or a
candidate failure as the method conclusion.

| Question | Owner | Decision; do not repeat |
| --- | --- | --- |
| Can a two-document learned direct-`z` calibration robustly retrofit mature OLMo? | [`research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) | Mean 2x moved favourably but the per-row gate failed. Do not run its PG-19, RULER, LoRA, or full-task continuation. |
| Can either tested analytic static table serve both 1x and 2x with frozen weights? | [`research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`](research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) | No: both tested tables improved 2x directionally and failed 1x retention. Do not sweep `tau`, protected bands, or gain from this result; the broader single-table objective remains open. |
| Does the already-frozen Native/s4 policy generalise to new natural text, and what owns each effect? | [`research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | Closed positive confirmation on disjoint shard002 rows. Do not rerun PG-19/RULER to reconfirm it or turn the ramp control into the method claim. |

The fresh result keeps four estimands separate:

| Contrast | Held fixed | Changed | What it establishes |
| --- | --- | --- | --- |
| Native versus Native/s4 session policy | checkpoint and rows | table, gain, and route as one bundle | zero-training policy persistence only |
| geometric versus derived/ramp at long lengths | sampled support `(a,R)`, gain, route, checkpoint, and rows | interior allocation `z` | length-dependent third-axis contrast among these tested `z` values; geometric is locally competitive at 8K and fails at 16K |
| derived versus coarse ramp | support, gain, route, checkpoint, and rows | fine profile detail within tested allocations | tested profiles are indistinguishable at current precision; no uniqueness or continuous-basin claim |
| static-s4 versus session-s4 | one frozen long table, gain, checkpoint, and rows | short-request routing | exact Native routing owns 4K retention |

Target-aware s2 versus session-s4 changes support and table together; it is an
operating-point comparison, never a pure `z` effect. Official YaRN is an
external reference, never the mechanism owner. Geometric/non-geometric is not
a quality classifier: a non-geometric `z` may perform well in-window and fail
under farther extrapolation.

Canonical reports and compact machine receipts are under
`research/attention-aware-retrofit/{results,evidence}/`; raw rows remain in an
ignored external owner. The authorised window also produced exactly
`1,000,000,000` CPU-tokenised FineWeb-Edu tokens from new shards002/003, absent
from the historical 000/001/004 set. The token file SHA-256 is
`223e466b1e829675e027e900fa4dbb8b0ff2e851f70d0c9ef301b1fec944b284`.
This corpus is data readiness, not training evidence. It remains on the stopped
instance's system disk; copy it to an explicitly chosen persistent owner before
ever releasing that instance or using the corpus on another machine.

The core method frontier remains a single allocation that jointly preserves
in-window performance and improves extrapolation. This is feasible, not ruled
out: the internal two-seed phase-chord owner reports mean delta NLL
`+0.00070/-0.16051/-0.15577/-0.20522` at `1x/2x/4x/8x` relative to FMRoPE.
Its seed scope and selection history prevent manuscript promotion, but it
already falsifies any narrative of an inherent one-table trade-off. The frozen
session route is the current verified engineering reference. The phase-kernel
and optimized collision/resolution owners already supply the theoretical
construction; the open evidence is whether that theory table realises an
acceptable in-window/OOD Pareto as a zero-training hard swap on a mature model.

Do not infer the intrinsic in-window cost of allocation from a frozen-table
swap. The governing decomposition is

\[
\Delta\mathcal L_{\rm in}
=\Delta\mathcal L_{\rm alloc}
+\Delta\mathcal L_{\rm adapt},
\]

where the second term is table/weight co-adaptation mismatch. The 2026-08-24
direct-`z`, analytic-table, and fresh same-support runs all freeze weights and
therefore include this mismatch; they cannot establish a fundamental
in-window/extrapolation trade-off. From-scratch exact-range, phase-chord, and
same-initialisation training owners govern the co-adapted allocation frontier.
This diagnosis does not create an adaptation fallback for the current
zero-training theory-table route.

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

There is no active GPU queue. A future large-model retrofit run requires a new
method-level hypothesis and explicit authorisation. It must jointly pass a
declared acceptable in-window-cost gate and an extrapolation gate before
cross-model confirmation. The phase-kernel theory table is a zero-training
hard-swap method; LoRA/continued adaptation is not its fallback. Neither the
failed direct-`z` protocol nor the two failed analytic candidates are valid
launch points.

Future compute is not ready merely because a script or token corpus exists.
Before opening a GPU, follow `../AGENTS.md` and additionally verify all of these
task-specific gates:

1. the question is absent from the closed ledger above and can change the paper;
2. the exact estimand states support, `z`, gain, route, weights, endpoint, and
   which of them change;
3. checkpoint, data, table, code, output, stop condition, and shutdown plan have
   frozen identities, and the raw data owner is portable or intentionally tied
   to the selected instance;
4. the 1.485B zero-training gate must pass before a larger checkpoint, broad
   capability suite, or cross-model confirmation is allowed.

A future run is not handed off as complete until its canonical report, compact
receipt, result/evidence indices, this handoff, raw-owner location, Git state,
and provider shutdown state agree. A plan, launch log, PID, output directory,
or tracked receipt alone never closes the loop.

The only active research implementation step is the report's M0 no-GPU table
materialization: bind the latest phase-kernel/collision-resolution theory owner
to one realised 1.485B table and a same-support Geo control, freeze identities,
and prepare the M1 zero-training hard-swap protocol. There is no small-model
search, 1× exact-parity requirement, or adaptation fallback.

Stop list:

- no new table, gain, beta, rank, step-count, or RULER sweeps for the current
  submission;
- no repetition of the shard002 Native/session confirmation or its 128/512-row
  controls unless the estimand, checkpoint population, or task family changes;
- no claim that coarse-ramp parity makes the method YaRN or makes `z`
  irrelevant;
- no claim that one shared table is mathematically unable to serve in-window
  and extrapolation regimes; this is neither proved nor supported by the
  candidate-specific frozen failures;
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
- Chrome showed the authorised A70/RTX 5090 instance as **已关机** on
  2026-08-24. This is volatile provider state: recheck the UI before assuming
  compute is running or before any release operation. The instance was stopped,
  not released.
- The 1B-token corpus is not in Git and is not yet a portable training-data
  owner; its tracked research receipt must not be mistaken for the raw corpus.
- No promoted scientific result is known incomplete. Remaining work is author
  review, live submission-policy verification, and explicitly authorized Git
  publication.
