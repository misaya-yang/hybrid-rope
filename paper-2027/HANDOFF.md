# ICLR 2027 active handoff

- **Updated:** 2026-08-25
- **Target:** ICLR 2027
- **Branch / upstream:** `main_0726` / `origin/main_0726`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Status:** final manuscript, theory/evidence owners, and PC-continuation
  documentation are validated and published on `main_0726`. Recheck live Git
  refs before pulling; no OpenReview upload or submission is implied.
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
use a fast-forward update; otherwise inspect ownership before changing Git
state.

Read in this order:

1. [`../AGENTS.md`](../AGENTS.md) — rules and claim ceilings.
2. [`../INDEX.md`](../INDEX.md) §2, §3.4, §6, §7 — theory, closed routes,
   agenda, and cross-machine boundaries.
3. This file — current manuscript, validation, and author actions.
4. [`main.pdf`](main.pdf) and `sections/` — reviewer-visible truth after a
   clean local build.
5. [`research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md`](research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md)
   — post-submission theory continuation.
6. The canonical numerical owner routed by
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
- Total PDF: 31 US-Letter pages.
- `paper-2027/main.pdf`
  - SHA-256: `5c5107126ae99d659c000fefd7b86fbd36178b4844a0b0b82e84bedd2db59c0e`
  - size: `697337` bytes
- `rope-spectral-budget-iclr2027-supplement.zip`
  - SHA-256: `f5587d994d9103d974f849930f2ff968b912386c5d6466ade6af4dd5831dc359`
  - size: `865317` bytes
- Immutable `paper/main.pdf`
  - SHA-256: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

Generated PDF, BBL, and supplement ZIP are local build products unless a later
explicit publication includes them. A tracked source commit is not an
OpenReview submission.

## 4. Latest validation

The final reviewer-path revision passed:

- active and isolated-package builds: 9 body / 31 total pages;
- zero undefined references or citations;
- `0pt` worst overfull box;
- anonymous Letter output, no Type-3 or unembedded fonts;
- exact-range/FMRoPE, fixed-support, repository-navigation, and package
  workspace suite: `28/28`;
- isolated supplement CPU suite: `144/144`;
- ZIP integrity and immutable-`paper/` checks;
- visual review of pages 1--9, Section 2 on page 3, both main figures, and the
  frozen-checkpoint table on page 29.

These receipts prove build/package health and the tested code paths. They do not
prove acceptance, policy currency, OpenReview state, or unmeasured claims.

## 5. Research continuation

The research agenda is not state. Its authority is [`../INDEX.md`](../INDEX.md)
§6; the theoretical reasoning is in the
[theory state](research/ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md).

Current missing bridge: a matched-content `table x virtual-gap position map`
2x2 that keeps tokens, answer, decoder, and rows fixed. Different-length RULER
rows cannot answer position failure versus model capability.

Do not start another shared-table static score, 50M candidate verdict, extra
oracle shell/step/seed/LR sweep, larger checkpoint, or GPU run from this
handoff. Any compute requires a new preflight and explicit user authorization.

## 6. What Git does not contain

| Artifact | Git status | Consequence on the PC |
| --- | --- | --- |
| manuscript source, owners, compact receipts, code, tests | tracked | available after verified sync |
| checkpoints and adapters | external | locate or transfer before evaluation |
| raw GPU rows and per-example outputs | external/ignored | compact means are insufficient for new per-position analysis |
| caches and token arrays | external/ignored | regenerate only from a pinned owner |
| prepared one-billion-token FineWeb-Edu corpus | previously machine-local; portability unverified | do not assume it exists on the PC or a new server |

A compact receipt is provenance, not the raw artifact. Missing local raw files
never mean the experiment did not run.

## 7. Author actions

Submission:

1. On the PC, rebuild from source and read pages 1--4 and 8--9 at normal zoom.
2. Confirm OpenReview title and abstract exactly match the PDF.
3. Recheck current ICLR policy, deadlines, anonymity, author profiles, and
   dual-submission state immediately before upload.
4. Upload only the curated paper/package, never a repository-root archive.

Research after submission:

1. Read the theory state and the complete closed-route ledger in INDEX §3.4.
2. Write the matched-content phase preflight without selecting on long-context
   results.
3. Request explicit GPU authorization only after code/config/data/table/output
   identities and a stop rule are frozen.
4. Require a 1.485B matched in-window + far-tail + capability gate before
   multi-seed or second-checkpoint expansion.

## 8. Git boundary

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
