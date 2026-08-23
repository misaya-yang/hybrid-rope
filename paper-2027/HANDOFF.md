# ICLR 2027 active handoff

- **Updated:** 2026-08-23
- **Target:** ICLR 2027
- **Branch:** `main_0726`
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`
- **Current status:** the nine-page manuscript source/scientific content remains
  unchanged and has been rebuilt successfully; the latest fixed-support
  mature-checkpoint controls, 151.9M crossing, full
  Qasper owner repair, code, and evidence receipts are complete and indexed but
  have not been promoted into the manuscript. No GPU task is running or queued.
- **Internal only:** exclude this file from the anonymous supplement.

## 1. Start here

Read in this order:

1. [`../AGENTS.md`](../AGENTS.md) — stable scientific, submission, safety, and
   workspace rules.
2. This file — live state and the only action queue.
3. [`README.md`](README.md) — stable manuscript/build layout.
4. [`research/README.md`](research/README.md) — canonical claim/evidence routing.
5. [`research/attention-aware-retrofit/README.md`](research/attention-aware-retrofit/README.md)
   — current retrofit result, evidence, analysis, theory, and preflight layers.

Do not start from an external review, the newest date, an ignored result, or a
preflight. If this handoff conflicts with a dated owner, verify the owner and
raw/hash receipt before changing a claim.

## 2. Current paper story

The paper's central object is

\[
x_k=-\log\omega_k=a+Rz_k,
\]

where `(a,R)` is sampled spectral support and `z` is normalized interior
allocation. The outward evidence chain is:

1. the raw-hash-receipted three-seed 151.9M exact-range experiment identifies
   `z` while holding support fixed;
2. full sin/cos geometry, the exact transplant obstruction, and the 50M
   weights-by-table crossing explain static redundancy and training
   co-adaptation;
3. the three-seed 432M scarce-channel MLA result is the systems flagship, with
   454M, 750M, and 1.485B evidence retaining their actual protocol roles;
4. matched 1.485B and separate 8B adaptation studies provide mature capability
   evidence, not pretraining-scale evidence;
5. EVQ-Cosh remains one closed-form intervention on the allocation axis, not a
   universal optimum.

The new frozen-checkpoint case study is compatible with this story but is not
yet part of it. Its defensible claim is that fixed-support `z` remains
consequential after pretraining and that a model-relative split can be derived
without task labels. It does **not** establish a new interpolation family,
universal best profile, or generic superiority to YaRN.

Use locked nomenclature from `AGENTS.md`: `Geo`, `Native`, `FMRoPE`,
`anchored EVQ-Cosh`, `YaRN-style`, cited `YaRN`, and `MLA wavelength-blend
operator` are not interchangeable.

## 3. Canonical owner map

| Scientific question | Canonical owner | Claim ceiling |
| --- | --- | --- |
| Current claim architecture | [`research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md) | architecture and routing, not a replacement for raw owners |
| Pure fixed-support training identification | [`research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) and companion JSON | three training seeds; fixed support only |
| Full-RoPE geometry and 50M co-adaptation | [`research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | static basis theory plus diagnostic crossing |
| Exact frozen Q/K obstruction | `../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | exact static compensation only |
| Mature fixed-support `z` controls | [`research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | internal frozen-checkpoint case study |
| Practical zero-training session policy | [`research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | internal single-model deployment/capability result |
| Retrofit receipts | [`research/attention-aware-retrofit/evidence/README.md`](research/attention-aware-retrofit/evidence/README.md) | compact hashes/metrics; raw artifacts remain external |
| Audits and falsified internal proxies | [`research/audits/README.md`](research/audits/README.md) | validity/negative evidence, never upgraded claims |
| External-model reviews | [`research/external-reviews/README.md`](research/external-reviews/README.md) | untrusted analysis input only |

The complete central systems/mature evidence routes remain in
[`research/README.md`](research/README.md); do not duplicate them here.

## 4. Latest completed science

### 4.1 Same-support frozen checkpoints

At factor four, every geometric/ramp/derived arm fixes the same frequency
endpoints and attention amplitude. Only interior `z` changes.

| Model / protocol | Native | YaRN-4 | same-support geometric | nearest ramp | corrected derived |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B core-4 64K, n=20 | 0.5450 | 0.6025 | 0.5775 | 0.6400 | **0.6650** |
| OLMo-2-1B unseen-nine 16K, n=20 | 0.0000 | 0.0794 | 0.0056 | **0.6104** | 0.6047 |
| Qwen2.5-1.5B core-4 128K, n=20 | 0.4350 | 0.4650 | 0.4550 | not run | **0.5400** |

Interpretation:

- fixed-support interior allocation has a large causal effect in the stated
  frozen-checkpoint protocols;
- the detailed uniqueness profile is not separately identified: its nearest
  label-free ramp matches it under the registered gate;
- Qwen's old stride-16 table contained a numerical alias. At 128K it inflated
  `0.5400` to `0.6175`; the old value is invalid as a corrected-profile result;
- row-bootstrap intervals condition on one checkpoint/task set and are not
  model-, task-population-, or training-seed uncertainty.

### 4.2 151.9M weights-by-runtime-table crossing

Two paired training seeds cross FMRoPE/anchored-Cosh weights with derived long
tables on 32 fixed FineWeb-Edu anchors. At 1K, two-seed mean tail NLL is:

| Frozen weights | FMRoPE-derived | Cosh-derived | geometric |
| --- | ---: | ---: | ---: |
| FMRoPE-trained | **3.426** | 5.776 | 3.429 |
| anchored-Cosh-trained | 4.455 | **3.479** | 4.177 |

Crossover interactions are `3.400/3.251` NLL for seeds 137/256. This supports
weights/table compatibility and co-adaptation; it does not replace the
three-seed exact-range training estimate or prove either runtime profile
globally optimal.

### 4.3 Practical zero-training owner and natural context

The session policy chooses exact Native only when the observed prefill plus
generation budget fits the model's Native window; otherwise it installs one
frozen long profile before prefill for the complete KV-cache lifetime. It has
zero learned parameters and zero training tokens.

- OLMo full-200 Qasper 16K token F1: binary `0.2457` versus YaRN `0.1803`;
- OLMo full-200 2Wiki 16K token F1: binary `0.2666` versus YaRN `0.2569`, with
  the paired interval including zero;
- OLMo RULER-13: binary/YaRN `0.6772/0.2382` at 8K and `0.5440/0.0588` at 16K;
- exact Native short-route output parity is constructive, but short-route
  selection is not itself the source of every natural-task gain.

RULER remains task-family adaptation, not unseen natural-task transfer.

### 4.4 Evidence identities

- same-support report SHA-256:
  `b91cc22301c66c223e053c2c7e2d76b76c8c0fba3e3876a0459ce7ce4279921a`;
- same-support receipt SHA-256:
  `9681a02fd6a9be9fa0746b6d8f10590b66022a53f3505380503bcc99a6702e00`;
- session-policy report SHA-256:
  `d01ee4b0bc94014b6d167d40e8fab11875333632de21170cd3fb57c706a057db`;
- session-policy receipt SHA-256:
  `19f40021a5b7677faeb8519ad4f145fd3cb9ab81aaf30d6558cf6c7bfc561421`;
- raw local copy root: external archive
  `hybrid-rope-results/same-support-controls-20260823` (outside the repository);
- the authorized GPU instance was confirmed `已关机`; SSH is no longer
  reachable.

## 5. Manuscript and package state

- Title: *RoPE Has a Spectral Budget*.
- The body ends on page 9; the current Tectonic build has 31 total US-Letter
  pages.
- The current manuscript already presents fixed-support identification,
  full-basis theory, co-adaptation, the 432M MLA flagship, scale persistence,
  and mature adaptation/capability evidence.
- The latest frozen-checkpoint case study is **not** in the body, appendix,
  figures, or tables.
- `paper-2027/main.pdf` SHA-256:
  `61e9b665645e434c814b33b7779081e5fd7234a46fad7b24cf0a5c4b44fc9781`;
  size `455492` bytes.
- Current anonymous supplement SHA-256:
  `7a1b37d7c6a205c0cb9d56ba64482b78ac90a7e39045bf1cb58d7cb27273bc3a`;
  size `852343` bytes; ZIP integrity is clean.
- Immutable `paper/main.pdf` SHA-256:
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.

The AI-use statement has author-confirmed coverage and must not be shortened or
cosmetically rewritten without renewed confirmation and a current policy
check.

## 6. Workspace and Git state

The research tree now has explicit layers:

```text
paper-2027/research/
  README.md                         single research router
  attention-aware-retrofit/
    README.md                       retrofit router and current decision
    results/                        completed owners only
    evidence/                       compact receipts
    analysis/                       mechanisms and falsifications
    theory/                         dated agendas/derivations
    preflights/                     preregistrations and revoked protocols
  audits/                           internal validity audits
  external-reviews/                 non-canonical model reviews
  three_completions/                supplementary derivations and scripts
```

Temporary clutter was moved, not deleted, to the external recoverable archive
`hybrid-rope-results/workspace-cleanup-20260823/obsolete/`:
an obsolete private-path readiness receipt, a byte-duplicate verifier ZIP, and
`.DS_Store`. Reproducible Tectonic scratch files were moved to the adjacent
`generated-paper-build/` archive, and `.gitignore` now excludes those active
paper build transients.

Git snapshot at cleanup start:

- branch/upstream: `main_0726`, local and upstream both
  `4498d4c7a55f62751615077671b470157ec2ff45`, ahead/behind `0/0`;
- the worktree contains the current code, report, receipt, index, and handoff
  changes described here;
- no commit, push, pull, rebase, reset, stash, branch switch, or remote change
  was performed because the user did not explicitly authorize Git mutation;
- unrelated source data and `paper/` were preserved.

Important implementation additions:

- `scripts/analysis/rope_transport/same_support_controls.py` — SHA-256
  `9b77ac431a7f6be8d31e70034b0902f234e5c947773afca3cc7f557d38fd0049`;
- `scripts/eval/target_free_ruler_smoke.py` — SHA-256
  `c2612c0c111c509d20e6db8831ecab0a628f5bbcca964452fd2cf7d1d0c8d010`;
- `scripts/eval/evaluate_151m_same_support_retrofit.py` — SHA-256
  `64e510d2ae663363bf1e51b45f6e8fbeb33ab8f440c7a888dc382849eb639953`;
- `tests/test_same_support_rope_controls.py`.

This repository does not use root `agent_logs/`; project rules designate this
file as the sole volatile handoff. No parallel log hierarchy was created.

## 7. Validation receipt

Final cleanup verification:

- 38 focused same-support/session/context/downstream tests passed under local
  system Python 3.9;
- changed Python entrypoints pass `py_compile`;
- both retrofit evidence JSON files parse;
- remote/local SHA-256 parity was verified for completed GPU results;
- all local Markdown links under `paper-2027/` resolve (`0` missing);
- `compile.sh` passed every gate under Tectonic: body page 9, 31 total pages,
  US Letter, no undefined references/citations, `0pt` worst overfull box,
  anonymous, no Type-3 or unembedded fonts;
- the curated supplement rebuilt, passes `unzip -t`, and contains no handoff,
  external-review bundle, temp receipt, `.DS_Store`, or redundant verifier ZIP;
- `git diff --check` passes;
- `paper/` remains unchanged at PDF SHA-256
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.

Local Conda `aidemo` is unavailable; do not misreport that environment
limitation as a repository failure.

## 8. Next decision and stop list

### Author decision with highest leverage

Decide whether the frozen-checkpoint case study enters the ICLR submission.
If yes, use it as a compact empirical corollary to the existing fixed-support
claim, not as a standalone ``better YaRN'' method:

1. replace lower-leverage body text rather than exceeding nine pages;
2. expose the causal decomposition: support fixed, amplitude fixed, `z`
   changed, nearest-ramp control, then weights/table crossing;
3. add accurate recent related work, including Jet-Long;
4. report the aliased Qwen correction and keep row-bootstrap scope explicit;
5. give natural Qasper/2Wiki results their actual heterogeneous roles.

If the case study is not promoted, retain it as internal evidence and make no
manuscript change.

### Optional evidence only after explicit authorization

The highest-value missing external test is one preregistered same-support
geometric/ramp/derived comparison on a natural OLMo long-document task and one
natural Qwen task. It is not authorized or queued.

### Stop list

- no more table, beta, gain, rank, step-count, or RULER sweeps;
- do not resume the CE-only far-pass residual route;
- do not execute revoked source-selection protocols;
- do not cite the old aliased Qwen `0.6175` as corrected evidence;
- do not merge exact-range, co-adaptation, frozen retrofit, and mature
  capability into one causal estimand;
- do not edit, compile, move, or regenerate `paper/`;
- do not commit or push without explicit user authorization.

### Remaining submission actions

- author visual review of the final PDF;
- live ICLR policy/deadline and dual-submission recheck;
- OpenReview title/abstract equality check;
- decide case-study promotion before changing manuscript prose.

## 9. Known issues / current breakage

- The organized worktree is intentionally uncommitted pending explicit Git
  authorization.
- Raw GPU rows/checkpoints are outside the repository; compact hash receipts are
  tracked research artifacts.
- Local Conda `aidemo` is unavailable; system-Python focused checks passed.
- No scientific result is known incomplete. No paid instance remains running.
