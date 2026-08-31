# ICLR 2027 active handoff

- **Updated:** 2026-08-31
- **Target:** ICLR 2027
- **Active manuscript:** paper-2027/
- **Immutable NeurIPS baseline:** paper/
- **Branch / upstream:** main_0726 / origin/main_0726
- **Published baseline before this iteration:**
  40bde33a47e3a43d7c9afaf6a57b0f4fd9f68234
  (checkpoint: prepare September paper iteration)
- **Published manuscript/package checkpoint:**
  09876d5 (`checkpoint: upgrade ICLR paper and freeze narrative`)
- **Published HEAD at the start of this follow-up sprint:**
  8e4d567 (`paper: reframe narrative around support-allocation decomposition`)
- **Published research-preparation checkpoint:**
  5a09464 (`checkpoint: prepare zero-training follow-up sprint`)
- **Published state before the current success-first documentation pass:**
  22477ae006315a2afbc05b58dc29838d37cb7107
  (`checkpoint: record follow-up sprint state`)
- **Published success-first workflow checkpoint:**
  5af6d46f844c92b87a582e96feb0891cc2745e7d
  (`Add zero-training tournament workflow`)
- **Published W0/F1 CPU-readiness checkpoint:**
  f508eb1153392a1a8460152f33b09ab8d8681978
  (`experiment: prepare W0 and F1 anchor stages`)
- **Verified divergence after the scoped W0/F1 push:** 0 / 0; local,
  `origin/main_0726`, and the live remote ref all resolved to `f508eb1`
- **Git publication state:** the W0/F1 code, root README invocation, index route,
  and frozen-protocol state are published at `f508eb1`; this handoff records
  that immutable code publication without changing experiment code

## 1. Current result

The evidence-complete September manuscript is **locally complete**. The latest
framing repair makes the scientific identity explicit before any benchmark or
method identity:

1. $x_k=-\log\omega_k=a+Rz_k$ decomposes a finite table into sampled support
   $(a,R)$ and interior allocation $z$;
2. the paired fixed-support intervention identifies $z$, while target-aware
   support retargeting reverses the tested ordering and establishes that the
   allocation result is conditional on the selected support policy;
3. exact phase-invariant geometry exposes the finite spectral budget and
   slow-end redundancy;
4. EVQ-Cosh is one analytic construction on this object;
5. frozen, adapted, and from-training evidence establishes distinct
   behavioural consequences without pooling estimands.

Fixed-support FMRoPE versus anchored EVQ-Cosh owns the clean allocation
identification. The target-aware reversal owns support-policy conditionality,
not a general support-by-allocation factorial law. FMRoPE is therefore a causal control, not a competitor to
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

The author authorized the scoped W0/F1 code and documentation commit/push; that
publication completed at `f508eb1`. Lightweight static validation remains
allowed on this PC.

It does **not** authorize new experiments, training, GPU inference/evaluation,
paid compute, OpenReview upload, unrelated remote mutation, branch operations,
force-push, or history rewriting. The author intends to run the experiment on
the home PC; its checkpoint/data/runtime/GPU readiness must still be verified
live before any model row is opened.

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
   target-aware retargeting reverses the ordering and shows that the tested
   allocation order is conditional on support policy;
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
  37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4
- size: 776774 bytes
- body: 9 pages
- references begin: page 10
- references end: page 13
- appendix begins: page 14
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
- citation-metadata hardening against the current 42-key TeX/BBL set: RULER
  now cites the COLM 2024 record; ordinary author lists, published venue/pages,
  applicable DOI fields, and canonical URLs were repaired; Llama 3 and Qwen2.5
  retain explicitly documented large-author `and others` exceptions;
- post-citation `paper-2027/compile.sh`: 9-page body, 31 total pages, 0 undefined
  refs/cites, 0pt worst overfull hbox, anonymity PASS, 0 Type 3/unembedded fonts;
  References pages 10--13 passed visual inspection and the appendix begins on
  page 14; TeX cites and BBL bibitems match 42/42 with `main.blg warning$=0`;
- post-citation hashes: `refs/references.bib`
  `bd09952fa53d8b1acf955bc2aef57a8b282c09f2a0aef6f0e488967ae6736f14`,
  `main.bbl`
  `845224e8f2e7a8f5f61c1625629adff8100653d235bcb0d82fcced9eb1334a58`,
  and `main.pdf`
  `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`;
- title/abstract/Figure 1/Contributions reviewer smoke test: the first recovered
  identity is the finite-table support--allocation decomposition and spectral
  budget, not a frequency-reallocation method;
- Figure 1 owner assertions and vector-PDF generation;
- final visual review of the abstract/intro, Figure 1, zero-training table,
  body closure, the assumption-bound derivation, and the expanded appendix
  evidence map;
- supplement static-theory allowlist-closure regression: 1/1;
- curated ICLR supplement packaging, leak scan, archive inspection, and
  isolated 9-body/31-total-page compilation;
- git diff --check;
- source/PDF synchronization at the hashes above;
- immutable paper/main.pdf hash unchanged;
- finite-K Cosh surrogate-regret audit: default
  $K\in\{8,16,32,64,128\}$ and $\tau\in\{0.5,1,2,4\}$ grid passed; final
  scaled-constant relative errors range from $1.66\times10^{-6}$ to
  $2.51\times10^{-4}$ in magnitude and tail slopes are within $0.0011$ of
  $-2$;
- 23/23 lightweight standard-library tests covering repository navigation,
  success-first portfolio routing, invalid band-restoration retirement, and the
  finite-K audit passed;
- changed-Markdown local-link audit, Python bytecode compilation, protected-scope
  scan, sensitive-diff scan, and `git diff --check` passed for the current
  documentation pass.

Still skipped for this iteration:

- full-repository and final cross-environment validation of the new W0/F1 patch;
- final cross-environment release receipt.

Work-machine validation for the success-first tournament code (run under Conda
`aidemo` on the host that owns it): `tests/test_success_first_portfolio.py`
13/13, `tests/test_repository_navigation.py` + `tests/test_rope_core.py` +
`tests/test_fixed_support_z.py` 160/160, all new Python modules bytecode-clean,
the stage driver passes `bash -n`, and an end-to-end CPU
freeze → candidate-assemble → contract chain passed on synthetic R0 data. No
model, checkpoint, or GPU was loaded in any of these checks.

W0/F1 CPU-readiness validation for checkpoint `f508eb1`:

- the installed current-PC `aidemo` suite passed `180/180` across
  `test_success_first_portfolio.py`, `test_repository_navigation.py`,
  `test_rope_core.py`, and `test_fixed_support_z.py`; this is changed-path
  validation, not the final cross-environment release receipt;
- Python bytecode compilation, stage-driver `bash -n`, `git diff --check`,
  sensitive-marker/private-path scans, and immutable `paper/` checks passed;
- the data-machine interpreter passed the focused tournament suite `20/20` with
  CUDA hidden; no model or GPU row was opened;
- the firewall split sizes are `D/S/T = 64/64/128`, pairwise disjoint, with
  replacement-source digest
  `22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb`;
- W0 contains exactly Native / frozen `s4` / official YaRN-4; its manifest file
  digest is `7e7336734253706e67b63089e4165256539246c98cea115fcc578417f9fd0af0`;
- F1 contains Native, six nonzero phase-chord morphs, and official YaRN-4; its
  manifest file digest is
  `6bc51248fe2e25b880fce7e067be3d9c4243be0dd5725c827af0153ecaf79899`;
- both no-GPU contracts returned `ZERO_TRAINING_TOURNAMENT_CONTRACT_OK`, bound to
  D-row digest `376ba06cd4d65addfea124331a5a1acfc2eb8231a2b3872e94135c96f7b45c37`,
  with `model_loaded=false` and `cuda_initialised=false`;
- checkpoint weight digest is
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`;
  the frozen `s4` tensor remains
  `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3`,
  and the Transformers 5.15.1 official YaRN-4 float32 table is
  `cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c`
  with gain `1.138629436111989`;
- an unauthorised `W0` launch stopped at the stage gate with exit `3` and wrote
  no W0 output.

These receipts establish CPU readiness and identity binding only. They are not
W0 measurements, candidate results, backend parity, or GPU authorization.

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
  longer source-synchronised after the framing and citation-metadata repairs;
  rebuild it before the final work-machine cross-environment receipt and
  OpenReview upload

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
| 1 | Publish W0/F1 CPU-readiness code and protocol | **completed:** scoped commit `f508eb1` is on `origin/main_0726`; remote SHA verified |
| 2 | Freeze success-first CPU assets | **completed:** portfolio, D/S/T firewall, W0/F1 manifests, official YaRN-4 table/gain, and both CPU contracts are hash-bound; no model result exists |
| 3 | Run W0 on the chosen machine | first verify the home-PC checkpoint/data/runtime/GPU/backend identities; evaluate only Native / official YaRN-4 / frozen `s4` on D and freeze `ABSOLUTE` or `ANCHORED` before reading F1 output |
| 4 | Run F1 only after W0 | separately authorize and pass the Native 1x-vs-4x parity smoke, then evaluate the frozen phase-chord morph grid on D; do not open S or T |
| 5 | Rebuild curated supplement | package contains the repaired manuscript source and passes leak/isolated-build checks |
| 6 | 9/17 abstract/metadata freeze | title, abstract, author roster, profiles, quotas, and AI statement confirmed |
| 7 | 9/25 full-paper freeze | owner-by-owner number read-through, final PDF/supplement approval, upload and downloaded-platform readback |

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
allocation optimization first and matched LoRA work second. The first target is
one static table that itself improves Native-window and long-range behaviour,
rather than preserving the former only through routing. Compute thrift is not
the primary optimization target: the current plan maximises the probability of
finding a result within four scientifically distinct candidate families.

The 2026-08-30 success-first portfolio is recorded in
[`research/attention-aware-retrofit/preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md`](research/attention-aware-retrofit/preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md):
phase-chord morph, Native-retention projection, five-degree behavioural
allocation, and one fixed support--allocation family each nominate one candidate
on development data. A separate selection split chooses one global winner and a
128-document confirmation split is opened once. Native-prefix, long-dense,
far-tail, and position bins come from the same physical `4x` forwards.

The archived abrupt band-restoration design is invalid because several arms
break frequency ordering; it is no longer a conditional queue. Matched-content
phase/gain work enters only when a confirmed winner has a mechanism ambiguity
that changes design. The finite-K audit remains an independent surrogate
certificate, not a candidate selector. These are research targets and protocol
gates, not manuscript claims or compute authorization. Durable ordering lives in
[`../INDEX.md`](../INDEX.md) §6.

## 10. Author actions

Before 9/17:

- align the abstract/introduction/contribution wording with the canonical
  support-policy-conditional reversal, without claiming a general
  support-by-allocation factorial law;
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
