# ICLR 2027 active handoff

- **Updated:** 2026-09-02
- **Role:** volatile Git/PDF/build/machine/compute/author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

The current research timeline and agenda live in [`../INDEX.md`](../INDEX.md)
§0/§6. The bottom-level mature-checkpoint research directory is
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).

## 1. Research status pointer

`PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED /
NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`

Current canonical owners:

- [`Headwise factorization and scale-flow result`](research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md):
  a 512-scalar headwise allocation+range field reaches `.23576` three-task
  natural-QA macro versus official YaRN-4 `.23817`, but fails the 4K PG-19
  retention gate. Exact-Native initialization improves 4K NLL over Native and
  fails long Hotpot. Free head gain is a protocol-specific negative. There is
  no SOTA or nonlinear scale-flow claim.
- [`Zero-training two-day synthesis`](research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md):
  frozen normalized-index improves Qwen K32 64K packed-natural NLL and
  far-source-conditioned answer likelihood, while the natural-generation QA
  gate, source-contrast decoding, and existing-candidate reranking do not
  recover a positive QA result. This local synthesis cleanly separates
  canonical 2026-09-01 owners from 2026-09-02 session receipts; the latter
  remain internal because their remote raw JSON/JSONL were not recovered.
- [`Native reference calibration`](research/attention-aware-retrofit/results/NATIVE_REFERENCE_LENGTH_CALIBRATION_RESULT_20260901.md):
  the original two-code instrument remains abstained; the single permitted
  replacement independently confirms a protocol-specific 4K operating
  reference without relabeling training length.
- [`Reference-corrected K128`](research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md):
  s2/s4 pass measured Native RULER/PPL gates and restore 8K/16K. The paired
  old-reference 16K control stays zero; physical-coordinate superiority is
  unresolved. The full 4K/8K/16K static-profile curve is complete.
- [`Qwen s2 baseline result`](research/attention-aware-retrofit/results/QWEN_S2_SAME_FAMILY_IDENTIFICATION_RESULT_20260901.md)
  is complete for K32/K64. The
  [`independent K32 N80 owner`](research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md)
  does not reproduce a physical long advantage: physical/index are unresolved
  at 64K, index is more Native-compatible, and the registered P3 entrance
  fails. A fixed YaRN completion ties index at 32K and is lower at 64K by
  `.064375`, paired 95% CI `[.0275,.102516]`; this is same-row baseline
  completion rather than an untouched final holdout.
- [`K128 coordinate confirmation`](research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md)
  completes on new seed `202609028`: index/physical are `.790000/.728125`,
  paired difference `+.061875` with 95% interval `[.028109,.096250]`. This
  closes physical-coordinate privilege and advances index only as the frozen
  engineering representative.
- [`K32 full-RULER confirmation`](research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md)
  completes as `CLEAR_ADVANCE`: Native/index/YaRN 64K macros are
  `.220513/.514551/.453654`; index-minus-YaRN is `+.060897`, paired 95% CI
  `[.027627,.095835]`, while the 32K index retention gate passes. QA and VT are
  mixed, so the result is not a task-universal or SOTA claim.
- The historical
  [`work-machine plan`](research/attention-aware-retrofit/WORK_MACHINE_NEXT_EXPERIMENT_PLAN_20260901.md)
  is no longer the action queue. Its natural-NLL/QA stage and the subsequent
  bounded headwise scope ladder are summarized by the latest owner above.
- Historical K32 Pareto, K64 scale consistency, and the original K128 negative
  remain routed by `INDEX.md`; none is silently replaced or pooled.
- Theory-only: the first-principles retrofit synthesis memo
  ([`FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902`](research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md))
  is complete under THEORY-ONLY mode (three isolated derivations +
  adversarial audit; no P0 findings). Verdict: partial structure with the
  unidentifiable quantity named (task-weighted tolerance to off-Native-arc
  joint phase configurations); no method, no experiment plan, no GPU queue.
  Indexed in `INDEX.md` §0/§2.2.

P3 cell-average remains rejected. Native-Q/K P3 is `ENTRANCE_FAILED /
NOT_EXECUTED`. No residual parameter, new curve, selector, SOTA sweep, or s8
branch is active.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Research-delivery baseline before this documentation refresh:
  `bbadf352e0bcdeb4d40131df864ab24bbee4a94d`.
- Local HEAD, `origin/main_0726`, and the live remote ref were independently
  read back at that SHA with divergence `0/0` before this documentation edit.
- This handoff is part of the documentation refresh and therefore does not
  embed its own final commit id; verify the final branch SHA live. Unrelated
  pre-existing worktree changes remain under `.agents/`, and
  two untracked analysis artifacts remain at
  `research/attention-aware-retrofit/analysis/h_n_diagonal_verification.json`
  and `research/attention-aware-retrofit/analysis/minimax_qp_phase1_results.json`.
  They were not modified or staged by this refresh.
- This run did not pull, rebase, switch, amend, force-push, or modify remotes.

Verify all volatile values live before Git operations.

## 3. Manuscript and PDF state

- Active PDF: `paper-2027/main.pdf`
- SHA-256: `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`
- Body / total: 9 / 31 pages
- Undefined references/citations: 0
- Worst overfull box: 0 pt
- Immutable `paper/main.pdf` SHA-256:
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`

The active manuscript has not been rewritten around the new research target;
that target is not yet a completed manuscript claim.

The curated supplement is not source-synchronised with the latest manuscript
repairs and must be rebuilt on the work machine before upload.

## 4. Machine and authorization state

- The 32 GiB RTX 4080 SUPER queue completed and was idle (`0%`, `0 MiB`) before
  shutdown. `shutdown -h now` returned exit code `0`; the follow-up SSH probe
  was immediately closed, confirming that the instance is no longer reachable.
- The 2026-09-02 headwise scope ladder, two-axis arm, three official natural-QA
  tasks, two 4K PG-19 retention panels, and corrected Native-start counterexample
  are complete. The raw/state/script bundle is archived locally outside Git;
  only the report and result-defining hashes remain in the repository.
- P0, P1 s2/s4, Qwen P2, K32 N80, matched K32 YaRN, K128 N80 and the K32
  full-RULER-13 confirmation are complete. The subsequent Qwen K32
  packed-natural NLL, far-evidence QA, table×gain, evidence-position bridge,
  source-contrast decode, and existing-candidate rerank also completed. Their
  aggregate statistics are recorded in the two-day synthesis, but their remote
  raw result owners were not recovered before shutdown. No old model was
  deleted; the protected 1.485B asset remains untouched.
- Cell-average, conditional Native-Q/K P3, and s8 were rejected by their
  registered entrance gates.
- No training or OpenReview upload is authorized.
- The work machine owns canonical `aidemo` validation and final packaging.
- The low-configuration personal PC remains a documentation/planning host and
  may run static checks, LaTeX/Tectonic, and visual PDF work; `aidemo` is not
  expected here and must not be recreated for this change.

## 5. Validation

Passed on the personal PC:

- 39 focused reference-gated coupling/baseline exporter tests;
- 43 Native calibration builder/decision/single-code tests;
- 55 fresh-NLL and paired RULER/NLL summary tests in the latest combined run;
- 29 same-family Qwen summary and 11 historical-replay synthetic tests;
- 101 K32 fresh-YaRN and K128 coordinate-summary identity/statistical tests;
- 37 full-RULER summary, 41 packed-natural data/evaluator/summary, and 6
  parallel-data-preparation compatibility tests;
- weight-identity and repository-navigation checks;
- focused `py_compile`, launcher `bash -n`, compact JSON parsing and
  `git diff --check` checks; no GPU validation is inferred from these.

Remote GPU owners separately record completed P0 and P1 s2/s4 raw rows, runtime,
checkpoint/data/table identities, and finite loss. The stock/custom canaries
and explicit chunked FP32-attention comparison are bounded diagnostics.
The s4 result is promoted only in its internal owner. `paper/` has an empty diff; no manuscript
compilation, canonical `aidemo` suite or supplement packaging was run.

The 2026-09-02 Qwen natural-NLL/QA/source-use chain has local code and parent-hash
bindings but no recovered remote raw JSON/JSONL. Its statistics are internal
decision evidence only until a raw owner is recovered and validated; no rerun
is implied by this handoff.

The 2026-09-02 local-only archive contains 42 files. All six state/log/receipt
triplets match their embedded hashes, all evaluation raw rows match their result
JSON hashes, and all receipt script hashes resolve to archived exact scripts.
Both corrected factorized baselines passed six-cell full-vocabulary bit-exact
parity before training.

## 6. Volatile action queue

1. No GPU method-development experiment is active or authorized. First decide
   whether the latest mature-checkpoint chain changes the manuscript claim set.
2. Recover and validate the missing 2026-09-02 Qwen raw results if another
   surviving copy exists. If not, keep those numbers internal; do not silently
   rerun or promote them.
3. If method research is separately reopened, require a CPU-identifiable
   bridge between the observed Native-compatible and long-capable basins. Do
   not begin with another frequency curve, gain, cutoff, curvature, or
   unrestricted per-frequency sweep.
4. Rebuild and validate the curated supplement on the work machine before
   submission.
5. Freeze author metadata and complete the final owner-by-owner number review.
6. Treat any new GPU run, OpenReview upload, commit, or push as a separate
   authorization; this handoff does not authorize them.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
