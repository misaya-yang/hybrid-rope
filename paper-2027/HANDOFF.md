# ICLR 2027 active handoff

- **Updated:** 2026-09-01
- **Role:** volatile Git/PDF/build/machine/compute/author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

The current research timeline and agenda live in [`../INDEX.md`](../INDEX.md)
§0/§6. The bottom-level mature-checkpoint research directory is
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).

## 1. Research status pointer

`P0_REFERENCE_INDEPENDENTLY_CONFIRMED / K32_N80_COMPLETE /
K128_COORDINATE_CONFIRMATION_COMPLETE / K32_FULL13_CLEAR_ADVANCE /
WORK_MACHINE_NLL_NEXT`

Current canonical owners:

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
- [`Work-machine plan`](research/attention-aware-retrofit/WORK_MACHINE_NEXT_EXPERIMENT_PLAN_20260901.md)
  freezes packed-natural NLL as the immediate next gate. Its model-free data
  receipt is ready, but model evaluation status remains `NOT_RUN`.
- Historical K32 Pareto, K64 scale consistency, and the original K128 negative
  remain routed by `INDEX.md`; none is silently replaced or pooled.

P3 cell-average remains rejected. Native-Q/K P3 is `ENTRANCE_FAILED /
NOT_EXECUTED`. No residual parameter, new curve, selector, SOTA sweep, or s8
branch has been opened.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Published research-delivery commit:
  `dd85086335efddfd3fc7c90675f1714e41236bf1`.
- The ordinary push to `origin/main_0726` succeeded; the remote ref was
  independently read back at the same SHA and divergence was `0/0`.
- The worktree was clean immediately after that delivery. This handoff-only
  refresh may advance HEAD once more; verify the final SHA live.
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
  shutdown. `shutdown -h now` returned success; the follow-up SSH check timed
  out during banner exchange, confirming the instance is no longer reachable.
- P0, P1 s2/s4, Qwen P2, K32 N80, matched K32 YaRN, K128 N80 and the K32
  full-RULER-13 confirmation are complete. Packed-natural data preparation is
  CPU-only and model status remains `NOT_RUN`. No old model was deleted; the
  protected 1.485B asset remains untouched.
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

## 6. Volatile action queue

1. On the work machine, execute only the frozen packed-natural
   Native/index/YaRN NLL gate. If it passes, preregister natural 2Wiki/Qasper/
   Hotpot confirmation, then matched static PI/NTK/Resonance baselines. Keep
   Native-Q/K P3, physical-x privilege, `G(x;K)`, residual, gain and boundary
   search closed.
2. Rebuild and validate the curated supplement on the work machine before
   submission.
3. Freeze author metadata and complete the final owner-by-owner number review.
4. Stay within the user's currently authorized evidence-driven GPU program;
   OpenReview upload and new Git publication are not authorized by that program.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
