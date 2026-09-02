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
K128_COORDINATE_CONFIRMATION_COMPLETE / K32_FULL13_RUNNING`

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
- The registered K32 new-seed full-RULER-13 Native/index/YaRN confirmation is
  preparing/running. No partial outcome may change its three arms, seed, table,
  gain, lengths, or decision rule.
- Historical K32 Pareto, K64 scale consistency, and the original K128 negative
  remain routed by `INDEX.md`; none is silently replaced or pooled.

P3 cell-average remains rejected. Native-Q/K P3 is `ENTRANCE_FAILED /
NOT_EXECUTED`. No residual parameter, new curve, selector, SOTA sweep, or s8
branch has been opened.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Last observed HEAD: `19850cd29cf57ee41f9ee72f2c0af9beded853c9`
  (background hourly workspace checkpoint; not committed by this run).
- Observed divergence against the existing upstream ref: `0/0`; no new
  fetch or remote publication verification was performed.
- Worktree: the latest K32 YaRN receipt/report update is modified/untracked and
  not yet in the observed background checkpoint.
- This run has not staged, committed, pushed, pulled, rebased, or switched.

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

- The earlier OLMo, K32, and K128 queues completed and their shutdown was
  verified. The user has now restarted the instance and explicitly authorized
  the new ordered attribution/calibration program; the prior shutdown plan is
  no longer active. Do not automatically shut down this session.
- The current GPU is a 32 GiB RTX 4080 SUPER. P0, P1 s2/s4, Qwen P2, K32 N80,
  matched K32 YaRN, and K128 N80 are complete. The K32 full-RULER new-seed data
  are being prepared and the three GPU arms are chained immediately after; no
  old model was deleted. The protected 1.485B asset remains untouched.
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
- weight-identity and repository-navigation checks;
- focused `py_compile`, launcher `bash -n`, compact JSON parsing and
  `git diff --check` checks; no GPU validation is inferred from these.

Remote GPU owners separately record completed P0 and P1 s2/s4 raw rows, runtime,
checkpoint/data/table identities, and finite loss. The stock/custom canaries
and explicit chunked FP32-attention comparison are bounded diagnostics.
The s4 result is promoted only in its internal owner. `paper/` has an empty diff; no manuscript
compilation, canonical `aidemo` suite or supplement packaging was run.

## 6. Volatile action queue

1. Finish and attribute the running K32 new-seed full-RULER Native/index/YaRN
   confirmation. Native-Q/K P3 and physical-coordinate superiority remain
   closed. After the result, decide the next fixed natural-NLL/deterministic-
   baseline confirmation; no `G(x;K)`, residual, gain or boundary search is
   authorized.
2. Rebuild and validate the curated supplement on the work machine before
   submission.
3. Freeze author metadata and complete the final owner-by-owner number review.
4. Stay within the user's currently authorized evidence-driven GPU program;
   OpenReview upload and new Git publication are not authorized by that program.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
