# ICLR 2027 active handoff

- **Updated:** 2026-09-01
- **Role:** volatile Git/PDF/build/machine/compute/author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

The current research timeline and agenda live in [`../INDEX.md`](../INDEX.md)
§0/§6. The bottom-level mature-checkpoint research directory is
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).

## 1. Research status pointer

`P0_REFERENCE_INDEPENDENTLY_CONFIRMED / REFERENCE_CORRECTED_K128_S2_COMPLETE /
S4_PAIRED_CURVE_RUNNING`

Current canonical owners:

- [`Native reference calibration`](research/attention-aware-retrofit/results/NATIVE_REFERENCE_LENGTH_CALIBRATION_RESULT_20260901.md):
  the original two-code instrument remains abstained; the single permitted
  replacement independently confirms a protocol-specific 4K operating
  reference without relabeling training length.
- [`Reference-corrected K128`](research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md):
  s2 passes measured Native RULER/PPL gates and restores 8K; physical/index
  remain statistically unresolved. Conditional s4 uses one frozen profile
  at 4K/8K/16K, with a paired old-reference 16K control.
- Historical K32 Pareto, K64 scale consistency, and the original K128 negative
  remain routed by `INDEX.md`; none is silently replaced or pooled.

P3 cell-average remains rejected. No residual parameter, new curve, selector,
SOTA sweep, or s8 branch has been opened.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Last observed HEAD: `9948db166dd0ea50f3b995d8d59bbd00175a8b51`
  (background hourly workspace checkpoint; not committed by this run).
- Observed divergence against the existing upstream ref: `0/0`; no new
  fetch or remote publication verification was performed.
- Worktree: current P0/P1 code, reports, and receipts include modified and
  untracked files. They are not all published.
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
- The current GPU is a 32 GiB RTX 4080 SUPER. P0 calibration/confirmation,
  implementation controls, and P1 s2 are complete. P1 s4 RULER and independent
  natural continuation are running concurrently; a recent live sample was
  100% GPU utilization, not a claimed whole-run average. No old model was
  deleted. The protected 1.485B asset must remain untouched.
- Cell-average/P3 and s8 were rejected by their registered entrance gates.
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
- weight-identity and repository-navigation checks;
- focused `py_compile`, launcher `bash -n`, compact JSON parsing and
  `git diff --check` checks; no GPU validation is inferred from these.

Remote GPU owners separately record completed P0 and P1 s2 raw rows, runtime,
checkpoint/data/table identities, and finite loss. The stock/custom canaries
and explicit chunked FP32-attention comparison are bounded diagnostics.
The s4 result is not yet promoted. `paper/` has an empty diff; no manuscript
compilation, canonical `aidemo` suite or supplement packaging was run.

## 6. Volatile action queue

1. Finish the running P1 s4 paired length curve and old-reference control;
   preserve raw evidence and update its owner before deciding P2. Prepare
   same-family K triangulation on CPU only meanwhile. No `G(x;K)`, residual,
   gain or boundary search is authorized; identification precedes SOTA.
2. Rebuild and validate the curated supplement on the work machine before
   submission.
3. Freeze author metadata and complete the final owner-by-owner number review.
4. Stay within the user's currently authorized evidence-driven GPU program;
   OpenReview upload and new Git publication are not authorized by that program.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
