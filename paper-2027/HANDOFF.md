# ICLR 2027 active handoff

- **Updated:** 2026-09-01
- **Role:** volatile Git/PDF/build/machine/compute/author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

The current research timeline and agenda live in [`../INDEX.md`](../INDEX.md)
§0/§6. The bottom-level mature-checkpoint research directory is
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).

## 1. Research status pointer

`MATCHED_K32_S2_COMPLETE / K128_SCREEN_UNRESOLVED_LONG_NEGATIVE /
REMOTE_SHUTDOWN_CONFIRMED`

The completed queue established four bounded results:

- frozen two-parameter C2 preserves OLMo/Qwen long behavior but misses the
  strict Native operating point;
- the same frozen K64 law zero-refit from s4 to s2 passes OLMo 1x and remains
  useful at 2x;
- on K32 matched s2, physical `x` wins 64K while normalized index passes the
  Native gate, establishing a Pareto crossing rather than uniform dominance;
- two exact Gemma-1 K128 artifacts recover nonzero 8K behavior under frozen
  tables but all Native/table 16K rows are zero and physical/index remain near
  parity. Gemma-1.1 Native 4K is `.9050`; physical table without gain is
  `.8350` at 8K; gain-only remains zero and Native/external canaries have exact
  output parity. This is an unresolved cross-K screen with a validated,
  replicated long negative—not a broken-model or wrong-artifact result.

P3 failed its CPU entrance gate; no `alpha` or GPU branch was created. The s8
stretch gate also failed and no hierarchical table was designed.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Published experiment/report commit:
  `3a3da401b8935065acd096cd0a72f4e2a6cf0290`
- Baseline divergence: `0/0`
- Current experiment code, reports, and receipts: published
- Git publication: complete on `origin/main_0726`

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

- All authorized OLMo, K32, and K128 queues completed; no GPU process remains.
- The compute instance accepted `shutdown -h now`; the SSH connection closed by
  the remote host and a subsequent connection probe timed out. Browser control
  was not needed. The protected 1.485B asset was neither moved nor deleted.
- Cell-average/P3 and s8 were rejected by their registered entrance gates.
- No training or OpenReview upload is authorized.
- The work machine owns canonical `aidemo` validation and final packaging.
- The low-configuration personal PC remains a documentation/planning host and
  may run static checks, LaTeX/Tectonic, and visual PDF work; `aidemo` is not
  expected here and must not be recreated for this change.

## 5. Validation

Passed on the personal PC:

- finite-K exact geometry audit reproduced without benchmark input;
- `24/24` focused exporter, weight-identity, and repository-navigation tests;
- `py_compile` for finite-K, cross-K exporter, identity helper, and runner;
- `bash -n` for the K-transport launch wrapper;
- compact JSON receipts parse successfully;
- `git diff --check`;
- `paper/` diff empty and immutable PDF SHA unchanged;
- branch divergence against upstream: `0/0` at local HEAD `c2c657c9cc56`.

GPU receipts record the RTX 4080 SUPER runtime, frozen table/checkpoint/data
hashes, completed K32 rows, completed OLMo s2 rows, and both K128 screens.
Canonical `aidemo` test suites and manuscript compilation were not run.

## 6. Volatile action queue

1. Do not fit `G(x;K)`, Native `alpha`, or hierarchical/s8 rescue from the K128
   screen. Before another arbitrary model, preregister Native-only natural and
   capability calibration to test whether config length differs from a
   checkpoint-level behavioral reference length. The next method comparison is
   then a matched deterministic static baseline panel such as Resonance-YaRN.
2. Rebuild and validate the curated supplement on the work machine before
   submission.
3. Freeze author metadata and complete the final owner-by-owner number review.
4. Obtain explicit authorization before any new GPU stage or OpenReview upload.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
