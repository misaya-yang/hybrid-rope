# ICLR 2027 active handoff

- **Updated:** 2026-09-01
- **Role:** volatile Git/PDF/build/machine/compute/author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

The current research timeline and agenda live in [`../INDEX.md`](../INDEX.md)
§0/§6. The bottom-level mature-checkpoint research directory is
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).

## 1. Research status pointer

`MAXENT_CANDIDATE_CPU_VERIFIED / NATIVE_DOUBLE_GATE_FROZEN / NO_GPU_AUTHORIZATION`

The deterministic MaxEnt family and its CPU audit are active. Stage A uses the
existing formal evaluator on multiplier `1` only and requires both PG-19 PPL
and five-task natural-downstream retention to reach `87.5%` of paired Native.
This status does not authorize training, GPU evaluation, or paid compute.

## 2. Git state

- Branch / upstream: `main_0726` / `origin/main_0726`
- Published baseline before this documentation correction:
  `e1611d4d936d233e3223e5124a30de941c93a62e`
- Baseline divergence: `0/0`
- Current documentation correction: local and uncommitted
- Git publication: not authorized by this documentation request

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

- No GPU experiment is active.
- No training, GPU inference/evaluation, paid compute, OpenReview upload, or
  external mutation is authorized by this documentation update.
- The work machine owns canonical `aidemo` validation and final packaging.
- The low-configuration personal PC remains a documentation/planning host and
  may run static checks, LaTeX/Tectonic, and visual PDF work; `aidemo` is not
  expected here and must not be recreated for this change.

## 5. Validation

Passed locally in `aidemo`:

- MaxEnt CPU audit: `MAXENT_DILATION_CPU_CONTRACT_OK`;
- MaxEnt, RoPE-core, and repository-navigation tests: `160/160`;
- `git diff --check`;
- `paper/` diff empty;
- branch divergence against upstream: `0/0` at local HEAD `e1611d4d936d`.

No GPU evaluation or manuscript compilation was run. The double gate is a
frozen protocol, not an LM result.

## 6. Volatile action queue

1. Bind the remote checkpoint, formal token manifest, generated table hashes,
   output paths, free space, and shutdown receipt in no-card mode; do not start
   GPU evaluation without explicit authorization.
2. Rebuild and validate the curated supplement on the work machine before
   submission.
3. Freeze author metadata and complete the final owner-by-owner number review.
4. Obtain explicit authorization before any Git publication, GPU stage, or
   OpenReview upload.

Before Git publication, verify branch/upstream/divergence, staged scope,
sensitive content, and unchanged `paper/`; record local/tracking/remote SHAs
only after a successful push.
