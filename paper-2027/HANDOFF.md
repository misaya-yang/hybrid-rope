# ICLR 2027 active handoff

- **Updated:** 2026-09-03
- **Role:** live Git, PDF, validation, authorization, and author actions only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

Read [`../README.md`](../README.md) for the paper core and
[`../AGENTS.md`](../AGENTS.md) for rules. Use [`../INDEX.md`](../INDEX.md) only
to resolve a specific scientific claim. This file owns no scientific verdict.

## Latest changes

- Native-isotonic improves natural retention/PG but loses fresh core-4 at
  4K/8K: an endpoint-dependent tradeoff, not a replacement or formal frontier:
  [`result`](research/attention-aware-retrofit/results/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md),
  [`receipt`](research/attention-aware-retrofit/evidence/NATIVE_ISOTONIC_PROFILE_RECEIPT_20260903.json).
- The frozen head-selective candidate is negative on the reused panel:
  [`result`](research/attention-aware-retrofit/results/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md),
  [`receipt`](research/attention-aware-retrofit/evidence/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RECEIPT_20260903.json).
- The post-run audit classifies the selector score as attention displacement,
  not `chi_func`, and separates frequency-only permutation from joint gauge:
  [`audit`](research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md).
- No manuscript/training/adaptation/upload changed; the research bundle is
  authorized for commit and push.

## Git

- Branch / upstream: `main_0726` / `origin/main_0726`.
- Before publication, local HEAD and refreshed tracking SHA were both
  `917dcc98313b5c76af9c2041c55971e59448ec7c`, ahead/behind `0/0`.
- Publication scope is only the 9/3 owners, audit, receipts, builder, and
  routers; verify final local/tracking/remote equality after push.

## Manuscript and PDF

- Active PDF `paper-2027/main.pdf` SHA-256:
  `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`
- Prior validated body / total: 9 / 31 pages; not rebuilt in this audit.
- Immutable `paper/main.pdf` SHA-256:
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`
- `paper/` and active manuscript source/assets have empty diffs. Neither
  manuscript was compiled.

## Authorization and machine

- At the final live check the reopened work machine had an idle GPU and no
  experiment process. No shutdown command was sent; the author may shut it down.
- Private raw rows, predictions, scripts, manifests, and calibration sidecars
  remain outside the repository and must not be committed.
- The final remote hash inventory covers 100 files; its SHA-256 is
  `0e803e12a08e1ce153afadee9a30020d82e5cb12c3984f90e80db6f821b1a4ca`.
- The work machine owns canonical PyTorch, packaging, and release validation.
- The low-configuration personal PC is a documentation/planning host. Do not
  install or recreate the work-machine environment here.

## Validation

Passed on the personal PC: repository navigation 25/25, scoped diff/public
safety checks, corrected-theory CPU counterchecks, receipt JSON parsing, PDF
hashes, and empty manuscript-source diffs.

Work-machine passes: frozen asset/table hashes, BF16 Flash-only 16K probe,
stock/custom Native bitwise identity at 32 and 16,309 tokens, `300/300` finite
six-arm cells, paired bootstrap, exact matched controls, recovered `12x256`
calibration matrix/all 66 stability pairs, and receipt-to-raw readback.

Skipped by the frozen stop rule: a nontrivial joint-relabel execution and any
head-count, scale, slot, or selector sweep. The joint-relabel algebra and source
path were audited only. Manuscripts were not compiled or packaged.

## What to do now

1. Keep both 9/3 results post-submission unless the author explicitly changes
   manuscript scope.
2. Do not continue either exact candidate with a sweep. A future mechanism
   study first needs the nontrivial joint-relabel gate, a prospective owner,
   and a fresh source.
3. Continue the existing submission audit and 2026-09-17/18/25 milestones.
