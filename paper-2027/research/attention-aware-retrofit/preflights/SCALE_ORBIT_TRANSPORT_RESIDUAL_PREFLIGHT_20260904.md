# Scale-orbit hard-panel transport-residual preflight

- **Date:** 2026-09-04.
- **Status:** `EXECUTED / RESULT OWNED ELSEWHERE`. See the transport follow-up
  in
  [`SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md`](../results/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md).
- **Question:** Does the repository's pre-existing isotropic in-window
  transport residual `D*` distinguish the useful log-p2 table from today's
  exact-chain, ULP-jitter, and same-support geometric failures when orbit
  counts and Gram-tail quantities do not?
- **Primary estimand:** `D* / d` under the exact causal 4K distance weight,
  reported separately from 8K/16K phase-safe fraction.
- **Compute:** CPU only, `CUDA_VISIBLE_DEVICES=-1`; no checkpoint, forward pass,
  optimizer, benchmark, or new table search.

## Frozen protocol

The input contains exact float32 Native, same-support geometric, legacy-u
log-p2, minimum-class chain, and ULP-jitter chain tensors from the completed
scale-orbit panel. The machine-private inline manifest SHA-256 is
`e9bd3187e3846645cd293d7de5e482439f73f2234398f4a4956ae0194809d6bd`.

Use `head_dim=128`, Native length 4096, target length 16384, exact causal
pair-count weighting reduced to 2048 support points, phase-safety endpoints
8192/16384, and 40 alternating transport iterations. Do not generate PI,
YaRN, or new budgeted candidates and do not run a rank sweep. The existing
runner/solver/table/weight SHA-256 values are respectively
`1b4b8a2d38d6479045ca1e1a2b2ada85de1565a5ee523760efa4b5820f302541`,
`33583aea5f37cf02aa8289044b1a66ead54b5989e7cf44caaacb3c26ab63c0b0`,
`f7a63653f54fe9b15f31eaf56c8ee3840dcf728d79a65e9e28f1e5f1980c6eff`,
and `b9c44de121a831a46c1a1c45a80190fd3d6fae44e03f9a7dd18e9c215e6bc4ad`.

## Outcome map

- If p2 has materially lower `D*` at matched or better phase safety, the old
  two-axis analysis survives this hard panel only as a post-outcome
  explanation; it is not a prospective LM selector.
- If p2 is tied with or worse than a failed table, `D*` also fails to explain
  this panel and the route closes at this scope.
- Any non-convergence, identity drift, or GPU visibility makes the run
  **Invalid**. Stop after these five frozen tables; no rescue settings.

No outcome identifies a unique `m_k`, validates an isotropic activation prior,
or promotes a static geometry quantity into a behavioural theorem.
