# Bounded-condition scale-conjugacy tightness preflight

- **Date:** 2026-09-04.
- **Status:** `EXECUTED / POSITIVE CONTROL PASSED / PRIMARY NEGATIVE /
  MULTILEVEL STOPPED`.
- **Question:** On the five frozen RoPE tables, can an optimized real
  invertible conjugacy approach the finite-window Theorem 5 lower bounds, or
  are those bounds too weak to describe the best sampled operator error?
- **Estimand:** for every table, scale level, window, and condition cap, the
  best-found sampled value of

  \[
  \max_{t\in\mathcal T}
  \|D_jT_tD_j^{-1}-T_{s^jt}\|_{\rm op}.
  \]

- **Evidence boundary:** this is a numerical candidate search. A maximum on a
  finite set is a lower estimate of the continuous supremum; a best-found
  nonconvex solution is not the global optimum. It can motivate or falsify a
  proposed construction, but cannot certify an operator upper bound.
- **Theory owner:**
  [`FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md`](../../theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md).
- **Result owner:**
  [`SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md`](../../results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md).

## 1. Live alternatives

| Outcome | What changes |
| --- | --- |
| sampled optimized error approaches the strongest lower bound and improves over exact identity/permutation controls | justify a separate certified-supremum refinement; do not yet claim sharpness |
| sampled error remains far above both Ky-Fan and fixed-native projection bounds across restarts and condition caps | the lower bound is diagnostically weak in this search regime; strengthen the theorem or change the error metric before LM work |
| improvement appears only as the condition cap grows | conditioning is a load-bearing variable; no unrestricted-invertible practical claim |
| optimizer fails to recover identity or a known permutation baseline | implementation/protocol invalid; repair before interpreting any table |
| theorem bound is zero for all tables while sampled errors saturate | retain the theorem as structural motivation only; stop the empirical bridge |

Current owners do not answer this question. They compute necessary lower bounds
and exact permutation controls, but they never optimize a non-permutation
`D_j` or measure the lower-bound gap.

## 2. Frozen implementation

- Optimizer: `scripts/analysis/optimize_scale_conjugacy.py`, SHA-256
  `97c6f8dfff59e8e3679dd7ec3af884ba1df9cc2422f027e7023a2d6d1f96fd76`.
- Unified driver: `scripts/eval/run_finite_scale_covariance_program.sh`,
  SHA-256 `775b71e4fe591b5ab8023befebcc0e8ef5519483a395febbadbf863f449604bb`.
- CPU-helper tests: `tests/test_optimize_scale_conjugacy.py`, SHA-256
  `7547a42cf9ed2fdc30eb2d2e3ab0ba8182d29f1e7f086c9fa79ad8a1b08429ce`.
- Table manifest and all five float32 table hashes are reused unchanged from
  [`SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md`](SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md).
- The implementation uses real `2K x 2K` RoPE rotations. Each free real matrix
  is projected after every update to a declared condition-number cap. Cap `1`
  is the orthogonal family; caps `8` and `64` test increasingly non-normal
  similarities.
- Every selected matrix is saved as `.npy` with a SHA-256. JSON records source,
  manifest and position-set hashes, seed, restart, realized condition number,
  exact identity/permutation baselines, Ky-Fan bound, fixed-native projection
  bound, sampled error, and gap.
- Failed executions emit a separate failure receipt and retain their
  `.incomplete` matrix directory.
- Work-machine no-card preflight: five tables, 15 primary configurations and 45
  trajectories planned; `torch_imported=false`, `cuda_initialized=false`;
  receipt SHA-256
  `37aebf84b11b26fa2d1c707abb0c86ae6a0e78bf156d798aaa84d2c60b497358`.

## 3. Staged protocol

### Stage 0: no-card preflight

Validate syntax, test CPU-only helpers, load and hash all requested tables, and
print planned counts without importing PyTorch or initializing CUDA.

### Stage 1: GPU smoke

- Tables: `native`, `legacy_u_p2_log_s4`, `min_q_exact_chain_s4`.
- `s=4`, `N=1`, `L=256`, condition caps `1/8`.
- 20 optimizer steps, one restart, 16 training positions, 32 evaluation
  positions, four power iterations.
- Required controls: finite losses/matrices, condition caps respected, output
  and matrix hashes present, and selected sampled error no worse than the
  identity control on the same positions.

### Stage 2: primary tightness panel

Run only after Stage 1 passes.

- All five frozen tables.
- `s=4`, `N=1`, `L=4096`, condition caps `1/8/64`.
- 300 steps, three independent restarts, 64 training positions, 256 frozen
  evaluation positions, ten power iterations.
- 15 configurations and 45 optimization trajectories.

### Stage 3: multilevel/scale panel

Run only if Stage 2 finds at least one optimized non-identity matrix and a
materially smaller sampled error than both exact controls.

- Tables: `native`, `legacy_u_p2_log_s4`, `min_q_exact_chain_s4`.
- `s in {2,4,8}`, `N in {1,2,4}`, `L in {4096,16384}`.
- Condition caps `1/8/64`, otherwise the Stage 2 optimizer contract.
- 162 aggregate configurations and 1,134 optimization trajectories.

This large panel is a conditional follow-up, not a default launch.

## 4. Authorization and stop rules

All execution modes require `SCALE_CONJUGACY_GPU_AUTHORIZED=YES` and a visible
CUDA device. They load no language model and perform no training/inference.

Stop immediately on input/code hash drift, missing CUDA, non-finite objective,
condition-cap violation, output collision, missing sidecar hash, failure to
retain the identity control, or author stop. Stop after Stage 2 without Stage 3
if every bound is vacuous, every selected matrix is identity, or the optimizer
does not improve on the exact baselines.

Do not open LM endpoints, tune a RoPE table, or rerun completed RULER/natural
tasks from this experiment. Its sole estimand is mathematical tightness.
