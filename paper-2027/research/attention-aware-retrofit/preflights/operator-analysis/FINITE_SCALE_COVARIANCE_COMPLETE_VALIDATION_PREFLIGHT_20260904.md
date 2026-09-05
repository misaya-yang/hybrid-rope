# Finite scale-covariance validation preflight

- **Date:** 2026-09-04.
- **Status:** `21 THEOREM CHECKS + PROOFS AUDITED / BEST-D_j POSITIVE CONTROL
  PASSED / 45-TRAJECTORY PRIMARY NEGATIVE / MULTILEVEL STOPPED`.
- **Question:** Which claims in the supplied Pro scale-covariance derivation are
  mathematically/numerically supported under their exact assumptions, which
  bounds are non-vacuous on frozen RoPE tables, and which theory--experiment
  bridges survive without treating any quantity as an allocation selector?
- **Scope correction:** this program does **not** search for or derive (m_k).
  It validates the finite-dimensional `leak / unresolve / alias` theory and its
  bounded empirical connection.
- **Evidence boundary:** a passing numerical audit is a reproducibility check,
  not a proof or literature-novelty certificate. Existing GPU behaviour tests
  remain separate observations.
- **Source boundary:** the supplied Pro text starts at Section 5 and ends at
  Section 12. Sections 1--4 were not supplied and are not claimed as covered.

## 1. Executable coverage of the supplied Sections 5--9 and 12

| Pro component | Code path | Required output |
| --- | --- | --- |
| Exact continuous scaling trilemma | `continuous_scaling_trilemma_report` | positive-scale nilpotent witness, polynomial norm growth, nonzero bounded-spectrum obstruction, explicit `s=-1` counterexample |
| Theorem 3 exact orbit growth | `exact_orbit_report`, rational exhaustive panel | both lower bounds, one-/two-sided boundary, equality iff every residue class is consecutive |
| Boundary allocation/sharpness | exact residue chains and `uniform_log_lattice_report` | packed-chain equality, commensurate index shift, incommensurate accumulated mismatch |
| Theorem 4 approximate packing | `best_coherent_matching`, partition exhaustive panel | one-sided multi-step partial-injection optimum, packed upper bound, leakage, tight constructions |
| Theorem 4 bidirectional case | `best_coherent_matching(two_sided=True)` | same coherent chain must survive both directions; verifies the stated `N -> 2N` replacement |
| Dimension--span inversion | `dimension_span_requirements` | one-/two-sided grid over factor, levels, leakage target, and tolerance |
| Continuous operator error to log tolerance | `operator_error_to_log_tau` | randomized phase-chord containment with absolute frequency anchor |
| Theorem 5 complex operator--orbit rank | `fourier_orbit_rank_report(real=False)` | continuous/discrete Gram, Ky-Fan bound, coherence and separated-frequency corollaries, fixed-native projection residual |
| Standard real RoPE specialization | `fourier_orbit_rank_report(real=True)` | signed `+/- omega` complexification and doubled representation dimension; not arbitrary real orthogonal RPE |
| Theorem 5 sharpness control | `_theorem5_orthogonal_sharpness` | exactly orthogonal Fourier characters attaining `1-K/M` PCA/Ky-Fan scaling |
| Dimension-free separation | `montgomery_vaughan_separation_bound` | `lambda_max <= 1+3pi/(2Ldelta)` and the resulting operator lower bound; inverse-`Ldelta` order, not best constant |
| Theorem 5 plus orbit growth | `orbit_growth_coherence_lower_bound` | explicit `1-K(1+rho)/(K+Nq)` bound, including vacuity clipping |
| Arbitrary invertible `D_j` diagnostic | `arbitrary_mixing_l2_report` | exact continuous/discrete L2 diagonal-error chain for sampled non-permutation invertible maps; no operator-supremum optimization |
| Exact discrete periodicity | `sharp_periodic_phases`, `approximate_alias_report` | zero-defect `BS(1,s)` cycles, `M=s^d-1`, coprimality and exact alias |
| Theorem 6 approximate alias | bottleneck spectral matching and cycle LCM | realized common period and theorem upper bound under perturbations |
| Anti-alias corollary | `anti_alias_margin`, `anti_alias_corollary_lower_bound` | direct numerical verification through horizon `s^d-1` |
| Multiple scale generators | `uniform_log_multi_generator_report` | exact jointly commensurate semigroup orbit and explicit failure when one `alpha/spacing` ratio is nonintegral |
| Frozen-table non-vacuity | optional `--table-manifest` | complex/real and continuous/discrete panels for every hash-bound table across all requested factors, levels, and horizons |
| Narrow behavioural discriminator | unified driver delegates to the frozen scale-orbit evaluator | tests whether boundary/Gram/one-step quantities rank named tables; it does not replay every Section 11 observation |

### Non-code and still-unresolved gates

| Pro item | Current coverage | Exact boundary |
| --- | --- | --- |
| Theorems 3--6 as formal mathematics | proof reconstruction completed in the theory owner | numerical harness remains only a countercheck |
| General real orthogonal RPE | proof completed by complexifying an arbitrary real orthogonal one-parameter group | executable panels still use standard RoPE blocks |
| Separated-frequency corollary | dimension-free Montgomery--Vaughan bound with sharp inverse-`Ldelta` order | best constant is not claimed |
| Matching upper construction | orthogonal-character PCA/Ky-Fan equality only | not an operator-similarity upper construction |
| Section 10 literature novelty | targeted current primary-source audit completed | non-discovery is not a novelty certificate |
| Section 11 behavioural bridge | routed to existing canonical owners below | no pooled theorem-to-behaviour causal claim and no redundant GPU rerun |
| Section 12 paper potential | narrow candidate contribution identified | best-found tightness result is negative; no current empirical basis for centering it |
| Pro Sections 1--4 | unavailable | cannot be audited from the supplied attachment |

The four Section 11 observations are owned separately: fixed-support causality
by `EXACT_RANGE_151M_3SEED_RESULT_20260820.md` and
`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`; final-frequency multiset
permutation collapse by `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`;
low movement MAE/RMSE without a Native-capability guarantee by
`CPU_LOW_DIM_COUPLING_LAW_20260901.md` plus
`LOW_DIM_COUPLING_GPU_RESULT_20260901.md`; and likelihood improvement without
generated-capability repair by `LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md`.
These observations support the safe bridge stated by Pro, but none is a test
of Theorem 5's operator premise.

The formal proof, primary-source comparison, stronger fixed-native projection
bound, and exact paper boundary are owned by
[`FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md`](../../theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md).

## 2. Frozen code and defaults

- Audit implementation: `scripts/analysis/finite_scale_covariance.py`, SHA-256
  `19222ff5f88438b812dd40d3ed58baecd9f6feccfeb43d798beef9c907980007`.
- Best-`D_j` optimizer: `scripts/analysis/optimize_scale_conjugacy.py`, SHA-256
  `97c6f8dfff59e8e3679dd7ec3af884ba1df9cc2422f027e7023a2d6d1f96fd76`.
- Unified driver: `scripts/eval/run_finite_scale_covariance_program.sh`,
  SHA-256
  `775b71e4fe591b5ab8023befebcc0e8ef5519483a395febbadbf863f449604bb`.
- Focused tests: `tests/test_finite_scale_covariance.py`, SHA-256
  `a03e6d3d87acf93a99e47eae554e3ac23b77b3efbdd540ec8b1f669c9d58d9e9`.
- Optimizer CPU-helper tests: `tests/test_optimize_scale_conjugacy.py`, SHA-256
  `7547a42cf9ed2fdc30eb2d2e3ab0ba8182d29f1e7f086c9fa79ad8a1b08429ce`.
- Default audit: factors `2,4`; levels `1,2,4`; continuous and discrete
  horizons `32,256,4096`; complex and standard signed real-RoPE panels; 64
  coherent-matching randomized cases; exact rational enumeration; discrete
  dimensions through 8 and scales 2/3.
- Existing behaviour driver and its exact table/checkpoint/data hashes remain
  owned by
  [`SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md`](SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md).
  This preparation does not mutate or relabel that completed experiment.

## 3. Execution gates

No-card mode may run only Python/shell syntax, focused unit tests,
`cpu-preflight`, and `operator-preflight`. The displayed provider CPU/RAM
inventory is not trusted for a full numerical run. `cpu-smoke` and `cpu-full`
remain unexecuted.

Locally, the current relevant suite passes `102/102`; the default CPU theorem
audit also passes. The work-machine no-card checkout matches all five current
code/test hashes and passes the combined finite-scale/scale-orbit/optimizer
suite `24/24`. Its 21-component/five-table theorem preflight receipt SHA-256 is
`f019c8504c723b0818fd589e82bb6547a9a8b4f5d9260cf17eabcae73a8fe788`.
The independent optimizer preflight covers five tables, 15 primary
configurations, and 45 trajectories without importing PyTorch; its receipt
SHA-256 is
`37aebf84b11b26fa2d1c707abb0c86ae6a0e78bf156d798aaa84d2c60b497358`.
The provider reported no GPU device during this no-card preflight.

A five-table `cpu-smoke` was mistakenly attempted on the no-card machine and
stopped after 1 minute 41 seconds without an atomic output. This is an invalid
execution attempt, not a scientific failure; the local no-table default audit
remains the only completed numerical run of the corrected harness.

Every GPU mode delegates to the already frozen driver and retains its
`SCALE_ORBIT_GPU_AUTHORIZED=YES` guard. No model load, inference, or training is
authorized by this document. Completed historical raw results should be reused;
rerunning them requires a new explicit authorization and reason.

The new `operator-smoke`, `operator-primary`, and `operator-multilevel` modes
instead require `SCALE_CONJUGACY_GPU_AUTHORIZED=YES`. They optimize general real
matrices projected to condition-number caps `1/8/64`, retain selected matrices
and hashes, and compare sampled errors against identity, block-permutation,
Ky-Fan, and fixed-native-subspace lower bounds. Sampled maxima are not certified
continuous suprema.

The later authorized RTX 4080 SUPER run completed `operator-smoke` and the
15-configuration/45-trajectory primary. A nontrivial synthetic conjugacy
positive control passed, but every primary selected identity with sampled
error near `2`; Ky-Fan was zero for four tables and `.052914` for Native.
`operator-multilevel` was stopped by protocol. Result owner:
[`SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md`](../../results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md).

Stop on input/hash drift, non-finite Gram spectrum, a violated exact inequality,
failed sharpness/equality construction, approximate-alias bound violation,
PyTorch import in the CPU audit, visible CUDA in a CPU mode, output collision,
or author stop. Preserve a failed receipt; do not weaken tolerances or remove a
case to obtain `PASS`.

## 4. Supported and unsupported claims before execution

Supported now: all explicitly numerical or finite-enumeration claims found in
the supplied Pro Sections 5--9 and 12 have a code path; local tests and a
default CPU audit supply implementation smoke. The separate theory owner checks
the proofs and derives the general real-orthogonal and separation-order results.

Unsupported now: that Theorem 5 is novel or globally non-tight, that any
theorem predicts LM ranking,
that a matching operator-similarity upper construction exists, that Sections
1--4 are covered, or that the paper-potential conditions are met. The completed
best-found search is negative at exact protocol scope, not an impossibility proof.
