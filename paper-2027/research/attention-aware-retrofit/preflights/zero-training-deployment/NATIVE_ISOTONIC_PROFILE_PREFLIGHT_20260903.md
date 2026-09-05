# Native-isotonic zero-training profile preflight

- **Date:** 2026-09-03
- **Status:** `EXECUTED WITHOUT PARAMETER RESCUE`; result owned by
  [`NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md`](../../results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md).
- **Authorization:** the author restarted the specified RTX 4080 SUPER machine
  and instructed continuation under the prior no-duration-budget, derivative-
  allowed, preserve-results, then-shutdown rules.
- **Training:** zero weight updates, zero table fitting, zero training tokens.
- **Primary question:** Does the declared squared Native-geometry surrogate
  `m = pinned-Iso(1-u)` provide a useful single-table Native/long operating
  point when paired with the already-frozen `c=.074` attention gain?
- **Theory owner:**
  [`NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md`](../../theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md)
- **Builder:** `scripts/analysis/derive_native_isotonic_profile.py`.

## 1. Alternatives and claim boundary

The experiment distinguishes four outcomes for one frozen composite candidate:

| 1x double gate | Cheap 2x/4x PG-19 | Interpretation |
| --- | --- | --- |
| pass | retained | supports the squared geometry surrogate plus inherited gain as a practical candidate |
| pass | lost | the declared surrogate is too conservative or allocates the wrong long utility |
| fail | retained | Native compatibility/inherited-gain operating point fails; long geometry remains viable |
| fail | lost | closes this exact `full-lag exact-u + pinned Iso + log-s4 + c=.074` composite |

No outcome identifies a latent checkpoint law or authorizes an exponent, gain,
normalization, rank cutoff, boundary, or head-selector sweep.

The experiment is not an untouched confirmation. The squared surrogate was
proposed after the existing `p=2`, C2, and head-selective outcomes were known;
`c=.074` was selected on the historical 1x PG-19 gate. The reused formal panel
is development/replay evidence. Fresh confirmation opens only after a positive
development result and receives its own frozen source/owner.

## 2. Frozen construction

### 2.1 Native identity

- Released OLMo-2-0425-1B-Instruct weight SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
- Native context `L=4096`, `K=64`, factor `S=4`.
- Authoritative Native `inv_freq` is the runtime float32 tensor with SHA-256
  `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34`.
- The token manifest must hash to
  `74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`.

### 2.2 Exact-u numerical contract

Use every integer lag `0..4095` with normalized causal pair-count weight
`2(L-d)/(L(L+1))`. Construct the full `128 x 128` real sin/cos Gram from the
closed triangular Fourier sum in arbitrary precision. The realized frequencies
satisfy `0 < omega_63 < ... < omega_0 < pi`; the complex nodes
`exp(+-i omega_k)` are distinct and `L >= 2K`, so the exact design has rank
128 and every leave-one-pair design has rank 126. No SVD cutoff, ridge, or lag
downsampling is admissible.

Compute all conditional residuals from the `2 x 2` blocks of one high-precision
inverse. The initial `200/240/280` ladder did not converge and produced no LM
outcome. A pre-model numerical amendment expanded the frozen ladder to
`360/440/520`; the last two points produced byte-identical float64 movement and
the same final float32 table. Entrance requires:

1. all three solves finite and all `u_k` in `[0,1]` up to the frozen point-MP
   residual tolerance;
2. last-two `u` and `m` maximum absolute differences `<=1e-12`;
3. identical pinned PAVA block structure and a stable final float32 table hash;
4. analytic triangular-kernel values match explicit high-precision lag sums on
   the self-test grid;
5. the final table is finite, positive, strictly decreasing, with exact movement
   pins `m_0=0`, `m_63=1`.

Failure stops before model execution. Point-MP convergence is a numerical
receipt, not an interval-arithmetic proof of the exact projector.

**Pre-model numerical amendment result.** The original `200/240/280` ladder
failed without loading the model. The expanded `360/440/520` ladder converged:
the final two float64 movement hashes were byte-identical and the primary
float32 table hash is
`3266663596a113b0bf8edbf5e89f57254fd96d980eb353614b9c66ea65f25bfa`.
Shifting every Native float32 frequency down or up by one ULP changed the
primary movement by at most `1.634e-7` (pair 14). The primary/current-p2
movement difference is `0.9971` (pair 16), so candidate separation is far
larger than this numerical sensitivity. No LM outcome informed the amendment.

### 2.3 Frozen 2 x 2 attribution table

Construct before any LM outcome is read:

| Native phase diagnostic | Link | Table |
| --- | --- | --- |
| historical 2048-lag/cutoff/min-max `u` | historical square | current `log_s4` positive control |
| historical `u` | pinned linear `Iso` | `legacy_u_iso_log_s4` |
| full-lag high-precision raw `u` | pinned square | `exact_u_p2_log_s4` |
| full-lag high-precision raw `u` | pinned linear `Iso` | **primary `exact_u_iso_log_s4`** |

All four use the log-frequency table `omega'=omega*4^(-m)`. This factorial is
an attribution control, not a search: no table is selected from an LM outcome.
If primary and current `p=2` are float32-identical, or their difference is no
larger than the numerical/one-ULP sensitivity audit, stop as experimentally
redundant.

Frozen float32 tensor hashes are:

- current legacy-u/p2:
  `56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`;
- legacy-u/Iso:
  `d6c529b4cc72a37396f68ec6862af6e3060870ff5df0550a07a3c5c194733919`;
- exact-u/p2:
  `6a3cbf3901e3398aed2c2906da4e3bbfe413ee1cabe078430ca6e778a2ff7ba4`;
- exact-u/Iso primary:
  `3266663596a113b0bf8edbf5e89f57254fd96d980eb353614b9c66ea65f25bfa`.

## 3. Runtime arms and controls

The matched system component is fixed for every non-Native table:

```text
attention_scaling = 1 + 0.074 ln(4) = 1.102585782722872
```

The arms are:

1. Native table, gain `1` — valid reference;
2. Native table, gain `1.102585782722872` — gain-only control;
3. current historical-u/p2 `log_s4`, matched gain — positive control;
4. historical-u/Iso, matched gain — projector-axis bridge;
5. exact-u/p2, matched gain — link-axis bridge;
6. exact-u/Iso, matched gain — primary composite.

Existing Native/current-p2 formal rows may be reused only after exact
checkpoint, token-manifest, evaluator, decoder, scorer, table, gain, and row
hash identity is verified. The three new profile arms and gain-only arm run on
the same rows. A negative primary result closes only the inherited-gain
composite, not unit-gain Iso as a class; no rescue gain run follows.

## 4. Stages and stops

### Stage A — CPU identity and fail-closed preflight

1. Run the high-precision builder and freeze receipt/table hashes.
2. Compare the four movement vectors, final tables, maximum ULP difference, and
   maximum Native-window phase displacement.
3. Run formal evaluator `--preflight-only` for every new table.
4. Verify stock/custom all-Native logits exactly and candidate outputs finite
   at a short prompt and a 16K-shape Flash-only smoke.

Stop before formal evaluation on any hash/rank/convergence/order/parity/Flash
failure, or if the candidate difference is below numerical uncertainty.

### Stage B — formal 1x replay

Use `scripts/eval/target_free_formal_eval.py`, multiplier `1`, limit 20, and
the existing formal rows:

- PG-19 final-tail NLL/PPL retention;
- equal-task official-score macro over Qasper, MultiFieldQA-en, HotpotQA,
  2WikiMQA, and GovReport.

The operational double gate remains a conjunction:

```text
PPL retention >= 0.875
five-task macro retention >= 0.875
```

This threshold is an author-chosen operating tolerance, not a scientific
discontinuity. Report paired row deltas and uncertainty as continuous results.

### Stage C — long diagnostic and conditional breadth

Regardless of the 1x gate, run the primary on PG-19 at `2x/4x`; this cheap
diagnostic distinguishes a Native-only failure from loss of the long backbone.
Compare against the frozen current-p2 values with a predeclared descriptive
non-inferiority band of `+0.02` NLL at each length.

Only if the 1x double gate passes, open:

1. the six-task natural `2x/4x` matrix on the same formal manifest;
2. RULER-13 at 4K/8K/16K on the existing frozen seed;
3. a separately frozen fresh-source confirmation if the development result
   remains useful.

The development result met that entrance. Before generating or reading any
fresh row, the confirmation is frozen as official RULER core-4
`niah_single_1/niah_multikey_2/niah_multikey_3/vt`, seed `202609037`, 20 rows
per task at 4K/8K/16K, comparing only primary versus current-p2 with the same
current evaluator, gain, decoder, and scorer. Primary reporting is the three
length-specific task-equal macros and paired row differences; no new selection
threshold or rescue follows.

RULER and task macros are reported by task and length; they are not pooled into
one universal success score. No generation or fresh-source result may repair a
failed 1x deployment gate.

## 5. Monitoring, evidence, and shutdown

- Monitor process alive, log growth, completed row count, GPU memory, finite
  metrics, and result-file creation every 120 seconds.
- Preserve command, PID, exit code, stdout/stderr log, builder receipt, every
  table/movement/u hash, evaluator manifests, raw JSONL, results, failures, and
  exclusions under one machine artifact root.
- A stage is complete only when its process exits zero and its result hashes
  read back. Do not silently retry or alter the candidate.
- Write a bounded result owner and compact machine-path-free receipt before
  shutdown.
- After every authorized stage and derivative is complete, issue the provider
  shutdown command and confirm that SSH no longer accepts the connection.
