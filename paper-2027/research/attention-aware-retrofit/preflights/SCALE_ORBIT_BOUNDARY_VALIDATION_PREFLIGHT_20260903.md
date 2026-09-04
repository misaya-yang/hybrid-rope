# Scale-orbit boundary and operator-rank validation preflight

- **Date:** 2026-09-03 project time (2026-09-04 on the work machine).
- **Status:** `EXECUTED / RESULT OWNED ELSEWHERE`. See
  [`SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md`](../results/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md).
- **Question:** Do exact scale-orbit boundary count, approximate channel
  matching, and finite-window Fourier-orbit rank remain distinct on a
  near-identical table pair, and does any resulting prospective ordering have
  mature-model behavioural relevance?
- **Primary estimand:** paired outcome difference between the frozen
  `min_q_exact_chain_s4` table and its ULP-jittered control on identical rows.
- **Training:** none; weights remain frozen and no table is selected from an LM
  outcome.
- **Authorization:** the author authorized preparation in no-card mode and GPU
  execution on the specified work machine after GPU activation, continuing
  until the author manually stops it. No GPU was visible and no model process
  was started at registration. GPU execution was later completed under that
  authorization; the result owner supersedes the prospective status here.

## 1. Claim boundary

This experiment does not test whether the stated combinatorial or Ky-Fan
inequalities are proofs; those are mathematical questions. It tests two
separate empirical bridges:

1. whether the discontinuous exact residue/boundary count is a plausible
   functional metric when an almost identical realized table changes that count;
2. whether the continuous finite-window Gram quantities supply a useful
   prospective ordering for one frozen OLMo checkpoint and protocol.

No outcome proves novelty, identifies a universal allocation, derives the
Native movement profile, or attributes a frozen-model failure solely to a
scale boundary. Static geometry and LM behaviour remain separate evidence
tiers.

## 2. Frozen alternatives and outcomes

All non-Native arms use `K=64`, the same fast endpoint, the same slow endpoint
`Native[-1]/4`, and the same fixed attention scaling

```text
1 + .074 ln(4) = 1.102585782722872
```

The arms are:

1. `Native`, gain `1`, as the unchanged reference;
2. `same_support_geometric_s4`, the geometric allocation over the common
   extended support;
3. `legacy_u_p2_log_s4`, the retained historical incumbent;
4. `min_q_exact_chain_s4`, six consecutive factor-four chains attaining the
   realized minimum-class construction;
5. `min_q_ulp_jitter_s4`, the primary control, obtained only by moving each
   non-endpoint chain value upward by `level+1` float32 ULPs.

The primary contrast has these pre-model interpretations:

| Result | Interpretation |
| --- | --- |
| behaviour is indistinguishable while exact `q,b` jump | exact boundary count is not a stable functional selector; approximate/Gram quantities survive this control |
| the tiny ULP perturbation causes a reproducible behavioural change | unexpected checkpoint sensitivity; repeat before assigning it to exact boundary structure |
| both tables fail while a matched-support control resolves | closes these chain constructions, not the theorem or allocation class |
| evaluator/control identity fails | `UNRESOLVED`; repair provenance, not parameters |

The secondary geometric/p2 comparisons are descriptive. They cannot rescue or
replace the primary near-identical contrast.

## 3. Frozen identities

### Model and data

- OLMo-2-0425-1B-Instruct weight SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
- Config SHA-256:
  `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c`.
- Native frequency float32 SHA-256:
  `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34`.
- Natural-text token manifest SHA-256:
  `74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`.
- Frozen RULER core-4 manifest SHA-256:
  `0e184255f006c212c9fc1db08860800cc3018c9de434e69a0d1d6207665fdb31`.
- Frozen panel manifest / metrics file SHA-256:
  `14945bc0b5faaedcdea16a860beaac7271b28abf436c3a845449b87cea5b043b` /
  `f859cc3cde76964e62d20f1f6eced4b1f982d53fd87c9d03b34be781324e4a40`.

### Table identities

| table | float32 SHA-256 |
| --- | --- |
| `Native` | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| `same_support_geometric_s4` | `2754c9c233fe6f65686e86189c4cee94b75efac0bd8df19f140f3e575ffb723f` |
| `legacy_u_p2_log_s4` | `56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b` |
| `min_q_exact_chain_s4` | `6344df4df8a9b39c50fdcd97c9ee22d736852566b2c1a396f2508ec472767475` |
| `min_q_ulp_jitter_s4` | `7896610d4dba6836233933eb71f1526b09ad9d2c833d53769f580cdff62d433b` |

## 4. CPU-derived preflight results

These are deterministic numerical properties of the frozen float32 tables, not
model observations. The continuous and discrete Gram calculations use
`s=4`, `N=1`, and `L=4096`; the operator entry is the best bottleneck error over
permutation conjugacies on the continuous window.

| table | realized `q` | boundary `b` | orbit modes `M` | Ky-Fan `epsilon` lower bound | permutation operator error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 64 | 64 | 128 | `.07765743` | `2.0` |
| same-support geometric | 64 | 64 | 128 | `.00012177` | `2.0` |
| historical log-p2 | 64 | 64 | 128 | `0` at float64 resolution | `2.0` |
| exact six-chain | 6 | 6 | 70 | `0` at float64 resolution | `2.0` |
| ULP-jittered chain | 64 | 64 | 128 | `1.01e-7` | `2.0` |

The exact-chain/jitter maximum phase difference at length `16384` is
`.0009765625`. At log mismatch `tau = 10^-6 log(4)`, both retain `58/64`
one-step matches despite their exact class counts being `6` and `64`.

Thus the current one-step operator-norm and Ky-Fan quantities do not
prospectively rank the practical extended tables: the operator error saturates
and the lower bound is zero or nearly zero. The GPU run is consequently a
falsification-oriented boundary-discontinuity test, not a promised method win.

## 5. GPU protocol

The existing hash-validated formal and RULER evaluators are reused unchanged.

1. **Contract and Flash smoke:** recheck every table/manifest hash, require
   CUDA, require Flash-only attention, reproduce Native/custom identity, and run
   one 1x PG-19 row for each primary arm.
2. **Primary natural likelihood:** primary pair only, fixed 20 PG-19 documents
   at 1x and 4x. Report paired document NLL differences separately by length.
3. **Primary capability:** primary pair only, frozen RULER core-4 at 4K and
   16K, 20 rows per task. Report task-level means and a paired-within-task,
   equal-task bootstrap interval.
4. **Conditional breadth:** only after the primary outputs are complete and
   controls resolve, run same-support geometric and historical p2 on the same
   endpoints. Breadth does not change the primary verdict.

The run is resumable by row identity. There is no silent retry, parameter
rescue, table search, gain sweep, head routing, adaptation, or training.

## 6. Monitoring and stop rules

- Monitor process liveness, completed row count, log growth, GPU memory,
  finite outputs, and result-file creation.
- Preserve commands, code/table/data hashes, manifests, raw rows, predictions,
  failures, exclusions, and partial completion under one private work-machine
  artifact root.
- Stop on any checkpoint/table/data hash drift, Native/custom identity failure,
  non-Flash attention, non-finite output, order/support mismatch, or author
  instruction.
- The author explicitly owns the wall-clock stop. Do not impose a synthetic
  three-hour cutoff and do not shut down before the author asks; after the stop,
  finish only bounded readback/summary work and write the result owner.

## 7. Code and validation state

- Builder/analyser: `scripts/analysis/scale_orbit_validation.py`, SHA-256
  `fc2c55122f0a813bc91cf8050b6a3c7fde0cffbea06a349f3580537783fad286`.
- GPU driver: `scripts/eval/run_scale_orbit_validation_5090.sh`, SHA-256
  `e0d523d92df8d8519aade360ae3e6112684e702bbc730deb0101f0ee9ab26cb2`.
- Focused test: `tests/test_scale_orbit_validation.py`, SHA-256
  `14f979813a3e5dc586628ac4dbec7bc7f085287bf8c1db6e6cdd1ca6ef982566`.
- Local standard-library/NumPy test result: `4/4` passed; Python compilation,
  shell syntax, deterministic asset construction, endpoint/order checks,
  exact-class separation, approximate-match continuity, phase-distance guard,
  and equal-task bootstrap logic passed.
- Work-machine code/assets were hash-read back successfully. A full evaluator
  contract preflight was intentionally stopped after the first cell because
  importing the existing Torch evaluator was disproportionately slow in the
  provider's no-card mode. It loaded no checkpoint and initialized no CUDA.

GPU smoke, model outputs, behavioural comparisons, result owner, manuscript
use, commit, and push remain unexecuted.
