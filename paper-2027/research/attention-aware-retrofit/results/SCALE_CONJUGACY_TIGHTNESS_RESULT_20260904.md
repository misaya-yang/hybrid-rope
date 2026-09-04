# Bounded-condition scale-conjugacy tightness result

## Material Passport

- **Date:** 2026-09-04.
- **Type:** CUDA numerical operator optimization; no model load or LM endpoint.
- **Status:** `COMPLETE BEST-FOUND SEARCH / POSITIVE CONTROL PASSED / PRIMARY
  NEGATIVE / MULTILEVEL STOPPED`.
- **Question:** Can bounded-condition real non-permutation conjugacies approach
  the finite-window Theorem 5 lower bounds on the five frozen RoPE tables?
- **Evidence labels:** the exact identity/permutation controls, positive-control
  values, sampled optimized errors, matrices, and hashes are **Observations**.
  Failure of this optimizer/protocol to improve identity is a **Negative
  result**. Global operator tightness remains **Unresolved** because the search
  is nonconvex and evaluates only a finite position set.
- **Preflight:**
  [`SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md`](../preflights/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md).
- **Theory owner:**
  [`FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md`](../theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md).

## 1. Decision

The primary best-found search is negative. Across all five frozen tables and
condition caps `1/8/64`, every selected solution is the identity control. The
sampled operator error is `2.000000` to displayed precision, as is the exact
best block-permutation error. Increasing the allowed condition number produces
no improvement.

The assay's post-primary nontrivial positive control passes: a known orthogonal
change of basis has realized error `5.16e-7`, while optimization from an
identity initialization reduces the identity error `1.998807` to `0.018797`,
well below the frozen threshold `0.199881`. The all-identity primary result is
therefore not explained by a completely inert optimizer.

This does **not** prove that no better general invertible `D_j` exists. It does
show that the supplied Pro lower bound does not furnish a tight or useful
comparative account of these tables under the tested search: four Ky-Fan bounds
are exactly zero at reported precision, Native is only `0.052914`, while every
best-found sampled error saturates near two. The stronger fixed-Native
projection lower bound is one for all five tables and is also non-ranking.

Per the registered stop rule, the 1,134-trajectory multilevel panel is not run.
No RULER, natural-generation, training, or model-inference experiment follows
from this result.

## 2. Frozen protocol and runtime

- Tables: Native, same-support geometric s4, legacy-u p2 log-s4, exact
  six-chain s4, and its ULP-jitter control.
- Table-manifest SHA-256:
  `14945bc0b5faaedcdea16a860beaac7271b28abf436c3a845449b87cea5b043b`.
- Primary: `s=4`, `N=1`, `L=4096`, condition caps `1/8/64`, 300 steps, three
  restarts, 64 training positions, 256 frozen evaluation positions, ten power
  iterations; 15 configurations and 45 trajectories.
- Optimizer SHA-256:
  `97c6f8dfff59e8e3679dd7ec3af884ba1df9cc2422f027e7023a2d6d1f96fd76`.
- Driver SHA-256:
  `775b71e4fe591b5ab8023befebcc0e8ef5519483a395febbadbf863f449604bb`.
- Positive-control script SHA-256:
  `6b1f48a3daf730a3cd2193071a788a93c6392237e5847f628f46f9c897d7ac17`.
- Runtime: RTX 4080 SUPER, PyTorch `2.8.0+cu128`, CUDA `12.8`, NumPy `2.3.2`,
  Python `3.12.3`.
- This is dense `128 x 128` operator algebra. FlashAttention is inapplicable;
  no attention kernel or language model is involved.

## 3. Primary observations

Each row below is identical across condition caps `1/8/64`. Full-precision
values and all restart records remain in the hash-bound JSON.

| Frozen table | Ky-Fan lower | Fixed-Native lower | Best sampled error | Exact block permutation | Selected solution |
| --- | ---: | ---: | ---: | ---: | --- |
| Native | `0.052914` | `1.000000` | `2.000000` | `2.000000` | identity at all caps |
| same-support geometric s4 | `0` | `1.000000` | `2.000001` | `2.000000` | identity at all caps |
| legacy-u p2 log-s4 | `0` | `1.000000` | `2.000001` | `2.000000` | identity at all caps |
| exact six-chain s4 | `0` | `1.000000` | `2.000000` | `2.000000` | identity at all caps |
| ULP-jitter chain s4 | `0` | `1.000000` | `2.000000` | `2.000000` | identity at all caps |

All 15 configurations have finite outputs, respect their condition caps, are
no worse than identity on the identical frozen evaluation positions, and have
matching matrix sidecar hashes. All 45 raw restart errors lie between
`2.000000` and `2.000001` to six decimals.

## 4. Positive control and validity boundary

The positive control constructs a target by conjugating four distinct planar
rotation frequencies with a known nontrivial orthogonal matrix. It uses the
same rotation builder, condition projection, power-norm training objective,
exact sampled spectral-norm evaluator, and Adam update path as the primary.

| Quantity | Value |
| --- | ---: |
| Identity error | `1.9988073111` |
| Known-witness error | `5.1633e-7` |
| Recovered error after 500 steps | `0.0187972840` |
| Required maximum | `0.1998807311` |

This validates that the machinery can recover a nontrivial conjugacy in a
resolving synthetic problem. It cannot establish global optimization on the
scale-covariance objective, nor can it convert sampled maxima into certified
continuous suprema.

## 5. Artifact identities

- Smoke JSON SHA-256:
  `b23fd0f60b8b1710b2d015f494a25d378c6cbe2389cb31d1f81dee43908a848b`.
- Primary JSON SHA-256:
  `f61cb7079d2c539e706db67cfbe9a3b47837bdb57f70131e507a7c383177eb28`.
- Primary 15-matrix hash inventory SHA-256:
  `ad6c89d5e5f1a90a8f4ad8a23e6f449cddb32d5e1a48f4f064e2fafc24f7efec`.
- Positive-control JSON / matrix SHA-256:
  `8a1a9ed80cf7ec7bc6423b174f2aed6fd6146958bdd0ef3d2872f08f26e7cecd` /
  `f6da6ca3cd5d7f970a404b401bf68425cca6f043917f779e39ffdf3c73e2c117`.
- Server-side raw JSON, matrices, logs, and inventory remain outside Git. Only
  machine-independent hashes and aggregate observations enter this owner.

## 6. Supported and unsupported claims

Supported at exact scope:

- the nontrivial synthetic control resolves;
- this 45-trajectory bounded-condition search finds no improvement over
  identity or permutation on the five frozen tables;
- the Ky-Fan lower bound is zero or tiny while sampled operator error saturates;
- these quantities do not rank the tables' known behavioural differences;
- the registered multilevel continuation has no execution qualification.

Unsupported:

- no general invertible conjugacy exists;
- the sampled maximum equals the continuous supremum;
- Theorem 5 is false;
- Theorem 5 predicts LM behaviour;
- any RoPE allocation or movement profile is selected;
- this result is itself a positive paper contribution.

## 7. Paper consequence

The exact theorem and its general real-orthogonal/separation extension remain
mathematically valid. This experiment removes the current empirical argument
for centering the paper on operator tightness: the lower bound is non-ranking
and the tested optimization saturates. Until a stronger similarity-specific
bound or a genuinely resolving averaged metric is derived prospectively, the
Pro operator route belongs in theory motivation or limitations, not as a
claimed explanation of the existing RoPE behaviour portfolio.
