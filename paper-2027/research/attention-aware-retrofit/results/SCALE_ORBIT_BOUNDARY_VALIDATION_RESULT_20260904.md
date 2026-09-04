# Scale-orbit quantities do not select the useful static retrofit

- **Date:** 2026-09-04.
- **Status:** `COMPLETE / VALID PRIMARY CONTRAST / NEGATIVE SELECTOR RESULT`.
- **Evidence labels:** CPU identities and checked bounds are **Derived results**;
  executed PG-19 and RULER values are **Observations**; failure of the proposed
  quantities to rank the tested tables is a **Negative result** at this
  checkpoint, support, gain, factor, and assay.
- **Question:** Do exact scale-orbit class/boundary counts, approximate matching,
  Gram-tail lower bounds, or one-step permutation error predict mature-model
  behaviour among fixed-support, ordered, request-static log-frequency tables?
- **Preflight:**
  [`SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md`](../preflights/SCALE_ORBIT_BOUNDARY_VALIDATION_PREFLIGHT_20260903.md).
- **Protocol:** released OLMo-2-0425-1B-Instruct; `K=64`; factor four; one
  static table and gain `1.102585782722872` at every length; 20 fixed PG-19
  documents at 1x/4x and fresh core-4 RULER at 4K/16K.

## 1. Decision

The tested scale-orbit quantities are not behavioural selectors. The exact
minimum-class chain and its one-ULP perturbation change realized exact orbit
classes and boundary count from `6/6` to `64/64`, yet their model behaviour is
indistinguishable. Conversely, the useful log-p2 table and the failed exact
chain share zero Gram-tail lower bound and saturated permutation error, yet
their 4x behaviour differs dramatically.

The underlying lower-bound statements may remain mathematically correct under
their assumptions. What fails is the promotion from a representation-theoretic
obstruction to a table-ranking rule or explanation of the successful retrofit.
No closed-form movement profile follows from this panel.

## 2. Exact-boundary falsification

The two primary tables have the same 64 channels, fast/slow endpoints, support,
gain, and strict order. Their maximum phase difference through 16K is
`0.0009765625`. The ULP perturbation destroys exact equality while retaining
`58/64` approximate one-step matches at tolerance `1e-6 log(4)`.

| Quantity | Exact chain | ULP-jitter chain |
| --- | ---: | ---: |
| Exact orbit classes / boundaries | `6 / 6` | `64 / 64` |
| Continuous/discrete epsilon lower bound | `0 / 0` | `1.01e-7 / 1.01e-7` |
| One-step permutation operator error | `2` | `2` |
| PG-19 NLL, 1x | `3.046837` | `3.046771` |
| PG-19 NLL, 4x | `7.253505` | `7.253491` |
| Core-4, 4K | `.6300` | `.6325` |
| Core-4, 16K | `0` | `0` |

Positive paired deltas favour the exact chain. PG-19 deltas are
`-0.000066`, 95% interval `[-0.001112,+0.001004]`, at 1x and
`-0.000013`, `[-0.003682,+0.003715]`, at 4x. Core-4 deltas are `-0.0025`,
`[-0.0075,0]`, at 4K and exactly zero at 16K. The discontinuous exact count is
therefore not reflected in the tested model endpoints.

Both primary arms also fail the intended extension: their 4x NLL is about
`7.2535` and every 16K core-4 cell is zero. Minimizing exact orbit growth is not
a usable construction rule here.

## 3. Stronger ranking counterexample

| Table | Gram epsilon lower bound | Permutation error | PG-19 4x | Core-4 16K |
| --- | ---: | ---: | ---: | ---: |
| Minimum-class exact chain | `0` | `2` | `7.253505` | `0` |
| Same-support geometric | `1.22e-4` | `2` | `7.116691` | `0` |
| Legacy-u log-p2 | `0` | `2` | **`3.081946`** | **`.4025`** |

The p2 and exact-chain arms have the same value for both proposed coarse
selectors but opposite behavioural utility. The geometric control further
shows that matching support, gain, and order is insufficient. Static geometry
can constrain what is possible, but the checkpoint's learned Q/K readout of the
specific table remains load-bearing.

The breadth driver began a redundant p2 RULER replay and was stopped after its
first row once the existing same-data current-p2 owner was verified. That
partial output is preserved but is not evidence and does not enter the table.

## 4. CPU transport-residual follow-up

A separately frozen CPU-only follow-up applied the repository's older
isotropic transport analysis to the five exact tensors after the behavioural
panel. `D*` is the best in-window rotation-basis reconstruction error available
to arbitrary fixed Q/K maps, normalized by the no-positional-signal energy. It
is paired with phase-safe fraction rather than used alone.

| Table | `D0` | `D*` | Safe at 8K | Safe at 16K |
| --- | ---: | ---: | ---: | ---: |
| Native | `0` | `0` | `.500` | `.500` |
| Legacy-u log-p2 | `.5220` | **`.2362`** | **`1.000`** | **`1.000`** |
| Minimum-class chain | `1.0102` | `.4821` | `.969` | `.594` |
| ULP-jitter chain | `1.0102` | `.4821` | `.969` | `.594` |
| Same-support geometric | `1.0085` | `.4831` | `1.000` | `.516` |

The p2 table strictly dominates all three failed non-Native tables on lower
`D*` and no lower phase safety at either target. The exact/ULP pair remains
identical under this continuous calculation, as its behavioural result
requires. Thus `D*` is a useful diagnostic for why these extreme constructions
are poor, while the proposed orbit quantities are not.

This hard-panel separation does **not** revive `D*` as a selector. The earlier
[`prospective axis falsification`](../analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md)
contains a stronger counterexample: a one-turn
floor table had `D*=.0192` and zero phase risk yet scored zero on 8K RULER, and
the eight-point `D*` ranking had the wrong correlation sign. That owner remains
valid. The new result is compatible with `D*` as an in-window repair bound, not
as a sufficient design objective or general behavioural explanation.

This does not identify a profile. The analysis uses isotropic content, loads no
checkpoint weights or activations, and was run after the behavioural outcomes
were known. The next evidential step would require a new table chosen by a
frozen cost/coverage rule and then tested prospectively; fitting a scalarization
to these five outcomes would be postdiction.

The frozen inline input manifest and output receipt SHA-256 are
`e9bd3187e3846645cd293d7de5e482439f73f2234398f4a4956ae0194809d6bd` and
`d1ec8b05fa683fdae33516b57eea246a580d9859f191ba3c2e4440f595353fd1`.
CUDA was hidden and the runner verified that PyTorch was never imported.

## 5. Artifact identity

- CPU asset manifest / metrics SHA-256:
  `14945bc0b5faaedcdea16a860beaac7271b28abf436c3a845449b87cea5b043b` /
  `f859cc3cde76964e62d20f1f6eced4b1f982d53fd87c9d03b34be781324e4a40`.
- Exact-chain PG-19 examples / results:
  `b932c8971dad6d5646d65752bff8a7201e3cff35f2447576ec2dbbc840527f88` /
  `f730e11eb81b99a61de75df984913c80f5075ff5e023f6237d75468549a584b5`.
- Jitter-chain PG-19 examples / results:
  `50600c4d84e934d6a22e25fe2abbd7a9e054e1a5ce18644e44c72a06829622ca` /
  `b06bf47ff00beb14e6c19b917776067ea0248266d78352a9e1da7e42b9655f97`.
- Exact-chain RULER examples / results:
  `f48986ffb1109bf7b1ec9e74af2709a9856103b81ff6b2845931092055d48156` /
  `d719c7ce41b0a2c352f03319ff9d51cf7df442a4cf1be3e5c14d10402b0020b1`.
- Jitter-chain RULER examples / results:
  `8a5611f384f5e3bdbfdec688d9a15d461e7f7c368b4b2e47c56bbb0de28ba422` /
  `356d8b377a265879491aedb9b4f348b94c9f74fc932737e85087b26a43cacf23`.

Raw rows, predictions, tables, manifests, logs, and partial outputs remain on
the work machine. Repository prose contains no private machine path.

## 6. Supported and unsupported claims

Supported: exact orbit count is numerically unstable to an ULP perturbation
that is behaviourally invisible in this panel; the tested approximate/Gram and
permutation quantities do not rank the successful p2 table over failed tables;
the minimum-class construction fails as a factor-four retrofit; and the older
`D*` plus phase-safety axes post hoc distinguish p2 from these three failures.

Unsupported: a universal impossibility theorem for every scale-orbit statistic;
a proof that no representation-theoretic quantity can help; revival or
prospective validation of `D*` as a selector; a unique Native checkpoint-
derived `m_k`; or any claim beyond this checkpoint, factor, support, gain, and
protocol. The orbit route closes as a selector, not as pure mathematics.
