# M4 exact-range factorial and frozen follow-up

Status: **180-run main grid and 12-run extreme follow-up complete; registered
runtime cross-swap and milestone analyses remain incomplete after the
bitwise-replay gate stopped. After result promotion, the user authorized
deletion of the run weights on 2026-07-27; retained artifacts are the raw
spec/result JSONs, audits, logs, curated evidence, and reports.**

Result report:
[`M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`](M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md).

Curated per-run evidence:
[`m4_exact_range_factorial_evidence_20260726.json`](m4_exact_range_factorial_evidence_20260726.json).

## Concern addressed

- `R27bE.1` / `AC.3`: separate the empirical operating rule from the Cosh
  family and from its finite-\(K\) range confound.
- `R27bE.2`: test held-out base and head-dimension structure.
- `R27bE.4` / `AC.3`: compare independently offset \(\tau\) values and a
  matched non-Cosh analytic schedule.
- `AC.1`: identify an interior-allocation effect while holding scalar
  range/support fixed.

## Existing evidence

The Phase16 99-run audit supports the rule only as a fallible basin prior. The
seed-42 exact-range experiment shows that interior allocation can matter after
sampled extrema and log-span are fixed, but it does not establish multi-seed
robustness, rule optimality, or Cosh uniqueness.

## Smallest missing evidence and executable plan

The registered main grid retains 180 runs:

\[
2\ \mathrm{bases}\times2\ L_{\rm train}\times3\ d_{\rm head}
\times3\ \mathrm{seeds}\times5\ \mathrm{arms}.
\]

The arms are Geo, Cosh at `0.75x/1.0x/1.25x` formula \(\tau\), and a matched
exponential. All arms share sampled extrema, log-span, initialization, token
order, optimizer, token budget, and evaluation anchors.

The follow-up does not enlarge that factorial. It adds only 12 boundary runs
at canonical base `500K`: the minimum-rule-\(\tau\)
`(L_train=1024,d_head=32,tau=1)` and maximum-rule-\(\tau\)
`(L_train=256,d_head=128,tau=8)` configurations, with Cosh `0.5x` and `1.5x`
formula \(\tau\), three seeds each.

## Matched-exponential definition

Let \(s_k\in[0,1]\) denote normalized log-frequency nodes after matching the
native sampled endpoints, and let \(u_k=k/(K-1)\). The pretraining-free
deformation budget is

\[
D(s)=\sqrt{\frac1K\sum_{k=0}^{K-1}(s_k-u_k)^2}.
\]

The exponential parameter is chosen by deterministic bisection so that
\(D(s_{\rm exp})=D(s_{\rm Cosh,rule})\). The two zero-displacement endpoints
are included. The audit records every normalized node, every actual frequency,
both deformation values, and their absolute error. The current numerical audit
covers all 12 structural configurations and has maximum matching error
`5.56e-17`.

## Frozen inference evaluation

Every final checkpoint is evaluated at `1x/2x/4x/8x` under:

1. `fixed_training_range`: retain the training base/span;
2. `target_matched_range`: retain the interior shape and set runtime base to
   the target window length, matching the repository FMR convention.

For Geo-trained and formula-Cosh-trained checkpoints, a `2x2` train-shape by
runtime-shape cross-swap loads both Geo and formula-Cosh interior grids under
the same runtime extrema/span. The co-adaptation metric is the mean penalty of
the wrong runtime shape:

\[
\frac12[(N_{\mathrm{GeoTrain,CoshRt}}-N_{\mathrm{GeoTrain,GeoRt}})
+(N_{\mathrm{CoshTrain,GeoRt}}-N_{\mathrm{CoshTrain,CoshRt}})].
\]

Two independent natural-text streams are frozen: WikiText validation and test.
Each stream/length uses four deterministic windows. Metrics are full
teacher-forced NLL, four equal-count prediction-position bins, and final-128
prediction-token NLL.

## Training dynamics

Geo, formula-Cosh, and matched-exponential retain checkpoints at
`25/50/75/100%`. Their fixed-range validation OOD rankings are reported
separately at `2x/4x/8x`. Runs completed before milestone retention was enabled
may be deterministically replayed only to recover missing checkpoint states;
the replay final model-state hash must exactly match the original.

## Statistical unit and aggregation

Anchors are averaged within a checkpoint and condition. Paired seed
differences are then averaged within each of the 12 structural configurations.
Headline means and the registered bootstrap operate on those 12 configuration
means, with three paired seeds per configuration. Evaluation lengths, anchors,
and 180 training runs do not increase the sample count.

## Stop conditions

- Missing or non-reproducible checkpoints stop the affected analysis.
- A replay whose final model-state hash differs is rejected.
- Frequency extrema/span or exponential-deformation mismatch stops training.
- Single-configuration or single-seed effects remain diagnostic and are not
  promoted into a general claim.

Raw outputs are isolated under
`results/theory/phase16_exact_range_factorial_m4_20260724/`.
