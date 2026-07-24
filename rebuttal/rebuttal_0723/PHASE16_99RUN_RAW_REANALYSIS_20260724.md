# Phase16 99-run raw reanalysis: what the formula actually supports

Date: 2026-07-24

Status: `internal_verified`

Scope: analysis only. No new training, no paper-table edits, and no claim upgrade.

## Technical summary

The local Phase16 source bundle is recoverable and reconciles exactly with the
tracked 99-row sanitized manifest. It contains 45 seed-42 pilot runs and 54
additional confirmation runs over nine model configurations.

Using one common metric for every arm---runner-defined weighted extrapolation
NLL---the preregistered formula

\[
\tau_{\mathrm{formula}}=d_{\mathrm{head}}/\sqrt{L_{\mathrm{train}}}
\]

beats midpoint-Geo in 7/9 configuration means and 18/27 paired
configuration-by-seed comparisons. The equal-weight descriptive mean is
\(-0.01331\) NLL, equivalent to about 1.32% lower PPL. This is a real but small
and variable signal.

It is **not evidence that the formula is near-optimal**. On the two confirmation
seeds that were not used to select the neighboring alternative, the formula
beats that alternative in only 3/9 configurations and 8/18 individual pairs.
Its mean gap is \(+0.02210\) NLL, about 2.24% worse PPL. Among the three
commonly evaluated candidates (Geo, formula, and one pilot-selected neighbor),
the formula ranks first, second, and third in exactly three configurations each.

The reviewer-safe conclusion is therefore:

> In this small staged study, the formula is a useful but fallible operating
> prior relative to midpoint-Geo. The experiment does not establish a global or
> near-optimal scaling law, and it does not isolate allocation shape from
> frequency range.

## Evidence recovered and validated

| Artifact | Audit result |
| --- | --- |
| `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/` | 99 run directories |
| `runs/*/result.json` | 99/99 present and readable |
| `runs/*/spec.json` | 99/99 present and consistent with plans/results |
| `pilot_plan.json` / `confirm_plan.json` | present; 45 pilot + 54 confirmation runs |
| `runs/*/checkpoint_last.pt` | 98/99 present; one confirmation checkpoint missing |
| `data/curated/phase16_99run_manifest.csv` | regenerated from raw source and byte-identical |
| `scripts/validate_rebuttal_evidence_bundle.py` | PASS on 2026-07-24 |

The missing checkpoint is
`confirm_L1024_H16_Dh32_tau1.25_seed256/checkpoint_last.pt`. Its `spec.json` and
`result.json` are present, so it does not affect this result-level reanalysis;
it prevents claiming complete checkpoint-level preservation.

The tracked metadata says the original source directory is absent because the
raw tree is ignored and is not part of a clean checkout. On this workstation,
however, the local ignored source is present. This distinction must remain
explicit in any public provenance statement.

## Protocol and common metric

Phase16 used the `local_m4` profile:

- 50M model tier and local WikiText;
- \(L_{\mathrm{train}}\in\{256,512,1024\}\);
- head counts \(H\in\{4,8,16\}\), hence
  \(d_{\mathrm{head}}\in\{128,64,32\}\);
- nine configurations total;
- 8,388,608 training tokens per run;
- seed 42 over five tau values per configuration;
- seeds 137 and 256 only for midpoint-Geo, formula tau, and one
  pilot-selected neighboring tau.

The generated historical report ranked `selection_score`, but that score is not
comparable across stages: pilot adds a passkey term and has DSR disabled, while
confirmation adds both passkey and DSR terms with different trial counts. It
also mixed one-seed pilot-only arms with three-seed confirmed arms.

This reanalysis therefore discards the composite score and uses only the PPL
component shared by all runs. For extrapolation ratios
\(r\in\{2,4,8\}\), define

\[
\operatorname{NLL}_{\mathrm{ext}}
=
\frac{\sum_r \log_2(r+1)\,\log(\operatorname{PPL}_{rL})}
     {\sum_r \log_2(r+1)}.
\]

All deltas below are `candidate - reference`; negative is better. Formula-vs-Geo
uses all three matched seeds. Formula-vs-alternative uses only seeds 137 and
256, which did not participate in selecting the alternative.

## The formula beats Geo modestly, but does not predict the best neighbor

| Configuration | Formula tau | Selected alternative | Formula - Geo NLL, 3-seed mean | Formula wins vs Geo | Formula - alternative NLL, held-out mean | Formula rank / 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L256_H4_Dh128` | 8.00 | 10.00 | -0.01948 | 2/3 | +0.02704 | 2 |
| `L256_H8_Dh64` | 4.00 | 5.00 | -0.03399 | 3/3 | +0.00385 | 2 |
| `L256_H16_Dh32` | 2.00 | 2.50 | -0.04887 | 2/3 | -0.00599 | 1 |
| `L512_H4_Dh128` | 5.66 | 4.24 | -0.01657 | 2/3 | -0.02546 | 1 |
| `L512_H8_Dh64` | 2.83 | 4.24 | -0.03657 | 3/3 | -0.01624 | 1 |
| `L512_H16_Dh32` | 1.41 | 1.77 | +0.04408 | 1/3 | +0.13758 | 3 |
| `L1024_H4_Dh128` | 4.00 | 5.00 | -0.01673 | 2/3 | +0.06104 | 2 |
| `L1024_H8_Dh64` | 2.00 | 2.50 | -0.00950 | 2/3 | +0.00597 | 3 |
| `L1024_H16_Dh32` | 1.00 | 1.25 | +0.01785 | 1/3 | +0.01115 | 3 |

Summary:

| Comparison | Result | Interpretation |
| --- | ---: | --- |
| Formula vs midpoint-Geo, configuration means | 7/9 wins | useful direction, not universal |
| Formula vs midpoint-Geo, paired runs | 18/27 wins | substantial seed/config variability |
| Equal-weight Formula - Geo mean | -0.01331 NLL | about 1.32% lower PPL, descriptive only |
| Formula vs selected neighbor, held-out config means | 3/9 wins | no robust local-optimum prediction |
| Formula vs selected neighbor, held-out paired runs | 8/18 wins | formula loses more often than it wins |
| Equal-weight Formula - neighbor mean | +0.02210 NLL | about 2.24% higher PPL |
| Held-out rank among three candidates | 3 first / 3 second / 3 third | not a stable best-point selector |

These configuration and seed observations are not independent iid samples, so
the equal-weight means are descriptive summaries, not pooled estimators with a
valid 27-sample confidence interval.

## The effect is not specifically a long-range effect

Formula-minus-Geo deltas averaged over the 27 paired runs are:

| Evaluation point | Mean NLL delta | Approximate PPL change | Pair wins |
| --- | ---: | ---: | ---: |
| In-domain \(1\times\) | -0.01612 | -1.60% | 17/27 |
| \(2\times\) | -0.01921 | -1.90% | 18/27 |
| \(4\times\) | -0.00987 | -0.98% | 17/27 |
| \(8\times\) | -0.01288 | -1.28% | 17/27 |

There is no monotonic strengthening with extrapolation distance, and the
in-domain effect is of similar magnitude. Phase16 therefore does not identify a
special long-context scaling mechanism. It is compatible with a general
frequency-table intervention whose benefit mixes range, endpoint, and internal
allocation effects.

The clearest failure corner is \(d_{\mathrm{head}}=32\) at
\(L\in\{512,1024\}\), where the formula loses to Geo on the three-seed mean.
This may reflect finite-channel behavior, a small-tau floor, optimization noise,
or this specific data/model regime. The experiment cannot distinguish those
explanations.

## What the historical report got wrong

The old generated report's `exact best 3/9`, `top-2 6/9`, and `top-3 8/9`
language is not a valid common-protocol rank analysis because it:

1. averaged pilot and confirmation `selection_score` despite different
   passkey/DSR components and trial counts;
2. ranked arms with one seed against arms with three seeds;
3. reused the pilot seed that selected the neighboring arm;
4. treated nine staged configurations as if they were 27 comparable
   configurations.

The raw data support the narrower 7/9 Formula-vs-Geo statement. The held-out
neighbor comparison added here is more damaging to the stronger claim: the
formula is best in only 3/9 configurations even among just three candidates.

## Relation to the current range-versus-shape question

Phase16 cannot determine whether Cosh is the right allocation shape. In the
actual midpoint implementation, changing raw tau simultaneously changes:

1. the highest sampled frequency;
2. the lowest sampled frequency and realized log-span;
3. the nonlinear spacing of channels inside that span.

Consequently, neither the Formula-vs-Geo gain nor the neighboring-tau ranking
can be attributed to shape alone. Phase16 also contains no matched exponential
shape arm and no independently controlled target-aware range arm.

Its main value for the current project is negative and diagnostic:

- it confirms that schedule choice matters at fixed model size;
- it shows that the closed-form formula is not a dependable optimum selector;
- it strengthens the need for the preregistered matched-range
  `Geo / Anchored-Cosh / Anchored-Exp` experiment;
- it argues against spending GPU budget on another raw-tau sweep.

The 59.9GB checkpoint tree does not need to be copied to the 5090 for this
conclusion. The raw results and plans are sufficient for Phase16 audit; the new
matched-range experiment requires its own three-arm checkpoints.

## Reviewer-safe disposition

Safe wording:

> A staged 99-run study covering nine small-model configurations found that the
> closed-form tau rule outperformed midpoint-Geo in seven of nine matched
> three-seed configuration means. However, it was best among Geo, the formula,
> and a pilot-selected neighboring value in only three of nine held-out
> comparisons. We therefore treat it as a useful but fallible operating prior,
> not as a proven or globally near-optimal scaling law.

Do not restore any of the following:

- `27 independent configurations`;
- `all configurations are within 1%`;
- `R^2 > 0.95/0.99 validates the scaling law`;
- `the formula is near-optimal across the grid`;
- `Phase16 proves the Cosh shape or the exponent -1/2`;
- `the comparison is against native endpoint RoPE`.

## Remaining limitations and next action

- This is a 50M-tier local-WikiText study with 8.39M training tokens per run;
  it is supporting evidence, not a large-model or downstream result.
- The baseline is midpoint-Geo, not native endpoint RoPE.
- Only two seeds are independent of pilot selection for the neighboring-arm
  comparison.
- The nine configurations share the same harness and corpus and should not be
  treated as nine independent scientific replications.
- One checkpoint is missing, although every result record needed here survives.

The next decision-useful experiment is the already specified matched-range
three-arm seed-42 gate. Run no additional raw-tau sweep unless that experiment
first establishes that allocation shape adds value once range is controlled.

## Source anchors

- `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/`
- `data/curated/phase16_99run_manifest.csv`
- `scripts/core_text_phases/phase16_formula_optimality_sweep.py`
- `scripts/core_text_phases/export_phase16_manifest.py`
- `rebuttal/pre_rebuttal/FULL_PAPER_INTEGRITY_AUDIT_20260713.md`
- `rebuttal/pre_rebuttal/FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md`
- `rebuttal/pre_rebuttal/THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md`
- `rebuttal/rebuttal_0723/ROPE_RANGE_SHAPE_MAPPING_THEORY_AND_5090_PLAN_20260724.md`
