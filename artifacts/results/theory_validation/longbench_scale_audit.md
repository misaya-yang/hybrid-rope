# LongBench Scale Audit

- Source: `batch_report_2026-02-23_downstream_eval/report/method_metrics_best_available.csv`
- Value column: `longbench_avg`
- Check: `score_pct == 100 * score_raw` (tol=1.0e-06)

- Max abs error: `0.000000e+00`
- Scaling check: `PASS`
- Ranking identity (raw vs pct): `PASS`

| rank | method | raw (0-1) | pct (0-100) |
|---:|---|---:|---:|
| 1 | anchored_sigmoid | 0.071724 | 7.1724 |
| 2 | sigmoid | 0.068671 | 6.8671 |
| 3 | pi | 0.066520 | 6.6520 |
| 4 | yarn | 0.065635 | 6.5635 |
| 5 | baseline | 0.062643 | 6.2643 |

## Ranking

- raw: `anchored_sigmoid, sigmoid, pi, yarn, baseline`
- pct: `anchored_sigmoid, sigmoid, pi, yarn, baseline`

## Conclusion

- The metric-unit conversion is a strict linear scaling. Relative ordering and pairwise deltas are unchanged; only presentation units differ.
