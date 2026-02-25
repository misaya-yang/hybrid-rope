# H100 TODO

## Immediate

1. Run `scripts/bootstrap_h100_env.sh` and archive env snapshot.
2. Fill `configs/experiment_matrix_1p5b.yaml` with real token budget and launcher args.
3. Start Phase P0 pilot (`geo_500k` vs `hybrid_a0.2_t100k`) and save raw JSON under `results/raw/`.

## Main

4. Complete 3-seed runs for three core configs:
   - `geo_500k`
   - `hybrid_a0.2_t100k`
   - `anchpoly_p3.9_omf0.3_t500k`
5. Run long-context benchmarks (LongBench/RULER) on best two configs.
6. Run lm-eval short-context sanity set (HellaSwag/PIQA/ARC/Wino).

## Reporting

7. Generate figures with `scripts/plot_h100_results.py`.
8. Write one-page claim summary with:
   - core win/loss statement,
   - confidence interval (seed variance),
   - failure boundary (if any).
