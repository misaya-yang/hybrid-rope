H800 3h Follow-up (Train+Eval) Summary

- ts: 2026-02-13_085358
- variants: 6
- lengths: [2048, 4096, 8192, 12288, 14336, 16384]
- random_start seeds: [42, 123, 777]

## Leaderboard (random_start @16K)

| rank | variant | group | ppl@16k | collapse_ratio(16k/2k) | delta_vs_geo10k |
|---:|---|---|---:|---:|---:|
| 1 | sigmoid_th100k_steep8_mid0.5_omf0.3 | sigmoid_high_theta | 25.847 | 1.232 | -51.142 |
| 2 | sigmoid_th500k_steep8_mid0.5_omf0.3 | sigmoid_high_theta | 26.116 | 1.258 | -50.874 |
| 3 | geo_500k | theta_scaling | 27.217 | 1.282 | -49.772 |
| 4 | hybrid_basegeo500k_alpha0.2 | hybrid_high_theta | 27.487 | 1.332 | -49.503 |
| 5 | sigmoid_steep8_mid0.5_omf0.3 | sigmoid | 27.870 | 1.432 | -49.120 |
| 6 | geo_10k_baseline | baseline | 76.989 | 3.615 | +0.000 |

## Multi-Length (random_start mean over seeds)

| variant | 2k | 4k | 8k | 12k | 14k | 16k |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid_th100k_steep8_mid0.5_omf0.3 | 20.977 | 19.180 | 23.891 | 24.226 | 25.103 | 25.847 |
| sigmoid_th500k_steep8_mid0.5_omf0.3 | 20.755 | 19.100 | 23.855 | 24.351 | 25.330 | 26.116 |
| geo_500k | 21.224 | 19.470 | 24.310 | 24.834 | 26.139 | 27.217 |
| hybrid_basegeo500k_alpha0.2 | 20.638 | 18.899 | 24.033 | 25.167 | 26.419 | 27.487 |
| sigmoid_steep8_mid0.5_omf0.3 | 19.459 | 17.989 | 22.995 | 24.081 | 25.992 | 27.870 |
| geo_10k_baseline | 21.297 | 19.785 | 34.406 | 53.802 | 65.555 | 76.989 |

## Notes

- All variants are retrained under identical data/order/seed, then re-evaluated with stricter slicing+seed settings.
- Lower ppl@16k and lower collapse ratio indicate better extrapolation stability.