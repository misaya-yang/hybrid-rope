# FOLLOWUP MECHANISM P0 SUMMARY

- Timestamp: 2026-02-13_075401
- Device: NVIDIA A800-SXM4-80GB
- Lengths (2x2): [12288, 14336, 16384]
- Strategies: ['sequential', 'random_start']
- Chunks per condition: 5

## 1) 2x2 Factor Ablation (subset_0)

| Variant | Strategy | PPL@12K | PPL@14K | PPL@16K |
|---|---|---:|---:|---:|
| base_orig | sequential | 53.25 | 127.74 | 190.566 |
| base_orig | random_start | 58.119 | 122.272 | 181.272 |
| base_hybridfreq | sequential | 6974.389 | 7517.583 | 7612.143 |
| base_hybridfreq | random_start | 6470.444 | 8002.426 | 6802.191 |
| lora_origfreq | sequential | 189.379 | 231.308 | 231.308 |
| lora_origfreq | random_start | 201.593 | 243.167 | 235.686 |
| lora_hybridfreq | sequential | 14.198 | 14.973 | 15.4 |
| lora_hybridfreq | random_start | 15.839 | 15.497 | 15.304 |

## 2) 16K Robustness Across Validation Subsets

| Model | Subset | Strategy | PPL@16K |
|---|---|---|---:|
| base_orig | subset_0 | sequential | 190.566 |
| base_orig | subset_0 | random_start | 181.272 |
| base_orig | subset_1 | sequential | 214.594 |
| base_orig | subset_1 | random_start | 210.608 |
| base_orig | subset_2 | sequential | 192.963 |
| base_orig | subset_2 | random_start | 204.129 |
| lora_hybridfreq | subset_0 | sequential | 15.4 |
| lora_hybridfreq | subset_0 | random_start | 15.304 |
| lora_hybridfreq | subset_1 | sequential | 15.067 |
| lora_hybridfreq | subset_1 | random_start | 15.594 |
| lora_hybridfreq | subset_2 | sequential | 16.24 |
| lora_hybridfreq | subset_2 | random_start | 15.594 |

## 3) 16K Loss Curve Jump Check

| Model | abrupt jump | head mean NLL | tail mean NLL | tail/head | max jump idx | jump/scale |
|---|---|---:|---:|---:|---:|---:|
| base_orig | False | 2.6422 | 9.5076 | 3.598 | 16250 | 7.218 |
| lora_hybridfreq | False | 2.6234 | 2.7209 | 1.037 | 2295 | 8.328 |

## 4) 16K Attention Probe

| Model | entropy mean | sink(128) mean | recent(512) mean | long(>=4k) mean | mean distance |
|---|---:|---:|---:|---:|---:|
| base_orig | 0.4374 | 0.3250 | 0.2433 | 0.5926 | 7542.4 |
| lora_hybridfreq | 0.6557 | 0.1083 | 0.2803 | 0.4357 | 4906.0 |

## Key Takeaways

- Numeric anomalies: 0
- base_orig vs lora_hybridfreq @16K ratio range across subsets/strategies: 11.845x to 14.243x
- lora_origfreq vs base_orig @16K (sequential, subset_0): 0.824x
- base_hybridfreq vs base_orig @16K (sequential, subset_0): 0.025x
