# Completed evidence reports

Read-only snapshot from the existing experiment server. No generation was rerun.
The source manifest distinguishes original report intervals, raw stored-score
aggregation, and full-output rescoring.

| Report | Role |
|---|---|
| [OLMo clean16K](olmo_clean16k_ruler200.json) | Full-13, 2600 pairs, second-family clean confirmation |
| [OLMo Natural-QA](olmo_naturalqa631.json) | Five tasks, 631 pairs, source-document clustered uncertainty |
| [Llama clean T/C/P](llama_clean_matched_dose_c.json) | Equal-displacement comparison at16K and32K |
| [Llama C/P](llama_clean_control_vs_mrpro.json) | Supporting three-table contrast |
| [Llama clean Native8K](llama_clean_native8k.json) | T/P/Native, 650 pairs |
| [Llama LongBench v2](llama_longbench_v2_8k32k.json) | Actual8K–32K input subset, 117 pairs |
| [OLMo Native half-turn](olmo_native_halfturn_four_arm.json) | Four-arm native allocation exploration |
| [OLMo Native NCP](olmo_native_ncp.json) | Public-parameter native allocation, same development panel |
| [Snapshot and sources](completed_results_snapshot_20260915.json) | Remote paths, hashes and verification scope |
| [Raw verification](completed_results_raw_verification.json) | Paired identities, stored scores and runtime/table contracts |
| [Derived findings](completed_results_derived_findings.json) | Rescoring, table identities and task concentration |
| [Pro6000 Llama S16 gate](pro6000_llama_s16_128k_gate.json) | 128K Full-13 and PPL endpoint pressure test |
| [Pro6000 Llama En.Dia](pro6000_llama_s16_infinite_en_dia.json) | 100K–128K InfiniteBench dialogue accuracy |
| [Pro6000 Llama En.QA](pro6000_llama_s16_infinite_en_qa.json) | 100K–128K InfiniteBench LongBook QA-F1 |
| [Pro6000 Qwen 256K](pro6000_qwen3b_s8_256k_health.json) | S8 single-NIAH and five-document LongBook PPL |
| [Pro6000 Qwen 128K En.QA](pro6000_qwen3b_s4_128k_en_qa.json) | S4 complete-context LongBook QA-F1 |
| [RULER sampling stability](pro6000_ruler_sampling_stability.json) | Finite-population n=5–200 subsampling calibration |

[Paper interpretation](../../../paper-2027/research/COMPLETED_EXPERIMENTS_PAPER_VALUE_20260915.md)
contains the proposed manuscript integration. [Execution entry points](../README.md).
