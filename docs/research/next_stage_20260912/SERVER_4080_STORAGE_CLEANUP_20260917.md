# 4080服务器资产归档与清理（2026-09-17）

## 结论

服务器`connect.westc.seetacloud.com:37849`的数据盘已从`64 GiB / 75 GiB`
（85%）降至`31 GiB / 75 GiB`（41%），释放约`33.15 GiB`。本轮没有删除
当前四模型、CA-NCP代码、正式实验逐行输出或仍在使用的CPU评测面板。

清理前先将本机论文正结果的最小可复核证据包同步到
`results/paper_positive_raw_20260917/4080/`。该目录是本机ignored raw镜像，
不进入Git；其清单为`ARCHIVE_MANIFEST.json`，SHA256为
`c9311dbe26d1b60fbc7c99ecd6cdeb2be2a943df9e89e104180a053aa23c9aca`。

- 归档大小：约45 MiB；
- 文件数：502；
- 逐行结果：64个`generations.jsonl`、`lm_rows.jsonl`或训练评价raw JSON；
- 远端/本地逐文件SHA256：64/64一致，无缺失、无不匹配；
- 每个结果包同时保留对应contract、summary、table和正式report（存在时）。

## 已覆盖的论文证据

| 证据 | 本地raw owner |
|---|---|
| A39：Llama classic与clean 32K | `tailspline_llama_s4_classic/`、`tailspline_llama_s4_32k_ruler200_clean/` |
| A40：OLMo classic | `tailspline_olmo_s4_classic/` |
| A45：Llama Natural-QA631 | `tailspline_llama_s4_naturalqa631/` |
| A46：Llama clean 16K | `tailspline_llama_s4_16k_ruler50_clean/` |
| A49：OLMo clean 16K | `tailspline_olmo_s4_16k_ruler200_clean/` |
| A50：OLMo Natural-QA631 | `tailspline_olmo_s4_naturalqa631/` |
| A51：matched-displacement C | `strong_evidence/llama_s4_clean_matched_dose_c/` |
| A52：Llama Native 8K三臂 | `strong_evidence/llama_s4_clean_native_x5/`、`iclr2027_three_track_sprint_20260915/native_classic8k/` |
| A53：Llama LongBench-v2 | `tailspline_llama_s4_longbench_v2_8k32k/` |
| A54/A62：OLMo Native NCP | `olmo_native_contrastive_proximal/`、`olmo_native_halfturn_phase/`、`native_research_20260916/` |
| A60：Llama/OLMo官方静态YaRN | `official_yarn_full13/`、`official_yarn_quick/` |
| A01：固定支持多seed | `iclr_exact_range_multiseed/`中的评价raw；模型checkpoint未保留 |

Pro6000产生的A55--A59以及70B A61不属于这台4080的本轮清理来源；其Git内正式
报告未被改动。本记录只声明4080上实际核对并同步的raw。

## 已删除

### 大型可再生中间量

- `native_research_20260916/runs/qkv_capture96`：约7.4 GB；96条机制样本的
  Q/K/V捕获矩阵。已在本地保留contract、完整index和partial index。
- `rope_pro_block_calibration_20260910/native_full_rows`：约7.5 GB；旧block
  calibration整行张量。已在本地保留runtime和manifest。
- `rotary_budget_20260908/data`与`inductor_cache`：约5.3 GB；可重下语料和
  编译缓存，不是论文评测raw。

### checkpoint与停用模型

- `fixed_support_joint_151m_20260912/.../runs`：约3.7 GB；保留服务器评价
  `rows.jsonl`，只删除训练checkpoint。
- `iclr_exact_range_multiseed/seed_{137,256}/runs`：合计约3.0 GB；本地已保存
  两个seed的评价raw，Git内A01摘要不变。
- `qwen25_1p5b_32k`：约3.1 GB；停用的Qwen-1.5B权重。当前Qwen-3B模型仍在
  `rope_qwen_baseline_20260907/model`。若重新启用1.5B旧线需重下权重。

### 未完成、重复或失败资产

- 未完成的`tailspline_llama_s4_32k_full500`与仅plan-only的
  `tailspline_qwen25_s4_64k128k_clean`；
- `fixed_rope_three_interfaces_20260913/queue`与旧通用`panels`；当前论文和
  当前模型专属面板均保留；
- `band_mini_20260913`、旧`pilot_01`、`wd_preserved_buggy_runs`、失效left-pad
  run、invalid-tokenizer资产和所有明确命名为canary/smoke/probe/partial-killed的
  运行目录；
- `native_windows_code_01/02`、旧tar包等未版本化代码快照。正式代码仍由Git
  仓库`hybrid-rope`持有。

## 已保留的CPU资产和模型

- Llama：clean 32K、clean 16K、classic/PPL46、Natural-QA631、LongBench-v2、
  S16/128K gate及NIAH heatmap面板；
- OLMo：clean 16K、classic/PPL46、Natural-QA631、Native RULER 13x10与
  13x100、LM128 token矩阵、Native-Z5数据；
- Qwen-3B：`ruler_prepared_01`、`prepared`、`prepared_v2`、128K相关manifest和
  现有逐行结果；
- GLM：`/root/models/GLM-4-9B-0414`完整保留。这台4080在清理前没有GLM专属
  tokenized benchmark panel；相关GLM结果来源在Pro6000，不把不存在的资产记成
  本轮保留项；
- 模型：Llama-3-8B约15 GB、Qwen2.5-3B约5.8 GB、OLMo-2-1B约2.8 GB、
  GLM-4-9B约18 GB；
- 当前`ca_ncp_native_20260917`构造、基线复用回执和CPU readiness全部保留。

## 暂不清理的历史目录

以下目录仍占约3 GB，但含历史raw或可复用数据，尚未逐项建立同等级本地raw
镜像，因此本轮有意保留：

- `nongeometric_screen_20260909`（约1.1 GB）；
- `position_observability_20260908`（约482 MB）；
- `ffn_review_execution_20260904`（约383 MB）；
- `bm_transfer_20260908`（约292 MB）；
- `sparse_memory_20260908`（约274 MB）；
- `olmo_fast_screen_20260908`（约270 MB）；
- `benchmarks/longbench_v2`（约444 MB，基准源数据）。

如果后续继续减盘，应先为其中已经进入论文或证据索引的逐行结果建立raw owner，
再删除远端副本；不能仅因目录旧而整目录清空。
