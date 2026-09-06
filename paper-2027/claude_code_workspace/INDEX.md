# INDEX — claude_code_workspace 细目（2026-09-06 更新）

状态标记：**活跃** / 历史（仍被引用）/ 已取代（结论作废或被覆盖）/ 存档（只读留档）。

根文件：`README.md`（总览与边界）、`INDEX.md`（本文件）、`LESSONS.md`
（Round 10-12 踩坑记录，含本次清理的提交纪律教训）。

## round12_20260906/ — 当前主线

| 文件 | 说明 | 状态 |
|---|---|---|
| `REPORT_ROUND12_20260906.md` | **总结报告 + 500M 恢复清单（§7）** | 活跃，首选入口 |
| `PREGLUCTION_ROUND12.md` | 预注册 v1：冻结表（N/Z/Y/M sha+gain）、任务、评分口径 | 历史（v2 增补有效） |
| `PREGLUCTION_ROUND12_V2_ADDENDUM.md` | v2：P0 诊断置顶、7B 探测门控、Track B 降级 | 历史 |
| `RUNBOOK_ROUND12.md` | 执行手册 v2（已更正 16GB→32GB 实测） | 历史 |
| `PHASE0_INTERPRETIVE_MEMO.md` | P0 诊断中文解读（§2.3/§5.2/分叉） | 历史 |
| `receipts/` | 空——Round 12 回执全在服务器（见下） | — |

### round12_20260906/code/（与服务器逐文件 md5 一致；改动须两端同步，服务器现关机）

- **表/任务/评分**：`rope_tables.py`（冻结表构造+verify）、`task_rows.py`、
  `build_ext_tasks.py`（Qwen 64K/128K EXT 行）、`build_y2_canon.py`（忠实 YaRN 表）、
  `check_tok_equiv.py`（1B≡7B tokenizer sha）、`equiv_check.py`、`scoring.py`
- **Track A 评估**：`track_a_eval.py`（含 --chat-template）、`driver_phase1.sh`（1B）、
  `driver_phase2.sh`（7B）、`driver_y2.sh`、`driver_qwen.sh`、`qwen_eval.py`
  （template 模式、128K 分块 prefill）、`chain_overnight.sh`、`chain_qwen_e1.sh`
- **评测审计**：`driver_audit_olmo_chat.sh`、`compare_audit.py`
- **P0 诊断**：`diag_fork_eb.py`（分叉 E/B）、`diag_write_decomp.py`（写分解）、
  `probe_capability.py`（探测，预期 16GB 已作废、实测 32GB 全过）、
  `driver_phase0.sh`/`driver_phase0_wd*.sh`
- **数据制备**：`data_prep_cpt.py`（V1 冻结）、`data_prep_cpt_v2.py`（500M，
  未运行）、`data_prep_sft.py`、`count_long_docs.py`
- **训练链（就绪未启动）**：`track_b_train.py`（V1）、`track_b_train_v2.py`
  （R12_7B_CPT_500M_V1）、`build_kl_cache.py`（教师预缓存）、`probe_7b_train.py`、
  `chain_500m.sh`、`chain_500m_waiter.sh`、`wait_and_prep_500m.sh`
- **E1**：`e1/driver_e1.sh`、`e1/e1_cross_cells.sh`、`e1/e1_fit_readout.py`
  （陈旧契约常数已改为参考记录+根因）、`e1/e1_summarize.py`
- **round11_harness_ref/**：round-11 引擎只读参考副本（generation_contract、
  single_table_generation 等），勿改。

## reports/ — 各轮执行报告（顶部均有 09-06 状态栏）

| 文件 | 状态 |
|---|---|
| `EXPERIMENT_REPORT_20260905.md` | 已取代（两方向审计时代；本地镜像已清理） |
| `ROUND10_LORA_RESULTS_20260905.md` | 已取代（Qwen 轮；**raw 模式崩溃警示**） |
| `ROUND11_OLMO_RESULTS_20260905.md` | 已取代（数字仍有效但限于 1B 三臂） |

## round11_20260905_olmo/ — OLMo 跨家族轮（已取代）

`PREGLUCTION_OLMO_Z1.md`、`PREGLUCTION_OLMO_ON_ADDENDUM.md`（冻结预注册，历史）；
`analysis/read_olmo_round.py`（读回执用；本地回执已清理，服务器原件在
`/root/autodl-tmp/claude_round11_olmo_20260905/`）。

## round10_20260905/ — §10 Qwen LoRA 轮（已取代）

`RUNBOOK_ROUND10.md`（历史）；`analysis/cpu_power_analysis.py`（历史）。
本地回执/配置 JSON 已清理，服务器原件在 `/root/autodl-tmp/claude_round10_20260905/`。
`package/`（当时代码库快照，484 文件）已于 09-06 删除——经逐文件比对与仓库
HEAD 逐字节相同，纯冗余。

## code/ — 两方向审计诊断脚本（已收官）

`first_divergence_kl_diagnosis.py`（1A）、`output_requirement_cross.py`（1B）、
`fixed_weight_support_shape_cross.py`（2）。各文件头已加 STATUS 行。

## runbooks/ — `BOOT_RUNBOOK_20260905.md`（已作废，见文件头状态栏）

## 服务器资产根（机器已关机，数据盘保留）

- `B12=/root/autodl-tmp/claude_round12_20260906/`：`tables/`（ROUND12_STATIC_TABLES_FROZEN_V1）、
  `tasks/round12_tasks.jsonl`、`data/cpt/`（V1 冻结）、`data/kl_cache 目标路径`、
  `track_a/`（全臂结果）、`track_a_audit/`（审计，7B Z chat 部分）、
  `runs/Z_CPT_500M/`（日志含 USER_HALT）、`datasets/pg19/`（14/15 片）、`diag/`、`probe4080/`
- `R11=/root/autodl-tmp/claude_round11_olmo_20260905/`
- `B=/root/autodl-tmp/ffn_review_execution_20260904/`（release008 只读 + 冻结控制表）
- 模型：`/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct`、
  `/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct`

## 清理记录（2026-09-06，用户指令"非核心实验结果 JSON 删除"）

已删：`results_20260905/`（全目录，含 2_ledger.json/2_report.json/1a/1b/2_runs/
claude_audit_results.tgz）、`round10_20260905/receipts/` 与 `config/`、
`round11_20260905_olmo/receipts/`、`configs/asset_paths_westc_20260905.json`
（自带"不得提交"标注）、空 `logs/`、`.DS_Store`。
保留的唯一 JSON = `round10_20260905/package/scripts/text_eval/ds_zero2.json`
（存档代码包的 DeepSpeed 配置，非实验结果）——随后连同整个 `package/`
（484 文件，与仓库 HEAD 逐字节相同的冗余副本）一并删除。
