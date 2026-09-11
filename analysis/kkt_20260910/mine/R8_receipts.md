# R8 — 评测凭据（receipt）挖掘：面板数字的原始来源与协议

日期：2026-09-10。角色：挖掘与整理（不推导）。
范围：`/Users/yang/projects/hybrid-rope/results/` 下的真实评测凭据，以及与之配套的评分脚本／协议。
纪律：只读；本文件是唯一写入物。每条结论带出处（文件路径＋行号/键路径）与证据等级。

---

## 0. 首要更正：任务给的两个起点路径都不存在（必须先说）

| 任务里给的起点 | 实际情况 | 出处 |
|---|---|---|
| `results/nongeometric_screen_20260909/` | **本地不存在**。它是**服务器端**运行目录：`/root/autodl-tmp/nongeometric_screen_20260909`（README 自己写明）。实验代码不在 `results/` 下，而在仓库根的 `experiments/nongeometric_screen/`（35 个 .py，2380 行）。 | `experiments/nongeometric_screen/README.md:60-61`；实测 `ls results/ \| grep nongeometric` 无输出 |
| `results/planned_controls/` | **本地不存在**。正确前缀是 `results/nongeometric_screen_20260909/planned_controls/...`，同样只在服务器上。被 6 处文档当作回执位置引用，全部悬空。 | `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:36,629`；`docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:173,235,281`；`docs/research/ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md:176` |

**连带后果（比上面两条更严重）**：

- `experiments/nongeometric_screen/reference_tables.json` **也不存在**——而它是 `analysis/unify_20260910/tables/GROUND_README.md:22` 与 `ground_truth_tables.json` 里 `methods.{Native,MrPro,...}.sources` 用来锚定"部署 fp32 原生表"的**唯一本地依据**。也就是说，G1 地面真值表声称的"18/18 bit-exact"，其对照物在本机一台都不在。
- `ground_truth_tables.json` 里 26 个 09-09 方法的分数来源写成 `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/results/nongeometric_screen_20260909/results/<method>/contract.json`（**另一台机器、另一个 checkout**）。例：`methods.E1_s28_less.sources`、`methods.E1_pair28_29.sources`、`methods.Smooth_MrBudget.sources`。
- 结论：**面板里除 MrPro／MrProBM／GapCapped 之外的 128K/32K 分数，在本机没有可打开的原始结果文件**。证据等级一律只能标 [部分证据-单源重建]。

### 0.1 `results/` 下另一条研究线的干扰（避免误认）

`results/` 里 09-09/09-10 有 **1480 个**新文件（`find results -newermt 2026-09-09`），但它们属于 **KV-cache 位置选择（NOSA / PC2 / PM）** 工作线，不是 RoPE 频率分配：

`broad_position_eval_20260909/`（`panel_v1/manifest.json:status = "PREPARED_NO_MODEL_INFERENCE"`，pc2.jsonl 41MB / pm.jsonl 69MB 是**准备好的输入**）、`nosa_position_20260909/`、`one_hour_decision_20260909/`、`position_overnight_20260909/`（E01–E12）、`pc2_cascade_same_state_v2/`、`pc2_ten_directions_20260910/`、`reference_position_20260909/`（其 `operator_checks.json:scope = "CPU positional-bias arithmetic only, not an LLM result"`）。
**这些都不是面板凭据**，引用前必须区分。

---

## 1. 主面板（36 行 Qwen2.5-3B，32K/128K）——方法 → receipt → 协议

统一协议（三条独立凭据一致）：
- 任务集 = 六任务 RULER 开发子集 `{niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1}`；32K 每任务 ×2 行、128K 每任务 ×4 行 = **36 行**（评测行数，不是方法数）。
  出处：`scripts/experiments/olmo_fast_screen/ruler_bench.py:5`；`docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json:scope`。
- 模型：`Qwen/Qwen2.5-3B-Instruct`，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`（`ROPE_BM_TRANSFER_RESULT_20260908.json:models.qwen3.revision`）。
- 评分：`ruler_bench.py:9-16` `score()` → 文本 strip + 控制字转 `\n` + lower；`reference in text` 子串命中；`qa_*` 取 `max(hits)`（任一命中即 1），其余取 `sum(hits)/len(hits)`（分数制）。
- 汇总：`ruler_bench.py:19-28` `summarize()` → 每长度对 6 任务取宏平均。
- 判决：`ruler_bench.py:31-51` `verdict()` → `DEVELOPMENT_WIN`（长端 Δ>0 且短端 Δ≥0）／`TRADEOFF`（长端 Δ>0）／`NO_LONG_GAIN`。硬编码 `evidence_scope='Six-task RULER development subset; not full RULER or independent confirmation'`。
- 解码预算**按任务不同**（不是统一值）：`niah_*`=128、`fwe`=50、`qa_1`=32、`vt`=30（实测 `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl` 全 36 行的 `max_new_tokens` 取值集合 {30,32,50,128}）。
- 参考数：`niah_single_2`=1、`niah_multikey_2`=1、`niah_multiquery`=4、`vt`=5、`fwe`=3、`qa_1`=3。
- 无 seed 概念（贪心解码 `do_sample=false`），改为**冻结输入哈希**配对：`manifest_sha256`／`prompt_sha256`／`input_sha256`。
- 运行时：`NVIDIA GeForce RTX 4080 SUPER`、torch 2.8.0+cu128、transformers 5.15.1、`backend="Flash SDPA only; no fallback enabled"`（`ROPE_BM_TRANSFER_RESULT_20260908.json:models.qwen3.runtime`）。

| # | 方法 | 面板 32K/128K | receipt（本地可打开） | 协议要点 / 配对 | 证据等级 |
|---|---|---|---|---|---|
| 1 | **MrPro**（基线） | 87.2222 / 78.1250 | `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl`（36 行，行级 full receipt：`row_id/prompt_sha256/input_tokens/references/correct/output_text/generated_ids/ended_eos`）；汇总 `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json:models.qwen3.arms.MrPro.summary` | 36 行；`eos_count=24`；`score_sum=29.2167`；`table_identity.tensor_sha256=33cbe3a4…6016f`，`gain=1.138629436111989`；`raw_sha256=e5dcf19a…2672`；elapsed 879.75s | **已验证-本地JSON** |
| 2 | **MrProBM** | 91.6667 / 70.8333 | `results/bm_transfer_20260908/run_qwen3_01/MrProBM.jsonl`；汇总同上 `.arms.MrProBM` | 同 36 行配对；`status=NO_LONG_GAIN`；`macro_delta_by_length = {32768:+0.04444, 131072:-0.07292}`；`paired_wins=6 / paired_losses=4`；`tensor_sha256=e5833f77…13193`；`raw_sha256=dd498dc2…cc32` | **已验证-本地JSON** |
| 3 | **GapCapped**（09-08） | 84.4444† / 62.1528† | `results/bm_transfer_20260908/gap_capped_run_01/GapCapped.jsonl`（36 行）+ `GapCapped.json`（36 行 row_ids、`score_sum=25.05`、`raw_sha256=2619b27a…6e3a`）；汇总 `docs/research/ROPE_GAP_CAPPED_RESULT_20260908.json` | `baseline_reused=true`（复用 MrPro 36 行）；`paired_wins=0 / paired_losses=7`；`status=NO_LONG_GAIN`；构造：BM 变体＋慢带载波帽 `c=1.2365e-5`，改动槽 24–38（`GROUND_README.md` §3 行 GapCapped） | **已验证-本地JSON** |
| 4 | **Qwen2.5-7B MrPro** | 83.3333 / 84.4444（**18 行**） | `results/bm_transfer_qwen7b_20260908/run_screen_02/{MrPro,MrProBM}.jsonl` + `.json`；汇总 `docs/research/ROPE_QWEN7_BM_RESULT_20260908.json` | **18 行**（非 36）：32K 每任务 ×1、128K 每任务 ×2；rev `a09a3545…`；`run_status.generations=30, reused_generations=6`；`baseline_reused=true` | **已验证-本地JSON** |
| 5 | **Qwen2.5-7B MrProBM** | 79.9999 / 71.1111（18 行） | 同上 `.result.candidate` | `status=NO_LONG_GAIN`；`macro_delta_by_length={32768:-0.03333, 131072:-0.13333}`；`paired_wins=0 / paired_losses=3`；`eos_count=10` vs baseline 11 | **已验证-本地JSON** |
| 6 | **E1_s28_less** | 87.2222 / 83.3333 | **无凭据**（服务器 `.../results/E1_s28_less/contract.json`；`ground_truth_tables.json:methods.E1_s28_less.sources` 指向本地不存在的 `experiments/nongeometric_screen/reference_tables.json` 与另一个 checkout） | 构造式有本地代码：`experiments/nongeometric_screen/select.py:76-80`（仅槽 28 `m28:=m27`）。分数只在 T2 重建表与文档叙述里 | [部分证据-单源重建] |
| 7 | **E1_s29_more** | 95.5556 / 77.9167 | **无凭据**（同上模式） | `select.py:76-80`；32K 增益来自单条 QA 行 | [部分证据-单源重建] |
| 8 | **E1_pair28_29** | 87.2222 / 73.9583 | **无凭据**（同 `E1_pair28_29.sources`） | 逐槽可加性反例（−4.17pp 的出处） | [部分证据-单源重建] |
| 9 | **Smooth_MrBudget** | 87.2222 / 68.3333 | **无凭据**；构造式本地：`experiments/nongeometric_screen/smooth_budget.py` | `B=16/3` 最小粗糙度 KKT 解 | [部分证据-单源重建] |
| 10 | **MrUni** | 64.5833 / 73.3333 | **无凭据**；构造式本地：`smooth_budget.py:38` | 过渡段线性斜坡 `m_j=(j−23)/17`（**不是全表 ÷4**） | [部分证据-单源重建] |
| 11 | **HighGapToLong** | 70.1389 / 67.3611 | **无凭据**；构造式本地：`experiments/nongeometric_screen/gap_budget_transfer.py` | `ΣT=1.6022`（违反端点固定） | [部分证据-单源重建] |
| 12 | **LongBridgeSlower** | 80.5556 / 80.0694 | **无凭据**；构造式本地：`experiments/nongeometric_screen/long_bridge.py` | 槽 36–39 `ν−=1/131072` | [部分证据-单源重建] |
| 13 | **LongBridgeFaster**（方向控制） | 87.2222 / 73.9583 | **无凭据**；同上 | 同幅反号 | [部分证据-单源重建] |
| 14 | **FullLagP2_Transfer3B** | 72.9167 / 81.6667 | **面板版本无凭据**；但**有独立的 1.5B 原始回执**：`docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`（见 §4） | 面板版 = 把 1.5B 资产迁到 3B；`gain=1.102585782722872`；128K 最强 far 项 | 面板版 [部分证据]；1.5B [已验证-本地JSON] |
| 15 | **Control_Mr_gain074** | 98.3333 / 75.3472 | **无凭据**；构造式 = MrPro 表 @ gain .074 | gain 析因 | [部分证据-单源重建] |
| 16 | **E3_BM_gain074** | 100.0 / 70.0 | **无凭据** | 表与 gain 的包，非 gain 孤立效应 | [部分证据-单源重建] |
| 17 | **E3_BM_gain1** | 89.5833 / 58.8194 | **无凭据** | 同上 | [部分证据-单源重建] |
| 18 | **E7_local_projection** | 90.0 / 68.6111 | **无凭据**；构造式本地：`experiments/nongeometric_screen/project.py` | `ground_truth_tables` 标 `bit=—`（公式未复原） | [部分证据-单源重建] |
| 19 | **E2_tail_more** | 100.0 / 54.7222 | **无凭据**；构造式本地：`select.py:87-89` | **12 行**面板（非 36） | [部分证据-单源重建] |
| 20 | **E8_zero51** | 100.0 / 50.5556 | **无凭据**；构造式本地：`select.py:90-92` | **12 行**；`ν_51=0` | [部分证据-单源重建] |
| 21 | **E4_pair25_29** | 100.0 / 64.4444 | **无凭据**；`select.py:146-163` | **12 行** | [部分证据-单源重建] |
| 22 | **E1_s28_reverse_matched** | 100.0 / 64.4444 | **无凭据**；公式**未复原**（部署反演 m28=0.132270 ≠ 候选式 0.130719） | **12 行** | [部分证据-单源重建] |
| 23 | **E1_s29_plus_matched** | 100.0 / 64.4444 | **无凭据**；公式**未记录** | **12 行** | [部分证据-单源重建] |
| 24–30 | **E5_layer21/27/32、E6_layer14_group0/1、E7_norm_matched_BM、E10_dual_frequency、E9_distance** | 全 100.0 / 64.4444 | **无凭据**；构造式本地：`operators.py`、`distance_operator.py` | **12 行**；`ground_truth_tables` 标"无单一 64 槽静态表（结构性排除）" | [部分证据-单源重建] |
| 31 | **BM_ScaleTaper** | 未执行 / 未执行 | **无分数凭据**；构造式本地 `scale_taper.py` | deferred 044d；表已重建（bit=True），分数 null | [部分证据] |
| 32 | **StackFrontBack** | null / null | **无凭据**（队列 0446 未执行） | 公式重建 | [假设] |
| 33 | **MrProN16 / MrProN15** | null / null | **无凭据**（队列 0448/0449 未执行） | 公式重建 | [假设] |
| 34 | **YaRN_linear_official / NTK_static / YaRN_smoothstep** | — / — | 面板外（历史/官方）；`sources` 指向 `paper-2027/research/attention-aware-retrofit/evidence/REFERENCE_CORRECTED_K128_S4_NLL_RECEIPT_20260901.json` 等 | 非本屏 | [叙事-未验证]（就面板而言） |

† GapCapped 的 84.4444/62.1528 与本地 `GapCapped.json:score_sum=25.05` 及 `ROPE_GAP_CAPPED_RESULT_20260908.json:result` 一致（**×100 后**）。

### 1.1 面板判决与行级失败例（可直接复用的负数据）

`ROPE_BM_TRANSFER_RESULT_20260908.json:comparison.loss_rows` 给出 MrProBM 相对 MrPro 的 4 条失败行原文，是"长端退化不是噪声"的一手材料：
- `niah_multikey_2_131072_1`：baseline `" 3954314."` → candidate `" 9289114."`（1.0→0.0）
- `vt_131072_1`：5 参考全中 → 只答出 1 个（1.0→0.2）
- `fwe_131072_3`：2/3 → 1/3
- `qa_1_131072_3`：1.0 → 0.0
`baseline_extreme_cells = ["32768/niah_single_2","32768/niah_multikey_2","32768/niah_multiquery","131072/niah_single_2"]`——即 32K 三条 niah 与 128K single 在基线上已满分（**天花板**，无上升空间，不可作为判别项）。

---

## 2. 长文 NLL 凭据 —— 与"PG19/ProofPile 128K"的说法不符

### 2.1 实际存在的 NLL 语料是 **FineWeb-Edu**，不是 PG19/ProofPile

三条 manifest 的 `scope` 字段一字不差地写着：

> `"Existing 16 long FineWeb-Edu prefixes decoded from frozen Qwen tokens and retokenized for target. Original parquet unavailable. Tail 512 next-token NLL, not full-document NLL or generation capability."`

出处：`results/olmo_fast_screen_20260908/prepared_nll_02/manifest.json:scope`；`results/bm_transfer_qwen7b_20260908/prepared_nll_01/manifest.json:scope`；脚本侧同句在 `scripts/experiments/olmo_fast_screen/natural_nll.py:34-38`。**每 cell 16 篇文档，tail 512 tokens。**

### 2.2 三个已完成的 NLL 回执（本地可打开）

| 模型 | 长度 | Native | MrPro | MrProBM | receipt |
|---|---|---|---|---|---|
| Qwen2.5-3B | 8192 | 2.276074 | **2.323329** | **2.325812** | `results/bm_transfer_20260908/run_nll_01/{Native,MrPro,MrProBM}.jsonl`（各 48 行 = 3 长度×16 文档） |
| | 16384 | 2.143172 | 2.188861 | 2.193111 | 同上 |
| | 32768 | 2.038502 | 2.088995 | 2.087888 | 同上 |
| Qwen2.5-7B | 8192 | 2.123450 | 2.176849 | 2.176893 | `results/bm_transfer_qwen7b_20260908/run_nll_01/*.jsonl` |
| | 16384 | 1.995638 | 2.039214 | 2.042067 | 同上 |
| | 32768 | 1.907776 | 1.946614 | 1.949421 | 同上 |
| OLMo-2-0425-1B | 4096 | 2.835046 | 3.210621 | **2.955060** | `results/olmo_fast_screen_20260908/run_nll_01/*.jsonl`（**Native 只有 16 行=仅 4096**） |
| | 8192 | — | 3.256548 | 2.954863 | 同上 |
| | 16384 | — | 3.687978 | 2.862058 | 同上 |

协议（`ROPE_QWEN3_BM_NLL_RESULT_20260908.json:runtime`）：`scoring = "FP32 cross-entropy on final 512 next tokens"`；`attention = "Flash SDPA"`；`torch=2.8.0+cu128`。配对统计用文档级 bootstrap：例 Qwen3 8K `BM_minus_MrPro = +0.002483`，95% 区间 `[-0.001167, +0.005815]`（跨 0）；16K `+0.004250`，区间 `[+0.000865, +0.007470]`（不跨 0）；32K `-0.001107`，区间 `[-0.007675, +0.004800]`。
OLMo 则相反且幅度大：4096 `BM−MrPro = -0.2556` 区间 `[-0.3401, -0.1823]`、8192 `-0.3017`、16384 `-0.8259`（16/16 文档方向一致）。

**关键限制：最长 32768（Qwen）／16384（OLMo）。本机不存在任何 64K 或 128K 的 NLL receipt。**

### 2.3 128K 长文 NLL 只在"散文"里，没有落盘

- 协议代码在本地：`experiments/nongeometric_screen/long_eval.py`（`lengths` 默认 `[65536, 131072]`，:26,:48），scope 硬编码为 `'Tail 512 next-token NLL following one real contiguous document prefix; not full-document PPL'`（:53）。数据源 `prepare_long_sources.py` 明确是**官方 Proof-Pile arxiv test + PG19 test books**（:54 `selection='First 32 Proof-Pile arxiv test rows with >=300000 chars; first 16 lexical PG19 test books with >=700000 bytes…'`，Proof-Pile 归档 sha256 `b1bc923a…1d835`）。
- 但**输出目录 `long_nll/` 与 `long_inputs/` 都不在本地**（本地无 `results/nongeometric_screen_20260909/`）。
- 唯一的 128K 数字出现在散文里：`docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:38-44` —— 同 4 篇文档前缀、tail-512：MrPro 官方 **1.705390 / PPL 5.503529**；同 gain MrPro **1.680924 / 5.370516**；P2 **1.678507 / 5.357549**；precheck 268.7s。P2 对 official −2.6525%，对 same-gain −0.2415%（2 升 2 降）。
- 证据等级：**[部分证据-单源]**（只有 markdown 文本，无 jsonl/csv 回执；同数字另见 `analysis/unify_20260910/digests/digest_nongeo-code.md:138,217` 与 `digests/digest_panel-results.md:90`，属同源复述，不构成第二来源）。

**任何"面板方法在 PG19/ProofPile 128K 上有 NLL 数字"的说法，本机零凭据。**

---

## 3. RULER 凭据（除 36 行主面板外的第二套）

### 3.1 OLMo-2-0425-1B Instruct @ 16384，350 行（7 任务 ×50）

| 方法 | 行 | score_sum | 宏平均 | receipt |
|---|---|---|---|---|
| MrPro | 350 | 24.80 | **0.070857 (7.09%)** | `results/olmo_fast_screen_20260908/run_ruler_newtasks_01/MrPro.jsonl`（行级）+ `MrPro.json`；汇总 `docs/research/ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json` |
| MrProBM | 350 | 145.85 | **0.416714 (41.67%)** | 同目录 `MrProBM.jsonl/.json` |

逐任务（我按 jsonl 重算，与 `.json:by_length` 一致）：

| 任务 | MrPro | MrProBM |
|---|---|---|
| niah_single_1 | 10/50 = 0.2000 | 45/50 = 0.9000 |
| niah_single_3 | 0/50 = 0.0000 | 25/50 = 0.5000 |
| niah_multikey_1 | 6/50 = 0.1200 | 30/50 = 0.6000 |
| niah_multikey_3 | **0/50 = 0.0000** | **0/50 = 0.0000** |
| niah_multivalue | 2/50 = 0.0500 | 30/50 = 0.6050 |
| cwe | 0/50 = 0.0060† | 2/50 = 0.0520† |
| qa_2 | 6/50 = 0.1200 | 13/50 = 0.2600 |

† 我按 `correct` 求和得 0/50 与 2/50，而 `.json` 的 `task_accuracy` 为 0.006 / 0.052——说明 cwe 的 `correct` 是**分数制**（非 0/1），我上面的整数计数对该行不适用。其余六任务两处一致。

配对：BM wins **156** / losses **9** / ties 185（我按 row_id 重算，与 `.result.paired_wins=156 / paired_losses=9` 一致）。
`eos_count`：BM 119 / MrPro 197 —— **EOS 口径不可与分数混用**（`ROPE_QWEN7_BM_RESULT_20260908.json:scope` 同句提醒）。
协议要点：`length_cap=16384` 单一（`input_tokens` 实测 13411–16351）；`max_new_tokens ∈ {32,120,128}`；OLMo-2-1B 原生上下文为 4096 ⇒ **16K 是 4× 外推**；参考数 `niah_*`=1、`niah_multivalue`=4、`cwe`=10、`qa_2`=1；模型 rev `48d788eca847d4d7548f375ad03d3c9312f6139e`（`run_ruler_newtasks_01/runtime.json`）。评分脚本与主面板同一个 `ruler_bench.py`（source sha `2c12dd09…620`）。
scope 原文：`"Matched frozen task panel, not full official RULER. Official recall and complete-string/EOS are distinct."`

### 3.2 36 行六任务之外的 RULER 小面板

- `docs/research/ROPE_OLMO_BM_RESULT_20260908.json`：scope `"Small six-task RULER panels. Scores use official substring matching; no full RULER or cross-model claim."`（arms: MrPro/MrProBM、MrProBM/BMSelectiveGain/BMUniformMatchedGain、MrPro/MrUni、MrUni/OfficialYaRN、BMCappedS4/BMFreq8Gain4）。
- `results/position_observability_20260908/primary_ruler_02|03/`：`manifest.json` 明确 `scope="RULER multiquery generator with declared output-format adaptation; not full official RULER"`，`format="Exactly four numerical strings in query order, single spaces, then EOS; no answer prefix or constrained decoding"`；两 split：`compact_dev` cap=2048 seed=20260910 行数 8；`frozen_long` cap=32768 seed=20260911 行数 32。**这是位置选择线（RoPEMean/Quest 等）的基准输入，不是 RoPE 分配面板。**
- `docs/research/ROPE_BM_128K_DIAGNOSIS_RESULT_20260908.json`：128K 诊断（含 `mode`/`arm` 字段）。

---

## 4. passkey 凭据

- **面板内的 passkey = `niah_single_2 @131072`（4 行）**，不是独立任务。MrPro 基线 4/4=1.0（`ROPE_BM_TRANSFER_RESULT_20260908.json:arms.MrPro.summary.by_length.131072.task_accuracy.niah_single_2 = 1.0`），因而被列为 `baseline_extreme_cells` 之一（**天花板，无判别力**）。
- `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:14`：P2 与同 gain MrPro "already scored **4/4 on the existing 128K single-key passkeys**; these are reused"——**复用**，无新回执。
- `docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md:87,120,131` 给出设计口径：旧 RULER 12 条内含 passkey；扩大时复用 `niah_single_2` 子集；"单 passkey 扩样（饱和 100% 无信息）"被明确**排除**。
- 早期 EVQ/几何路线的 passkey 回执（与本次面板方法无关）：`results/rebuttal_primary1_20260713/raw/primary1_{seed42,125m_s42}/{evq_seed42,geo_seed42}/passkey_nll.json`；`results/legacy/passkey_long/report.md`；`results/core_text/phase14_yarn_passkey/phase14.log`。
- **无凭据**：P2 / Control 在 128K passkey 上的独立新回执——**只有散文复述**（[部分证据-单源]）。

---

## 5. 检索行级结果与其他一手回执

| 项目 | receipt | 要点 |
|---|---|---|
| Cross-cache（attention≠generation 判决） | `results/bm_transfer_20260908/cross_cache_run_01|02/results.json`（逐 token `top_ids/top_logits/eos_margin`）+ `qualification.json`（`keys_bitwise_equal:true`、`diagonal_matches_original:true`）+ `attention_source.txt` | run_01 与 run_02 各含 `live/qualification/results/runtime/status.json` |
| 长端自然 QA（OLMo，LongBench 子集） | `docs/research/ROPE_OLMO_BM_NATURAL_RESULT_20260908.json`（`rows_per_arm=567`）+ `run_natural_01`(360 gen)/`run_natural_02`(774 gen)；`ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`（`rows_per_arm=778`，含 `run_natural_extra_01` 422 gen） | `task_equal_macro.extended`: MrPro 0.21373 / BM 0.26008（Δ+0.04636，bootstrap 95% `[0.01316, 0.06292]`）；`within_native_length`: 0.39703 / 0.41595（Δ+0.01892，区间跨 0）。scope：`"Frozen untruncated in-range LongBench subset. Natural token-F1, not exact-string/EOS capability or all LongBench."` 按任务：hotpotqa/2wikimqa/qasper/narrativeqa/multifieldqa_en |
| 1.5B FullLagP2（P2 面板的来源资产） | `docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json` | `rows_total=120`（64K: 3 任务×8；128K: 3 任务×8）；`decoder={do_sample:false,num_beams:1,repetition_penalty:1.1,eos_token_id:151645}`；`all_rows_ended_eos=true`；64K MK2 37.5 vs MrPro 12.5、FWE 70.83 vs 45.83、VT 87.5 vs 82.5；**128K MK2 0/0、FWE 50/50**（`limits` 原文："At 128K only variable tracking retains an aggregate gain; MK2 is 0/0 and FWE 50/50, so the three 64K gains do not all persist."）；另附 3B 迁移 24 行（MK2 75.0 vs 50.0、VT 90.0 vs 95.0、FWE 66.67 vs 75.0） |
| OLMo NLL/QA/RULER 的 prepared 输入 | `results/olmo_fast_screen_20260908/prepared_{nll_02,natural_extra_01,ruler_newtasks_02}/manifest.json` | 冻结输入哈希链 |
| 09-08 一夜实验 | `docs/research/ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json` | `status="RESEARCH_STOPPED_FOR_AUTHOR_REQUESTED_REPORT"`；逐 job `exit_code/elapsed/plan_sha256/state_file_sha256/log_sha256`；含一条 `FAILED`（`CARRIER_NATIVE_MEANS_01`，`interpretation="INPUT_HASH_TYPE_ENGINEERING_FAILURE_BEFORE_MODEL_FORWARD"`）——**失败例子保留完整回执的正面样板** |
| 位置可见性/中层探针（非 RoPE） | `results/position_observability_20260908/evidence_manifest_20260909.json` | `research_goal_achieved: false`；`scope="Two complete paired quality comparisons; not proof of the broader research/paper goal"` |

### 5.1 明确"无凭据"清单（找不到 receipt 的方法/任务）

1. 26 个 09-09 非几何筛方法（§1 表中 #6–#30）的 **32K/128K 分数** —— 只有 T2 重建表与文档叙述。
2. **128K P2 precheck（tail-512 NLL，4 篇前缀）** —— 散文数字，无 jsonl。
3. **P2_QA128 pilot（4 行 128K QA，12 generations）** —— 本地无回执；`digests/digest_panel-results.md:90` 声称"pilot 本地 jsonl 实测 MrPro 1/4（row_3）、P2 0/3、Control 0/3，P2/Control 缺第 3 行"，但**该 jsonl 在本机找不到**（`grep -rl "P2_QA128\|qa_pilot\|pilot4"` 只命中文档与 digest）。该行结论（"pilot 未构成 P2 QA 收益的独立确认"）引自 digest，标 [部分证据-单源]。
4. `planned_controls/` 全部：`long_first_protocol.json`、`p2_gap_comparison.json`、`source_subspace_transport_audit.json`、`qk_operator_gram.npz`、`weighted_source_subspace_audit.json`、`softmax_harmonic_audit.json`。
5. `deferred_queue/20260910_candidate_quality/`（HighGapToMid 的中止数据 16/36 行，以及候选质量判决）。
6. `long_inputs/`、`long_nll/`（PG19 + Proof-Pile 的 64K/128K 输入与输出）。
7. `experiments/nongeometric_screen/reference_tables.json`（原生表 bit-exact 的对照物）。
8. HighGapToMid 的裁决（**中途中止于 16/36**，`NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:24-26`）——未计入任何已测表。
9. E1_s28_reverse_matched / E1_s29_plus_matched 的构造式（`ground_truth_tables.json:mismatches` 自认"仅部署张量为真值"）。

---

## 6. 评分脚本与判决口径（写 KKT 材料时必须照抄的部分）

| 项 | 值 | 出处 |
|---|---|---|
| RULER 任务集 | `('niah_single_2','niah_multikey_2','niah_multiquery','vt','fwe','qa_1')` | `scripts/experiments/olmo_fast_screen/ruler_bench.py:5` |
| RULER 匹配 | 小写子串；`qa_*` 取 max，其余取命中比例 | 同文件 :9-16 |
| RULER 聚合 | 每长度 6 任务宏平均 | 同文件 :19-28 |
| 判决门 | `DEVELOPMENT_WIN` / `TRADEOFF` / `NO_LONG_GAIN` | 同文件 :48-49 |
| 面板强度断言的上限 | `evidence_scope='Six-task RULER development subset; not full RULER or independent confirmation'` | 同文件 :52 |
| NLL 单位 | "final 512 next tokens" cross-entropy，FP32；**不是 full-document PPL** | `ROPE_QWEN3_BM_NLL_RESULT_20260908.json:runtime.scoring`；`natural_nll.py:34-38` |
| NLL 每 cell 样本量 | 16 篇文档 | prepared manifest `docs[16]` |
| PG19/ProofPile 长 NLL | tail-512、长度 65536/131072、每数据集前 2 篇 | `experiments/nongeometric_screen/long_eval.py:20,:26,:53` |
| 配对方式 | 行级配对（同 `row_id`/`doc`），断言 `prompt_sha256/task/length_cap/references` 全等，否则 raise | `ruler_bench.py:31-38` |
| 无 seed | 贪心解码，用输入哈希冻结 | 各 `runtime.json` 的 `generation_config.do_sample=false`（`ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json:contract.generation_config`） |

---

## 7. 与权威文档的矛盾与口径不一致

| # | 冲突 | 两边出处 | 判定 |
|---|---|---|---|
| C1 | 任务描述与多份文档说"长文 NLL = PG19/ProofPile 128K"，实际面板 NLL 回执是 **FineWeb-Edu ≤32K** | 文档：`NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:38-41`（4 篇 128K 前缀）；回执：`prepared_nll_02/manifest.json:scope`、`natural_nll.py:38` | **口径必须拆开写**：面板 NLL = FineWeb-Edu/≤32K【有回执】；128K 长文 NLL = PG19+Proof-Pile/4 篇【只有散文】 |
| C2 | `GROUND_README.md` 结论"18/18 bit-exact"与 `ground_truth_tables.json:evidence_policy` 的 [已验证] 定义，都要求"公式重建与**部署张量**逐位一致"；但部署张量的镜像目录本地不存在 | `analysis/unify_20260910/tables/GROUND_README.md:22`；`ground_truth_tables.json:meta.evidence_policy`；`rebuild_ground_truth_tables.py:31` 的 `MIRROR` | **本机不可复现**。与 INTEGRATION §9"地面真值表 v1 在产"是同一件事的两种字体；引用这些分数时不得标 [已验证] |
| C3 | `GROUND_README.md:§3` 表尾注称 GapCapped 的 summary"不在本地镜像 `results/` 目录内"，但 `results/bm_transfer_20260908/gap_capped_run_01/GapCapped.json` 与 `.jsonl`（36 行）**确实在本地** | GROUND_README 表尾 †；实测两个文件存在 | **GROUND_README 此处过时/有误**，GapCapped 可升到 [已验证-本地JSON] |
| C4 | 任务书说"面板 36 行"，但 E2/E8/E4/E1_s28_reverse/E1_s29_plus/E5×3/E6×2/E7_norm/E9/E10 是 **12 行**面板 | `ground_truth_tables.json:methods.E2_tail_more.panel = "12行 (100/54.7222)"`、`E8_zero51.panel="12行"`、`E5_*.panel="12行 (持平/无模型分)"` | **36 行面板不覆盖全部方法**；混用 36 行与 12 行分数做比较是口径错误 |
| C5 | INTEGRATION §3 把"128K response 终态"标 UNKNOWN，但 §3 同段把 P2 128K NLL 数字当已知引用 | `analysis/unify_20260910/INTEGRATION_20260910.md:52`；`NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:38-41` | 不矛盾但**容易误读**：P2 的 128K NLL 是 tail-512 短评（4 篇），不是 §8-3 要求的"真实连续 128K 模型级判定" |
| C6 | `digest_panel-results.md:90` 说 pilot 是"本地 jsonl 实测"，但本地无此 jsonl | `analysis/unify_20260910/digests/digest_panel-results.md:90` vs 实测 grep | 该 digest 的 pilot 行结论必须降级为 [叙事-未验证] 或明确注明 jsonl 已不可得 |
| C7 | Qwen7B 面板是 **18 行**，不是 36 行；7B 的 `qa_1` 两侧均为 0.0（与 3B 不同） | `docs/research/ROPE_QWEN7_BM_RESULT_20260908.json:result` | 跨模型比较必须带行数标签 |

---

## 8. 覆盖度

### 读了（全部本地只读）
- `results/` 顶层全列；`bm_transfer_20260908/`、`bm_transfer_qwen7b_20260908/`、`olmo_fast_screen_20260908/`、`zero_param_single_table_20260824/` 逐目录与关键 jsonl/json 逐行解析。
- `docs/research/` 下 11 个结果 JSON 逐字段读：`ROPE_BM_TRANSFER_RESULT`、`ROPE_GAP_CAPPED_RESULT`、`ROPE_QWEN7_BM_RESULT`、`ROPE_QWEN3_BM_NLL_RESULT`、`ROPE_QWEN7_BM_NLL_RESULT`、`ROPE_OLMO_BM_NLL_RESULT`、`ROPE_OLMO_BM_RESULT`、`ROPE_OLMO_BM_FIVE_QA_RESULT`、`ROPE_OLMO_BM_NATURAL_RESULT`、`ROPE_OLMO_BM_EXTRA_RULER_RESULT`、`ROPE_QWEN15_FULL_LAG_P2_RESULT`、`ROPE_OVERNIGHT_EXPERIMENT_LEDGER`、`ROPE_BM_128K_DIAGNOSIS_RESULT`、`ROPE_BM_CROSS_CACHE_RESULT`。
- 评分/判决代码：`scripts/experiments/olmo_fast_screen/ruler_bench.py`（全 55 行）、`natural_nll.py`、`prepare_ruler.py`、`bench.py`；`experiments/nongeometric_screen/{README.md,worker.py,long_eval.py,holdout_eval.py,prepare_long_sources.py}`。
- 权威文档：`analysis/unify_20260910/INTEGRATION_20260910.md` 全读；`tables/GROUND_README.md` 全读；`tables/ground_truth_tables.json` 的 `meta/reconciliation/mismatches` 与 38 个 `methods.*` 的 `panel/sources/bit` 字段。
- `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md` 前 120 行。
- 目录存在性实测：`results/nongeometric_screen_20260909`、`experiments/nongeometric_screen/reference_tables.json`、`planned_controls`、`deferred_queue` 均确认不存在。

### 跳过 / 未读
- `results/` 下 1480 个 09-09/09-10 新文件（NOSA/PC2/PM 位置选择线）——只抽样确认其性质，未逐条读。
- `results/legacy/`（23 子目录，2026-03 前的 EVQ/几何路线）、`results/core_text/`（21 子目录）、`results/theory/`、`results/video_dit/`、`results/m4_max_36gb/`、`results/supporting_*` —— 属早期阶段材料，与本次 36 行面板无直接凭据关系，未展开。
- `results/rebuttal_primary1_20260713/`：只确认 passkey 文件位置。
- `analysis/unify_20260910/raw/`（2.9MB transcript 镜像）与 `digests*/`：未读；同一数字的复述不构成第二来源。
- 未接触 `~/.codex`（严格只读且本次无需）。
- 未运行任何重建脚本（`rebuild_ground_truth_tables.py` 因 MIRROR 缺失本机跑不起来，且 G1 代理当时在编辑）。
