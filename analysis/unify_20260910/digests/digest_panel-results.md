# digest panel-results（36 行开发面板与全部已测结果数字钉死）

日期：2026-09-10。方法：python3 直接读取本地 JSON/JSONL 原始文件重算 macro；与 docs/research 各 md 引用逐条对账。铁律执行：每个数字给出文件路径（相对仓库根 `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/`）；代理指标不写作能力结果；未测不写作否证。

---

## 1. 来源清单

### 1.1 权威逐行结果（本地镜像，64 MB，最后同步 09-10 05:36）
目录 `results/nongeometric_screen_20260909/`（远程 `/root/autodl-tmp/nongeometric_screen_20260909` 的同步副本）：
- `results/<method>/{contract.json, summary.json, ruler.jsonl, nll.jsonl}` × **26 个方法目录**（§5 全列）。ruler.jsonl 每方法 36 或 12 行，含 row_id、prompt_sha256、correct、baseline_correct、output_text、ended_eos。
- `long_nll/{MrPro, E1_s28_less, E1_s29_more, Control_Mr_gain074, FullLagP2_Transfer3B}.jsonl`：PG19/ProofPile 长文 NLL，每 (dataset,length) **每方法只有 2 篇**（audit 警告与本地一致：prepared 8 篇/源，只测 2 篇）。
- `transfer/{qwen7b, olmo1b}/{summary.json, E1_band_transfer.jsonl, nll.jsonl, qualification.json}`：s28 规则冻结迁移。
- `holdout_results/mixed_s20260910/{MrPro(4 行), Control_Mr_gain074(3 行), FullLagP2_Transfer3B(3 行)}.jsonl` + `cohort.json`：**只完成 qa_1@128K**，P2/Control 各缺第 3 行（0441 pilot 收尾状态）。
- `native_reference/summary.json`：Native S=1/gain=1，12 短行。
- `planned_controls/`：16 个 json（p2_gap_comparison、gap_budget_transfer、long_bridge_confirmation、full_lag_p2_confirmation/transfer、long_first_protocol、scale_taper(_cross_model_geometry)、softmax_harmonic_audit、qk_operator_gram*、bias_*、remote_PI_distortion_audit、weighted/source_subspace、regularized_worst_direction）。
- `development_summary.json`：**audit 明示已过期**（stale，描述部分面板），本 digest 不以它取数，只用 `results/*/summary.json`。

### 1.2 docs/research/*.json（grep -l "gain|128K|panel" 命中 23 个 + 任务点名文件）
| 文件 | 字节 |
|---|---|
| ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json | 18,270 |
| ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json | 7,169 |
| ROPE_RECOVERED_QWEN_P2_20260907.json | 8,029 |
| ROPE_MRPRO_BM_CANDIDATE_20260908.json | 94,519（含 BM 定义公式） |
| ROPE_BM_TRANSFER_RESULT_20260908.json | 8,641（3B 36 行 MrPro/BM 配对） |
| ROPE_BM_128K_DIAGNOSIS_RESULT_20260908.json | 13,941 |
| ROPE_BM_CROSS_CACHE_RESULT_20260908.json | 11,731 |
| ROPE_GAP_CAPPED_RESULT_20260908.json | 69,940（逐行 36 行） |
| ROPE_OLMO_BM_RESULT_20260908.json | 111,379（dev/replication/gain_followup/scale8/existing_controls） |
| ROPE_OLMO_BM_NLL_RESULT_20260908.json | 19,353 |
| ROPE_OLMO_BM_NATURAL_RESULT_20260908.json | 356,519 |
| ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json | 645,972 |
| ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json | 1,087,110 |
| ROPE_QWEN7_BM_RESULT_20260908.json | 36,818（18 行逐行） |
| ROPE_QWEN7_BM_NLL_RESULT_20260908.json | 17,874 |
| ROPE_QWEN3_BM_NLL_RESULT_20260908.json | 17,868（16 篇尾 512 NLL） |
| ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json | 36,386（34 个 job 回执） |
| OVERNIGHT_FAILURE_AUDIT_20260909.json | 3,500 |
| SPARSE_MEMORY_INTERFACE_RESULTS_20260908.json | 41,142 |
| ROPE_P2_TRANSFER_MECHANISM_ANALYSIS_20260908.json | 8,612 |
| ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json | 397,749 |
| ROPE_SCALE_TRANSPORT_ASSUMPTIONS/FOLLOWUP、SOURCE_COVERAGE、LAYER_POLICY_SPEC、GENERAL_ALLOCATION_CPU、CARRIER_REMOVAL、NATIVE_SECTOR、GAP_CAPPED_CANDIDATE、OLMO_MRPRO_SOURCE、BM_TRANSFER_MODELS | 各 1.5–8 KB |

### 1.3 关键 md（引用对账对象）
| 文件 | 行数 |
|---|---|
| docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md | 143 |
| docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md | 63 |
| docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md | 630 |
| docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md | 72 |
| docs/research/PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md / AUDIT_20260910.md | 227 / 127 |
| docs/research/ROPE_BM_TRANSFER_RESULT_20260908.md (57), ROPE_GAP_CAPPED_RESULT_20260908.md (54), ROPE_OLMO_FAST_SCREEN_20260908.md (55), ROPE_BM_SELECTIVE_GAIN_20260908.md (23), ROPE_QWEN7_BM_TRANSFER_20260908.md (42), PC2_FAILURE_AND_CLAIM_AUDIT_20260910.md (130) | — |

### 1.4 已知缺失（不可本地复核）
- **`planned_controls/evidence_distances_20260910.json` 本机不存在**（find 全仓无匹配）。UNIFIED §4 距离列（89K/106K/127K/96K/117K）与逐行得分联表源自服务器文件；GLM 复核稿（ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:30）声明"已读取服务器上的该文件并 CPU 解码核对"。本 digest 用本地 ruler.jsonl 独立复核了**得分部分**（全部一致，见 §2.5），距离部分只能引用文档：88,726≈89K、95,676–95,682≈96K、117,487–117,493≈117K 有本地文字出处（NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:348,443–444），vt=105,880(max)/20,563(min) 有 GLM 复核稿出处（其 :33），127K(vt_1) 无本地独立出处 [部分证据]。
- 0446 StackFrontBack / 0448 N16 / 0449 N15 / 0450 E1 holdout16 / 0451 P2 长端确认：**未执行**（GPU 已关，BUDGET §4 与 UNIFIED §9 一致），不得写作结果。

---

## 2. 任务时间线

### 2.1 2026-09-07（P2 谱系建立）
- **ROPE_RECOVERED_QWEN_P2**：从历史资产恢复 64 频表（Qwen2.5-1.5B 资产，`historical_gain_coefficient=0.074`；tensor sha `ecd0c280a11788e0…61096b`，log_s4_float32 数组在 json）。limits 原文："Historical result owner reports core4 64K/128K=0.7000/0.5875; raw generations not rechecked here"→ [部分证据]（历史数字未重算）。
- **ROPE_QWEN15_FULL_LAG_P2_RESULT**（1.5B，3 任务×8，64K+128K，COMPLETED）：64K niah_multikey_2 = P2 **37.5** / MrPro 12.5 / MrPro-matched-gain 25.0；vt = 87.5/82.5/77.5；fwe = 70.83/45.83/45.83。128K（同冻结表）mk2 = 0/0（全员 0，配对 0 胜 0 负）；vt = **85.0 vs 72.5**（paired 4 胜 1 负）；fwe = 50/50。3B 64K 迁移屏（rows=24）：见 json `checkpoint_transfer_3b64`。→ P2 短/中端收益首次出现 [已验证于 1.5B+局部 3B]。

### 2.2 2026-09-08（BM 主对照 + 跨模型 + GapCapped）
- **BM(MrProBM) 定义**（ROPE_MRPRO_BM_CANDIDATE json `definition`）：ε_i=6i(N+1−i)/(N(N+1)(N+2))，m_q=q(q+1)(3N+2−2q)/(N(N+1)(N+2))，freq=Native·s^(−m_q)。相对 MrPro 只改零基槽 24–39，中段频率下降约 1.5%–31.5%，最大在槽 34（ROPE_BM_128K_DIAGNOSIS_20260908.md:7）。
- **ROPE_BM_TRANSFER_RESULT（3B，36 行面板，配对完成）**：MrPro 基线 32K **87.2222% / 128K 78.1250%**；BM 32K **91.6667% / 128K 70.8333%**；macro_delta +4.44/−7.29pp；status=**NO_LONG_GAIN**；配对 6 胜 4 负；1.5B 分支保持用户中止（`user_cancelled_1p5b.reason = "User switched directly to the official-paper 3B checkpoint."`，完成 baseline 19 行）。
- **ROPE_GAP_CAPPED_RESULT**（GapCapped@3B 36 行，逐行 json 已本地重算）：32K 84.4444（vs 87.2222，**−2.78pp**）/ 128K 62.1528（vs 78.1250，**−15.97pp**）→ 与 ROPE_GAP_CAPPED_RESULT_20260908.md:8–9 一致 [已验证]。
- **ROPE_QWEN7_BM_RESULT（7B，18 行面板）**：MrPro 基线 32K 83.3333 / 128K 84.4444；BM 80.0000 / 71.1111；delta −3.33/−13.33；**0 胜 3 负** 15 平（ROPE_QWEN7_BM_TRANSFER_20260908.md:37 一致）→ 7B NO_LONG_GAIN [已验证]。
- **NLL 面板（探索性，16 篇冻结自然前缀尾 512）**：3B（ROPE_QWEN3_BM_NLL）8K/16K/32K：Native 2.27607/2.14317/2.03850；MrPro 2.32333/2.18886/2.08899；BM 2.32581/2.19311/2.08789（BM−MrPro：+0.0025/+0.0042/−0.0011，16K 的 bootstrap 区间 [0.00086,0.00747] 不含 0）。7B：Native 2.12345/1.99564/1.90778，MrPro 2.17685/2.03921/1.94661，BM 2.17689/2.04207/1.94942。→ NLL 是尾部代理，不是能力 [已验证为 NLL 本身]。
- **OLMo（ROPE_OLMO_BM_RESULT，4 组实验全在本地 json）**：dev 36 行 4K/16K：MrPro 37.2222/14.9306，BM **79.4444/49.0278**（DEVELOPMENT_WIN，+42.2/+34.1pp）；seed_replication 72 行（独立 seed 20260910）：MrPro 37.8472/2.7778，BM **81.8056/51.3194**（44 胜 0 负 28 平，与 ROPE_OLMO_FAST_SCREEN md:49 一致）；gain_followup（不晋级）：BMSelectiveGain 80.6944/44.7917、BMUniformMatchedGain 79.4444/44.2361（16K 密集多键检索 25%→0%，VT 15%→0/5%——ROPE_BM_SELECTIVE_GAIN md:21）；scale8（S=8）：MrPro 4.1667/0.6944，BM 23.8889/6.9444；existing_controls 72 行：MrUni 76.8750/32.1181，OfficialYaRN 54.3750/6.9444；native 短参照 24 行 4K 75.625。→ BM 在 OLMo 冻结权重静态 4× 下真实局部收益 [已验证]，不泛化到 3B/7B 的 128K。
- **OVERNIGHT_FAILURE_AUDIT_20260909.json**（09-09 01:29 UTC 对 position_observability 的只读审计）：`primary_compact_01` Dense 8 行 raw_exact_eos=0 而 trimmed=4（saved_text 与 raw 不一致 4 行）等——记账型审计，无面板数字。
- **LEDGER_20260908**：34 个 job 回执（含 FAILED→重跑），evidence_tiers 明确 "capability: Only completed raw generations…"。
- **SPARSE_MEMORY_INTERFACE_RESULTS**：phase_status=**CLOSED_ASSAY_UNQUALIFIED**，scientific_verdict=无 TP 合格证据（非面板结果，与 32K/128K 得分无关）。

### 2.3 2026-09-09（nongeometric 36 行全面板屏，本 digest 核心）
26 个方法目录在 `results/nongeometric_screen_20260909/results/`（构成见 §5 总表）。36 行 = 6 任务 ×（32K 2 行 + 128K 4 行）：row_id 全列于 §2.4 代码输出（ruler.jsonl，任务集 {niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1}）。所有 36 行方法的 baseline 均为同一 MrPro(gain .1) 32K **87.2222** / 128K **78.1250**（score_sum 29.2167，逐行 baseline_correct 与 09-08 BM_TRANSFER json baseline 完全一致——两次实验复用同一基线，无重跑）。12 行早期子集 = 每任务 `*_0` 行（6 短 6 长），其 MrPro 基线为 32K 100.000 / 128K **64.4444**（含 vt_0=.2、qa_0=0）。
- 赢家：**E1_s28_less**（36 行 87.2222/**83.3333**，2 胜 0 负，改的两行=`niah_multikey_2_131072_2` 0→1、`niah_multiquery_131072_3` .75→1）；**FullLagP2_Transfer3B**（72.9167/**81.6667**，6 胜 6 负；vs official +3.5417pp、vs same-gain control **+6.3194pp**、短端 −14.3056/−25.4167pp）；**LongBridgeSlower**（80.5556/**80.0694**，3 胜 3 负；128K VT .75→.95，即 "VT 75→95"）。
- gain 析因（4 格全本地）：MrPro@.1 87.2222/78.1250；BM@.1 91.6667/70.8333；MrPro@.074（Control_Mr_gain074）**98.3333/75.3472**；BM@.074（E3_BM_gain074）**100.0000/70.0000**（+12.778/−8.125 vs MrPro）；BM@1（E3_BM_gain1）89.5833/58.8194（相对 BM@.1 长端 **−12.0139pp** 而 NLL 改善 0.023–0.035 nats，本地 nll.jsonl 均值核算：8K 2.28882、16K 2.16604、32K 2.05513，均低于同输入各法 ≈2.32/2.19/2.09）。
- 方向对：LongBridgeFaster 87.2222/73.9583（唯一变化行 qa_1_131072_2 1→0，0 胜 1 负）；E1_pair28_29 87.2222/73.9583（**数值与 Faster 巧合相同、改的行相同**，但构造不同——引用时必须带方法名，不能只写"87.2/74.0"）。
- 反方向/端点破坏：Smooth_MrBudget 87.2222/68.3333（−9.7917pp）；HighGapToLong **70.1389**/67.3611（−17.0833/−10.7639，0 胜 7 负）；MrUni 64.5833/73.3333；E2_tail_more（12 行）100/**54.7222**（基线 64.4444，−9.72）；E8_zero51（12 行）100/**50.5556**（−13.889）。
- 12 行持平（各 0 胜 0 负，得分逐行等于基线）：E10_dual_frequency、E1_s28_reverse_matched、E1_s29_plus_matched、E4_pair25_29、E5_layer21/27/32、E6_layer14_group0/1、E7_norm_matched_BM、E9_distance。E7_local_projection 完成 36 行：90.0000/68.6111（+2.778/−9.514pp）。
- HighGapToMid：**中途中止**（16/36），数据在 `deferred_queue/20260910_candidate_quality/`（NONGEOMETRIC md:24–26），未计入任何已测表。
- 冻结表构造核对（contract.json `values_float32` + native ω=b^(−j/64), b=1e6 反推 m_j；脚本输出）：MrPro m28=**0.0980**、m36–39=**0.5948/0.6863/0.7843/0.8889**、m40=1.0；s28_less m28=**0.0654**（其余槽与 MrPro 逐点相同）；LBS 只改 36–39 → **0.6252/0.7295/0.8465/0.9799**（对应 D=W·4^m=77,954/90,082/105,953/**127,473**，与 GLM 复核稿 :24 完全一致）；P2 m30=0.8506、**m31=0.9979、m32 起 =1.0**（"31/32 槽即完成"成立），m28=0.0741（前端比 MrPro 贴原生）。
- P2 gap 增量（planned_controls/p2_gap_comparison.json）：高频带 gaps 0–23 合计 **+0.0007754**（max 相对改动 0.0775%）；gap28 +0.149445、**gap29 +0.809123**（部署绝对 gap=ln(13267/4468)=**1.088 巨洞** ✓ BUDGET 行）、gap30 +0.131703、gaps31–39 全为负。

### 2.4 2026-09-10（汇总、复核与新候选）
- 0441 P2_QA128_pilot（long_first_protocol.json）：passkey 复用单键 4/4=100%（P2、Control 均 1.0）；tail-512 PPL（4 篇 128K 前缀）：MrPro **5.503529** / Control_Mr_gain074 **5.370516** / P2 **5.357549**（P2 vs matched −0.2415%，2 升 2 降；NONGEOMETRIC md:38–48 一致）→ 放行 4 行新 QA pilot（268.7s）。pilot 本地 jsonl 实测：MrPro **1/4 正确（row_3）**，P2 0/3，Control 0/3，**P2/Control 缺第 3 行**（MrPro 唯一正确的那行）→ pilot 未构成 P2 QA 收益的独立确认 [中断/未执行完毕]。
- BUDGET_ALLOCATION_MODEL_AND_CANDIDATES / UNIFIED_BUDGET_ALLOCATION_THEORY：把上述面板总结为理论表（引用数字全部与本地原始文件一致，除 §3 两处标注）。三新候选 0446/0448/0449 与确认 0450/0451 **未执行**。
- ROPE_GLM_6PRO_REVIEW：对 UNIFIED 的多处因果表述纠偏（见 §3 纠正列）；其核对的 LBS m/D、质心表（Σm：MrPro 29.3333、s28_less 29.3007、LBS 29.5602、P2 34.1789；新增 log-gap 质心 34.667/34.699/34.440/**29.821**）与本地重算一致。
- PARALLEL_20X10_AUDIT：独立复核 E1/E3/E7/E8 数字（+8.333/−0.208、+12.778/−8.125、+2.778/−9.514、−13.889 全部与本地 summary.json 一致）；并声明 development_summary.json 过期、逐行以 results/*/summary.json 为准。

---

## 3. 理论主张表（主张 | 证据等级 | 出处 | 后续是否被纠正/推翻）

| # | 主张 | 等级 | 出处 | 纠正/后续 |
|---|---|---|---|---|
| A1 | MrPro 36 行锚点 32K 87.22 / 128K 78.13 | [已验证] | results/*/summary.json baseline；ROPE_BM_TRANSFER_RESULT json（09-08）；ROPE_GAP_CAPPED json 逐行重算；三源同一组 36 行，score_sum 29.2167 | 无；精确值 87.2222/78.1250，78.13 是四舍五入 |
| A2 | s28_less 128K 83.3 = +5.21pp、2 胜 0 负、修 mk_2@89K 与 mq_3 | [已验证]（仅开发面板） | summary.json + ruler.jsonl 逐行；AUDIT:17 | 确认队列 0450 未跑；AUDIT 警告开发证据不可当泛化 |
| A3 | LBS 80.6/80.1、"危险区完成手术 m36–39→0.63–0.98、修 106K VT(.2→1.0)" | [已验证得分]；机制归因 [部分证据] | summary.json；contract m 重算；GLM 复核 :24（D_39=127,473） | GLM 复核指出 LBS 并未"完成到 m=1"（0.98≠1）；UNIFIED "m_36–39=0.51–0.89" 应为 0.59–0.89（0.51 是槽 35）——**文档笔误** |
| A4 | P2 72.9/81.7、长端 +3.54pp（official）/+6.32pp（matched-gain），短端死于 29 槽 1.088 巨洞 | [已验证得分+算术]；巨洞归因 [部分证据] | summary.json（6W/6L）；NONGEOMETRIC :576–600；p2_gap_comparison | P2 新 QA pilot 0/3 未确认（n=3，缺关键行，不构成否证）；0451 未跑 |
| A5 | I1"bank 恒等"：从高频抽预算必输（HighGap −17.1/−10.8，36 行 0 提升） | [已验证]（该构造） | HighGapToLong summary；NONGEOMETRIC :564–574 | GLM 复核纠正同表 **MrUni 定义**："MrUni 不是全表 PI，是同一过渡区内线性累计压缩、外部与 MrPro 相同"（其 :11）→ UNIFIED §I1 "MrUni 全表 ÷4" 表述错误；64.6 得分本身 [已验证] |
| A6 | I2"尾部精确 ÷S"：E2 ÷4.93 → 128K 崩至 54.7 | [已验证]（12 行子集！） | E2_tail_more summary：100/54.7222，基线 100/64.4444 | 引用时必须注明 12 行面板，54.7 不可与 78.125 锚点直接比；实际 delta −9.72pp |
| A7 | "所有赢家都在把预算右移，跨 8 构造单方向梯度" | **被推翻** | UNIFIED §3 结论 | GLM 复核质心表：s28_less 质心 34.699（**右移**），LBS 34.440 与 P2 29.821（**左移**）——不同方向各有收益，方向梯度不成立 |
| A8 | 危险区 75–112K 地平线带 = MrPro 特异性失败的机制解释 | [假设→部分证据被降级] | UNIFIED §2/§4 | GLM 复核 :28–38：FWE 行距离定义不成立、VT"106K"只是链 max、MQ 行混合 77 与 120,934 token；两行翻盘样本过小；75–112K 不得称"已验证危险带" |
| A9 | gain 只改 logit 幅度、与相位弧正交（E3 四格证据） | [部分证据] | 四格本地可复算（§2.3） | GLM 复核 :26："softmax(g²z_ν) 同时依赖两者，通常存在非零交互"——四格不能证明效应正交；保留为操作事实（gain 不改频率） |
| A10 | E7 的 160× NMSE 差距 = 局部线性化失败 | **被纠正** | AUDIT:29–40 分解表（线性 7.9367e-8 vs 理想 7.9364e-8 vs FP32 7.9325e-8 vs **BF16 1.2807e-5**） | 差距由 BF16 实现主导，非一阶理论失败；下游失败另因（不能归因舍入也不能排除） |
| A11 | E8 说明固定态代理分数 ≠ 全模型能力 | [已验证]（反例本身） | E8 12 行 −13.889pp；AUDIT:21 | 已成标准教训（铁律#1 的案例来源） |
| A12 | "BM@gain1：NLL 改善 0.02–0.035 而 128K −12.0" | [已验证]（NLL 为尾 512 代理） | 本地 nll.jsonl 均值 + summary：58.8194 vs 70.8333 = −12.0139 | NLL 改善≠能力，出处均带此限定 |
| A13 | OLMo 局部双升（BM vs MrPro 4K/16K，两 seed）+ 与 QA 保持张力（Native 4K 75.625 vs BM 81.8056 但 QA 100→75） | [已验证]（OLMo 冻结、静态 4×） | ROPE_OLMO_BM_RESULT 全组 | FAST_SCREEN md:49/53 自限：不是全 RULER、不是 SOTA、不逐任务保持 |
| A14 | s28 规则冻结迁移 7B/OLMo | [已验证，负向/混合] | transfer/*/summary.json：7B 18 行 83.3333/82.7778 vs 基线 83.3333/84.4444（1W/1L，长端 −1.67pp）；OLMo 36 行 33.8889/18.7500 vs 37.2222/14.9306（4K −3.33 / 16K +3.82，4W/2L） | 说明单槽规则不可直接跨 checkpoint；"跨模型表述"主张目前只有 OLMo 16K 局部一致信号 |
| A15 | 长文 NLL 128K：P2 最优（PG19 2.27090 vs MrPro 2.30775；ProofPile 1.08611 vs 1.10303） | [部分证据]（**每源每长只有 2 篇**） | long_nll/*.jsonl 本地重算 | AUDIT:23 明示 2/8 篇、尾 512 度量；只作筛查 |
| A16 | "本轮 27 个已测面板" | [部分证据]（对账差异） | UNIFIED :3 | 本地 `results/` 恰 **26** 个方法目录；第 27 个最可能是 HighGapToMid(16/36 中止) 或把 0441 pilot 计入——未钉死，见 §7 |
| A17 | BUDGET §2 行 "E3 gain074 | 改 gain 非表 | 98.3/75.3" | 数字 [已验证]，**标签错位** | 98.3333/75.3472 = **Control_Mr_gain074**（MrPro 表+gain.074）；E3_BM_gain074 实为 100.0/70.0（PARALLEL_PLAN:221 的"E3 gain074×BM 100/70"正确） |

---

## 4. 失败机制清单

1. **BM 长端崩（3B 70.83@128K，−7.29pp；7B −13.33pp）**：中段槽 24–39 过度压缩（最大降幅槽 34）。复发警告：任何"给中段加压缩"的构造先对照 78.125 锚点核 128K，再看逐行（mk2@128K 判别面 75→50）。
2. **HighGapToLong（−17.1/−10.8，0 胜 7 负）**：EVQ 字面"从高频带搬 gap"对冻结模型全输；donor 0.2158674 log 单位直接从缠绕致密带取走。P2 只从高带取 +0.0007754（少 280 倍）——预算来源与放置必须分开归因。
3. **Smooth_MrBudget（−9.79pp 长端）**：固定预算下最小粗糙度反把危险区预算抽走；平滑先验 ≠ 弧安全。
4. **pair28_29 / Faster（各 −4.17pp）**：同位置叠加超调（洞 1.46×；Faster 单行 qa_1_131072_2 翻转）；"两个 36 行方法得分巧合相同（73.9583）"是引用陷阱，方法名必须随行。
5. **E2 平台过压 / E8 槽 51 置零**：破坏精确 ÷S 端点即付出 128K 代价（12 行 −9.72 / −13.89pp）；E8 还出现格式/截断失败（AUDIT:21）——**E8 的固定态高选分与实际退化并存**，是"代理≠能力"的最强本地反例，UNIFIED §8.3 已引为校准-KL 降级的理由。
6. **E7 局部投影**：局部保护准则未换来下游收益（−9.51pp）；其"160×"被复盘为 BF16 实现误差（AUDIT:29–40）。复发模式：**把实现数值问题当理论否证**，历史上已发生（E7 的 160× 教训被 UNIFIED §8.1-3 再次引用以约束 J_r 定位）。
7. **cached-path 归因失效**：QA s29 增益在缓存路径 4 格全 0（AUDIT:82）——同表缓存路径本身丢失了 full-prefill 成功，任何用 cached 记录做的机制归因要先验证通路保留。
8. **development_summary.json 过期**、**E5 layer27 提名因归一化下溢失效**（改后第二名 27→32）：根目录汇总不可作引用源，逐方法 summary 才是权威（AUDIT:13、:127）。
9. **0441 pilot 缺行**：P2/Control 各缺 qa_1_131072_3（恰是 MrPro 唯一正确行）→ 小样本比较被截断偏置；引用 pilot 必须写"3/3/4 行、不可比"。
10. **面板级 12 行 vs 36 行混用**：54.7/50.6 的基线是 64.444 不是 78.125；跨子集比大小是当前文档最容易复发的错误（本 digest §5 全部标注面板列）。

---

## 5. 方法定义与总表（方法 × 面板 × {32K, 128K, gain}）

构造规则出处：contract.json 表数组（本地重算 m）与 NONGEOMETRIC/BUDGET/CANDIDATE 各节；官方 gain=1+0.1·ln4=1.13863，P2 历史 gain=1+0.074·ln4≈1.10265。**[U]=该数字出现在 UNIFIED_BUDGET §3 表或 §1 不变量段。**

### 5.1 Qwen2.5-3B · 36 行开发面板（基线 MrPro .1 = 87.2222/78.1250）
| 方法 | 构造 | 32K | 128K | Δpp(32K/128K) | 胜负 | [U] |
|---|---|---:|---:|---|---|---|
| MrPro | 官方 (23,40) 过渡，m28=.0980,m39=.8889 | 87.2222 | 78.1250 | — | — | ✓ 87.2/78.1 |
| E1_s28_less | 槽28→前驱指数 m=.0654，其余同 MrPro | 87.2222 | **83.3333** | 0/+5.2083 | 2/0 | ✓ 87.2/83.3 |
| LongBridgeSlower(LBS) | 槽36–39 各 +1/131072 rad/token | 80.5556 | 80.0694 | −6.6667/+1.9444 | 3/3 | ✓ 80.6/80.1 |
| FullLagP2_Transfer3B | 历史 64 频表(m31=.998,m32=1)+gain.074 | 72.9167 | 81.6667 | −14.3056/+3.5417 | 6/6 | ✓ 72.9/81.7 |
| Control_Mr_gain074 | MrPro 表 @ gain .074 | 98.3333 | 75.3472 | +11.1111/−2.7778 | 4/2 | ✓（但被误标"E3 gain074"） |
| E3_BM_gain074 | BM 表 @ .074 | **100.0000** | 70.0000 | +12.778/−8.125 | 6/4 | ✓（四格；PLAN:221） |
| E3_BM_gain1 | BM 表 @ gain 1 | 89.5833 | 58.8194 | vs BM@.1 长端 −12.0139 | 7/13 | ✓（E3 四格/gain 正交条） |
| E1_s29_more | 槽29→后继指数 | 95.5556 | 77.9167 | +8.333/−0.208 | 2/1 | （AUDIT 表） |
| E1_pair28_29 | 28+29 双倍集中 | 87.2222 | 73.9583 | 0/−4.1667 | 0/1 | ✓ 87.2/74.0 |
| LongBridgeFaster | 槽36–39 各 −1/131072 rad/token | 87.2222 | 73.9583 | 0/−4.1667 | 0/1 | （NONGEOMETRIC:389） |
| Smooth_MrBudget | 固定 B=16/3 最小粗糙度，末 gap→.042 | 87.2222 | 68.3333 | 0/−9.7917 | 3/5 | ✓ 87.2/68.3 |
| HighGapToLong | gap0–22 均匀抽 .2158674 → gap36–39 | 70.1389 | 67.3611 | −17.0833/−10.7639 | 0/7 | ✓ I1 |
| MrUni | （定义被 GLM 稿纠正：过渡内线性累计压缩，非全表 PI） | 64.5833 | 73.3333 | −22.6389/−4.7917 | 4/6 | ✓ 64.6 |
| E7_local_projection | 局部保护准则选槽投影 | 90.0000 | 68.6111 | +2.778/−9.514 | 6/7 | （AUDIT） |
| BM(.1)（09-08） | β 加权中段加密（定义见 §2.2） | 91.6667 | 70.8333 | +4.44/−7.29 | 6/4(配对) | （PLAN/AUDIT） |
| GapCapped（09-08） | BM 变体 c=1.2365e-5 | 84.4444 | 62.1528 | −2.78/−15.97 | — | （§8 未直接引用） |

### 5.2 Qwen2.5-3B · 12 行子集（基线 MrPro = 100.0000/64.4444）
| 方法 | 32K | 128K | Δ128K | 备注 |
|---|---:|---:|---|---|
| E2_tail_more（平台推到 ÷4.93） | 100.0000 | **54.7222** | −9.72 | [U] "崩至 54.7（12 行面板）"✓ 已注明 |
| E8_zero51（槽 51 置零） | 100.0000 | **50.5556** | −13.889 | [U] "128K 50.6" 与 §8.3"−13.9pp" ✓ |
| E10/E1_s28_reverse/E1_s29_plus/E4/E5(21,27,32)/E6(g0,g1)/E7_norm_matched/E9 | 100.0000 | 64.4444 | 0 | 全部 0 胜 0 负；reverse/plus_matched 为镜像对照持平 |
| Native S=1/gain=1（native_reference，12 短行） | 83.3333 | — | — | QA=0、VT/FWE=1；6 任务均值对 2 个 QA 行敏感（NONGEOMETRIC:301–305） |

### 5.3 跨模型
| 模型/面板 | 方法 | 短 | 长 | Δ |
|---|---|---:|---:|---|
| 7B · 18 行 32K/128K | MrPro 基线 | 83.3333 | 84.4444 | — |
| 7B · 18 行 | BM(.1) | 80.0000 | 71.1111 | −3.33/−13.33，0W3L |
| 7B · 18 行 | s28 冻结规则（transfer/qwen7b） | 83.3333 | 82.7778 | 0/−1.67，1W1L |
| OLMo-1B · 36 行 4K/16K | MrPro 基线 | 37.2222 | 14.9306 | — |
| OLMo · 36 行 | BM(.1) | 79.4444 | 49.0278 | +42.2/+34.1 DEVELOPMENT_WIN |
| OLMo · 72 行(新 seed 20260910) | BM vs MrPro | 81.8056 vs 37.8472 | 51.3194 vs 2.7778 | 44W/0L/28T |
| OLMo · 72 行 | MrUni / OfficialYaRN | 76.8750 / 54.3750 | 32.1181 / 6.9444 | BM 对照 |
| OLMo · 36 行 | BMSelectiveGain / UniformMatched | 80.6944 / 79.4444 | 44.7917 / 44.2361 | 均 NO_LONG_GAIN（vs BM dev） |
| OLMo · S=8 · 72 行 4K/32K | MrPro / BM | 4.1667 / 23.8889 | 0.6944 / 6.9444 | BM 仍 DEVELOPMENT_WIN 但绝对分极低 |
| OLMo · 36 行 4K/16K | s28 冻结规则（transfer/olmo1b，界 14/32→槽19） | 33.8889 vs 37.2222 | 18.7500 vs 14.9306 | −3.33/+3.82，4W2L |
| OLMo · 24 行 4K | Native 短参照 | 75.6250 | — | BM 81.81 但 QA 100→75 |
| 1.5B · 3 任务×8 64K | FullLagP2 / MrPro / matched | mk2 37.5/12.5/25；vt 87.5/82.5/77.5；fwe 70.8/45.8/45.8 | 128K：mk2 0/0；vt 85/72.5；fwe 50/50 | 见 §2.1 |
| 3B · holdout mixed_s20260910(128K QA) | MrPro 1/4；P2 0/3；Control 0/3 | — | — | 缺行不可比，非确认也非否证 |

### 5.4 证据距离 × 方法 128K 关键行联表（UNIFIED §4 复核；得分=本地 ruler.jsonl，距离=文档出处）
| 行 | 距离[出处] | MrPro | s28_less | LBS | P2 |
|---|---|---:|---:|---:|---:|
| niah_multikey_2_131072_2 | 89K（88,726/88,725，NONGEOMETRIC:348;GLM:32） | 0.0 | **1.0** | 1.0 | 0.0 |
| vt_131072_0 | 106K（max=105,880，GLM:33） | 0.2 | 0.2 | **1.0** | 0.8 |
| vt_131072_1 | 127K（本地无独立出处 [假设]） | 1.0 | 1.0 | 1.0 | 1.0 |
| niah_multikey_2_131072_1 | 96K（95,676–95,682，NONGEOMETRIC:443） | 1.0 | 1.0 | **0.0** | 0.0 |
| niah_multikey_2_131072_3(记作 mk_0/117K) | 117K（117,487–117,493，:443） | 1.0 | 1.0 | 1.0 | 1.0 |
| qa_1_131072_0/1 | 90–110K（GLM 稿限定：QA 距离只是定位线索） | 0/0 | 0/0 | 0/0 | 0/**1** |

（UNIFIED 表把 4 条 multikey 行记作 mk_0/1/2 + "multikey_1@96K"；其 LBS 列 0.0 带 * 号，与本地一致。）

---

## 6. 用户指令与纠正（原文引用）

1. （ROPE_BM_TRANSFER_RESULT json `user_cancelled_1p5b.reason`，09-08）："User switched directly to the official-paper 3B checkpoint."
2. （ROPE_BM_TRANSFER_20260908.md:35）："1.5B保持用户中止状态。"
3. （NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:9–13）："**Latest protocol correction:** the user's subsequent instruction supersedes the old automatic full-panel completion rule. The active objective is long context: first inspect target-length PPL and passkey, then selectively release long downstream tests. No new 32K experiments or automatic 36-row completion are queued."
4. （同上 :21–31）："**Current priority correction (2026-09-10, after the user's candidate-quality critique):** the hand-built HighGap, G1/G2/G3 pair-gap, and BM_ScaleTaper branch has been withdrawn from the active queue… Unrun proposals are not labeled empirical failures."
5. （同上 :327–336）："The user identified an important error in the recent research emphasis: fixed budget, smoothness, and neighboring-gap geometry were being analyzed without making protection and improvement of long-distance interactions the primary criterion… The user explicitly cautions against turning a local long-distance diagnostic into the entire research program."
6. （ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:61）："用户 11:06 明确告知 GPU 恢复并允许验证。"（且 :61 同时记录"不自动执行 GLM 原来的五个候选"）
7. （MEMORY/AGENTS 铁律，本轮所有产出适用）：不得把代理指标说成能力结果；不得把未测写成否证；每个结论标注证据等级。

---

## 7. 未决问题

1. **27 vs 26**：UNIFIED 称"27 个已测面板"，本地逐方法权威目录恰 26 个。第 27 个是 HighGapToMid（16/36 中止，deferred_queue 有部分数据）还是把 0441 pilot / MrPro 基线计数在内？需要向 UNIFIED 作者钉定义。
2. **evidence_distances_20260910.json 本机缺失**：89K/106K/127K 距离联表只能靠服务器文件 + GLM 复核稿转述；vt_1=127K 一行连文档数值出处都没有。恢复 GPU/服务器后应把该 json 同步进本地镜像并逐条复算。
3. **0446/0448/0449/0450/0451 全部未执行**：StackFrontBack 双机制可叠加性、N′∈{16,15} 单调族、s28/LBS/P2 的新样本确认、holdout 距离-准确率曲线——方向规则目前全部停在"开发面板 [已验证得分] + 机制归因 [部分证据]"。
4. **0441 pilot 缺行**（P2/Control 的 qa_1_131072_3）：补齐前 P2 新 QA 收益方向悬置（既不确认也不否证）。
5. **A7 被推翻后的统一解释问题**：s28（右移）与 LBS/P2（左移）各有收益，UNIFIED §3 "单方向梯度"表述需要重写；候选统一变量（弧安全 D_j 联合版、洞比率、内容条件 margin）均未闭环（AUDIT "Minimal useful continuation" 与 GLM §验证中 1–4 是既定路线）。
6. **LBS 32K −6.67pp 的短端代价**（multikey 1.0→0.5）在文档中只被一句带过；与 0446 判定规则（"32K≥80%"）耦合，未预注册单独归因。
7. **每方法 48 NLL 输入哈希一致但 audit 限定 16 篇/尾 512**：NLL 与面板判决的一致性校准（PARALLEL_PLAN 实验设计中的"OOD-NLL 与面板判决一致性"）未做。
8. **12 行子集基线 64.4444 的引用规范**：UNIFIED 已标注两处（E2、12 行），但 BUDGET/其他稿若沿用 54.7/50.6 仍可能误导与 78.125 的比较——建议所有汇总表格强制带面板列。
