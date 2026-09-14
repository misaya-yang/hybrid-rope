# R1 — 开发面板数据完整挖掘（K2 泛函拟合训练集）

日期：2026-09-10。角色：挖掘与整理（不推导）。范围：36 行 near/far 开发面板，Qwen2.5-3B，32K/128K。
纪律：只读；本文件是本次唯一的写入物。

---

## 0. 本文档的可靠性分级与一个必须先讲的重大缺口

### 0.1 三档证据来源

| 档 | 来源 | 可复核性 |
|---|---|---|
| **T1 本地原始 JSON**（09-07 / 09-08 实验） | `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json`、`ROPE_GAP_CAPPED_RESULT_20260908.json`、`ROPE_QWEN7_BM_RESULT_20260908.json`、`ROPE_OLMO_BM_RESULT_20260908.json`、`ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`、`ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json` | 本地可逐行打开，可重算 |
| **T2 数值化重建表** | `analysis/unify_20260910/tables/ground_truth_tables.json`（321KB，38 方法）+ `tables/GROUND_README.md`（169 行）+ `tables/rebuild_ground_truth_tables.py` | **可打开，但其输入镜像本地缺失——见 §0.2** |
| **T3 前序代理 digest / 报告叙述** | `analysis/unify_20260910/digests/`、`digests_codex/`、`docs/research/*.md` | 只能引用，不能独立复核 |

### 0.2 【重大缺口，逐格追溯性的前提】部署镜像 `results/nongeometric_screen_20260909/` 本地不存在

- `rebuild_ground_truth_tables.py:31` 定义 `MIRROR = ROOT/'results/nongeometric_screen_20260909'`，其 `load_contract()`（:52-59）从 `MIRROR/results/<method>/contract.json` 读部署张量，`load_summary()`（:61-64）从同一镜像读 `summary.json`。
- 实测：`/Users/yang/projects/hybrid-rope/results/nongeometric_screen_20260909` **不存在**（该目录下只有 `bm_transfer_*`、`olmo_fast_screen_*` 等 09-08 及更早目录）；全仓 `find` 也搜不到 `reference_tables.json`。
- 因此 `ground_truth_tables.json` 里形如 `sha256_deployed == sha256_rebuilt` 的"逐位一致"声明，**在本机无法复现**（脚本现在跑不起来）。它的 `sources` 字段指向 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/results/...`，同样是本机不存在的另一个 checkout。
- **后果**：T2 档的 26 个 09-09 非几何方法，其"32K/128K 分数"与"构造式 vs 部署张量 bit-exact"两项，本文只能标注为 **[部分证据-单源重建]**，除非该数字另有本地文档锚点（→ 升为 **[已验证-双源]**）。这是本文对 K2 拟合集最大的诚实性限制。

### 0.3 分级符号

- `[已验证-双源]`：T2 重建表 + 一份独立本地文档（`INTEGRATION_20260910.md` §3 / `PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md` / `NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md` / `digests_codec/digest_nongeo-code.md`）取值一致。
- `[已验证-本地JSON]`：T1 档，本地原始 JSON 直读。
- `[部分证据-单源]`：仅 T2 重建表或仅 T3 叙述。
- `[假设]` / `[叙事-未验证]`：标注假设性构造与叙述性说法。

### 0.4 坐标与常量（K2 直接复用的定义与数字）

出处：`tables/ground_truth_tables.json:meta`（`constants` / `definitions` / `evidence_policy` / `score_units`）。

| 量 | 值 | 定义 |
|---|---|---|
| W / S / L | 32768 / 4 / 131072 | 训练窗 / 外推倍率 / 目标长 |
| base b | 1e6 | `ω_j = b^(−j/64)`，`j` zero-based，末对 = `b^(-63/64)` |
| lnS | 1.3862943611198906 | `ln4`，过渡段增量守恒常数 |
| native_log_gap | 0.21586735246819178 | `ln(b)/64` |
| native_period_ratio | 1.2409377607517196 | `b^(1/64)`，原生最小洞比 |
| official_gain | 1.138629436111989 | `1+0.1·ln4` |
| p2_gain | 1.102585782722872 | `1+0.074·ln4` |
| `m_j` | — | `log(ν_j/ω_j)/log4`（由 fp32 部署值反演） |
| `T_j` | — | `2π/ν_j` |
| `D_j` | — | `W·4^{m_j}`（识别地平线） |
| `r_j` | — | `W/T_j^native`（原生周期数，全方法相同） |
| `gap_j` | — | `ln(ν_j/ν_{j+1})`，`j=0..62`（0-based gap g 介于槽 g、g+1） |
| 守恒式 | `Σ_{g=23}^{39}(gap_g − ln b/64) ≡ ln4` | **端点固定表的恒等式**（坐标点名：过渡段 log-gap 增量） |
| 洞比 | `ρ_g = T_{g+1}/T_g` | 原生 1.2409 |
| 分数单位 | 百分数 | `= summary.json candidate.by_length.macro_accuracy × 100` |

**未定坐标**：自由增量在槽 24–39，共 16 个 Δ（17 个 gap）；`Σm` **不是**守恒量（`ground_truth_tables.json` 各方法 `sum_m_all_slots` 从 25.21 到 34.18 不等，见 §4）。

---

## 1. 主面板表（36 行，Qwen2.5-3B，32K/128K）——逐格可追溯

评测协议统一说明：**六任务 RULER 开发子集** = `{niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1}`；32K 每任务 ×2 行、128K 每任务 ×4 行 ⇒ 合计 **36 行**；分数 = 长度内六任务宏平均。（出处：`ROPE_BM_TRANSFER_RESULT_20260908.json:scope = "Six-task RULER development subset; not full RULER"`；`digests/digest_nongeo-code.md:72`。）"36 行"是**评测行数**，不是方法数、不是任务数——这是一个被反复写混的口径。

| # | 方法 | 32K | 128K | 行 | W/L | gain | 构造（出处） | 证据等级 |
|---|---|---|---|---|---|---|---|---|
| 1 | **MrPro**（基线） | 87.2222 | 78.1250 | 36 | — | 1.1386294 | `m_j=q(q+1)/(N(N+1))`, `q=clip(j−23,0,17)`, `N=17`, `j≥40 m=1`；出处 `ground_truth_tables.json:methods.MrPro` + `USER_PROMPT_TRANSCRIPT_20260909.md:214` + `scripts/analysis/build_boundary_matched_mrpro.py:59`；T1 直读 `ROPE_BM_TRANSFER_RESULT_20260908.json`（87.22222222222222/78.125） | **已验证-双源+本地JSON** |
| 2 | **MrProBM**（BM，官方 gain） | 91.6667 | 70.8333 | 36 | 6/4 | 1.1386294 | `ε_i=6i(N+1−i)/(N(N+1)(N+2))`, `m_q=q(q+1)(3N+2−2q)/(N(N+1)(N+2))`, `N=17`；T1 直读 `ROPE_BM_TRANSFER_RESULT_20260908.json`（91.666…/70.833…，`status=NO_LONG_GAIN`） | **已验证-本地JSON** |
| 3 | **E1_s28_less**（s28; 128K 最强） | 87.2222 | **83.3333** | 36 | 2/0 | 1.1386294 | 仅槽 28：`m28 := m27`（0.098039→0.065359）；出处 `experiments/nongeometric_screen/select.py:76-80`；改善行 `niah_multikey_2_131072_2`、`niah_multiquery_131072_3` | **已验证-双源** |
| 4 | **E1_s29_more** | **95.5556** | 77.9167 | 36 | 2/1 | 1.1386294 | 仅槽 29：`m29 := m30`；32K 全部增益来自**一条 QA 行**，128K 由一 VT 改善与一 multiquery 回退近乎抵消 | **已验证-双源** |
| 5 | **E1_pair28_29** | 87.2222 | 73.9583 | 36 | 0/1 | 1.1386294 | 槽 28 与 29 同时刻双手术 → `ε28+ε29+ε30` 集中入 gap28（1.4608× 洞）；逐槽收益**不可叠加**的判决性反例 | **已验证-双源** |
| 6 | **Smooth_MrBudget** | 87.2222 | 68.3333 | 36 | 3/5 | 1.1386294 | 固定预算 B=16/3 的最小粗糙度 KKT 解累加（界 23/40，端点同 MrPro）；NLL Δ = +0.000975/+0.001925/+0.003423 @8K/16K/32K | **已验证-双源** |
| 7 | **MrUni** | 64.5833 | 73.3333 | 36 | 4/6 | 1.1386294 | **过渡段均匀斜坡 `m_j=(j−23)/17`**（槽 24–39），段外与 MrPro 逐位相同；**不是"全表 ÷4"**（UNIFIED 旧表述已由 GLM 复核纠正，本文 bit_exact=True 独立确认） | **已验证-双源** |
| 8 | **HighGapToLong** | 70.1389 | 67.3611 | 36 | 0/7 | 1.1386294 | gaps0–22 各减 `ln(b)/64/23=0.009386`（总抽 0.2158674），gaps36–39 各加 1/4；**ΣT=1.602162（违反 I1，m23=−0.1557）** | **已验证-双源** |
| 9 | **LongBridgeSlower** | 80.5556 | **80.0694** | 36 | 3/3 | 1.1386294 | 槽 36–39 `ν_j −= 1/131072`（公共相位 −1 rad@128K）；D≈77954/90082/105953/127473；VT .75→.95、FWE .75→.6667 | **已验证-双源** |
| 10 | **LongBridgeFaster**（方向控制） | 87.2222 | 73.9583 | 36 | 0/1 | 1.1386294 | 同幅反号 `ν_j += 1/131072` | **已验证-双源** |
| 11 | **FullLagP2_Transfer3B** | 72.9167 | **81.6667** | 36 | 6/6 | **1.1026** | 历史 FullLagP2 整表（1.5B 资产恢复，`m31=0.998` 完成、`m32` 起 =1.0）；`Σm=34.178915`，**max 洞 2.9696@29**；短端 −14.31/−25.42pp | **已验证-双源** |
| 12 | **Control_Mr_gain074**（gain 析因） | 98.3333 | 75.3472 | 36 | 4/2 | 1.1026 | MrPro 表 @ gain .074（表同、幅度不同） | **已验证-双源** |
| 13 | **E3_BM_gain074**（gain 析因） | **100.0** | 70.0 | 36 | 6/4 | 1.1026 | BM 表 @ gain .074；相对 MrPro：32K +12.778 / 128K −8.125 → **表与 gain 的包，不是 gain 的孤立因果效应** | **已验证-双源** |
| 14 | **E3_BM_gain1**（gain 析因端点） | 89.5833 | 58.8194 | 36 | 7/13 | 1.0 | BM 表 @ gain 1；长端相对 BM@gain1 自身 −12.0139pp | **已验证-双源** |
| 15 | **E7_local_projection** | 90.0 | 68.6111 | 36 | 6/7 | 1.1386294 | BM 方向局部响应约束投影（`project.py`；修正约束后 ≡ BM，历史部署表保留） | **已验证-双源** |
| 16 | **GapCapped** (09-08) | 84.4444† | 62.1528† | 36 | 0/7 | 1.1386294 | BM 变体，慢带载波帽 `c=1.2365e-5`；T1 `ROPE_GAP_CAPPED_RESULT_20260908.json`（84.4444…/62.1527…，Δ=−0.0278/−0.1597，`baseline_reused:true`） | **已验证-本地JSON** |

† `ground_truth_tables.json` 该行 `sum_transition` 为 1.386294、`maxhole=1.44759@39` 与 BM 同；但 T1 JSON 的 `candidate` 值是 84.4444/62.1528，与 BM 的 91.6667/70.8333 不同 ⇒ **GapCapped 与 MrProBM 不是同一张表**（帽位/帽值不同）。这一点留作未解问题（§7-Q3）。

**gain 的精确语义（F 零件，勿混）**：实现把旋转后的 Q 与 K **都**乘 `gain` ⇒ 固定状态下 attention logits ×`gain²`（出处：`PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md:44`）。因此改 `c` 只改 softmax 温度，**不改频率、不改角跨度**；"BM×gain074 恢复相位弧"的说法被明确否决（同处 :44-46）。

---

## 2. 12 行子面板（`*_0` 行）——**与 36 行不可比**

出处：`digests/digest_nongeo-code.md:14`（`panel small=12 行(_0)`）、:88-90、:202-209。

| 方法 | 32K | 128K | W/L | 构造 | 证据等级 |
|---|---|---|---|---|---|
| **12 行基线（MrPro 表）** | **100.0** | **64.4444** | 0/0 | 由 E4 / 镜像控制臂的 `panel_scores` 反推（`ground_truth_tables.json:methods.{E4_pair25_29,E1_s28_reverse_matched,E1_s29_plus_matched}.panel_scores`） | **部分证据-单源** |
| **E2_tail_more** | 100.0 | **54.7222** | 0/2 | 槽 40–63 频率 `×1e6^(−1/64)` ⇒ 平台水平 `4^{1.15572}≈÷4.931`（**违反 I2**，`m40=1.15572`，ΣT=1.602162） | **已验证-双源** |
| **E8_zero51** | 100.0 | **50.5556** | 0/2 | `ν_51 = 0`（慢带单槽置零）。失败分型 = **终止/格式型**，非检索型 | **已验证-双源** |
| E4_pair25_29 | 100.0 | 64.4444 | 0/0 | 槽 25/29 共享 `m=(m25+m29)/2=0.078431` | 部分证据-单源（空结果） |
| E1_s28_reverse_matched | 100.0 | 64.4444 | 0/0 | 槽 28 反向镜像，部署 `m28=0.132270`；**候选镜像式 `2m28−m27=0.130719 ≠ 部署值`，精确构造式未找到** | 部分证据-单源 |
| E1_s29_plus_matched | 100.0 | 64.4444 | 0/0 | 槽 29 镜像，部署 `m29=0.094729`，构造式无记录 | 部分证据-单源 |
| E5/E6/E7_norm_matched/E9/E10（算子臂） | 100.0 | 64.4444 | 0/0 | 层/头/KV 组粒度算子，无独立 64 槽静态表 | 部分证据-单源 |
| native 短参照（S=1, gain=1） | **83.3333** | — | — | `ν_j=ω_j, m≡0`；NLL 2.27607/2.14317/2.03850 @8K/16K/32K | 部分证据-单源（见 §5-F3 口径疑问） |

**口径旗标 F1【必标】**：12 行基线的 32K = **100.0**、128K = **64.4444**，而 36 行基线为 87.2222/78.1250。E2 的 54.7222 与 E8 的 50.5556 **只能**与 64.4444 比，不能与 78.1250 比；把它们写进 36 行表会凭空造出 −23pp 的假落差。K2 拟合时这两点必须带一个 `rows=12` 的指示变量或单独归一。

**口径旗标 F2【必标】**：12 行子集在 32K 上**全员满分 100.0**（含 native 之外的 11 个干预臂中除 MrUni 外的全部）⇒ **32K 维度在 12 行上没有区分度**，只有 128K 有信息。K2 的 F 若要用 12 行点，等于只用了 128K 一维。

**口径旗标 F3【必标】**：`digests/digest_nongeo-code.md:116,191` 把 native 短参照写成"12 行 83.3333（vs MrPro 87.2222）"——**分母口径混排**：87.2222 是 **36 行 32K** 基线，83.3333 是 **12 行**参照。若 native 12 行确为 83.3333，则 12 行 32K 并非全员满分（与 E4 臂的 100.0 矛盾）。此二说不可同真，**未解**（§7-Q1）。

---

## 3. 非 36 行面板的对照数据（**不得与主面板混排**）

### 3.1 OLMo-2-0425-1B-Instruct（4K/16K；S8 臂为 4K/32K）——不同模型 + 不同长度

出处（T1）：`docs/research/ROPE_OLMO_BM_RESULT_20260908.json`（111KB）。

| 臂 | 4K | 16K | 行 | 证据 |
|---|---|---|---|---|
| MrPro（development） | 0.3722222 | 0.1493056 | 36 | 已验证-本地JSON |
| MrProBM（development） | **0.7944444** | **0.4902778** | 36 | 已验证-本地JSON |
| MrPro（seed replication） | 0.3784722 | 0.0277778 | 72 | 已验证-本地JSON |
| MrProBM（seed repl.） | 0.8180556 | 0.5131944 | 72 | 已验证-本地JSON |
| BMSelectiveGain | 0.8069444 | 0.4479167 | — | 已验证-本地JSON |
| BMUniformMatchedGain | 0.7944444 | 0.4423611 | — | 已验证-本地JSON |
| MrUni（control） | 0.76875 | 0.3211806 | — | 已验证-本地JSON |
| OfficialYaRN（control） | 0.54375 | 0.0694444 | — | 已验证-本地JSON |
| BMCappedS4（scale8） | 0.7340278 | **0.0** | — | 已验证-本地JSON |
| BMFreq8Gain4 | 0.2909722 | 0.0555556 | — | 已验证-本地JSON |
| MrPro（scale8 4K/32K） | 0.0416667 | 0.0069444 | — | 已验证-本地JSON |
| MrProBM（scale8） | 0.2388889 | 0.0694444 | — | 已验证-本地JSON |
| native 参照 | 0.75625 | — | 24 | 已验证-本地JSON |

另有汇报口径（`ROPE_OLMO_BM_RESULT_20260908.md` + `INTEGRATION_20260910.md:52`）：OLMo **16K 350 条**上 BM 41.67% vs MrPro 7.09%（+34.59pp，156W/9L；`multikey_3` 双方皆 0）。**EOS 计数：BM 119 / MrPro 197** —— 两个口径（整串完成 vs 官方 recall）**不得混用**（出处：`INTEGRATION_20260910.md:52`，明写"评分口径不得混用"；`ROPE_OLMO_BM_RESULT_20260908.json` 各 record 的 `scope` 字段）。OLMo QA 一列 100→75。

### 3.2 Qwen2.5-7B（18 行）

出处（T1）：`docs/research/ROPE_QWEN7_BM_RESULT_20260908.json`。

| 臂 | 分数1 | 分数2 | Δ | W/L |
|---|---|---|---|---|
| MrPro | 0.8333333 | 0.8444444 | — | — |
| MrProBM | 0.7999999 | 0.7111111 | −0.03333/−0.13333 | 0W3L |

### 3.3 Qwen2.5-1.5B FullLagP2（64K/128K）

出处（T1）：`docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`。

| 臂 | 分 |
|---|---|
| summary64 mk2：P2 / MrPro / matched | 37.5 / 12.5 / 25.0 |
| summary128 mk2：P2 / MrPro | 0 / 0 |
| `checkpoint_transfer_3b64`（3B, 64K, 12 行）：P2 / MrPro | 75.0 / 50.0 |

（该 JSON 内还有 `user_cancelled_1p5b.reason = "User switched directly to the official-paper 3B checkpoint."` —— 说明 1.5B 线被主动放弃，**这不是失败**。）

### 3.4 外部参照 YaRN-ref（**Qwen 3B 36 行面板上不存在**）

- `ground_truth_tables.json:methods.YaRN_linear_official`：`role=external-baseline`, `panel="非本屏（历史/官方）"`, **分数 null**。`bit_exact=true` 只针对**表构造**（`scripts/analysis/export_static_rope_baselines.py build_tables()`，`low=23, high=40`，线性 ramp `ν=ω/4·ramp+ω(1−ramp)`），**不含面板分数**。
- 唯一可引的 YaRN 数字来自 **MrRoPE 论文 Table 2**（`docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md:93`）：Qwen2.5-3B zero-training 32K→128K，**13 任务 RULER**：MrPro 82.3/82.9/78.5/70.4/53.2、YaRN 78.1/77.7/75.6/63.2/50.1 @8/16/32/64/128K。
- **口径旗标 F4【必标】**：论文表是 **13 任务**、且是**另一次运行**；本文主面板是**六任务开发子集**。**78.1 与 87.2222 不可比，50.1 与 78.1250 不可比**。K2 若需要 YaRN 点，必须先在 36 行面板上跑 YaRN，或把 YaRN 点单独标注为异协议外点。
- 另有 OLMo 上的 `OfficialYaRN` 4K/16K = 0.54375/0.0694444（§3.1）——同样是异模型异长度。

---

## 4. 构造—几何诊断表（K2 的自变量侧：ν、Δ、D、洞）

数值出处：`tables/ground_truth_tables.json` 各方法 `sum_m_all_slots` / `sum_transition_gap_increments_gaps23_39` / `hole_ratio_max_transition` / `endpoint_delta_m`。这些是**表的静态函数**，可以逐位复核（前提见 §0.2）。

| 方法 | `Σm` | `ΣT`(过渡增量) | max 洞 (@gap) | `m23` | `m40` | I1 | I2 |
|---|---|---|---|---|---|---|---|
| Native | 0.0 | 0.0 | 1.24094@28 | 0 | 0 | — | — |
| MrPro | 29.333333 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| MrProBM | 32.000000 | 1.386294 | 1.39340@31 | 0 | 1 | ✔ | ✔ |
| E1_s28_less | 29.300653 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| E1_s29_more | 29.379085 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| E1_pair28_29 | 29.346405 | 1.386294 | **1.46077@28** | 0 | 1 | ✔ | ✔ |
| E2_tail_more | 33.070502 | **1.602162** | 1.79637@39 | 0 | **1.15572** | ✔ | **✘** |
| E8_zero51 | 28.333333 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| Smooth_MrBudget | 29.333333 | 1.386294 | 1.45134@35 | 0 | 1 | ✔ | ✔ |
| MrUni | 32.000000 | 1.386294 | **1.34637@28** | 0 | 1 | ✔ | ✔ |
| HighGapToLong | 25.206876 | **1.602162** | 1.52786@39 | **−0.1557** | 1 | **✘** | ✔ |
| HighGapToMid | 26.608314 | **1.602162** | 1.44759@39 | **−0.1557** | 1 | **✘** | ✔ |
| LongBridgeSlower | 29.560179 | 1.386294 | **1.49298@38** | 0 | 1 | ✔ | ✔ |
| LongBridgeFaster | 29.125312 | 1.386294 | 1.61920@39 | 0 | 1 | ✔ | ✔ |
| FullLagP2_Transfer3B | **34.178915** | 1.385519 | **2.96956@29** | 0.0006 | 1 | ≈✔ | ✔ |
| Control_Mr_gain074 | 29.333333 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| E3_BM_gain074 / gain1 | 32.000000 | 1.386294 | 1.39340@31 | 0 | 1 | ✔ | ✔ |
| E7_local_projection | 30.393463 | 1.386294 | 1.55152@34 | 0 | 1 | ✔ | ✔ |
| GapCapped | 28.000000 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| BM_ScaleTaper（未跑） | 30.801913 | 1.386294 | 1.44759@39 | 0 | 1 | ✔ | ✔ |
| StackFrontBack（未跑） | 29.527499 | 1.386294 | 1.49298@38 | 0 | 1 | ✔ | ✔ |
| MrProN16（未跑） | 30.000000 | 1.386294 | 1.46077@38 | 0 | 1 | ✔ | ✔ |
| MrProN15（未跑） | 30.666667 | 1.386294 | 1.47573@37 | 0 | 1 | ✔ | ✔ |
| YaRN_linear_official | 30.104186 | 1.386294 | 1.45993@39 | 0 | 1 | ✔ | ✔ |
| YaRN_smoothstep（假设） | 30.516166 | 1.386294 | 1.42076@35 | 0 | 1 | ✔ | ✔ |
| NTK_static | 32.000000 | 0.374079 | 1.26855@28 | **0.3651** | 0.6349 | **✘** | **✘** |

**读表须知（红线）**：`Σm` 在表中从 25.21 到 34.18 跨度很大，且 MrUni / NTK / MrProBM / E3_BM 四者 `Σm` 均为 **32.000000** 却分数天差地别（64.58/—/91.67/100.0）。**`Σm` 不是守恒量、也不是选择子**。（另注：`MrProBM` 与 `E3_BM*` 的 `Σm=32.00000007223662` 相同，因为三者是同一张 BM 表；`E1_pair28_29`、`E1_s28_less`、`E1_s29_more` 的 `Σm` 分别 29.3464/29.3007/29.3791，均 **≠** `MrPro` 29.3333，虽然它们在 gap 守恒下必然近似相等——这是 fp32 落地误差，**不是**自由度。）

---

## 5. 必须显式标注的口径不一致（K2 使用前先处理）

| 编号 | 冲突 | 两边出处 | 处理建议 |
|---|---|---|---|
| **F1** | 36 行基线 (87.2222/78.1250) vs 12 行基线 (100.0/64.4444) | `ground_truth_tables.json:methods.*.panel_scores` vs `digest_nongeo-code.md:14` | 加 `rows` 指示；E2/E8 只与 12 行比 |
| **F2** | 12 行 32K 全员满分 ⇒ 无区分度 | `ground_truth_tables.json`（E4/镜像臂 100.0） | 12 行点只贡献 128K 一维 |
| **F3** | native 短参照到底 83.3333 还是 100.0 | `digest_nongeo-code.md:116,191`（83.3333，且拿来比 87.2222=36行） vs E4 臂 12 行 100.0 | **未解**，见 §7-Q1 |
| **F4** | YaRN-ref 无 Qwen 3B 36 行分数 | `ground_truth_tables.json:methods.YaRN_linear_official`（score null） vs 论文 Table 2 的 13 任务值 | 要么补跑，要么标异协议外点 |
| **F5** | OLMo 4K/16K 与 Qwen 32K/128K 混排 | `INTEGRATION_20260910.md:52` vs 主面板 | 禁止混排；OLMo 用独立子表 |
| **F6** | EOS 计数（BM 119 / MrPro 197）与 recall 口径 | `INTEGRATION_20260910.md:52` 明写"评分口径不得混用" | OLMo 行必须带口径标签 |
| **F7** | "E3 gain074 = 98.3/75.3" 标签错位 | `INTEGRATION_20260910.md:53` vs `ground_truth_tables.json`：**98.3333/75.3472 是 `Control_Mr_gain074`**；`E3_BM_gain074` = **100.0/70.0** | 见 §6-C4 |
| **F8** | `ground_truth_tables.json` 自报 bit-exact，但输入镜像本地缺失 | `rebuild_ground_truth_tables.py:31` vs 本机 `results/` 列表 | 26 个 09-09 方法降为 [部分证据] |
| **F9** | 09-09 非几何屏是 26 还是 27 个方法目录 | `digest_panel-results.md` 称 27 tested panels vs 本地 `ground_truth_tables.json` 38 项中 26 个 09-09 方法 | **未解**，见 §7-Q2 |
| **F10** | "危险带 75–112K 已验证" | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:§4` vs `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:38`（"不能据此将 75–112K 宣称为已验证的危险带"；36 行**不是同一种距离—准确率样本**；FWE 的距离定义**不成立**） | 采信 GLM 复核；危险带表述降级为 [叙事-未验证] |
| **F11** | `evidence_distances_20260910.json` 本地缺失 | GLM 复核 :30 说已读服务器上 `planned_controls/evidence_distances_20260910.json` | 异距诊断不能本地复核；见 §7-Q4 |
| **F12** | 分数单位为百分数 vs 0–1 | `ground_truth_tables.json:meta.score_units` 明示 `_pct` 为百分数；OLMo JSON 用 0–1 小数 | 合并时统一乘 100 |

---

## 6. 矛盾清单（与权威文档 / 报告叙述）

| 编号 | 命题 A | 命题 B | 判定 |
|---|---|---|---|
| **C1** | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md §3`："所有赢家同方向移动预算" | `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:14-22` 槽位质心表：E1 s28 less 质心 34.699（**后移**）、LBS 34.440（前移）、P2 29.821（**前移**） | **B 成立**。无单一方向梯度 ⇒ 任何"压缩高频/让给低频"的单调方向假设**不能**作为 F 的结构约束。 |
| **C2** | `UNIFIED…§3` / 任务描述："MrUni = 全表 ÷4" | `ground_truth_tables.json:methods.MrUni`（bit_exact=True）：过渡段**内部线性斜坡** `m_j=(j−23)/17`，段外与 MrPro 逐位相同；`ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:11` 同 | **B 成立**。A 是错的；MrUni 是 I1/I2 保持下的**内部重分配**。 |
| **C3** | `UNIFIED…`："MrPro m36–39 = 0.51–0.89" | `ground_truth_tables.json:mismatches[6]`：部署与公式重建均为 `m36=0.5948…0.8889`；**0.5098 实为 m35**（槽位错位） | **B 成立**（GLM 复核 :10 判笔误；重建表独立复验，D 带 74,737/84,845/97,197/112,361 与 0.5948–0.8889 一致）。 |
| **C4** | `INTEGRATION_20260910.md:53`："E3 gain074 98.3/75.3" | `ground_truth_tables.json`：98.3333/75.3472 = **Control_Mr_gain074**；E3_BM_gain074 = **100.0/70.0** | **B 成立**（`digest_panel-results.md` A17 已标标签错位）。 |
| **C5** | `BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:45-47`：三个 queue 候选 max 洞 = 1.86 / 1.76 / 1.80 | `ground_truth_tables.json:mismatches[0-5]` 按 §1-2 定义 `ρ=T_{g+1}/T_g` 重算 = **1.49298 / 1.46077 / 1.47573** | **B 成立**；A 的"洞"公式未给出 ⇒ 文档需澄清定义或修正（两组数字都在最多 6 个 mismatch 条目里重复出现）。 |
| **C6** | `docs/research/*`："YaRN 递减 vs MrPro 递增" | `STARTING_POINT_YARN_VS_MRPRO.md` F1/F2：**该叙述已被证伪**，两者都凸、都递增 | **B 成立**（权威文档自陈）。 |
| **C7** | `PARALLEL_NONGEOMETRIC_20X10_PLAN` 的两力框架叙述 | `PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md:7`：E7 不证明局部线性化失败；gain 不恢复相位弧；增大频率一般**不**增加周期性位置分离 | **B 成立**（审计为独立复核）。 |
| **C8** | "E7 局部线性化低估 160 倍" | 同审计 :29-38：161× 是 **BF16 落地误差**（linear 7.9366844e-8 / ideal 7.9364028e-8 / FP32 7.9325473e-8 / **BF16 1.2807392e-5**），且是**平方范数比**不是幅度比 | **B 成立**；A 降级为 [叙事-未验证]。 |
| **C9** | "17 个 log-gap 总和 = ln4" | `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:7`：17 个**实际** log-gap 总和 ≈ **5.056039**（原生部分 ≈3.669745）；= ln4 的是**增量** `Σ(m_j−m_{j−1})` | **两者都对但坐标不同**：把"守恒"写成 gap 和是错的；写 `ΣΔm = ln4` 才对。**必须点名坐标**（红线）。 |

---

## 7. 未解问题（K2 拟合前需裁决）

- **Q1（F3）**：12 行 native 短参照究竟是 83.3333 还是 100.0？若是 83.3333，12 行 32K 的"全员满分"就不成立，F2 结论要改。
- **Q2（F9）**：09-09 非几何屏到底跑了 26 还是 27 个方法目录？（`GROUND_README.md` 38 项 − 09-07/08 项 − 未执行/假设项。）
- **Q3**：GapCapped（c=1.2365e-5）与 MrProBM 的静态表在 `ground_truth_tables.json` 里诊断量完全相同（`ΣT=1.386294`、`maxhole=1.44759@39`、`Σm` 28 vs 32 不同），但 T1 JSON 分数差 7.22/8.68pp。**帽位与帽值到底是哪几个槽**，本地无 `ROPE_GAP_CAPPED_CANDIDATE_20260908.json` 的完整表；需补。
- **Q4（F11）**：`evidence_distances_20260910.json` 本地缺失 ⇒ 89K/96K/106K/117K/127K 异距诊断在 K2 中**不可用**，除非重新取回。
- **Q5**：`E5_layer27` 因 selector 归一化下溢被作废（"层 27 是已测候选，不是有效的修复后亚军排名"，审计 :127）⇒ 层粒度的 K2 特征若含层号，需排除 27。
- **Q6**：MrPro 的 `Σm` 从部署表反演 = 29.333333269（非精确 29.3333…），fp32 落地误差量级 ~1e-7。K2 若把 `Σm` 或 `Σν` 当特征，需先定义是**公式值**还是**部署值**。
- **Q7**：`E1_s28_reverse_matched` / `E1_s29_plus_matched` 的**精确构造式无记录**，只有部署张量。作为"匹配幅度镜像控制"其匹配准则未能本地复核。
- **Q8**：`E2_tail_more` 只跑了 12 行，36 行全面板 pending；`E10_dual_frequency` 36 行 pending。K2 的 far 端若依赖它们，样本量不足。

---

## 8. 死路登记（**绝不能再试**，含失败原因；与 K2 的 F 约束直接相关）

| 机制 | 失败事实（出处） | 失败原因（机制解释） |
|---|---|---|
| **改平台水平**（E2_tail_more：槽 40+ 再 ÷4.93） | 12 行 128K 崩至 54.7222（vs 12 行基线 64.4444）；`ground_truth_tables.json`（`m40=1.15572`）；`digest_nongeo-code.md:171` | 违反 **I2**（`j≥40` 必须精确 `m=1`）。÷S 平台 = `p→p/S` 的**精确重参数化**，改水平即破坏 |
| **全表 PI / 端点整体挪**（MrUni 式） | 32K 崩至 64.5833（−22.64pp）；`ground_truth_tables.json:methods.MrUni`；`digest_nongeo-code.md:171` | 均匀 PI 保跨度但破坏 I2 的水平；"与端点无关的自由度根本不存在" |
| **压缩高频带（快槽）换取低频**（HighGapToLong） | 36 行 70.1389/67.3611（−17.08/−10.76pp，0W7L）；`m23=−0.1557`，`ΣT=1.602162` → **违反 I1** | I1（`j≤23` 恒等）是 bank 不可压；抽快带预算七行全负、零行改善 |
| **压平洞/新造洞 ≥1.46**（E1_pair28_29） | 36 行 87.2222/73.9583（−4.17pp，0W1L）；`maxhole=1.46077@28` | 逐槽收益**不可加**：`s28_less`(+5.21pp) + `s29_more`(−0.21pp) 的组合产出**负**和。`INTEGRATION_20260910.md:74`："每次编辑后重算"是唯一正确解释器 |
| **平滑度当选择子**（Smooth_MrBudget，固定 MrPro 预算最小化粗糙度） | 36 行 87.2222/**68.3333**（−9.79pp，3W5L）；NLL Δ +0.000975/+0.001925/+0.003423 @8K/16K/32K；`digest_failure-audits-1.md:19` | 更平滑的场可以**损害证据竞争**；粗糙度是**承载信息**的，不是噪声 |
| **慢带单槽置零**（E8_zero51） | 12 行 128K 50.5556（−13.889pp）；失败分型 = **终止/格式型** | 冻结态 selector 高分**不**预测全模型任务改善；置零后保留内容通道 ≠ 位置通道可删 |
| **局部线性化 + 数值匹配投影**（E7_local_projection） | 36 行 90.0/68.6111（+2.78/−9.51pp）；审计 :27-40 | (i) 局部导数**确实**预测理想干预；(ii) **BF16 落地**不实现该干预（161× NMSE，且是平方范数比）；(iii) 下游失败。三问必须分开；**不能**用"数值误差"单因解释下游失败，也不能因此废弃全部梯度/敏感度工具 |
| **BM 表本身在 128K** | 36 行 91.6667/**70.8333**（−7.29pp，6W4L，`status=NO_LONG_GAIN`）；`ROPE_BM_TRANSFER_RESULT_20260908.json` | BM 在长端是**净负资产**；作混合成分先验更差（审计 :92） |
| **BM @ gain 1** | 36 行 89.5833/**58.8194**（长端再 −12.01pp）；7W13L | gain 是**独立幅度变量**（logits ×gain²），不改频率/角跨度 ⇒ "gain 恢复相位弧"被否决 |
| **把"平滑度/覆盖率/有效秩/能量/Σcos 首根/碰撞能/Gram"当 F 分项或选择子** | 红线（任务书）；且 `digest_constructive.md:114` 记录 band-polytope 与 `D_j` 作为硬失效边界均被 veto | 静态几何代理量在本数据上与结果**不一致**（s28 在 5 个测试距离里有 3 个**降低**了平方弦距却赢，审计 :60-63） |
| **"training-free + static + nongeometric frequencies 无前例"** | 审计 :115 | MrRoPE-Pro 自身就用 progressive radix schedule（累积中带指数非线性、非单一几何级数）；YaRN 也是分段 schedule ⇒ 新颖性边界过宽 |
| **Σm 当守恒量 / "waterbed 已证明"** | 红线；§4 表：`Σm∈[25.21, 34.18]`，且 32.000000 对应 64.58–100.0 四种结果 | 守恒只在**点名坐标**下成立（`ΣΔm=ln4` 或 `Σ(ν)` 恒等）；Σm 不是不变量 |

---

## 9. 覆盖度

**已读（本地，完整）**：
- `analysis/unify_20260910/`：`INTEGRATION_20260910.md`、`NEXT_DERIVATION_KKT_PROBLEM.md`、`STARTING_POINT_YARN_VS_MRPRO.md`、`tables/GROUND_README.md`、`tables/ground_truth_tables.json`（全 38 方法 + 10 mismatch，逐字段 dump）、`tables/rebuild_ground_truth_tables.py`（头部 + 关键函数签名）、`digests/digest_panel-results.md`、`digests/digest_nongeo-code.md`、`digests_codex/digest_constructive.md`、`digests_codex/digest_failure-audits-1.md`、`mine/panel_extract.json`、`mine/A4_sol06_10.md`、`mine/A5_sol11_15.md`、`mine/A7_source_inputs.md`。
- `docs/research/`：`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md`、`ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`、`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md`、`PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md`、`PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md`、`NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`（630 行，读 1-120/230-450/520-630）、`NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`、`ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md`（部分）、`ROPE_CARRIER_REMOVAL_PILOT_20260907.md`（75-110）、`ROPE_OLMO_BM_RESULT_20260908.md`。
- `docs/research/*.json`（T1）：`ROPE_BM_TRANSFER_RESULT_20260908.json`、`ROPE_GAP_CAPPED_RESULT_20260908.json`、`ROPE_QWEN7_BM_RESULT_20260908.json`、`ROPE_OLMO_BM_RESULT_20260908.json`、`ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`、`ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json`。

**未读 / 跳过**：
- `analysis/unify_20260910/raw*`（~2.9MB transcript 快照）——仅按需 grep，未通读。
- `digests_codex/` 其余 5 份；`digests/` 其余文件。
- `docs/research/rope_allocation_20260910/`（codex 原报告 28 份）——未展开。
- `paper-2027/` 下 `.tex` 与 supplement——未动（用户纪律：Claude 只审稿不动 .tex）。
- `outputs/`、`.tmp_sync/`、`internal/`、`rebuttal/`——未扫描。

**本地缺失（重要）**：`results/nongeometric_screen_20260909/`（部署镜像，含 `reference_tables.json` 与 26 个 `<method>/{contract.json,summary.json,ruler.jsonl}`）；`planned_controls/evidence_distances_20260910.json`；`ROPE_GAP_CAPPED_CANDIDATE_20260908.json`；`/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/`（另一 checkout）。**未访问任何网络、未访问 ~/.codex、未写入除本文件外的任何路径。**

---

## 10. 给 K2 的拟合集建议（我的整理结论，非对下游的命令）

1. **可直接入拟合的 22 个 36 行点**（§1 表 1–16 + §3.4 若补 YaRN）：自变量用 `(Δ_j 向量或其低维投影, gain)`，因变量用 `(L_near, L_far) = (32K, 128K)`。
2. **必须带 `rows` 指示的 2 个点**：E2_tail_more、E8_zero51（12 行）。
3. **必须作为多目标分量而非单点的 1 个点**：FullLagP2（32K 72.9167 是全场第二差、128K 81.6667 是全场第二好 ⇒ 它是 K2 里 **Pareto 张力最大**的样本，也是"两力权衡"最干净的一个数据点）。
4. **禁入的静态几何量**：Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量、`Σm`。
5. **建议保留为自由参数的两条硬约束**（它们是**归纳**而非定理，但当前无一反例）：I1（`j≤23` 恒等）、I2（`j≥40` 精确 `m=1`）；以及 `Σ_{g=23}^{39}Δgap = ln4`（这个是**恒等式**，端点固定时自动成立）。
6. **最强的单点反例**：`E1_pair28_29`（逐槽不可加）与 `MrUni`（内部重分配而非压缩）——任何把 F 写成"每槽独立代价求和"的形式，必须能解释这两点，否则形式错误。

---

*本文件为只读挖掘的产出，所有数字均带出处；T2 档数字受 §0.2 缺口限制，已在每格标注证据等级。*
