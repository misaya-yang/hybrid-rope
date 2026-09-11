# 预注册：OLMo 幸存者移植表在 Qwen2.5-3B 上的 36 行配对筛选（run_qwen3_02）

**提交时刻早于任何一行新结果被读取。** 判定规则 = 现成评估脚本 `ruler_bench.verdict` 原文，不改一字。

## 一、问题

OLMo 上的宽带幸存者（a1_b64 主、wide_b4 次）经 **P5 冻结迁移规则**（无目标调参）移植到
Qwen2.5-3B（θ=1e6，32K 原生，s=4）后，是否仍在 128K 上赢 MrRoPE？

背景事实（已归档，非本轮测量）：
- Qwen2.5-3B 上 MrRoPE **不崩溃**（32K macro 0.8722，128K macro 0.7813）——公平战场。
- 已测过的 MrProBM（sum_m=32.0000）≠ 移植 a1_b64（S_Q=33.470817；max|m diff|=0.1346，
  2026-09-11 数值核验）⟹ **OLMo 赢家从未在 Qwen2.5-3B 上被测过**。
- MrProBM 判定 = NO_LONG_GAIN（Δ32K +4.44pp，Δ128K −7.29pp，6W/4L）。

## 二、面板与基线（§2 面板纪律）

- **冻结 36 行 6 任务开发子集**：12 行 @32768 + 24 行 @131072（6 任务 × 每 cap 2/4 行），
  prompts/评分/解码与归档 MrPro 运行**逐字节相同**（prepared_qwen3_02 的 screen/qualification/
  generation_config 为原文件副本；run identity digest = `bb86e623d7597506…` 与 run_qwen3_01
  核验相等）。
- **基线 = 归档 MrPro（MrRoPE）臂缓存复用**（identity/raw_sha256/table/row_ids 四重校验通过才复用）。
  不重跑、不混面板。
- **不用 13 任务全量 RULER**：§4 规定昂贵 Qwen128K 只评最终固定规则；移植表尚未通过任何
  Qwen 筛选，无资格。36 行是开发筛选，不是确认。

## 三、臂（构造在读结果前冻结）

| 臂 | 构造 | S_OLMo | S_Q | tensor_sha256 | gain |
|---|---|---|---|---|---|
| QwenA1B64T（主） | `m_incr_beta(1.0,n=21,low=11)` → `pro_tables_20260911.transport`（越界即 raise，不裁剪） | 42.0（校验通过） | **33.47081678952053**（= math_checks.json 预注册预测 33.4708167895，git 35063d5，逐位一致） | `f16eae8dadeeeee9a7b55d20033a1fa1f94f83c74bf280e94c32d1121ac33275` | 1.138629436111989（冻结于参照） |
| QwenWideB4T（次） | `m_incr_beta(4.0,n=21,low=11)` → 同一规则 | 46.681784536706814（= holdout180 归档值，校验通过） | **37.91770839642992**（本文件提交前记录） | `25900f9801f3ad60b2ebc56c63492a1c256295b6a98e733002663d45c642def0` | 同上 |

ν = ω_Q·4^(−m_Q)；约定核验：归档 MrPro 表反推 sum_m=29.3333 ✓、Native==ω_Q ✓。
构建脚本 14/14 校验通过后才写盘（BUILD OK）。

## 四、判定（脚本原文规则，逐字沿用）

对每臂 vs 缓存 MrPro，同 36 行配对（row 身份四字段校验）：

- `DEVELOPMENT_WIN`：Δmacro(128K) > 0 **且** Δmacro(32K) ≥ 0
- `TRADEOFF`：Δmacro(128K) > 0，Δmacro(32K) < 0
- `NO_LONG_GAIN`：Δmacro(128K) ≤ 0

同时报告 paired W/L/T、逐任务×cap 表、eos_count、基线极端格。
harness 自带早停：主臂若 DEVELOPMENT_WIN 则队列停止（次臂不跑）——这是预注册行为，不补跑。

## 五、解释分支（现在写死）

1. **主臂 DEVELOPMENT_WIN** → 冻结迁移规则在公平战场通过开发筛选 →
   下一步是**功效化独立确认**（新内容、新 seed、n≈7.84σΔ²/δ²，σΔ² 用本次配对差估计，
   确认跑之前另行预注册）。不构成最终结论。
2. **主臂 TRADEOFF** → 与 MrProBM 同签名（用原生买外推，但方向不同于 MrProBM 的
   +32K/−128K）→ 部署规则必须长度条件化；收缩解空间。
3. **主臂 NO_LONG_GAIN** → 移植幸存者在与 MrProBM 相同的面板上同败 →
   **「OLMo 赢家可跨模型移植」被证伪（限本面板）**；收缩：宽带收益是 OLMo 特异的，
   MrRoPE 在 Qwen 的优势维持；下一步转向长度条件部署或承认三段路线的模型边界。
4. **次臂只作描述性报告**：OLMo held-out 上 a1_b64−wide_b4=−0.42pp（t=−0.16，不可分辨）。
   若两臂在 Qwen 上分道超过配对噪声 ⟹ b 轴在移植后可分辨——记录，但**不得事后提拔更好的臂**。

## 六、事先声明的局限

- n=36 功效不足（先例：MrProBM 6W/4L）；**不显著 ≠ 等价证明**（§P5）。
- 本面板曾用于 MrProBM 判定，但幸存者选择只用了 OLMo 数据 ⟹ 对本臂而言 Qwen 面板是新鲜的。
- 不加行、不混面板、读结果后不再改表。

## 七、GPU 让位记录

2026-09-11 ~11:26 按用户指令终止 OLMo 队列腾卡（评估脚本硬性要求 GPU 独占）：
walk/h22/gain2x2/step42/natural/walkconf 及各 chain（服务器 `phase1_20260910/GPU_YIELD_20260911.md`）。
本筛选两臂 ≈2×15min（先例：2 臂×36 行 = 1766s）。结束后按原命令恢复 OLMo 队列。
