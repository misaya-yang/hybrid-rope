# 结果真值表（VALID / INVALID）

更新时间：2026-02-23 00:20（CST）

目的：给论文写作和后续 AI 分析提供唯一“可引用结果边界”。任何不在 `VALID` 区域的数据，默认禁止进入论文主表。

## A. VALID（可进入论文主线）

### A1. 从零训练 Scaling 主线（已稳定）

来源：`knowledge_base/01_已完成实验核心数据.md`

| 实验 ID | 关键结论 | 可用数值 |
|---|---|---|
| `EXP_50M_3SEED` | Hybrid 稳定优于 Geo | 16K PPL: Geo `18.2±0.8` vs Hybrid `17.3±0.4` |
| `EXP_100M_FINAL` | 跨规模改进成立 | 16K PPL: Geo `10.88` vs Hybrid `9.41`（-13.5%） |
| `EXP_350M_FINAL` | 改进继续存在 | 16K PPL: Geo `14.65` vs Hybrid `12.65`（-13.7%） |
| `EXP_50M_YARN` | 与 YaRN 对照优势明确 | 16K PPL: Geo `17.97`, YaRN `39.48`, Hybrid `16.86` |

### A2. 理论与纯计算验证（2026-02-22 计划）

来源：`results/theory_2026-02-22/mentor_plan_execution_summary.json`

| 任务 | 结论 | 关键数值 |
|---|---|---|
| Task 1（Ediag exact vs affine） | 理论近似成立 | `R^2(mid)`：0.9942（b=1e4），0.9954（b=5e5） |
| Task 2（Phase transition L/b 扫描） | 现象与 `L/b` 强相关 | `L/b=100,1000` 出现 crossing；`L/b=1.6,10` 无 crossing |

### A3. 8B 公平套件（部分完成，已落盘）

来源：服务器 `/root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/*/summary.json`

| 方法 | 状态 | 关键训练结果 |
|---|---|---|
| baseline | 已完成 | `train_loss=0.34198`, `eval_loss=3.543e-06` |
| pi | 已完成 | `train_loss=0.08251`, `eval_loss=5.956e-07` |
| yarn | 已完成 | `train_loss=0.08459`, `eval_loss=1.172e-06` |
| anchored_hybrid | 进行中 | 截止 00:15 已到 step 30（loss 6.42 -> 2.64 -> 0.352） |

注意：该套件是“先训练四个，再统一评测”，当前 NIAH/LongBench 还未产出最终对比。

## B. PENDING（待定，不可写最终结论）

| 项目 | 当前状态 | 进入论文前的必要条件 |
|---|---|---|
| `EXP_8B_FAIR_LORA` 最终结论 | 仅完成前三模型训练，第四模型进行中 | 四模型训练+统一评测全部完成，并生成对比汇总 |
| NIAH / LongBench（overnight_8h） | 尚未进入 Gate2/Gate3 汇总阶段 | 产出统一热力图/表格并核验协议一致性 |

## C. INVALID（当前判定不可用，禁止主表引用）

### C1. Task3 / Task4（你指出的问题）

来源：`results/theory_2026-02-22/mentor_plan_execution_summary.json` 与  
`sigmoid_rope_experiments/phase4_geo100k_124m_2026-02-22_rerun1/data/phase4_corrected_summary.json`

判定依据：
1. `PPL≈1.x` 显著偏离历史尺度（不可信）。  
2. 该 run 元信息显示：`dataset_name = Synthetic-Passkey`，`tokenizer = byte`。  
3. 与历史 A100 协议（TinyStories + HF tokenizer）不一致，不能横向比较。

结论：Task3/Task4 当前版本全部标记为 `INVALID`，仅可用于“错误复盘”，不可用于论文正面结论。

### C2. 旧版 8B 非公平对照

来源：`docs/EXPERIMENT_REGISTRY.md`（`BAD_8B_LORA_OLD`）

结论：涉及 `rope_scaling` 与 monkey patch 混用，协议不公平，禁止主文引用。

## D. 一句话执行规则

只要结果不是：
1) 协议一致，  
2) 元信息可追溯，  
3) 数值尺度与历史一致，  
就一律先归入 `PENDING/INVALID`，不进入论文主表。

