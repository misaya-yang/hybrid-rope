# ds_workspace 主索引（2026-09-11 整理版）

> 全项目文档地图。按"想做什么"选入口；文件全文不在这里，这里只负责指路。
> 维护规则：新判决/读出 → `recon_20260910/verdicts/`；新预注册 → `recon_20260910/prereg_protocols/`；
> 理论/机制分析 → `recon_20260910/theory/`；加入后在本页对应分类补一行。

---

## 1. 入口文档（按需选一层，不必全读）

| 层 | 文件 | 回答什么 |
|---|---|---|
| **10 分钟总览** | [EXPERIMENT_THEORY_MASTER_20260911.md](EXPERIMENT_THEORY_MASTER_20260911.md) | 全项目每个实验/理论 ≈100 字一句话版 + 一页总账（立住/立不住/欠着） |
| **证据核查层** | [recon_20260910/YARN_MRROPE_RESEARCH_EVIDENCE.md](recon_20260910/YARN_MRROPE_RESEARCH_EVIDENCE.md) | YaRN→MrRoPE-Pro 问题的完整证据链：算子对照、论文核查、可信度修正、复算命令 |
| **战役判决层** | [recon_20260910/index.md](recon_20260910/index.md) + [recon_20260910/MASTER_SUMMARY_20260911.md](recon_20260910/MASTER_SUMMARY_20260911.md) | 9/10–11 战役的 23 条结论清单与效应层级（⚠ 审计前快照，数值以审计与 EVIDENCE 为准） |
| **当前接管状态** | [recon_20260910/RESEARCH_OWNER_20260911.md](recon_20260910/RESEARCH_OWNER_20260911.md) | 接管后的决策、服务器操作纪律、live 状态 |
| **审计** | [recon_20260910/audit/pro_decision_20260911/REPORT.md](recon_20260910/audit/pro_decision_20260911/REPORT.md) | 六个已确认缺陷 + 900/700 行重打分复算（check_results.json） |
| **流程教训** | [LESSONS.md](LESSONS.md) · [README.md](README.md) | 静默失败案例库与工作区说明 |

## 2. 目录地图

```
ds_workspace/
├── index.md                        ← 本文件
├── EXPERIMENT_THEORY_MASTER_20260911.md   （10 分钟总览）
├── LESSONS.md / README.md
├── codex_failures_20260910/        （失败案例存档）
├── receipts/                       （回执）
└── recon_20260910/                 ← 9/10–11 战役工作区
    ├── index.md / MASTER_SUMMARY / RESEARCH_OWNER / YARN_MRROPE_RESEARCH_EVIDENCE
    ├── MIGRATION_20260911.md       ← 2026-09-11 整理的新旧路径对照表
    ├── theory/            （12 份：理论与机制框架）
    ├── prereg_protocols/  （12 份：预注册与执行协议）
    ├── verdicts/          （28 份：结果判决与读出）
    ├── audit/             （代码审计：REPORT + check_results + 快照）
    ├── code/              （纯 numpy 复算脚本：coverage_theory 等）
    ├── work/              （本地同步的原始 jsonl，按臂分目录）
    ├── _reports/          （7 份子代理深报告）
    ├── _archive_20260911/ （23 份被后续结果覆盖的早期文档）
    ├── design/ · recon/
    └── （服务器原始数据：ssh -p 27741 [REDACTED_EMAIL]
        /root/autodl-tmp/phase1_20260910/ 与 /root/autodl-tmp/rope_decision_20260911/）
```

## 3. recon_20260910 分类清单

### theory/ — 理论与机制框架
| 文件 | 一句话 |
|---|---|
| `THE_ANSWER_20260911.md` | ★ 全项目核心经验律：表间差大小与符号由"救援量"(native−BM) 决定，跨 3 模型 4 长度 |
| `COVERAGE_CEILING_THEORY_20260911.md` | 覆盖/天花板理论：两公理两常数解释整场战役，n_int 排序全家族 |
| `FOUR_CORNERS_20260911.md` | Lε forcing 框架：BM=常数 forcing、MrRoPE=慢端点源，四角单纯形 |
| `MRROPE_RECONCILIATION_20260911.md` | MrRoPE 论文对账：YaRN→Pro 只改中段装填（⚠ 需按官方/论文 YaRN 之分限定，见 EVIDENCE §1.2） |
| `SYNTHESIS_20260911.md` | 为什么 gain 的解析推导管用、频率分配全不管用（RATIO 定量分界） |
| `NLL 反向相关` → verdicts/NLL_VS_TASK | （归判决） |
| `GPT6PRO_MECHANISM_BRIEF_20260911.md` | 机制缺口简报：M1–M4 失败机制、E1–E5 判别实验设计、Q1–Q8 |
| `NO_STATIC_FUNCTIONAL_20260911.md` | 12 个静态泛函全部出局（三层受控对） |
| `CONSTRAINT_IS_SLACK_20260911.md` | 带内重分配在 NLL 上免费（⚠ 任务代价见 NLL_VS_TASK 反向） |
| `WHY_THE_FISHER_ROUTE_DIED_20260911.md` | Fisher 二次代价的精确死因（native 不是最优 ⟹ 距离≠代价） |
| `COST_IS_NOT_QUADRATIC_20260911.md` | 代价曲线非二次的实测 |
| `RELEASE_AXIS_20260911.md` | 释放 m=1 平台轴：KNIFE 悬案解决 + rel_* 三臂判决 |
| `PLATEAU_20260911.md` | 连续仪器上的平台：五成员统计打平、两方向相加 |

### prereg_protocols/ — 预注册与协议（全部"写在读数之前"）
`COVERAGE_PREREG`（P1–P8+判分）· `HOLDOUT180_PREREG` · `LONGBRIDGE_PREREG` · `NATURAL_PREREG` ·
`QWEN3_SURVIVOR_PREREG` · `QWEN4X_POWER_PREREG`（跨模型符号判决，跑中被杀）· `SATPRO_PREREG`（P9–P12）·
`S8_PREREG` · `STEP42_PREREG` · `WALK_PREREG` · `DOSE_PREREG` · `RUNBOOK_OFFICE`（执行手册）

### verdicts/ — 结果判决与读出（按主题）
| 主题 | 文件 |
|---|---|
| **头条与总判决** | `HEADLINE`（平台四成员+选择效应警告）· `PHASE1_CONCLUSION`（★ 不存在通用"超过 MrRoPE"，规则=按救援量选压缩量）· `THE_ANSWER` 在 theory/ |
| **MrRoPE/BM 对比** | `PRO_TABLES_RESULT`（Pro 两张派生表全差于 BM）· `CROSS_MODEL_VERDICT`（Qwen 两仪器判不了）· `THIRD_MODEL`（Qwen7B：负救援量，两表都不如 native）· `QWEN3_SURVIVOR_RESULT`（移植两臂皆败） |
| **held-out / 泛化** | `HOLDOUT_VERDICT`（72 行，冠军归零）· `HOLDOUT180_RESULT` / `HOLDOUT180_VERDICT`（未决+长度交易；⚠ 唯一提示修正见 audit）· `OUT_OF_SAMPLE_ALL_NULL`（三预注册全阴）· `FRESH72_COMPLETED`（★ 独立样本六臂：YaRN≈Pro≈灾难、校准回归、b4wide 条件存活） |
| **形状/阶跃** | `STEP42_RESULT`（选择效应反转）· `S8_RESULT`（8× 全零）· `TRADEOFF_ASYMMETRY`（增益=逐项召回）· `DPATTERN`（churn 恒定、方向由面板决定）· `HISTORY_HAS_CANDIDATES`（历史候选与坏对照） |
| **gain 轴** | `GAIN_AXIS`（+38pp 主效应）· `GSWEEP_RESULT`（★ YaRN 解析 mscale = 实测最优）· `GAIN_TABLE_2x2_FINAL`（gain 让表生效：native 零、BM +38pp）· `DOSE_G2X2_READOUT` · `DOSE_RESULT`（a*(L)≡1，最优按模型不按长度） |
| **走线/自然 QA** | `WALK_RESULT_AND_CONFIRM_PREREG`（PURE TRADE+非线性）· `OUT_OF_SAMPLE_ALL_NULL` §走线内点 |
| **LongBridge 符号对** | `LONGBRIDGE_RESULT`（面板 +11.8pp t=5.85，65% 来自 niah_single_3）· `LONGBRIDGE_HOLDOUT`（★ out-of-sample 翻负，方向不成立）· `SIGNED_CONTROL_RESULT`（CPU 复读确认面板值） |
| **EVQ** | `EVQ_LONGRANGE_VERDICT`（三 τ 全零，零训练替换死亡） |
| **机制/形态学** | `NLL_VS_TASK`（★ NLL 与任务反向律）· `INSTANCE_OUTAGES`（两次实例级中断的取证） |

## 4. 按问题检索

| 想知道… | 去哪 |
|---|---|
| YaRN→Pro 到底改了什么 | EVIDENCE §1（算子对照表）+ theory/MRROPE_RECONCILIATION |
| 哪个结果可信/哪个被推翻 | EVIDENCE §4–5 + audit/REPORT + verdicts/HOLDOUT*、STEP42、LONGBRIDGE_HOLDOUT |
| 为什么 BM 赢 MrRoPE | theory/COVERAGE_CEILING + theory/THE_ANSWER + verdicts/PHASE1_CONCLUSION |
| 现在该跑什么 | EXPERIMENT_THEORY_MASTER §7"欠着" + prereg_protocols/ 里未跑的判据 |
| 某个数的原始数据在哪 | EVIDENCE 附录 A + recon_20260910/work/jsonl/ 或服务器路径 |
| 重算某个结果 | EVIDENCE 附录 B 命令 + recon_20260910/code/ |

## 5. 状态速览（2026-09-11 整理时点）

- **GPU/服务器**：两次实例级中断（verdicts/INSTANCE_OUTAGES）；重启后曾 GPU 不可见；LongBridge faster 腿、qwen4x_power 等中断臂以 verdicts/ 各文档"边界"节为准。
- **仍欠着**：qwen4x_power（预注册已写，语料已建）· P1 scale8x · P5–P9 · E2 注意力归因 · OLMo-1B PPL 型四臂。
- **不要重跑**：EXPERIMENT_THEORY_MASTER §7"立不住"清单。
