# ICLR 2027 投稿前行动计划（只读分析产物）

- **日期：** 2026-08-22
- **性质：** 计划报告。本文件不修改任何论文源文件、图、owner 或配置；
  所有列出的动作在执行前需按第 8 节的授权边界逐项确认。
- **证据基础：** 全量手稿审读（`ICLR2027_FULL_MANUSCRIPT_READONLY_REVIEW_20260822.md`，
  含图表视觉检查附录）、全部 canonical owner 复核、NeurIPS 官方审稿记录、
  前沿线（attention-aware-retrofit）全部五份报告、provenance manifest、
  submission checklist、HANDOFF。
- **时间基准：** 今天 2026-08-22；摘要截稿约 2026-09-18 AoE（约 4 周），
  全文截稿约 2026-09-25 AoE（约 5 周）。以官网 live 核对为准（见 P0-2）。

---

## 1. 现状快照（计划起点）

| 维度 | 状态 |
| --- | --- |
| 科学内容 | 无已知 claim-ceiling 违规；理论无缺陷；数字与 owner 一致到舍入精度 |
| 构建 | main.pdf 30 页、66 条参考文献、正文 9 页用满；上次编译 08-21 |
| 已知缺陷 | 2 个图相关（P2）：Fig 1(c) 标签遮挡、App. C "+1.1%" 舍入伪差 |
| 最高杠杆写作项 | 叙事压缩未完成：intro ¶2–4 残留审计腔（P1） |
| 未决实验 | RULER-13 确认矩阵被平台中断；2Wiki 一项 physical-budget 不变量待复查（P3/P4，均需授权） |
| 投稿有效性 | checklist 尚有未勾项；12 张未使用图 PDF 含 internal-only oracle 图，需确认不入 supplement（P0/P5） |

---

## 2. P0 — 投稿有效性门槛（零内容改动，最先做）

这些动作不改变科学内容，但任何一项失败都直接取消投稿资格。

| # | 动作 | 具体内容 | 通过标准 |
| --- | --- | --- | --- |
| P0-1 | 完整构建健康检查 | 仓库根目录跑 `(cd paper-2027 && ./compile.sh)` 与 `conda run --no-capture-output -n aidemo python scripts/package_supplement.py --profile iclr2027` | 两命令退出 0；记录精确 receipt 进 HANDOFF |
| P0-2 | live 政策与截止日期核对 | 在 ICLR 2027 官网/OpenReview 核对摘要与全文截稿时刻（AoE）、页限、模板版本、AI-use 政策是否有更新 | 与 `SUBMISSION_CHECKLIST.md` 假设一致；不一致则先修订 checklist |
| P0-3 | 匿名性审计 | 确认 PDF、supplement、代码附件、PDF 元数据中无作者身份/路径/checkpoint 痕迹；确认内部 handoff 类文件不在包内 | 打包清单 allowlist 逐项核对 |
| P0-4 | supplement 泄漏检查 | 核对打包器输出不含 12 张未使用图，特别是 internal-only 的 `fig_lerope_profile_oracle.pdf`（LeRoPE oracle 证伪属内部结果）与 kappa raw receipts | allowlist 外零文件 |
| P0-5 | 作者逐数复核 | 作者对照正文/附录每个对外数字与其 owner（第 7 节给出路由表） | 作者签字确认；agent 只准备对照表，不代替签字 |
| P0-6 | OpenReview 一致性 | 提交时标题/摘要与 PDF 逐字一致；NeurIPS 2026 若已接收，自引改第三人称且不违反 dual-submission 规则 | 提交前最后一次核对 |

## 3. P1 — 叙事压缩（分数上限的最大杠杆，纯写作）

**目标读者契约：** 忙碌审稿人前两页读完应能复述单一发现，而不是面对八个独立限定的主张挑最弱一环。

**建议的核心句（供作者采纳或改写，不是既定文本）：**

> RoPE 的频率表是一组有限谱基：静态几何决定什么是可辨识的，
> 训练共适应决定模型实际使用什么。因此 interior allocation 是一个真实、
> 可分离识别的训练期设计轴——其收益边界（不叠加于 target-aware range
> transport、不能经冻结移植存活）由受控实验明确划定。

**具体编辑点（全部为替换、非堆叠；页限已满）：**

1. **Intro ¶2–4 去审计腔。** 现在这三段按"证据分层"组织，读起来像内部
   audit 的外化。建议按发现顺序重排：几何事实（r_2=2.00、46 个名义维度）
   → 共适应诊断 → 因果识别实验 → 边界定理。每段以结论开头。
2. **Abstract 保持现状骨架。** 以 r_2=2.00 开头的当前版本已符合单发现
   契约，只需通读确认与重排后的 intro 无缝衔接。
3. **target-matched 反转的定位升级。** 从"防御性一句话"升级为识别主张的
   组成部分：正是"固定支持下 3/3 赢、target-matched 下 3/3 输"把结论钉在
   正确的主张上。放在 Discussion 现有位置即可，改措辞方向不改位置。
4. **禁止事项（执行时逐条自查）：**
   - 不引入任何新数字；所有保留数字从 owner 重算核对（不从旧稿复制）；
   - 不恢复 surrogate-first 叙事（canonical 报告 RISKY_REGIONS 第 1 条）；
   - 不合并 exact-range、mature adaptation、50M retrofit 的因果层级
     （RISKY_REGIONS 第 3 条）;
   - 锁定命名表（Geo/Native/FMRoPE/anchored EVQ-Cosh/YaRN-style）不动；
   - AI-use statement 覆盖范围不改（需重新确认才能动）。

**验证：** 重编译后页数 ≤9；图表引用完整；交叉核对三个 headline 数字
（−0.281/−0.176/−0.146、31.1%、16K tail −0.2179）与 owner 一致。

---

## 4. P2 — 两个已定位缺陷的修复（小改动，需重新生成图 + 编译）

### P2-1 Fig 1(c)：EVQ-Cosh ×2 柱标签被 callout 遮挡

- **现象：** "21.5" 标签被红色 "8B causal deletion" callout 完全遮住，
  六根柱中唯一不可见标签，恰承载 caption 核心主张。
- **根因：** `figs/make_fig_evidence_overview.py` 第 128–139 行；标签画在
  y≈22.1，callout 位于 axes 分数坐标 (0.98, 0.78)，正好覆盖。
- **修复方案：** 将 callout 移至面板左上空白区，或该柱标签加垂直偏移加细
  引线。二选一，优先前者（不动数据标签逻辑）。
- **验证：** 重渲染 PNG 视觉复核六根柱标签全部可见；图 PDF hash 更新记录；
  caption 无需改动。

### P2-2 App. C："+1.1%" 应为 "+0.9%"

- **裁定（已完成）：** 底层数据相同；curated JSON 未舍入均值
  35.77/35.44 = +0.93%，真实值 +0.9%；正文的 +1.1% 来自舍入表格值
  35.8/35.4 相除。**图正确，附录句子错。**
- **修复方案：** App. C 该句一处单词级改动。
- **验证：** grep 全仓库无其他 "+1.1" 残留；与 P2-1 合入同一次编译。

### P2-3（可选，随 P2-1 顺带）

Fig 1 面板 (a)/(b) 注释字号 5.8–6.2pt 接近印刷下限。若做 P2-1，顺带把
注释字号提至上限并做打印比例目检；不单独为此重新生成。

---

## 5. P3 — RULER-13 确认矩阵（唯一登记的 GPU 继续项，需明确授权）

- **审稿人问题映射：** `AC.2`/`RzWsa.3`（更强 benchmark）；同时是前沿线
  结果能否晋升为稿件证据的门。
- **现状（owner 记录）：** 三臂 × 两长度 × 13 任务注册；core-4 cell 身份
  锁定；word 资产已恢复验 hash；中断前 ≥18/26 cells 已生成但 manifest
  未写入，post-shutdown 状态未验证。
- **重启协议（owner §Interrupted breadth extension 原文规定）：**
  1. 先检查持久卷上的既有输出与 runtime receipts；
  2. 若无完整已验证 26-cell manifest，则生成全新 versioned 输出，
     不把残缺目录当作完成；
  3. core-4 预测仅在四个 cell hash 与已完成 selection manifest 完全一致时
     复用，否则重算；
  4. 九个 unseen 任务 confirmation-only，方法不得因此改变；
  5. 九任务确认 macro 与完整 13 任务 macro 分开报告，8K/16K 分开。
- **付费前置条件（AGENTS.md 第 4 节全项）：** 冻结代码/config hash、数据与
  checkpoint 身份、realized frequency tensor、optimizer/budget、输出 schema、
  磁盘空间与关机预案；CPU/no-GPU preflight 通过后才启动。
- **预算：** 启动前按 core-4 单臂实测吞吐折算三臂×两长度×九任务墙钟并报
  作者批准；分 cell 落盘 + 每 cell 即时 hash 收据，保证再次中断可恢复。
- **结果处理（两种结局预先声明）：** 方法冻结不变；若 Native 在 unseen
  任务广泛反超，一句平铺陈述，不加解释性包装；任何结局先更新 owner，
  再谈稿件整合。
- **不授权的替代路径：** 本项保持 frontier 状态，论文不受影响（正文没有
  任何句子依赖它）；Discussion 维持现有前瞻表述边界即可。

## 6. P4 — 2Wiki physical-budget 不变量复查（轻量，随 P3 同会话）

- **背景：** post-run 审计发现未截断 fast path 在加 chat-template overhead
  前检查了 raw prompt tokens。
- **动作：** 对 200 行 raw 逐行检查 `input_tokens + 32 <= nominal_length`；
  违规 cell 用修正后检查重跑。
- **影响面：** 仅 2Wiki held-out 结果；不影响 RULER 与 runtime parity。
- **地位：** 前沿线任何稿件晋升的硬前置；即使不晋升也应在 owner 补记
  复查 receipt，消除悬挂的限制条款。

---

## 7. 数字→owner 路由表（P0-5 作者复核用；未来文字修改的唯一数字来源）

| 正文位置 | 主张/数字 | Canonical owner |
| --- | --- | --- |
| Abstract/§4/App. E | −0.28073/−0.17599/−0.14571；+0.02619；target-matched +0.06032/+0.22720/+0.45959 | `research/EXACT_RANGE_151M_3SEED_RESULT_20260820.{md,json}`（禁用 07-24 旧聚合 −0.3159 系列） |
| §4/App. F | MLA 31.1%（16K 三种子聚合 95.59 vs 138.81）；EVQ+YaRN(s=4) 71.13 | `data/curated/table18_mla_3seed_aggregate.json` |
| App. D | M4：rule 7/12 p=0.125；1.25× 10/12 p=0.0273 未校正；Cosh vs Exp +0.00074 p=0.836 | `theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` |
| §3 定理 | r_2=2K/(1+(K−1)c̄)；低频坍缩；移植障碍（含 2π alias） | `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` + obstruction 07-26 |
| §4/App. G | 1.485B：+7.51%/+3.88%/−4.28%/−12.64%；tail −0.2179 [−0.2296,−0.2063] | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| §4/App. H | QK-phase RULER-13 42.44/31.63/5.03 vs 72.19/2.02/0.38；2Wiki 21.48%@8K | `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` |
| finite-τ 表述 | fallible operating prior；离散网格非 basin | `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md` + M4 边界臂 |
| LeRoPE 段落 | concurrent work；Fixed-LeRoPE 63.6%；oracle 内部 only | `LEROPE_CONCURRENT_WORK_NOTE_20260728.md` + oracle audit 08-20 |

（theory_results 前缀 = `rebuttal/rebuttal_0723/theory_results/`）

---

## 8. 执行顺序与授权边界

```
本周（验证/分析类，无需授权）：
  P0-1 构建健康检查 → P0-2 live 政策核对 → P0-3/P0-4 匿名与泄漏审计
  → （并行）P1 叙事压缩草稿（写成 diff 建议，不动源文件，交作者过目）
需要作者点头的小改动：
  P2-1 图脚本 callout 移位 + 重新生成 + 编译
  P2-2 附录一个数字改正（合入同一次编译，共用一份 receipt）
需要明确 GPU 授权：
  P4（轻量）→ P3（重量）；顺序不可倒置（2Wiki 复查是 P3 解释的前置）
截止前最后一周：
  P0-5 逐数复核签字 → P0-6 OpenReview 一致性 → 最终编译+打包+视觉终检
全程红线：
  paper/ 零接触；internal/、results/、audit_v3/、audit_v4/、.codex/、
  .claude/ 零接触；不做 git stage/commit/push 除非明确要求；每次变更后
  HANDOFF 记录 changed files、receipt、immutable-paper 状态、git 状态。
```

---

## 9. 本计划的限制

- 截止日期以 08-21 handoff 记录为准，P0-2 完成前不应视为最终；
- P3 预算数字需启动前用 core-4 实测吞吐折算，本报告不给估计值以免未经
  测量就锚定预期；
- P1 的核心句与编辑点是 agent 分析建议，采纳与否属作者的科学判断；
- 本报告自身不构成任何实验结果、证据或协议身份的变更。


