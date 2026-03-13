# EVQ-Cosh 协作中心

## 30 秒概览

**论文核心**: EVQ-Cosh 是 RoPE 频率分配的闭式变分解，τ 是唯一控制参数，geometric RoPE 是 τ→0 退化极限。

**当前状态**: 99-run τ* scaling law 验证完成。750M 模型确认 EVQ+YaRN 100% passkey vs Geo+YaRN 61-65%。LaTeX 初稿完成 (10页正文 + 8页附录)。

**投稿目标**: NeurIPS 2026

---

## 阅读顺序（老师/合作者请按此顺序）

### 第一步: 战略概览 (5 min)
1. **`briefs/advisor_brief.md`** → 论文三锚点 + 最强证据 + 待决策问题

### 第二步: 执行状态 (10 min)
2. **`status/WORKFLOW_AND_PAPER_GAPS.md`** → ⭐ 最重要文件
   - P0/P1/P2/P3 优先级矩阵
   - 理论矩阵 (T1-T8 claims 状态)
   - 实验矩阵 (C1-C6 claims + 单点风险)
   - 缺失图表清单
3. **`status/open_gaps.md`** → 当前缺口快速清单

### 第三步: 下一步计划 (按需)
4. **`plans/README.md`** → 10个实验计划索引

### 快速写作入口
| 需求 | 路径 |
|------|------|
| 论文 LaTeX | `paper/main.tex` |
| 核心叙事线 | `internal/mainstory.md` |
| 所有图表 | `paper/figs/` |
| 论文↔实验完整映射 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| 复现指南 | `docs/overview/REPRODUCE.md` |

---

## 目录结构

```
team/
├── briefs/           → 给老师/合作者的战略摘要
│   ├── advisor_brief.md
│   └── senior_collab_brief.md
├── status/           → 当前执行状态和缺口跟踪
│   ├── WORKFLOW_AND_PAPER_GAPS.md  ⭐ 核心文件
│   ├── open_gaps.md
│   └── downstream_strategy_2026-03-12.md
├── plans/            → 下一步实验计划
│   └── (10 个实验计划文件)
└── archive/          → 历史 handoff (仅供追溯)
```

## 规则

- team/ 中的所有文件应可一遍读完
- 不要放原始实验数据或运行日志
- 新计划按 `plans/README.md` 的三问题模板创建
