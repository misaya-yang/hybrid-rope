# 2026年4月工作区

## 文件夹结构

```
2026_04_run/
├── README.md                 ← 本文件
├── docs/                     ← 文档与分析
│   ├── narrative_strategy.md ← 论文行文策略分析（核心）
│   ├── theory_problems.tex   ← 6个理论开放问题（可发导师/师兄）
│   ├── theory_problems.pdf   ← 编译后 PDF
│   ├── theory_analysis_report_20260331.md ← 理论分析报告 v2
│   ├── KEY_FILE_MAP.md       ← 全项目关键文件索引
│   ├── tau_strategy.md       ← τ 优化策略
│   ├── sprint_plan.md        ← 3月 sprint plan（已完成）
│   ├── p0_runbook.md         ← P0 任务 runbook
│   └── p1_p2_p3.md           ← P1-P3 任务列表
├── figs/                     ← 图表
│   └── fig_evq_method_v3.*   ← 方法图（最新版）
├── scripts/                  ← 分析脚本
└── results/                  ← 实验结果
```

## 当前状态

- **论文行文**：已完成策略分析（narrative_strategy.md），核心是从 "optimal formula" 转向 "principled framework" 叙事
- **理论问题**：6个开放问题已审核修正，theory_problems.pdf 可发导师
- **关键发现**：
  - λ ≈ 1.17 ± 0.13 (CV=11.1%)，近似普适
  - 连续泛函 J[ρ] 在小τ时确实减少碰撞，论文理论框架正确
  - Q6 小τ展开公式原有错误，已修正
- **方法图**：v3 版本已生成，待插入论文

## 待办优先级

1. P0：论文 narrative 重构（principle > formula）
2. P0：τ* optimality basin 图提到正文
3. P1：λ cross-validation table 加 appendix
4. P2：EVQ + LongRoPE2 实验（师兄建议）
