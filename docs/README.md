# Docs — 研究文档中心

本目录同时包含当前维护文档和明确隔离的历史材料。是否能够支持 reviewer-facing claim，不由文件是否位于 `docs/` 决定，而由 `overview/RESULT_PROVENANCE_MANIFEST.md` 决定。

---

## 目录结构

```
docs/
├── overview/       当前 claims、provenance、复现指南与审计控制面
├── exp/            实验报告 (YYYY-MM-DD_slug.md 格式)
├── theory/         理论推导与数值验证
├── tau_algor/      tau/scaling 的历史推导与诊断
├── archive/        明确退役的文档
└── superpowers/    已执行计划的历史记录，不是当前入口
```

---

## 阅读顺序

### 快速入门 (10 min)

1. **`ai-handoff.md`** → 当前工作树、已知问题和继续位置
2. **`REPO_MAP.md`** → 目录职责与 source-of-truth
3. **`overview/README.md`** → 当前 overview 权威顺序
4. **`overview/PAPER_CLAIMS_MAP.md`** → 论文↔实验↔脚本↔结果导航
5. **`overview/RESULT_PROVENANCE_MANIFEST.md`** → reviewer-safe artifact 与哈希

### 深入了解 (30 min)

6. **`overview/METHODOLOGY.md`** → EVQ-Cosh 方法论、评估协议（若与 audit stack 冲突则降级）
7. **`overview/TERMS_AND_PROTOCOLS.md`** → 统一术语表和命名规范
8. **`exp/README.md`** → 实验报告索引
9. **`theory/THEORY_MATH_VALIDATION.md`** → 历史理论数值验证；rebuttal 数学以 `rebuttal/THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` 为准

### 复现实验

10. **`overview/DATA_PREPARATION.md`** → 数据来源
11. **`overview/REPRODUCE.md`** → 核心结果复现路径

---

## 关键文件速查

| 需求 | 文件 |
|------|------|
| 从 Figure/Table 找到生成脚本 | `overview/PAPER_CLAIMS_MAP.md` |
| 判断一个结果是否可用于 reviewer/rebuttal | `overview/RESULT_PROVENANCE_MANIFEST.md` |
| 复现论文结果 | `overview/REPRODUCE.md` |
| 理解数据来源 | `overview/DATA_PREPARATION.md` |
| 查看特定实验结果 | `exp/README.md` → 找到对应报告 |
| 理解理论推导 | `theory/THEORY_MATH_VALIDATION.md` |
| 统一术语 | `overview/TERMS_AND_PROTOCOLS.md` |

---

## 文档维护规则

- 实验报告使用 `YYYY-MM-DD_slug.md` 命名，放入 `exp/`
- 理论文档放入 `theory/`
- overview/ 的 README、provenance manifest、claims map 和 reproduce 文档是当前维护入口；其余 audit 文档按索引使用
- 历史或已完成计划进入 `archive/` 或保持明确的 archived/superseded 状态
- 缺 raw artifact 时只能写 report-backed / missing-artifact，不能用叙述文档升级证据
