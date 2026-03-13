# Docs — 研究文档中心

本目录包含 EVQ-Cosh 项目的所有策略、理论和实验文档。仅保留与当前 NeurIPS 2026 投稿直接相关的内容。

---

## 目录结构

```
docs/
├── overview/       高层概览、方法论、复现指南、追溯地图
├── exp/            实验报告 (YYYY-MM-DD_slug.md 格式)
└── theory/         理论推导与数值验证
```

---

## 阅读顺序

### 快速入门 (10 min)

1. **`overview/PROJECT_OVERVIEW.md`** → 项目全貌和当前状态
2. **`overview/PAPER_CLAIMS_MAP.md`** → ⭐ 论文↔实验↔脚本↔结果的导航中枢

### 深入了解 (30 min)

3. **`overview/METHODOLOGY.md`** → EVQ-Cosh 方法论、评估协议
4. **`overview/TERMS_AND_PROTOCOLS.md`** → 统一术语表和命名规范
5. **`exp/README.md`** → 所有实验报告索引 (按时间排序)
6. **`theory/THEORY_MATH_VALIDATION.md`** → 理论推导的数值验证

### 复现实验

7. **`overview/DATA_PREPARATION.md`** → 四个数据源的获取方式
8. **`overview/REPRODUCE.md`** → 从环境搭建到核心结果复现的完整路径

---

## 关键文件速查

| 需求 | 文件 |
|------|------|
| 从 Figure/Table 找到生成脚本 | `overview/PAPER_CLAIMS_MAP.md` |
| 复现论文结果 | `overview/REPRODUCE.md` |
| 理解数据来源 | `overview/DATA_PREPARATION.md` |
| 查看特定实验结果 | `exp/README.md` → 找到对应报告 |
| 理解理论推导 | `theory/THEORY_MATH_VALIDATION.md` |
| 统一术语 | `overview/TERMS_AND_PROTOCOLS.md` |

---

## 文档维护规则

- 实验报告使用 `YYYY-MM-DD_slug.md` 命名，放入 `exp/`
- 理论文档放入 `theory/`
- overview/ 中的文件是长期维护的参考文档，不随单次实验更新
- 只保留满足以下条件的文档: 支撑论文 claim、记录核心实验、保存相关理论推导
