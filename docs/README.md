# 文档目录 (Documentation Index)

> 最后更新：2026-02-22

欢迎查阅本项目的文档与实验索引。为了保证论文数据的严谨性和代码复现的确定性，请遵循以下结构：

## 1. 优先级文档 (Priority Docs - Current)

这些是**唯一权威**的参考资料，可直接用于论文写作：

| 文档 | 用途 |
|------|------|
| ⭐ **[EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)** | **实验事实表**：所有跑数记录、结论状态与原始 JSON 证据链汇总。任何未在此表中标为 ✅ 的数据，严禁放入论文正文。 |
| ⭐ **[TERMS_AND_PROTOCOLS.md](TERMS_AND_PROTOCOLS.md)** | **术语与公平协议标准**：定义何为可用的评测分数、跨方法比较的红线，以及禁止引用的黑名单。 |
| [RESULTS.md](RESULTS.md) | **论文口径核心结果**：直接为论文 Table/Figure 提供数据的简明汇总页。 |
| [METHODOLOGY.md](METHODOLOGY.md) | **实现方法与评价方法**：核心频率映射函数、滑动窗口取样、Token 切片规范。 |
| [REPRODUCE.md](REPRODUCE.md) | **最短复现路径**：包含如何一键跑出核心数字的三条路径（小模型/大模型验桩）。 |

## 2. 详细分析资产 (Knowledge Base)

存放在 `../knowledge_base/`，用于论文撰写素材和理论自洽性证明：

| 编号 | 文档 | 用途 |
|------|------|------|
| `00` | [项目与结论总览](../knowledge_base/00_项目与结论总览.md) | Claims / 核心贡献与理论脉络 |
| `01` | [已完成实验核心数据](../knowledge_base/01_已完成实验核心数据.md) | 提供至实验事实表的最快指针 |
| `08` | [8B 实验分析](../knowledge_base/08_8b_experiment_analysis.md) | 分析为何需要完全公平条件、Failures & Fix |

*(其他文件参见对应目录)*

## 3. 遗留归档文献 (Legacy - Do Not Cite)

以下文件仅供历史追溯，**其内容已过时或包含不公平测试结果，不得引用**：

| 文档 | 说明 |
|------|------|
| `EXPERIMENT_OVERVIEW.md` |已被 `EXPERIMENT_REGISTRY.md` 替代。|
| `RESEARCH_STORYLINE_*.md` | 已被合并入 `knowledge_base` 和 `paper_draft`。 |
| `QWEN_STANDARDIZED_COMPARISON...`| 已被更加严格的公平注入协议要求替代。 |

---
**阅读建议**：
若您是评审委员或合作者，请严格从 `EXPERIMENT_REGISTRY.md` 循迹以复核所有声称效果的真实性。