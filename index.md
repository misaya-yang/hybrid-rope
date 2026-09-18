# Hybrid-RoPE：当前工作入口

当前论文：**Beyond the Base: Frequency Allocation in RoPE**。
研究主线是z：实际频率范围与旋转预算给定后，内部配置怎样改变位置行为和模型质量。
受控干预、完整旋转对结构、显式构造与模型任务证据组成论证。TailSpline承担冻结扩展，
NCP保留原生固定支持实验；Cosh承担训练、适配与外推证据。
当前稿正文9页、全稿35页，**标题和167词摘要已封板**。9月18日Kanana、理论解释与逐章优化已入稿，概念首图已替换；后续以实质证据和有价值的Pro建议决定正文增量。
本轮未安排新增GPU队列；结果由各自owner维护。OpenReview表单已填好，最终提交留给作者，仓库不宣称已经提交。

| 当前任务 | 入口 |
|---|---|
| 研究目标与下一步 | [当前研究索引](docs/research/next_stage_20260912/index.md) |
| 修改论文、编译与交付 | [论文索引](paper-2027/index.md) |
| 核实主张与实验数字 | [证据索引](paper-2027/research/evidence/index.md) |
| 实验代码与结果确认 | [实验索引](experiments/index.md) |
| 新会话与家里PC接续 | [当前工作handoff](paper-2027/HANDOFF.md) · [跨机器同步](docs/maintenance/CROSS_MACHINE_CONTINUATION.md) |
| 作者要求与反复纠错 | [工作约定](paper-2027/research/AUTHOR_WORKING_CONTRACT.md) |

按任务选择一条路线，已知文件直接读取局部上下文。既有 scratch、LoRA、BM 等结果仍是论文基础；当前研究排序不改变其证据效力。

[完整目录与历史检索](CATALOG_20260913.md)只在追溯旧结果、旁线或定位其他目录时使用，不是启动阅读清单。[维护入口](docs/maintenance/index.md)用于新增文档和导航检查。
