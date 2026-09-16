# Hybrid-RoPE：当前工作入口

当前论文：**Beyond the Base: Frequency Allocation in RoPE**。
研究主线是z：实际频率范围与旋转预算给定后，内部配置怎样改变位置行为和模型质量。
受控干预、完整旋转对结构、显式构造与模型任务证据组成论证。TailSpline承担零训练扩展，
NCP承担原生窗口增强，Cosh承担训练、适配与外推证据。
当前进入[已完成证据的论文增量与新旧稿对读](paper-2027/research/revision_20260916_evidence_update/README.md)。
后续在跑实验完成后再增量加入；结果和执行状态由各自owner维护。

| 当前任务 | 入口 |
|---|---|
| 研究目标与下一步 | [当前研究索引](docs/research/next_stage_20260912/index.md) |
| 修改论文、编译与交付 | [论文索引](paper-2027/index.md) |
| 核实主张与实验数字 | [证据索引](paper-2027/research/evidence/index.md) |
| 实验代码与结果确认 | [实验索引](experiments/index.md) |

按任务选择一条路线，已知文件直接读取局部上下文。既有 scratch、LoRA、BM 等结果仍是论文基础；当前研究排序不改变其证据效力。

[完整目录与历史检索](CATALOG_20260913.md)只在追溯旧结果、旁线或定位其他目录时使用，不是启动阅读清单。[维护入口](docs/maintenance/index.md)用于新增文档和导航检查。
