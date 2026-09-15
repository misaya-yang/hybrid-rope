# Hybrid-RoPE：当前工作入口

当前论文：**Beyond the Base: Frequency Allocation in RoPE**。
现稿研究频率覆盖范围确定之后，内部配置怎样改变有限窗口的位置结构及模型对频率的使用。
论证链为受控干预 → 完整旋转对几何与学得使用 → 解析构造及模型验证。TailSpline是主要冻结
扩展构造，Cosh是辅助外推搬运实例；同一静态表在2L/4L的任务质量提升是核心实证贡献。
研究下一步与服务器执行状态分别由当前研究索引和其链接的执行owner维护。

| 当前任务 | 入口 |
|---|---|
| 研究目标与下一步 | [当前研究索引](docs/research/next_stage_20260912/index.md) |
| 修改论文、编译与交付 | [论文索引](paper-2027/index.md) |
| 核实主张与实验数字 | [证据索引](paper-2027/research/evidence/index.md) |
| 实验代码与结果确认 | [实验索引](experiments/index.md) |

按任务选择一条路线，已知文件直接读取局部上下文。既有 scratch、LoRA、BM 等结果仍是论文基础；当前研究排序不改变其证据效力。

[完整目录与历史检索](CATALOG_20260913.md)只在追溯旧结果、旁线或定位其他目录时使用，不是启动阅读清单。[维护入口](docs/maintenance/index.md)用于新增文档和导航检查。
