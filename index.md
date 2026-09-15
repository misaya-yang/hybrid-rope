# Hybrid-RoPE：当前工作入口

当前论文：**Beyond the Base: Frequency Allocation in RoPE**。
核心主张：固定频率支持下，内部allocation具有独立且可利用的作用；位置基结构与学得的频率—坐标匹配可通过受控干预区分。已有构造收益分别按其训练/部署协议成立；当前TailSpline候选不代表这些证据已经统一成同一方法。
实验主线分别研究高频保持边界、低频缩放终点、固定两端后的中频过渡；每项给出数学解释、可计算方案和最小验证。论文主线为 z 发现 → EVQ → full-z → 基于 z 变化超过 YaRN/MrRoPE 的实验现象 → 更优理论。全窗口质量是评价方式。

| 当前任务 | 入口 |
|---|---|
| 研究目标与下一步 | [当前研究索引](docs/research/next_stage_20260912/index.md) |
| 修改论文、编译与交付 | [论文索引](paper-2027/index.md) |
| 核实主张与实验数字 | [证据索引](paper-2027/research/evidence/index.md) |
| 实验代码与结果确认 | [实验索引](experiments/index.md) |

按任务选择一条路线，已知文件直接读取局部上下文。既有 scratch、LoRA、BM 等结果仍是论文基础；当前研究排序不改变其证据效力。

[完整目录与历史检索](CATALOG_20260913.md)只在追溯旧结果、旁线或定位其他目录时使用，不是启动阅读清单。[维护入口](docs/maintenance/index.md)用于新增文档和导航检查。
