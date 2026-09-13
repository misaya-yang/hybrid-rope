# Hybrid-RoPE

研究有限 RoPE 位置基的分配、学习与长度行为。当前论文是 **Beyond the Base: Exponent Allocation in RoPE**，位于 `paper-2027/`。

**从 [index.md](index.md) 开始查询。** 当前研究以既有分配识别和学习证据为基础，推进一张固定RoPE表在任务可用短端、原生段、中段和目标端的上下文质量，并与YaRN、MrRoPE等强方法比较。transition、band和低频终值是设计手段；已有方法差异解释不作为论文主线。最新定位见[当前研究方向](docs/research/next_stage_20260912/PAPER_INTERVAL_DIRECTION_20260913.md)。

- [当前研究](docs/research/next_stage_20260912/index.md)：目标、固定表确认与当前证据。
- [论文与构建](paper-2027/index.md)：稿件及交付。
- [AGENTS.md](AGENTS.md)：稳定工作原则。

快速构建：在仓库根运行 `bash paper-2027/compile.sh`；源码打包：`python3 paper-2027/package_source.py`。运行环境和可选数字复核见论文索引。文档浏览无需运行模型。

分类入口统一为小写 `index.md`。README 解释目录用途，日期报告保存时点事实，实验原始行/回执保留来源路径；当前论文数字沿证据索引查原件。旧 `paper/` 等精简前材料可从 `main_0726` 只读查看。
