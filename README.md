# Hybrid-RoPE

研究有限 RoPE 位置基的分配、学习与长度行为。当前论文是 **Beyond the Base: Exponent Allocation in RoPE**，位于 `paper-2027/`。

**从 [index.md](index.md) 开始查询。** 当前主线是内部指数分配如何组织位置基、参与模型表征学习，并通过具体构造改善长度泛化与上下文利用。历史的“超越 MrRoPE-Pro”、稀疏注意力和算子研究有独立记录，不替代当前论文目标。

- [论文资产与解释](paper-2027/research/PAPER_REVISION_HANDOFF_20260911.md)：重要性、证据和相互关系。
- [下一阶段研究计划](docs/research/next_stage_20260912/index.md)：理论、实验、论文三方评估与综合安排。
- [当前论文与构建](paper-2027/index.md)：PDF、源码、复核和两轮审稿。
- [AGENTS.md](AGENTS.md)：稳定工作原则与维护约定。

快速构建：在仓库根运行 `bash paper-2027/compile.sh`；源码打包：`python3 paper-2027/package_source.py`。运行环境和可选数字复核见论文索引。文档浏览无需运行模型。

分类入口统一为小写 `index.md`。README 解释目录用途，日期报告保存时点事实，实验原始行/回执保留来源路径；当前论文数字沿证据索引查原件。旧 `paper/` 等精简前材料可从 `main_0726` 只读查看。
