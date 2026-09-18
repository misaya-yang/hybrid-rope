# Hybrid-RoPE

研究 RoPE 内部频率配置、学习与上下文利用。当前论文是 **Beyond the Base: Frequency Allocation in RoPE**，位于 `paper-2027/`。

**从 [index.md](index.md) 按任务进入。** 当前论文通过固定范围和等位移控制识别配置的作用，分析位置结构与学得使用，并以TailSpline、原生窗口干预和学习／适配验证实际价值。研究与执行状态由[当前研究入口](docs/research/next_stage_20260912/index.md)维护。

- [当前研究](docs/research/next_stage_20260912/index.md)：目标、固定表确认与当前证据。
- [论文与构建](paper-2027/index.md)：稿件及交付。
- [AGENTS.md](AGENTS.md)：稳定工作原则。
- [跨机器接续](docs/maintenance/CROSS_MACHINE_CONTINUATION.md)：把本工作树安全同步到家里PC；仓库内技能无需依赖这台Mac的个人记忆。

快速构建：在仓库根运行 `bash paper-2027/compile.sh`；源码打包：`python3 paper-2027/package_source.py`。运行环境和可选数字复核见论文索引。文档浏览无需运行模型。

分类入口统一为小写 `index.md`。README 解释目录用途，日期报告保存时点事实，实验原始行/回执保留来源路径；当前论文数字沿证据索引查原件。旧 `paper/` 等精简前材料可从 `main_0726` 只读查看。
