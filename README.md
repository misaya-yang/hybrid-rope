# Hybrid-RoPE

研究 RoPE 频率分配与长上下文能力，目标是构造有实际收益的位置编码方法。
活动论文位于 `paper-2027/`。

## 入口

- [HANDOFF](paper-2027/HANDOFF.md)：当前目标、工作状态与下一步。
- [INDEX](INDEX.md)：代码、论文和研究记录的位置。
- [AGENTS](AGENTS.md)：通用工作原则。

## 仓库

| 路径 | 内容 |
| --- | --- |
| `scripts/`、`tests/` | 实验、分析代码与测试 |
| `docs/research/` | 研究方案、实验结果与复盘 |
| `paper-2027/` | 活动论文及相关研究材料 |

论文构建：`bash paper-2027/compile.sh`。
旧 `paper/` 等精简前材料保存在 `main_0726` 分支，作为只读历史保留。
