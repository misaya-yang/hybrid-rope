# Hybrid-RoPE：论文与证据总索引

更新：2026-09-12。当前中心是 **Beyond the Base: Exponent Allocation in RoPE**。先按问题进入下列索引，无需顺读整个仓库。

## 最重要的五个入口

1. [论文交接总档](paper-2027/research/PAPER_REVISION_HANDOFF_20260911.md)：科学主线、P0–P3、A01–A28资产、解释与缺口。
2. [证据索引](paper-2027/research/evidence/index.md)：逐项来源、可达性、统计身份、正文位置。
3. [下一阶段计划](docs/research/next_stage_20260912/index.md)：三方独立判断与综合取舍；按科学价值安排，资源由作者协调。
4. [当前论文](paper-2027/index.md)：PDF、源码、构建与审稿。
5. [主张—图表—来源映射](paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)：定位具体论文结论。

## 按科学问题查询

| 问题 | 入口 |
|---|---|
| 固定支持与多形状是否确立分配作用 | [A01–A02](paper-2027/research/evidence/index.md) |
| 完整位置基、有限窗、Cosh推导 | [理论基础](paper-2027/research/foundations/index.md) |
| 权重交叉、同谱置换、成熟模型调整 | [成熟模型研究](paper-2027/research/attention-aware-retrofit/index.md) |
| MLA/续训/适配/完整答案EOS/QA | [论文证据](paper-2027/research/evidence/index.md) |
| BM/C42/gain/开发与独立面板 | [战役证据](ds_workspace/index.md) |
| 更广理论推导与失败记录 | [研究文档](docs/research/index.md)、[分析工作台](analysis/index.md) |
| 旧稿、旧理论、更正与Git历史 | [论文历史](paper-2027/research/history/index.md)、[历史文档](docs/archive/index.md)、[rebuttal](rebuttal/index.md) |

## 按文件类型查询

| 目录 | 用途 |
|---|---|
| [paper-2027/index.md](paper-2027/index.md) | 活动稿件与内部研究导航 |
| [docs/index.md](docs/index.md) | 研究、协议、复盘、环境和月度历史 |
| [experiments/index.md](experiments/index.md) | 全部实验实现按用途分类；保留代码位置 |
| [scripts/index.md](scripts/index.md) | 训练、表构造、评价、分析和文档检查 |
| [data/index.md](data/index.md) | curated与数据目录；本地缓存并非Git交付 |
| [results/index.md](results/index.md) | 本机结果镜像；不迁移原始流 |
| [tests/index.md](tests/index.md) | 代码检查入口，不等于模型能力证据 |
| [维护索引](docs/maintenance/index.md) | 全文件分类清单、迁移记录、检查和新增文件规则 |

## 维护约定

`AGENTS.md`只保留稳定规则，README只介绍项目/目录，`index.md`负责导航，带日期报告负责具体内容。来源文件更新时同步最近一级index与证据登记。过去文档里的“当前/今晚/已排队”均按记录时点读取，不构成新运行指令。不要把内部Pro意见、模拟审稿或尚未完成实验当作已验证结论。

## 两台机器的路径约定

核心入口路径保持稳定：`index.md`、`paper-2027/index.md`、`paper-2027/research/PAPER_REVISION_HANDOFF_20260911.md`、`paper-2027/research/evidence/index.md`。Markdown链接相对当前文档，文字/JSON来源相对Git仓库根；不依赖工作机的绝对目录。新增未提交文件需要随后纳入Git才会在PC出现；本轮没有提交或推送。

`results/`等ignored原始材料单列为本机存储，未同步时不算仓库导航断链。默认运行`python3 scripts/check_repository_docs.py`检查Git可迁移入口；需要核对本机镜像时加`--local-evidence`。
