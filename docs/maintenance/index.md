# 仓库维护与检索

本次以论文为中心建立分层小写index，保留有依赖的代码/raw/回执位置。

- [组织说明](REPOSITORY_ORGANIZATION_20260912.md)：改了什么、未动什么、完整性检查。
- [文件清单](repository_inventory.json)：Git可见文件逐项路径/分类，含新增未提交文件；ignored原始目录单列。
- [迁移与保留路径](relocations.json)：实际移动、因代码/JSON来源依赖保留的文件。
- [验证结果](organization_validation.json)：新索引链接、52来源、原文件保持情况。

- [Astra环境整理](ASTRA_ENVIRONMENT_AUDIT_20260913.md)：按任务读取、项目技能范围和验收边界。

- [默认上下文入口整理](CONTEXT_ROUTING_20260913.md)：当前路线与按需历史目录。

## 后续如何维护

1. 新结果：保留配置/代码/数据/权重/表身份、指标与单位、完整结果和原始行，登记到论文asset_registry（若用于本稿）。
2. 新文档：理论放theory/foundations，协议放protocols，复盘放reviews/audits，历史放history/archive；实验家族代码和结果仍与已有运行路径一致。
3. 更新最近一级index；改变论文主张时同步claim map与交接。根index只保留稳定分类，不堆所有日期文件。
4. 来源路径要移动时先查脚本、JSON、Markdown和hash引用，写迁移清单并检查链接；不要以目录清爽为由移动活跃runner/原始流。
5. 运行 `python3 scripts/check_repository_docs.py` 检查受管理入口。历史快照的相对链接保留原路径语境，不当作当前有效导航。
6. 可移植实验报告不得写入个人电脑绝对路径。当前next-stage结果、论文证据目录和冲刺便携报告会由检查器拒绝`/Users/...`与Windows用户目录；使用文档相对链接、仓库根相对registry路径，远端raw只记录必要的实验服务器路径。

README介绍用途；AGENTS保留稳定原则；日期状态只写在具体报告。计划、准备、CPU核验、模型完成四种状态分别表达。

## 跨机器路径检查

默认检查Git可见导航（包括拟随Git提交的新增文件），路径严格按Git大小写匹配。新增文件/目录后运行`python3 scripts/check_repository_docs.py --refresh-inventory`刷新文件清单；本机原始镜像复核另用`--local-evidence`。忽略目录只保留仓库根相对路径文本，不成为跨机器必需链接。

- [Sol每小时研究监督](SOL_RESEARCH_SUPERVISION_20260913.md)：用户授权范围、首轮反馈和待跟进问题。
