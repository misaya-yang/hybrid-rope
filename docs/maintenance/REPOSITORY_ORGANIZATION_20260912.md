# 以 Beyond the Base 为中心的仓库整理

日期：2026-09-12。当前统一入口：[根index.md](../../index.md)。

## 完成的整理

1. **重写论文交接总档**：按科学问题组织，区分论证重要性、证据强度和后续工作优先级。A01–A28覆盖固定支持、多形状、完整位置基、范围/共适应/槽位、构造、模型收益、恢复资产和新发现的成熟z学习历史；逐项解释与其他成果的联系。原502行R1完整保存在论文history中。
2. **建立证据登记**：28项资产、52处source记录，给出来源路径、本地可达性、SHA、论文位置和必要解释。52是引用次数，部分来源由多资产共享，不表示52个独立实验。MLA本地原始评价JSON已核SHA；不由此声称旧cache独立。
3. **形成三个独立研究报告与综合计划**：理论、实验、论文分别审查；用户最新资源纠正已覆盖旧D6/D14限制。主合同推荐实际2K、两个预指定support、Geo/Cosh/full-z、三seed充分训练；机制与同批模型共用，独立MLA复评、C42/C2确认和gain析因各自回答问题。尚未执行这些模型实验。
4. **统一分层index**：根、论文、研究、证据、理论、成熟模型、战役、实验实现、脚本、数据/结果、测试、档案和维护入口统一为小写`index.md`。README改为用途介绍，AGENTS补稳定导航/维护约定，不堆具体实验结果。
5. **修正时序误导**：根README不再以旧“打赢MrRoPE-Pro”替代当前论文主线；ai-handoff、rebuttal、ds_workspace旧“正在跑”和旧评审计划原文归档，当前入口不将它们当实时状态或执行授权。旧Top15保留为历史选材依据，不作为当前价值上限。
6. **分类而不破坏来源**：普通独立文档归入主题文件夹；代码/JSON/hash引用的结果原件保持原路径，通过results/theory/protocols/reviews/history分类index查询。21个实验家族保持实现路径，逐个建立导航。ignored模型/缓存/raw没有迁移或公开打包。
7. **保存外部指导**：作者指定Pro手册原样存入`paper-2027/research/external-reviews/pro-guidance-20260911/`并登记SHA；指令属于参考输入，具体事实以来源和当前作者要求判读。

## 实际迁移与路径保持

[relocations.json](relocations.json)记录5个实际位置/命名变化：

- `INDEX.md` → `index.md`（通过临时名完成大小写转换）；
- `analysis/unify_20260910/INDEX.md` → 同目录`index.md`（内容不变）；
- `docs/research/PC2_FAILURE_AND_CLAIM_AUDIT_20260910.md` → `docs/research/reviews/`；
- `docs/research/PC2_TEN_EXPERIMENTS_RESEARCH_PLAN.md` → `docs/research/protocols/`；
- `docs/research/ROPE_ALLOCATION_PROGRESS_20260910.md` → `docs/research/history/`。

另有59个候选叙述文件发现代码/JSON/来源清单依赖，保持原位并登记依赖。原文、日期报告和raw的存在是为了复查；文件名不自动授予科学权威。对仍含历史相对链接的原样快照，按原路径语境阅读；它们不纳入当前导航检查。

## 恢复点

整理前关键入口原文保存在[导航快照](../archive/navigation_20260912/index.md)，完整R1在[论文历史](../../paper-2027/research/history/PAPER_REVISION_HANDOFF_R1_20260912.md)。

本轮最初创建的Downloads完整markdown备份曾成功写入，但随后作者清理Downloads后已不在原位置，位置未追回；不能把它列作目前可用恢复包。当前恢复点补存在仓库忽略目录`internal/local_snapshots/repo_organization_20260912_012317/`，内有当前markdown/论文、Git patch/status、SHA清单。它是整理过程中的恢复点，不冒充完整整理前基线。后续备份优先放仓库忽略目录。

## 验证与范围

运行 `python3 scripts/check_repository_docs.py` 只读检查当前受管理文档链接、28项资产来源及迁移目标；结果写入[organization_validation.json](organization_validation.json)。[repository_inventory.json](repository_inventory.json)列Git可见的存在文件（含未提交），ignored原始目录只登记目录级范围，不声称扫描了全部模型数据。

本轮没有修改训练/评价实现，没有模型运行、远端操作、提交或推送。新增脚本仅为文档检查；`.gitignore`只增加root results/index.md例外，原始results与internal恢复点仍被忽略。原有其他未提交实验工作保留。

当前PDF和源包与本次整理前的交付哈希相同：

- PDF：`602073f3ab00e7ebdb3fae2fd4fc5bc1d034ff85944ac8306f75ec7c731f7c0d`
- ZIP：`6f900e566cca51d52b3dbcd1c9103c9357e0df89500d67946f08ae4bbb071e8c`

论文原有科学内容/版面未改变，因此本轮验证导航、来源与包身份，不以重复模型或编译充当文档工作完成。`git diff --check`通过。

## Git路径与跨机器复核

核心文档不依赖工作机绝对目录，Markdown使用文档相对链接，JSON来源使用仓库根相对路径。可见文件含tracked和新增未提交；后者需纳入未来Git提交才随两机同步。本轮未提交。ignored结果镜像采用路径文本并标本机存储，不混入Git可迁移链接。

原根INDEX历史文本从本轮初次完整读取恢复，详见导航快照的snapshot_provenance.json；其他关键快照直接复制。目录index与原root_INDEX_before.md区分命名，避免大小写不敏感文件系统覆盖历史原件。

最终跨机器模拟：仅将2,369个Git可见文件置于独立目录，不包含ignored raw镜像，79个受管理入口/文档中的1,249个相对链接通过；51处可随Git提供的来源核验通过，另1处ignored MLA原始评价在本机模式核验，总计52处引用、49个唯一source路径。新增文件尚未提交，需随未来Git提交同步到另一机器。

Git大小写处理：macOS普通文件重命名不足以让Git记录大小写变化，因此使用`git mv`记录根和analysis/unify两个INDEX→index重命名。暂存区仅含这两项R100路径变化；其余内容改动未暂存。没有commit或push。
