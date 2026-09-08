# Hybrid-RoPE：频率分配与长上下文研究

本项目研究 RoPE 的有限频率分配如何影响原生能力、长上下文生成及训练后的部署表现。活动论文是 `paper-2027/` 中的 **RoPE Has a Spectral Budget**，当前分支为 `main_0726_09_06`，正在进行 ICLR 方向的研究与论文重构。

目标是从 MrRoPE、CoPE 和本方已有方法出发，找到更有效的中段分配、低频缩放与区域衔接，并按当前顺序研究零训练改进、LoRA适配与能力保持、稀疏注意力中的位置编码。SOTA 是研究目标，现有小型开发结果尚未达到这一结论。

## 当前研究定位

- **优先本方方法。** 初期复用论文与已有对手结果，不默认重跑完整基线、对手微调或笛卡尔积消融；比较条件不同处简要注明。
- **从有效方法改进。** MrRoPE 提供累计 radix 与有效中段分配，CoPE 提供深尾稳定机制，本方已有 Z/分配结果提供候选与经验。Cosh 保留为历史证据，不继续优化它的曲线。
- **按实际终点判断进展。** 区分CPU数学检查、真实模型验证、开发结果和能力
  结论；研究已知失败及后续纠正，从已有有效方法推导可区分预测。
- **先审核实验代码，再做必要检查。** 复用稳定路径的验证与资产记录，避免反复
  hash和smoke；科学预测、工程检查与实用验收分开，结果分支在运行前写清。

公司PC接续先读[HANDOFF](paper-2027/HANDOFF.md)，其中维护阶段状态、资源与下一
行动。具体实验结果、失败谱系和准备中的协议按下表读取，README不另维护作业状态。

原有 support/allocation 表述、固定端点和统一符号本身不足以承担新颖性。论文重构应围绕真正有效的构造、可复现输出和有用的机制解释展开；完整理论不是开始有价值试验的门槛。

## 阅读和文档职责

| 文件 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 核心约束、阶段内自主执行、GPU 资源管理和必要验证 |
| [INDEX.md](INDEX.md) | 详细文件索引；区分当前结果、设计输入与历史材料 |
| [paper-2027/HANDOFF.md](paper-2027/HANDOFF.md) | 当前暂停/运行状态、授权预算、资产位置和恢复工作入口 |
| [研究主线](docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md) | 方法关系、文献核对和历史方案 |
| [本夜实验报告](docs/research/ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md) | 全部阶段结果、成本、局限、失误复盘及证据路由 |
| [本地失败谱系](docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md) | 已证事实、具体否证与未知；整改报告机制/定理复核及三项执行纠正 |
| [固定位置可见性协议](docs/research/ROPE_FIXED_POSITION_VISIBILITY_PROTOCOL_20260908.md) | 有条件的单行判别、代码审核、一次必要检查、评分分支与停止条件 |
| [外部材料](paper-2027/research/external-reviews/README.md) | Pro原始输入、覆盖清单及来源身份 |
| [REVISION_BRIEF](paper-2027/REVISION_BRIEF.md) | 论文重构契约；旧阶段排序只作历史 |

恢复工作先读 HANDOFF，再按 INDEX 打开相关 owner。不要批量阅读全部理论史，也不要把一个旧计划文件当作待执行队列。

## 代码与已有资产

[scale_transport](scripts/experiments/scale_transport/)包含本轮统计、缓存重放、有限表构造、问答诊断和 RULER 子集入口。[cross_audit](scripts/experiments/cross_audit/)提供已验证的早期准备、冻结评估、训练探针及作业监督组件。脚本和结果的实际状态由协议/HANDOFF 区分，不能因文件存在就推定它已运行。

早期 OLMo E0/E1 与 scratch overlays 已完成，见 [ROI/结果核对](docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md)。旧 Z 训练提议、三臂微调和 seed42 权重恢复均不是当前继续入口。已有来源和方法身份保留，新的候选收益需要自己的证据。

## 仓库边界与本地工作

活动分支保留 `paper-2027/`、`docs/`、`scripts/`、`tests/` 和根路由文件。`paper/`、`rebuttal/`、旧顶层 results 等 pre-slim 内容在 `main_0726` 历史分支；不恢复、修改或编译该归档来迁就旧脚本。

个人电脑做阅读、代码/文档准备和轻量 CPU 检查；实际 PyTorch/GPU 验证沿用工作机环境，不在家用机重新搭建训练环境。只执行与变更相关的 AGENTS 验证入口。活动论文构建命令为 `bash paper-2027/compile.sh`，构建成功不代表科学结论成立。

模型、缓存、服务器私有回执和凭证不进入 reviewer-facing 包。当前 supplement allowlist 仍需与精简 checkout 对齐后才能发布。本地修改、Git 发布、服务器运行和论文提交是不同状态，具体身份见 HANDOFF。
