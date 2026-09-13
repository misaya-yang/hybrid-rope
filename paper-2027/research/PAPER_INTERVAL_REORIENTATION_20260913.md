# 论文主线更新：固定表的全窗口上下文质量

2026-09-13。本轮读取作者指定的“5090实验负责agent-5.6sol”任务最新消息、近期结果owner和理论审计，随后以作者在本任务中的最后纠正为最高依据。解释YaRN/MrRoPE差异不是论文主线；实际质量和强基线比较才是目标，三个变量是研究手段。

## 实际修改

- 改写摘要、引言贡献和结论，以固定表服务整个可用窗口为部署目标；保留已有BM与MLA代表数字，不把新开发信号升级为确认结果。
- 成熟模型章节前移为§5，学习期构造为§6；保留EVQ完整理论、MLA、750M、8B来源使用、OLMo独立EOS与selective-QK结果。
- 新§5.3定义band、过渡、终值作为设计变量，明确端点目标和区间目标不同。没有把“解释前人进步”列为贡献，没有声称已经全面超过强基线。
- 新附录H给出固定表评价对象、条件深度定理完整证明、任意长度分布下Native/端点加权解，以及周期失真的局部最大值反例。checkpoint replay仅为候选预测器。
- 纳入Llama匹配C42V24衍生shape的两条长度曲线及OLMo324行mini表，明确开发身份。报告依据来自[BAND_NEXT_STAGE_RESULT](../../docs/research/next_stage_20260912/BAND_NEXT_STAGE_RESULT_20260913.md)；本轮未读取服务器raw输出或重算bootstrap。旧350行/E3/QA面板没有被混合覆盖。
- 当前目标、优先级、后续实验和理论升级统一归入[当前研究方向](../../docs/research/next_stage_20260912/PAPER_INTERVAL_DIRECTION_20260913.md)，并同步稳定入口、论文handoff、narrative guide和证据映射。

## 研究判断

全面胜过YaRN/MrRoPE是值得追求的结果，但不需先证明它不可能才能直接研究全窗口固定表。区间目标必须在评价前说明：task-valid短段、Native、中间点与SL端点同时考虑，允许明确的合理权衡，不能事后换目标制造胜利。transition、band、是否/s用来构造更好的表，并非三个独立已解决的新方法。当前没有充分确认的统一赢家。

下一阶段优先补齐已有强表的可比曲线与原生短段，直接做首次深度对照，再以少量匹配band/shape控制推进有效候选；replay是否值得成为方法由实际预测和生成结果决定。无新模型运行或实验调度。

## 来源与理论核对

核对了[MrRoPE原论文§3.2](https://arxiv.org/html/2601.22181v1)与[YaRN原论文](https://arxiv.org/html/2309.00071v2)。前者的实验明确采用零训练推理比较；不能把“YaRN原方法有微调”写成永远禁止静态YaRN对照的理由。新的强基线比较需明确采用静态运行模式还是完整训练方法。本次保留现有已定义对照，没有新增训练臂。

独立Simpson积分、有限差分及代数残差核验：区间导数、二次/四次最优值、加权解、非对称解和周期反例通过。开发表AUC由四舍五入数据复算与原汇总差小于0.01pp；不把该算术检查称raw验证。脚本为[verify_interval_design.py](../figs/verify_interval_design.py)，输入为[interval_development_inputs.json](../figs/interval_development_inputs.json)。

## 验收

构建、逐页视觉检查、源码包独立编译与最终分页以[验收记录](interval_reorientation_validation.json)为准。本版没有新增独立PDF-only审稿；先前两轮属于20260912_story版本，不冒称审过本版。

修改前恢复点位于仓库ignored的`internal/local_snapshots/interval_story_20260913_090930/before.tar.gz`。保存了修改前paper源码、PDF、源包和核心入口；不是必须跨机器导航的Git文件。无提交、推送或模型实验。
