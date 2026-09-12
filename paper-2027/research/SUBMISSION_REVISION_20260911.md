# 2026-09-11 正式投稿稿修订说明

本次以 `review_report_and_optimization_plan.md` 和 `gemini-code-1789141359563.md` 为审稿输入，按“已有仓库证据优先，清除未执行描述”的要求修订。没有启动新的模型训练、推理或远程 GPU 实验，没有提交或推送 Git。

## 已完成的正文修改

- 重写摘要、引言、贡献和结论：以固定支持训练、范围重设、权重—表交叉和实际方法收益为主线。
- Figure 1 合并固定端点示意、三 seed 固定支持曲线、三 seed 范围重设曲线与两 seed 交叉矩阵；自然 QA 图恢复正文。小型 Qwen-3B gain 表保留为附录实测对照。
- 原 §4.3 的混合“静态指标不可能性定理”被替换为频率分配与槽位指派的区别、精确块置换补偿关系和已测干预结果。附录仅保留有明确风险差前提的预测误差下界及其三角不等式证明。
- 增补锚定 Cosh 的有限表性质：逆 CDF 凸性给出内部归一化指数不大于均匀网格、固定端点下对应频率不小于 Geo。
- §6.4 只保留可证明的相位有效距离/导数和可重算的十二臂响应描述；不再声称相位包含保证校准、慢频内容退化或压缩无害。
- 区分 432M midpoint 与 750M anchored-inclusive 协议；Llama 的 300-step LongAlpaca 适配与 516-step RULER-family continuation 分开标注。
- Qwen-0.5B 的 +6.09 明确属于 table–amplitude 联合配置；小型 Qwen-3B 对照仅报告该 panel 的条件差异。

## 删除、纠正与证据边界

- 清除正文/附录中的 `not yet run`、planned/ongoing baseline、未执行的 YL/IH 析因表格行，以及 X1/X2/X3 修订标记。
- 删除无矩阵计算记录支撑的 Fisher/Hessian 条件数、主特征向量、slot 19 最敏感方向和“任意接近参考仍失败”的渐近断言。
- 删除未用于主结果且容易混淆观测对象的候选 Jacobian/内容耦合机制段落；不把它们改写为已完成实验。
- 移除未在本次检索中定位到原始测量记录的 0.61-turn/−0.046 NLL、−0.023 NLL/−10.6 task-points 等单槽及机制叙述，不据此认定这些实验从未发生。
- 原 orbit-jitter 段落混用了符号和区间。按实际结果修正为 16K jitter-minus-exact NLL −0.000013，95% 区间 [−0.003682, +0.003715]。
- 十二臂由保存的 350-row 输出逐行重算。其评分含部分得分，改称平均任务分数。拟合仍为 −20.4532 + 1.19068 N；66 对中 39 对计数不同且同向、27 对计数相同。因此不能称为完整排序预测或独立验证。
- 十二臂表 hash 是按现有构造重建的 float32 tensor digest，正文与附录不将其冒充运行时 attestation；输入核对为保存的 row_id 一致性。
- AI use statement 改为工具用途和作者责任说明，删除“所有表项均已重算、所有证明均已独立重导”的无条件全称自证。

## 主要仓库证据

- 固定支持三 seed：`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json`。
- 151.9M crossing：`paper-2027/research/attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`。
- 自然 QA：`docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`；778 个保存评分，631 个长输入，五任务宏平均已重算。
- C2：`paper-2027/research/attention-aware-retrofit/results/coupling-transfer/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`。
- Orbit jitter：`paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md`。
- 十二臂构造：`ds_workspace/recon_20260910/code/midband_n.py` 与 `experiments/curvature_20260910/tables.py`；实际输出：`ds_workspace/recon_20260910/work/jsonl/{archive,olmo,olmo_knife}/`。重建及逐文件 hash 已写入 `paper-2027/figs/profile_diagnostic_inputs.json`。

## 两份审稿意见的取舍

采纳两份意见共同指出的未执行实验残留、贡献弱化、样本量与 gain 归因问题。没有采纳 Gemini 对旧不可能性定理的肯定，也没有把其建议的“物理守恒/容量再分配”补成未经证明的机制。Resonance RoPE 的作者按正式来源修正为 Wang et al.，不是 Gemini 所列 Shen et al.。

补充并核验的文献：
- [RULER, COLM 2024](https://arxiv.org/abs/2404.06654)
- [Resonance RoPE, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.32/)
- LongRoPE 与 [LongRoPE2](https://proceedings.mlr.press/v267/shang25a.html) 的机制分别描述。

## 验证与产物

- `bash paper-2027/compile.sh`：通过。正文含声明止于第 9 页，参考文献从第 10 页开始，总计 41 页。
- 0 未定义引用，0pt overfull hbox；匿名、Letter 纸张、字体嵌入检查通过。
- 全 41 页渲染查看，并放大检查主文联合图、理论页、QA 页及十二臂表。
- `verify_explicit_geometry.py`：8 项基函数计算，最大差异 2.71e−14。
- `verify_profile_diagnostics.py`：12 个重建频率表、评分聚合、拟合和 66 对分类一致。
- 锚定 Cosh 不等式数值核对覆盖 60 组 K/τ/grid 设置；这是公式核对，不是新模型实验。
- 源码包 `exponent-allocation-source.zip`：62 个文件，ZIP 与 SHA-256 manifest 校验通过。
- 原有未提交 PDF 备份：`/Users/yang/Downloads/paper_revision_20260911_backup/main_before_revision.pdf`。

本次完成的是投稿稿及其源码整理，不等同于会议提交或外部审稿通过。后续实验建议不作为本稿已完成成果陈述。
