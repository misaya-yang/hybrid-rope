# 指数分配稿：主张、公式、图表与引用对应表

2026-09-09。内部审查表；不进入匿名稿件。实验优先级及逐项裁决见 [Top 15](EXPERIMENT_ASSETS_TOP15_20260909.md)，完整文件SHA256见 [source index](EXPONENT_REVISION_SOURCE_INDEX_20260909.json)。新图/表的直接输入及运行检查另见 [figure receipt](../figs/exponent_revision_source_receipt.json)。

## 理论与构造

| 主张/构造 | 公式与位置 | 直接证据与核验 |
|---|---|---|
| 非均匀指数分配可与频率范围分开描述 | §2 `omega=b^(-phi)`；`x=a+Rz`，K≥2、b>1、严格有序正频率表 | 这是定义与控制工具；geometric的z始终等距。形状的行为后果由Top15 #1、#3识别。 |
| 完整位置对象是sin/cos二维子空间 | §3.1 `Q=S_omega^(-1/2) H S_nu^(-1/2)`、`c=||Q||_F^2/2` | `appendix/a1_proofs.tex`给完整trigonometric Gram与相位不变性；foundation report §2保存数值检查。 |
| 平均canonical collision关联有效rank | §3.1 `r2=2K/[1+(K-1)c_bar]` | 同一block-whitened Gram的trace恒等式，完整证明在A1；不是raw entropy rank或LM loss。 |
| 慢频率共享位置子空间 | §3.2 `V_omega→span{1,Delta}`；`2-||Q||²=O(epsilon⁴)` | Uniform[0,L]下证明与展开；标准k/K网格K64,b500K,L4096有23慢pair、r2=2.00013。endpoint-normalized网格的24pair是另一配置。 |
| Cosh是明确变分目标的闭式解 | §3.3平方密度+min(phi,psi) interaction；rho_tau、inverse CDF | alpha>0,beta≥0、正C²、单位积分；严格凸与边界条件导出cosh。正文一次说明surrogate，完整推导与tau规则在A1。 |
| 成熟表的指数位移统一三种操作 | §5.1 `d=log(omega_N/omega')`；频率混合、log-shift、mixed-radix各自公式 | 频率混合是`-log(1-w+w/s)`，不能把w误当log-shift的m；radix乘积取log成为sum。 |
| BM平滑的是radix增量 | 附录A7，epsilon_q=6q(N+1-q)/[N(N+1)(N+2)]，累加得三次m_q | `ROPE_MRPRO_BM_CANDIDATE_20260908.json`保存OLMo N18、Qwen N17、逐项exponents；绘图代码逐项断言相等。 |
| finite-grid profile placement有不同构造 | A7参考K64插值与target local-gap直接计算 | `export_frozen_coupling_transport.py`；K128/K32两份confirmation identity逐字段核对，gain在每一比较内一致。Gemma reference4K是operating reference。 |
| 454M历史scaler的准确身份 | A2 `R_s(omega)=omega/[s^r T(s)^(r/2)]`, T=1+.07log2(s) | `official_yarn.py`中的legacy fixed-index operator；K32、cutoffs6/28、s8、T1.21、cos/sin gain1。与官方YaRN单独命名。 |

## 图表契约与最终表面

使用可复现Matplotlib矢量PDF；新图同时导出PNG预览。论文固定宽度内检查字体、零线、范围、标签、图注和黑白可辨性。基线蓝色/圆点、构造橙色/菱形；co-adaptation矩阵用同一橙色根表达相对匹配格的NLL增量。图源为现有实验，未执行模型。

| 图/表 | 问题与直接主张 | 形式/数据粒度 | 输入与呈现 |
|---|---|---|---|
| `fig_evidence_overview` | 相同端点下改变什么、是否有实证后果？30个interior改变，所有9个seed×OOD length差值为负。 | 有序样点图+四个真实评价长度的paired line；3训练seed，均值不代替单seed | Top15 #1；正文首图。固定range结果不与retargeted结果拼接。 |
| `fig_8b_length_curve` | 在适配窗口之外发生什么？完整8/16/32K PPL曲线显示长程收益与窗口内成本。 | 两条有序长度曲线，log-PPL轴；24 packs/length | Top15 #2；300-step pair，不能与516-step RULER后续pair混用。 |
| `fig_weight_table_crossing` | 为什么第二阶段相对native表调整？权重偏好与其训练表相容的运行表。 | 两个2×2矩阵；左标PPL，右标tail NLL；阴影统一表示diagonal-relative ΔNLL | Top15 #9；50M来自报告§5.2，151M来自`small_model_crossing`，不引用Qwen K32 receipt。 |
| `fig_bm_natural_qa` | 指数调整能否在自然输入获益？五任务均值均增加，macro21.62→25.44%。 | 五任务+task-equal mean的paired dot；样本166/173/119/61/112 | Top15 #6；对778个row_id去重、重算两length strata逐任务均值，再绘631长输入。 |
| `fig_bm_exponent_profiles` | BM与MrPro究竟改了什么？同边界/总量，增量向band中间重分配。 | 两个离散transition width的cumulative profile，N18/N17 | 直接对比recorded exponent arrays；放附录，数值差异不是新的模型实验。 |
| `table_index_full13` / `table_coordinate_confirmation` | 静态表的breadth及不同placement | 分任务官方RULER分数/配对差值CI | macro按task均分；K32两个长度用97.5% CI，K128用95%；旧pilot不pool。 |
| `table_bm_tasks` / `table_bm_qa_all` | 保留模型、任务与生成端点细节 | 三模型六任务分项；自然QA完整输出F1%与EOS计数 | 不把FWE/VT部分分数称exact accuracy；不从generation_config默认值猜实际cap。 |
| `table_evq_ramp` | 训练时分配与后续固定scaler的组合 | 四臂、三训练seed；PPL与teacher-forced PK分列 | 只使用full-sequence summary；不拼接per-document PPL与早期s4单seed数据。 |

三点/四点评价长度是作者明确要求保留的离散模型评价网格，并非时间序列抽样不足；不插值生成新观测。所有caption写实际条件和metric。

## 最近相关工作的实质关系

| 文献 | 本稿比较的实际内容 | 原始来源 |
|---|---|---|
| FMRoPE / Frequency Bands | base选择保持geometric normalized exponents等距；本文固定端点改变内部形状并做配对训练 | [Oka et al.](https://openreview.net/forum?id=PR1PPxvG9Q) |
| YaRN | NTK-by-parts频率混合与cos/sin amplitude；指数位移是混合后的负log | [YaRN](https://arxiv.org/abs/2309.00071) |
| LongRoPE | dimension-wise factors加token-position threshold；位移公式只对应频率因子，不能代替位置阈值机制 | [LongRoPE](https://proceedings.mlr.press/v235/ding24i.html) |
| MrRoPE | mixed-radix累乘转成指数位移累加；BM在同band边界下修改增量形状 | [MrRoPE](https://arxiv.org/abs/2601.22181) |
| LeRoPE | 每pair的log-space scale跨层/头共享，并与weights联合学习；不是因名称不同就与指数分配无关 | [LeRoPE](https://arxiv.org/abs/2607.10134) |
| AdaRoPE | head-specific learned frequencies和attention scaling；本文聚焦几何、显式密度与受控指数问题 | [AdaRoPE](https://arxiv.org/abs/2607.19363) |
| DoPE | 以truncated matrix entropy分析rotated activations低秩并修改PE；本稿比较supplied sin/cos subspaces | [DoPE](https://arxiv.org/abs/2511.09146) |
| Du et al. | 长上下文position/token辨识理论；本稿有限分离区间的basis geometry与学得表征交互是不同对象 | [Du et al.](https://arxiv.org/abs/2605.15514) |
| Gemma | 新加入实验的模型家族来源；具体1.1 checkpoint identity在本地confirmation receipt | [Gemma paper](https://arxiv.org/abs/2403.08295), [官方model card](https://huggingface.co/google/gemma-1.1-2b-it) |

相关工作保持紧凑。核对表为内部工具；正文不复制逐项novelty审查，也不复述一年的试错过程。

## 复现与记录层级

Figure builder重算分任务均值、样本数与部分记录一致性，并绑定原始输入hash；这不等于重新运行训练或生成。历史报告、curated summaries和raw-backed receipts按其实际层级使用。论文源包包含完整TeX、styles、bibliography、所有引用PDF图与成稿；模型checkpoint及原始流单独维护。
