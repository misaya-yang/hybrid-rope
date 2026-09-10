# Cosh 改版：代理目标、非几何网格与适配范式的证据复核

本轮基于 `09_09`、起始 HEAD `fe10272`。任务是深入分析仓库，判断 EVQ 改版真正缺什么；没有启动模型训练、GPU 推理、远端作业或改动论文。除读取既有材料外，本轮重算了 Phase16 全部99个结果文件、三臂303题 QA，并重跑既有纯 NumPy 理论核查脚本。

## 结论及研究目标

用户希望获得比 Cosh 更好的非几何分配及扎实能力证据。把当前论文重述为“指数形状有作用”的机制论文，并不能代替这个目标。当前已有结果证明分配这个变量值得研究，但尚未建立能够可靠选出更优分配的理论或算法。

Cosh 是一个由特定 surrogate 选出的单参数曲线族；某个部署 τ 才是其中一个点。它有真实正结果，但“闭式可解”“碰撞减小”“PPL 改善”“任务能力改善”是四个不同命题，现有材料没有把它们连成充分的因果链。

零训练、不接触原窗口外距离、只允许短 LoRA，都不是本次任务自动继承的要求。新方法需要与相同训练/适配范式下的 Cosh 比较；不能用一个充分训练的新方案击败适配不足的 Cosh，再把全部收益归给网格。

## 1. Cosh 推导的断点比“一个近似”更多

旧理论入口：[EVQ_COSH_THEORY.tex](../theory/EVQ_COSH_THEORY.tex)，§Exact Kernel and Broadband Projection。

旧版取距离先验 `p(Δ)=1/(Δ log L)`，以

\[
K_{\cos}(\omega,\nu)=\mathbb E_p[\cos(\omega\Delta)\cos(\nu\Delta)]
\]

为所谓 exact collision kernel，再用 `αδ(φ−ψ)+β min(φ,ψ)` 近似它。

但真实单个 RoPE pair 的 logit 项为

\[
f_k(X,W,\Delta)=C_k(X,W)\cos(\omega_k\Delta)+D_k(X,W)\sin(\omega_k\Delta).
\]

这里至少有以下不同层级：

1. 距离先验是否代表任务相关的读取需求；token重复距离、因果pair计数、平均attention距离不是同一个测度。
2. cosine-only 核是否表示完整位置对象；它没有包括另外三个 sin/cos Gram 分量。
3. 完整位置子空间几何如何进入有符号的内容竞争；白化相关会消去幅度及条件数，不能直接表示训练模型的使用价值。
4. surrogate 对所选几何目标的逼近是否保持最优点，而不仅是矩阵拟合分数较高。
5. 几何最优点如何在训练之后改善部署任务；训练所学的 C/D 系数、跨层状态、softmax 竞争及生成过程都没有被前述静态目标确定。

因此旧文“唯一近似是 broadband projection”只能指其已选定标量模型内部的代数步骤，不能描述从真实任务到 Cosh 的全部假设。

### 1.1 先验曾被反向选择来提高拟合

[3月11日 broadband 检查](../exp/2026-03/2026-03-11_test3_broadband_r2_validation.md)记录了：初始合成先验结果低于目标；token co-occurrence 测度得到约0.65；GPT-2平均attention测度在所列配置得到约0.90；随后扫描先验、base、长度、网格与拟合区域，在24,000个配置中找到886个 `R²_mid>0.99`。

这些结果可以说明某些条件下近似较好，不能说明真实 RoPE 需求恰好服从为了提高拟合而选出的条件。GPT-2 的 attention 统计也不是目标 RoPE 模型的任务敏感距离分布。该文件是历史报告，本轮没有重新下载数据或重测这些模型。

### 1.2 Cosh 由目标形式选出，不由拟合结果发现

当前稿件明确写成

\[
J[\rho]=\frac\alpha2\int\rho^2+\frac\beta2\int S_\rho^2,
\qquad S_\rho(t)=\int_t^1\rho(\phi)d\phi.
\]

在单位质量约束下，变分方程微分两次得到 `ρ''=(β/α)ρ`，于是解为 Cosh。存在性、唯一性与闭式是真实数学贡献；但一旦选定常系数局部平方项和 min-kernel，无论怎样拟合 α、β，都会返回 Cosh 族。这不能作为独立证明“真实 RoPE 应当采用 Cosh”的证据。

更高的拟合 R²也不能自动给出任务优化保证。甚至准确优化完整几何 rank，也存在窗口内正交而窗口外周期混叠的表。现有稿件附录的 harmonic-lattice、cosine/full-subspace 排序反例与跨长度反转，已经阻止把“换成更准确的碰撞目标”直接当下一代方法。

来源：[完整 RoPE 分析](../../paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)、[当前数学附录](../../paper-2027/appendix/a1_proofs.tex)。

## 2. τ 的解析缺口已经有直接反证

本轮重跑 [既有 NumPy 核查](../../rebuttal/strong_model_verdict_numerics_20260720.py)，只采用其明确 uniform-prior cosine-Gram 拟合条件下的以下结果，不将之说成所有先验的数值：

| d=64、b=500K | surrogate拟合所得 √(β/α) | 部署规则 d/√L | 比值 |
|---|---:|---:|---:|
| L=2048 | 6.244 | 1.414 | 4.42 |
| L=4096 | 5.704 | 1.000 | 5.70 |

这两个目标点明显不同。关于 Cosh 的 surrogate 最优定理没有解析确定实际部署 τ。

更有决定性的是 Phase16 的真实训练结果。本轮读取本机全部99个 `result.json`，以统一的加权外推 NLL 重算：

| 比较 | 配置均值胜数 | 配对胜数 | 平均NLL差 |
|---|---:|---:|---:|
| 公式 − Geo，3个seed | 7/9 | 18/27 | −0.01331075 |
| 公式 − pilot选出的邻点，另2个seed | 3/9 | 8/18 | +0.02210473 |

与7月24日重分析一致。早期“near-optimal law”的排名混用了不同stage的复合评分和不同seed数；不能继续采用。该结果既不否定 Cosh，也不表明 τ 完全不可选择，而是证明旧公式没有完成可靠选点。

来源：[Phase16旧报告顶部纠正](../exp/2026-03/2026-03-09_phase16_formula_optimality_sweep_results.md)；完整后续分析可读取 `git show main_0726:rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`；本机原始目录 `results/theory/phase16_formula_optimality_sweep_local_m4_wikitext/`。

## 3. 仓库已经试过哪些非几何分配

| 构造 | 已有证据 | 对改版的含义 |
|---|---|---|
| 早期高频Geo、低频多项式尾部混合 | `compute_hybrid_inv_freq` 中有 split/alpha/p/最低频率缩放；它同时改变形状与慢端范围 | 不是 Cosh，也不是固定范围对照；代码存在不等于已验证成功 |
| 高频Geo、低频Cosh的 Hybrid | `hybrid_evq_inv_freq` 固定前r个pair、重分配余下区间；750M/2K/1B-token有旧报告 | 混合方向确实做过，但没有建立普遍优于纯Cosh的结论 |
| deformation-matched exponential | 50.9M、12配置×3seed；与参考Cosh的差为+0.00074 NLL，区间跨零 | Cosh的特殊形状优势没有分离出来；这是必须保留的简单相关对照 |
| phase-isotropy、pair-volume、min-eigenvalue | 固定端点、实际训练小矩阵，存在正负及长度反转 | 完整几何指标也没有成为可靠的任务选表器 |
| 学得的固定范围分配及小结点参数化 | 已有代码和共适应/剂量实验；learned方向可以改变full/tail取舍 | “放开频率学习”已经试过，也不是自动得到更优方法或新颖性 |

代码：[Cosh Hybrid](../../scripts/core_text_phases/run_evq_sweep.py)、[多项式 Hybrid](../../scripts/supporting_eval/eval_niah_recall.py)、[5自由度结点分配](../../scripts/lib/rope/knot_allocation.py)。

### 3.1 早期 Hybrid 的远程检索数字不能当完整生成

[Phase9F报告](../exp/2026-03/2026-03-03_phase9f_50pct_checkpoint_report.md)在750M、seed42、完整训练点记录：8K PPL Geo/Hybrid 为115.010/121.583；20题checkpoint口径的检索诊断为60%/80%。

但最终40题表将 `ret/AR` 分开：8K为50%/0%对62.5%/0%。两边远程AR exact均为0。报告中的整体AR提高来自其他长度，不能说已经得到8K完整生成优势。并且那组实验没有同条件纯Cosh臂，不能据此宣布Hybrid胜过Cosh。

### 3.2 确实有比 Cosh 更好的小范围点，但没有胜出方法

本轮核对了原始 `stageB_base500k_25m/reports/summary.json`：加权OOD tail NLL 为 Geo5.78930、Cosh5.71740、phase-isotropy5.76712、min-eigenvalue5.67675。

min-eigenvalue 在这一格胜过 Cosh，但只有一个seed、四个评价anchor及25M-token训练。另一base/预算的Stage A所有候选都输Geo；两个stage改变了base和训练预算，不能把它们当纯base消融。该记录是可复用线索，尚无足够理由把min-eigenvalue直接升为下一代主方法。

来源：[M4完整结果](../../paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md)、[factorial表](../../paper-2027/tables/table_m4.tex)。

## 4. PPL 改善而模型能力受损：本轮逐题复核

读取三臂原始QA结果，每臂303题；确认 example_id、prompt SHA256 和 references 一致，并重算宏平均F1及exact。

| 相同303题 | Base-Native | Native-LoRA | EVQ-LoRA |
|---|---:|---:|---:|
| task-macro F1 | 23.09% | 21.10% | 11.26% |
| 完整字符串exact | 33/303 | 25/303 | 4/303 |
| ≤8K子集宏F1（194题） | 49.46% | 50.88% | 17.49% |
| >8K子集exact（109题） | 0/109 | 0/109 | 0/109 |

子集任务比例不同，不能跨子集解释成纯长度曲线；同子集配对仍有效。同一批adapter之前的temporal文本PPL Native-LoRA/EVQ-LoRA为8K 6.817/10.068、16K 108.958/24.068、32K 991.475/127.911。

这不是一般意义上的“PPL指标没用”：它准确度量了所测文本的平均条件概率。错误是把它当作指令遵循、目标绑定及自由生成成功的替代。7月50-step检索micro-tune又把训练loss降得很低，却仍未修复真实16K的S-NIAH/KV exact。来源依赖干预说明EVQ确实使用了远端信息，但没有识别最终失败究竟由哪个内部环节主导。

原始数据：[QA summary](../../results/qa16k_three_arm_s42_20260715/summary.json)、[EVQ逐题输出](../../results/qa16k_three_arm_s42_20260715/evq_lora.json)。解释与协议：[QA报告](../exp/2026-07/2026-07-15_lora_qa16k_three_arm_results.md)、[检索转换报告](../exp/2026-07/2026-07-14_lora_retrieval_conversion_probe.md)。

所以当前Cosh问题不是“从来没做下游”，而是已有部分下游失败、部分早期无效测量，尚未形成与醒目PPL收益相称的稳定能力证据。BM自然QA属于另一种构造，不能替Cosh填补该缺口。

## 5. YaRN 主实验与我们的短 LoRA 并非相同难度

本轮读取 [YaRN v2 §§4.1–4.2](https://arxiv.org/html/2309.00071v2) 及[作者训练代码](https://github.com/jquesnelle/yarn/blob/master/finetune.py)。论文从原生4K Llama-2出发，在64K真实序列上训练400步、global batch64；128K版本继续在64K数据训练200步。64K→128K是相对于适配长度的外推，原生4K→64K这一段已经接受长序列训练。前400步的名义token预算约1.68B。官方当前代码默认直接优化全模型参数，LoRA需显式开关；代码也支持LoRA，因此不能声称YaRN在任何设置下都必须全参。

我们7月300步协议是rank64 Q/K/V/O LoRA、batch2×accum4、max length8192。仅用steps接近作比较会忽略更新参数范围、真实长度、数据及token预算的差异。核心错误是未经证据便要求原窗口内短适配同时修复全局换表、保留原能力并泛化到更远距离。

“OOD训练”在这里应理解为超出原模型原生窗口的真实长度。经过频率缩放，部分相位被映射回已见范围；不宜将所有频率的训练相位都称为完全OOD。

另一方面，现有结果没有在相同曝光和预算下分离LoRA秩限制与长度曝光不足的因果份额，故不能推导“LoRA永远无效”或“换成全参必然成功”。低成本虚拟position gaps可用于某些相位诊断，但不自动复现真实长输入的干扰数量、softmax分母及各层隐藏状态。

## 6. 为什么外推会改善：已经知道什么，仍不知道什么

**已证实：**固定端点的151.9M三seed比较建立了内部形状的真实行为效应；范围重设会反转排序；权重×表交叉建立共适应；某些Cosh模型有远程来源依赖；这些都不是纯拟合假象。

**未证实：**减少几何碰撞是Cosh外推收益的主导原因；Cosh比其他形状更适合学习；τ规则能预先选出较好点；一个静态几何分数能判断下游胜负。

三种仍可区分的解释是：

- 有效位置方向增加，从而让模型学得更好的距离区分。
- 分配改变了训练中形成的有符号位置偏置/频率使用方式，外推行为由这些系数与相位共同决定。
- 平均外推PPL主要因局部预测更稳定或错误远端注意力更少而改善，必要的目标绑定/生成链仍未恢复。

当前证据允许这些效应同时存在，没有完成贡献分解。对静态kernel作更漂亮的近似不能自动区分它们。

一个直接数学提醒：固定内容系数时，频率变化的相位差是 `δω·Δ`，且 `|exp(iωΔ)−exp(iω'Δ)| ≤ min(2,|δω|·|Δ|)`。两个表在短距离上近似相同，不表示在更远距离仍相同；这个界本身也不决定哪一个表更好。

## 7. 更优秀非几何网格的研究问题应如何落地

分配必须与训练后行为一起评价。可以写成

\[
W_\Omega=\mathcal T(\Omega,D_{\rm train},\mathcal U,B),\qquad
\text{evaluate }(W_\Omega,\Omega)\text{ on held-out capabilities and LM risk},
\]

其中 `U` 是可更新参数集合，`B` 是实际预算。这是问题定义，不是新算法。只在固定W上优化某个proxy，或者让不同表接受不同适配条件，都不能直接回答这一问题。

对EVQ改版最有价值的比较应做到：

1. 新网格与Cosh在同条件下有明确质量或成本优势；只赢Geo不足以证明升级。
2. 简单匹配非几何对照也在场，以区分一般的形变收益与新构造的增量；已有同条件结果可复用。
3. 先证明适配后的模型在适配长度内具备目标能力，再判断超出适配长度的泛化；短端能力与远端能力分开报告。
4. 在实际效果上验证构造所声称的机制，而不要求先完成全局最优理论，也不拿proxy过关代替任务。

LeRoPE已经构成实质近邻：[原文 §§3、6–7](https://arxiv.org/html/2607.10134)包含逐频率学习、固定已学网格重训及与YaRN组合。它的裸外推会更快退化，并通过主导频带的符号翻转解释部分现象。因此“学一个非几何表”或“再叠YaRN”不够成为新贡献；需要新的、可验证的选择依据或真正更好的质量—成本关系。本轮只核对这些直接相关段落，没有作穷尽查新。

当前研究决定：保留Cosh及已有受控正结果作为基线；放下它是特殊最优族的预设，也不继承零训练和原窗口短LoRA的硬限制。下一代方法尚未找到，不能把本次复核或当前论文重包装当成已完成EVQ升级。

## 8. 本轮验证边界

- 新做的是CPU重分析和旧数学脚本复算，没有新的模型结果。
- Phase16与QA的关键数值由本机原始结果直接重算；M4由保存的实际运行summary核对。
- 早期Hybrid数字使用历史报告，未重新生成，也没有将它的检索诊断改称AR成功。
- NumPy核查的τ数字绑定其uniform-prior、离散网格和拟合约定，不外推到所有先验；脚本其它近似重现项不作为本报告的逐位复现声明。
- 旧文中的“最优”“验证通过”“下一步必须运行”等表述是待核对的历史内容，不是本轮结论或执行授权。
