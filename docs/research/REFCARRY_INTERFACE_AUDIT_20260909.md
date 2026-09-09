# RefCarry：地址接口的独立数学与因果核验

本记录核验用户提供的 **Position Is a Retrieved State / Reference-Carrying Attention**。它是继续研究的证据，不是已验证的新方法或论文结果。当前只使用原始文献、实现与本地 CPU；没有启动 GPU、下载模型或排入训练队列。

## 当前判断

值得继续研究的是“模型是否能组合内容寻址与参照相对读取”。现有材料尚未证明一个现代 hybrid 架构特有的 reference 丢失瓶颈，也没有证明所给 group-moment 接口具有独立新意或生成优势。其精确代数恒等式成立；概率语义、最小性条件及 oracle 的识别范围必须修正。

可复现 CPU 检验：`python3 experiments/native_sparse_position/aux_refcarry_math_checks.py`。脚本无第三方依赖；以下所有数值仅为有限维数学例子。

## 1. 精确的对象是平均 logit

固定 query、所有 reader keys、原生频率和地址分布。令 `ell[a,j] = qᵀ R(p_j-a) k_j`，则

\[
\bar\ell_j=\mathbb E_\mu\ell_{a,j}
=(M_\mu q)^\top R(p_j)k_j,
\qquad M_\mu=\mathbb E_\mu R(a).
\]

100 个随机 CPU 例的最大误差为 `3.885780586188048e-16`。该式没有 softmax，也没有 value readout。若 `P_a=softmax(ell[a,:])`，实际 RefCarry 的读取分布满足

\[
P_{\rm moment}(j)
=\frac{\prod_a P_a(j)^{\mu_a}}{\sum_t\prod_a P_a(t)^{\mu_a}},
\qquad
P_{\rm marginal}(j)=\sum_a\mu_a P_a(j).
\]

前者是归一化的加权几何平均，后者才是先按各参照读取、再对不确定参照边缘化。二者都是可定义的算子，不能混称为精确保留同一种读取。

**直接反例。** 一个 RoPE pair，频率 π/2，参照位于 0 和 1，query 为 `(1,0)`。选合法 post-RoPE keys `(10,-10),(-10,10),(1,1)`，得到两个参照的 logits `(10,-10,1)` 与 `(-10,10,1)`。每个参照都偏好前两项之一。等权 moment 的 logits 为 `(0,0,1)`，却以 `0.5761169` 的概率选择第三项；真实读取混合给第三项的概率仅 `0.0001233946`。

**更强的状态碰撞。** 同一频率下，`mu_A=(delta_0+delta_2)/2` 与 `mu_B=(delta_1+delta_3)/2` 的全部该频率一阶 moment 都为 0。固定 keys `(4,0),(-4,0),(0,0)` 时，实际边缘读取分别为 `(0.49101,0.49101,0.01798)` 与均匀分布，总变差为 `0.3153530`。因此该 moment 对一般边缘读取不是充分状态。

若所有有质量的参照对同一目标具有正 logit margin，取平均仍保留该目标的 margin；上述反例不否定这个受限情形。应先定义真实任务需要的是单参照、多个参照的共识，还是多个参照的边缘读取，避免以定义出来的算子代替能力需求。

## 2. 概率 simplex 上的最小性需要中心化

把位置特征作为列组成 `Phi in R^(2K×N)`。地址分布满足 `1ᵀmu=1`，可变方向是 `H={delta:1ᵀdelta=0}`。当线性 sketch 后允许任意解码、且归一化常数已知时，所需最小维数为

\[
r_{\min}=\operatorname{rank}(\Phi|_H)
=\operatorname{rank}\begin{bmatrix}\mathbf1^\top\\\Phi\end{bmatrix}-1.
\]

证明：若存在 `delta in H` 同时满足 `S delta=0` 和 `Phi delta!=0`，可从 simplex 内点沿 ±delta 作足够小扰动，得到相同 sketch 而不同目标 moment。故 `ker(S|H)` 必须包含于 `ker(Phi|H)`；由维数公式得到下界。取中心化特征行空间的一组基，并用已知归一化常数作 affine 恢复可达到该下界。这是有限维线性代数结论，不应包装成未经查新的新定理。

原文无条件写 `r>=rank(Phi)` 不成立：两个地址 0、1，频率 1，未中心化的 Phi 秩为 2，但仅传 `mu_1` 一个实数即可用 `Phi_0 + mu_1(Phi_1-Phi_0)` 恢复全部 moment。CPU 恢复误差小于 `1.2e-16`。对足够多的普通位置和满秩频率表，中心化秩仍可能是 2K，不能把这个修正夸大为所有实际配置都少一维。

若任务只有一个选定参照，传一个地址整数并在 reader 重建 `R(a)` 即可。这个离散地址族不需要恢复所有概率分布，不能拿全 simplex 上的线性 sketch 下界证明多维 moment 在该任务中的必要性。

## 3. 架构与已有接口必须直接对照

[TAPE §3.2 Eq.7](https://arxiv.org/html/2501.00712v1)已经用 attention 权重加权位置矩阵，再以内容条件化变换更新位置。RefCarry 的 `sum A exp(iωp)` 与这个更新有直接计算重叠。冻结 key 的原始坐标、只跨指定层传给 query 是更受限的部署设计；这一限制是否带来新的能力成本优势须另证，不能只把近邻描述成“更新所有位置场”后宣布地址矩是新接口。

[RePo §3](https://arxiv.org/html/2512.14391v1)从 hidden state 预测可微连续位置，并用于位置变换。对单参照任务，应同时比较廉价的 scalar-pointer/query-rebase 构造；“传 scalar”不自动等同于完整 RePo 实现，但它是必要的构造对照。

[Neural Turing Machines §3.3.2](https://arxiv.org/pdf/1410.5401)更早就把上一读取权重作为地址状态，与新的内容寻址权重插值后作位置平移。这不是现成的预训练 hybrid LLM 基线，也没有覆盖 RefCarry 的具体实现，但“先按内容读、保存读取位置、再相对该位置读取”作为一般计算接口已有明确先例。论文必须说明现代模型中新的困难与解决价值，不能把这个接口概念本身当作首次发现。

[Qwen3.5 官方实现的 DecoderLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)在 GDN 或 full-attention 更新之后均保留逐 token 残差加法，MLP 后再次残差相加。第 19 层 query 的表征沿深度到第 23 层仍有该通路。沿序列的 recurrent 压缩不能直接推出这段沿深度的表征必然丢失。单个 affine map 不能表示任意地址与 query 的乘法，也不能证明含门控、MLP 与残差的真实 bridge 无法表达它。

“两个输入具有相同压缩状态 C”的不可区分性命题，仅对下游全部相关输入相同的 decoder 成立。真实 reader 还看到 K/V 字典及其它缓存；它们若不同，`C(X)=C(X')` 本身不构成完整模型碰撞。需要明确固定的 side information。

## 4. 目前 oracle 不能识别的内容

**Gold-reference 是一种带正确地址的 phase 干预，不是恢复上界。** 预训练 reader 未必学过这种 query 原点替换。它失败可否定“当前 checkpoint 上这个接口直接有效”，不能单独否定参照信息有用；它成功也可能修复了 writer 原本没找到参照的问题，而非三个 GDN 层造成的 handoff 丢失。应把已找到参照、参照跨层保留、下游使用参照分开定位。

**目标在参照之前不阻止答案经参照传递。** 因果 mask 禁止 target 看到未来 anchor，但允许 anchor 看到过去 target。每条记录只有一个 value、相对 offset 固定时，anchor 可以携带该 value，形成普通关联检索。把 offset 在 query 中晚揭示、让同一 anchor 对应许多可能目标，可减少这个特定捷径；不能把它称为彻底证明无旁路。

**Writer-value-zero 与 No-target-KV 不是完整路径切断。** 只清一个 writer 的语义输出不清更早层或残差中的内容；只删除 reader 的原始 target KV 不删除后续 token 已经携带的答案。失败也可能来自破坏正常计算，不能机械作为整个方向的 kill 条件。

**更有识别力的 value 干预：** 构造同长度、同格式的 donor/recipient 目标值。在指定 reader 处保持 recipient 的 query、keys、所有非目标 values、上游状态不变，仅替换目标 value 张量。先用 Gold-target 验证 donor value 能被完整解码，再检验 reference 干预的答案是否随目标 value 改变；做双向交换和同规模错误位置交换。这只证明指定 value 路径的因果作用，不等同于自然样本总体收益。相比删除 KV，它保留候选集合和 logit 归一化。具体 token 对齐与后续解码干预区间必须在运行前固定。

**地址平移不普遍保证目标按斜率 1 移动。** CPU 例仅改变 anchor 0→1、保持单位 query 范数，合法固定 keys 仍连续选择位置 2。内容竞争可以主导位置作用。平移扫描是可检验的任务预测，需要近似平移同构的内容/结构条件，不能从群表示恒等式直接推出通用数值门槛。

**主终点必须保持完整生成加 EOS。** 提案中 Gold-target 的“准确率提升或 NLL 降低 0.75 nat”不能用后半项替代用户要求的能力恢复。NLL 只能帮助定位，不能令主 oracle 通过。

## 5. 对继续工作的具体影响

不启动提案中的 13 GPU 小时训练。当前本地阶段先完成一个明确判断：在实际 read-to-read 任务中，writer 是否已经获得可用参照，而 native reader 的失败能否由仅补充这个参照、保留其它计算条件的干预恢复。该问题尚未有真实模型证据。

若之后获得实验授权，先复用已有 Qwen3.5 checkpoint 的 Native、Gold-target、Gold-reference 和匹配错误参照；只在出现完整恢复后，用 target-value 交换等干预辨别路径。没有必要在第一次资格判断前把八臂、三种子和自然任务全面铺开。Gold-reference 失败应如实归类为接口失败，避免据此虚假排除一整个机制。

若单参照接口通过，应优先与 scalar-pointer/query-rebase 和忠实 TAPE 对照比较；若核心是软参照，必须明确采用几何聚合还是边缘化读取，现有精确公式不能替后者背书。只有真实能力与成本优势支持时，才有理由构造完整论文。此审查本身没有完成原研究目标。
