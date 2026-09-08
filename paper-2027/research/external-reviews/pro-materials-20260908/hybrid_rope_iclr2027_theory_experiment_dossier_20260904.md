# Hybrid-RoPE / EVQ-Cosh：单表工作点、生成迁移与 ICLR 2027 研究执行档案

**版本：2026-09-04 · 审查执行稿 v1.0**  
**适用设备：用户报告的 4080 Super 32GB、5090 32GB；以服务器实际显存为准**  
**主要模型：OLMo-2-0425-1B-Instruct；第二模型族用于外部有效性，不用于挑选最好看的结果**  
**文件性质：理论分析、证据审计与预注册实验设计，不是已经完成的新实验报告。**

> 核心决策：不再搜索频率曲线。先把已经存在的一张表变成可独立复核的工作点，再用一个固定、受 Native 功能约束的适配协议，验证原有短程任务能力能否迁移为完整长程生成。论文的核心仍是固定 support 下 allocation 的独立作用及其与 learned computation 的关系，而不是把论文改成另一个通用 LoRA 方法。

---

## 阅读导航

- **第 0–2 节：** 结论、上一版需要修正的地方、失败事实与证据等级。
- **第 3–5 节：** 两大问题的核心理论；哪些能严格推出，哪些不能。
- **第 6 节：** 相关工作与本项目的真实剩余空间。
- **第 7–10 节：** 固定实验矩阵、数据、损失、训练与选择规则。
- **第 11–13 节：** 因果诊断、统计、资源预算与运行伪代码。
- **第 14–16 节：** 论文主线、扩展思考、截止日期与最终 Verdict。
- **附录 A–D：** Codex 审查清单、预注册配置、内部资料与公开参考文献。

给 Codex 的最低阅读范围：第 0–2、7–13 节和附录 A–B；不能只截取某一段 LoRA 配置就开始训练。

# 0. 最终决策与不可混淆的目标

## 0.1 两个问题分别是什么

**问题 Z：冻结权重的单表工作点。** 对一个指定 checkpoint，所有短请求和长请求都使用同一张有序频率表、同一个 gain、同一个标准 attention 算子。目标不是保证任意长度有效，而是找到并验证一个同时具有可解释 Native 代价和真实 long 收益的工作点。当前要验证已有候选，不再发明候选。

**问题 T：少量训练后的生成能力迁移。** 固定上述位置底座，仅允许少量模型参数适配。训练序列及 position IDs 不超过 16K，使用同一套参数在 Native、16K 和更远长度评测。成功必须体现为对证据有因果依赖的完整答案、合理终止及 Native 保留；不是 teacher-forced loss 好看。

两条路线必须有独立结论。T 成功不追认 Z 已经无损；Z 的 NLL 改善不替代 T 的生成成功。

## 0.2 唯一推荐的研究路线

**已有单表冻结 → 独立评分与配对诊断 → 全线性低秩残差适配、实际部署函数上的 Native 约束 → 盲测生成迁移。**

不把 head-selective 改造设为新的主方法，不做新的 `m_k` 搜索，不把当前表替换成 YaRN 后再继续宣称自己的 allocation 贡献。YaRN 与 MrRoPE-Pro 是外部对照，不是根据本轮结果选择的初始化。

训练使用所有 Transformer blocks 的 Q/K/V/O 和 MLP 线性映射的低秩更新，固定 rank 16。这个选择是对“尚未唯一定位的寻址—写入—后处理瓶颈”保留必要表达范围的预算决策，**不是最小充分秩定理，也没有证据可给它标注一个客观成功概率。**

## 0.3 本轮必须分开的四种“成功”

| 结论等级 | 要求 | 禁止升级成 |
|---|---|---|
| 几何事实 | 固定 support、K、算子，只改 z；几何量或函数发生变化 | 通用最优 allocation |
| 单表有效工作点 | 同一表在短、长条件下完整报告收益与损害 | 无损 Native、任意长度可用 |
| 合格联合工作点 | 同一表/adapter 通过预设 Native 保留标准，且完整长程输出改善 | 所有 Native 任务零遗忘 |
| 真正迁移 | 未见语义实例和模板上，远端证据驱动完整生成；更远长度另报 | 训练过一种 QA 后获得所有 long 能力 |

不能在结果出来后放宽指标，把前一等级改叫后一等级。

## 0.4 证据访问边界

本稿依据用户提供的目标、历史对话、Library 中的结果报告、交接档案、上传论文与审稿文本，以及截至 2026-09-04 核对的公开资料。GitHub connector 对 `hybrid-rope` 的已安装仓库检索没有返回可读取仓库，因此**本稿没有核验当前 live repository 的代码、服务器或最新原始 JSON**。

截图显示 Codex 正在建立 `generation_contract.py`，不代表本稿已审查该实现。代码与数据的最后核验由有仓库访问权的 Codex 按本文完成。

# 1. 对上一版方案的实质修正

## 1.1 保留的结论

Native-only 信息不能唯一识别真实 long utility；端到端输出 KL 比纯频率距离更有资格定义 Native 功能变化；head sensitivity 不是 head benefit；局部 Fisher 不是全局最优证明；恢复 evidence likelihood 不等于恢复生成。这些结论继续成立。

## 1.2 必须纠正的七件事

**第一，同一请求内静态不等于所有请求同一张表。** 上一版按请求选择不同 scale、短请求切回原表，只满足 request-static。当前主实验使用更强的 global-single-table 定义：一个模型实例、一个频率文件 hash、一个 gain，在所有评测长度不变。

**第二，Native 约束必须测实际部署函数。** 若部署使用表 ΩZ，约束应比较 `student(weights+adapter, ΩZ)` 与 `teacher(native weights, Ω0)`。不能在 Native replay 时给 student 换回 Ω0，再称“同一模型保住 Native”。

**第三，位置替换本身可能已经产生 Native 损伤。** adapter 零初始化不意味着总 KL 为零。相对于原模型的总 damage 在初始化处一般有常数项和一阶项，不能直接套用以零 damage 为中心的纯二次 Fisher 近似。

**第四，不能把已有失败简化成“以前只训练了 Q/K”。** 旧档案还记录了 Q/K/V/O、稀疏 Native-teacher KL、首 token 加权、位置蒸馏、读出修补和后处理重排的失败或不足。本轮必须有明确的新变量，而不是给旧方案换名字。[I01, I03, I05]

**第五，“100 个样本”必须附带物理长度和 token presentations。** AdaRoPE 的相关设置使用 65,536 的训练长度；Randomized YaRN 会采样更长位置编号；它们不能直接证明严格的、未见更长位置的 16K→64K。[R07, R13]

**第六，不再把四种不同意义的 long loss 堆成默认最优目标。** 本轮主目标采用完整答案 CE 加一个小权重的逐轨迹最差 margin 项；两种合法证据版本分别监督自己的正确答案。Native KL 用于保留约束。不给错误证据搭配原答案，不使用 long teacher 蒸馏作为主要能力来源，也不以 long NLL 选模型。

**第七，不能为了“唯一协议”而伪造唯一理论选择。** rank、学习率、预算和阈值必须固定，但它们是预注册工程选择，不是 checkpoint 推出的常数。实验验证这个选择，不证明它在所有可能参数化中最优。

# 2. 已有证据、失败经历与必须先解决的冲突

## 2.1 证据等级

- **E0：原始可复核证据。** checkpoint/表 hash、行集合、配置、raw generations、日志能一一对应。
- **E1：owner 结果报告。** 指向原始产物，但本稿未重新执行或逐文件核验。
- **E2：聚合汇总或交接。** 用于提出审计问题，不能独立替代 owner。
- **H：理论假设、分析稿、计划。** 不能写成已完成结果。

这些等级描述的是本稿的核验范围，不是断言某份内部结果为假。

## 2.2 关键证据台账

| 记录 | 目前支持什么 | 不支持什么 | 本轮处理 |
|---|---|---|---|
| 固定 endpoints/span 的训练与 frozen 干预 | interior allocation 是独立干预变量 | 某一解析曲线唯一最好 | 保留为论文主干；核查 exact-range 约束与 raw receipts [I06] |
| 同频率多重集合、不同 slot assignment 的退化 | ordered slot-frequency coupling 有实质影响 | 几何频谱本身决定性能 | 增补 joint relabeling 精确等价控制 [I02, I04] |
| log-scaling 比旧 arithmetic movement 更稳定 | 旧公式随 s 的运动饱和是一个真实公式缺陷 | scale composition 推出模型的尺度等变 | 冻结旧表，不再扫 m 或 scale [I01] |
| s4 的 RULER 与 long NLL 收益 | 有值得独立确认的成熟 checkpoint 工作点 | 所有自然 QA 已恢复 | 增加完整输出及 matched near/far 数据 [I01, I02] |
| frequency MAE/RMSE 很小仍有 Native 损伤 | 几何距离不能代替功能兼容性 | 某个新几何指标自然解决问题 | 以实际输出 KL/任务保留为约束 [I02] |
| Q/K LoRA、Q/K/V/O、first-token repair 失败 | 单纯寻址或局部 loss 改善不足 | 所有 PEFT 不可能有效 | 不重跑旧路线；采用第 9 节的明确 delta [I03] |
| task-family supervision 曾恢复部分 QA | 少量监督可以转换部分能力 | 泛化到未见任务族、保住 Native | 本轮设置 held-out family 和实际部署 Native 约束 [I02] |
| contrast decoding / 少量候选重排不足 | 该后处理分支没有可信增量 | 所有解码器理论上无效 | 不把 beam、候选数或系数变成新搜索轴 [I02] |
| head sensitivity 存在差异 | shared intervention 在 heads 上作用不均匀 | 低敏感 heads 就是最优修改对象 | 保留诊断，不升级为主方法 [I04] |

## 2.3 一个必须解决的数字冲突

8 月 31 日 owner 报告给出的 OLMo `log_s4` full-13 RULER，4K/8K/16K 为约 **0.7101 / 0.6671 / 0.5486**。9 月 2 日聚合汇总对同名配置记录约 **0.71397 / 0.66705 / 0.49859**。[I01, I02]

这不能靠“取最新”“取较大值”或平均处理。可能原因包括行集合、checkpoint、表、gain、评分器、版本或汇总方式不同；目前没有材料足以替它们选定原因。

Codex 必须建立以下连接：

```text
数字 -> owner 文档 -> run_id -> result JSON -> raw rows
     -> model revision -> table bytes -> gain convention
     -> prompt/tokenizer -> decoder -> scorer version
```

若无法恢复旧 run 的原始材料，则将其标记为历史 preliminary，在新协议下独立重测。不能把无法复核的差异写进摘要。

另外，9 月 2 日汇总明确区分了本地 owner 与尚未回收的部分远端 JSON/JSONL。后者在回收并核验之前，只能作为线索，不是论文确认结果。[I02]

## 2.4 Native retention 数字不能被误读

若 `PPL retention = PPL_native / PPL_method = 0.875302`，则对应 PPL 约增加 **14.25%**，ΔNLL 约为 **0.1332 nats/token**。这只是按该比值定义计算，Codex 必须先确认指标定义。[I02]

这种工作点可能具有研究价值，但不是“Native 几乎无损”。它也解释了为什么把总 Native KL 预算直接设成 10^-3、同时假设零 adapter 起点可行，会造成协议内部矛盾。

## 2.5 明确关闭的分支

本轮不重新开启：新 cosh dosage / protected curve、one-turn floor、按低 sensitivity 手工选 heads、双路径/混合坐标、length router、目标相位下用 QA margin 优化表、EOS-only 修补、beam/contrastive/rerank 搜索、继续旧任务特化 adapter 谱系。[I03, I05]

关闭理由是本轮的信息收益与证据不足，不是证明整个数学类别永远无效。将来重启必须说明新机制、新证据、与旧失败的唯一差别及可证伪预测。

# 第一部分：两大问题的核心理论

# 3. 问题 Z：冻结权重到底能推导到哪里

## 3.1 真正的研究对象是有标签的频率—内容耦合

对某层某 head，给定该层输入，记未旋转的 query/key 为 q_i、k_j。使用 logit gain a：

$$
s_{ij}=\frac{a}{\sqrt d}q_i^\top R_\Omega(j-i)k_j,
\qquad \alpha_{ij}=\operatorname{softmax}_j(s_{ij}),
\qquad o_i=\sum_j\alpha_{ij}v_j.
$$

按二维 rotary pairs 展开：

$$
s_{ij}=\frac{a}{\sqrt d}\sum_k
[A_{ijk}\cos(\omega_k\Delta)+B_{ijk}\sin(\omega_k\Delta)],
\qquad \Delta=j-i.
$$

A、B 是 checkpoint 与输入内容决定的系数，不是可随频率任意重排的常数标签。[R01]

因此，同一个频率 multiset 放到不同 slots，一般形成不同函数。低频的几何冗余不等于对应 learned channels 没有语义作用；成熟模型中低频承担内容匹配的机制证据尤其不允许把 slow bands 一概当作可免费改动。[R02]

给定 q/k 的单层计算只是条件化表达。修改早层 RoPE 后，后层 q/k/v 也会变化。只记录一次原模型 activations 再计算 bilinear distortion，不能代表完整网络的非线性响应。

### 单表的物理坐标与 pure-z 控制

对按频率递减排序、K>1、非零 span 的正频率表，定义

$$
x_k=-\log\omega_k=a+Rz_k,\qquad
a=\min_k x_k,\quad R=\max_k x_k-\min_k x_k.
$$

固定 a、R、K 后，geometric interior 为 z_k=k/(K−1)，剩余 K−2 个内点可以改变。这里必须用**实际采样到的最快和最慢频率**，不能把名义 base 的某个幂直接当成没有采样到的端点。G 对照就在 Z 的实际 endpoints 上重新等距放置内部点，保持算子与 gain 不变。[I06]

`frequency-only permutation` 则保留频率 multiset、改变 slot 标签；它可以离开保序构造子类，是另一项机制干预，不能与“排序后相同的 z 直方图”混淆。固定 support 的因果对比与任意 post-hoc scaling 也要分开：后者可能同时改变 a、R、z。

## 3.2 一个重要的对称性控制：joint relabeling

令 P 只在同一 head 内置换完整二维 rotary pairs，且

$$
R'=PRP^\top,\quad q'=Pq,\quad k'=Pk.
$$

则严格有

$$
q'^\top R'k'=q^\top Rk.
$$

这给出两个不同干预：

- **frequency-only permutation：** 只改 R，不改 q/k；真实改变内容—频率绑定。
- **joint permutation：** 同时改频率与 Q/K 输出坐标；只是重新命名内部坐标，函数应等价。

实验上须同步变换 Q/K projection 的输出 rows、相关 bias，以及所有按该坐标作用的 norm 参数；OLMo 的 Q/K normalization 不能漏。实现使用 split-half 还是 interleaved pairs 也必须先确认。若变换跨 head，V/O 的对应关系另需处理；本轮不跨 head。

这是已知线性代数对称性的应用，不把它包装成新数学定理。价值在于：它能排除“置换代码破坏了张量布局”“任意打乱都会坏，所以说明不了什么”等替代解释，明确 ordered coupling 的因果对象。

该控制含坐标重参数化的权重修改，因此不是部署零训练方法；它是等价性控制。

## 3.3 为什么 frequency movement MAE/RMSE 不够

二维旋转有精确关系：

$$
\|R_{\omega'}(\Delta)-R_\omega(\Delta)\|_2
=2\left|\sin\frac{(\omega'-\omega)\Delta}{2}\right|.
$$

因此单 pair 的 bilinear 变化受 q/k 能量、距离与相位共同决定。频率差小，乘上大距离后未必小；频率差大，也可能在某些距离发生相位回返。多个 pairs 的贡献还有增强与抵消。

更关键的是：即便 attention-logit 变化很小，也可能恰好跨越一个低 margin 的答案决策边界。反之，很多大的内部变化可能被后续层吸收。频率 MAE 既不包括内容系数，也不包括输出决策裕量。

这解释现象，不自动产生一个通用最优的新相位距离指标。

## 3.4 Long benefit：哪些对象可识别，哪些不行

已知证据集合 E 时，定义

$$
\Gamma_E=\log\sum_{j\in E}e^{s_{ij}}
-\log\sum_{j\notin E}e^{s_{ij}}.
$$

证据 attention mass 精确为 sigmoid(ΓE)。这是有意义的 attention competition 指标，但不等于完整任务效用：E 本身来自任务，且正确 aggregation 后仍需绑定、推理、词汇输出与终止。

完整效用至少是

$$
U_{\mu_L,\mathcal R}(x)=
\mathbb E_{(C,y)\sim\mu_L}
\mathcal R\big(\operatorname{Decode}(p_{\theta,x}(\cdot|C)),y\big).
$$

Native checkpoint、Native 文本和目标长度不能唯一决定 μL、正确标签以及评价规则。两个长程分布可以有相同的所有短程可见边际，却要求不同的远端依赖。没有 long outcome 或额外任务先验，就不能辨识哪个是实际部署世界。

**所以，真正的 long optimum 在当前信息条件下欠定。** 这不意味着不能设计合理的先验或证明条件定理，而是不能把 stable rank、Gram condition number、phase coverage、最小频率间距中的任意一个换名为“真实 long benefit”。

## 3.5 Native compatibility 的合适对象

给定 Native prefix 分布 μN，定义

$$
D_N(x)=\mathbb E_{c\sim\mu_N}
D_{KL}(p_{\theta,x_0}(\cdot|c)\|p_{\theta,x}(\cdot|c)).
$$

该 KL 衡量函数偏离，而不是所有偏离都等于性能损伤。因此仍需独立的 Native NLL、完整任务输出和终止指标。

| 对象 | 有资格回答的问题 | 主要缺口 |
|---|---|---|
| Q/K bilinear perturbation | 当前输入下，slot 变化如何进入某层 logits | 后续表示变化与输出决策 |
| attention-logit distortion | 分数竞争变化多少 | 行常数平移对 softmax 无效，普通 MSE 会误罚 |
| attention KL / Fisher | 当前 attention 分布变化多少 | value、W_O 与后续 computation |
| **端到端 Native 输出 KL / Fisher** | **原模型实际输出函数变化多少** | 校准分布外、有限样本与 greedy 稳定性 |
| 真标签 loss Hessian | 特定标签损失的二阶曲率 | 可不定，不等于 KL 的功能度量 |

原模型点的 KL 局部展开为

$$
D_N(x_0+\delta)=\tfrac12\delta^\top H_N\delta+o(\|\delta\|^2),
$$

$$
H_N=\mathbb E J_x^\top[\operatorname{diag}(p_0)-p_0p_0^\top]J_x.
$$

这是输出 Fisher 的 pullback；相关条件下与 generalized Gauss–Newton 对应。它不是经验标签梯度 outer product 的无条件等价物，也不是 full loss Hessian。[R16]

## 3.6 自然梯度到底什么时候是最优结构

给定合法、可微的 U，在局部问题

$$
\max_\delta g^\top\delta
\quad\text{s.t.}\quad \tfrac12\delta^\top H_N\delta\le\epsilon
$$

且 HN 正定时，解为

$$
\delta^*=\sqrt{\frac{2\epsilon}{g^\top H_N^{-1}g}}H_N^{-1}g.
$$

若 HN 奇异，而 g 在 ker(HN) 上有非零投影，二次近似可能无界；伪逆会删除这个分量，不会神奇修复原优化问题。必须加入有限 trust radius、结构约束或真实非线性 KL 检查。

只有 g=0 且 benefit 的首个有效项确实是二次型时，才谈广义特征问题。全局问题的 KKT 仅是满足正则条件下的必要条件之一，非凸性不会消失。

**本轮执行不要求估计完整 Fisher 或运行自然梯度优化器。** 这些公式负责限定理论含义；没有可识别 gL 时，花 GPU 精算 HN 也不会自动产生 long 最优表。

## 3.7 Shared / headwise：自由度多不等于应当先扩张

若允许 layer/head/group 变量，自然解一般是 dense 的。稀疏 head selection 通常需要显式的修改成本或 group sparsity 约束；低 Native sensitivity 只是代价的一部分，不代表能带来 long benefit。

GQA 下，要保持共享 KV cache，相关频率自由度必须满足 KV-group 约束；不能在 query heads 上随意独立变换，同时把同一份已旋转 K 当作所有 heads 的匹配坐标。AdaRoPE 的实现正是 group-level frequency、head-level scaling。[R07]

但当前论文的主张是 shared single-table 的能力与边界。扩张成每头表会改变研究问题，增加选择偏差，也进入已有工作的核心空间。本轮不做这项扩张。

## 3.8 不要把几何恒等式升级成行为上的“守恒定律”

固定 trace 的 positional Gram 可以有

$$
r_2(\Gamma)=\frac{2K}{1+(K-1)\bar c}.
$$

它解释某个分离先验下的 effective dimension；不推出“改善 long 就必须同等损害 Native”的水床定理。两个窗口、两种数据分布及非线性网络不是同一个固定 trace 优化问题。

同样，慢频 subspaces 趋近 span{1,Δ} 不证明这些 learned slots 可免费移动；完整一周期不是学习可用位置特征的必要条件；有限维 exact scale-equivariance 的谱障碍不给出 4×、8×、16×这样的工程上限。[I06, R02]

若写尺度不可兼得命题，应明确 s>0、s≠1、连续有界群等条件：A 与 sA 相似且有限非零谱无法对正倍率反复缩放闭合；有界性再排除非零幂零生成元。s=-1 可以有反射共轭，不能遗漏。该论证是背景性限制，不是本轮独立投稿的“新大定理”。

## 3.9 Zero-training 的最终可执行答案

选择已经存在的、按规范可复现的 `log_s4` 单表作为固定 witness。它承载的是一个明确的结构先验与历史工程结果，不是由 Native 信息唯一推导出的 task-independent optimum。

原来的 m 向量主要由 Native 位置基、K、窗口和数值定义构造；没有读取真实 W_Q/W_K 或输出函数的构造，不能称为 checkpoint-weight-derived。两个这些 profile 相同但 learned weights 不同的模型可以得到同一 m，却有不同的 Native damage。功能检查负责验证这种兼容性，不把它倒写成 m 已经包含权重信息。[I01, I03]

本轮只验证它，不用新 long 结果选择频率、gain、head mask 或 scale。若历史表曾受开发 benchmark 影响，如实标为 method-development；新留出样本提供的是前瞻性确认，不能追认历史选择为完全无标签。

# 4. 问题 T：新增自由度究竟应该解决什么

## 4.1 不是“更多位置参数”，而是补足不可达的决策方向

给定某层输入、固定 values，任何 attention 权重重分配都满足

$$
o_i\in\operatorname{conv}\{v_1,\ldots,v_n\}.
$$

写入 residual 的方向受 W_O 作用后的集合限制。若证据相关信息没有编码进这些 directions，或后续冻结计算无法将它转成正确词汇决策，单纯把 attention 更集中到正确 token 上仍可能失败。

这是条件化的单层可达集结论。改变早层 Q/K 可能间接改变后层 values，所以不能把它升级成“整个深层模型一旦某次 oracle routing 失败，就证明所有位置干预均无效”。

可操作的瓶颈表述是：

> 对已知短程可解的任务，长条件下正确证据的影响已经进入模型，但冻结计算没有把它稳定映射到完整正确的输出轨迹。需要允许内容寻址、信息写入以及后续非线性映射共同调整，而不是继续只优化几何。

是否“已经进入”必须用 counterfactual/source intervention 支持，不能只看 attention heatmap。

## 4.2 参数化的结论与限制

频率-only 主要改距离响应；Q/K 更新直接改内容寻址；V/O 更新增加信息提取与写入方向；MLP 更新允许对新组合的 residual features 做非线性决策修正。完整微调自由度更广，但本项目的小数据与显存预算不支持把自由度增加本身当作成功理由。

**本轮唯一训练参数化：所有 blocks 的 Q/K/V/O + MLP linear layers 的 LoRA，r=16。** 原始参数、norm、embedding、LM head、RoPE、gain 全部冻结。

其理由是覆盖尚未排除的 bottleneck，而不是已证明每个模块都必须更新。LoRA 的理论表达能力结果不保证本 checkpoint 的特定任务只需 rank 16；它较少遗忘也不代表不会遗忘。[R17, R18]

在 OLMo 官方配置 hidden=2048、intermediate=8192、16 layers、普通 Q/K/V/O 投影与三层门控 MLP 线性映射下，可训练数约为 12,058,624，约占 1.485B 的 0.81%。实现以实际模块计数为准，不能把同名模型的估算当作代码审计。[R26]

## 4.3 “最小充分”目前能严谨写到什么程度

设 A_S 是某训练子空间 S 对一组目标 margins 的 Jacobian，b 为需要补足的 margin 缺口。局部充分性需要存在 δφ 使

$$
A_S\delta\phi\succeq b,\qquad
\tfrac12\delta\phi^\top F_{N,S}\delta\phi\le\epsilon,
\qquad \|\delta\phi\|\le r.
$$

这个可行性与 checkpoint、任务、距离分布、budget 都有关，不能仅凭“这是 LoRA”确定。

在 F 正定且忽略额外 trust radius 的局部模型里，最小 Native 二次成本的对偶形式为

$$
\max_{\lambda\succeq0}\lambda^\top b
-\tfrac12\lambda^\top A_SF^{-1}A_S^\top\lambda.
$$

它提供条件化的能力—保留 trade-off 表达。未满足近似误差控制时，局部不可行也不是全局不可行。

本轮不要求构造这个巨型 Jacobian。不能为了“理论最小充分”新增一个昂贵的全参数 Fisher 估计项目。最终可以说“这组低秩自由度在测试条件下足够”，不能说“这是数学上最小的参数集合”。

## 4.4 一条直接连接训练与生成的严格结论

对固定输入、固定表、固定 decoding semantics，设目标输出为 y1,…,yT,EOS。定义 gold-prefix margin

$$
m_t=z(y_t\mid C,y_{<t})-\max_{v\ne y_t}z(v\mid C,y_{<t}).
$$

如果所有 t 都有 m_t>0，则 deterministic greedy 从该输入出发会逐 token 生成这条完整轨迹，并输出 EOS。证明是简单归纳：第一个 token 是唯一 argmax；正确前缀成立后，下一个仍是唯一 argmax。

这条结论有三个用途：

1. NLL 下降、第一 token margin 变好，都不足以保证整条轨迹。
2. 对短答案，监督全答案与 EOS、关注最差 margin，比只修首 token 更贴近目标。
3. 若实测所有 gold-prefix margins 明确为正，greedy 却走错同一轨迹，应先检查 cache、位置、dtype、mask、模板、logit processors 或对齐错误，而不是立刻发明 exposure-bias 解释。

限定：这是针对该输入与规范化目标轨迹的充分条件。自然语言可能有多个语义正确答案；未匹配某一 token 序列不等于语义错误。浮点 tie 和不同 backend 的微小差异应单独处理。

## 4.5 一个更精确的 Native 保留证书：KL 到 argmax 边界

平均 KL 小不能保证 greedy 不变，可以给出一个比泛泛引用 Pinsker 更直接的点态判据。

对 teacher 分布 p，正确且唯一 top-1 token 为 a。固定竞争 token b，考虑所有满足 q_b≥q_a 的 student 分布。最小 forward KL 在

$$
q_a=q_b=\frac{p_a+p_b}{2},\qquad q_j=p_j\ (j\ne a,b)
$$

处取得，因此

$$
\kappa_b(p)=p_a\log\frac{2p_a}{p_a+p_b}
+p_b\log\frac{2p_b}{p_a+p_b}.
$$

取 κ(p)=min_b κ_b(p)，最小值由最高概率竞争者，即 teacher top-2 给出。若

$$
D_{KL}(p\|q)<\kappa(p),
$$

则 student 不可能改变该 top-1。对 teacher 的完整 greedy 轨迹逐 prefix 检验，即可得到该样本完整输出保持的充分证书。

**证明要点：** KL 对 q 的约束最小化；边界约束使 a/b 两个质量相等；其他质量保持 p。κ_b 随 p_b 接近 p_a 而下降。该结论是信息投影的直接推导，不主张新的普适学习理论。

这给了论文一个有用的解释工具：相同平均 KL 下，靠近决策边界的 token 更脆弱。但不要把 `KL/κ` 的均值当作全样本证书；必须逐 token、逐轨迹满足不等式。κ 极小时要报告数值不确定性，不除以一个人为 epsilon 后声称证书成立。

本轮将该量用于少量 Native decision-critical tokens 的诊断，不增设复杂新优化器。

## 4.6 为什么实际部署 Native KL 的起点不是零

令 ΩZ 为固定 retrofit 表。实际约束是

$$
D_N^{total}(\phi)=
\mathbb E D_{KL}\left(
 p_{\theta_0,\Omega_0}\;
\|\;p_{\theta_0+\Delta\theta(\phi),\Omega_Z}
\right).
$$

在 φ=0 时，D_N^total 通常非零。因此

$$
D_N^{total}(\phi_0+\delta)
=D_0+g_D^\top\delta+\tfrac12\delta^\top H_D\delta+\cdots.
$$

其中 HD 也不应无条件冒充原模型点的纯 Fisher。原点局部 Fisher 对的是相同输出分布间的 infinitesimal departure。

应分别报告：

- 表替换带来的初始损害；
- 加 adapter 后相对 retrofit 起点的变化；
- **最终实际部署函数相对原模型的总差异。**

KL 不满足这里所需的加法分解，不能用两个阶段 KL 相加充当总 KL。

## 4.7 为什么选择功能约束，而不是仅靠 EWC / L2-SP

EWC 对重要参数方向加惩罚，L2-SP 保持参数接近原点，二者都具有合理先验，但不会直接检查有限更新后的部署输出。Replay 解决覆盖，teacher KL 定义保留对象，约束优化处理 long–Native 冲突；这三者组成同一个方案。[R19, R20]

本轮不并列叠加 EWC、L2、Fisher、attention KL、hidden MSE 五类正则。低秩参数化已经提供一个结构限制，核心保留约束放在真实输出上。

旧稀疏 Native KL 失败并不证明所有功能约束都无效；它提醒我们，**一组少量 token 的平均 KL 可能遗漏重要 Native 行为。** 因此本轮分层校准、关键轨迹检查和 held-out Native 任务必须同时存在，不能只改一个 KL 系数。

## 4.8 更直接的 transport 条件：要保的是决策方向，不是全部 hidden state

设 LM head W,b 冻结。对同一语义实例、同一个正确答案前缀，h_C 和 h_F 分别为 compact/far 的最终 **post-normalization** hidden states。令 δh=h_F−h_C，词表 logit 差恰为 δz=Wδh。

对正确 token y 与竞争 token v，严格有

$$
m^F_{y,v}=m^C_{y,v}+(w_y-w_v)^\top\delta h.
$$

因此，正确决策被保留的充分条件是

$$
\max_{v\ne y}(\delta z_v-\delta z_y)<\min_{v\ne y}m^C_{y,v}.
$$

另一种较保守的充分条件为：存在行常数 c，使

$$
\|\delta z-c\mathbf1\|_\infty<\tfrac12\min_{v\ne y}m^C_{y,v}.
$$

共同 logit 平移不影响 argmax。对整条正确轨迹逐 prefix 满足这个条件，再由第 4.4 节得到 greedy 生成的充分保证。

这解释一个实际选择：不把长条件下所有 hidden states 的 MSE 压到零。落在 W 的零空间或非关键方向上的巨大变化可能无害；与最强竞争 token 对应的一点变化却可能使答案失败。允许修改 V/O 和 MLP，是让 adapter 有机会修正这些内容到决策的方向；不是已证明它们的具体 rank 足够。

这个式子是已实现 states 之间的精确关系，不是我们已经拥有对更远未见输入的 δh 上界。论文不能用它承诺外推，只能用它定义要恢复的计算对象，并解释相同 NLL/KL 下不同样本的生成差异。teacher 本来错误时，保持其决策也不会变成正确性保证。

# 5. 16K 数据为什么可能、又为什么不保证支持更远长度

## 5.1 需要学习的是不变量，不是长度编号

对于答案不依赖文档排列位置的任务，保持证据关系不变，移动证据和加入无关内容不应改变答案。理想训练结构是

$$
\text{相同关系结构 + 不同距离/干扰} \longrightarrow \text{相同正确计算}.
$$

它不是“删除所有位置信息”：最近、先后、对应顺序等任务本来就需要位置，应有 Native 位置敏感任务作为负控制。

本轮使用共享 weights、共享 adapter、共享表，不向模型提供一个专门的 length-id classifier，不使用 16K 专属 adapter。训练中改变的是证据位置、相对距离和干扰布局。

## 5.2 三种外推不能混在一起

| 外推轴 | 训练后更长输入新增什么 | 训练 ≤16K 的局限 |
|---|---|---|
| 位置/相位 | 更大的 relative separations | 同一表下可能出现未见联合相位 |
| softmax competition | 更多 keys、更多潜在相似干扰 | virtual gaps 不能模拟真实竞争数量 |
| 任务计算复杂度 | 更多跳、更多绑定、更多输出步骤 | 固定关系结构的迁移不保证更复杂算法 |

## 5.3 Softmax crowding 的精确小模型

一个证据 key 分数为 u，N 个等分干扰 keys 分数为 v，则

$$
p_E=\frac{1}{1+Ne^{v-u}}.
$$

维持同样 pE，需要 u−v 随 log N 增长。在此简化模型中，干扰数翻四倍，需要额外 log 4 的 attention gap。

这不是 vocabulary answer margin 的 log 4 定律，也不是“统一 gain 加 log N 必然正确”。真实 heads 可能在选择、平均或多证据聚合，不同功能对 sharpening 的需要不同。[R07]

因此本轮必须使用真实长背景训练，不能只给 2K token 拉开 position IDs 后，把 64K 实体上下文泛化当作已训练覆盖。

## 5.4 严格 held-out 的定义

本轮训练条件：

```text
physical tokens per example, including supervised output <= 16,384
maximum position_id <= 16,383
no virtual positions beyond 16K
one fixed frequency table and one fixed gain
no 32K/64K validation used for method or checkpoint selection
```

公开模型的预训练历史不由我们控制。对配置 Native window 已经是 32K 的 Qwen，16K post-training 后的 32K 结果不能叫“模型一生未见 32K”。必须分别写 post-training held-out 和 pretraining/configuration window。[R27]

## 5.5 固定 16K 工作表去 32K/64K的地位

本轮不根据目标测试长度切换 s8/s16 表。固定原先的 ΩZ 在更远条件下接受盲测，得到的是该表和该 adapter 的真实适用范围。

这种做法牺牲了一些可能的极限性能，但避免把“更远时换了一张更合适的表”混入生成迁移结论。即使 32K/64K 未通过，只要 16K 的完整迁移与 Native 保留成立，仍是独立、明确的结果。

**Verdict：有条件的结构性泛化机会，没有现成的 16K→64K 保证。** 本轮不以最大倍率作为论文是否成立的唯一条件。

# 6. 文献研究：哪些空间已经被占据，哪些仍值得做

## 6.1 与两个问题直接相关的工作

| 工作 | 真正相关的结论/方法 | 本项目不能据此推断的事 |
|---|---|---|
| RoFormer / PI / NTK / YaRN [R01, R03, R04] | RoPE 机制、坐标缩放及实用的扩展初始化 | 默认参数是某 checkpoint 的 Native-aware 全局最优 |
| LongRoPE / LongRoPE2 [R05, R06] | 非均匀频率搜索、扩展训练与短程恢复 | Native-only zero-training；10B tokens 不能叫本项目意义的极少量训练 |
| AdaRoPE [R07] | group/head 级频率和 gain 的异质性有实证意义；有冻结 backbone 的位置参数优化 | 100 个 PG19 样本是在 ≤16K、未见更长位置下训练；相关配置实际为 65,536 |
| LeRoPE [R08] | 联合学习 RoPE 频率能改善训练；频率表不是不可动的常量 | 成熟 checkpoint 可任意换表；我们首次发现频率可学习 |
| FMRoPE [R09] | 频带、base、训练长度的关系是直接近邻 | 只重述 dead-frequency 就有充分 novelty |
| MrRoPE [R10] | mixed-radix 视角、training-free extensions，是比只用 YaRN 更接近的对照 | 本项目无需讨论现有非均匀 scaling |
| Resonance RoPE [R11] | 周期对齐与 PosGen 分离位置困难/生成复杂度 | 每个坐标有某个回访点，就等于整个向量在同一个 Native 位置回访 |
| GeNE [R12] | 训练期随机化 extrapolation scale，有 train-short/test-long 证据 | 300 steps × batch 128 × 16K 的语言建模设置是百样本级训练；按满长计算约 6.29 亿 token presentations |
| Randomized YaRN [R13] | 短物理长度、较大位置范围和 curriculum 可改善长程推理 | 严格 unseen-position；模型的配置 Native windows 也已是 32K/64K |
| PoSE / CLEX [R14, R15] | 物理长度与位置范围解耦、连续长度变换 | virtual target-position exposure 等价于完全未见目标相位 |
| LongLoRA [R21] | 高效长上下文适配与训练 attention 的工程方案 | all-linear r16、冻结 embedding/norm 就有现成保证；其公开方案有训练稀疏 attention 与可训练 embedding/norm 的额外条件 |
| LongReD / LinearARD [R22, R23] | 位置修改后用 Native teacher 恢复能力是已经存在的方向 | 本项目的蒸馏分解本身就是新方法；内部 relation matching 等价于端到端保留 |
| Retrieval Heads [R24] | 一些 heads 对检索存在因果作用 | 对位置干预低敏感的 heads 就最值得修改 |
| Deconstructing Positional Information [I08] | 内容—位置相互作用及特定任务的 head deposit 机制 | 该合成机制自动证明本成熟 checkpoint 的故障位置 |

公开论文的数字来自其各自模型、预算和评分器，不能与本项目数值直接横向相减。

## 6.2 两个特别容易误用的理论结论

**坐标覆盖不等于函数覆盖。** 对每个 k，存在 Native 位置 pk 使某个单坐标值相同，不推出存在同一个 p 使所有坐标同时相同；更不推出带内容系数的 attention 或模型输出相同。整数波长或一周期论证必须保留量词顺序。[R11]

**Self-relation distillation 不等于 cross-QK 或输出等价。** Q'=QU、K'=KV，U/V 分别正交，可以保持 QQᵀ、KKᵀ，却在 U≠V 时改变 QKᵀ；V/V self-relations 也不单独保证与固定 W_O 的兼容。LinearARD 的效率和实证值得重视，但不能从它的 relation 目标推出任意 Native 函数的严格保留。[R23]

## 6.3 本论文仍可能成立的独立空间

不是“首次修改 RoPE 频率”，而是下列闭环：

1. 固定 sampled support、pair count 和 operator，识别 interior allocation 的独立影响；
2. 用对称性控制区分无害坐标重命名与真实 slot-frequency 错配；
3. 显示几何改善、Native compatibility、证据依赖、决策轨迹是不同对象；
4. 在相同数据/预算的比较下，验证固定位置底座能否降低把已有能力迁移成长程完整生成的困难。

第 4 条即使成功，也不能反过来证明第 1 条的特定曲线最优。它给的是可使用性与模型生命周期上的联系。

# 第二部分：固定实验计划

# 7. 实验总图：少量实验，每个只回答明确问题

## 7.1 主矩阵

| 实验 | 核心问题 | 固定对照 | 主要产物 |
|---|---|---|---|
| **E0：独立有效性审计** | 数字、算子、cache、数据、评分是否可信 | identity、joint relabel、独立 scorer | provenance map、CPU/GPU parity 收据 |
| **E1：固定单表确认** | 不训练能做到哪里，Native 代价是什么 | N / Z / G / Y / M | 单表 Native–long 表，compact/near/far/counterfactual readouts |
| **E2：受约束生成迁移** | 同一 table 与共享 adapter 能否恢复完整生成 | N / Z / Y × 无训练/同协议训练；固定 3 seeds | 完整输出、Native 保留、语义实例迁移 |
| **E3：机制补充** | 收益是不是只学了短任务；参数位置是否重要 | compact-only；参数量匹配的 attention-only | 两个预注册的解释性消融，不作为新候选 |
| **E4：外部与更远盲测** | 结论是否跨模型族；长度外推到哪里 | 冻结配置、未见语义实例 | 第二模型族、32K/64K 边界 |

N = 原 checkpoint + 原表 + 原 gain。  
Z = 原 checkpoint + 已有 `log_s4` 表 + 已冻结 gain。  
G = 与 Z 完全相同 sampled endpoints/log-span/K/gain 的 geometric interior 对照。  
Y = 官方 YaRN 的固定目标工作点，不用本轮结果调参。  
M = 官方 MrRoPE-Pro 的固定工作点，仅作为零训练外部对照。

**G 是因果解释控制，N/Y/M 是实用比较。** G 不是经过训练的 FMRoPE 模型，也不能因此给它冠上某个公开方法的名字。FMRoPE 的训练比较优先使用已存在且经核验的 fixed-support paired runs，不把不匹配的 post-hoc FMRoPE collapse 当作优越性证据。

## 7.2 为什么主训练只用 N/Z/Y

N/Z 的训练前后对比回答“位置底座对能力转换是否有增量价值”；Y 是必须包含的成熟实用基线。G 的纯 z 因果识别已在 E1 和既有 fixed-support training 中承担。

因此 E2 的结论应写成“该完整固定 retrofit 底座的适配收益”，不能写成“E2 已独立证明所有训练收益只由 z 引起”。若论文要后一条更强主张，就需要 G 的匹配训练；本稿不把这笔新增成本伪装成已覆盖。

## 7.3 先后次序不是搜索规则

E0 和 24 个语义实例的 E1 诊断先运行。它们允许修复实现错误和阻止无效任务，不允许据此换 Z、换 gain、换 heads、换 loss、换 rank。

E2 第一 seed 在 N/Z/Y 三臂全部完成同样预算后，依据第 10 节的预定 feasibility gate 决定是否投入其余固定 seeds。不能先跑 Z，挑到好 seed，再补 N/Y。

若方法失败，记录为该冻结协议失败；下一份协议要新版本、新开发数据、重新声明测试暴露，不能悄悄覆盖本轮。

# 8. E0/E1：独立协议与单表确认

## 8.1 E0 不得变成无止境的仓库整理

E0 只审查会改变本轮结论的资产：主 checkpoint、Z/G/Y/M 表、gain、生成路径、数据与 scorer。上限见第 12 节。不能重新汇总整个仓库三个月历史来延迟第一个有效结果。

旧 owner 文档和脚本是查找入口，不默认正确，也不要求无理由重写全部基础设施。建议新建一个独立的最小 evaluator，使它不复用旧的答案 parser、结果聚合和成功判断逻辑；模型加载和可靠算子可以复用，经 parity 检验后固定。

## 8.2 最小运行 manifest

```yaml
run_id: content_addressed_identifier
protocol_version: single_table_transport_v1
model:
  repository: allenai/OLMo-2-0425-1B-Instruct
  revision: REQUIRED_EXACT_COMMIT
  weight_digest: REQUIRED
  tokenizer_digest: REQUIRED
position:
  inv_freq_sha256: REQUIRED
  dtype_at_construction: float64
  runtime_phase_dtype: float32
  shared_across_layers_and_heads: true
  global_single_table: true
  gain_kind: qk_amplitude_or_direct_logit
  gain_numeric_value: REQUIRED
  dynamic_rope_updates: disabled_and_asserted
training:
  max_physical_tokens: 16384
  max_position_id: 16383
  max_context_includes_answer: true
data:
  semantic_ids_digest: REQUIRED
  token_ids_digest: REQUIRED
  split_ledger_digest: REQUIRED
runtime:
  code_commit: REQUIRED
  attention_backend: REQUIRED
  torch_cuda_driver_versions: REQUIRED
  device_name_and_total_memory: REQUIRED
```

`REQUIRED` 表示实验前必须从实际资产读取，不是让 Codex 根据 long 分数填写。尤其不能把结果 JSON 的 SHA 当成 frequency tensor 的 SHA。

## 8.3 算子与 cache 的强制单元测试

**Identity parity。** 插件开启但表/gain 为原值时，与原模型同输入 logits 一致到预定数值容差；32 个短样本同时比较 token IDs、完整生成和 EOS。容差先在 baseline backend 的重复运行与高精度小尺寸参考上校准，不能看到实验输出不同后放宽。

**Global table invariance。** 在 2K、4K、8K、16K 和 audit-only 的更长 cache 分配操作前后，定义几何的 inv_freq 与 gain 都不变。只检查生成元、有效 scaling 和禁止动态重算的配置，不对会合法增长的 sin/cos cache 或 seq_len 缓存元数据要求 hash 不变。这里只检验张量与代码路径，不运行更长任务来选配置。

**Cache parity。** 同一 token 序列使用整段 teacher forcing 和逐 token KV decode，固定抽查多个位置的 next-token logits；加入输出 token 后不能更新频率，不能 reset position IDs。

**Gain convention。** 若 Q/K 各乘 c，则 logit gain 是 a=c²。将公式、tensor hook 和 SDPA scale 都打印进收据，禁止多乘一次或漏平方。N 原始 a=1，G/Z 用完全相同的 a；Y/M 的官方实现另报。

**GQA / rotary layout。** 在实际模型上验证 head mapping、KV sharing、partial rotary dimensions、interleaved/split-half 和 norm 参数形状。不能通过重复 K heads 后重新独立旋转，悄悄增加 cache/算子自由度。

**Gradient parity。** 用一个短任务的少量 batch 验证选定模块有梯度、冻结模块无梯度，LoRA 零输出初始化但不是 A/B 同时全零。检查只监督 assistant 答案和真实终止 token，不意外监督 prompt 或 padding。

## 8.4 评分器必须通过故意构造的反例

以下输入都须有预期评分：正确首数字但完整答案错误；答案后额外错误项；答案正确但达到 max tokens 未终止；提前 EOS；空串；复述全部候选；prompt 中包含答案导致错误截取；包含多个数值时取错位置；合法别名与大小写/标点变化。

RULER 同时报告官方 scorer 和本项目的 complete-output/termination 指标。严格完整输出指标不再叫“官方 RULER 分数”；官方的 substring 或结构规则也不能被当作完整生成无误。

Raw record 必须保存 input IDs、generated IDs、未清洗文本、scorer 前文本、命中的别名、stop reason、EOS IDs、输出长度以及异常状态。失败不能被清洗器悄悄删掉。

## 8.5 Z 表的冻结规则

主 Z 来自已有 8 月 31 日 scale-consistent log-profile 的确定性构造及其受核验导出文件。优先恢复原频率字节和 lineage，不根据 9 月 2 日某个较好/较坏分数选择版本。

若旧表找不到，只允许依据唯一明确的 owner 公式、原始 m 向量和参数重建一次。若 m 向量或数值定义也有歧义，先解决资料歧义，标记 `BLOCKED_TABLE_PROVENANCE`，不得让 Codex重新设计“近似一样”的曲线。

固定表后不为更长长度修改 support、不改变 gain，不对短请求回退。

## 8.6 小型但有区分力的数据结构

每个语义实例 i 至少有以下三个输入；除 compact 外，near/far 的物理长度相同：

- **C：compact。** 只保留足够证据、原问题和固定回答格式，≤2K。
- **N：near-full。** 真实长背景中，证据在问题附近。
- **F：far-full。** 相同 block multiset，交换等 token 长度的文档 blocks，使证据远离问题；问题及回答指令位置不变。

再对 C/N/F 各构造两种**合法证据版本** b=0,1，对应不同正确答案 y0,y1。两版本问题一致，目标答案由证据内容决定。

near/far 的块交换不能改变因果、时间顺序等任务的正确答案；这种不变量仅对当前选定的关系任务成立。按时间顺序问“最后发生什么”的任务不能盲目做保持答案不变的 shuffle。

证据不是带显眼 `EVIDENCE` 标签的特殊 block；文档标题、标点、长度和位置不能泄漏哪个 block 正确。counterfactual 替换要保持全篇事实一致，排除干扰段中重复泄漏旧答案。

## 8.7 首轮 24 个实例只用于诊断

固定 8 个单证据、8 个双证据关系、8 个绑定实例；先验证 Native compact 的可解性。记录全部候选及被排除原因，不按 Z 表表现筛选。

这 24 个实例不得进入最终训练或测试。它们区分：

| 现象 | 合理判断 | 不可作出的判断 |
|---|---|---|
| Native C 失败 | 不是干净的已有能力迁移实例 | 位置外推失败 |
| C 成功，N 失败 | 真实背景、竞争或指令跟随已是障碍 | 仅远距离相位失败 |
| N 成功，F 失败 | 距离/位置布局与计算存在交互 | 某个特定 head 必然失效 |
| F 的证据替换明显改变输出分布，但答案错 | 有 source dependence，decision conversion 未完成 | 证据已被完整正确理解 |
| gold-prefix margin 到中途/EOS 变负 | 找到一条具体失败轨迹 | 只修第一 token 就够 |
| 全轨迹 margin 为正但 greedy 错 | 实现、数值或对齐不一致优先 | 需要新增 RL 才能解决 |

## 8.8 E1 正式确认集

在审计前固定新的 128 个语义实例，按相同三类分层；主确认的 near/far 都固定为总长不超过 16K，compact 不超过 2K；所有方法使用完全相同的 C/N/F/证据版本。先报告完整生成的共同指标，再报告条件化的 Native-short-correct 子集。

零训练选项 N/Z/G/Y/M 全部预先固定。正式集不用于调参数，也不选择“最适合某任务”的模型。既有 RULER/full-NLL suite 作为外部 readout，优先重用已生成且可复核的数据行，不再生成不同随机难度的比较。

32K/64K 的任务输出不在 E1 阶段打开；到 E4 与最终训练模型一起盲测，避免提前看到某一长度失败后针对性改变训练。

# 9. E2：唯一微调协议

## 9.1 与旧失败的明确区别

本轮不是“再来一次 Q/K LoRA”。同时固定以下必要改动：

| 旧风险 | 本轮明确改变 |
|---|---|
| 短程模型本来不会做任务 | 训练对必须由 Native compact 完整正确、证据敏感地完成 |
| 主要是短 token + 虚拟长位置 | 使用真实 8K/16K 背景，position IDs 不超过 16K |
| 只优化长 NLL / 首 token | 全答案与 EOS 的 CE，另看最差 gold-prefix margin |
| wrong-context 仍监督原答案 | 两种合法证据各自监督自己的答案 |
| Q/K 或 Q/K/V/O 限制后处理修复 | 所有 blocks 的 attention 与 MLP linear 低秩适配 |
| Native replay 用错 RoPE 配置 | student 在 Native replay 也固定使用本臂部署表 |
| 单一 sparse KL 均值代替保留 | 分层 KL、完整 Native 决策轨迹与独立任务 gate |
| 看到 long 结果后换方案 | 表、rank、loss、预算与 splits 先固定 |

这是一个复合协议，所以第一轮成功不能归因于其中某个单独组件。E3 只补最有信息量的两个拆分，不声称所有组件都必要。

## 9.2 参数设置

```text
base weights: frozen, BF16
LoRA targets: all Q/K/V/O projections + all MLP linear projections
rank: 16
LoRA alpha: 16    (scaling = alpha/r = 1)
LoRA dropout: 0
LoRA initialization: one factor random, output factor zero
trainable norms/embeddings/LM head: none
RoPE table and attention gain: frozen throughout
attention: ordinary dense causal attention using an exact memory-efficient kernel
adapter: one shared adapter, always enabled for this model arm
```

不使用 QLoRA、量化 KV、shifted sparse attention、head routing 或额外 learned temperature。它们各自可能有效，但会引入新的机制/数值变量。

## 9.3 训练数据：128 个独立语义实例，而不是 128 个随机长串

固定构成为：64 单证据自然文本 QA、32 双证据关系 QA、32 绑定/检索实例。建议复用合法可访问的 SQuAD 风格证据与 2Wiki 类关系材料，以及受控的绑定生成器。公开数据的许可、原 split 和来源必须记录；本稿不声称已生成这些数据。

每个实例两种证据版本，每个版本三个物理长度上限：

$$
L\in\{2048,8192,16384\}.
$$

输出计入长度上限，因此实际 context 要预留正确答案及 EOS。总共 128×2×3=768 个训练视图，一次遍历。满长 token presentations 上限为

$$
128\times2\times(2048+8192+16384)=6,815,744.
$$

这不是 768 个独立语义样本，也不是只有 128×16K 的计算量。

三个长度分别承担：已有短能力锚点、过渡位置、目标工作长度。8K/16K 证据位置与干扰关系分层平衡；随机 seed 在训练前固定，不能每次重新生成直到任务简单。

## 9.4 数据筛选与反事实合法性

只有 Native 在 compact 的两种证据版本上均完整回答正确且正常终止，才进入迁移训练集。teacher 的自然答案可以作为规范化目标，但必须经数据真值或明确规则校验，不能把 teacher 自信当正确。

同时要求证据改变后答案相应改变，防止 teacher 仅凭世界知识或问题模板答对。

若从预先固定的 2,000 个候选中不能获得所需数量，不通过改模板、改模型或挑 Z-short-correct 来填满。报告可用实例数，进入 `BLOCKED_DATA_QUALIFICATION`，人工审查问题构造；任何新构造须产生新数据版本，不能沿用已暴露测试集。

训练筛选用于隔离已有能力迁移；最终评测还必须保留未筛选总体和 short-correct/short-incorrect 分层，报告 qualification rate。否则只证明了选择性子集上的修复。

## 9.5 唯一 Long loss

设 compact teacher 的已验证目标轨迹为 \(\bar y=(y_1,\ldots,y_T,\mathrm{EOS})\)。对每种合法证据版本分别计算：

$$
L_{CE}=-\frac1{|\bar y|}\sum_t\log p_\phi(\bar y_t\mid C,\bar y_{<t}).
$$

teacher 的正 margin 只用来设置一个保守、封顶的轨迹目标：

$$
\tau_t=\min(m_{0,t},1),\qquad m_{0,t}>0.
$$

$$
L_{margin}=\max_t[\tau_t-m_{\phi,t}]_+.
$$

最终

$$
\boxed{L_{task}=\mathbb E_{i,b,L}[L_{CE}+0.25L_{margin}]}
$$

各语义实例等权，各证据版本等权，先在序列内归一化。0.25 和上限 1 是预注册稳定性选择，不是理论最优数值。不做系数 sweep。

两种证据各自有完整 CE，已直接惩罚“忽略证据、总输出同一答案”。不再叠加一个可能通过任意降低 wrong-context likelihood 取巧的无界 contrastive 目标。

不对 16K prompt 的全部背景 token 施加 LM loss。本轮目标是已有任务计算迁移，不是微型 continued pretraining。Native 自然语言功能由下面的约束覆盖；long NLL 作为诊断指标，不是优化目标或选模标准。

EOS 是模型/模板真实的 end-of-turn 终止 token，不一定是通用的某个字符串。禁止 `min_new_tokens` 人为屏蔽 EOS，再声称 EOS 恢复。

## 9.6 Native 数据与实际函数约束

Native replay pool 独立于 long 任务，共 512 条，四组各 128 条：自然文本、一般指令/知识、短程推理/QA、位置敏感/格式/终止。自然文本包含两个领域，避免只保住 PG19 一种分布。

对自然文本，每条预先按固定位置规则选 32 个 valid prefix positions；对短回答指令，保留完整已验证 teacher answer-prefix 序列及终止位置。prefix 采样规则在任何训练之前固定，不按训练后 loss 大小挑 token。

每组的 sampled-prefix functional KL 是

$$
D_j(\phi)=\mathbb E_{c\sim\hat\mu_{N,j}}
D_{KL}\left(p_{\theta_0,\Omega_0}(\cdot|c)
\|p_{\theta_0+\Delta\theta(\phi),\Omega_{arm}}(\cdot|c)\right).
$$

该均值是相对于明确采样分布的估计，不叫“所有 Native tokens 上的精确 KL”。完整词表 KL 与只保留 top-k 的近似也必须区分。

训练目标形式：

$$
\min_\phi L_{task}(\phi)
\quad\text{s.t.}\quad D_j(\phi)\le0.02\ \forall j.
$$

0.02 nats/token 是本轮设计预算，不是无遗忘阈值定理。最终 Native 接受标准还要求：

- 两个自然文本领域分别 ΔNLL≤0.03 nats/token；
- Native 完整生成任务宏平均不低于原模型 2 个百分点；每个预设任务组不低于 5 个百分点；
- 未正常终止比例的增加不超过 2 个百分点。

正式确认使用预设置信区间而非单一均值；不能根据训练结果放宽这些门限。若更偏好不同容忍度，只能在任何新训练/验证结果出现前由负责人整体修改协议版本。

## 9.7 为什么不直接把 KL 系数固定为一个魔法数

使用非负 dual multipliers 调整约束力度：

$$
\mathcal L=L_{task}+\sum_j\lambda_j(D_j/0.02-1),\quad\lambda_j\ge0.
$$

每个得到新估计的组更新

$$
\lambda_j\leftarrow\max\{0,\lambda_j+0.05(D_j/0.02-1)\}.
$$

初始 λj=1。没有抽到某组时不拿陈旧 minibatch 值继续更新该组。λ 是训练约束算法的状态，不是通过 long benchmark 选择的外部超参数。

小批量 primal-dual 优化**不保证每一步或最终总体严格可行**。只接受在独立 Native calibration/validation 上通过检查的 checkpoint；最终测试另行验证保留。若优化不稳定，不用“约束理论正确”掩盖实现或预算失败。

## 9.8 固定两阶段日程

**阶段 R：Native restoration，32 steps。** 每 step 8 个 Native replay 样本，最小化四组等权 teacher KL。student 始终用该臂实际表。对 N 臂，理论上原始函数已匹配，更新可以接近零；仍保留同样阶段与计数。该阶段不训练频率，不接触 long 标签。

**阶段 T：paired generation transfer，96 steps。** 遍历 768 个视图，effective task batch=8；每 step 另取 2 个 Native 样本，组别轮转。使用上述 task loss 和 primal-dual Native 约束。

8K/16K 视图在整个阶段分层交错，而不是先把所有简单短样本耗完，再在最后突击长样本。每种证据版本、长度、任务族的出现次数相同。

优化器统一：AdamW，学习率 1e-4，β=(0.9,0.95)，eps=1e-8，weight decay=0，global grad norm clip=1。每阶段前 10% warmup，余下 cosine decay；每阶段 optimizer state 明确重置，三臂一致。microbatch=1，用 gradient accumulation 实现 effective batch。开启 activation checkpointing，训练关闭 KV cache。

总 128 steps，保存 step 0/32/64/96/128。不能因 loss 尚在下降而延长某一臂，不根据第一 seed 的结果增加 rank。

主 seeds 固定为 42、43、44；对应运行顺序固定为 N/Z/Y、Z/Y/N、Y/N/Z，各臂尽量使用同一实际 GPU/backend。允许完全相同 checkpoint 的故障续跑，不允许换 seed 当作“重试稳定性”。

## 9.9 初始 Native 不可行怎么办

Z 的零 adapter 起点可以超出 Native budget。阶段 R 的目的正是恢复实际表下的 Native 兼容性。32 steps 后若未完全可行但无数值故障，仍按固定计划完成阶段 T；不通过额外 restoration sweep 选出新初始化。

只有最终可行 checkpoint 才具有“联合工作点”资格。无可行 checkpoint 时，结论是**当前参数空间、目标与预算未恢复 Native**，不允许把“相对 Z 起点没再变差”当成“相对原模型无遗忘”。

# 10. 数据切分、模型选择、正式成功与停止条件

## 10.1 数据权限隔离

| 数据池 | 用途 | 允许影响什么 |
|---|---|---|
| 24-instance diagnostic | 发现无效构造/实现故障 | 数据和实现修复；不挑方法 |
| 128 semantic training instances | 固定适配 | 训练权重，不调位置表 |
| 独立 64-instance ≤16K validation | 有限 checkpoint 选择 | 只选本协议中的保存点 |
| E1 fresh single-table confirmation | 验证冻结 Z/G/Y/M | 不能调 Z/gain/训练设计 |
| 最终 generation holdout | 主结果 | 不能选择方法或 checkpoint |
| Native train/calibration/validation/test | 训练约束、可行性检查、最终保留 | 按明确职责分离 |
| 32K/64K blind outputs | 长度泛化边界 | 所有配置和 checkpoints 冻结后一次打开 |

语义实例、源文档、实体关系、模板 lineage 做 group split。一个实例的不同长度、两种证据版本绝不跨 train/validation/test。两条语句只是实体名字不同，也不自动算独立模板族。

Native 数据另行分组隔离：第 9.6 节的 512 条仅为 replay train；另建 128 条 calibration（每组 32）检查实现与约束估计、256 条 validation（每组 64）用于保存点可行性选择；正式 Native test 使用下一节的独立样本。不能将 replay 上 KL 通过写成 held-out 保留，也不能让 E1 已公开的 Native 诊断样本同时承担 E2 最终盲测。

## 10.2 最终测试组成

最低正式 generation set 共 256 个独立语义实例：其中 192 个按三组各 64，任务族与训练可以相同但实例/模板/文档均未见；另外 64 个是完全未训练任务族的自然 QA 实例，例如训练使用单证据/2Wiki 类关系时，预先指定 HotpotQA 与 Qasper 的可验证任务部分。

若 Qasper 原任务不可在 16K 内保留全部必要证据，不做截断后继续沿用原标签；记录不适用。可以报告符合长度约束的明确子集，不能冒充完整 benchmark。

官方 RULER 使用全部预注册 families，不因为某一族低分就删除。绑定类训练与某些 retrieval families 有结构重叠，必须标为 family-overlap，不把它们叫 unseen-task transfer。[R25]

Native formal test：两个自然文本领域各 128 个独立片段；另有一般指令/知识、短推理、位置/格式/终止三组各 500 个短任务。若现实预算只支持更小样本，保留原容忍度并报告区间不足，不以“没显著下降”宣布非劣。

原来用于 Native retention 的固定五任务套件也应按原定义并列报告，保持历史可比性；但其 log-likelihood/选择题分数不能改称自由生成成功率。新增生成式 Native 样本优先从现有有真值、可独立评分的回归池按预定顺序取；缺少可验证真值或可靠评分规则时标记数据不足，不让一个临时 LLM judge 自动裁定无遗忘。

## 10.3 选模规则

每一臂独立选同样预定保存点，不选方法：

1. 先看独立 Native validation 的可行性；
2. 在可行 checkpoints 中，最大化 16K validation 的**完整答案正确且正常终止**的任务宏平均；
3. 并列时 Native KL 更低者优先，再并列选更早 checkpoint；
4. 没有可行 checkpoint，则该臂记为 infeasible，不把未通过 Native 的最佳 long 点放进“保留能力后的方法排名”。

阶段 R 的 step 32 是天然的 restoration-only 控制，无需再训练一个同义新分支。训练 loss、平均 NLL、第一 token margin 不作为最终选模依据。

在打开正式 holdout 前，为选定 checkpoint 写入不可变 hash 清单。

## 10.4 主成功条件

**Z 路线：** E1 的同一张表在独立数据上存在明确的真实 long 收益，并完整报告 Native 代价。只有通过 Native gate 才称合格联合工作点；未通过者只能称有代价的单表工作点。

**T 路线：** 在 Native gate 成立的前提下，Z+adapter 相对 Z 的完整生成收益有正向置信下界，且证据 counterfactual 的双版本成功率同步提高；不能只有某一版本、某一个 first token 或 teacher-forced 分数改善。

**位置底座的增量价值：** 比较

$$
I=[S(Z,T)-S(Z,0)]-[S(N,T)-S(N,0)].
$$

I>0 是一个描述训练增量差异的交互量，需同时报告绝对分数和 N/Z 的起点；floor/ceiling 可影响它，不能直接把它当成某一内部机制的证明。

**跨任务族：** 只有完全未训练的任务族上也有可信收益，才扩大为跨任务族迁移。否则明确限定为已训练关系族的新实例迁移。

**更远长度：** 32K 和 64K 各自报告。16K 成功不追认更远成功，32K 成功不追认 64K 成功。

## 10.5 预定 kill criteria

| 状态 | 动作 | 结论范围 |
|---|---|---|
| identity/cache/joint-relabel parity 失败 | 禁止正式 GPU 实验，修实现 | 不能解释成方法失败 |
| 旧核心结果无法对齐 run/table/rows | 降级旧结果，按新协议重测 | 不编造 canonical 数字 |
| compact Native 可解实例不足 | 停止长程迁移训练，审查数据构造 | 不用本来不会的任务证明传输失败 |
| 训练 loss 非有限、参数冻结错误、数据泄漏 | 中止，废弃受影响 run | 不继续等它自行恢复 |
| 第一 seed 三臂完成后，Z 的 ≤16K validation 没有完整生成增量，且 Native 无可行点 | 不扩大后两 seeds 与 E3 | 当前固定方案无可用 feasibility 信号 |
| 完整生成改善，但 Native gate 失败 | 不称联合成功，不放宽门限 | 记录 trade-off |
| 只有 NLL / retrieval / margin 改善 | 不称能力恢复 | 记录断点，停止解码器搜索 |
| 16K 成功，更远失败 | 保留 16K 结果，报告边界 | 不临时换 s8/s16 或暴露目标相位 |
| 置信区间太宽 | inconclusive | 不等价于有效，也不等价于无效 |

Kill 是控制投入和结论范围，不是证明任何更大方法类别不可能。

# 11. 机制诊断：解释失败，但不把诊断变成新方法搜索

## 11.1 每条样本的最低 readout stack

每个模型臂、每个 C/N/F、每个证据版本记录：完整答案正确性；规范化答案 F1/EM；自然 EOS/turn-end；first-error index；gold-prefix 所有 token 的正确 log-prob/rank/margin；必要时 student-prefix 上的后续行为；证据替换效应；长度、位置和 stop reason。

“first-error”同时保留 token-exact 与语义级标注。token-exact 首次不同可能只是合法同义表述，不应自动记成推理失败。

Attention 只在 24-instance diagnostic 上记录少数 answer queries 的证据 mass / logit gap，禁止为了指标把完整 N×N attention map 常驻显存。

## 11.2 证据效应的对称定义

对两种证据版本 C0/C1 和各自答案 y0/y1，定义 length-normalized sequence score S。使用

$$
E=\tfrac12\{[S(y_0|C_0)-S(y_0|C_1)]
+[S(y_1|C_1)-S(y_1|C_0)]\}.
$$

同时报告二者完整生成均正确的概率

$$
P(\mathrm{correct}(C_0)\land\mathrm{correct}(C_1)).
$$

E>0 只说明答案概率对来源有区分，不代表最后选择正确。E 与双版本生成成功共同改善，才更支持真正的 evidence-conditioned generation。

证据删除控制应保持近似相同长度/位置，用中性内容替代；直接删除可能同时改变 query 的 position IDs 和干扰数量，不能把所有变化都归于“没有证据”。

## 11.3 Compact、near、far 的差分各回答什么

记 G 为越大越好的能力指标：

- `G(N)-G(C)`：加入真实背景的影响；
- `G(F)-G(N)`：在相同实体上下文规模下，远置证据的影响；
- 方法相对 Native 的上述差分变化：隔离它改善了哪一种条件差异。

绝对分数必须同时列出。若 compact 被方法损伤，单看 far 与 compact 的差距缩小会得到伪改善。

这些差分仍不是无假设的全部机制分解；near/far 文档排列可以改变浅层上下文表示。它们比“一个短 prompt 对比一个随机长 prompt”更可解释，但不宣称纯粹只改了一维相位。

## 11.4 可选的少量 activation patching

仅在证据依赖已明显存在、但完整生成仍失败时执行。优先分析仓库已经保存的 matched layerwise readout tensors；读取旧 8B 张量不等于新增 8B GPU 项目，也不能代替本轮主模型确认。[I07]

新 patching 最多使用 diagnostic 中预先按 semantic ID 排序的 8 个合格实例，深度固定为 1/4、1/2、3/4 blocks，不根据效果寻找最佳层。使用同样长度、同样 query token 的 near/far 对作为 donor/recipient。

比较正确语义 donor、另一答案的 counterfactual donor 和 sham patch。只 patch 预先指定的 query/answer positions，不把目标答案 token 直接塞进模型输入。

若正确 donor 对 margin 有定向恢复、错误 donor 朝错误答案移动、sham 无影响，这支持该位置的 residual state 对决策有因果作用。**它不证明自然推理时该位置是唯一瓶颈，也不证明所有合法位置干预都无法到达相同状态。**

不要把 final hidden-state 全量替换后恢复答案当成高价值发现：这几乎把答案表征本身直接注入。诊断必须说明改变了什么、不改变什么，并有错误 donor 控制。

## 11.5 E3 的两个解释性训练消融

仅在主协议通过 feasibility 后运行，不能作为从中选最好方案的候选集。

**E3a：compact-only，检验是否只是学会了任务。** Z、同一 LoRA、同样的完整答案 presentations、相同 Native 约束与 optimizer steps，将所有任务视图替换为 compact 条件。它匹配的是语义与答案监督次数，不匹配处理的 context tokens/FLOPs；须明确标注。若与主训练同样改善 far，不能声称真实 long exposure 是必要因素。

**E3b：attention-only、参数量匹配，检验修正的分布位置。** OLMo 的 all-linear r16 参数数为约 12.06M；Q/K/V/O-only r46 恰好在该结构下匹配同样的参数计数：

$$
16\,[4(2048+2048)+3(2048+8192)]\times16
=16\,[4(2048+2048)]\times46.
$$

因此固定 r46、alpha46 的 attention-only 控制，而不是拿参数数量少约三倍的 QKVO-r16 作不公平比较。模型实际 shape 不符时按上述参数计数公式确定最接近整数秩，差额如实报告；不能根据 outcome 改秩。

该消融区分“同样低秩参数预算放在更广模块”与“集中在 attention”。即便 all-linear 更好，也不能推出每个 MLP 都必要或 r16 最小；参数分配和优化几何仍不同。

两个消融只运行预先固定 seed 42；属于解释性证据，不能用单 seed 声称普遍稳定机制。主结论仍由 N/Z/Y 三臂多 seed 承担。

# 12. 统计、算力和 32GB 实际可执行性

## 12.1 独立单位是语义实例与训练 run

同一实例的三种长度、两种证据版本和 near/far 不是六个或十二个独立样本。置信区间按 semantic instance 聚类重采样；方法间使用 paired differences。

训练 seed 是独立 run 级别变化。报告每个 seed 和跨 seed 的范围/均值；只有三个 seeds 时，不把几千条 token 当作几千个独立训练重复。针对实例 bootstrap 的区间不覆盖全部训练随机性，应明确。

正式主要 endpoint 固定为 Native-gated 16K 完整输出成功率及其相对 Z0 的增量。32K/64K、task-family breakdown、NLL 和机制指标是明确分层的次要结果。多项显著性结论采用预声明的 Holm 校正；不要在十几个指标里只挑一个 p<0.05。

Native 非劣使用预声明的单侧 95% 配对区间：损害型指标看上界，保留型准确率差看下界；按 semantic/document cluster 计算，不能用 token 当独立重复。各组联合过线是交并检验的判定，不把多组均值相互抵消。Native 非劣按预设百分点范围和配对区间检验；“没检测到显著下降”不是“证明没下降”。区间太宽时保持 inconclusive。

## 12.2 硬件约束必须以实际机器为准

NVIDIA 官方 4080 Super 规格为 16GB，5090 为 32GB。用户报告的 4080 Super 32GB 可能是服务商配置或改装版本；本稿按用户资源规划，但 Codex 上机必须记录实际显存、驱动、功率/时钟限制与设备标识，而不是按名称推断。[R28]

两张卡不自动构成 64GB 连续显存。即便在同一机器上，默认也不使用 tensor parallel、跨机梯度同步或训练 teacher/student 双模型常驻同卡。

**推荐分工：** 5090 执行主适配；4080 执行 teacher 缓存、Native 校准、零训练评估和独立评分。没有并发资源时顺序执行，不改变实验条件。

## 12.3 KV cache 的最低内存账

对标准 KV cache，batch=1、每元素 b 字节：

$$
M_{KV}=2\,n_{layers}\,n_{KVheads}\,d_{head}\,N\,b.
$$

根据官方配置计算的 BF16 KV 大小如下，不包含模型权重、prefill workspace、输出与碎片：[R26, R27]

| 长度 | OLMo 16 layers / 16 KV heads / head_dim128 | Qwen2.5-1.5B 28 layers / 2 KV heads / head_dim128 |
|---|---:|---:|
| 4K | 0.50 GiB | 0.109 GiB |
| 8K | 1.00 GiB | 0.219 GiB |
| 16K | 2.00 GiB | 0.438 GiB |
| 32K | 4.00 GiB | 0.875 GiB |
| 64K | 8.00 GiB | 1.750 GiB |

这说明更远生成在单卡上不必然受 KV 的硬上限阻断，但不保证 prefill/训练已经能放下，更不保证耗时很小。

## 12.4 训练内存最容易踩的坑

1. 1.485B BF16 基础权重约 2.77 GiB，但 adapter 训练仍需为 backbone 激活反传；“只训练 0.8% 参数”不等于只花 0.8% 激活内存。
2. 必须使用数学等价的 memory-efficient dense causal attention；FlashAttention 类实现避免完整 attention matrix 常驻，但不把 O(N²) attention arithmetic 消除。[R29]
3. 16K×100352 vocabulary 的 FP32 logits 单张量约 6.13 GiB。仅仅 mask 掉 prompt loss，若仍计算全长词表 logits，显存仍被占用。
4. 对答案 CE/margin，先取完整 Transformer 的目标位置 hidden states，再做 LM-head 投影；Native KL 同样只投影预注册 sampled positions。这是逐 token 线性 head 的精确计算重排，需与标准短序列实现做 parity。
5. Native teacher 在另一张卡或离线计算；固定 sampled positions 的全词表 logits 分片保存在 CPU/disk。精确 KL 目标保存 FP32 logits；若为了存储压成低精度或 top-k，明确写成近似，并另有全词表确认。
6. 训练 `use_cache=False`，开启 activation checkpointing，microbatch=1。NLL 评估按 head-projection chunks 累积，不改变上下文窗口或给每个 chunk 重置位置。
7. 不请求 `output_attentions=True` 物化 16K/64K 全矩阵；需要的证据 mass 只对 selected query rows 重构或流式计算。

## 12.5 训练 token 与 forward 预算

每个主训练 run 的上限：

| 项目 | 上限 token presentations |
|---|---:|
| 任务视图，128×2×(2K+8K+16K) | 6,815,744 |
| restoration，32×8×4K | 1,048,576 |
| joint Native replay，96×2×4K | 786,432 |
| **student 总计** | **8,650,752** |

这不包括 teacher cache、validation、generation、失败调试或因 activation checkpointing 增加的计算。必须分开记录输入 token、受监督 token、student forward/backward、teacher forward 和 decode tokens。

主 N/Z/Y 三臂 × 三 seeds 共九个 run。E3 两个单 seed 消融另计，不默认再为每个模型族复制全部训练矩阵。

## 12.6 不伪造 GPU 小时预测

先对最终 backend 测 5 个 warmup + 10 个稳定 step，覆盖实际长度混合，分别记录 task batch、Native replay 和 restoration 的时间/峰值显存。再计算

$$
\widehat T_{run}=32\,t_R+96\,t_T+T_{validation}+T_{serialization}.
$$

generation 以 prefill 与每 token decode 分开估算；不能用 4K 吞吐线性外推 64K prefill。

建议投入上限是管理预算，不是预计必然耗时：E0 GPU 审计 2 GPU-h；E1 diagnostic 与单表确认 6 GPU-h；第一 seed 三臂训练及验证 12 GPU-h；剩余 seeds 24 GPU-h；E3 8 GPU-h；正式/更远/第二模型评估 12 GPU-h；预留 8 GPU-h。合计上限 72 GPU-h，可按实际吞吐在正式运行前整体缩减范围。

若预计超出预算，优先删 E3、第二模型的扩展评估和可选 patching（第二模型训练本来就不在主矩阵内），保留主模型的完整输出、Native test、匹配基线与 raw receipts。不能通过删困难样本、改 max output、换量化配置或放宽标准来省出“成功”。

上限不是自动授权花费：Codex 先把实测 ETA 与资源需求交给负责人，再启动正式批次。运行时空等下载、失效任务或无用 sweeps 不应占着付费 GPU。

## 12.7 OOM 与故障规则

第一次 OOM：保存配置和显存事件，停止该 run。仅允许数值/计算等价的 memory 修复，如去掉全长词表 logits、减 microbatch 并保持 accumulation、修复缓存、启动 checkpointing。

不得未经新协议把 16K 改成 8K、改 sparse attention、改量化或删层后沿用相同 run_id。若标准算子在实际机器确实放不下，记录工程约束失败，停止该分支，不无限原样重试。

# 13. 供 Codex 实现的核心伪代码

以下是算法合同，不绑定某个库版本，函数名是需要实现或对应到现有代码的伪接口，并非声称某个公开库已有这些 API。Codex 要根据已安装模型实现审查 API，并保证语义一致。伪代码中的 N/Z/Y 指模型臂，第 8 节 C/N/F 中的 N 指 near 输入，实现中分别命名 `arm_native` 与 `layout_near`，避免混淆。

## 13.1 构造并审计数据

```python
# CPU first; do not launch a large GPU evaluation here.
candidates = load_source_locked_candidates(max_count=2000)
train, val, final_test = split_by_source_document_relation_and_template(candidates)
assert no_lineage_overlap(train, val, final_test)

qualified_train = []
for item in fixed_order(train):
    pair = build_two_valid_counterfactual_worlds(item)
    assert answers_differ(pair)
    assert query_and_output_instruction_unchanged(pair)
    assert no_answer_leak_or_inconsistent_fact(pair)

    teacher_runs = native_greedy_on_compact(pair)
    save_all_teacher_runs_and_rejection_reasons(teacher_runs)
    if not both_truth_correct_and_naturally_terminated(teacher_runs):
        continue

    targets = exact_verified_teacher_completions_including_end_token(teacher_runs)
    for world in pair:
        for max_len in (2048, 8192, 16384):
            view = build_real_text_view(world, max_len, fixed_layout_seed)
            # Input length budget includes the target answer for training.
            assert len(view.prompt_ids) + len(targets[world]) <= max_len
            assert all_position_ids_contiguous_and_below_16k(view, targets[world])
            assert all_required_evidence_present(view)
            save_immutable_view(view, targets[world])
    qualified_train.append(item.semantic_id)
    if quotas_reached(qualified_train):
        break

assert_exact_training_quotas_or_block()
write_data_manifest_and_hashes()
```

## 13.2 冻结配置与 cache 合同

```python
model = load_exact_checkpoint_and_tokenizer(manifest)
install_one_shared_table(model, table_bytes, fixed_gain)
expected = hash_immutable_frequency_generators_and_effective_gain(model)

for test in identity_layout_gain_cache_tests:
    receipt = run(test)
    require_pass(receipt)

# Audit real tensor bytes at installation and request boundaries.
# Runtime uses a write/version guard: do not copy/hash CUDA tensors per token.
# Legitimately growing sin/cos caches are excluded from geometry identity.
def assert_same_position_configuration(audit_boundary=False):
    assert immutable_geometry_version_unchanged(model)
    assert no_request_length_router_is_active(model)
    if audit_boundary:
        assert hash_immutable_frequency_generators_and_effective_gain(model) == expected

# Hash/version guards do not replace numerical parity; both are required.
```

## 13.3 精确目标位置投影

```python
hidden = transformer_forward_full_causal_sequence(
    input_ids, position_ids, attention_mask,
    use_cache=False, activation_checkpointing=True
)

# Index uses the hidden state that predicts each next token, not the target token's
# own post-embedding state. This off-by-one must have a unit test.
selected_hidden = gather_prediction_positions(hidden, positions_for_this_loss)
selected_logits = original_lm_head(selected_hidden)
loss = compute_loss_in_fp32(selected_logits, targets_or_teacher_logits)

# Reference parity on small sequences:
# selected_logits == gather(full_standard_logits, same_prediction_positions)
```

## 13.4 固定适配

```python
for seed in (42, 43, 44):
    for arm in balanced_predeclared_order("N", "Z", "Y"):
        student = fresh_native_weights_with_arm_fixed_table(arm)
        attach_all_linear_lora(student, rank=16, alpha=16, seed=seed)
        freeze_base_norm_embeddings_head_rope_gain(student)
        teacher = immutable_native_teacher()

        for step in range(32):
            batch = native_replay_batch(size=8, balanced_groups=True)
            d = full_vocab_kl_at_declared_prefixes(teacher, student, batch)
            # Student uses arm table even on Native data.
            optimizer_step(mean_by_group(d))
        save_checkpoint("restoration_32")
        reset_optimizer_with_declared_schedule()

        dual = {group: 1.0 for group in native_groups}
        for step, task_batch in enumerate(fixed_768_view_schedule(batch=8)):
            native_batch = native_replay_batch(size=2, rotating_groups=True)
            task = complete_answer_ce(task_batch) + 0.25 * worst_margin_hinge(task_batch)
            d = full_vocab_kl_at_declared_prefixes(teacher, student, native_batch)
            objective = task + sum(dual[g] * (d[g] / 0.02 - 1) for g in observed_groups(d))
            optimizer_step(objective, clip_norm=1.0)
            for g in observed_groups(d):
                dual[g] = max(0.0, dual[g] + 0.05 * (stop_gradient(d[g]) / 0.02 - 1))
            assert_frozen_base_and_position_hashes()
            save_fixed_checkpoints_and_complete_run_receipts()

        # No train/validation mixture and no 32K/64K access here.
        select_only_among_predeclared_checkpoints_using_native_then_16k_validation()

    if seed == 42:
        apply_predeclared_feasibility_gate_to_all_three_arms()
        # Either continue the fixed seeds or stop; do not pick new hyperparameters.
```

若多组 batch 的大小不同，应先按每个 group 内 valid sampled positions 归一化，再做预定组权重平均，不让最长文本自动控制 Native loss。

## 13.5 完整生成审计

```python
for example in locked_evaluation_order:
    cache = new_empty_cache()
    prompt = exact_tokenized_prompt(example)
    assert_same_position_configuration(audit_boundary=True)
    output = []

    for t in range(max_new_tokens_for_task_family):
        logits = next_token_with_standard_kv_cache(prompt, output, cache)
        token = deterministic_argmax_without_hidden_processors(logits)
        output.append(token)
        assert_same_position_configuration(audit_boundary=False)
        if token in true_model_end_of_turn_ids:
            stop_reason = "natural_end_token"
            break
    else:
        stop_reason = "length_cap"

    assert_same_position_configuration(audit_boundary=True)
    save_raw_ids_text_cache_metadata_and_stop_reason()
    score_official_task_metric_separately_from_full_output_metric()
    run_gold_prefix_margin_audit_with_matching_tokenization()

# Only after all selected checkpoints are hashed and final configurations locked:
run_32k_and_64k_blind_suites()
reveal_both_lengths_together()
```

## 13.6 每个 run 必须输出的机器可读汇总

```text
identity: model/table/tokenizer/code/data/decoder/scorer hashes
budget: physical_tokens, supervised_tokens, max_position_id, steps,
        teacher_forwards, student_forwards_backwards, walltime, peak_memory
native: total_KL_by_group, NLL_delta_by_domain,
        full_generation_macro, termination_delta, feasible_status
long: compact/near/far complete_success, official_task_scores,
      two_world_joint_success, evidence_effect,
      first_semantic_error, gold_prefix_min_margin, EOS_status
statistics: semantic_cluster_CIs, all_seed_values, adjusted_tests
status: PASS / FAIL / INCONCLUSIVE / BLOCKED with exact failed clause
artifacts: raw_generation_paths, checkpoint_path, manifest_path
```

以上字段缺失时不能只给一个“score=0.63”作为完成报告。

## 13.7 E4：外部模型与更远长度的唯一规则

第二模型固定为 Qwen2.5-1.5B-Instruct 的明确 revision，不通过比较多个候选的 long 分数选择。它的官方配置是 32K window、GQA、不同 base，适合检验跨 checkpoint 的函数兼容性，但不能当作另一个原生 4K 模型。[R27]

本轮第二模型只做零训练复制和少量机制确认，不默认复制九个训练 runs。位置方案沿用已经冻结的 normalized-index movement profile；若源/目标 K 相同，则逐项拷贝 m；不同则按既有唯一插值定义映射。s 固定为 2，用于其 32K→64K 工作点，gain 使用同一已冻结规则，所有 short/32K/64K 请求使用这一张表。该构造不读取 long outcomes，不是新的 m curve。

Codex 必须明确：这张第二模型表可能需要按既有构造首次导出，本稿没有声称仓库已存在它的 hash。若原 normalized-index 插值规则没有明确 owner，不重新发明一版，而是暂停外部复制；不影响主模型实验。

外部复制用最终 holdout 中预先固定的 64 个语义实例，N/Z/Y/M 相同输入。主 OLMo 的 32K/64K 和第二模型结果在所有配置冻结后一起打开。第二模型结果支持“一个额外模型族上的外部有效性/边界”，不支持 universal cross-checkpoint law 或 K 因果定律。

# 第三部分：论文与扩展思考

# 14. 最大化 ICLR 2027 投稿质量：先消除真正的拒稿理由

## 14.1 原评审记录告诉我们的不是“再加几个 benchmark”

上传的评审/AC 记录主要质疑：与 FMRoPE/频带分析的 novelty 重叠；小模型与诊断任务的外部有效性；从 surrogate 到 cosh 和 operating rule 的推导链缺口；方法选择与比较是否足够公平。[I11]

因此，最高价值工作是让几个核心主张变得难以被替代解释击穿，而不是继续增加十几条互不相干的理论与负结果。ICLR 官方 reviewer guide 也强调问题、动机、证据与新增知识，并未要求所有工作都达到 leaderboard SOTA。[R31]

## 14.2 建议保留的一条论文主线

> **有限 RoPE 频率分配是独立设计变量；它的几何收益必须经过 checkpoint 的坐标绑定与决策计算才能转成能力。固定位置底座的作用，应分别在训练、冻结改造和少量适配中检验。**

当前上传稿的 “RoPE Has a Spectral Budget” 方向可保留，但“budget”必须指清楚有限维配置/Gram 结构，不暗示已经证明 Native 与 long 性能有严格守恒或不可突破的 trade-off。[I06]

不同生命周期使用的具体构造可以不同：from-scratch 的 EVQ-Cosh 与成熟模型的 log-profile retrofit 不是同一条频率公式。共同点是研究同一个有限 allocation 自由度；论文应公开这个区别，不把它们拼成一个从头到尾都被同一定理保证的算法。

## 14.3 三个主贡献，而不是七个小故事

**贡献一：受控识别。** 在 exact sampled support、pair count、training budget 匹配下，改变 allocation 引起可复核变化。利用已经完成的多 seed 训练证据，优先修复 owner 链，不重做大规模预训练。

**贡献二：成熟 checkpoint 的兼容性机制。** 用 frequency-only 与 joint relabeling 区分错误配对和纯坐标重命名，说明频谱多重集合和简单 movement 距离不能决定函数保留。理论呈现真实 bilinear 和输出功能几何，明确其局部/条件边界。

**贡献三：从证据利用到生成的可操作边界。** 展示同一表下 source dependence、gold-prefix trajectory margin、完整生成并不等价；若 E2 成功，再给出受 Native 约束的小数据修复作为建设性验证。如果 E2 失败，则只保留被数据支持的边界结论，不能写成已经完成的 conversion 方法。

基本线性代数、KL Hessian、argmax 证书和 greedy 归纳本身不是足够的独立 novelty。它们的价值在于约束实验与解释，而不是堆砌 theorem 数量。

## 14.4 主图与主表建议

| 位置 | 内容 | 审稿人应该一眼看懂什么 |
|---|---|---|
| Figure 1 | 固定 support 的 allocation、ordered binding、输出轨迹三层示意；连到真实干预 | z 不是 base 的换名；几何不是行为 |
| Figure 2 | 原表 / frequency permutation / joint relabeling，Native 输出与几何指标并列 | 内容绑定是对象，不是张量实现事故 |
| Figure 3 | 每个实例 compact→near→far 的完整输出与最差 margin；至少一个正反 counterfactual | long NLL 变好但为何仍不生成 |
| Table 1 | 固定 support 的既有多 seed 训练，包含 FMRoPE 相关公平控制 | 与最接近工作不是只讲概念差异 |
| Table 2 | N/Z/G/Y/M 的同表 Native+真实 long 结果 | 既有单表收益与代价可见，不靠 routing |
| Table 3 | N/Z/Y × 训练前后；Native gate、完整输出、held-out family、更远边界 | 少量训练到底多带来了什么 |

若图表过多，把机制消融、完整 RULER breakdown 与更远失败放附录；主文必须呈现决定结论的负结果与 Native 代价，不能只放 appendix 隐藏。

## 14.5 九页主文的分配建议

ICLR 2027 初稿正文上限为 9 页，参考文献/附录另计；不要按往年记忆安排页数。[R30]

建议约：引言与结论预览 1 页；问题设置与相关工作 1 页；有限 allocation/兼容性理论 1.5 页；受控识别实验 1.5 页；单表与生成迁移 2.5 页；机制讨论、限制与结论 1 页；其余留给关键图表排版。不是要求严格到小数页，而是防止新微调协议把原有主贡献挤掉。

解析 surrogate 的完整变分推导、regularity 假设、small-τ/finite-τ 范围、数值求积与有限 K 对比，放技术附录并从主文明确引用。若 operating rule 缺少实证支持，就删除 universal optimal 语言，不用一段“理论直觉”替代验证。

## 14.6 近邻比较的优先顺序

FMRoPE 是原评审已指出的直接近邻，应给出可核验的同支持域比较；MrRoPE-Pro 是当前 training-free scaling 的近邻，不能只打一个弱 YaRN 实现；YaRN 需要同时澄清无训练配置与训练后的配置，不能以它零训练在某个小模型上低分，推翻其公开训练结果。[R04, R09, R10]

LeRoPE/AdaRoPE 表明“频率能学”“不同 head 需要不同频率”已不是空白。LongReD/LinearARD 表明“改 RoPE 后用 Native teacher 恢复”也不是新颖性本身。相关工作应主动区分这些重叠，而不是依赖审稿人没见过新论文。[R07, R08, R22, R23]

不要求本轮复现所有工作，尤其不把 10B-token、8-GPU 或新自定义 kernel 的训练复现塞进三周预算。需要明确哪些是直接数值基线、哪些是方法层面的边界比较，不能把无法匹配的结果放进同一张 SOTA 表。

# 15. 除此之外，还能做什么

## 15.1 最高 ROI 的额外贡献：把评测合同做成可复用资产

比另造一个 benchmark 名字更有价值的是发布一个小而可审计的 transformation suite：compact/near/far、合法 counterfactual、原始 token/stop records、同一表 hash 检查和官方/完整输出评分并列。

它应附带“哪些任务允许移动证据而不改答案”“哪些操作改变答案”“哪些结果只能作条件化迁移”的数据合同。同行可以用自己的表或 adapter 运行，且容易发现 NLL、retrieval 与 generation 之间的断点。

不要仅凭 128 或 256 个实例宣称建立了通用长上下文 benchmark。定位为可复用的 causal evaluation protocol 和公开实例集即可。

## 15.2 可以低成本补强的理论—实证连接

利用第 4.5 节的点态 KL-to-argmax 边界，检查 Native 出错 tokens 是否集中在低 κ 区域，并展示平均 KL 无法保护低 margin 决策的具体反例。这个分析只需已有 logits，不需要新训练。

预测性检验必须在固定的 held-out instances 上做，不能看完错误位置再设计一个新的加权指标，然后在同一数据上报告优越性。更不能用这个指标重新挑表。

它的贡献是把“兼容性不能看频率 MAE”推进到“兼容性还必须考虑输出决策裕量”，并与真实完整生成连接，而不是提出一个未经比较的通用新 loss。

## 15.3 已有 8B / 多尺度资料的合理使用

若仓库有完整 owner、raw outputs 与 checkpoint/table receipts，可以作为较大模型的补充证据，明确它们属于历史协议。只分析已有张量通常远比现在临时训练一个 7B 更有 ROI。[I07]

但旧 8B 结果与本轮 OLMo 配对设计不匹配时，不能混进同一统计比较；旧某次 PPL 改善而 exact match 为零，可以支持现象存在，不能替本轮证明特定模块机制。

## 15.4 本轮不投入的扩展

不启动新的 recurrent/relational positional framework，不转多模态 token compression，不为了文章变“宏大”加入 spectral trilemma 的独立大项目，不将每头 learned RoPE 作为未检验的逃生路线，不拿更大模型堆算力掩盖小模型的因果缺口。

PPE 的启示是清晰定义被保存的结构并做相匹配的任务验证；不是提示本论文必须加视频/图像实验。Deconstructing Positional Information 的启示是理论预测要落到能区分假设的任务；不是直接借用它的 deposit-head 故事解释所有模型。[I08, I09]

## 15.5 不同结果下论文如何收敛

| 本轮结果 | 可以主张 | 不能主张 |
|---|---|---|
| z 因果识别稳定；单表有收益；生成迁移成功且 Native 过线 | 生命周期一致的 allocation 研究，带可用的小数据转换实例 | 通用最优位置编码/无限长度 |
| 单表有收益，生成未修复 | 明确几何—功能—生成分离与成熟模型兼容边界 | 已解决长上下文生成 |
| 微调有效，Z 与 Y 或 N 同样好 | 数据/函数约束有价值；Z 不是必需底座 | z 提供独特训练效率优势 |
| 只有训练族有效 | 任务族内、未见实例的远程能力迁移 | 通用 long reasoning |
| 旧结果无法复核，核心新结果也弱 | 缩小主张、诚实报告，重新评估稿件成熟度 | 以更多理论术语或图表替代证据 |

没有任何一行对应“保证录用”。这份计划优化的是证据质量、可解释性与投入，不是一个可估计的录用概率模型。

# 16. ICLR 2027 时间线与最终 Verdict

## 16.1 已核对的官方时间和规则

截至 2026-09-04，官方列出的摘要截止为 **2026-09-18 23:59 AoE**，全文为 **2026-09-25 23:59 AoE**。北京时间分别是 **9 月 19 日 19:59**、**9 月 26 日 19:59**。[R30]

摘要必须真实反映论文；摘要截止后不能新增作者，作者顺序可在规定范围内调整。需要提前核对真实合作者的 OpenReview 资料与 reviewing 资格，不为满足资格而挂名。若旧稿仍在 NeurIPS 审理，官方 FAQ 允许先提交 ICLR 摘要，但全文须另按双投政策和当时状态检查。[R30]

ICLR 2027 要求披露 AI 在研究与写作中的使用，包括理论/假设、实验设计、实现等相关协助；作者对最终内容负责。本项目应保存审查记录，而不是在声明中写没有实际完成的人类验证。[R32]

## 16.2 从今天起的建议安排

| 日期 | 唯一主要交付 | 截止后不再做什么 |
|---|---|---|
| 9/4–9/6 | E0、冲突数值对齐、数据与表冻结、24-instance diagnosis | 不继续无限整理历史、不加新表 |
| 9/7–9/9 | 第一 seed N/Z/Y 完整协议与 feasibility 报告 | 不换 rank/loss/gain “救火” |
| 9/10–9/12 | 固定复制 seeds；并行整理已有 fixed-support 图表 | 不开启新研究分支 |
| 9/13 | 锁定 checkpoints 与所有统计规则，运行更远盲测 | 不看 32K 后改 64K 配置 |
| 9/14–9/16 | 主图、主表、论文 claims 与 owner 对齐 | 不用 speculative theory 填实验空缺 |
| 9/17 | 完整可读初稿及真实摘要；作者资料确认 | 不把 placeholder 当占坑 |
| 9/18 AoE 前 | 正式摘要提交 | 不再新增作者 |
| 9/19–9/22 | 复核、必要的既定控制、局限、匿名代码包 | 不追加方法搜索 |
| 9/23–9/24 | 投稿 PDF、附录、引用与双投状态检查、提前上传 | 不把最后一天当实验日 |
| 9/25 AoE 前 | 正式全文截止 | 以官方系统状态为准 |

若某阶段落后，先删可选扩展，不推迟基本可信性检查。更远最大长度不是阻止写出完整论文的理由。

## 16.3 四个 Verdict

**Verdict A — Zero Training：FUNDAMENTALLY UNDERDETERMINED。** 对“仅凭 Native 信息推出唯一真实 long optimum”而言，缺少 long task distribution 与效用。可推导 Native 几何、条件最优结构和边界；本轮采用已有固定表作可复核工作点，不冒充普适最优。

**Verdict B — Minimal Finetuning：固定位置底座 + 全线性 r16 LoRA + 实际部署 Native 功能约束 + 合法 counterfactual 的完整轨迹监督。** 最小充分模块与秩尚未识别；本轮选择覆盖寻址、写入和后续计算的低秩预算。成功后只报告该预算足够，不声称数学最小。

**Verdict C — 更远泛化：有条件可泛化。** 不变量与共享计算提供理由，联合相位、干扰数量和任务复杂度构成限制。严格区分物理长度、position range 与模型预训练历史；所有更远结果盲测。

**Verdict D — 研究投入：MINIMAL-FINETUNING FIRST。** 指新增研究预算投入生成迁移，而不是继续搜索零训练曲线。执行上 E0 与已有单表确认必须先完成，它们是确保训练命题有效的前置条件，不是另一次 zero-training 搜索。

**最终研究问题：不是“再找一个看起来更合理的 z”，而是“在有序耦合和真实 Native 约束下，固定几何究竟恢复了哪一段计算，少量学习是否能补上剩余的决策轨迹”。**

---

# 附录 A. Codex 审查与交付合同

## A.1 开始写训练代码之前

确认 live repository 的当前 owner 与本稿资料是否一致；打印冲突而不是自行选择。旧 AGENTS 的 ≤4K backward 约束与当前用户允许 ≤16K 不一致，须在本轮实验 scope 中显式更新，不把旧 4K+长 position IDs 方案偷偷算成严格 16K 外推。[I05]

Codex 首次交付应是一个短的审查结果：`PASS/BLOCKED`，列出缺失资产、会影响方法语义的代码差异、最终 manifest、实测资源需求及计划执行的固定 matrix。允许指出本文推导或工程选择存在问题；不能默认本文每个假设均成立。

## A.2 可以自行做的修复

修复可复现 bug、off-by-one、重复 gain、cache 重置、数据泄漏、错误 tokenization、未按声明冻结的参数、完全等价的内存优化。每项修复有单元测试和前后差异收据；受影响旧结果失效重算。

## A.3 不能自行改变的研究内容

新 frequency curve、新的 gain/scale 选择、head/layer selector、新 LoRA target/rank、目标长度位置曝光、训练实例数与任务族、阈值、训练时长、主 endpoint、train/test 切分和完整生成判定。需要改变时停止正式 run，给出明确理由和新协议版本，不能沿用未暴露的名义。

## A.4 每轮结果报告的必要五句话

1. 本轮到底改变了什么，唯一对应哪个假设？
2. 与哪个 matched control 比较，哪些量真的匹配，哪些没有？
3. 实际 Native 与完整生成结果是什么，不只汇报 NLL？
4. 最先断掉的是任务可解性、背景鲁棒性、远端 source dependence、margin 还是终止？哪些仍无法区分？
5. 根据事先规则，下一步是继续固定复制、停止、还是标记 inconclusive？

不能只写“有信息量”“发现长程 backbone”“值得继续”而不给读数和边界。

# 附录 B. 预注册配置摘要

```yaml
protocol: hybrid_rope_single_table_transport_v1
status: design_not_executed
as_of: 2026-09-04

primary_model: allenai/OLMo-2-0425-1B-Instruct
secondary_model: Qwen/Qwen2.5-1.5B-Instruct
primary_table: existing_log_s4_exact_owner_export
primary_table_hash: BLOCKED_UNTIL_READ_FROM_ACTUAL_BYTES
routing: none
dynamic_scale_changes: none
head_or_layer_selection: none

zero_training_arms: [N, Z, G_same_sampled_support, official_YaRN, official_MrRoPE_Pro]
training_arms: [N, Z, official_YaRN]
seeds: [42, 43, 44]

trainable: all_attention_and_mlp_linear_lora
rank: 16
lora_alpha: 16
lora_dropout: 0
frozen: [base_weights, norms, embeddings, lm_head, frequency_table, gain]
attention: exact_dense_causal_memory_efficient
quantization: none

train_semantic_instances: 128
counterfactual_worlds_per_instance: 2
max_total_sequence_lengths: [2048, 8192, 16384]
max_position_id: 16383
virtual_target_positions: forbidden
training_views: 768
training_passes: 1
restoration_steps: 32
transfer_steps: 96
native_batch_restoration: 8
task_effective_batch_transfer: 8
native_batch_transfer: 2
microbatch: 1
learning_rate: 0.0001
adam_betas: [0.9, 0.95]
weight_decay: 0
clip_grad_norm: 1
warmup_fraction_each_stage: 0.1
optimizer_reset_between_stages: true

long_loss: answer_and_eos_ce_plus_0.25_worst_gold_prefix_margin_hinge
teacher_margin_cap: 1.0
native_constraint: actual_deployed_student_vs_original_teacher
native_sampled_prefix_KL_budget_each_group: 0.02
native_NLL_delta_budget_each_domain: 0.03
native_macro_generation_noninferiority_margin: 0.02
native_group_generation_noninferiority_margin: 0.05
native_nontermination_increase_budget: 0.02

saved_steps: [0, 32, 64, 96, 128]
selection: native_feasible_then_16k_complete_generation_then_lower_KL_then_earlier
primary_endpoint: native_gated_16k_complete_generation_improvement_over_Z0
blind_lengths: [32768, 65536]
reveal_32k_64k_together: true
bootstrap_unit: semantic_instance
report_all_seeds: true

optional_explanatory_controls:
  - Z_compact_only_matched_answer_presentations_seed42
  - Z_attention_only_rank46_parameter_matched_on_OLMo_seed42
optional_controls_are_not_method_candidates: true

budget_gpu_hours_cap: 72
budget_is_not_runtime_prediction_or_spending_authorization: true
```

所有 `BLOCKED` 字段必须在正式运行前解决。它们是资产事实，不是根据成绩优化的自由度。

# 附录 C. 内部资料索引与证据权限

本稿中的 [Ixx] 为可移植的内部引用。Codex 应按文件名、日期、章节和 file ID 交叉定位，再找 owner/原始产物。Library file ID 不是服务器路径；不能将其拼成磁盘地址。

**[I01] `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831(2).md`**  
Library file ID：`file_00000000ed9c81f5a5c71aa4b5bf7899`。重点：log/arithmetic 公式、冻结 m、gain ablation、full-13 结果与 hash。报告给出的 result JSON SHA-256 为 `ff8ebb9488da4ebdbc3b1442093c21b3226a901704e11aa17521e381a773357e`；这是结果文件 hash，不是表 hash。本稿等级 E1。

**[I02] `ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md`**  
ID：`file_000000003ad481f5ae381b13835ef66d`。重点：旧训练失败、C2 几何距离与 Native 退化、跨模型迁移、s4 汇总值、远端 receipt 缺口、后处理失败。文件自述为聚合汇总；不同条目的 owner 完整性不同。本稿等级 E2。

**[I03] `Hybrid-RoPE_ZERO_TRAINING_RESEARCH_DOSSIER_20260827.md`**  
ID：`file_00000000858081f5bbcb326184cb0a39`。重点：运行静态与跨请求 routing 的区别、gain 的 Q/K amplitude 定义、one-turn floor、piecewise position、Q/K 与 Q/K/V/O repair 失败、早期结果失效。属 owner 引用型 dossier，不是所有 raw files 已经被本稿验证。

**[I04] `hybrid_rope_session_handoff_20260903(1).md`**  
ID：`file_00000000c57881f5b0eb9d40a2041024`。重点：论文定位、ordered coupling、head sensitivity preflight 与尚未完成的 causal pilot。区分真实观察与计划，不能把 pilot 表当结果。

**[I05] `hybrid_rope_AGENTS.md`**  
ID：`file_0000000023f8822fa5f2ed63b0990eb9`。重点：旧 ≤4K scope、谱系规则、matched controls、真实 counterfactual、raw generation/EOS、Native 保留、停止重复失败。旧约束按当前用户请求显式更新，不把历史文档当作高于当前任务的指令。

**[I06] `main(20260827-115835).pdf`**  
ID：`file_00000000dac481fdb18be68c45bc2eb3`。本稿核对其摘要及开篇理论/实验主线，而非声称它等于 live repository 最新正文。重点：RoPE Has a Spectral Budget、finite Gram、固定 support 多 seed、geometry/function dissociation。旧 NeurIPS 稿与该稿不能混为同一版本。

**[I07] `粘贴的 markdown (1)。md`，readout/conversion 审计文本**  
ID：`file_00000000883c81fd8c204334abd232e2`。重点：oracle routing、source effect 与生成分离、已保存但未完整分析的 layerwise tensors、task-family transfer 边界。引用的是审计文本；必须回到其中 run/manifest 才能升级为原始证据。

**[I08] `3159_Deconstructing_Positional.md`**  
ID：`file_00000000ae4881f59d7d6c688baaa76d`。上传的 ICLR 2026 论文，题名 *Deconstructing Positional Information: From Attention Logits to Training Biases*。用于内容—位置耦合与定向因果任务的相关工作，不是本项目实验。

**[I09] `10300_PPE_Positional_Preservat.md`**  
ID：`file_00000000cca881f5b79fb60dc1e8f69c`。上传的 ICLR 2026 论文 *PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models*。只用于范围与实验设计启示，不延伸成新多模态项目。

**[I10] `5551_MrRoPE_Mixed_radix_Rotary.md`**  
ID：`file_000000009be481f580cb8d39cf8c0030`。上传的 ICLR 2026 MrRoPE 论文，与公开 [R10] 交叉核对。

**[I11] `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`**  
ID：`file_00000000fcd0820c922dd30504142ee7`。作者保存的评审/AC 文本记录，含来源与完整性说明。用于识别被明确提出的拒稿风险；不在本稿宣称该文件证明当前投稿的最终官方决定。

**[I12] `粘贴的 markdown (1)。md(20260904-194700)`**  
ID：`file_00000000410081f598686db72e368da8`。上一轮分析稿，不是新增独立实验或文献证据。本稿第 1 节主动修正其中的 single-table、retention、初始化与训练设计问题。

# 附录 D. 公开文献与官方来源

以下编号对应正文。查询日期 2026-09-04；涉及模型/政策/新预印本时以列出的版本为准。来源地址采用代码形式，便于移交和版本核验；不把公开摘要的结论扩张成本项目保证。

**[R01] Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding.**  
`https://arxiv.org/abs/2104.09864`  
用途：真实 RoPE 相对位置计算。频率 multiset 的独立解释需要结合内容系数，不能仅引用 RoPE 原论文推出本项目的新结论。

**[R02] Barbero et al. — Round and Round We Go! What makes Rotary Positional Encodings useful?**  
`https://arxiv.org/abs/2410.06205`  
用途：成熟模型高/低频的不同使用方式，反对把所有 slow dimensions 视为无用。观察的模型范围不能自动泛化到全部 checkpoints。

**[R03] Chen et al. — Extending Context Window of Large Language Models via Positional Interpolation.**  
`https://arxiv.org/abs/2306.15595`  
用途：PI 的缩放/适配基线。NTK-aware scaling 在本文只作为相关缩放族，不把名称当作严格的神经切线核最优性保证。

**[R04] Peng et al. — YaRN: Efficient Context Window Extension of Large Language Models.**  
`https://arxiv.org/abs/2309.00071`  
已核对页面版本为 v3（2026-02-06）；历史 v2 的训练描述不可无标注混用。官方实现、gain convention 与工作长度须固定，不靠本轮 outcomes 调参数。

**[R05] Ding et al. — LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.**  
`https://arxiv.org/abs/2402.13753`  
用途：非均匀缩放/搜索与渐进扩展背景，不是 Native-only 的唯一 optimum。

**[R06] Shang et al. — LongRoPE2: Near-Lossless LLM Context Window Scaling.**  
`https://arxiv.org/abs/2502.20082`  
用途：needle-driven 搜索与 mixed-context training 的条件；约 10B-token 预算与本轮 small-data 设定不同。

**[R07] Wang et al. — AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally.**  
`https://arxiv.org/html/2607.19363v2`  
用途：group frequency/head gain 异质性，冻结 backbone 但优化位置参数的条件。§5.2 提及前 100 个 PG19 样本，附录训练长度 65,536；不是严格 ≤16K unseen-position 实验。两者必须一起看。

**[R08] Karypis et al. — LeRoPE: Learnable RoPE Frequencies Improve Language Modeling.**  
`https://arxiv.org/abs/2607.10134`  
用途：联合学习频率的近邻，不能再声称频率可学习本身是空白；与 mature frozen retrofit 的学习条件不同。

**[R09] Oka et al. — Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off.**  
`https://openreview.net/forum?id=PR1PPxvG9Q`  
ICLR 2026 / FMRoPE 相关工作。公开检索与已有稿件确认相关性；OpenReview 正文抓取遇到 browser verification，具体实现按作者论文/代码另行读取，不在本文伪造其完整公式。

**[R10] Tian et al. — MrRoPE: Mixed-radix Rotary Position Embedding.**  
`https://arxiv.org/abs/2601.22181`  
用途：直接的 training-free 非均匀 scaling 对照；另有上传全文 [I10]。

**[R11] Wang et al. — Resonance RoPE: Improving Context Length Generalization of Large Language Models.**  
`https://arxiv.org/abs/2403.00071`  
用途：周期性位置特征与 PosGen 的问题分离。正文关于联合坐标覆盖的量词检查是本稿分析，不声称该论文已经提供完整函数不变性。

**[R12] Li and Zhang — Context Length Extension via Generalized Extrapolation Scale.**  
`https://aclanthology.org/2024.findings-acl.249.pdf`  
用途：GeNE 的 scale randomization。训练设定见论文第 4 页：16K、global batch 128、300 steps；token 数为按该设定的上限计算，不是另行测量。

**[R13] Mehta, Yin, Durrett — Randomized YaRN Improves Length Generalization for Long-Context Reasoning.**  
`https://arxiv.org/html/2606.23687v1`  
用途：短物理序列的任务适配与位置 curriculum。§2.2 明确随机采样更大 position range，§3 的模型配置 Native windows 已为 32K/64K；不能当成原生 4K 模型的严格 unseen-position 证明。

**[R14] Zhu et al. — PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training.**  
`https://arxiv.org/abs/2309.10400`  
用途：区分 token 数与 position range。

**[R15] Chen et al. — CLEX: Continuous Length Extrapolation for Large Language Models.**  
`https://arxiv.org/abs/2310.16450`  
用途：连续长度变换的相关方向，不把 ODE 参数化当作未知长度上的普适正确性保证。

**[R16] Martens — New Insights and Perspectives on the Natural Gradient Method. JMLR, 2020.**  
`https://jmlr.org/papers/v21/17-678.html`  
用途：Fisher/GGN、trust region、empirical Fisher 与参数化问题。本文的局部式有明确条件，不据此声称全局最优。

**[R17] Zeng and Lee — The Expressive Power of Low-Rank Adaptation.**  
`https://arxiv.org/abs/2310.17513`  
用途：低秩适配表达能力的条件结果；不是本文 rank16 或特定模块足够的证明。

**[R18] Biderman et al. — LoRA Learns Less and Forgets Less.**  
`https://arxiv.org/abs/2405.09673`  
用途：learning/forgetting trade-off，反对把 PEFT 当作无遗忘保证。

**[R19] Kirkpatrick et al. — Overcoming catastrophic forgetting in neural networks.**  
`https://arxiv.org/abs/1612.00796`  
用途：EWC 的重要方向约束背景。

**[R20] Li, Grandvalet, Davoine — Explicit Inductive Bias for Transfer Learning with Convolutional Networks. ICML 2018.**  
`https://proceedings.mlr.press/v80/li18a.html`  
用途：L2-SP，区分参数距离先验与实际模型函数保留。

**[R21] Chen et al. — LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models.**  
`https://arxiv.org/abs/2309.12307`  
用途：长上下文 PEFT 的工程可行性及其 sparse training / embedding / norm 条件；本轮不复制这些额外变量。

**[R22] LongReD — Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation.**  
`https://arxiv.org/abs/2502.07365`  
用途：位置扩展后的短程恢复已是已有研究方向。

**[R23] Yang et al. — LinearARD: Linear-Memory Attention Distillation for RoPE Restoration, v2.**  
`https://arxiv.org/html/2604.00004v2`  
用途：final-attention-layer Q/Q、K/K、V/V relations 蒸馏与线性内存实现。v2 报告 4.25M-token 实验与相对 Native 的约 93–94% 短程恢复；本文不将其近似恢复表述为无损，也不混用其他版本/参照基线的百分比。

**[R24] Wu et al. — Retrieval Head Mechanistically Explains Long-Context Factuality.**  
`https://arxiv.org/abs/2404.15574`  
用途：retrieval heads 的消融证据，不提供按 Native sensitivity 选择频率修改对象的最优规则。

**[R25] Hsieh et al. — RULER: What's the Real Context Size of Your Long-Context Language Models?**  
`https://arxiv.org/abs/2404.06654`  
用途：完整 task-family 覆盖与 benchmark 定义，不能用 single-needle 代替全部能力。

**[R26] AllenAI — OLMo-2-0425-1B-Instruct 官方 config。**  
`https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/raw/main/config.json`  
查询值：hidden2048、intermediate8192、16 layers、16 attention/KV heads、window4096、theta500000、vocab100352。实际实验须 pin revision，不依赖可变 main。

**[R27] Qwen — Qwen2.5-1.5B-Instruct 官方 config。**  
`https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json`  
查询值：hidden1536、28 layers、12 query heads、2 KV heads、window32768、theta1000000。配置上限不是完整预训练数据谱系证明。

**[R28] NVIDIA — 4080 family / RTX 5090 官方产品规格。**  
`https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/`  
`https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/`  
用途：标称 16GB 与 32GB 的区别；租赁/改装设备以实际检测为准。

**[R29] Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.**  
`https://arxiv.org/abs/2205.14135`  
用途：避免完整 attention matrix 的精确算法思路，不把论文年份当作当前 5090 软件兼容版本。驱动/backend 支持由服务器测试并固定。

**[R30] ICLR 2027 — Call for Papers / Author Guidelines。**  
`https://iclr.cc/Conferences/2027/CallForPapers`  
`https://iclr.cc/Conferences/2027/AuthorGuidelines`  
用途：摘要/全文截止、9 页正文、作者与真实摘要、双投及补充材料规则。提交前再次检查官方更新。

**[R31] ICLR 2027 — Reviewer Guidelines。**  
`https://iclr.cc/Conferences/2027/ReviewerGuidelines`  
用途：问题、定位、证据与新增知识，而不是以 SOTA 或模型尺寸作为唯一价值判据。

**[R32] ICLR 2027 — AI Policy for Authors。**  
`https://iclr.cc/Conferences/2027/AIPolicyForAuthors`  
用途：AI 协助研究/写作的披露及作者责任。声明应反映实际使用和实际核验，不承诺未执行的审查。

---

**结束条件：** 本稿不是邀请 Codex 再生出十个候选方案。它要求把两个研究问题压到可辨识的对象、固定的干预、合法的标签、真实的输出与可接受的 Native 代价上。未经验证的更长倍率、漂亮但不对应模型函数的几何指标、以及换名后的旧训练配方，都不计作完成。
