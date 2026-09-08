# Hybrid-RoPE：回到零训练优化与 LoRA 外推

**计划日：下一工作日（按当前会话日期为 2026-09-06）**  
**依据：最新 `SPECTRAL_BUDGET_LORA_20260905.md`、两版服务器报告、原论文及本轮核验的公开原始论文。**  
**算力边界：全部后续工作最多 100 GPU 小时；首日最多 20 GPU 小时。两张卡同时运行按小时相加，不把双卡一天算成一个 GPU 日。**

本文是新的研究与执行方案，不是已运行结果。未连接服务器，未核验实时仓库、当前模型资产、吞吐或显存。所有旧负结果保留。文中用“报告事实”“推导”“工作假设”“工程设定”区分证据地位。

## 0. 先改正方向，不再追加诊断主线

当前用户的两个主目标并列：

1. **零权重训练：继续寻找能保住 Native、改善真实长生成的单一全局静态表。**
2. **有限训练：让小模型在固定长位置底座上形成可用的窗口外能力，并进一步验证超出适配长度的泛化。**

最新附件将两项目标都改写成零训练，并把 LoRA 降为次级补救；这不作为本轮指令。之前把格式盲标、FFN 必要性、逐项零损失和层间机制审查放在方法验证之前，已经失去优先级。这些工作不再阻止下面两条主线。

本轮不承诺理论保证成功，也不再把“理论不能给唯一全局最优”当作暂停方法研究的理由。做法是：明确必要条件，采用最接近公开有效训练条件的固定方案，以两个训练臂检验它；零训练只增加一个公开静态对照和一个更强的原生短窗 checkpoint，不发明频率曲线。

**明日核心交付：一个零训练结果表、一个 Z/Y 同配方适配结果表。不是新的审查平台。**

---

# 1. 最新报告究竟说明了什么

## 1.1 现在已有的不是“一概失败”，也不是“表定死了内容上限”

以下是附件汇总，不是本轮重新计算的 raw 结果。[I1]

| OLMo，Native 4K，测 16K | far 双世界完整精确成功 | near 成功 | far EOS | gold 字面出现，诊断项 |
|---|---:|---:|---:|---:|
| T0：原表、不训练 | 0/32 | 0/32 | 0 | 0/64 |
| Z0：固定 Z、不训练 | 1/32 | 3/32 | 0.62 | 26/64 |
| ZC：Z + compact-only 适配 | 1/32 | 4/32 | 0.70 | 27/64 |
| ZF：Z + 完整长输入适配 | 4/32 | 16/32 | 0.91 | 29/64 |
| ON：原表 + 完整长输入适配 | 0/32 | 0/32 | 1.00 | 0/64 |

可以据此判断：

- 当前计算预算和任务配方下，改变位置底座产生了实际效果；但 Z 不是合格的完整解决方案。
- 在 Z 上增加长输入训练，near 改善明显，far 改善有限。距离/布局仍与计算发生强交互。
- ON 已会正常停止却完全答错，排除了“只修 EOS 即可解决”的解释。
- 26→29 的字面出现数量，不是冻结表的理论内容上限，不证明已有信息被正确绑定，也不证明换更合理的训练就无效。
- 不能由一个 ON 失败推出“所有不改频率的训练都不可能”；它只限制本次 recipe。

Qwen 的 Z/Y/compact-only 对照已经完成，不能再把 N_compact 当成明日待办。它们主要测试 **32K 原生窗口内的 16K 输入适配**，不提供原生窗口外扩展证据。[I1]

## 1.2 一个应立即撤销的额外失败条件

OLMo compact 的 27→25，相应计数比为 92.59%。这不是“逐例无遗忘”，但也不是低于 88%。

如果实验原先另立“compact 一个也不能丢”的标准，原判定保留；**下一轮不要再把逐例零损失自动升级为用户目标，更不要用它叫停超过 88% 的候选。**

同样，总体 Native ≥88% 与每个子任务均 ≥88% 是不同要求。保持原来已声明的总体标准，完整报告分项，不暗中增加新否决项，也不把总体通过包装成所有能力无损。

若PPL retention定义为PPL_native/PPL_candidate，那么88%等价于ΔNLL≤−log(0.88)≈0.12783 nats/token。此前额外使用0.03或0.01的NLL容忍度更严格，不能默认为用户要求。行为指标与置信区间仍需独立报告。

## 1.3 目前还没有测试到的关键条件

没有材料证明已经执行了下列完整组合：

> OLMo 的 Z 与 YaRN 固定长表 + 充分覆盖长位置的普通 next-token 持续训练 + 有效的低成本参数自由度 + 少量内容依赖的任务适配 + 同表 Native 保留。

旧 Q/K、Q/K/V/O、全线性答案微调和从头训练 YaRN，都不等于这个组合。新方案如果成功，也不能仅凭与旧结果的差别，把功劳全归给其中一个组件。

---

# 2. 与 YaRN、LongLoRA 对齐：此前漏掉的三个条件

## 2.1 “少量”是相对原始预训练，而不是只有几千个答案标签

YaRN 原论文 §4.1 的 Llama-2 s=16 配置：400 updates、global batch 64、64K 物理序列，普通语言建模持续训练。按其设置计算：

$$400\times64\times65536=1,677,721,600.$$

约 16.8 亿 token。它没有声称几百个短答案即可替代这种训练。[R1]

你们最近配方有约 632.8 万输入 token，但只有 3,642 个 answer/EOS 监督位置，平均每个视图约 4.74 个。[I2]

这不意味着前文没有梯度；答案 loss 能沿 attention 反传。但两者对整个长序列内部计算的训练覆盖非常不同。

**结论：目前未成功，不足以否定小模型的长上下文适配；首先不能把当前实验称为 YaRN 式训练已经失败。**

还要改正一个过去的逻辑跳跃：**NLL 不能单独证明生成能力，不等于不该用 NLL 作为主要训练目标。** 应当用合适的训练目标学习，再用独立生成任务验证，而不是为了避免 NLL 代理误读，反过来删掉本来需要的语言建模适配。

## 2.2 YaRN 的 64K→128K 不是“原 64K 表原样继续往外跑”

YaRN 先得到 s=16 的 64K 配置；随后安装 s=32、从此前 checkpoint 再训练 200 steps，物理数据仍为 64K。其 Table 1 中，s=16 系统在 128K 的 PPL >10，s=32 系统显著更好。[R1]

因此需要分开记录四个长度：

| 符号 | 含义 |
|---|---|
| L0 | 原始 checkpoint 的训练/配置窗口，不能任意缩小分母 |
| Lt | 本次适配最大物理长度 |
| Ldesign | 此部署表在训练前确定的目标长度 |
| Leval | 评估长度 |

本轮先得到 **L0=4K、Lt=16K、Ldesign=16K** 的可靠 4× Native 扩展。再用预先定义的 **Lt=16K、Ldesign=32K** 续训验证 32K。这两个产物各自始终使用一张表；不能称它们合起来是同一个永不变表的系统。

第一阶段的 16K 是真实 Native 窗口扩展，但不是超出微调长度。第二阶段的 32K 才同时超出 Native 与适配物理长度。

## 2.3 all-linear LoRA 仍然不是公开配方里的全部有效自由度

LongLoRA 的 Table 2 在其 Llama-2-7B、32K、S²-Attn 设置中，普通 LoRA 扩大 rank 未消除与 full finetuning 的差距；rank 8 加 embedding 与 norm 得到 PPL 8.12，接近 full FT 的 8.08，普通 rank 8 为 11.44。[R2]

这不是 OLMo 的充分条件定理，也不是 norm 必要性的跨模型证明。但它比继续猜 FFN 层位更直接地支持一个执行选择：

> 在低秩适配之外，允许 input embedding 与所有既有 normalization affine 参数更新。

本轮不引入 LongLoRA 的 shifted-sparse attention。始终使用标准 dense causal attention 的精确内存高效实现。因此应称 **受 LongLoRA 启发的 PEFT 参数设置**，而不是复现了完整 LongLoRA。

---

# 3. 第一性原理：训练究竟要修复什么

## 3.1 RoPE 决定了坐标；成熟模型学到的是带内容系数的计算

一个 head、固定当前层输入时：

$$s_{ij}=\frac{a}{\sqrt d}\sum_k\left[A_{ijk}\cos(\omega_k\Delta)+B_{ijk}\sin(\omega_k\Delta)\right].$$

A、B 来自 Q/K 的真实内容。优化 positional basis，不等于优化这些系数形成的任务计算。[I3]

零训练只能改变位置响应；训练则能改变内容匹配、信息写入与后续使用它的映射。这不预设瓶颈在 FFN。

对于固定 q、k 与 gain，改频引起的直接 logit 变化满足：

$$|\delta s_{ij}|\le\frac{2a}{\sqrt d}\sum_k\|q_{ik}\|\|k_{jk}\|\left|\sin\frac{\Delta\delta\omega_k}{2}\right|.$$

这是二维旋转差的算子范数界。它解释了为什么小 frequency MAE 不能保证小功能损伤；它不包含前面层改动导致 q/k 本身变化的额外项，不能拿来认证全网。

原生保持与长收益需要共同可行，不由频率表平均距离决定。当前四个 OLMo 候选失败并没有排除其它表或其它成熟 checkpoint。

## 3.2 监督短答案时，优化可能优先走容易的路径

对某个 attention 节点：

$$\frac{\partial L}{\partial s_{ij}}=\alpha_{ij}\left\langle\nabla_{o_i}L,v_j-o_i\right\rangle.$$

如果远证据几乎没有获得 attention，且其余因素有界，相关局部梯度会受抑制。模型可能先学到回答格式、常见答案偏好或 EOS，这些路径更直接。

这是条件性机制解释，不是已经测得当前模型全部远程梯度接近零。跨层路径和反向梯度会改变实际大小。

**设计后果：不要只让一段 16K 输入末尾的几个 token 承担全部坐标适应。先用整段语言建模适配计算分布，再用明确依赖远证据的监督形成可用行为。**

## 3.3 Dense LM 有必要的作用，但不自动教会远程依赖

在真实分布上，给定近上下文 C、远证据 E 和目标 Y，最优 log-loss 改善为：

$$H(Y\mid C)-H(Y\mid C,E)=I(Y;E\mid C).$$

若远端几乎没有额外信息，普通 LM 可以通过局部捷径下降。LongFilter 等数据研究也明确区分长度与长程信息量。[R3]

所以本轮既不是“只加 128 个随机 prefix 标签”，也不是“纯 PG19 NLL 下降就算成功”。使用两个互补阶段：

- 完整长文本 LM：适配被换表影响的分布式计算。
- 单证据/绑定任务：直接要求输出使用远处内容。

数据中的真实因果依赖必须成立；自然文本本身不能保证每个 token 都依赖很远。无需为此新建长文本打分/筛选研究项目。

## 3.4 位置范围与竞争 key 数量是两个不同外推变量

一个证据分数为 u，N 个干扰分数均为 v 时：

$$p_E=\frac{1}{1+N e^{v-u}}.$$

保持相同证据质量需要 u-v 随 log N 增长。虚拟 position gaps 不会增加 key 数；纯相位曝光不能代替真实长输入的竞争。

这个简化模型不决定所有 heads 的最优 temperature，也不允许由此发明统一 gain 公式。

**本轮主适配用真实 16K tokens；更远测试保持任务关系复杂度不变，只增加距离和干扰。**

## 3.5 为什么开放 norm 不是“随便再加一些参数”

简化的 RMSNorm 后接线性映射为：

$$W\,\operatorname{diag}(\gamma)\frac{h}{\operatorname{rms}(h)}.$$

改变 γ 的直接效果含 W diag(δγ)，一般不是 rank-16 更新。少量 affine 参数可以改变很多通道的尺度关系，而不是在一个单层线性映射上只提供 rank-r 的任意残差。

这只说明参数化不同；不证明整个多层 LoRA 网络无法间接近似，也不证明 norm 对每个 checkpoint 都必要。

RoPE 本身保范。需要适配的不是“旋转把向量长度弄坏了”，而是相位加权的内容匹配、信息混合和后续表示分布。开放 norm 的选择由这一表达差异与 LongLoRA 实证共同支持。

input embedding 的支持主要来自公开实验，不从 RoPE 公式强行推导其必要性。本轮开放它是为了先得到有效工作点，成功后再按预算缩减自由度。

## 3.6 为什么不能把初始不兼容都记为 forgetting

总体 Native 变化包含：

$$\text{换表即时损伤}+\text{适配后相对换表起点的变化}.$$

后一项既可能恢复也可能继续退化。全部 Native 比较必须对照原权重、原表，student 则始终使用部署表。

平均 KL 与任务保留不等价，所以最终仍需 Native 行为测试。但明日不新建逐 token 决策证书平台，直接在训练中使用已有框架能实现的函数蒸馏与独立行为评估。

---

# 4. 失败归类：哪些解释成立，哪些不能下定论

| 失败/不足 | 理论上最可靠的解释边界 | 对下一步的含义 |
|---|---|---|
| OLMo Z/Y 及单位幅度均未过联合 Native | 这些有限配置未找到联合工作点；gain 与频率有交互 | 不重跑单位幅度，不宣布静态表类别无解 |
| 频谱 rank/coverage 很好、RULER 却差 | 几何指标没有纳入成熟内容系数、竞争和决策 | 关闭代理排序法，不再挑最大 rank 表 |
| 保 multiset 但 permute slots 坍塌 | 学到的内容子空间与具体频率绑定 | 不做无序频谱优化；不把 permutation 当无害随机化 |
| Q/K、Q/K/V/O repair 的 NLL 好但生成坏 | 平均 log-loss 能改善大量非决策 token；寻址和输出是不同条件 | 新协议直接评估生成，但不让几个答案 token承担全部适配 |
| Qwen compact-only 已能改善 16K | 该评估在 Native32K 内，可能主要改变已有能力的调用 | 结束这条归因支线；不拿它否定窗口外训练需求 |
| OLMo ZC≈Z0，ZF 主要改善 near | 旧短监督没有建立足够稳健的远端计算；不能区分优化/数据/容量的全部原因 | 更换训练范式，而不是再加一次同配方 steps |
| ON 学会 EOS 但答案仍零 | 终止是可独立学习的更容易行为；不能补偿当前坐标下的内容错误 | EOS-only 不再是主改动 |
| YaRN 从头训练差于 Geo | 安装于随机初始化并学习，与成熟模型坐标适配不是同一个问题 | 不能由此排除标准持续训练的 YaRN 对照 |
| learnable τ 的某些初始化无梯度 | 可能存在坐标退化；已存审计具体记录 τ≈0.01 的 softplus 死区 | 原因不能统称“位置参数无可学信号”；明日不重启 τ 调参 |
| Power-Shift 差 6–22× | 历史材料将其放在 video DiT，非本轮 OLMo；方向亦可能与 Cosh 相反 | 不混入“文本 LoRA 全部失败”的因果链 |
| Wan2.1 LoRA GEO win | 当前只有结果摘要，缺具体训练/频率/评价协议 | 不能编造机制；明日不投入视频实验 |
| Native 总体高但部分格式/索引下降 | 分项损失真实，均值不能抵消解释；也不等于全面灾难遗忘 | 保持总体88%与分项披露，不追加逐项零损失硬门槛 |

[I1–I6] 支持表中事实。对没有原始记录的项，只分析适用边界，不把猜测记成已证原因。

## 4.1 learnable τ：一个真正能算清楚的旧失败

已有审计给出：

$$\phi_\tau(u)=u-\frac{\tau^2}{6}u(1-u)(2-u)+O(\tau^4).$$

若 τ=softplus(a)，当 τ 很小时：

$$\frac{d\tau^2}{da}=2\tau\sigma(a)\approx2\tau^2.$$

因此，“τ 几乎不动”可由参数坐标本身解释，不能直接升级成 task risk 对 allocation 没有信息。该审计有具体 τ_init=0.01 的历史记录；不能假设最新列出的每个 flat run 都用了同一初始化。[I4]

这项修正写入理论/局限即可，明日不花 GPU 修 learnable-τ，因为它不是当前最短成功路径。

## 4.2 Power-Shift：不能把相反方向的构造当成同一个实验

旧资料写 φ_PS(u)=1-(1-u)^(1+α)。对 α>0 和 0<u<1，φ_PS(u)>u；而正 τ 的 Cosh warp 使 φ_Cosh(u)<u。若采用 ω=B^(-φ)，两者移动方向相反。

这是由给定公式推出的结论，不说明 Power-Shift 的每个失败都是该原因，也不排除其它参数符号。需要原 run 确认 α 的取值。旧资料的 video 解释与强断言不直接搬到当前文本实验。[I6]

---

# 5. 模型大小与 benchmark：如何取舍

## 5.1 不是简单的“1.5B 太小，所以必败”

OLMo 与 Qwen 的实际参数量接近，却在当前 compact 能力与原始窗口上不同。不能把这些差异唯一归因于规模。[I2]

原稿 750M、500M-token full continuation 得到过受控 AR 检索结果，说明“小于 2B”不是这种任务能力的普遍禁止条件。但它不保证当前 Instruct + tiny LoRA 能复现。[I3，p23]

**训练主模型继续用 OLMo-2-0425-1B-Instruct。增加一个 7B 只做零训练/规模锚点，不先把全训练预算升级到 7B。**

## 5.2 新增模型只选 OLMo-2-1124-7B-Instruct

官方配置同样是 L0=4096、RoPE base=500000、head_dim=128，因此和 1B 有同样的默认 frequency profile。[R4–R5]

这有三个实际好处：原生4K→16K是真4×；大部分加载/算子路径可复用；可以测试相同频率构造面对不同成熟计算是否得到更好工作点。

不是纯参数规模因果实验：层数、宽度、训练和后训练历史仍不同。正式描述为同族不同规模的外部验证。

不要根据哪个 long 分数更好再从十几个模型中挑一个。该 7B 身份在看结果前固定。

## 5.3 评估问题的确被设得过重

“完整答案必须逐字一致 + EOS + 两个世界同时成功 + 很难的多证据推理”可以作严格压力测试，但不是唯一的 PE 能力定义。

若两个世界边际成功率均为 p、且近似独立，联合成功率为 p²；不独立时只能用 Fréchet bounds：

$$\max(0,p_0+p_1-1)\le P(\text{both})\le\min(p_0,p_1).$$

旧严格结果保持不变。下一轮增加标准生成任务指标并列报告，不再让格式/EOS 把内容能力完全遮住，也不把“字符串中出现答案”升级成语义正确。

**正文候选端点：RULER 生成检索 + 单证据自然 QA 的生成 EM/F1 + Native 常识/语言建模保留。**

多跳、复杂格式、长摘要终止仍报告，但不再单独承担所有训练是否继续的决定。

---

# 6. 明日路线 A：零训练的有界直接尝试

## 6.1 固定矩阵

| Checkpoint | N 原表 | 现有 Z | 官方 YaRN-s4 | MrRoPE-Pro-s4 |
|---|---|---|---|---|
| OLMo 1B | 精确匹配时复用 | 精确匹配时复用 | 精确匹配时复用 | 新运行 |
| OLMo 7B | 新运行 | 新运行 | 新运行 | 新运行 |

**只新增一个公开算法 MrRoPE-Pro，不创造本项目的新曲线。**它是静态、training-free 的已发表对照；没有保证它在 OLMo 上超过 Z 或保住88%。[R6]

其作用是同时回答：既有 Z 是否在更强底模上能直接工作；以及旧 Z/Y 的失败是否仍留有已知静态构造可利用的余地。

如果 MrRoPE 最好，归功于 MrRoPE；不能写成 Hybrid-RoPE 的新方法。如果 Z 在7B得到合格点，可以形成现有方法的实质新证据。若都不合格，不在当天继续增加第五张表。

## 6.2 表和幅度如何冻结

- Z：复用当前字节、m 向量与幅度来源；不临时将1.1026替换成另一较好值。两模型默认 profile 经实际核对相同后，可复用同一 Z 数值表；否则按原确定性定义重新导出，不能插值凑表。
- Y：采用已有、核验过的官方 s4 定义与默认幅度。
- M：按 MrRoPE-Pro §3.2 的已发表定义，使用原生窗口下32圈/1圈的物理分界及与其参考实现一致的幅度规则。该论文附录的 α/β 字母有易混读之处，manifest 直接记录“哪个是32圈、哪个是1圈”及实际索引，不能通过试分数消歧。
- 每个方法所有短、长请求使用同一张表/同一个幅度。绝不 Native routing，不修改 KV 存活期间的频率。

M 的核心是对中间 band 的相邻 radix 增量作递增分配，而非当前 Z 的 uniqueness mask。按一基索引，n=d_h-d_l：

$$\epsilon_j=\frac{2(1+j-d_l)}{n(n+1)},\quad\lambda_j=s^{\epsilon_j},\quad d_l\le j<d_h,$$

其它 λ=1，实际频率为：

$$\omega'_j=\omega_j\Big/\prod_{d<j}\lambda_d.$$

注意累计积，不是直接除以 λ_j。分界、索引和幅度是公共算法的固定实现，不用 long benchmark 拟合。[R6]

累计经过 r 个中间增量时，其归一化 log-frequency 位移为 r(r+1)/[n(n+1)]，不超过均匀 radix 分配的 r/n。它把更多移动留到较慢的中间通道；这是明确的结构差异，不是随手换一条曲线。但它不推出 Native 功能更好，因为真实内容系数仍未知。

这不是理论最优声明，只是有外部方法依据的一次直接尝试。

## 6.3 首日规模和判定

7B 先在固定64条 compact 单证据/检索任务运行 Native，确认基本内容能力；不为模型反复改 prompt。随后各静态臂在相同4K/8K/16K数据上评估。

首日量级：每类32条检索、64条自然QA；相同实例跨长度，证据位置分层。以生成内容正确性和 Native 配套指标做对比，并保留已有完整精确/EOS结果。

旧 88% Native 指标的失败不被新评分覆盖；只有该指标也通过，才称达成原来严格联合目标。标准 Native 指标通过而旧严格套件未过，只能作范围更窄的 PE 能力结论。

**路线A最多3 GPU小时。**下载、数据准备和公共公式检查在付费运行前完成。不根据结果转入 gain sweep 或 head selection。

---

# 7. 明日路线 B：唯一主训练配方

## 7.1 正面回答“LoRA 到底怎么训练”

训练不是继续当前 ZF，也不是再做首 token 修补。两臂从相同的原始 OLMo1B Instruct 重新开始：

| 臂 | 固定位置系统 | 其它训练条件 |
|---|---|---|
| Y-CPT | YaRN-s4 + 官方冻结幅度 | 完全相同 |
| Z-CPT | 已有 Z-s4 + 当前冻结幅度 | 完全相同 |

没有新 N 训练臂；近期 ON 已提供旧配方下的失败，不值得明日再烧一份。新 Y 是必要的公开位置方法参照。

Y/Z 的幅度若不同，比较是**位置系统整体的适配效果**，不是 pure-z 因果效果。论文已有 pure-z 控制承担后者。

## 7.2 训练参数

**开放：**

- 所有 attention Q/K/V/O 和 MLP linear 的 LoRA，r=16、alpha=16、dropout=0；
- 模型原有全部 normalization affine 参数，包括存在的 Q/K norm、block norm、final norm；
- input token embedding 全矩阵。

**冻结：**原始 attention/MLP 线性权重、untied LM head、RoPE、gain。不新增 temperature、head mask 或 gating。

这是 **LoRA + Norm + Embedding 的部分参数适配**，不是“仅约1%参数”的纯 LoRA。官方 OLMo1B 配置 vocab=100352、hidden=2048，input embedding 有205,520,896参数，约占本体13.84%；加 LoRA/norm 后总开放量约15%，由实际模型枚举确认。[R4]

选择它的理由是优先得到工作点，避免把已经被文献指出的自由度排除掉。并非证明这15%都必要。成功后才考虑减参，不先做六种模块 ablation。

## 7.3 阶段 A：长文本持续语言建模

- 2048 条真实16K序列，来源为两个现有公开训练数据池：PG19 train 与 FineWeb-Edu 的长文档，各1024条。
- 按 source document/book 划分训练与验证。一个长文档可切连续片段，但同源片段不能跨集合。
- 不足16K的文档不能靠重复模板或无意义padding凑成“真实长依赖”。数据资产不足时先完成 CPU 准备，不占着GPU等待。
- 自然文本使用普通 causal LM 输入，预测每一个合法 next token。只屏蔽 padding、确实不可定义的目标；人工截断不能误标成真实结束。
- microbatch1，gradient accumulation4，512 optimizer steps。
- token上限33,554,432；合法预测位置略少，日志分别记录。

$$L_A=L_{\mathrm{dense\ LM}}+D_N.$$

每个update另取1条≤2K Native replay；D_N是该条/该组明确采样位置上的完整词表 teacher→student KL。不是把整条16K隐藏状态送到teacher，不要求存储全长词表分布。

**这一阶段比原配方更接近有效的坐标适配，但不是 YaRN 原预算复现。32M 是否足够是工作假设。**

## 7.4 阶段 B：少量确实依赖上下文的生成适配

共512个视图，64 updates、effective batch8：

- 128个单证据自然QA实例，每个8K与16K两种视图：256条；
- 128个随机键值绑定实例，每个两种合法内容世界：256条。

每步4条自然QA、4条绑定，组内均衡长度与证据位置。两臂使用相同实例、布局、顺序和seed。

自然QA用现成可验证来源和答案，不让模型编造新真值。背景以真实文档填充，过滤直接答案泄漏；query与证据有明确一致性。绑定世界改变value时同步改变gold，不能给错误世界监督旧答案。

训练只要求数据真值正确，不要求底模在两个世界都逐字输出且EOS才准入。**旧严格qualification不是新训练的入场条件。**评估仍并列报告 Native compact-correct 子集，区分已有能力迁移与新任务学习。

仅监督完整正确回答及真实turn-end token：

$$L_B=L_{\mathrm{answer+EOS}}+D_N.$$

每步2条Native replay。两个任务组各自先序列内取均值、再等权；不让回答长度决定任务重要性。不添加首token最大margin、contrast系数或prefix128×0.1附加项。

选择简单单证据任务不是通过改分数救旧结果，而是新一轮将计算复杂度固定，集中训练长距离内容使用。double/复杂格式原结果继续保留，不再阻止本轮。

## 7.5 Native replay 与优化器

Native池512条、四组各128：自然文本、一般指令、短QA/推理、格式/索引。仅用训练来源；此前确认集上的失败样本不回灌。

自然文本固定32个分布于序列的位置；指令样本缓存原模型实际生成的回答前缀，最长128个输出token。正常终止的teacher轨迹包括终止位置；截断轨迹不伪造EOS。这里是训练数据准备，不另开teacher-prefix机制论文。

student在replay上仍使用Z或Y实际部署表。以相同原模型原表作为teacher。

优化器固定AdamW，所有可训练参数学习率2e-5，betas=(0.9,0.95)，weight_decay=0，clip=1；各阶段5%warmup，cosine降到2e-6；阶段B重置optimizer，两个臂一致。

KL权重固定为1，两个均值项分别归一化；不沿用D/0.02再乘初始dual=1的隐含50倍尺度。**这不是声称旧KL过强已被证明，也不保证λ=1最优。**这是新复合配方的一项公开工程选择，无系数搜索。

最终以真实Native保留判定；没有任何固定KL系数能保证88%。若模型超过Native损伤预算，不能放宽评分来接受它。

## 7.6 保存与选择

保存：原始起点、CPT128/256/512、SFT64。前两份CPT用于预算—损失曲线，不作为大型模型选择池。

最终候选只有CPT512与SFT64。每臂按独立≤16K验证：先满足Native标准，再优先生成内容；同分优先Native损伤小者。选择后冻结，再打开32K数据。两份都不可行，就报告该臂不可行。

不因看到第一臂结果而改第二臂数据、rank、步数或loss。两臂完整结果是一个比较单元。

## 7.7 精确成本计数

| 部分 | 每臂输入/序列token上限 |
|---|---:|
| 512 × 4 × 16384 CPT | 33,554,432 |
| 256条8K + 256条16K任务视图 | 6,291,456 |
| CPT Native：512 × 2048 | 1,048,576 |
| SFT Native：64 × 2 × 2048 | 262,144 |
| 合计 | 41,156,608 |

teacher、validation、generation和activation recomputation另计。不是4100万独立样本；token之间有相关性。

**每臂最多6 GPU小时，训练两臂合计最多12小时。**开始前按实际 dense-loss backward 测吞吐；若预计无法完成，记录所需时间，由负责人从总预算调配。不能自动缩短训练数据、删embedding或改成8K后沿用相同实验名称。预算截停的run不是完成的负结果。

---

# 8. 16K→32K：如何像 YaRN 那样推进，而不是提出更苛刻的隐含要求

明日先冻结并评估 s4 产物在32K的表现，作为不改表压力测试。它的失败不等于第二阶段不可能。

接下来的预定义延伸方式是：在训练开始前切到s8的32K目标表，仍只用16K物理数据，做一次有限续训。不是prefill过程中切表，不是看到测试答案后选择scale。

只有第一阶段在16K有实际生成收益且Native可行，才值得花这笔钱；优先仍做Y/Z匹配，不以某一个漂亮点省掉对照。

后续固定预算建议128个dense updates、effective batch4，共8.39M long tokens，再32个任务updates。数据与目标构造仍≤16K；32K/64K结果不可用于选checkpoint。该扩展使用后续预算，**不是明日20小时默认自动运行任务**。

这样两项研究结论分开：

1. 4K→16K：小模型经有限训练实现Native窗口外能力。
2. 16K训练→32K测试：在预先确定的32K位置配置下继续长度泛化。

不宣称“只看16K数据就知道任意64K任务该怎么做”。64K保留为更远验证；不在32K尚不可靠时用它消耗大部分预算。

---

# 9. 明日评价：足以指导方法，不做完整审稿规模

## 9.1 固定三类读数

**检索生成：**RULER标准生成器中单键、multi-key各32实例，4K/8K/16K；官方scorer与完整输出精确、EOS并列。标准scorer不是纯字符串出现诊断，保持其原定义。

**自然生成：**64个独立single-evidence QA，compact/8K/16K，保留足够证据、不改变gold；使用正式定义的答案EM/F1。输出是自由自回归生成，不是候选答案打分。原32组双世界严格集仅作历史比较，不用于选新的频率构造。

**Native：**两个独立文本域各64片段，加PIQA、ARC-Easy、HellaSwag各固定256题的官方评估设置；另复用旧Native严格套件作回归。选择题likelihood准确率只用于Native保持，不改称生成。

对自然QA中合法不同措辞按该任务官方答案规范处理，不用答案字面出现计为正确。实际内容错误、额外冲突答案、未结束输出保存在raw记录并分别统计，不重开数百条盲标项目。

## 9.2 首日通过与正式论文确认分开

首日只判断是否值得继续：

- 16K生成内容得分必须相对本臂未训练底座有实际增加；
- 不能只改善EOS或平均NLL；
- Native总体点值至少88%，分项完整披露；
- 分数的置信区间和小样本不确定性仍报告，不把点值叫最终非劣证明。

不人为要求所有小组CI下界同时超过88%才允许完成另一臂。

正式“可信工作点”需冻结产物后在新的数据、更多独立实例与训练seeds上确认。旧严格Native门槛未通过时，不能用新增较易题集宣称旧目标已实现。标准PE评估与旧压力测试的结论分开。

## 9.3 两种统计单位

同实例不同长度、近远版本、两个世界，不是独立重复。bootstrap按语义实例/源文档分组。训练seed变异与评价样本变异分开。首日一个seed是方法可行性测试，不是稳定性证明。

---

# 10. 如果失败，下一步仍朝目标走，但不能乱加组件

| 结果 | 解释与后续唯一优先动作 |
|---|---|
| Z/Y都能恢复16K生成并过Native | 先做第二seed匹配；再做预定义32K目标适配 |
| Y明显成功、Z失败 | 训练系统已有可行参照；承认当前Z与适配不匹配，不能怪小模型普遍不会。用剩余零训练/理论预算研究Z而不是改动Y评分 |
| Z成功、Y失败 | 保留Z工作点；公开方法在本协议失败不是对原YaRN的反驳。先复制，再隔离gain/shape而非立即SOTA宣传 |
| NLL明显改善，两个生成端点仍不改善 | 分布适配发生但任务转化不足；不算成功。不先上FFN/decoder搜索；先用已计划的CPT-only→SFT差异确认任务训练是否在训练集上学会 |
| 训练集也学不动，Y/Z都失败 | 下一笔受限预算做Y下full FT，固定相同数据前缀和预算，检查PEFT是否限制了该训练；不是rank sweep |
| 训练集好、验证差 | 优先数据多样性/过拟合问题，不能由此推导容量不足；不再重复128个任务到更多epochs |
| Native先坏、内容收益明显 | 仍未联合成功；从已保存点报告trade-off，后续只能做一个固定更强保留方案，不加回已失败样本再用原确认池 |
| 7B零训练明显好于1B | 得到更好的成熟工作点；不能声称仅参数数目是因果原因；训练主线仍先结束1B比较 |
| MrRoPE成功而Z/Y失败 | 固定单表空间仍存在可行候选；这是他人方法，诚实作为强baseline，不能据此认证我们的曲线 |

full FT是后续容量参照，不是从失败推导出的必然救法。即使full FT也失败，仍要区分数据量、数据依赖、优化与位置配置；不能宣布理论不可能。

---

# 11. 预算与明日顺序

## 11.1 明日最多20 GPU小时

| 阶段 | 主要机器/执行方式 | GPU小时上限 |
|---|---|---:|
| 资产完成后的短runtime检查、dense CE测量 | 任一；复用已有可信路径 | 0.5 |
| 零训练：OLMo7B固定矩阵，1B新增M | Pro6000优先 | 3 |
| OLMo1B Y-CPT+SFT | 5090或Pro6000 | 6 |
| OLMo1B Z-CPT+SFT | 同上，尽量相同backend | 6 |
| 固定checkpoint评价与收据 | 两卡按显存需要分工 | 3 |
| 余量 | 故障回收，不做额外研究 | 1.5 |
| **合计** | 双卡同时执行按小时相加 | **20** |

这些是支出上限，不是虚构ETA。完整训练预计超过上限时，在正式启动前报告，不能边跑边将明日预算吞成100小时。

## 11.2 全项目100小时保留分配

| 用途 | 上限 |
|---|---:|
| 明日两条主线 | 20 |
| 固定32K目标的≤16K续训，或必要的Y-full容量参照 | 20 |
| 有效两臂的第二、第三seed复制 | 30 |
| 独立标准任务确认与一次第二checkpoint外部验证 | 20 |
| 故障/数值与论文复现余量 | 10 |
| **总计** | **100** |

不是所有项目都自动授权运行；只有上一阶段结果决定需要哪个已声明的后续比较。不能把每个上限都当作应花完的额度。

## 11.3 工程只保留必要项

- 先用nvidia-smi确认Pro6000具体型号与显存，不默认它就是96GB Blackwell。官方该版本为96GB；其它代际另算。[R7]
- 不重新安装已工作环境，不换推理框架，不进行仓库文件整理。
- 支持HF分片权重加载即可，7B不因旧单文件loader限制而被当成模型问题。
- BF16权重，数值稳定的loss；position×frequency及sin/cos计算保留FP32，再按模型接口转换，不能把长位置相位直接在BF16计算。attention使用精确内存高效kernel，禁止N×N attention缓存。
- dense CE不能存下全长N×V logits及其多份梯度；采用验证过的fused/streaming linear-cross-entropy。词表投影分块但保留所有位置的精确目标和梯度，不用抽128位置冒充dense。
- 若现有分块autograd仍保留全部softmax activation，就没有达到预期内存节省，需修这一点而不是删训练序列。
- 新开放的norm、embedding和LoRA用FP32可训练参数或FP32 master weights累积更新、BF16/AMP前向；不能让2e-5级更新被BF16权重量化直接吞掉。一次真实step检查有效参数差即可，不另开梯度研究。
- 新开放input embedding和norm必须随产物保存；只存LoRA会丢失本轮真正训练的参数。
- 训练关闭KV；推理前清空KV并固定表。沿用已有identity/cache测试，修改相关路径才补测试。
- 原始失败保留，新的训练recipe另命名；不假称“旧引擎逐字不改”还能包含新的参数更新与dense目标。

---

# 12. Codex核心伪代码

```python
# Phase 0: CPU/network preparation; do not hold a paid GPU idle.
resolve_original_checkpoints_and_tokenizers()
freeze_existing_Z_and_official_Y_and_published_M()
prepare_document_disjoint_cpt_train_and_validation()
prepare_small_sft_pool_and_fixed_evaluation_rows()
cache_native_teacher_distributions_on_native_training_sources()

# Track A: no optimization, one static table per method at all request lengths.
for model_id in [OLMO_1B, OLMO_7B]:
    for system in [NATIVE, Z_S4, YARN_S4, MRROPE_PRO_S4]:
        if exact_existing_receipt_matches(model_id, system, rows, scorer):
            reuse_receipt_without_substituting_other_data()
        else:
            model = load_original_weights(model_id)
            install_static_position_system(model, system)
            run_fixed_native_and_4k_8k_16k_generation(model)
            save_raw_outputs_and_all_declared_scores()
# No choose-best-frequency and no next-candidate loop.

# Track B: a paired system comparison, not a search over modules or ranks.
for arm in [YARN_S4, Z_S4]:
    student = load_original_olmo_1b()
    teacher_identity = ORIGINAL_OLMO_1B_NATIVE
    install_static_position_system(student, arm)
    install_all_linear_lora(student, rank=16, alpha=16)
    unfreeze_existing_norm_affine_parameters(student)
    unfreeze_input_embedding(student)
    keep_original_linear_weights_and_output_head_frozen(student)

    optimizer = fixed_adamw(lr=2e-5, betas=(.9, .95), weight_decay=0)
    for step in range(512):
        # Four real 16K documents, full valid next-token loss.
        loss_lm = accumulated_dense_lm_mean(student, cpt_batch(step, size=4))
        # Student uses arm's deployment table even for Native examples.
        loss_native = native_full_vocab_kl(student, native_batch(step, size=1))
        optimizer_step(loss_lm + loss_native)
        save_declared_intermediate_points_only(step)
    save_complete_model_delta("cpt_512")

    optimizer = reset_fixed_sft_optimizer(lr=2e-5)
    for step in range(64):
        task = balanced_sft_batch(step, natural_qa=4, binding=4)
        loss_task = complete_answer_and_turn_end_ce(student, task)
        loss_native = native_full_vocab_kl(student, native_batch(step, size=2))
        optimizer_step(loss_task + loss_native)
    save_complete_model_delta("sft_64")

    candidate = select_between_two_declared_points_using_native_and_16k_dev()
    freeze_candidate_before_opening_32k_outputs(candidate)
    evaluate_complete_generation_and_native(candidate)

report_both_arms_even_if_one_loses()
report_runtime_tokens_trainable_parameters_and_unfinished_work()
```

不写新“理论指标排序器”。不让报告生成器根据long结果发起下一次实验。需要追加工作时按第10节给出一项明确选择，而不是列十个可尝试项。

---

# 13. 论文应该怎样用这些结果

原稿的核心不是“凭一张表解决所有LLM长推理”，而是固定support下的allocation独立性、有限basis几何以及安装阶段的作用。[I3]

这不是放弃两个方法目标，而是避免把投稿成败绑在比YaRN原命题还强的全局无损系统上。

**正文优先保留：**已有三seed因果证据；静态mature表的可复核生成效果与Native代价；一个在固定预算下有效的适配对照。纯检索不替代自然QA，NLL不替代生成。

本轮若Z/Y适配都有效但相当：说明位置扩展在预算内可做，不证明Z独特。论文须主要依靠已有pure-z因果实验与更清楚的scope；不能强行写新的“LoRA只修接口”机制。

若Z在等预算下更好：才将适配效率列为新增贡献，之后做重复与必要的gain-matched比较。新增norm/embed/CPT组件本身来自相关工作，不是论文方法新颖性。

如果仅7B的zero Z有效：这仍是有价值的部署尺度证据，但不能隐藏1B负结果或称跨checkpoint普适。

数学上不要再把spectral-budget恒等式写成必然牺牲Native的“水床定律”；不要把严格全域transplant不可能定理用来否定允许12%行为代价的近似适配；不要把有限候选没有成功写成全静态空间无解。

**这次应该补的是：合理位置配置能否在足够相关的训练下被模型利用，而不是为每次失败再命名一个机制。**

---

# 14. 可移植证据与文献索引

## 内部材料

- **[I1]** `SPECTRAL_BUDGET_LORA_20260905.md`，本轮用户附件，正文标题日期2026-09-06。以包含ON完成结果、已降级强机理叙事的版本为准；不使用同名更早草稿替代。
- **[I2]** `SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904(1).md`，含unit-amplitude完成、N128独立Native确认及训练计数的版本。方法、模型、计数单位不能跨版本拼接。
- **[I3]** `main(20260905-052024).pdf`，31页。p4–5为几何/变分/移植边界；p6–7为成熟冻结与适配；p23为750M续训；p25–26为固定support与weights×table；p29–30说明zero系统与routing。
- **[I4]** `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md`，2026-07-25，§2.3。来自File Library；specific softplus/init记录而非对所有flat runs的归因。
- **[I5]** `Hybrid-RoPE_ZERO_TRAINING_RESEARCH_DOSSIER_20260827.md`，§5–6，包含one-turn、piecewise、direct-z和QK/QKVO失败。
- **[I6]** `mainstory.md`，历史v22，2026-03-16/17版本。仅用于识别Power-Shift的公式和video实验所属；其“唯一可行曲线”“普适规律”等过强叙事不被采纳。

## 本轮核验的公开原始资料

- **[R1] YaRN — Efficient Context Window Extension of Large Language Models.** §4.1、Table1、§4.3–4.4。https://arxiv.org/abs/2309.00071 。本轮查看PDF p7–8图像；主要用来区分训练量、目标配置与物理长度，不用其数字保证本方案效果。
- **[R2] LongLoRA — Efficient Fine-tuning of Long-Context Large Language Models.** §3.3、Table2、§4.1。https://arxiv.org/abs/2309.12307 。本轮查看PDF p6图像；norm/embed结果来自特定Llama2+S²设置。
- **[R3] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data.** https://arxiv.org/abs/2510.25804 。只支持数据长度与可用远程信息并不相同；本轮不采用额外数据排序系统。
- **[R4] OLMo-2-0425-1B-Instruct 官方config.** https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/raw/main/config.json 。本轮读取L0、vocab、hidden、heads、base与untied head；部署须另pin实际revision。
- **[R5] OLMo-2-1124-7B-Instruct 官方模型卡及config.** https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct ；https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct/raw/main/config.json 。核验默认4K与相同head_dim/base；不是证明两代训练完全匹配。
- **[R6] MrRoPE: Mixed-radix Rotary Position Embedding, ICLR2026.** §3.2、Eq14–16及AppendixB。https://proceedings.iclr.cc/paper_files/paper/2026/file/69413f87e5a34897cd010ca698097d0a-Paper-Conference.pdf 。本轮只采用公开固定静态构造作外部对照，不采用其理论编码上界当作OLMo生成保证。
- **[R7] NVIDIA RTX PRO 6000 Blackwell Workstation Edition 官方规格.** https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ 。96GB是该具体版本，不默认等于用户租用的所有“6000”。

## 一句话给执行者

**先把零训练工作点和“固定Z/Y + 长文本CPT + LoRA/norm/embedding + 少量内容任务”两臂做出来；不重跑compact-only、不新增FFN/格式归因主线、不把128步短答案训练当成YaRN已经被复现、不在第一天花掉100小时。**
