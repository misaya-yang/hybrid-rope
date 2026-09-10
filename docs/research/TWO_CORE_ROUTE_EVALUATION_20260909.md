# 位置研究路线评估：12条候选、PC2与PM-Keep

2026-09-09。三份独立评估覆盖12条候选；附第13项PC2的严格理论审查。用户随后加入Pro的PM-Keep，并明确最终只推进PC2和PM-Keep两条主线。执行入口见[TWO_CORE_SOL_HANDOFF_20260909.md](TWO_CORE_SOL_HANDOFF_20260909.md)。下文是方案与反例，不是12组GPU结果。

PM-Keep的完整用户方案：[原始文件](/Users/yang/Downloads/ICLR_Position_Research_Decision_Codex_20260909.md)。其理论和接口已转成experiments/pm_keep中的算子、原模型适配、作者EA基线及完整生成代码。


---

# 一、几何四路线

# 四条几何路线的独立审查：今晚可以做什么、不能凭什么下注

日期：2026-09-09。范围：动态 cache 一致性 RoPE、压缩前 source-phase 绑定、PPE-inspired 多位置表示、稀疏图 edge-relative phase。本文是设计审查，不是 GPU 结果。未修改仓库代码、未操作远程或 GPU。

先给判断：**这四条都不支持把“改动预训练 reader 的位置几何”直接设为今晚 10–12 小时主实验。动态 cache 适合最多 45–75 分钟的有限判别；source-phase 最有价值的转化是诊断原生 selector 的相消与异质性遗漏，再补充 selector 统计，而不是重新旋转 reader。** 后者需要与当前主任务的 moment selector 方案合并，不应当作第五条平行路线启动。

已阅读的证据包括用户调研、`SPARSE_POSITION_EVALUATION_LITERATURE_20260909.md`、`PROJECT_DIRECTION_SYNTHESIS_20260909.md`、`COMPRESSED_POSITION_INDEPENDENT_20260909.md`、`deepseek_mini_position/runtime.py`、`cache_schedule_attention.py`、`RTX5090_BLACKWELL_PROFILE.md`，以及已下载 NOSA 原始实现 `results/nosa_position_20260909/source/modeling_llama_long_infllmv2.py`。动态 cache 另由独立子代理核对已有实验记录和接口。文献判断主要依赖本地已经复查的一手来源记录，没有把用户报告里的不可追溯 turnXX 引用直接当作新颖性证据。

## 1. 动态 cache 一致性 RoPE

### 初始候选与可证明的内容

把第 t 个 query 的长程位置分组设为

\[
G_t=\max\{1,\lceil(t+1)/W\rceil\},\quad
\ell_{tj}=q_t^T R(f_{G_t}(j)-f_{G_t}(t))k_j,
\]

其中当前实现远程使用 \(f_G(p)=\lfloor p/G\rfloor\)，近程保留原始位置。每个 query 的规则只依赖其逻辑位置、已经到达的前缀和固定参数，不依赖本次 API call 的末端长度。

**可证命题。** 固定 token 序列、逻辑位置、模型参数、因果可见集合及确定性 tokenwise 运算，若每一层每一位置采用上述 prefix-consistent 规则，则任意合法分块执行得到相同的精确实数算术输出。证明按层和位置归纳：该位置的前层输入和可见历史已相同，attention 算子也相同，故新状态相同。这个证明不依赖该特定 ceil 公式；任何固定的因果位置规则，包括静态 scaling，都能满足。

**不可跳接。** 分块不变不等于 native 等价、不等于最终 horizon oracle 等价、不等于任务更准确。当前 BM 交叉 cache 结果说明两个静态位置表下，仅在 decode 重旋转 K 不一般恢复不同前缀形成的 K/V；它没有证明真实动态 chunk schedule 已导致任务失败。toy 数值反例也不是模型能力结果。

### 自我否定

最强反例是“稳定执行一个不适合任务的规则”。前 W 个 token 的状态永久按 native 几何形成；到了更长前缀，query 用更强压缩相位读取这些状态。早期层已经没有保留的关系不会因未来分块一致而恢复。固定最终 horizon 从开头形成的状态可能更适合长距离回答，并且更快。

其次，\(f_G\) 合并 remote 邻近位置，即使 raw K 相同也可能丢失重复记录的顺序可分性。近程保护不能修复任意远程 ordinal 任务。全历史重算可能恢复，row-wise 却不恢复：这支持状态形成问题，拒绝当前修复，不能自动允许加另一适配器来延续同一个成功 claim。

### 最近邻与应保留的 claim

动态缩放下 key 相位不一致已有直接先例；Jet-Long/bifocal 框架和 Consistent Dynamic NTK 都必须按实际代码比较。当前 wrapper 的 raw K 重旋转已经排除了最浅一层旧相位错误，但“prefix-consistent 函数定义”仍不是独占理论。用户调研把 RoPE layout bug 解释成两条路径必须共享几何也过强；不同布局可以实现同一内积。

本轮另读本地存档 `/tmp/hybrid-consistent-dntk-20260909/scale_rope/consistent_rope_for_llama_patch.py`：文件头日期为 2023-07-22，代码明确 cache 未旋转 K，读取时按当前 sequence length 重旋转全部 keys。其 [作者说明](https://normxu.github.io/A-Potential-Rotation-Inconsistency-of-Dynamic-Scaled-RoPE/) 已讨论 PPL 固定长度和真实逐 token generation 的差别。它不等价于 row-wise 历史形成修复，但直接覆盖“缓存 raw K 并重旋转”的机制，不能把这一部分作为2026年的新意。

唯一有潜力的实用 claim 是：**在未知最终长度的 append-only 流中，某个明确因果执行规则保持完整任务质量，并相对现实固定策略减少重算，形成更好的质量—成本关系。** 必须同时赢质量或成本；“相同文本分块不同会有不同 logits”只能支持运行语义观察。

### 今晚公平、可判决的实验

优先用已有 Qwen2.5-1.5B/3B，不用尚未接入的 Qwen3.5-2B。当前 install 只支持 qwen2/qwen3、全维 split-half、无 padding/position jumps/sliding/压缩 cache。Qwen3.5-2B 的 262144 原生窗口、partial rotary 和 recurrent state 使 32–64K 测试既不触发真实扩展也不满足 wrapper 条件。

数据需真实超过原生 W：先取 1.25–2W 的有限检索/重复来源 streams，保证有至少一个强对照能完成任务。当前 MRCR 18–25K 在 W=32768 下不触发，W=128 只配做算子诊断。所有臂同 token IDs、同生成预算，流中的中间回答不能遗留 KV 改变后续可见上下文。

首批臂：phase-corrected call-wide、row-wise、独立 dev 上选定的部署静态因子；少量样本加最终 horizon 静态 oracle 与逐点全历史重算。后两者是参考条件，不是准确率上界。完整 prefill、固定 chunk、错开 W 边界 chunk 是同一输入的配对条件；逐 token 只用于极少算子样本。

主指标是 raw 完整答案 exact+EOS，以及源文档/stream 级配对成功和失败；自然 QA 另用其标准指标。记录端到端累计延迟、各臂独立 reset 后的峰值、重算 token 数。模型 logits 差与 K/V 差只是机制诊断。

CPU 先修当前 probe 的 ids/references schema；有外部数据时仍执行 native parity；检查 A+B−C merge 在集中注意力下的消减数值风险；逐臂 reset memory。不能带这些已知缺口开始无人值守长队列。

### 结果分支与预算

- call 分块改变正确答案，row 恢复且优于部署静态：追加独立 streams 和另一已资格骨干，值得成为候选论文问题。
- 只有数值小差、任务相同：停止本路线，得到“此工作负载不需要修复”的有限部署结论。
- 重算恢复、row 不恢复：历史状态形成有作用，当前 schedule 修复失败；停止系数扫描。
- row 与静态同质量但更慢：选静态，方法 claim 不成立。
- 所有臂都失败：数据上没有已确认可恢复 headroom；它是不可判定，不是 row-wise 普遍不可能，更不值得用剩余整夜反复测。

**今晚裁决：最多 45–75 分钟有条件的旁路判别，不能取代稀疏/压缩来源绑定主问题。** 成功也只够决定扩展，不能据此保证 ICLR accept。

## 2. 压缩前 source-phase 绑定

### 初始候选、精确公式与真正命题

在 V4 Mini 风格 shared-KV 压缩器中，令 \(u_j=a_j\odot c_j\)，其中 gate 已包含原生块内位置参数，块锚点为 \(p_b\)。

\[
C_b^{\mathrm{native}}=R(p_b)N(\sum_j u_j),\qquad
C_b^{\mathrm{source}}=R(p_b)N(\sum_j R(p_j-p_b)u_j).
\]

旋转仅作用于原来的 partial-RoPE 维度；非 rotary 分量保留。N 在块局部坐标里计算，保留原生 learned norm，最后才赋绝对锚点。

**可证命题 A：在受限固定特征下对交换敏感。** 忽略 N，两个来源具有相同标量 gate 时，交换 A/B 所致差异为

\[
\Delta T=(R(\delta_1)-R(\delta_2))(A-B).
\]

一个频率平面中，若 \(\omega(\delta_1-\delta_2)\notin2\pi\mathbb Z\) 且 A−B 在该平面非零，这个差异非零。

**可证命题 B：条件性的平移协变。** 若平移不改变 content/gate，所有 p_j 与 p_b 同移 d 后，块内 offset 不变，输出增加 R(d)。这只陈述该算子坐标行为，不包含真实网络 hidden states 在平移后保持不变的假设。

**不可跳接。** source sensitive 不等于 source recoverable；N 或 reader 投影可消掉差异；有限维聚合不是无损。原生 APE/gate 已经随 j 变化，原生交换差可为 \((a_1-a_2)\odot(A-B)\)，所以不能声称原生压缩绝无顺序信息。完整网络的因果 hidden states 更不是置换不变特征。

### 最强反例：新加入的相位能把原本一致的内容清零

等幅同向 c、均匀 gate、B 个连续位置时，一个频率平面的均值幅值比为

\[
\left|\frac{\sin(B\omega/2)}{B\sin(\omega/2)}\right|.
\]

它可为零。RMSNorm 不能恢复已经相消的信息；它还可能把残存噪声放大。对任意 u_j 有

\[
\|T-S\|\le\sum_j\|(R(\delta_j)-I)u_j\|,
\]

但当 \(\|S\|\) 很小时，这个绝对扰动界对归一化后输出没有小相对误差保证。把“群作用正确”写成“所有预训练内容关系更完整”是错误推论。Prism 已分析先旋转再平均的相消；Still 反向解旋后压缩的经验正好构成竞争解释。RoVE/V4 原生 OV 运输也不是这里新发明的部分。

### 与当前 NOSA 的决定性接口核对

**NOSA selector 已经在做本候选的核心 source-phase 聚合。** `LlamaSdpaAttention.forward` 先对 Q/K 调用 `apply_rotary_pos_emb`；随后 `_sparse_attention_forward` 把旋转后的 K 传入 `CompressK`，后者 32-token window、stride16 取 `filtered_k.mean(dim=1)`。所以摘要就是

\[
\mu_b=\frac1B\sum_{j\in b}R(p_j)k_j.
\]

它再用于选 64-token 原始 KV blocks，reader 最终读的是被选中的原始 KV，并保留 learned DMA/cis。它不是 V4 的“压缩 shared-KV 直接承担 reader value”。因此把 source-phase 移植到 NOSA 后宣称新方法，实际上可能是 no-op；把它再旋一次则改错几何。把 Qwen3.5 的 full-attention 强行改为 pooled-memory 也不是已有预训练压缩接口。

### 自我修正与公平实验

若未来真要在已有预训练 V4-like compactor 上研究，可用 native-parity 的残差插值

\[
C_b(\eta)=R(p_b)N(S_b+\eta(T_b-S_b)),\qquad \eta(0)=0,
\]

配同等适配训练的 native compactor、PPE-inspired、同预算多槽强对照。不能把 native 冻结、只给候选训练；也不能只赢删掉原生 APE 的弱控制。\(\eta\) 是受限可学习干预，不意味着训练必能找到收益。冻结主模型时需同时训练所有臂相同范围的 compressor 参数并采用相同答案侧训练目标；有 LM 保持数据以防仅学格式。

主任务：普通内容 KV 检索、重复 key 第 n 次/最近一次、来源交换反事实、两个未训练长度/块边界，以及模型能够完成的公开 RULER 子任务。完整生成 exact+EOS；按反事实 family 配对，内容与顺序能力同表。若只有随机 Mini 任务训练通过，只支持机制可学性，不能称预训练语言模型方案成功。

**今晚不建议启动这种 compactor 训练：现成 Mini 是 random-init，NOSA 接口已含 source-phase，Qwen3.5 不含该压缩 reader。** 为凑训练跑道而临时造 pooled-reader 会让负结果混合任务能力、预训练几何失配、压缩器未训练和方法假设四种原因，10小时没有清晰归因。

### 成败分别留下什么

- matched 已训练压缩接口上源交换与内容任务都改善：支持 joint binding 的具体构造，继续公开任务及第二真实接口。
- 顺序收益伴随内容退化：明确相消—辨序权衡；不能宣布解决压缩位置问题。
- native APE 已很强、候选无收益：该接口不需要这项操作，停止；不否定所有位置信息研究。
- 只在随机 Mini 训练有效：留作机制或可学习性证据，主论文预训练质量 claim 仍未完成。
- NOSA 中证明原生 source-phase 已存在：得到一次有价值的方向排除，避免重复实现已有操作。下一步合理对象是它遗漏的 score 分布，而不是重加相位。

## 3. PPE-inspired 多来源位置表示

### 初始候选与可证明的内容

将一个 compressed content 向量 c 的不同 rotary frequency pair 分配给不同来源位置：

\[
\tilde c_r=R(\omega_r p_{\sigma(r)})c_r,
\qquad
\ell_{ib}=\sum_rq_{ir}^TR(\omega_r(p_{\sigma(r)}-p_i))c_r.
\]

\(\sigma(r)\) 可取块内均匀位置、内容权重分位点，或原 PPE 的明确文本适配。若它只用位置集合的均值、方差或 Fourier moments，也需要明确内容摘要和位置摘要之间是什么联合关系。

可证的是每个 band 使用其分配位置的相对相位，以及在固定 features 下相应的平移规律；**不是“一条 compressed token 能代表所有源内容的精确位置”。** 每个 band 只收到一个坐标，不是让每个语义特征同时携带所有来源。

### 自我否定与最强反例

若压缩 content c 与多位置集合分开保存，那么 A@p1/B@p2 和 B@p1/A@p2 在同一个 c 和位置集合下给出完全相同表示。交换不变的边际位置统计不包含 content-source 关联。该反例必须限制为固定上游 features/聚合器，不能扩展成有因果 features 的整个网络不可能性定理。

把一个已预训练语义通道的坐标从 anchor 改成另一个来源也可能损坏原有匹配：只有其内容与那个来源确实对应时才有意义。一个 band 的 c_r 若同时混有 A/B，把它标成 p_A 并不会把 B 的部分消除。把更多位置装进固定 d 又会减少每来源可用频谱；多 ID 不等于无成本增加信息。

在 NOSA 中，压缩后 c 已是 \(\sum_jR(p_j)k_j\)。再给各 band 赋多个 p 是二次旋转，并不是合理 PPE 移植。要做它只能先改变 NOSA 原生 post-RoPE mean 的统计定义；这改变 selector，而非增加“缺失的位置”。需要与原生已联合编码的摘要比，不与无位置 mean 比。

### 最近邻碰撞与适用 claim

[PPE](https://arxiv.org/html/2510.22936v1) 已把多个源位置 ID 分配到一个视觉压缩 token 的不同频率段。其主实验包含大量 SFT，不能因为“参数少”就视为文本预训练模型的可靠零训练插件。文本适配必须明确标为 PPE-inspired adaptation，首次保留多位置的 claim 不成立。它在今晚最合理的身份是强近邻对照，而非主方法。

若获得正结果，最多起步声称“在特定压缩接口、相同状态预算下，多位置的频率分配改善某些来源任务”；还要排除收益来自额外训练、容量或 slot 分配。不能把多模态 PPE 的训练成果拿来担保文本 reader 的兼容性。

### 公平实验与失败价值

未来真正有预训练 pooled-reader 时：native anchor、PPE-inspired、source joint binding 三臂同参数/训练范围、同原始 compressor。控制原生 APE 不变。内容、重复来源、块内交换、边界平移与公开检索共同评估，至少两个压缩预算。另作同总持久状态字节的多槽/更细分块 frontier；它是实际资源强对照，不冒充只改位置的单变量消融。

不把“两个 d 维槽”与“一个 d 维槽”叫等预算；如果为对齐字节改了全局摘要数或压缩窗口，必须报告这一代价。额外位置 ID、moments、temporary cache 也计入成本。

- PPE-inspired 胜 source，但两者都不胜 native：新增相位没有必要。
- PPE-inspired 与 source 都改善，等字节多槽更好：应采用更直接的多槽压缩，不能把位置包装成唯一机制。
- 只在分布外顺序任务提升但公共检索不变：有限机制结果，可作补充，主质量 claim 不完整。
- 全部新方法破坏内容能力：预训练接口不兼容，不能归结为位置无用。

**今晚裁决：不独立跑。已有压缩 reader 主线若成立，把它放进后续强近邻对照；当前 NOSA 不支持直接套原 PPE。**

## 4. 稀疏图 edge-relative phase

### 初始候选与可证明的边界

报告建议对选中边加入 routing 类型/块尺度/置信度：

\[
\ell_{ij}=q_i^TR(\omega(p_j-p_i)+\beta_{\tau(i,j)})k_j,\quad (i,j)\in E.
\]

\(\beta\) 零初始化可保证在固定计算实现下 native parity。若要求每个 token 只旋转一次、复用普通 K cache，则相位必须存在 node potential \(a_i\)，使 \(\theta_{ij}=a_j-a_i\)。在连通图上，其必要且充分条件是任意环的有向相位和为0（mod 2π）；在 causal DAG 的三点有边子图上同样要求

\[
\theta_{ik}=\theta_{ij}+\theta_{jk}\pmod{2\pi}.
\]

证明：potential 差沿路径 telescoping；反之选根并用路径积分定义 a，零环和保证路径无关。

### 自我否定：图一致性理论可能正好限制掉提案的“新能力”

一般 edge-type 常数不满足上述闭合。三条 global 边都加 β 时，直边加 β、两跳加2β，只有 β=0（mod2π）才能满足。强行说它仍只需每 token 一次标准 RoPE 是错误的。若把它限制成 node potential，就退回重定位/额外 token 坐标，RePo/GRAPE 等近邻更近，“稀疏图新几何”的独特性反而变小。

非可积边相位不是数学非法，但要真实逐边算子，或对有限 edge types 分别执行 attention 并正确合并 softmax normalizer。两个路径独立 softmax 后任意加权不是同一个统一 attention。小参数量不能推出低 latency；原有 kernel 若不支持，就可能把省下的 sparse FLOPs 花在重排与多次 kernel 上。

### 预训练上的最强反例

若原生 selector 已选对证据，reader 原来的 RoPE 就是其预训练匹配函数，额外 β 可直接翻转证据/干扰项排序。若证据根本被删掉，β 不能创造丢失内容。更重要的是 edge confidence/type 常与距离、token age、摘要规模、DMA decay 共变；即使可学习 β 提升，也可能只是额外 branch 容量/温度校准，并不证明新的位置几何。

对 NOSA 的 local/global branch 加常数 query 旋转是容易实现的特例，但这只是 branch-specific Q 变换；如已有不同 branch Q 投影，某些常数旋转可直接被投影吸收。需要同容量 Q 校准和 logit-bias 对照才能声称 PE 作用。

### 最近邻与错误动机

DeepSeek V3.2 的 indexer non-interleaved / MLA interleaved 官方修复，是让各路径匹配自己的权重约定；不同基底实现相同旋转并不必然造成几何冲突。DSA 还已有主 attention 分布监督 indexer 的 KL 训练。不能以布局 bug 或“要 router-main consistency”为新问题已证的证据。CABLE/GAPE、RePo、GRAPE 与 existing DSA supervision 是实际近邻，不应只比独立随机位置的弱基线。

### 如果坚持实验，最小公平版是什么

只在 native 有完整任务能力、且已有明确 reader 相位导致排序错误的诊断后，做两类分开的比较：

1. 总效应：相同 pretrained NOSA、同 DMA、同 TopK 预算，候选与所有控制允许其自身因果 hidden/selection 改变，测完整生成和成本。
2. 机制：固定输入和固定可见结构的 teacher-forced reader 比较，或使用所有臂一致的、与生成内容无关的 deterministic sparse mask。不要把 baseline 在未来真答案上的 masks 泄漏给候选自由生成，再称固定-mask任务比较。

候选 β 仅少量 heads/branches、zero-init，训练目标和 token budget 与同容量无位置 Q adapter、标量 logit bias/temperature、native matched adaptation 一致。随机或置换 edge labels 是必要诊断：若它也同样改善，位置解释失败。固定 read budget 与固定 masks 是两个不同 estimand，不能混用。

需要量 kernel、selector、prefill、decode、训练时间及总状态；任何 Python per-edge reference 只能给机制 evidence，不能当作部署成本。若实现需要新 sparse kernel，今晚没有依据保证10小时内同时拿到可靠训练和公平性能比较。

### 成败分支与今晚裁决

- 超过同容量 content/branch 校准、位置置换破坏增益、公开任务与成本成立：值得继续，可立更窄的 relation-specific reader claim。
- β 与普通 Q adapter 相当：结论是适配容量帮助，几何解释不成立。
- 只有人工 corruption 曲线更平：先看扰动后是否还保留正确内容，不能称修复自然路由错误。
- 任务改善但显著更慢：按真实质量—成本取舍报告，不能因参数少称高效。
- kernel/训练不资格：工程不可判定，不否定函数类，但停止让它消耗今晚主预算。

**今晚裁决：不跑主实验。没有现成自然失败点和合适 kernel 时，这是高概率把一次长夜消耗在预训练兼容性与工程资格上的方向。**

## 四条路线合并后真正值得迁移的机制

NOSA 的原生 mean 保存的是联合 source phase 的第一矩，而非完全没有位置信息。正确问题是：**一个均值摘要能否低估包含稀有高 logit token 的块，造成有限 TopK 下遗漏，而补充少量 score dispersion 能否在 reader 完全原生的条件下恢复完整任务？**

令旋转后的 key 为 \(z_j=R(p_j)k_j\)，块均值 \(\mu_b\)、协方差 \(\Sigma_b\)，固定 query q 的真实 score 方差是

\[
v_b=q^T\Sigma_bq/d.
\]

真实 block logsumexp 满足 cumulant 展开

\[
\log\frac1B\sum_j e^{s_j}=\bar s+\tfrac12 v+\tfrac16\kappa_3+\cdots.
\]

这解释 mean-only 对 heterogeneous score 分布的潜在遗漏，但不是全局性能保证。若只存 diagonal covariance，则丢掉跨维协方差：z 在 ±(1,1) 上、q=(1,−1) 时真实 variance=0，diagonal approximation 却为2；q=(1,1) 时真实4、diagonal2。相同均值与方差也不能确定稀有高分尾部的 logsumexp。故“二阶近似定理准确”绝不能推为“所有块排序更好”。

source-phase 相消与一般内容异质性应区分：相消是后者的一个可能来源，不能把所有 variance 收益归因于位置。可以在固定 pre-rotation content 下受控改变源位置做机制实验；主要质量仍在原生真实数据、完整生成、原生 DMA 和同 TopK 下验证。强对照至少应包含原生 mean、保留更多局部均值的细分块方案、可用的 Quest/envelope 或等开销的峰值候选，不让“同摘要个数”掩盖 moment 更贵。

**相较这四种直接改 reader 的候选，moment selector 的优势是干预可落在 NOSA 已有接口、保留预训练读取几何；它的成败都更容易解释。** 成功说明一个具体 selector 统计修复了 task-visible omission；失败能区分没有可恢复路由 headroom、二阶摘要不够、原生 DMA 抵消、或额外成本不值得。它并不自动获得 PE 新颖性，也不能仅凭 selector recall/attention-KL 就进入论文主结论。

今晚值得托管的是一个主问题的决策树，而非四条各分两小时。先用少量完整任务确认 NOSA 原生和强 selector 对照具有能力且存在可恢复差距，再决定 moment 方案是否扩展；若没有差距，保留预算、输出清楚否定该 workload 上的需求。10–12小时是资源上限与预计容量，不是必须烧满的成功条件。


---

# 二、路由四路线

# 四条 routing 方案的独立审查：2026-09-09

结论先行：四条中，没有一条现在能承诺 ICLR accept。今晚最值得执行的 routing 工作是**在真实 NOSA 上，把 native、等开销细粒度均值/Quest、位置保留二阶摘要、内容二阶摘要同时放进最终生成比较**。它能产出明确的方法质量结果，也能结束“平均摘要误差究竟是不是关键因素”的悬案。但把 `logmass≈mean+variance/2` 当新理论，或把 router KL 对齐当新贡献，均与已存在工作正面重合。这一 routing 对比只能是待竞争的方案，不应由于可写代码就压过别的压缩路线。

本审查读了用户 deep-research-report、SPARSE_POSITION_EVALUATION_LITERATURE、SPARSE_POSITION_CLAIM_DISCUSSION、OVERNIGHT_FAILURE_POSTMORTEM、CORE_DIAGNOSIS、ORACLE_FIRST_SHARED_ROUTING_PLAN；直接核对 NOSA、SparDA、Prism、FASA、COBS 原文及本地 NOSA 官方源码。未修改仓库、未运行 GPU。对当前四条路线的裁决是：R1 不作主线；R2 不重复已有失败构造；R3 可作今晚主执行候选，但 novelty 很薄且必须正面打 COBS；R4 不应在没有必要路径证据前占用整晚。

## 共用的真实接口与质量要求

**NOSA 是选原始 KV 块的模型，不是把整块永久压成一个 value 的模型。** 官方源码 `results/nosa_position_20260909/source/modeling_llama_long_infllmv2.py`：`CompressK` 在 32-token 窗口、stride16 上求 post-RoPE key 均值；selector 经 head 聚合和 64-token block max-pool，先选 query-aware 部分，再由训练得到的 query-agnostic `cis` 分数填满总预算；main attention 读取对应原始 K/V 并包含 `cis` attention bias。保持原生频率、Q/K、V、trained `cis`、sink/local 和总块数。改变 query-aware 选择后，按照原规则重新填满 query-agnostic 部分，不能误把旧填充集合强行固定而改变预算或不允许重叠。

NOSA 常用总预算是 4096 tokens，其中 query-aware 1024；64-token blocks，固定 sink/window 占 17 块，其余是动态块。该事实来自 [NOSA v2 §3、Appendix B](https://arxiv.org/html/2510.13602v2)，运行仍需读取实际 checkpoint/config，不能硬编码论文配置。以两个总块预算 64 和 48 作公平比较时，所有方法采用相同 sink/window 与相同 query-aware 配额；配额比例的改变本身不是方法收益。全 cache 仍存在于 GPU/CPU，因此只能声称 sparse access，不得声称压缩了总 KV 容量。

Qwen3.5-0.8B 有 GDN recurrent 和少量 full-attention 层。在 full-attention 上后加 mask 的实验可研究另一个读取接口，但**不是**原生 NOSA selector，也不证明重训练过的 sparse 模型有同一瓶颈。已归档 Qwen 结果显示 PSR 与同数量 contiguous 无分离、Pre/PostMetric4 被同预算 Quest32 超过；不能用“换成新模型”重置这项负证据，只能说明 NOSA 的训练补偿和 shared-GQA/`cis` 形成了新的可识别条件。

每个方案的一晚预算均以**已下载、可运行的 NOSA-1B 或 3B**为前提，不以当晚重新训练 1B backbone 为前提。32GB 可容纳 1B frozen model 加合理长度 cache，但 kernel、32K 峰值和秒/样本尚未实测；不虚构吞吐保证。先以同一 8/16K 输入运行完整 native 与全支持 gather parity；CPU 无卡阶段准备模型配置、参数名和数据，不得把 CPU 数学测试写成 READY 的 GPU 结果。

每晚的共同执行配额（四个方案是互斥选项，绝非四个夜间队列叠加）：

1. 0–0.5h：确认 native 正常生成、数据长度/支持覆盖、修改关闭时输出一致；用 8 个不进入测试集的实例量 16K/32K 端到端 p95 时延。它只决定作业规模，不是科学主实验。
2. 0.5–2h：固定 64 个开发提示的 native/必要对照；若需要校准，再用独立 train prompts 采集真实 Q/K/`cis`，仅抽取预声明 query positions，避免保存 O(T²) 矩阵。
3. 2–6h：一个预注册候选加最强相关对照的自由生成。先完整完成 128 个配对样本，然后向 256–384 扩展；任务至少含官方 RULER 的内容检索与多值/追踪类、重复来源的完整字符串读取、一个真实长文 QA 子集。不能把 8 个 smoke 结果升级成主结果。自然 QA 用官方整体答案指标；机械唯一答案另外保留 literal raw exact+EOS、规范化完整答案、格式成功率，不能抽取 substring/首数字。
4. 6–9h：固定方法后第二长度或第二总预算的确认集，至少 128 个配对样本。应优先取得完整质量–预算比较，再铺更多机制图。
5. 9–10h：对最终两个方法做端到端时间、所有 descriptor/cache/CPU transfer 字节、显存峰值测量和输出复算。10–12h 只扩充已开始的确认样本或配对校准种子，不因首轮失败自动再开一个不同方法。

实际样本上限必须用 `n_max=floor((剩余秒数-收尾保留秒数)/(arms*p95_seconds_per_sample))` 计算。若 n_max 连预定最小确认集都不够，缩短到已有能力且有稀疏区分度的长度，不能无声缩成 16 条然后宣称稳定。样本不确定性与训练种子不确定性分别报告。不要用基线错误样本筛选取代全分布成绩；另列 native 错→候选对和 native 对→候选错配对计数。

**前缀干预边界必须写清楚。** 先 native 编码无问题文档，再从问题首 token 开始修改 sparse reader，能够回答“共享文档 cache 的 query-time 检索”问题，并自然复用前缀；它不证明 sparse prefill 加速或整个输入全程使用新方法的结果。若工具仅支持这个入口，主 claim 就限于 cached-prefix serving，并在独立 holdout 保留完整生成。要声称端到端全 prefill 方法，必须让原生 sparse prefill 的 selector 同样使用候选，单独记录其成本。不能对全 prompt native prefill 后只修改第二个输出 token 的实验声称检索机制已完整干预。

## R1：在原生 reader 几何下做 router KL 校准

### 具体干预

令原生 post-RoPE reader logits（包含 learned bias）为

`z_hj = q_h^T k_j / sqrt(d) + cis_j`。

按真实 GQA group 和合法 token 计算块质量 `p_hb = sum_{j in b} softmax(z_h·)_j`，teacher `P_b=(1/H) sum_h p_hb`。这是 full-key diagnostic/training teacher，不能在部署阶段全量扫描。

保留 native selector `a_hc=q_h^T mu_c/sqrt(d)`，构造少量位置相关特征，例如 `phi_hc = a_hc - qpre_h^T mean(kpre)_c/sqrt(d)` 和该差分的平方；原生均值 query-aware score 加 `u_h phi_hc+v_h phi_hc²`，后续 head aggregate/max-pool/`cis` fill 完全原生。`u=v=0` 精确回到 native。训练只拟合这些标量或极小 gate，主模型冻结；目标为 teacher-to-router KL，包含 top-target 集合之外的 rest mass，不能仅在候选自己已选中的集合监督而永远看不见漏选。

比较等参数 content 特征（native 分数及平方、key norm、native `cis`）校准；若加训练，则这组对照用相同 teacher、train prompts、步数和 seeds。只比较训练后的 phase scorer 与未训练 native，会混淆校准和位置设计。

### 严格理论、预测、反例

若 `KL(P||Pθ)≤ε`，Pinsker 给 `TV(P,Pθ)≤sqrt(ε/2)`。令 `S*` 为 P 下的 top-K，`Sθ` 为 Pθ 下的 top-K，则 `P(S*)−P(Sθ)≤2 TV≤sqrt(2ε)`。它只保证**选中 attention mass 的后悔值**，并不保证完整答案、更好的 evidence coverage 或 head-specific 必要路径。

反例：某必要来源在 teacher 的总 mass 很小，但经 value/output/residual 后影响唯一答案；teacher mass 最优可能仍漏掉它。另一反例：在训练点几何差分与内容相关，而测试反事实打断该相关性，phase gate 学到 shortcut。均值后的 KL 再低也无法恢复摘要所缺的高阶信息。理论不能推出修复 RoPE 本身。

可证伪预测：phase 特征应当在正确官方几何下、同样训练的 content scorer 之外，改善**与位置因素相关的真实漏选以及完整答案**；如果只是 KL 降低或总体 mass 更好而生成不变，位置主张失败。

### 最近邻与创新冲突

[DeepSeek-V3.2 §2](https://arxiv.org/html/2512.02556v1) 已用主注意力聚合分布蒸馏 indexer，正确 RoPE layout bug 修复不是几何一致性新定理。[SparDA §4.2](https://arxiv.org/html/2606.04511v1) 已在 NOSA/InfLLMv2 sparse-pretrained 模型冻结 backbone 训练 Forecast，使用 top-target+rest KL 和更细 teacher summaries。它还有跨层预取系统收益。仅把目标改成包含 phase 的 KL，贡献很难超越它；反向 KL 也不是充分创新。

### 一晚执行与失败价值

四个最终臂：native；content-only matched calibration；phase calibration；预算内等开销细窗 native scorer。2–3h 采集/拟合，6h 留给真生成和成本。可用不同初始化的两个小校准重复，不能让低维模型“3 个种子”冒充完整训练稳定性。只要 teacher 不需全模型梯度，一晚可跑完，但不等于复现 SparDA 的完整训练规模。

- phase 赢 content 且赢细窗、代价可接受：得到可继续的新经验事实，claim 为“原生 sparse reader 的位置相关校准改善了指定共享 cache 检索”，仍需更多架构/任务建立论文贡献。
- content 与 phase 一样好：得到训练校准效应；撤销位置解释，方法作为已知 distillation adaptation 归档。
- KL 降但答案不变：直接否定该分布代理在当前任务的瓶颈性，避免再扩参数。
- exact teacher 选块都不能改善：只能否定这一 mass teacher，不可宣称所有路由都不重要；不用再训练拟合它。

**今晚裁决：不推荐为主线。** 计算可行，新颖性和目标特异性都太弱。只有现成数据已显示 phase-conditioned 系统误选，才值得短支线，而不是从“理论漂亮”推断它会成功。

## R2：查询条件的内容–phase 多代表检索

### 具体干预

每个物理块先独立于未来 query 构造 R=2 或4 个代表：对 native post-RoPE keys 的分组 `G_br` 缓存 `(n_br, mu_br)`；query-time 分数为

`ell_hb = log sum_r n_br exp(q_h^T mu_br/sqrt(d) + mean(cis)_br)`。

它是 query-conditioned **scoring**，不是 query-conditioned 缓存分组。若每个 query 临时对全块原始 keys 按 `q^T k` 重新聚类，就已经全扫描 raw K；必须按真实额外访问计费，不能称 cacheable selector。

位置版用 `q` 的独立校准二阶矩 M_Q 作为度量，联合原始内容与 phase 选择 `G_br`；公平对照是同 R、同 FP precision、同字节的 contiguous、random、post-key content clustering、pre-key content clustering、Quest 更细物理页。正文不能把联合分组仅改名为新 PE。

### 理论、预测、反例

令所有 `j∈G_br` 的投影残差 `e_j=q^T(k_j−mu_br)/sqrt(d)+(cis_j−mean cis_br)` 绝对值≤δ，则每组真 logmass 与代表 logmass 相差至多δ，合成 LSE 仍相差至多δ。实际可用平均误差界只约束平均 query，不能推成每个 query 排序稳定。用 block 分数间隔>2δ 才能保住候选对排名。

这解释“多代表为什么可能比一个均值好”，并不证明 phase 选择优于 content。反例包括少数 answer-key 被多数内容吞没、代表恰落在相消方向、query 的所需内容在校准 M_Q 的零空间内，以及四个 contiguous 子块已经满足同样的误差界。不同 key/value 语义也可能具有同样旋转相位，phase 多样性不等于 evidence 多样性。

预测必须是：在同 R/字节和同查询分布下，phase-aware 代表比 content/contiguous 降低真实错误候选对误差，最终生成领先；若收益只来自 R 倍描述符，则不成立。

### 最近邻与创新冲突

本项目此前 Pre/PostMetric4 在 32 个冻结生成样本得到 11/10 个 exact+EOS，而更细页 Quest32 是13；PSR 与 contiguous 平均保留率也未分离。这不是全称不可能性，却直接反对原样重跑。

[Prism](https://arxiv.org/html/2602.08426v2) 已解释 RoPE 与 pooling 的相消并分频带选择；[FASA](https://arxiv.org/html/2602.03152v1) 已做 frequency-aware selection；[COBS](https://arxiv.org/html/2607.09052v1) 明确讨论 first-order summaries 的缺陷及 compressed covariance。把 counts+LSE、多 centroid、位置聚类拼在一起仍处于已知 clustering/mixture 家族。查询相关分数也不是新特征。

### 一晚执行与失败价值

若执意选这条，只做 native、R=4 phase-aware、R=4 content-only、同字节 finer-Quest 四臂；R 不扫描，使用已有最经济的 R。先保留真实原生 NOSA trained sampler；所有候选从相同压缩窗口出发并明报原始 KV 总读取。4–7h 做最终生成，2–3h 第二预算和时延，余下分析 conditional flips。它能够产生方法答案结果，但构建代表的预填充成本可能很高，必须记在总时延内。

- phase 大幅超过 content/Quest：新的训练后架构条件值得验证，不能抹掉旧失败，也不能直接 claim universal position preservation。
- 多代表都比 native 好、相互接近：确认摘要容量主导，位置方法主张失败。
- frozen 分数好而部署答案不变：说明 trajectory/reader 缓冲了选择误差，不能继续只优化代理。
- 代表构建/字节开销让质量–成本被细页支配：部署方案失败，保留为 oracle 近似而停止。

**今晚裁决：不推荐。** 该构造最容易复现“数学正确、弱均值基线可赢、强基线又输”的上轮循环。NOSA 的训练区别本身不足以赋予创新性。

## R3：保持原生 RoPE 的二阶分散度校正，并只在候选池内重排

### 具体干预

这是四条中最具体的可执行方法比较，但应称 **position-preserving COBS adaptation**，直到找到并证明额外贡献。

对每个原生 32-token subwindow，用 reader 的 augmented key `x_j=[k_j/sqrt(d), cis_j]`，augmented query `a_h=[q_h,1]`。缓存均值 μ 和结构化协方差 Σhat。这是完整处理 `cis` 的一种表示：NOSA reader 的 `cis` 在真 logits 里，不能忽略；第13候选最终采用附录的指数倾斜表示，在概率权重中精确处理它，无需这套 augmented key–cis covariance。

`ell_hc = log|c| + a_h^T μ_c + (β_l/2) a_h^T Σhat_c a_h`。

`β_l=1` 是二阶截断；若校准，仅在训练集拟合每层标量 β，允许收缩到0，不能 test 后反复调。`Σhat` 有三种预声明且不可混称的实现：

1. **RoPE pair 2×2 blockdiag**：每个真实旋转对保存3个数，另保留 cis variance 与 key–cis 相关项（后者增加 d 个数）。RoPE 的配对必须遵从实际 split/interleaved 布局，不能按数组相邻下标猜。
2. **等字节低秩 post-RoPE covariance**：rank由完整 descriptor 字节确定；这是最直接的 COBS-style 强基线，而不是位置方法的自己人。
3. **content/NoPE covariance**：仅 selector 去旋转，原生 reader 保留原样；须公平匹配主模型/预算，不把它当成 NoPE 模型。

主部署形态：native 分数先产生 `C=2Kq` query-aware 候选池，对其计算上述纠正，选回相同 Kq，随后 native `cis` fill。这个过程**不访问候选块 raw K/V**，只访问缓存 descriptor，才有机会维持 offload 收益。如果 exact rerank 要读取 C 个 raw key blocks，即使最终只读 K 个 value blocks，也已增加大量通信，必须单列 expensive oracle。

Shortlist 外漏掉的块无法修复，因此同时记录原始候选覆盖率；不能把全局 exact oracle 的 ceiling 偷换成 shortlist rerank 的 ceiling。另一个细节是 native selector 对窗口概率进行 GQA 聚合与 max-pool，`ell` 应在相同 compressed grid 上替换原分数；直接把整块64的LSE塞进去会同时更改 pooling policy，须作为分开的消融。

### 严格理论和最危险的反例

对某窗口的真 token score z，`logsumexp(z)=logB+E z+Var(z)/2+R3`。令 z 的范围为 Δ，则 log-MGF 的三阶导数是 tilted distribution 的三阶中心矩；保守有限范围界给 `|R3|≤Δ³/6`。所以当 Δ 很小，二阶有可控误差；真实长上下文检索常有稀有高分 token，Δ 很大，此界会完全无用。它绝不是“二阶必定比均值准确”的定理。

pair-blockdiag 的额外误差是 `1/2 a^T(Σ−Σhat)a`，可正可负，且并不因 Σhat 半正定就保证低估。例子：跨两个频率成分 `Σ=[[v,-v],[-v,v]]`，`a=(1,1)`，真方差0，而去掉跨频率协方差得到2v，给纯虚假的大幅加分。这个反例恰好打破“按旋转对保存协方差就自然匹配真实几何”的直觉。相反 `a=(1,-1)` 会漏掉真方差。

还存在稀有尖峰反例：一个窗口绝大多数低分、仅一个相关高分；校正可能严重低估LSE；另一个双峰但无关窗口被高估。过大的 query norm 会放大二阶，造成 spurious top-k。β收缩能缓和数值但不补回缺失高阶信息。

新颖性冲突非常直接：[COBS §4/5](https://arxiv.org/html/2607.09052v1) 已有完全同一 cumulant expansion、low-rank covariance、query-subspace projection、quantization 和 KV traffic 分析。其选择路径使用 NoPE；“保留 RoPE并用2×2协方差”可以是有意义的实证反驳或工程结构化近似，不能再声称发现二阶重要性。Prism 的原版双频带加温度选择也是必须比较的近邻；固定 K 版本要标为 adaptation，因为它原本按 top-P 取 union。

### 一晚主实验（可行但不能偷工）

训练不是必要项：先 β=1 和 native=0，避免又开调参。四个候选中只推一个pair模型进入确认，另外两个协方差与细均值进入开发/最终强对照，不能把已失败对照删除。建议最终四臂：native；同 descriptor 字节的细均值/Quest较强者；post-RoPE pair covariance；同字节 content/low-rank COBS较强者。开发选 strongest 的规则事先写定，确认集冻结。若算力容许保留第五个 Prism-style 固定K对照；不够时它属于投稿必须补齐的 baseline，报告明确不作SOTA。

2h 内至少完成开发集的真实答案；不用先写一整套分析再碰生成。用 native 前缀共享的 late-query retrieval 实验可降低成本，但 claim 必须限于这个入口。成功后当晚后半段必须补整条 trajectory 上的一个完整子集，前提是候选 prefill implementation 真已接好；没接好就如实保留局限，不能自动承诺加速。

最终主指标是 paired raw exact+EOS、成对 occurrence/order 成功、自然 QA task score；机制附表只报它們对应的 true mass error、rank flips、phase/content/cross-term contribution。对于 scalar β calibration，使用相同数据训练 content/lowrank β，不给位置臂更多训练。

### 结果如何改变判断

- post-RoPE pair > 同字节低秩/NoPE/细均值，跨预算和任务成立：可立候选 claim“保留 reader 原生位置几何的结构化 summary 改善已训练 sparse model 的质量–访问成本”；创新需要证明 pair结构为何在部署/转移上有额外优势，且最终必须面对 COBS/Prism。
- covariance 都优于均值，但 pair无优势：确认二阶信息在NOSA上有价值，得到 COBS迁移结果，不包装成新 PE。
- exact shortlist oracle恢复，但所有 cached cov不恢复：否定二阶结构在当前误差分布下足够，保留高阶误差/候选必要性结论；不自动增加rank/三阶直到赢。
- pooled oracle也不恢复答案：停止优化这个proxy；路线没有证成任务瓶颈。
- 方法质量略好却总descriptor/cache访问或端到端时延被细页支配：不能claim实用提升；记录 tradeoff 限制。

**今晚裁决：四条中优先级最高，但属于有风险的方法比较，不是已确定论文主线。** 比 pure KL 和重复 PSR 更有信息量，因为它直接与近期强方法对撞；其最终生成价值可在一晚决定。若整体12方案中有更清晰的不可替代贡献，R3适合作强对照而不是独占主线。

## R4：同读取预算的位置多样性/必要路径覆盖

### 具体干预

保留原生每-head归一化窗口/块相关性估计 `p_hat_hb` 与固定集合 L（sink/window）。对 query-aware 预算 Kq，不再按head总质量取topk，而最大化

`F(S)=sum_h w_h log(ε+sum_{b in L∪S} p_hat_hb) + λ logdet(I+sum_{b in S} u_b u_b^T)`。

`u_b = sqrt(r_b) ψ((p_b−p_q)/T)` 是位置特征，`r_b` 必须由query相关性产生，防止对无关远块盲目奖励；ψ可使用固定少量distance buckets或低维Fourier phase，不训练backbone。严格同 Kq，`cis` fill 原生。常用 log coverage/PSD logdet 都是已有次模形式，不是新数学。λ仅在训练/dev上选一次，λ=0和w均匀是强非位置对照。

更简单且更经济的臂是 min head retained-mass 的贪心选择，无logdet；必须比较它，以免“位置diversity”只是另一种head coverage。

### 理论、预测和反例

固定非负 p_hat 和固定PSD特征时，上述 F 是单调次模；cardinality约束下 greedy有1−1/e近似保证。该保证只关于**人为写出的F**，完全不保证答案质量。

平均head保留率无法强控制最差head：准确下界为 `min r_h≥max(0,H*mean(r)−H+1)`，不能说绝无下界。高平均值也可漏掉单个必要head，所以coverage有动机；但“少数head”是不是任务因果必要路径必须通过可控支持集替换和完整答案确认。

反例：正确答案需要连续两个相邻块、多样性规则将它们判为冗余，选择远处的无关块；有些head只承担停词/sink或冗余功能，均匀log覆盖会浪费预算；频率相位周期性复现，同phase可能语义完全不同，different-phase也可能是同一事件。因此 positional diversity不等于source diversity。

### 最近邻与创新冲突

项目 ORACLE_FIRST_SHARED_ROUTING_PLAN 已审查 [K-VEC](https://arxiv.org/html/2606.29563v1)、[BumbleBee](https://openreview.net/pdf?id=8w0RApM5yG) 的重要性/覆盖/多样性选择；COBS 已界定GQA additive mass只是输出误差的松弛。log覆盖不能重命名成新位置理论。真正潜在贡献只能是：证明某类重复来源/顺序任务确实需要同时保留的阅读路径，提出更便宜又有效的选择，并在真实任务质量–成本上胜出。

### 一晚执行与失败价值

先用同一个 dev batch 比 native exact additive mass、head log coverage、position logdet cover和强 cached selector，各自保留相同K。需要 expensive exact mass只是早期区分“估计误差”和“目标误差”，绝非最终方法。最终生成须将 cached p_hat用于部署臂并记录 greedy开销，不能只报精确矩阵下的选择胜利。

完成原生 128例与固定16个真实失败/必要证据干预后，若 additive exact 已经修复绝大部分，就把剩余预算给可部署估计而不训练coverage；这并不否定coverage在别的条件有用，却说明当前没有针对它的缺口。只有 additive exact仍缺少多个必要路径、而相同K合法替换能恢复完整答案，才让coverage进入其余确认集。

- 非位置head coverage有效，phase/logdet无增益：贡献属于共享选择目标，位置主张撤销。
- phase coverage优于head coverage且强selector，严格相同cost有质量收益：才获得位置作用的候选证据，仍需排除data distance shortcut。
- diversity有更高覆盖但答案下降：明确否定该覆盖指标作为目标，避免继续做美观覆盖图。
- oracle合法替换不能恢复：当前任务没有被证实的这个瓶颈，不再盲扫λ。

**今晚裁决：不推荐直接作为主线。** 上一轮并没有发现“exact加性目标本身导致任务失败”的证据。先验增加coverage比较可能值得，但在该证据之前分配整夜会重新变成漂亮目标函数驱动实验。

## 能支撑论文的 claim 与今晚实际产物应分开

不应写：“sparsity破坏RoPE相对性”、“router/main必须共享PE”、“二阶统计无损保留来源”、“attention mass提高保证答案更好”、“完成一晚实验就stable accept”。

可作为待验证的routing主张是：**在模型已支持的上下文内，正确实现且已训练过的稀疏读取仍存在可定位的summary误选；在相同总读取和descriptor预算下，保持原生位置几何的低成本摘要在完整检索和来源关系答案上优于native及最近强对照，并有真实时延收益或明确的质量–成本前沿。**

这句话的每一个部分都要有证据。若R3只得到二阶摘要通用收益，真正claim应降为COBS在原生NOSA上的适配结果，不能强保位置创新。若只能跑late-query共享前缀，则scope写这个实际部署入口。今晚最有价值的deliverable是一个可复算的paired生成表、两档预算、现成强baseline和足以关闭错误解释的附表，不是第四份新的研究路线名。

## 补充：第13候选 PC2 的严格评价与实际优先判断（指数倾斜版本）

主任务将 R3 收紧为 PC2：保持 native RoPE，只存每个 RoPE pair 的2×2 covariance，并在真实 trained NOSA 上比较质量–成本。以下以**对原生 `cis` 作指数倾斜**的最终版本为准；上文 R3 的 augmented `(k,cis)` 是另一种合法参考表示，其存储数字不用于这里的PC2，也不能据它批评本版本必须存key–cis cross covariance。

### 指数倾斜恒等式与成本纠正

对某个合法窗口 b 中的 tokens j，令 `c_j=cis_j`、`t_j=q^T k_j/sqrt(d)`。在每个实际KV head分别计算：

`Z_cis,b = sum_{j∈b} exp(c_j)`；

`w_j = exp(c_j)/Z_cis,b`，因此 `w_j≥0` 且 `sum_{j∈b} w_j=1`；

`logmass_b(q)=log sum_{j∈b} exp(c_j+t_j)=log Z_cis,b + log E_{j∼w}[exp(t_j)]`。

最后一个等式是**精确恒等式**，不是bias近似。倾斜权重逐窗口归一化，不能用跨窗口权重直接套此式；窗口常数 `log Z_cis,b` 也不能省略，否则会抹掉窗口间的真实bias质量差别。`cis` 是此处token对应的原生reader bias，不是再次对 block max-pool 结果作softmax。

缓存 `μ_w=Σ_j w_j k_j`、`Σ_w=Σ_j w_j(k_j−μ_w)(k_j−μ_w)^T`，PC2 分数为

`ell_b(q)=log Z_cis,b + q^T μ_w/sqrt(d) + β_l/(2d) q^T Π_pair(Σ_w) q`。

β=1为二阶截断，β=0为同倾斜测度下的weighted-mean强对照。这里**不需要**显式保存key–cis covariance；cis通过w进入所有key矩，而其总尺度由logZ保存。若 `c_j=c` 为常数，则w=1/B且logZ=c+logB，回到普通cumulant公式加精确常bias。

最终PC2存储为 weighted mean d、pair covariance 3d/2、logZ 1，即 **2.5d+1个数/窗口**。在同一tilted measure上，rank2 COBS-style baseline包含mean d、两个吸收特征值的factor共2d、logZ 1，即 **3d+1个数/窗口**；PC2在相同dtype下比rank2少约1/6 descriptor。4个disjoint子窗口weighted means若各保留logZ，则约4d+4个数/原窗口；实际还要匹配overlap stride、token coverage、索引、dtype、更新buffer，不能仅靠这些代数数量宣称总KV或时延降低。COBS原式未包含NOSA这个bias倾斜，故两个baseline都应标为NOSA adaptation，并用**相同w与同logZ**比较。

PC2每窗口构建O(Bd)，可用稳定流式一/二阶加权统计；rank2 PCA构建通常更贵。但必须量prefill/update/selector/main/decode各部分和端到端代价，不能从免eig直接推出更快。权重/logZ与二阶累加至少采用FP32稳定计算，保存时量化另列；`E[kk^T]−μμ^T`的抵消误差不能当真实负方差。新token进入窗口时还需正确更新normalizer，不能沿用旧w只加一个样本。

### 等变命题与它不能保证的东西

设完整旋转为 `R=diag(R_1,...,R_m)`，`Π_pair` 保留协方差每个2×2对角块、删除跨pair项。则严格有

`Π_pair(R Σ_w R^T)=R Π_pair(Σ_w) R^T`。

在同步变换query/key的全局原点、保持原生cis值不变时，w和logZ不变，`q^T Π_pair(Σ_w)q`及PC2完整分数不变。这只说明**摘要近似不引入额外的全局坐标原点依赖**，不是“不丢顺序”、“比PCA更准确”或“保证生成更好”。full-space per-block weighted PCA covariance也随正交共轭等变；若截断处有eigenvalue简并，必须讨论重构协方差及完整子空间，任意切开简并子空间不能自动宣称唯一等变。固定全局投影一般不等变，除非它与旋转兼容或随同变换。

主要风险依旧存在：旋转pair之间并不因为使用RoPE或指数倾斜而统计独立。不同频带由共同隐藏状态产生，cross-frequency covariance可能正是稀有相关token的信号。m个pair投影完全正相关时，PC2估出的方差可比真实值小m倍；完全抵消时，真方差接近0而PC2仍很大。原点等变不约束这种query-dependent误差。

指数倾斜还带来一个必须面对的**低bias稀有重要token反例**。两token窗口令 `(c_1,t_1)=(-A,2A)`、`(c_2,t_2)=(0,0)`。真logmass=`log(exp(A)+1)≈A`；但 `w_1≈exp(-A)`，一阶项约 `2A exp(-A)`，二阶项约 `2A² exp(-A)`，两者都趋0，而logZ也趋0。即使保留完整covariance，二阶也会严重低估真正被query激活的低cis token。bias在恒等式中精确处理，**不代表二阶近似对所有query精确**。这应在真实失败对上检查，不能仅测试q=0或constant-key例子；低有效样本数 `1/sum(w²)` 只能作为这种风险的诊断，不是自动失败判据。

### NOSA配额与“重复计权”检查

**无代数上的双算条件**：selector用上述ell一次，后续只是native quota/union和原生reader的 `softmax(qk+cis)`。选择分数并不是再次乘进reader概率，因此主attention保留cis不是重复计权。也不要在ell之后再加mean(cis)、乘exp(cis)或另加logZ；那才会明确把bias算两次。

**有真实的配额目标冲突风险**：NOSA原生query-aware分支按qk召回，query-agnostic分支按cis驻留；它有意保留“低cis但当前query重要”的独立入口。改成tilted reader logmass后，query-aware也可能偏好原本会由cis填充的块。去重并填满K能保持真实预算，也不会使主概率乘两次；但可能减少query-aware名额带来的**新增召回**，让完整答案下降。原生kq/ke分配下的训练补偿可能加剧或缓解它，不能从真logmass更准确直接推出更好的NOSA系统。

为区分两种效应，最低必要对照是：native；tilted weighted mean；tilted PC2；同tilted measure的rank1/2 PCA或等字节细均值。若开销允许，再加只按原生qk构建的unweighted PC2，用来区分“二阶校正”与“把cis带入query-aware目标”。每个方法按同一原生规则先产生query-aware集合，再用**原生未修改cis**填满总预算；不固定旧e集合、不减小最终独特token数。

保存实际q集合、最终union、仅靠cis的对照集合及其交并，用以统计q名额新增了多少块、其中低cis但高query-score块的召回/丢失，以及相应完整生成结果。不要只看q集合重合高就判方法错；真正判据仍是确认集质量–成本。如果strictly保留原生query-agnostic规则，任意query-aware scoring一般不破坏其配额结构的locality结论，但本版本需按实际边界/填充实现验证，不能把额外bias偏好误称已经破坏数学保证。

### 今夜优先判断

- 在这四类route方案里，tilted PC2比单纯KL或重跑PSR更值得一晚，因为它存在清晰、可实现、可输可赢的系统差别，并能立刻与native、同tilted rank1/2 COBS-postRoPE、细均值/Quest、Prism adaptation比较完整答案。
- descriptor修正提高了它的工程可行性，**没有自动提高任务成功概率**；主要未知仍是跨pair/高阶误差与NOSA配额目标。不能用省去cross-cis统计暗示这些风险已消失。
- 若PC2只胜native而不胜weighted mean或rank2/finer means，不因等变命题漂亮就留作论文主线。若同质量下总成本更低，仍可能有部署价值，需真实速度与跨长度/模型证据。
- 若tilted mean与PC2都受损、unweighted变体正常，结果支持“query-aware配额目标改变”的解释；不要把它误归为二阶方法普遍失败，也不要未经确认集冻结就随意去掉cis再报胜利。
- 若PC2跨预算胜出强对照且成本好，继续扩大同一方法的确认。需要另一个真实sparse模型/接口验证；Qwen3.5后加mask只能是外部迁移补充。
- 邻近工作多不意味着无条件退回旧EVQ/频率外推论文。这里仍有可比较的工程差别，也符合当前研究目标；允许一场强对照裁决，失败就关闭这个构造，不通过不断变名、加rank或泛化等变性质维持期待。

因此tilted PC2可以排为**今晚方法质量对比的一号执行项**，但计划应如实写明：这是检验“结构化位置兼容统计是否具有质量–成本价值”，论文新颖性和任务收益尚未建立。10–12h最合理的可保证交付是完整、强对照、能够决定继续与否的生成结果；不能保证positive或accept。


---

# 三、其它四路线

# 四条替代路线的独立审查：不能用机制恒等式承诺整网任务收益

日期：2026-09-09。仅本地读资料与数学审查；没有连GPU、启动训练或修改repo。依据用户提供的`deep-research-report.md`，以及本repo `SPARSE_POSITION_EVALUATION_LITERATURE_20260909.md`、`SPARSE_POSITION_CLAIM_DISCUSSION_20260909.md`、`COMPRESSED_POSITION_INDEPENDENT_20260909.md`、`experiments/native_sparse_position/DESIGN_SOURCE.md`。使用`research-experiment-judgment`技能的判断原则。文中一手来源链接来自上述已核查材料；本次没有重新联网认证全部论文。

## 共同判断与资源边界

这四条里，今晚最应该真正加入有效实验的是**相同cache字节下的连续多槽/首尾保留强对照**。它新颖性最低，却最可能决定“复杂source-position算法是否值得”；不要把该基线改名后独立宣称新PE。四条里没有一条现在可以独立承诺ICLR accept。若要求从四条里选一项独立新方法，则我不会押注训练后phase removal、直接RoVE改装或RefCarry新理论：强近邻重合和预训练失配同时存在。

本地Qwen3.5-2B配置证据：24层、6个full-attention层与18个linear层，full head256维，partial rotary64维；在这些full层加入压缩是**sparse conversion**，不是其原生压缩训练。主任务随后给出NOSA实际源码`results/nosa_position_20260909/source/modeling_llama_long_infllmv2.py`，已重新核对：它是原生训练的稀疏模型，CompressK以32 tokens、stride16对post-RoPE K求均值，**这只是selector摘要；主reader读取raw KV并有DMA bias**。因此它不是压缩后直接读pooled KV的模型，不能用它证明payload压缩来源恢复。NOSA的吞吐与训练就绪仍未由本次只读审查确认。

32GB对2B BF16冻结主干+少量adapter的2K/4K训练从容量上可行，仍取决于实现；不能从权重约4GB直接承诺长训练可跑。仓库5090的12.5K–24K tokens/s证据属于OLMo2 1.5B的特定16K LoRA shape，不能拿来给Qwen GDN、自定义压缩或NOSA填预算。10–12h应由一次真实前后向/生成吞吐回执划分有限工作，不能虚构每臂可完成的token数。小模型100–150M从零训练可研究专门检索任务，但一晚上不能把它当已有自然语言MRCR能力。

共享有效实验合同：共同初始化、相同数据顺序与有效训练tokens，原生压缩gate/APE不删；baseline获得同样适配；精确任务保存raw token完整答案与终止EOS；holdout的payload、模板、长度、压缩block offset分离；prefix压缩时看不到问题/答案；问题和生成都走被测读取路径。普通内容检索与重复来源/顺序任务同表。相同选中条目数不等于相同常驻cache字节，侧状态、metadata、临时完整KV、训练与生成成本分别计。

## 路线A：learned phase removal / DroPE-style gates

### 机制、公式与能成立的理论

最清晰的静态head/frequency gate是

\[
s_{ij}^{(g)}=q_i^T R(g_h\omega(p_j-p_i))k_j,\quad g_h\in[0,1].
\]

这里Q与K必须使用相同门控频率。报告只写了`q'=R(g(n)θn)q`：如果K仍按原生频率旋转，g=0并不等于NoPE；这是需要补齐的实现约定。静态g本质是频率重标定，二值g才是指定头/频带RoPE删除。若g_i按token内容变化，实际变为`R(ω(g_j p_j-g_i p_i))`；一般共同平移p→p+c会多出`ωc(g_j-g_i)`，不能再无条件主张原生平移相对性。若g依赖当下总长度，还另有旧缓存一致性问题。

一个真实但有限的扰动关系是

\[
\|R(g\omega\Delta)-R(\omega\Delta)\|_2
=2|\sin((1-g)\omega\Delta/2)|.
\]

它说明移除会改变已经训练的logits，不说明改变方向有益。不能把网络每层新投影、残差、softmax的作用粗略等同于同一旋转角的L倍累加，再声称少层RoPE一定有更好的整网稳定区间。报告援引depth phase理论时必须满足原文具体假设。

### 严谨反例与训练失配

固定的两个source表示拥有相同k但不同v，任务要求“第一次/第二次出现”的值。g=0后两source获得相同content score；若下游可见状态没有其他顺序字段，读出无法选择所需occurrence。反过来，强位置phase恰可帮模型区分它们。该反例只针对固定表示/read接口，不宣称整个因果NoPE网络无法学会顺序。

还存在原生模型专门在某rotary pair上训练了`q^TR(ωΔ*)k`识别固定关系的情况；g变小会毁掉已学信号。远程纯内容检索可能改善而最新版本/终止行为恶化，不能用PPL或单needle替代两类能力。

LoRA+gate只在短context自然文本上训练，测试长重复对话时，gate可能只是学会减少当下语料的局部位置负担；未训练的order query和EOS无保证。若只训练gate且loss平坦，也可能是gate不能补偿重排后的表示，而非方法本体不行。

### 最强近邻与可守claim

DroPE已经覆盖训练时有PE、校准后移除；R2N覆盖分层RoPE/NoPE；frequency scaling已有巨大文献。sigmoid gate、layer schedule、compile为静态本身不足新颖性。能争取的窄claim必须是“在已训练的某异构稀疏接口中，任务条件决定哪些位置通路应保留，并以同成本schedule得到稳定质量–成本改善”，且需优于原生、公平DroPE/R2N、固定频率缩放及参数匹配适配。

### 10–12h有效实验与失败价值

不建议今晚主跑。若必须验证，使用已有Qwen3.5-2B，只改6个full层的partial rotary；做native+matched LoRA、静态g+同LoRA、二值layer schedule三臂，shared prefix与相同训练tokens。g从native初始化，不能给候选额外任务适配却让原生零训练。训练包含短LM replay和内容/版本/次序任务；测试2K/4K训练范围与8K holdout，并报告终止和一般短能力。第一配对种子足够裁决是否值得后续，不把三个候选层数网格铺满。

若纯内容涨而顺序跌：得到可复用的task-dependent位置需求证据，但削弱“全局去PE”主张。若学习到g≈1且可靠梯度、任务已经学会：支持当前接口不需删除，不是无效训练。若相同LoRA基线也获得全部增益：归因于适配而非gate。若各臂都不会基准内容任务：未判定，应修训练/任务，不能宣称DroPE失败。只有这些明确判别才值得有限GPU时间。

**今晚评级：不作为新主线；最多保留一个低成本静态NoPE/gate对照。**

## 路线B：RoVE / value-path relative phase

### 机制与公式

RoVE典型操作是

\[
\alpha_{ij}=\mathrm{softmax}_j(q_i^TR(p_j-p_i)k_j),\quad
y_i=\sum_j\alpha_{ij}R(p_j-p_i)v_j.
\]

全局坐标先旋V再加权，输出逆旋query位置，确实使V传输带相对位置。这是算子性质。对于压缩条目可写为`v_b^phase=Σ_j a_bj R(p_j-p_b)v_j`，读出再乘`R(p_b-p_i)`；线性读出阶段有`R(p_b-p_i)v_b^phase=Σ_j a_bjR(p_j-p_i)v_j`。该恒等式不覆盖重新用压缩K计算的softmax权重，不能叫完整dense attention无损恢复。

DeepSeek-V4已有共享K=V与输出inverse RoPE，给它再加RoVE不是新的独立操作；只改Q参考还会造成score frame与V frame不同，必须明确控制两条通路。

### 反例与真正困难

取同一payload v，相隔一个频率半周期的两个source，a=1/2，则`(R(0)v+R(π)v)/2=0`。原生不旋V的平均恰保留v。RMSNorm不能恢复零向量。来源绑定可能改善，但纯内容路径发生可预见损伤。

在冻结预训练Qwen中，V各维的语义是按原生非旋转输出投影学出的。即使attention只指向一个正确source，`R(Δ)v`也可让正确语义在输出中变号；不是“正确选中所以必然正确输出”。需要适配V/output的坐标约定。给新路径一个γ=0原生旁路`y=(1−γ)y_native+γy_RoVE`能避免初始化损坏，不能保证γ可学、也不保证学出的好处超出额外adapter容量。

所谓rotate-before-pool精确保留联合相位，只是在其保留的有限线性特征上精确；相消、信息维数和query依赖softmax都仍在。report中“position-blind V”应理解为算子未显式携带位置，不等于上下文化v没有任何顺序信息。

### 近邻和claim

最强近邻正是RoVE；本repo已核对其124M/354M、FineWebEdu-10B训练、4×H100，RULER使用answer NLL。今晚不能把其10B-token full training收益视为Qwen短LoRA移植的实证保证。压缩前旋转V还与Prism的相消分析、V4原生KV/output运输直接邻近。

可守claim仅是某个具体compression接口的“内容与位置通路分离保存/相对读取，在同字节数下比已实现的RoVE及原生更保留source binding”，这需要新的构造，不能直接把RoVE改名。若QK同时改变，则需V-only/QK-only/combined消融才归因。

### 真实实验与失败价值

今晚若主候选确为source-binding，此路线适合作为**一个V-only机制对照**：same backbone/common train tokens，原生、QK-only source phase、V-only source phase、QK+V。为控制额外容量，γ旁路与必要output adapter在全部条件都匹配。主任务必须包含content retrieval与交换来源后的成对完整生成，另测dense-trained→compressed adaptation对照。

不要在Qwen全模型突然旋转V然后用16个低分样本杀死路线；这只证明预训练不兼容。最好在已有训练过compressed reader的模型里共同微调，或在小模型共同初始化从头训练专门生成任务。NOSA不满足这里的compressed-reader条件；若用它做V-only改造，研究对象是raw-KV sparse reader，不能再用payload压缩丢失作为动机。后者只能给机制证据。

若V-only在换序任务胜出，QK-only无效：定位贡献来自value运输而非router/score；这对论文归因非常有用。若纯内容按phase跨度下降：证实相消代价、给构造的频宽/多槽需求。若LoRA原生也恢复同样能力：主要是适配。若γ始终0且训练充分：当前预训练reader拒绝此归纳偏置，缩小适用范围。所有路径都保存原始生成、loss曲线与byte账本。

**今晚评级：可作因果控制，不能独立担任新颖主claim。**

## 路线C：boundary-aware multi-slot / first-last compressed memory

### 最简单的机制与可证明性质

把block划为M个有序子区间S_bm，分别保存

\[
c_{bm}=\mathrm{Norm}(\sum_{j\in S_{bm}}a_{bmj}\odot h_j),\quad K_{bm}=R(a_{bm})c_{bm}.
\]

或在预算内保留首/尾两条原始KV，其余内容压入低成本summary。它保留更细的来源分辨率，减少跨很大距离的混合。对于固定标量权重、无Norm的key线性摘要，锚点近似误差有

\[
|q^T\sum_j a_j[R(p_j-p_i)-R(a_{bm}-p_i)]k_j|
\le \|q\|\sum_j|a_j|\|k_j\|\min(2,\omega_{max}|p_j-a_{bm}|).
\]

连续均匀子区间将位置半径从约B/2变为B/(2M)。这是对固定summary的误差上界改良，不保证最终softmax、任务或平均误差单调改善。它比“保存均值和位置方差就能找回来源”更直接：表示实际保留了不同位置部分的不同内容。

### 自我攻击：最强反例也是最重要的公平性问题

1. 连续M槽/B tokens，在相同总条目数和same gates时，通常等价于单槽/B/M tokens的更细分块。若不优于这一对照，**不存在新位置方法**；只是改变compression粒度。
2. 首尾保留不等于按query找第一/最后出现的同名实体。block首尾可能都是irrelevant tokens，目标第2/第3次发生在中部。任务只考first/last会奖励人为先验。
3. 同cache字节增加槽数通常必须降低每槽维数、提高其他位置的压缩率或少保留block；不能让M槽用M倍内存再声称更好的PE。
4. 对长程平均/全体证据整合任务，多槽拆分可能被softmax只选一槽而丢失其余质量；正质量block-mass/multiplicity处理要一致，否则收益/损失可能来自softmax权重计数。
5. learned native gate/APE可能已经表示块内顺序。与去掉APE的弱基线比无效。
6. 最恶意的heldout是同content、不同block offset；如果收益只发生在训练对齐边界，学到的是边界捷径。

### 近邻与claim

最强近邻是普通更细连续分块、PPE多位置ID、NSA原生块内position/gate、各种多summary/保留最近tail的缓存方案。首尾/分层保留不应宣称“首次恢复压缩来源”。该臂的价值是建立实际可打败的强基线；若一个新联合phase方法只打赢单槽弱baseline却输给同bytes多槽，论文应转向资源分配结论。

可守的结果claim是“在固定KV字节下，source-resolution allocation比单点位置重标记更决定重复出现读取能力”；要把这升级为研究贡献，需跨接口、多任务、真实质量–成本曲线与可预测的分配原则，不能止于first/last heuristic。

### 今晚推荐的真实实验

把它纳入今晚主实验，**不另外启动独立模型项目**。如果主任务用已训练compressed模型，主表最少native、candidate、same-byte finer-block/multi-slot三臂；若candidate占额外side state，则baseline用这些字节加raw/multi slots，不把额外预算闲置。

共同训练数据含：unique KV；重复key的first/last；内部第n次；`event A之后第一次B`；query不含答案位置。最后两类非常关键，否则first/last的成功毫无泛化意义。内容和模板holdout，block offset均匀扰动；测试两个真实compression预算与2倍未见长度。生成完整2–4 token payload+EOS，保存全部样本和成对反事实成功。

不需要为此新训三份1B full model：优先已有预训练compressed接口的matched adapter；若该资产不存在，small shared-init model可跑机制对照，Qwen本身可为独立语言baseline而非强行临时改成原生compressed模型。

失败仍有明确价值：candidate不如同bytes多槽→停止“phase优先”构造，保留资源分配/来源分辨率证据；multi-slot只改善first/last不改善第n次→证明boundary prior范围；两者都没涨而内容检索可靠→当前读出瓶颈不由这一分辨率控制，需要看合法source access而非继续频率扫参；整体都没学会则未判定。若所有候选都有同等收益，归因为训练/容量，不能制造位置独特性。

**今晚评级：四条中最高，作为主实验不可缺的强基线；独立新意最低。**

## 路线D：occurrence/order-conditioned readout / query参考系

### 机制与成立范围

一种实现读取writer注意力地址测度μ_i(a)，保存Fourier矩`M_μ=Σ_a μ_i(a)R(p_a)`，之后用

\[
s_{ij}=q_i^TM_\mu^TR(p_j)k_j=\mathbb E_{a\sim\mu}q_i^TR(p_j-p_a)k_j.
\]

单点μ=δ_a时确实相对该anchor重基准。概率混合后是平均bilinear logit，通常不是纯旋转，也不是多anchor readout的正确边缘化。理论要写在这个层次。

一个更直接的occurrence-aware reader是按query关系r预测某种relative bias `b_r(p_j−a)`，或用分立anchor slots对每个anchor做各自softmax，再混合输出。前者需要学query→relation映射；后者增加memory/head/read计算；两者都要计入成本与合法anchor获取。

### 数学反例与泄漏

两个anchor的logits分别为[10,0]、[0,1]，各概率1/2。这在二维旋转配合恰当q/k时可实现。正确混合softmax的第一个key概率是

`0.5*(sigmoid(10)+sigmoid(-1))≈0.63445`，

对平均logit[5,0.5]做softmax则是`sigmoid(4.5)≈0.98901`。

所以`E softmax(s) != softmax(E s)`，正确地址的不确定性可能变成错误的过度自信。矩模长小还会改temperature；单看gate收益可能是在调logit幅度而非改善参考系。

更基本地，压缩已经把`A@p1/B@p2`与`B@p1/A@p2`映成相同所有下游状态时，query参考系不能找回丢失的payload–source关联。给oracle正确anchor或第n次occurrence标签等于额外提供部分答案，不能当部署成绩。writer若用gold answer token、final answer位置或事后dense正确路径产生μ，同样泄漏。

如果writer本来就能正确识别所问occurrence，那任务可能已被其解决；必须构造“内容定位anchor→读取邻接/后续不同payload”的真正组合关系，并确认第一跳不是隐含完整答案。同时避免只以lookup table数据训练再声称自然对话泛化。

### 最强近邻与claim边界

本地`COMPRESSED_POSITION_INDEPENDENT_20260909.md`已公式核对：TAPE的attention更新位置字段与双线性读取覆盖核心矩形式；RoVE已有地址Fourier矩生成和frame变换；RePo已有动态点reference；NTM已有内容寻址后相对位置移动。RefCarry剩余差别是one-sided query更新、key原坐标冻结、跨GDN bridge的更新日程和概率约束，不是全新寻址理论。

可以争取某hybrid桥的低成本adapter价值，但要与相同功能约束的TAPE变体比，不能拿完整TAPE每层更新的高成本去对照本方法跳层旁路。V4/shared KV还须同步理解output frame，Qwen的Q-only公式不能无条件迁移。

### 今晚实验与失败价值

不建议以当前RefCarry作为主线再开完整夜训。先复用既有oracle/负例，只有新证据指明某桥的address-state确实丢失且合法writer可获得所需anchor，才做有限matched适配。第一组实验证据应是native+等容量LoRA、合法单点reference、合法moment-reference、功能匹配TAPE、amplitude-only控制；oracle只是诊断旁列，不能混入主表。

数据应同时覆盖单点anchor和有两个合理候选anchor的歧义例，分别测单跳、两跳、反向关系、未见spacing/模板。绝不能只测one-hot oracle后宣布soft reference理论成立。若资源仅够两臂，应优先native vs合法reference，并用CPU确定moment实现及反例，无需满矩阵。

失败价值：oracle好但合法writer差→定位anchor获取，不继续重排read公式；单点好、混合坏→地址不确定性与softmax边缘化不匹配；amplitude-only同样好→温度效应而非新reference；功能匹配TAPE相同→确认实现精简贡献而非新算子；全部合法版本无效且训练充分→结束当前桥的下注。若baseline不会组合任务，仍是未判定。

**今晚评级：当前不押；不要因为公式最漂亮或已有代码最多而选它。**

## 从四条中选最强，然后攻击自己的选择

我选择C进入今晚队列，但明确它是**强对照和决策工具**。理由不是它能独立带来新paper，而是它用最少新工程回答新方法是否真有固定资源的优势。理论支持“减少来源混合尺度”这一局部机制，反例也可直接由任务测出；比在预训练Qwen上大改phase/V路径更少受表示失配牵制。

对这个选择的最强攻击：它仍可能只得到普通compression granularity的常识、无新意，甚至因为原生模型有learned block APE而无效。因此今晚不能只跑C然后写论文；应让它与主任务最终选择的一个有差异构造**共同**训练/评估。只要它胜过复杂候选，就省下后续大规模浪费；但不能把这种负向价值冒充已经有accept论文。若一个新构造能在同bytes下同时优于native与C，提升内部occurrence/顺序反事实、保持内容能力，再跨第二接口重复，才有比“更好的RoPE公式”更可信的主要claim基础。

## 一个可接入主任务的10–12h分配建议（不是四路线都启动）

1. 0–0.5h：真正作业入口的最短finite forward/backward+decode、已有checkpoint原生任务样本，记录新shape吞吐。CPU同时产生固定holdout、byte账本和上述反例；不反复验证无关hash。
2. 0.5–5.5h：一个主骨干、一个主候选、native和同bytes multi-slot的共同配方first seed；按首次回执选所有臂能在这一窗口完成的**相同有效token预算**，而非每臂各跑随意小时。若资产允许，先从有能力的预训练compressed模型适配。
3. 5.5–7.5h：同一冻结holdout的自由生成，优先内容、内部第n次、顺序swap、未见block offset与长度；至少两种预算。留够生成时间，不能训练吞满一夜只剩NLL。
4. 7.5–10.5h：若first seed体现可靠可解释增益，做第二paired seed或已有第二接口的有限迁移，选择能区分当前最大不确定性的一项；若无增益但实验有效，完成预先定义的V-only/边界/幅度中**一个**因果消融，而非四新方法轮流碰运气；若工程/学习不足，修已定位问题并续跑同一比较。
5. 最后0.5–1h：完成未结束的生成、逐样本得失和paired CI、cache/latency报告，保存checkpoint与原始输出。10–12h只能承诺有限队列和充分可读的结果包，不能预先承诺正效应、所有模型或多个训练种子都完成。

每项GPU工作需要回答一个具体分叉；“GPU持续忙”和“失败也产生日志”不构成科研价值。若今晚数据证明只对边界或温度有益，应改claim。若native与强对照在目标任务都胜过候选，应停止这个具体构造，不能用新的名字维持同一论证。

## NOSA实际接口补充：把C改成selector强对照，避免研究对象偷换

以上四路线的compressed-reader讨论不应因NOSA资产可用就自动套用。NOSA源码表明：先对真实Q/K做RoPE，CompressK求32-token/stride16均值；QK摘要得分与CIS分支共同选64-token raw blocks；main以raw K/V读取，并通过两次attention归一化实现DMA权重。配置中的`max_position_embeddings=4096`与`original_max_position_embeddings=32768`及已有LongRoPE factors需要按实际运行实现解读，不能只看其中一个字段推导有效长窗。

若主任务最终选择“保留post-RoPE logit分散度的moment selector”，最适合从本报告借走的是C的**连续subblock均值对照**：每32-token摘要维护两个16-token均值，作为与`mean+diagonal variance`约2d标量状态匹配的基线，reader、原生RoPE、DMA、top-k预算及raw KV都保持相同。该路线是routing summary改良，不是KV永久压缩，也不是修复已丢payload。

设一个目标block内真正logits为`x_j=q^T k_j`，划分成非空子块S_m。用

`F_split(q)=log Σ_m |S_m| exp(q^T mean_{j∈S_m} k_j)`。

由Jensen有`log|B|+q^T mean_B k ≤ F_split(q) ≤ log Σ_{j∈B} exp(q^T k_j)`，逐级细分使这一下界不减。该性质对包含RoPE的实际k成立，不需要假定相位是坏的；最终top-k ranking仍可能变化而不一定更准确。它需要跟原生overlap/max-pooling的block映射统一，不能重计重叠token后宣称精确质量。应清楚区分直接替换32-summary评分与新的64-block logmass估计。

对于moment候选`mean(x)+Var(x)/2`，一个竞争攻击是稀有尖峰：31个0、1个A。`Var/2`为O(A²)，真实logmeanexp为O(A)，大A下候选严重过估；相反较小A时可更接近。两子均值可能漏掉尖峰但不会在固定分组的质量估计上超过精确logmass。两者的得失应按自然logit偏度/峰值与真实漏选后果分层，不能只证明方差非负就预测routing必涨。

NOSA的预算需分三项：原始raw KV常驻/CPU offload状态；GPU selector额外summary字节；每query实际访问的raw blocks。若只是固定raw访问量而增加selector状态，应称固定read预算、报告完整metadata成本；不能称严格等总cache。C在这里的最强价值是检验moment额外状态是否胜过最简单的空间分辨率增加。QK只负责原生选择的一部分，CIS补足选择；要分别记录QK proposal changes、最终selected set以及source coverage，不将全部top64错误归因于QK均值。

今晚该接口可以优先零训练有限完整生成，继而视实际错配需要安排matched轻量适配；不能因为此前计划写了训练就耗时重训所有路线。若NOSA基础生成及修正summary路径已有可靠回执，Qwen可作为后续迁移，省下的时间用于更多真实paired generation而非新增候选。


---

# 四、PC2独立理论审查

# 第13方案 PC2：独立理论审查与可失败边界

日期：2026-09-09。对象：冻结预训练 reader、原生 post-RoPE key 与 NOSA cis，只替换可缓存的 block selector 统计。本文没有 GPU 结果，没有声称准确率改善；三个数值反例用 CPU 双精度直接核算。

**判断：PC2 是一个明确、可实现、O(d) 状态与评分复杂度的 structured second-cumulant selector；它值得进入与强基线同批的有限实验，但绝不是被理论担保的论文方案。** 基本 cumulant 原理由 COBS 等已有工作覆盖；RoPE-pair block covariance 的价值只能在强对照下表现为实际质量—成本优势。它不比 full-space per-block PCA 更“合法”或更“位置一致”。

## 1. 精确定义：先确定在估计哪个 reader

固定 query head h、其映射 KV head g、一个因果可见的物理块 B，以及实际 attention scale \(1/\sqrt d\)。省略 h/g 下标。令

\[
x_j=R(p_j)k_j,\quad q=R(p_i)q_i^{\rm raw},\quad
s_j=q^Tx_j/\sqrt d+b_j,\qquad j\in B.
\]

这里 q 和 x 均为 reader 实际使用的已旋转向量。若实际 reader scale 为 \(\tau\ne1/\sqrt d\)，下文一次项、二次项分别替换为 \(\tau\)、\(\tau^2\)，不能只换其中一个。b_j 是 reader 真实加性 cis；不是 selector 的 mean-cis、也不是重新训练的外部重要性分数。

定义

\[
Z_B=\sum_{j\in B}e^{b_j},\quad
w_j=e^{b_j}/Z_B,\quad
\mu_B=\sum_jw_jx_j,\quad
\Sigma_B=\sum_jw_j(x_j-\mu_B)(x_j-\mu_B)^T.
\]

真实 block logmass 为

\[
L_B(q)=\log\sum_{j\in B}e^{s_j}
=\log Z_B+q^T\mu_B/\sqrt d+\log\mathbb E_w e^Y,
\quad Y=q^T(x-\mu_B)/\sqrt d.
\]

把空间分成真实 RoPE 的二维 invariant planes \(V_f\)，正交投影记为 \(P_f\)，令

\[
\mathcal D(\Sigma)=\sum_fP_f\Sigma P_f,
\qquad C_f=P_f\Sigma P_f|_{V_f}.
\]

PC2 定义为

\[
\widehat L_B^{\rm PC2}(q)=\log Z_B+
\frac{q^T\mu_B}{\sqrt d}+
\frac1{2d}\sum_fq_f^TC_fq_f.
\]

同 cis weighted-mean 控制是前两项，不能省掉 logZ；否则块长度、cis 总质量或不完整块导致的误差会被错归因于 covariance。

### NOSA cis 是否被完整正确计入

本地原始源码 `modeling_llama_long_infllmv2.py:1123–1164` 先令 \(u_j=e^{\mathrm{cis}_j}\)，一次 sparse attention 的 value 为 \(u_jv_j\)，另一次 value 为 \(u_j\)，最后相除。在两个调用有相同 Q/K、mask、scale、TopK、dropout=0 的精确算术下，结果恰为

\[
\frac{\sum_{j\in S}e^{q^Tx_j/\sqrt d}e^{b_j}v_j}
{\sum_{j\in S}e^{q^Tx_j/\sqrt d}e^{b_j}},
\]

所以把 cis 精确吸收到 w 和 logZ 是正确的恒等变形。**它正确处理的是原生 reader 的 cis，不是复现原生 selector 的启发式配额。** 后者另做 mean-cis 排序，并强制保留 qk shortlist；两者是不同计算。

成立条件必须完整保留：

- 使用同 KV head 的 b_j；GQA query heads 可以共享该 KV head 的 summary，但用各自 q 评分。
- block 与 w 只含真实可见 token；不能把 causal 当前块的未来 token 放进 summary。完整历史块可复用，当前块用 prefix summary 或保留 native local 分支。
- 原生 cis 在当前源码是 \(A_g\,\mathrm{softplus}(\delta(V_j))_g\)，没有从这段源码推出 query-specific 累积衰减。若另一后端的 bias 是 b_ij，必须检查是否可分离成 token 部分与对所有 j 相同的 query 部分；否则一个 query-independent summary 不够。
- 不再额外加 mean(cis) 或把 e^cis 再乘一次。对每块计算 w 时减 max(b) 做稳定 softmax，logZ 保留被减去的常数；不能只存归一化 w 而忘掉 logZ。
- 源码使用两个 attention 调用、BF16 exp 与相除；数值下溢、量化或两个执行路径误差可能使其偏离精确 s+b。用 FP32 stable-logsumexp 的参考 reader 要先验证 native parity，不能把这种数值差记作 PC2 收益。
- block logmass 使用哪个合法 token 集合必须一致。NOSA 原生是32-token/stride16均值摘要再映射为64-token块 max；这不等于64-token块 logmass。如果 PC2 直接对64-token disjoint blocks构建summary，应保留原生路径为独立baseline，并让 weighted-mean/PCA/split controls使用相同64-token集合。若继续对overlapping32窗取max，理论还需加 window aggregation error，不能只报告下文两项误差。

## 2. 可证命题一：block-diagonal projection 与 RoPE 旋转共轭可交换

固定频率表和 rotary layout。共同改变坐标原点 a 时，令 \(U=R(a)=\bigoplus_fU_f\)。每个 \(P_f\) 都与 U 交换，因此

\[
\begin{aligned}
\mathcal D(U\Sigma U^T)
&=\sum_fP_fU\Sigma U^TP_f\\
&=U\left(\sum_fP_f\Sigma P_f\right)U^T
=U\mathcal D(\Sigma)U^T.
\end{aligned}
\]

若固定 pre-RoPE content、block 成员及 b_j，同时平移 query 与 key 的位置，那么 \(x'_j=Ux_j,q'=Uq\)，于是

\[
\mu'=U\mu,\quad\Sigma'=U\Sigma U^T,\quad
Z'=Z,\quad\widehat L'_B(q')=\widehat L_B(q).
\]

这是一项**固定内容、固定 cis、固定频率表的坐标原点不变性**。它不等于在实际文本前面插入 filler 后模型输出不变；真实 hidden states、V、cis、窗口边界或动态 scaling 都可能改变。也不能只转 keys 不转 query 然后把 score 变化称作违反定理。

实现必须按实际 pair 配对。NOSA/HF split-half 的二维对通常是 (r,r+d_rot/2)，不是相邻元素 (2r,2r+1)；pair 配错会使定理对该实现失效。缓存和代码中应把 layout 作为明确参数。

### 这个命题没有给 PC2 哪些独占优势

**Full-space per-block PCA 也正交等变。** 若 \(\Sigma=V\Lambda V^T\)，rank-r spectral truncation为\(\Sigma_r=V_r\Lambda_rV_r^T\)，且 cutoff有eigengap \(\lambda_r>\lambda_{r+1}\)，则

\[
(U\Sigma U^T)_r=U\Sigma_rU^T.
\]

所以它的二阶 score 在同一原点变换下也不变。eigenvector 的独立正负号变化不影响 covariance 或平方投影。若 cutoff 切开重复特征值，rank-r 子空间不唯一，任意坐标 tie-break 可能破坏数值等变；可通过保留完整退化 eigenspace/在 covariance 层比较处理。这不是 PCA 本身不支持 RoPE 的证明。

更固定的 global query-subspace、固定投影或量化 PCA 变体是否等变，要看它们的具体投影与量化方式；不能拿这类变体代替 parent 指定的 full-space per-block PCA 强对照。

PC2 的真实结构优势候选是：无需每块 eigen decomposition，用 O(d) 构建/更新和存储保留**每个原生二维 plane 的完整协方差**。它可能优于少数全空间主成分的资源取舍；这需要测量，不是旋转合法性优势。

### 非 rotary 维度

在 unrotated 子空间上 U=I；保留完整协方差、对角协方差或其低秩近似都不破坏上述原点不变性。选择必须写进方法与成本。

- NOSA D=128、全旋转：64个2×2对称协方差，3×64个数；mean128+cov192+logZ1=321个标量/块/KV head。
- 同样存储精度下，full-space rank1 PCA 若把 \(\sqrt\lambda\) 吸收进向量，mean128+factor128+logZ1=257；rank2为385。PC2位于两者之间，不能说rank2和PC2完全等字节。
- Qwen部分旋转若 D=256、d_rot=64、d_nope=192，非旋转部分保留完整对称协方差额外需要18528个数，已不是低成本O(d)状态。保留非旋转diagonal时总量为256+96+192+1=545。只保留非旋转mean则353，但遗漏项更大。不要把这三种实现用同一个PC2成本数字描述。

\(\mathcal D(\Sigma)\) 是 Frobenius norm 下 \(\Sigma\) 到该 block-diagonal 线性空间的正交投影，且每块是PSD，所以总矩阵PSD。但 \(\mathcal D(\Sigma)-\Sigma\) 一般既非PSD也非NSD；它不保证方差只低估或只高估。

## 3. 可证命题二：精确误差是 cross-block covariance 加 tilted Taylor remainder

令 \(Y=q^T(x-\mu)/\sqrt d\)，其 cumulant-generating function

\[
K(t)=\log\mathbb E_w e^{tY},\quad 0\le t\le1.
\]

有限 token 集合、有限 logits 下所有导数存在。定义 tilted weights

\[
w_j^{(t)}=\frac{w_je^{tY_j}}{\sum_kw_ke^{tY_k}}.
\]

有

\[
K(0)=0,\quad K'(0)=0,\quad
K''(0)=q^T\Sigma q/d,
\]

\[
K'''(t)=\mathbb E_{w^{(t)}}[(Y-\mathbb E_{w^{(t)}}Y)^3].
\]

Taylor integral remainder 给出精确恒等式

\[
L_B-\widehat L_B^{\rm PC2}
=\frac{q^T(\Sigma-\mathcal D\Sigma)q}{2d}
+\frac12\int_0^1(1-t)^2K'''(t)dt.
\]

若满足**uniform tilted third absolute central moment bound**

\[
\sup_{t\in[0,1]}\mathbb E_{w^{(t)}}
|Y-\mathbb E_{w^{(t)}}Y|^3\le M_{3,B}(q),
\]

则

\[
|L_B-\widehat L_B^{\rm PC2}|
\le\frac{|q^T(\Sigma-\mathcal D\Sigma)q|}{2d}
+\frac{M_{3,B}(q)}6.
\]

进一步可用

\[
\frac{|q^T(\Sigma-\mathcal D\Sigma)q|}{2d}
\le \frac{\|q\|^2}{2d}\|\Sigma-\mathcal D\Sigma\|_{op},
\]

或者单独使用更方向化的界

\[
\frac{|q^T(\Sigma-\mathcal D\Sigma)q|}{2d}
\le\frac1d\sum_{f<g}\|q_f\|\,\|\Sigma_{fg}\|_{op}\,\|q_g\|.
\]

最后一个写法将 nonrotary子块与rotary子块的所有交叉项也算入；不能只遗漏“频率之间”，却漏掉rotary/nonrotary coupling。

### 不能偷换的条件

- 只控制 t=0 的第三矩不足以控制余项；稀有正尾会在 t接近1时被exponential tilt放大。需要0到1整条tilt路径的界。
- 原始分布 skewness=0 不使余项为0。对称 ±a 在 t=0 的第三矩为0，但tilted分布一般不再对称。
- 小 covariance Frobenius error不保证具体query方向误差小；query可对齐被删掉的cross项。
- 有限块必有某个M3，不意味着这个界有用。若score range很大，界可远大于topK margin。
- 这些误差量一般要访问更多token信息或做离线审计才能计算。不能宣称PC2以O(d)摘要自动给出了低成本topK可靠性证书。

若需要一个简单有限范围界，设Y的取值区间长度为D，则任意tilt后的中心偏差绝对值≤D、方差≤D²/4，故第三绝对中心矩≤D³/4，余项≤D³/24。这是合法但可能很松的界，不能作为“真实分布近似Gaussian”的替代证据。

## 4. 可证命题三：top-K稳定只在margin足够时成立

对某个固定head和固定eligible block集合，若所有块都有 \(|L_B-\widehat L_B|\le\epsilon_B\)，真topK集合S无并列，且

\[
L_i-L_j>\epsilon_i+\epsilon_j,\quad
\forall i\in S,j\notin S,
\]

则近似topK等于S。统一误差\(\epsilon\)下，充分条件为第K与第K+1真logmass间隔大于2ε。也可在候选集合上使用区间分离形式：

\[
\min_{i\in\widehat S}(\widehat L_i-\epsilon_i)
>\max_{j\notin\widehat S}(\widehat L_j+\epsilon_j).
\]

这只证明相对**真实block mass目标**的选集稳定，既不证明native NOSA选集相同，也不证明答案正确。如果有init/local强制块、QK与DMA配额，须先明确哪些块固定、哪个剩余集合用哪个分数排序；定理只适用于该具体受限选择。

在等大小块、单head、固定K条件下，选取最大真实mass块最大化保留attention概率质量。若values满足\(\|v_j\|\le V\)，忽略省略块后重新归一化的attention output与完整输出之差≤\(2V(1-P_S)\)。这是说明mass目标相关性的条件性界，不保证深层残差、MLP、autoregressive生成或完整答案；最重要的答案token也未必拥有最高总体attention mass。

## 5. 必败反例：三个方向的错误都真实存在

下述logmeanexp省略各块相同的logZ常数；cis均为0，块大小均为64，因此不借助不公平的cis/长度差异。

### 5.1 高方差对称尾：PC2严重高估，full covariance也不能救二阶截断

块A有32个score −4、32个score +4。真值

\[
L_A-\log64=\log\cosh4=3.3071882258,
\]

而mean=0、variance=16，PC2=8。块B全部score=4，则真值与PC2均为4。**PC2选A，真实mass选B。** 此例可以全部放在单一RoPE pair里，cross-covariance误差为0，失败完全来自Taylor remainder；rank1/rank2 full covariance的二阶公式同样失败。

随a增大，\(\log\cosh a\sim a-\log2\)，二阶项却为a²/2，过估可无限增长。给correction加一个未经机制推导的cap会形成新超参，不是原PC2理论的推论。

### 5.2 稀有正尾：PC2严重低估

块A有1个score 10、63个score −10/63，因此mean=0，variance=100/63。真值为

\[
\log\left[(e^{10}+63e^{-10/63})/64\right]
=5.8435543386,
\]

PC2仅为\(50/63=0.7936507937\)。块B全部score=3。**PC2选B，真实mass选A。** 只看到variance不大不能推断余项小；exponential tilt迅速把权重集中到正尾。

该例说明：若错选恰来自稀有高相关token，二阶项未必是足够的摘要；Quest/envelope、分槽或tail-sensitive estimator可能更合适。

### 5.3 跨频带协方差：真实score恒为0，PC2却虚构高方差

设两个不同RoPE pair对scaled score的贡献为 \(Y_1=Z,Y_2=-Z\)，Z等概率±2。总score恒为0，真实logmeanexp=0，但每pair variance=4，PC2 correction=(4+4)/2=4。常数score1的块会被PC2错误排在其后。

此例full-space covariance是rank1；**rank1 PCA精确保留负cross-covariance，PC2丢掉它而必败。** 所以“PC2比PCA保留更完整的位置信息”不成立。

相反，若covariance能量均匀分散在很多独立planes，PC2可保留全部二阶项，rank1/2 PCA会丢很多方向。这是两者真正应由实验决定的结构权衡。

### 5.4 split means的优劣也取决于结构

在±4反例中，如果前半块全−4、后半全+4，split2分别保留精确条件均值并合并logmass，会精确恢复\(\log\cosh4\)，胜过PC2；如果正负交替且每个子块mean为0，split2/4仍可能完全遗漏。不能事后按测试query把tokens分到有利split再称query-independent缓存摘要。

## 6. 多head归一化与NOSA配额：即使每head logmass精确，也可能错选

### 6.1 不能把各head未归一化mass直接相加

单head排序可忽略全局normalizer，因为它对所有block相同。跨head共享选块通常需要

\[
A_{hB}=e^{L_{hB}-\log\sum_Ce^{L_{hC}}},\quad
M_B=\sum_h\alpha_h A_{hB}.
\]

各head可以拥有任意不同的logit常数偏移；这不改变其attention，但会任意改变\(\sum_he^{L_{hB}}\)。CPU反例：两块A/B，两head logmass分别为(100,99)和(0,4)。每head归一化后，A/B平均mass为0.374522/0.625478，应选B；未归一化求和被head1压倒而选A。即使PC2在所有head都精确，这种聚合仍错。

GQA中“同一个KV head的queries共享summary”不等于“可以把其query直接求均值后再算PC2”。二次项满足一般\(E_h[q_h^TCq_h]\ne(E_hq_h)^TC(E_hq_h)\)，再叠加softmax后差别更大。应先各query head计算score与normalizer，再按既定group聚合。

若所有合法块在head h都有logmass误差≤ε_h，则其lognormalizer误差也≤ε_h，归一化mass相对误差介于\(e^{-2\epsilon_h}\)与\(e^{2\epsilon_h}\)。因此group score误差还包含normalizer影响；可用

\[
|\widehat M_B-M_B|\le\sum_h\alpha_h A_{hB}(e^{2\epsilon_h}-1)
\]

作保守界，再用group层margin条件。不能直接把单head logmass的2ε门槛当group topK证明。

归一化还应包含fixed local/init块的mass；仅在remote候选内部归一化会改变“当前head多大比例注意力本来在remote”的权重，是另一种目标。

### 6.2 原生query-agnostic DMA配额可以使PC2改进不产生任何任务作用

源码 `compressed_attention` 先按QK分数选一份shortlist并把这些block在cis分数里设为+∞，再按cis补齐剩余TopK。cis由token V生成，在当前query下不重新计算语义匹配。若PC2只替换QK shortlist分数：

- PC2改善的块本来已被cis填充选中，最终mask可能完全不变。
- 真证据位于超过QK quota的下一名，却被大量高cis无关块占掉其余名额；即使QK logmass完全精确也无法保证它入选。
- 若fixed/init/local占了预算大部分，可改变的remote名额太少，score保真提升也可能无task-visible headroom。

反过来，将所有配额合并成单一PC2 joint-mass ranking，改变的不只是covariance，而是query-aware/cis联合路由策略。必须有同logZ、同w、同quotas的weighted-mean控制，以及保持native quotas和统一joint ranking两组分开的消融。不能把去掉query-agnostic quota的收益全归因于RoPE-pair covariance。

## 7. 必须同批出现的强对照

1. **Official native NOSA**：原生32/16均值、64block映射、QK/DMA quotas、cis reader完整保留。它回答真实baseline是否被改善。
2. **Same-cis weighted mean**：logZ+qμ/√d，与PC2同block集合、同head归一化、同配额策略。它隔离covariance项。
3. **Full-space rank1/rank2 COBS-style**：相同w与logZ，post-RoPE actual keys上每block PCA，score加入真实低秩二阶项。明确这是matched implementation/adaptation，不冒称完整COBS官方复现。记构建、更新、summary bytes与query成本。
4. **Split2/split4 means**：固定、query-independent、因果合法分区；每子块自己有logZ_r与weighted mean μ_r，score为\(\log\sum_r\exp(\log Z_r+q^T\mu_r/\sqrt d)\)。它由Jensen给出真实block logmass的下界，不能直接用不带counts/logZ的mean相加。对原分区作嵌套细化时，在精确算术下不会降低这个下界；任意两套不同分区没有这个单调保证。两者都不保证topK或答案。
5. **部署强selector（可用时Quest/envelope）**：同实际读取预算和总成本，检验二阶statistic是否比已有peak-sensitive方法值得。PC2不能只赢同为低阶截断的controls就声称解决稀有相关token问题。

PCA与split都要公平包含cis。若native保留原生quotas，候选与matching controls也应先保留；统一joint-mass ranking是另一条明确消融。Qwen迁移没有NOSA cis时令b=0，保留原生reader，不能用不同模型不同budget做混合平均。

## 8. 成功后可写什么，什么仍不足以支撑新意

### 如果只获得理论性质或offline score收益

最多可写：“RoPE-plane block covariance是一个可缓存、原点不变的structured second-order approximation，误差可分解为cross-plane covariance与higher cumulants。” 这是一项明确数学说明，但projection、cumulant expansion和topK margin本身都很常规。**不足以单独支撑ICLR方法创新，更不足以称稳定accept。**

只比均值好也不足：COBS已使用均值+低秩协方差估计softmax mass；Prism已分析RoPE均值相消。把两者接起来再命名不产生充分新颖性。

### 若完整实验成功，合理窄claim

“在冻结reader频率与权重、保留原生KV的预训练稀疏模型中，按原生rotary planes保存cis-weighted covariance，可在给定summary与读取成本下减少mean-selector遗漏；相对full-space低秩和多均值摘要，在特定资源范围实现更好的完整任务质量—实际成本关系。”

要让这个claim有说服力，至少同时需要：

- 相对native、same-cis mean、rank1/2、split2/4的完整生成或公开任务收益，而不是只提高attention-KL/recall。
- 明确哪个budget区间PC2赢；如果rank2在同延迟下更好，应选择rank2并拒绝PC2主方法。
- 第二个实质不同预训练接口上的迁移，或更充分的主模型质量—成本证据；跨接口不成功就收窄范围。
- 以真实missed-block与可恢复任务为依据，显示收益对应pair内分散能量，且cross-frequency covariance/higher-tail反例对应其失败，而不是只挑漂亮相消toy。
- 正确计入cis、head normalization、quotas和summary构建；不能把这些额外修复的收益归给PC2。

理论能让失败可解释、让实验有针对性；它不能提前保证成功。今晚最有价值的结局有三种：PC2确实赢强对照而值得扩展；PCA/split更好而选择已有结构更合适的方案；所有selector没有可恢复任务headroom而停止这项投入。第三种不是科研突破，但比在错误接口上整夜训练更有决策价值。

## 9. 一晚决策的最低充分结论

PC2可以作为第13个、且目前最可落地的候选参加首批对照；**不要在开始前认定PC2就是最终论文方法**。已知必败例要求它接受PCA与split的正面竞争，不能事后移除这些baseline。第一批应保留raw token输出及EOS、逐样本mask/预算、matched强对照与耗时；如果只记录summary拟合误差，就无法回答今晚研究问题。

首批失败时先按可恢复headroom、covariance approximation、higher cumulants、head aggregation、quota作用和额外成本定位。只有定位到具体机制且新干预可在剩余预算内独立验证时才修正；不能把“二阶不够”当理由自动开启任意rank、分频、temperature或cap扫描。

来源：NOSA已下载官方模型源码；本地`DESIGN_SOURCE.md`对 [COBS](https://arxiv.org/html/2607.09052v1)、[Prism](https://arxiv.org/html/2602.08426v2) 的既有一手核对。本文没有新增穷尽性查新结论。

### CPU 数学核算回执

用64个8维随机向量、非均匀cis权重、split-half配对、独立二维旋转，NumPy双精度参考计算得到：blockdiag共轭等变最大差0；PC2原点平移score差0；full-space rank2 PCA的同一score差3.33e−15。以1000区间Simpson积分核算tilted第三中心矩，精确误差分解残差5.48e−15。前述三个解析反例也直接计算出本文数值。它们核查公式和反例，不核查NOSA GPU实现、kernel成本、真实任务或PC2有效性。

