# EVQ-Cosh 的低遗忘长窗适配方案
## OLMo-2-1.485B → Llama-3-8B；RTX 5090 32GB

日期：2026-09-12。

**本文是一份待执行方案，不包含新训练成绩。** 历史数字来自附件；外部事实来自列出的原始论文和官方文档；新超参、预算吞吐、验收门槛均为设计值或假设。配套 YAML 是研究训练合同，不是可以不加修改就传给 TRL 的配置文件。

## 0. 最终建议

采用 **固定的、原生采样端点锚定的 EVQ-Cosh 表 + 真实长序列 + 短长混合训练 + Q/K/V/O LoRA + 低学习率 FFN LoRA + 原模型短窗输出约束**。

第一轮不要继承已经发生遗忘的旧适配器，不联合学习频率，不增加新的注意力结构，不通过短输入切回 Native 来获得“无损”。从发布的 Instruct 原始权重重新开始，用同一张 EVQ 表处理短输入和长输入。

主配方：

| 项目 | OLMo 主合同 | Llama-3-8B 扩展合同 |
|---|---|---|
| 原模型 | OLMo-2-0425-1B-Instruct，实际约 1.485B | Meta-Llama-3-8B-Instruct，不换成 3.1 |
| 原生窗口 | 4096 | 8192 |
| 真实长窗阶段 | 8K → 16K | 16K → 32K |
| Q/K/V/O | 全层 r64、α128 | 相同 |
| FFN gate/up/down | 全层 r16、α32 | 相同；先不增加 rank |
| QK / VO / FFN 峰值 LR | 5e-5 / 2.5e-5 / 1.25e-5 | 2e-5 / 1e-5 / 5e-6 |
| 每次更新 token slots | 65,536 | 131,072 |
| 更新次数 | 4000 | 2000 |
| 总 token slots | 262,144,000 | 262,144,000 |
| 长度阶段分界 | 1000 步 | 500 步 |
| warmup | 200 步 | 100 步 |
| 精度 | BF16 | 先资格测试 BF16 32K；不适配时统一改为 NF4 QLoRA，见 §10 |
| EVQ | 默认 anchored-midpoint τ=1；开发期完整比较 τ=2 | 搬运 OLMo 锁定的构造与 τ，不重新搜表 |

说明：slots 包含必要的批内 padding，**不是自动等于实际看过的 tokens**。要求 token fill ≥95%，因此主合同至少包含 249,036,800 个 non-padding 输入 tokens；日志同时记录 slots、non-padding、loss-bearing tokens、unique documents/examples 及重放次数。预算公式按 slots 作保守工作量估计。若 fill 不足，先修 packing/分桶，不把空位记作训练规模。

这是约 0.25B 级实际输入的训练，而不是 50 步修补，也不是原来 1376 行反复循环三遍。

---

## 1. 历史证据真正支持什么

### 1.1 已确认的三件事

**第一，PPL 改善不能代替任务成功。** 第一代 8B 的 Native-LoRA / EVQ-LoRA 在 32K PPL 上为 991.48 / 127.91，但注册 QA 的 F1 为 0.2110 / 0.1126，短输入缺口尤其明显。[F1 §2.1]

**第二，只有 answer-only 加少量 replay 也没有解决保留。** 第二代 8B 用 1376 行、516 步、33.8M 输入 tokens；EVQ 的 16K RULER 提升到 0.140，但 8K 自然文本 PPL 为 12.27，未微调模型为 7.95。其 128/1376≈9.30% 是 replay 的**样例占比**，不能直接当作其监督 token 或梯度占比。[F1 §2.2；9.30% 为计算值]

**第三，显式任务监督与终止监督确实可以修复部分读出。** OLMo 的 8K/16K 完整答案加 EOS 达到 98/60%，但这是物理 ≤4K、目标区相位暴露和合成任务下的结果。Q/K 续训继承并冻结已训练的 V/O；它不是“从原始模型仅训练 QK 就足够”的证据。[F1 §2.4–2.5]

论文第 39 页 Table 24 还保留了 OLMo Q/K-only 的短窗 RULER 代价：Native 72.19%，EVQ 42.44%。因此，EOS 个别任务成功并不意味着原生能力已经稳定。[F2 p39]

### 1.2 不能据此确定的原因

不能认定“全序列 CE 必然破坏检索”“FFN 不训练就是唯一瓶颈”“33.8M 太少就是唯一原因”。历史协议同时改变了表、训练分布、监督方式、预算和优化路径。750M 全参数 500M-token 成功与 8B LoRA 失败也不是容量的单因素对照。[F1 §2.7、§4]

新的计划要分别回答：能否学稳；是否比合理基线更好；FFN 或短窗约束是否承担了可识别的作用。

### 1.3 两处来源审计

1. 档案时间线把 OLMo 的 182.7→159.6 写作“零训练换表”；当前论文 p39 的对应记录却明确是从 step-0 起、2.097B-token 匹配训练。本计划**不把该数字作为零训练基线**，必须以真实 released checkpoint 重新测 step 0。[F1 §2.0；F2 p39]
2. 新档案提示旧 S-NIAH/passkey 归零序列的臂与长度归属无法从原始报告核实。即使稿件现有文字给了归属，也不在这里拼成新的比较。新测试明确保存每个方法、长度和 token 输出的身份。[F1 §2.1、§4]

---

## 2. 外部经验怎样改变本次配方

| 外部工作 | 原文支持的做法 | 在本计划中的用途与边界 |
|---|---|---|
| YaRN | 在真实长序列上适配；原论文用 PG19 64K segments、400 steps、global batch64，LR2e-5，AdamW β=.9/.95，零 WD | 支持真实长上下文训练，不支持“所有 YaRN 训练都是 QA SFT”；也不能把其 400 步照抄成小 batch400 步 |
| LongAlign | 长指令数据与短通用指令混合；packing 要处理跨序列注意力和 loss weighting | 支持短长并训、任务级 loss 归一化，不让答案长度决定样例权重 |
| LongRoPE2 | 短窗用原 RoPE、长窗用扩展 RoPE，训练与推理都区分两者 | 它的短窗保留不能直接证明“同一张长表也无损”；本计划主线更严格，短窗学生同样用 EVQ |
| LongLoRA | 长上下文适配中考察 embedding/norm 的训练，并使用 shifted-sparse attention | 借鉴训练目标与模块经验，但不直接照搬稀疏算子，避免引入另一个核心变量 |
| QLoRA / LoRA Learns Less and Forgets Less | 全 transformer 线性层目标、rank、数据与 LR 对结果都有影响；不同任务上 rank 结论不相同 | 支持受控引入 FFN，不支持“all-linear 一定成功”或“rank 一定无关” |
| PEFT / TRL 官方文档 | rank/alpha patterns、经典与 rsLoRA 缩放、assistant mask 的具体约束 | 落实参数与 mask 验收，避免开关打开但目标模块或监督区域错误 |

按 YaRN 原文名义值计算，400×64×65536=1,677,721,600 token slots。它与本计划约 0.26B、原档案约 0.034B 是不同数量级。这个计算不是在声称本计划一定需要相同数据量，而是说明**步数本身不可比较**。[W1；计算]

---

## 3. 为什么要用真实长输入，而不是继续只训练原生窗

原生窗口训练中的 token 对最大真实距离受窗口约束。把 position IDs 拉大可以暴露特定相位，但没有同时复现长输入中的 token 数、干扰项竞争、远程证据排列和真实文档结构。

因此本次训练直接提供两种信号：

- 长自然文本提供扩展窗口中的语言建模和上下文适配信号。
- 长监督任务提供“必须利用远处来源才能生成完整正确输出”的信号。

短文本不是删掉，而是作为保留目标与长数据交错训练。训练到 16K 后，16K 成绩称为**相对原模型的长窗扩展/适配**；只有未在该长度训练的 32K 评价，才属于该适配模型的额外长度外推。不能再将两者混称。

本计划不声称短窗训练绝不可能外推；论文已有特定外推结果。判断是：为了当前要求的可靠长程能力，不能再把希望主要放在未覆盖的长距离分布自动迁移上。

---

## 4. EVQ 表：有限重选，不做大扫描

### 4.1 固定端点与网格

取 checkpoint 实际的 FP32 native inv_freq，提升至 FP64 做构造：

\[
a=-\log\omega_0^N,\qquad R=\log(\omega_0^N/\omega_{K-1}^N),\qquad u_k=(k+1/2)/K.
\]

\[
Q_\tau(u)=1-\frac{\operatorname{asinh}((1-u)\sinh\tau)}{\tau},\quad
z_{\tau,k}=\frac{Q_\tau(u_k)-Q_\tau(u_0)}{Q_\tau(u_{K-1})-Q_\tau(u_0)}.
\]

\[
\omega^E_k=\exp[-a-Rz_{\tau,k}].
\]

输出 FP32 后，首尾显式覆盖为原生 FP32 首尾；验证正值、严格递减、K 不变、实际采样端点逐位一致。τ=0 的实现直接返回原始 tensor，不重新生成一个近似 Native。上述公式按论文 Eq.(9)–(10)、Eq.(52) 构造。[F2 p5、p25]

这里的“同量化器/网格”指频率表的有限采样约定，不是 4-bit 权重量化。

### 4.2 主线 τ=1，开发对照 τ=2

推荐起点 τ=1，是因为成熟模型已经与 Native 槽位绑定；先采用低于历史 OLMo τ=2 的分配强度，再给足真实长数据与监督，比直接假定历史默认表最优更合理。**这只是偏保留的设计选择，不是理论最优值或安全证书。**

τ=2 完整跑同样预算，以检验较强分配是否在真实长训练后提供更好的长端收益。两个候选都使用上面的 anchored-midpoint，不在本轮同时扫描 grid、base、gain、每层频率和可学习 z。

不默认增加 2.5：scratch factorial 中 1.25×Cosh 更常获益，不能推出成熟 LoRA 的最优点也应更强。只有已经学稳且确实要检验强度敏感性时，才作为额外预定邻点，不能在 final test 上不断搜优。[F1 §0、§4；F2 p27]

### 4.3 表不随训练长度变化

8K→16K 时，不自动用 d/√L 重新计算 τ；Llama16K→32K 也不改表。否则同一次训练就在追逐不断移动的位置基，保留和长度收益无法对应到一张固定表。

OLMo 选中的公式与 τ 搬到 8B，用 8B 自己的实际采样端点，重新记录 tensor hash。两个模型头宽、K 等配置须以 runtime config 验证，不把模型名称当作足够证据。

### 4.4 失败与备选

- τ=1 长端不足而 τ=2 通过联合门：选 τ=2，再用独立种子确认。
- τ=2 native 代价大而 τ=1 学稳：选 τ=1，论文称为新的 anchored 工作点，不宣称旧表被原样修好。
- 两者均失败：先依据 §11 区分数据、优化和表问题。只有 native 代价明确成为共同瓶颈时，增加一个预指定 τ=0.5 分支；不是开启连续扫描。
- 联合学习 z 不列入主线：已有 oracle 显示 tail NLL 与全序列代价并存；它会把本来要验证的静态 EVQ 适配变成另一个问题。[F1 §2.6]

---

## 5. 模块、rank、α 与优化器

### 5.1 主模块安排

**Q/K/V/O 保留 r64/α128，FFN gate/up/down 新增 r16/α32，覆盖所有层。**

Q/K 是直接承接新旋转基的投影；V/O 和 FFN 给后续表征转换与输出组织留出适应空间。后半句是动机而非已识别机制：FFN 的作用必须由对照检验。不给 FFN 和 QK 同样的 LR，避免同时大幅改动原模型所有功能路径。

其余：classic LoRA、dropout0.05、bias=none、B 初始化为零；冻结 embeddings、lm_head、所有 norm、base weights 和 inv_freq。不新增 token，不修改 tokenizer vocabulary。

### 5.2 FFN 对照必须怎样解释

开发阶段直接比较：

- E1-A：同表、同数据、QKVO-r64。
- E1-F：同表、同数据、QKVO-r64 + FFN-r16。

两臂共用 QKVO 初始 adapter bytes；新增 FFN 不能改变其他模块随机初始化。所有基座权重一致，数据顺序、batch、loss 和 schedule 一致。

基于 OLMo 官方 config：hidden2048、intermediate8192、16层，LoRA 参数量按 `r*(in+out)` 求和：[W7；计算]

- QKVO-r64：16,777,216。
- FFN-r16：7,864,320。
- 合计：24,641,536。

因此 E1-F 优于 E1-A，首先说明“**增加这组 FFN 适配容量**”有用，不能仅凭它断言 FFN 位置不可替代。

要强化定位结论，增加 attention-only **r94/α188**：它在该 OLMo 架构上恰好也是 24,641,536 参数。若 E1-F 仍优于 r94 控制，才能更有力地说明参数放在 FFN 比继续增加注意力 rank 更有效。该严格参数等量关系只在已核验 OLMo 结构成立，不照搬给 GQA 的 8B。

### 5.3 LR、调度与防遗忘

| 设置 | 主选择 | 理由 | 备选/回退 |
|---|---|---|---|
| QK LR | OLMo5e-5；8B2e-5 | 保留历史 QK 可修复的量级，8B 比旧1e-4更保守 | loss 不稳定时各组减半；不是首先砍 rank |
| VO LR | QK 的1/2 | 允许输出通路适配但限制偏移 | native 任务回退时先下调 VO，而非冻结已学出的行为冒充原生 |
| FFN LR | QK 的1/4 | 新增容量同时控制更新速度 | 原生代价只在加 FFN 后出现：减半 FFN LR；仍无收益则选择 no-FFN 臂 |
| rank / α | QKVO64/128；FFN16/32 | α/r=2，避免变 rank 时同时改变经典 scaling | 容量不足的明确迹象下，单独试 FFN32/64 或等参数注意力控制 |
| AdamW | β=.9/.95，ε1e-8，WD0，clip1 | 与长窗训练及历史 OLMo 优化设置接近；因子 WD 不等于功能保留约束 | 梯度异常先检查 mask/precision/accumulation，再减 LR |
| schedule | 5% warmup；cosine 降到0.1×peak | 切长窗时还保留足够 LR，最后收敛；不另设突然的高 LR 长窗阶段 | 追加训练使用所有比较臂一致的新续训合同，不只续落后臂 |

LR、rank、dropout 和 KL 系数均是本计划设计，不是已经验证的最优超参。

### 5.4 为什么不先叠 rsLoRA / DoRA / LoRA+

PEFT 的 classic scaling 是 α/r，rsLoRA 是 α/√r。r64、α128 时，它们分别为2和16。直接开 rsLoRA 而保留 α 会把缩放放大8倍，无法当作纯 rank-stability 比较。[W6；计算]

先稳定普通 LoRA，再决定是否需要新参数化。norm 解冻可作为受控容量备选，但 embedding/lm_head 全量解冻不是“少几个参数”，尤其在大词表模型上会明显增加优化状态与遗忘风险。LongLoRA 的 embedding/norm 经验不能无条件移植成当前最优设置。[W4、W7]

---

## 6. 数据：以实际 token 与真实依赖距离定义

### 6.1 两阶段比例

比例按固定 token-budget updates 安排，每20次更新使用下表给出的家族计数，再在周期内固定随机交错；不是先训完一种家族再换另一种。

| 数据家族 | 阶段A：前25%预算 | 每20步 | 阶段B：后75%预算 | 每20步 |
|---|---:|---:|---:|---:|
| 短自然 LM |25%|5|20%|4|
| 短通用 SFT |25%|5|20%|4|
| 长自然 LM |25%|5|30%|6|
| 长自然任务 SFT |20%|4|20%|4|
| 长合成检索 SFT |5%|1|10%|2|

阶段A：OLMo 长样本用真实8K，Llama用真实16K。阶段B：分别真实16K/32K，至少80%的长输入 token 量来自长度≥目标窗75%的单个 causal segment。

整个预算约42.5%短、57.5%长。短样本同样安装当前方法的静态表，只有冻结 teacher 用 Native。

### 6.2 五种家族分别做什么

**短自然 LM。** 用训练合法、与评价按文档隔离的自然文本；OLMo 主要2/4K，Llama主要4/8K，也包含更短块。用于约束自然语言分布，而不是只保住一个 passkey 模板。

**短通用 SFT。** 包含知识问答、推理、格式要求、不同长度回答和适量多轮。保留原 chat template，不把全部任务改成“只输出一个数字”。优先使用权利与来源可核查的通用指令集或项目已有合规 replay；附件没有给出现成规模，必须实际审计。

**长自然 LM。** 真实长文章、书籍或文档流；主力是有连续性的文档，不把很多相互隔离的短段 pack 成16K后称为16K依赖训练。若使用 PG19 等，应固定训练 split 和文档名单；现有 temporal holdout 的三个域不能作为回放来源直接混进去。[W1]

**长自然 SFT。** 原始 LongAlign / 原始 LongAlpaca 中经过新 tokenizer 长度审计的完整样例，以及公开 QA 的 training split。覆盖单文档抽取、多跳证据和文档理解；不只使用既有8K切片。LongAlign 的8K–64K和“10K样例”是发布数据描述，实际可用16K/32K数量必须在筛选后统计。[W2]

**长合成 SFT。** 复用 RULER/反事实生成器接口，不复用旧训练行。新 key/value、模板、文档与评测隔离；覆盖单针、多键、多值、变量关联等，平衡证据位置和答案长度。可用配对来源值替换作数据增强，但主线先用普通正确答案 CE，不再额外叠大权重 margin loss。

### 6.3 必须验收的长依赖

长 QA 至少50%样例的可验证 gold-source→answer 距离超过原生 L0；近、中、远位置均覆盖。多跳任务分别记录每个 gold 块距离，不能只靠离答案很近的最后一条提示完成。

每条记录保存：实际 tokenizer 后长度、每个 causal segment 长度、source offsets、answer offsets、loss mask、generation reserve。任何裁剪不得移除 gold 或正确答案尾部。

长自然 QA 的有用性不能靠增加无意义 filler 保证；合成任务里的 distractors 必须与长度/位置作为明确实验因素记录，不能把该协议成绩伪装为原始自然 LongBench 榜单成绩。

### 6.4 规模与泄漏

监督样例默认最多访问2次；若合格独立样例不足，先补数据，不能重放旧1376行十几遍却声称获得0.25B独立信息。自然 LM 也要报告 unique documents / tokens 与重复比例。

旧 temporal24packs、2Wiki200/长度和旧 RULER 面板继续作为开发/回归资产。确认集必须在配方锁定后使用预留文档与新生成样例；换随机文件名不等于独立。训练集、开发集、最终确认集在**原始文档/基础问题**层分割，所有改写与不同长度版本跟随同一基础样本。

---

## 7. 损失：不是全序列与 answer-only 二选一

### 7.1 自然 LM 和指令监督分开归一化

自然文本：

\[
L_{LM}=\frac{\sum_{t\in V}-\log p(x_t\mid x_{<t})}{|V|}.
\]

指令任务：

\[
L_{SFT}=\frac1n\sum_{i=1}^{n}\frac{\sum_{t\in A_i\cup E_i}-\log p(y_{it}\mid x_i,y_{i,<t})}{|A_i\cup E_i|}.
\]

A 是完整 assistant answer，E 是实际 end-of-turn / EOS token。每个样例先平均，再在同一 family 的 update 中平均；不能因为一个回答更长就让它支配整个任务。

prompt label 是 -100，**不代表 prompt 的前向或梯度被切断**。必须保留从答案 loss 经注意力、prompt hidden states 到 QK/FFN 的反传。对 prompt 采用 no_grad 或 detached KV 是另一个训练问题，本计划不采用。

家族通过 §6 的 update 频率控制，**不要在 sampler 按20%抽样后，又把 loss 乘0.2**。记录各家族 CE、有效 target 数、每样例权重和共享 adapter 上的梯度量级。

### 7.2 短窗行为约束

teacher 为发布的原始 BF16 模型、原生表、adapter关闭；student 为当前方法表和当前 adapter。只在不超过原生长度的短 LM/SFT 上计算：

\[
L_{short}=L_{LM\text{ or }SFT}+0.2\,D_{KL}(p_T^{Native}\Vert p_S^{current\ table}).
\]

温度T=1。0.2是设计值。保留目标是原模型短窗输出行为，不是所有短任务都照抄 teacher 的错误；正确标签 CE 始终存在。

不在长输入上用原生 teacher 蒸馏，否则可能把要修复的长端失败也固化。行为保留蒸馏的思想有已有研究基础，但它在 EVQ+LoRA 上能否达到这里的联合门槛仍是待检验问题。[W8]

### 7.3 5090 上可实现的 KL

离线为短训练样本生成 teacher cache；同一模型的所有比较臂复用同一个 cache，不常驻第二个8B模型。

每个短样例预定抽取至多128个有效目标位置。保存 teacher top64 token IDs，与 gold token、end-of-turn IDs 取并集，并保存剩余词表的总概率质量。student 仅对这些位置计算 logits 与归一化，再计算“保留项+其余项”的聚合分布 KL。

这是真正的**聚合分布 KL**，不是完整词表 KL；按 log-sum inequality，它不会大于全词表 KL。不能把 top-k 重新归一化后假装剩余概率不存在。teacher tail mass 与 student logsumexp 用稳定精度计算；极小 tail 做数值保护并报告。

若实现复杂度成为阻塞，可以改为较少选中位置上的完整 KL、在线串行 teacher 前向，但必须计入耗时和显存；不能把取消 KL 默默当作同一配方。

### 7.4 EOS 不另外放大

主线用正常 answer+end-of-turn CE，不增加独立大权重 EOS term，不在答案开始处教模型提前终止。记录 terminal-EOS、空回答、cap hit 和回答长度分布。

自然 LM 的人工裁剪处不强行标注文章结束；指令数据则必须保留正确结束标记。Llama 的 end-of-turn 与 generic eos 可能不同，以 tokenizer/chat template/generation config 实际定义为准。

若使用 TRL `assistant_only_loss=True`，必须确认模板支持 assistant mask；官方文档明确需要相关 generation markers。不能仅打开一个开关就认为监督范围正确。最小验收打印若干样例的 token IDs、mask 和目标文本逐一核对。[W6]

---

## 8. 训练预算、阶段与正式对照

### 8.1 一个完整训练合同

OLMo 4000更新，每次65536slots；8B 2000更新，每次131072slots。两者的token预算和阶段比例相同，8B更大的更新包避免32K SFT每次只有两个长样例造成过大的样例梯度噪声。该作用是设计动机，不是实测结论。

前25%预算是真实2×原生长窗，后75%是真实4×。所有阶段同时有短输入；不另设一次“纯短训练修复 native”的收尾，以免最后又抹掉长程使用。

保存25/50/75/100%预算点。主终点固定最终 checkpoint；最后两个开发 checkpoint 需稳定通过联合门后才称为“学稳”。中间点属于同一完整 schedule，不是分别为较小预算优化过的最终模型。

### 8.2 OLMo 开发五臂

所有臂从相同 released Instruct 原始权重重新开始，单开发 seed42，完整相同数据和预算：

| 臂 | 频率与 gain | LoRA | 回答的问题 |
|---|---|---|---|
| N | 原始 Native，g1 | QKVO64+FFN16 | 原始表在同样真实长训练下能学到什么 |
| E1-A | anchored Cosh τ1，g1 | QKVO64 | 无 FFN 基准 |
| E1-F | anchored Cosh τ1，g1 | QKVO64+FFN16 | 推荐主线 |
| E2-F | anchored Cosh τ2，g1 | QKVO64+FFN16 | 强度与适配的交互 |
| Y | 固定官方 YaRN s4，官方 gain | QKVO64+FFN16 | 实际竞争基线 |

YaRN gain 按锁定实现的公式，常见g=1+0.1ln4≈1.138629。它与 E/N 的 gain 不同，因此 Y 是**部署配置级比较**，不是纯频率形状因果对照；N/E 的固定端点、固定gain对照承担形状识别。[W1]

Y 从新的原始模型起、用新的真实长数据训练，与历史“给失败 adapter 后置套 YaRN 看能否恢复”的 pilot 不是同一实验。保留旧负结果，不重复那个 pilot。

每张表都先跑 step0 无 adapter 的相同开发评价。分开记录：换表当下代价；训练新增变化；最终相对原始 Native 的总代价。只比较最终对原模型的差，无法区分 conversion shock 与训练遗忘。

### 8.3 锁定与确认

先依据开发联合门选 τ 和模块。若多个配置通过，先选长自然 QA 更高者；相差不足1pp时选 native NLL 更低者；仍接近时选更少参数。这个排序是设计规则，不在看到 final test 后修改。

锁定后对 selected-EVQ、Native、YaRN 使用 paired training seeds137、256，共六条确认训练。各 seed 内数据与初始化配对。两个确认 seed 必须都完成，不能因为第一个不利就改第二个配方。

可选的两个机制消融优先于新模型堆叠：OLMo attention-only r94等参数控制；同一配方去掉短 KL。它们必须跑同预算，不能用50步结果承担机制阴性结论。

---

## 9. 稳定判据与 8B 触发条件

以下全部是**新方案的工程/研究验收目标，不是现有成绩**。阈值可以在正式结果开封前由作者调整，一旦锁定，不以模型刚好差一点为由追着结果移动。

### 9.1 原生保留

| 指标 | 判据 |
|---|---|
| 自然文本2/4K（8B为4/8K） | domain-macro PPL ratio相对untouched Native的95%上界≤1.05；各域点估计≤1.10 |
| 窗内自然 QA | whole-response F1差的95%下界≥−3pp |
| 窗内 RULER | official macro差的95%下界≥−3pp；另列所有任务避免平均掩盖崩坏 |
| 通用能力小面板 | 固定 HellaSwag/ARC-Challenge 等短选择题面板，宏平均差95%下界≥−2pp；任务与prompt模板先锁 |

PPL门等价于 ΔNLL的上界≤ln1.05≈0.048790；各域10%对应ln1.10≈0.095310，均为算术换算。

原生 QA 不通过，不能用“PPL几乎无损”代替能力保留；反之亦然。只是不显著变差也不是通过非劣门。

### 9.2 真实长端能力

| 指标 | OLMo 工程目标 |
|---|---|
| 真实8K/16K RULER13 macro | 分别≥60%/50%；相对同表step0的改善需有配对证据 |
| 简单 passkey / S-NIAH 完整答案+终止 | 8K≥90%，16K≥70% |
| 16K自然QA | 对 matched Native-LoRA 和同EVQ表step0的F1改善点估计分别≥3pp，且配对区间下界>0 |
| 输出终止 | 针对短答案、可回答任务 terminal rate≥98%，空回答≤1%；cap hit与检索/格式失败分别列出 |

70%等值是“值得继续投入8B”的最低工程目标，不是 Cosh 理论推导出的成功率。复杂自然任务不要求全部使用同一个小生成cap；cap和结束规则按任务标签分布提前设定。

### 9.3 竞争力与论文主张分开

能通过保留与长端门，支持“EVQ可稳定适配”。要宣称比现有方法更优，还必须比较同数据、同预算的 YaRN。

启动8B前建议再要求：EVQ在预定长自然QA与RULER主要宏指标上对YaRN的非劣边界为−3pp，没有明显较差的证据；同时保存 tokens-to-gate 的同schedule曲线。若EVQ仅超过没有扩展设计的Native，而远输YaRN，就先解决OLMo差距，不急着把成本放大到8B。

### 9.4 最终触发条件

必须同时满足：两条新确认 EVQ 训练完成且通过保留、真实长端、终止门；两个最后阶段的开发点稳定；对照与数据身份完整；YaRN竞争门没有失败；8B长窗实现资格测试通过。

不能拿单个seed、单一合成任务98%、或者tail NLL胜利来替代这个触发器。任一关键区间跨门槛，结论为未定，不是默认通过。

### 9.5 测试资产与样本量

沿用现有 temporal、RULER、2Wiki、EOS测试器，不重新发明评分方法。但旧数据已经参与反复分析，开发/回归与新确认必须分开。

建议新确认 temporal：三域各32个独立长packs，合计96；按原文档分组、domain-macro NLL计算；一次固定表的最长因果前向可聚合原生前缀与中间前缀，前提是不触发动态RoPE或长度条件gain。

RULER：新生成13任务×40例×三个长度。EOS：每长度200条简单闭合答案案例。自然QA：旧2Wiki200/长度用于开发；确认目标至少600个独立基础问题作原生/长输入配对，并补足不同自然任务类型。600是待构建的设计规模，不是假设附件已有。自然完整文档不够短的任务不能粗暴裁剪后计入native门。

样本量应在配方开封前用开发配对差标准差定精度：
\[
n\approx (1.96\hat\sigma_D/h)^2.
\]
例如仅作为规划假设，σD=.35、半宽h=.03，需要约523例，选600留余量。这不保证任意数据都达到3pp精度，也不是三训练seed变成600个训练重复。最终报告逐seed结果；prompt/document bootstrap仅度量给定训练产物下的评价变异。

若合格新样本不足或区间太宽，不夸称已经证明无损。可以保留现有结果为开发证据，再补确认数据。

---

## 10. 5090 32GB 的执行方案

### 10.1 先核验，不拿推理KV估训练显存

使用现有锁定环境开始：torch2.8.0+cu128、transformers4.57.6、peft0.17.1、trl0.24.0等。新路径的硬件资格包括：真实长batch、全部目标模块、loss反传、optimizer step、gradient checkpointing、teacher-cache KL、重载后的同表输出。[F1 §1]

OLMo官方结构为16层、hidden2048、MHA16头、intermediate8192、native4096、base500000。[W7]

8B的旧8K LoRA吞吐8383tokens/s是档案实测，但其训练JSON没有记录运行卡身份，**不能直接标为5090实测**。20.4GB长前向峰值也不能推出32K训练一定装得下。[F1 §1]

### 10.2 主内存路径

- long microbatch1；短样本分桶/packing；梯度累积达到更新token预算。
- BF16基座，稳定精度adapter与优化状态；attention为全因果Flash路径。
- `use_cache=False`训练；不存解码KV；不detach prompt。
- gradient checkpointing；长MLP必要时对token块分段并配合重算，**仅切块但保留全部中间反传张量并不一定节省峰值**。
- Liger fused linear CE / chunked lm_head，不能生成完整 `[batch,L,vocab]` FP32 logits。
- KD只计算抽样位置logits；teacher离线，8B不能同时常驻两个BF16副本。
- 8B32K评测串行，不复用历史四路并发OOM方案。

对新packing作等价检查：两条短样本独立前向，与同一packed buffer的block-diagonal/varlen前向应在数值容差内一致；长样本与其prefix在静态表条件下应相符。标签mask不能代替attention隔离。

### 10.3 8B32K BF16不适配的回退顺序

先micro1、checkpoint、fused/chunked head、真正可重算的MLP分块；均须实测。

仍不适配，则在正式8B训练前选定以下一种，并冻结所有比较臂的选择：

1. **主推荐的5090回退：全部8B训练臂统一NF4 QLoRA。** Native、EVQ、YaRN使用相同量化基座、配置、dtype、数据和训练预算；RoPE相位构造仍在FP32，不能量化inv_freq。分别报告相对BF16 untouched Native的总代价，以及相对同NF4 untouched基座的增量。论文写QLoRA，不写作旧BF16协议复现。[W5]
2. 大显存机器实际可用时，完整比较块用Pro6000 BF16训练；不能只让EVQ换精度或硬件后仍宣称完全匹配。
3. 严格不接受权重量化且只能5090时，先确认真实16K训练、32K作为未训练长度评价。它仍是8K→16K真实扩展，但**不再声称完成32K物理训练**。

不在 BF1616K 已训练了一半后无记录地切为4-bit继续跑；选择改变时建立新实验身份。QLoRA的训练吞吐也要单独测，不能直接沿用下表BF16规划假设。

---

## 11. 按症状回退，不再靠随机组合超参碰运气

| 观察 | 首先能判断什么 | 下一步，单一主要改动 | 不允许的解释 |
|---|---|---|---|
| step0换表即损失大，训练后持续恢复 | conversion shock与可恢复程度 | 完成预算比较τ1/2；必要时预定τ.5 | 把初始差直接判成训练遗忘或理论不可能 |
| native LM和短任务同步变差，长端上升 | 保留与新适配冲突 | 短KL .2→.5；其余保持；新开发合同 | 宣称更多长数据一定免费修复 |
| native LM保住，但短QA/指令退化 | 文本建模约束未充分保护任务行为 | 阶段B短SFT20→30，长LM30→20 | 只增加LM replay或只看PPL |
| 长PPL/答案NLL改善，但完整生成差 | 概率改善没有转为行为 | 先审mask/结束token；再长自然SFT20→30、长LM30→20 | 又拿attention hit替代成功 |
| 加FFN后才显著退化 | 新容量/更新路径可能造成代价 | FFN LR减半；与noFFN/r94比较 | FFN永远不能动，或FFN必然必要 |
| native都好，远证据仍访问失败 | 目标距离/证据竞争或频率工作点不足 | 验证真实gap覆盖；完成τ1/2比较 | 用统一position-ID平移冒充新依赖 |
| 两个Cosh均不过，但Native/YaRN正常 | 当前Cosh配方/工作点的适配失败 | 新的共享range-operator对照，见下文；或诚实保留负结果 | 宣称所有内部指数分配无效 |
| 训练与开发都欠拟合且末段仍持续改善 | 预算不足是候选解释 | 全部关键比较臂同起点增加同样续训预算，使用新增数据 | 只把EVQ续到赢，或把总processed当unique |
| 8B失败而OLMo成功 | 跨模型可迁移性尚未建立 | 审GQA映射、mask、dtype、batch与KV；再考虑受控rank调整 | 把1.485B成功直接推广为8B结论 |

任何数据配比、LR、τ、KL回退都只在开发阶段进行，保留失败轨迹。确认阶段不临时改配方；工程错误则先停止修正，并重新标识受影响结果。

### 可选的共享范围算子后备

若共同问题指向扩展范围而非读出，可将同一组、按Native一次性定义的官方index-ramp权重w_k冻结：

\[
\mathcal R_s(\omega)_k=\omega_k\big[(1-w_k)+w_k/s\big].
\]

从原始模型分别训练 `Native + R_s + LoRA` 与 `anchored EVQ + R_s + LoRA`，同端点、同ramp、同gain、同数据预算。w不能在EVQ表上另算一遍。该比较问的是“在共同扩展策略下增加内部非几何分配是否有收益”。

这不是在失败旧adapter上后置套YaRN，也不能将成功写成“EVQ alone零额外处理”。它是新的组合协议；只在诊断需要时执行，不占第一轮五臂预算。

---

## 12. 预算：算清真实长窗成本

### 12.1 吞吐假设

下面不是实测，也不是租赁报价。q表示该完整训练配置下的规划tokens/s；实际资格测试后必须替换。

| 模型 | short | 阶段A long | 阶段B long |
|---|---:|---:|---:|
| OLMo |18,000（≤4K）|12,000（8K）|8,000（16K）|
| Llama8B |5,000（≤8K）|3,000（16K）|1,500（32K）|

按已指定预算与比例，名义分量：短111,411,200slots；A长32,768,000；B长117,964,800。

\[
H=1.2\left(\frac{T_S}{q_S}+\frac{T_A}{q_A}+\frac{T_B}{q_B}\right)/3600.
\]

其中1.2是加载、保存、编译及执行余量的规划因子，不包括独立评价、数据准备和teacher cache。

| 工作 | 训练GPU·h估计 |
|---|---:|
| OLMo一臂 |7.89|
| OLMo开发五臂 |39.44|
| OLMo确认三方法×两seed |47.33|
| 额外r94或去KL，每臂 |约7.89|
| 8B一臂 |37.28|
| 8B初次三方法比较，一seed |111.85|
| 8B每追加一个完整三方法seed |再约111.85|

核心OLMo开发+确认与首轮8B训练合计约198.62GPU·h。若q减半，可变训练成本近似翻倍，不能用20%余量掩盖。

### 12.2 评价绝不是免费

生成评价按：
\[
H_{eval}=1.2\sum_L\left(I_L/q_{prefill,L}+O_L/q_{decode,L}\right)/3600+H_{load}.
\]

建议规划OLMo prefill在4/8/16K分别20K/12K/8K input tokens/s、decode80 output tokens/s；8B在8/16/32K分别6K/3K/1.5K、decode40。均为假设。

最终新确认面板包括RULER、EOS、扩大的自然QA与native能力面板。按完整规模，预留**OLMo每产物约5–8GPU·h、8B每产物约20–30GPU·h**，再用实际prompt长度与输出cap重算。开发期只跑固定小开发面板和回归仪器，不能每100步重复完整确认集。

teacher cache独立一次性生成：按短数据量约111Mslots、规划teacher prefill分别30K/8K tokens/s，纯前向约1.0h/3.9h；算上选中位置head、写盘和加载，分别先预留2h/5h。同模型所有方法复用，不重复计每臂teacher训练成本。

如果按假设5090单价¥3/h换算，198.62训练GPU·h约¥596；**这不含评价、teacher、失败重跑、闲置费、存储与人工**。完整工程投入明显高于这个纯训练数。

两台5090优先分工为独立匹配训练/评价或两臂并行，缩短wall time但不减少总GPU·h。已有4080训练链不打断。Pro优先接确认评价与teacher缓存，长序列训练需要时整块承接。

---

## 13. 执行顺序与交付物

**第一步，数据/实现资格。** 锁源文件、模型与环境；确认table/no-adapter forward、目标模块、loss mask、source/answer真实距离；在8/16K测OLMo完整反传与内存。对新增8B32K路径单独测试，不将一次推理通过当作训练通过。

**第二步，OLMo五个开发臂。** 每臂完成预定预算，记录同阶段能力曲线、native代价、step0与训后变化。所有数字落在同一个只读结果清单，不仅保存“最好那条”。

**第三步，锁定配方并做两seed三方法确认。** 根据§9验收。配置或数据改变即重新区分开发和确认，不能保留旧“held-out”名字继续选型。

**第四步，8B三方法首轮。** 先锁BF16/NF4实施路线，再从原始权重开始。移植的是方法和训练原则，不是把OLMo adapter跨架构搬过去。首轮通过后再安排额外完整seed，不先并行租满机器押注。

**第五步，论文图表。** 一张联合图同时显示native保留和长端真实任务；一张同schedule学习曲线；FFN受控对照与必要消融进入附录。旧两代8B负结果保留，标明新协议相对旧协议改了哪些因素；不能声称单一因素已经解释全部历史失败。

若仍沿用9/18摘要、9/25全文节点，摘要只列已经完成的比较；未完成8B不预写成功。排程按实际吞吐和资格结果重算，不降低正式确认标准赶日期。

---

## 14. 正负结果如何进入论文

| 最终观察 | 能写的主张 | 不能写的主张 |
|---|---|---|
| EVQ联合门通过，Native相对较弱 | 真实长窗、低秩适配能把该非几何表转成保留良好的长任务能力 | 普遍优于所有扩展方法 |
| EVQ和YaRN都通过，质量近似但EVQ更早过门 | 在该schedule/模型上适配更高效；需完整曲线与同等比较 | 任意最优schedule下都更省token |
| EVQ同预算长端可靠优于YaRN且native非劣 | 该构造具有实用增量，成为适配线的强正结果 | 由Cosh密度先验推出全局最优 |
| EVQ通过但不胜YaRN | 适配可行、历史失败并非必然；方法优势仍有限 | 把工程修复包装成新的SOTA |
| PPL好，任务没过 | 仍只有概率/建模收益，适配线没有闭环 | “学会长程能力” |
| 合成检索好，自然QA不过 | 合成任务条件下有效 | 通用长上下文修复 |
| 只在short→Native routing下保留 | 双路部署方案保留了短窗 | 单表EVQ无损 |
| FFN胜noFFN，不胜等参数r94 | 额外容量有帮助，FFN特异性未识别 | FFN是唯一或不可替代机制 |
| OLMo通过、8B未过 | 小模型配方成立，跨架构/规模迁移未建立 | 8B已修复或方法可任意放大 |
| 当前τ1/2均失败、YaRN正常 | 本轮静态Cosh工作点的低预算适配不成功 | 所有内部指数分配不可能学稳 |

**本轮的成功定义只有一个：同一张固定EVQ表、同一个训练产物，在原生保留门和真实长任务门上同时成立，并在合理训练基线前有明确定位。**

---

## 参考依据与数字溯源

### 用户材料

- [F1] `LoRA微调简报_OLMo与Llama8B_20260912(1).md`，2026-09-12，§1硬件、§2历史协议、§3已试/未试、§4关键张力、§5可复用资产。文中行号对应对话所载210行版本。
- [F2] `main(20260912-090735).pdf`，《Beyond the Base: Exponent Allocation in RoPE》，49页，Eq.(2)/(9)/(10)/(52)、Appendix F、Tables22–24、28。
- [F3] `RoPE实验计划简报_供Pro细化_20260912.md`，训练seed与评价样例区分、开发与确认隔离、实际token计数等原则。

### 外部原始资料

- [W1] Peng et al., **YaRN: Efficient Context Window Extension of Large Language Models**, arXiv:2309.00071v2 / ICLR2024，§3.4、§4.1。原文PG19长段训练不是全部为instruction SFT。
- [W2] Bai et al., **LongAlign: A Recipe for Long Context Alignment of Large Language Models**, arXiv:2401.18058；Findings of EMNLP2024；作者数据仓库LongAlign-10k。采用长短指令混合并处理packing的loss权重。
- [W3] Shang et al., **LongRoPE2: Near-Lossless LLM Context Window Scaling**, arXiv:2502.20082 / ICML2025，§3.3。短长使用不同RoPE与inference路由。
- [W4] Chen et al., **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**, arXiv:2309.12307v3 / ICLR2024。
- [W5] Dettmers et al., **QLoRA: Efficient Finetuning of Quantized LLMs**, arXiv:2305.14314；Biderman et al., **LoRA Learns Less and Forgets Less**, arXiv:2405.09673v2 / TMLR2024。前者部分短指令实验rank影响较小，后者在更大分布迁移任务发现rank与性能有关，不能合成万能结论。
- [W6] Hugging Face **PEFT LoRA configuration**（检索v0.17.0文档）与 **TRL SFTTrainer v0.24.0** 官方文档。项目已锁peft0.17.1，实际部署前以安装版本签名验证，不在实验中途升级。
- [W7] AllenAI官方模型配置：`allenai/OLMo-2-0425-1B-Instruct/config.json`，hidden2048、intermediate8192、16层、16Q/16KV头、native4096、rope_theta500000。
- [W8] Li & Hoiem, **Learning without Forgetting**, arXiv:1606.09282。作为功能/输出保持蒸馏的历史依据，不作为EVQ成功的实验证明。

外部资料检索日期：2026-09-12。方法选择、验收阈值、τ1推荐、混合比例、LR、rank组合、吞吐与时间都是本方案提出的设置，除显式注明的原文数字外，不归因给这些文献。
