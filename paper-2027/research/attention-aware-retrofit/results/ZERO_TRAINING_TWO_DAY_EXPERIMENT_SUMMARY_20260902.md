# Zero-training pure-`z`：2026-09-01—02 两日实验汇总

- **日期：** 2026-09-02
- **性质：** 内部两日综合报告；不是新的独立实验 owner
- **范围：** mature-checkpoint、权重冻结、单张静态 RoPE 表、无长度 routing
- **当前判决：** `LONG_POSITION_METHOD_ESTABLISHED / REPEATED_QA_CONVERSION_BARRIER_CONFIRMED`

## 1. 总结

这两天没有得到“全面解决长上下文”的方法，但也绝不是没有结果。

已经建立的是：成熟 checkpoint 中存在不可交换的 rotary
subspace–frequency coupling；一张冻结的 normalized-index pure-`z` 表可以在
OLMo、Qwen 和 Gemma 的已测协议上保留较高 Native 能力，并显著改善长程 NLL、
RULER/NIAH 与远端 source likelihood。这个效果主要来自频率表，而不是 attention
gain。

尚未建立的是：同一张表可以稳定改善自然生成 QA。2026-09-02 的 30-row
far-evidence panel 中，index 的自然 QA point estimate 低于 Native，区间跨零；后续
source-contrast decoding 和三候选 sequence reranking 也没有恢复生成分数。因此当前
最准确的结论是：

> normalized-index pure-`z` 已经是有效的长位置语言建模与证据传输方法，但从
> “远端证据进入 logits”到“正确答案成为 autoregressive top-1 并正确停止”仍有断点。

这不支持“`z` 无用”，也不支持“自然 QA 已解决”。

### 1.1 这个 QA 断点不是今天才发现

此前 EVQ-LoRA 已经多次暴露同一层级差异，今天的结果必须放在这条历史链上理解：

- generic-data Q/K-only LoRA 可以改善 long-position natural NLL，却不能恢复 broad
  task capability；
- 七个不含 RULER/NIAH rows 的 4K-only LoRA 方案全部低于 length-matched controls。
  最好的 8K RULER screen 只有 `0.0750`；最强 one-token natural-retrieval arm 在自己
  的 held-out task 上 first-token top-1 仅 `0.0391`，mean answer NLL 为 `3.7357`；
- sparse Native-teacher KL 也在该搜索中失败，不能因为今天重新写成 teacher
  consistency 就视为未经测试的新答案；
- 显式加入 2Wiki/RULER task-family supervision 后，EVQ-LoRA 才在长程 2Wiki 上从
  matched Native 的 `0.48%/0%` 提升到 `15.09%/4.63%` F1（8K/16K），同时仍明显
  损伤 4K。后续 request-length routing 保住了 Native 短程，但它不是当前要求的单表
  pure-`z` 方法；
- 另一轮 complete-family continuation 的 SQuAD/Hotpot QA 基本不变，收益主要来自
  VT/CWE/FWE 与 NIAH；8B 结果也早已显示 probability/source dependence 未稳定转成
  top-1 generation、QA accuracy 或整体 LongBench 提升。

历史 owners：

- [`OLMO2_1B_SHARED_TASK_LOCALIZATION_20260730`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SHARED_TASK_LOCALIZATION_20260730.md)
- [`OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_NON_RULER_ADAPTATION_SEARCH_20260731.md)
- [`OLMO2_1B_OVERNIGHT_EXPERIMENT_SUMMARY_20260726`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_OVERNIGHT_EXPERIMENT_SUMMARY_20260726.md)
- [`EVQ_COSH_REBUTTAL_PRINCIPLES`](../../../../rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md)

所以 2026-09-02 不是首次发现 QA 问题。它新增的是：同一障碍在**零权重更新的
normalized-index 静态表**上再次出现；table×gain 对照排除了 gain 主因；
full/ablated bridge 又把断点缩小到“远端 source 已改变正确答案 likelihood，但没有
稳定改变 autoregressive winner”。把它表述成一个全新的 QA 问题是不准确的。

## 2. 证据层级

本报告分开两种证据。

1. **本地 canonical owner：** 2026-09-01 的报告和 JSON receipt 已在仓库中，下面的
   数字可由链接 owner 核验。
2. **2026-09-02 会话回执：** 远端运行完成后记录的统计量；本地脚本已绑定父结果
   SHA-256，但远端 raw JSON/JSONL 尚未回收到本工作区。服务器现已不可连接，因此
   这些数字可用于内部决策，不能在导入 raw owner 前升级为论文证据。

## 3. 2026-09-01：静态 pure-`z` 方法与跨模型证据

### 3.1 OLMo 单表结果与 ordered coupling

OLMo 的冻结 log-s4 表使用

\[
\omega'_k=\omega_k s^{-m_k},
\]

同一张表覆盖 1x、2x 和 4x。其 1x PG-19 PPL retention 为 `0.875302`，五任务
macro retention 为 `0.915103`；2x/4x PG-19 NLL 为 `3.083278/3.081946`，完整
RULER-13 在 4K/8K/16K 为 `0.71397/0.66705/0.49859`。这些结果属于
[`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md)。

同一最终 frequency multiset 的 slot permutation 造成决定性坍塌：OLMo 1x NLL
由 `3.10423` 变为 `6.86493`，Qwen 64K core-4 由 `0.7000` 变为 `0`。因此无序
频谱分布 \(\pi(\omega)\) 不足以描述 retrofit；真正受 checkpoint 约束的是 learned
rotary subspace 与 frequency/dilation 的有序配对。

### 3.2 低维压缩并不等于功能等价

两参数 clipped-affine `C2` 对 OLMo empirical movement 的 MAE/RMSE 只有
`0.001223/0.006118`，zero-refit Qwen geometry 也很接近原 transport 表。CPU 几何
结果见
[`CPU_LOW_DIM_COUPLING_LAW_20260901`](CPU_LOW_DIM_COUPLING_LAW_20260901.md)。

GPU 行为却暴露了误差度量的局限：C2 的 OLMo PPL retention 为 `0.870971`，Qwen
Native macro retention 为 `0.868902`，均略低于预设 `0.875` 门；long behavior
基本保留。结论不是“再加一个参数”，而是 movement-space RMSE 不能代替
checkpoint 的功能距离。见
[`LOW_DIM_COUPLING_GPU_RESULT_20260901`](LOW_DIM_COUPLING_GPU_RESULT_20260901.md)。

### 3.3 normalized-index transport 是当前工程规则，不是普适定律

K32 fresh N80 中，physical/index 的 64K macro 为 `.466250/.461250`，长程差异未
解决；但 32K Native retention 为 `.859459/.923243`，只有 index 通过门。index
在同批 64K 比 matched YaRN 高 `+.064375`，paired 95% CI
`[.027500,.102516]`。见
[`K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901`](K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md)。

独立 K128 N80 中，index/physical 为 `.790000/.728125`，差 `+.061875`，95% CI
`[.028109,.096250]`。这关闭了 physical-`x` 作为跨 K 特权坐标的强主张，但没有
证明 normalized rank 是唯一或普适坐标。见
[`K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901`](K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md)。

K32 full RULER-13 给出最强 breadth confirmation：

| Arm | 32K macro | 64K macro | 32K retention |
| --- | ---: | ---: | ---: |
| Native | `.547821` | `.220513` | `1.000000` |
| normalized-index | `.559167` | `.514551` | `1.020711` |
| YaRN-s2 | `.559423` | `.453654` | `1.021181` |

64K index-minus-Native 为 `+.294038 [.250192,.338271]`，index-minus-YaRN 为
`+.060897 [.027627,.095835]`。但是 index 在 variable tracking 和两个 QA rows
上没有统一胜出。见
[`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901`](K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md)。

K64 同代对照进一步说明结果不是统一支配：C2-s2/YaRN-s2 的 64K core-4 为
`.6325/.6950`。见
[`QWEN_S2_SAME_FAMILY_IDENTIFICATION_RESULT_20260901`](QWEN_S2_SAME_FAMILY_IDENTIFICATION_RESULT_20260901.md)。

Gemma K128 在修正 Native reference length 后，index s4 的 4K/8K/16K 为
`.9450/.8650/.7950`，并通过 4K retention 门；但该结果不能单独识别 K 因果或坐标
唯一性。见
[`REFERENCE_CORRECTED_K128_RESULT_20260901`](REFERENCE_CORRECTED_K128_RESULT_20260901.md)。

## 4. 2026-09-02：自然文本、自然 QA 与断点定位

本节是对既有 EVQ-LoRA capability-conversion 障碍的 zero-training 复现与机制收缩，
不是新发现一个 QA 问题。自然 QA gate 本身仍有价值，因为它第一次在当前冻结
normalized-index 表、自然 far-evidence prompts 和 matched Native/YaRN 下直接测试该
问题；但 gate 失败后再继续做 decoder/rerank rescue 的信息增益很低，未应继续扩张。

### 4.1 Packed-natural NLL：通过

固定 Qwen K32 normalized-index 表在 32 条 paired packed streams 上得到：

| 长度 | Native NLL | Index NLL | 主要判定 |
| --- | ---: | ---: | --- |
| 32K | `2.597666` | `2.615399` | PPL retention 约 `0.9824`，短程代价可接受 |
| 64K | `2.754945` | `2.630842` | index-minus-Native `-0.1241`，95% CI `[-0.1479,-0.1029]` |

64K 的 32/32 streams 均改善。index 与 YaRN 的 64K 差异未解决，因此该实验建立
的是自然长位置 NLL 改善，不是对 YaRN 的统一优势。

本地代码中的父回执绑定为
`ba489f47070d2dd9058afe50fa7c9db1229f50eb2bc364445dfdb5c7815712a9`；raw
receipt 尚待回收。

### 4.2 Far-evidence natural QA：未通过正向门，但不是显著负效应

30-row 2WikiMultihopQA/Qasper/HotpotQA panel 的 macro 为：

| Arm | Macro |
| --- | ---: |
| Native | `0.13229` |
| normalized-index | `0.10174` |
| YaRN | `0.11197` |

index-minus-Native 为 `-0.03055`，paired 95% CI `[-0.1168,+0.0486]`。因此正向
QA gate 未通过，但区间也不支持“index 已被证明伤害自然 QA”。正确表述是自然 QA
仍未解决。父回执 SHA-256 为
`6109434ea596b42706e5ce295a1348052ec4d01b22c0849529bf23f2dbef793c`。

### 4.3 Table × gain 归因：NLL 主要来自表

Native/index table 与 unit/index gain 的 2×2 对照显示：index table + unit gain 保留
了几乎全部长 NLL 收益；gain 不是自然 QA 断点的主要来源。当前自然 NLL 结果应归因
于 exponent table 为主，而不是 gain-only。factorial result 的本地绑定 SHA-256 为
`5d6f2f2e7dc4cc8961e1d42d87931e03d3cf540a3af4a7b7d9c776d66b5a3d9b`。

### 4.4 Evidence-position bridge：远端证据确实进入了答案 logits

在 far/near/ablated 的 matched prompt bridge 中：

- Native far source-use 为约 `+0.066`，区间跨零；
- index far source-use 为约 `+1.504`，区间为正；
- index-minus-Native source-use 为 `+1.438`，95% CI `[.720,2.230]`；
- far 条件下 canonical answer NLL：index `3.695`，Native `5.123`；source ablation 后
  两者均约 `5.19`。

这说明 index 的改善不只是“模型能在绝对位置 64K 继续做局部 LM”：远端 source
对 canonical answer likelihood 的影响明显增强。但 near 条件受 instruction scope
混杂，不能从该对照声称纯粹的 position penalty 已被分离。

### 4.5 两条 readout rescue 均未成功

1. **Source-contrast greedy decoding：** index contrast-minus-greedy 为
   `+0.00079 [-0.0271,0.0292]`，没有恢复分数；index 与 Native contrast 的点差约
   `-0.0288`。该路线关闭，不做 contrast coefficient sweep。
2. **现有三候选 sequence rerank：** rerank 相对 index 为
   `+0.0067 [-0.0173,0.0330]`，macro `0.1142`；相对 Native 仍低约 `-0.0181`。
   三候选 oracle macro 为 `0.18656`，说明候选中存在一定 headroom，但现有 source
   score 无法可靠选择。该路线关闭，不做 beam/alpha/rerank sweep。

这些负结果定位的是 readout rescue，不是否定 pure-`z` 已建立的 NLL、RULER 与
source-utilization 效果。

## 5. 两日后可以与不可以声称什么

### 已建立

- 同一 frequency multiset 的 slot permutation 会崩溃；unordered spectrum 不充分。
- mature checkpoint 的有效对象是 learned rotary subspace 与 dilation 的 ordered
  coupling。
- 一张冻结的 pure-`z` 静态表可以跨多个已测 checkpoint 改善 long NLL 和 synthetic
  long-context capability，同时保留可接受的 Native 能力。
- normalized-index 是目前最有证据的跨 K transport rule；physical-`x` 没有实证
  特权。
- 在 Qwen K32，自然 64K NLL 与远端 source-conditioned answer likelihood 均明显改善。
- gain 不是上述自然 NLL 收益的主要解释。
- zero-training 表重复了历史 EVQ-LoRA 的 capability-conversion barrier；今天新增的是
  更干净的 table/gain 归因和 source-to-logit 定位，不是 QA 障碍本身。

### 尚未建立

- normalized-index 是 canonical、唯一或普适的 K-law；
- 当前 profile 是第一性原理推导或全局最优；
- pure-`z` 已稳定改善自然生成 QA；
- index 统一优于 YaRN、PI、NTK 或 Resonance；
- source likelihood 改善必然转化为正确首 token、完整答案和停止行为；
- C2 的低 movement RMSE 能保证功能等价。

## 6. 对下一步理论的约束

今天可以再关闭一个看似合理、实际退化的想法。设部署尺度为 \(S\)，
\(\omega'_k(m)=\omega_kS^{-m_k}\)。如果只把同一短序列的所有 position ids 统一乘
\(r=S^c\)，则

\[
\omega'_k(m)\,r\Delta
=\omega_kS^{-(m_k-c)}\Delta.
\]

因此“整体稀疏拉伸后的 teacher matching”与把所有 \(m_k\) 平移同一个常数严格
等价。它主要识别全局 PI 模式，局部二阶近似不能凭空导出 non-affine allocation。
这是一条代数结论，不是新的 GPU 结果。

## 7. 从失败分叉推出的候选解

现有证据没有推出另一条解析 ramp；它推出的是一个更具体的选择问题：能否仅移动
`z`，把模型**已经具备的短程任务计算**运输到 target-range phases，同时不破坏
Native computation。

### 7.1 先修正问题定义

当前 Qwen-0.5B 在 far-evidence QA 上的 Native macro 只有 `0.13229`。对模型短程本来
就不会回答的样本，任何 `z` 表都不可能凭空创造推理能力。候选构造因此只能读取：

1. Native 在 compact/short prompt 上答案正确；
2. source ablation 确实降低正确答案 margin；
3. calibration source、question 与最终 long test 完全分离。

选择只使用 Native short behavior，不读取任何 candidate long outcome。最终 estimand
不是任意 QA 的绝对分数，而是候选表能否把已存在的 short capability 运输到 long
layout；绝对 long QA 仍需作为冻结后的外部结果另报。

### 7.2 唯一允许的干预

从当前 normalized-index 表 \(m_0\) 出发，仍只允许

\[
\omega'_k=\omega_kS^{-(m_{0,k}+d_k)}.
\]

模型权重、LM head、gain、prompt、decoder 和 KV-cache 语义全部冻结。校准数据复用
历史成功 query-gap 链中真正必要的结构，而不是它的 LoRA 参数：

- correct-source/value-swapped counterfactual pair；
- compact 与 target-range query/source phase exposure；
- target-range view 使用非仿射 source/query layout，并包含真实 distractor，避免 §6 的
  全局缩放等价；
- answer first token、完整 suffix 与 terminal EOS；
- 独立 compact natural-LM/QA retention rows。

### 7.3 优化对象必须是“谁赢”，不是单个答案 NLL

今天已经看到 canonical answer NLL 大幅下降但 greedy QA 没有改善。因此对每个
short-correct 样本，主损失应直接使用正确首 token \(y\) 相对最强竞争 token 的 margin：

\[
\mathcal L_{\rm margin}
=\operatorname{softplus}
\!\left(1-\left[\ell_y-\max_{v\ne y}\ell_v\right]\right),
\]

再加 answer suffix/EOS CE 与 correct-source 对 swapped-source 的 preference margin。
Native compact KL/PPL 和当前 packed-natural NLL 只作为约束，不能代替这个目标。

只求一次最小共同方向，而不是训练 32/64 个自由参数到收敛。令 \(g_j\) 是不同
task、scale 和 source-counterfactual cell 的 signed behavioral gradient，求

\[
\max_{d,\gamma}
\quad \gamma-\frac{\lambda}{2}\lVert d\rVert_2^2
\quad\text{s.t.}\quad
g_j^\top d\le-\gamma\ \ \forall j,
\]

并加入 Native retention、packed-natural NLL、frequency order 与有限步长约束。
只有在 source-disjoint splits 上都得到 \(\gamma>0\) 的同向结果，才保留这一条
`d`；步长只由独立 compact/target-phase calibration gate 冻结，不能看最终 64K QA
后再调。

### 7.4 它如何逃出已经失败的类

| 已失败路线 | 为什么失败 | 本候选的实质差异 |
| --- | --- | --- |
| 2026-08-24 两文档 direct-`z` | 62 effective DOF 用两篇文档的 2x tail NLL 标定，row heterogeneity 直接过拟合 | 多个 short-correct、source-sensitive task cells；目标是 answer margin/EOS；只取一个 robust direction |
| 2026-07-31 sparse Native-teacher KL | 在 Tulu/retention batch 上保持 teacher；没有要求同一任务在 target-range source/query phase 下保持答案 | compact teacher 与 target-phase student 成对；显式 source/value counterfactual |
| answer-only / natural-span LoRA | generic objective 未教会 held-out task computation，最强 one-token arm 连自己的 held-out task 都失败 | 只运输 checkpoint 已经正确执行的 short task，不要求 `z` 发明新能力 |
| source-contrast decoder / rerank | 后处理放大已有 source signal，却不改变产生错误竞争 token 的内部 logits | `z` 方向直接由 first-token competitor margin 的梯度决定 |
| lm-head readout 草案 | 修改输出权重，离开 pure-`z` 主线 | LM head 与所有 weights 冻结，只改一张静态 exponent table |

这仍然是未运行的候选，不是结果。但它是目前唯一同时解释历史成功与失败、并且没有
退化为 geometry selector、新 ramp、gain sweep 或 readout adapter 的方案。

### 7.5 最小判别与失败含义

第一门不需要 64K generation：在 compact physical token budget 内，用 target-range
position IDs、source/value counterfactuals 和真实 distractor，读取几十个
short-correct rows 的 `z` 梯度。

- 若跨 task/scale 的 robust \(\gamma\le0\)，则当前单表 pure-`z` 在 \(m_0\) 附近无法
  同时运输这些已知任务；不启动 long QA，也不换曲线。
- 若 \(\gamma>0\) 但独立 target-phase validation 不复现，说明方向过拟合；停止。
- 只有方向和有限步长都通过，才冻结一张表，做一次 packed-natural retention 与一次
  short-solvable far-evidence 64K confirmation。

该门若失败，限制的是“全层全头共享的一张静态 `z` 表”的表达能力，不是否定训练期
allocation、LoRA co-adaptation 或 dynamic/headwise 方法。若通过，它第一次给出从旧
LoRA 成功机制到 pure-`z` 单表的直接桥，而不是再猜一个频率公式。

## 8. Owner 与回收状态

| 内容 | 当前 owner / 状态 |
| --- | --- |
| OLMo log-s4、Qwen self profile、permutation | 本地 canonical owners，见 §3 链接 |
| C2 CPU/GPU | 本地 canonical owners，见 §3 链接 |
| K32/K64/K128 transport 与 RULER | 本地 canonical owners，见 §3 链接 |
| 2026-09-02 packed-natural NLL | 会话回执；raw remote owner 未导入 |
| 2026-09-02 natural QA | 会话回执；raw remote owner 未导入 |
| table×gain、evidence bridge、contrast、rerank | 本地代码含父哈希绑定；raw remote owner 未导入 |

在远端 raw JSON/JSONL 回收并校验前，§4 只能作为内部决策证据。对外论文数字仍应从
§3 的 canonical owners 引用。
