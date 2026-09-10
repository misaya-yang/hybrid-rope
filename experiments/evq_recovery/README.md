# EVQ：真实长输入适配与非几何网格比较

本轮只做开卡前准备。研究问题是：包含 FFN 的 LoRA 在足够的真实长输入监督下能否恢复能力，以及 Cosh 相对其他非几何网格是否有优势。训练 loss、PPL 和任务生成分别判断。

## 基座与训练范式

使用已缓存的 `allenai/OLMo-2-0425-1B-Instruct`，revision `48d788eca847d4d7548f375ad03d3c9312f6139e`，实际约1.485B参数、原生4096 tokens。选择它是为了在单卡上以真实16K输入测试超出原生窗口的适配；不把仍位于Qwen原生32K内的结果叫作外推。它的指令能力存在局限，因此同时保留原始Native和同配方YaRN对照，不把共同地板当作某个网格失败。

模型来源：[官方模型卡](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)。

- 全部层的 Q/K/V/O、gate/up/down FFN 使用 LoRA，r32、alpha32、dropout0；基座其它参数冻结。
- AdamW，LR2e-5，betas(.9,.95)，无weight decay，梯度裁剪1；相同初始权重、adapter seed、数据顺序和token预算。
- 每次更新：一段完整16K PG19密集下一token CE、一条完整长指令的答案＋EOS CE、一条短能力replay CE。三个平均loss的权重分别1、1、0.25，各自记录真实监督token数；不是仅在长输入末尾几个答案token上训练。
- 第一观察点8M CPT tokens，第一训练段32M，可沿同一预先固定的128M调度继续到64M/128M；同时另外记SFT/replay的token成本。32M不是LoRA能力上限，也不按墙钟时间强制杀任务。
- 保存可续训的未合并adapter及optimizer。判断学习是否充分要看验证loss走势和完整生成；预算结束本身不证明方法失败。
- 同数据的全参数入口已准备，只用于必要时区分LoRA限制与共同训练问题，不自动加入首轮。该路径的GPU内存仍需实测。

这些是待验证训练配置，不宣称最优超参数。当前不是从零预训练比较，结果也不能自动推出从零训练时的网格排名。

## 有限对照集与执行顺序

| 表 | 构造 | 作用 |
|---|---|---|
| Native | 原始几何表 | 原能力、通用适配收益及纯形状基线 |
| Cosh | τ=2的midpoint分位数，归一化并锚定Native两个端点 | 当前论文固定范围构造的参考点，不称最优τ |
| Exponential | 与Cosh匹配归一化指数RMS形变，λ≈1.47547 | 简单、已知接近Cosh的非几何对照 |
| Hybrid | 保留前16个高频pair，低频Cosh重分配；匹配相同RMS，局部τ≈2.83714 | 早期Hybrid思想的受控比较 |
| YaRN | 固定s4官方方程和幅度1.138629 | 实用扩窗参照；范围和幅度不同，不是纯形状归因 |

前四臂频率端点一致，gain=1；三个非几何臂的RMS形变约0.129239。Hybrid保持高频一半时无法达到该形变量，因此在模型输出产生前固定为历史r16对照。匹配的是一个明确的形变幅度，不代表全部几何性质相同。实现和精确数组见准备目录的 `tables.json`。

1. 真实16K训练smoke只验证更新、内存和耗时，丢弃其adapter；不能作为能力结果。
2. 原始Native、未训练Cosh、未训练YaRN完整生成；随后Cosh与YaRN接受同配方适配，比较训练前后和短能力保留。
3. 训练配方有可判读能力结果后，Native、Exponential、Hybrid使用相同数据和预算，复用Cosh结果。
4. 若Cosh-LoRA仍失败：先判断是否仍在学习、是否共同任务地板；同表全参数比较可用于进一步定位，不继续靠增加推理补丁代替训练。

“Cosh最好”最多指实际受测范围及预算。τ=2胜过匹配对照并不能证明整个Cosh族最优；若要提升为方法优越性，需要给竞争构造相同的开发机会。首轮可以直接发现Cosh参考点可被替代，不能预先承诺某个候选获胜。

## 已准备数据

训练与保留：

- PG19：128本官方train书，1,573个连续16K预测窗口，一遍25,772,032预测tokens。窗口不拼短文、不重复filler、不拉伸position IDs。官方validation/test各24本。
- LongAlign：1,536条完整英文长指令训练行；开发/测试各192条。实际prompt超过4096，prompt＋完整回答不超过16384。来源为固定revision的公开LongAlign-10k；assistant答案保留其合成数据身份。
- 短replay：512行，四类各128。历史source-separated Native池的开发256行、测试1756行；不冒称这些历史材料从未被查看。
- LongAlign按文档前缀group分离；对QASPER全部开发/测试原文做32-word exact-passage重叠检查，包含错位匹配。它不等于语义去重证明。

能力与LM评测：

- 官方QASPER：167开发、186测试问题，完整原文不截断；按原始论文source分组。主要指标为完整响应token F1，normalized exact及EOS分开。
- RULER：固定作者生成器，single-key、multi-key、variable tracking，开发/测试各288行。保存实际token长度，保留原生成器的未填满行，不用名义长度冒充真实长度。
- LM：同一PG19留出窗口上的4K/8K/16K/32K full与tail NLL，单次backbone计算两个读数。
- core阶段使用各任务/长度固定前8行及各Native类别前16行；full阶段使用准备好的完整池。模型输出不参与挑题。多答案RULER使用官方答案项recall，不把“命中其中一个答案”标成完整exact。

QASPER的16–32K桶只有开发8题、测试2题，不能承担强自然任务外推结论。16K主要回答训练后恢复；32K的RULER和LM测试超出适配长度的泛化。这些数据是首轮研究准备，不是完整论文验证矩阵。

来源：[PG19](https://github.com/google-deepmind/pg19)、[LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k)、[QASPER](https://huggingface.co/datasets/allenai/qasper)、[RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)。

## 使用

服务器准备根目录：`/root/autodl-tmp/evq_recovery_20260910`。从其 `code` 目录运行；数据、plan、频率表位于上一层。所有入口默认只检查或打印；`--execute`才加载GPU。

```bash
cd /root/autodl-tmp/evq_recovery_20260910/code
/root/miniconda3/bin/python -m experiments.evq_recovery.validate --root .. --verify-model
/root/miniconda3/bin/python -m experiments.evq_recovery.launch --root .. --phase smoke --execute
/root/miniconda3/bin/python -m experiments.evq_recovery.launch --root .. --phase recovery --cpt-tokens 8388608 --execute
```

首段结果分析后，可用相同命令将token端点改为33554432继续；已完成的baseline和checkpoint会复用。形状比较使用 `--phase shapes`，同样指定匹配token端点。全参数条件使用 `train --regime full`，不混用LoRA checkpoint。

配对分析入口：

```bash
/root/miniconda3/bin/python -m experiments.evq_recovery.compare --candidate CANDIDATE_EVAL_DIR --baseline BASELINE_EVAL_DIR --out paired.json
```

QA按论文source聚类，RULER按生成输入聚类；分别输出各任务/长度的胜负、均值差和区间，不把不同指标拼成一个分数。

## 准备检查及限制

CPU检查覆盖：实际OLMo频率表及匹配形变、密集CE数值/梯度、FFN实际梯度、adapter保存重载、超窗KV-cache、完整答案评分、多答案语义、数据索引、续训顺序和错位原文去重。完整模型的GPU内存、吞吐和能力仍必须在开卡后测量。

数据下载首次遇到截断，通过Range补齐并核对PG19云MD5；LongAlign全文件SHA256匹配官方LFS身份。生成器依赖缺失及初次数据字段错误均已在无卡阶段处理；旧失败日志保留于准备材料。源码、实际计数与服务器读回结果见同目录 `PREPARATION.json`。
