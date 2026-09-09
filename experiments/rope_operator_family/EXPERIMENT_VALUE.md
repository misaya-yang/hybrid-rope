# 本轮实验：Core Experiment、用途与失败处理

2026-09-09。当前处于**方法定位阶段**：推进一个方法，并准备一个与核心归因问题对应的对照。本轮对照明确为同架构、同初始化、同优化预算的 attention 输出蒸馏。每次运行一份配置；不生成多任务或参数网格。

## 1. 本轮要验证的 claim

**在相同 KV 存储预算下，内容加权的 RoPE 算子族拟合，能够比简单频率折叠更好地保留相对位置计算，并修复真实的远距离依赖。**

它包含两个相连的效果：相对位置计算保留得更好，以及这种改进对模型回答有用。实验直接测这两个效果。方法定位阶段取得清楚的配对改进，就是值得继续的结果；不要求先完成一套完整论文的所有验证。

## 2. 最能体现 claim 的 Core Experiment

### 同预算的远距离依赖修复

**一个当前方法 + 一个简单对照 + 同一种依赖任务。**

| 项目 | 具体设置 |
|---|---|
| 当前方法 | 当前代码拟合后的 A/B、旋转频率和内容/value latent |
| 唯一归因对照 | 同一紧凑架构的 attention 输出蒸馏；A/B、频率、内容/value 都训练，优化机会与当前方法匹配 |
| 共同起点 | 从同一份保存好的 FreqFold/PCA 参数加载；它是共享初始化，不充当归因对照 |
| 预算 | 主配置保持内容192 + rotary64个实数；两者缓存布局、dtype和总字节相同 |
| 输入 | 相同的长上下文、问题、正确答案、tokenizer和生成设置 |
| 任务 | 一个有确定答案的远距关联检索，例如从相似干扰记录中读出指定随机key的value |
| 机制读数 | 同一份留出 Q/K 上的相对位置响应误差，以及完整 attention 输出误差 |
| 实际效果 | 同一问题的答案是否正确；需要连续读数时测正确答案的条件 NLL |

例如上下文远处有 `key=amber_728, value=blue`，其他位置有类似但不同的key/value，最后询问 `amber_728` 的值。该任务直接需要保存内容与远距离关联，答案可以核对。先固定一个长度和预算，测试这一种任务；需要复核时沿用同一种任务的实例，不先换一排 benchmark。

最清楚的正结果是：**经过匹配优化的输出蒸馏对照读错，本方法读对；同时，本方法的位置响应与完整输出误差更小，缓存字节相同。** 这直接支持本轮算子族拟合目标的额外收益。若实际回答改善来自内容/value 而非位置响应，仍保留方法收益，并据实定位来源。

### 关键归因：算子族拟合目标是否优于普通输出拟合

两边共享相同参数化、A/B/频率/内容/value 可学习参数、初始张量、校准记录、每步文档和位置尺度、优化步数、Adam设置及缓存布局。唯一改变的是以下目标中的第一项：

\[
L_{operator}=\operatorname{NMSE}(\hat s_\Delta,s_\Delta)+\operatorname{NMSE}(\hat V,V),
\qquad
L_{KD}=\operatorname{NMSE}(\hat o_\Delta,o_\Delta)+\operatorname{NMSE}(\hat V,V).
\]

`o` 是整行因果 softmax 后的 value 聚合。两边看到相同的真实位置和伸展位置，也都可以学习频率；因此差异不能归给额外距离曝光、只让本方法训练或只让本方法调频。相同步数提供相同优化次数，实际梯度成本与运行时间另行记录。

这对比较明确归因于**拟合目标**。共享的算子参数化属于共同方法基础，不能把双方共有的结构写成这一对比较隔离出的因素。已有 `--freeze-frequency` 则回答另一个精确问题：在同样拟合下，允许旋转频率更新是否有额外作用；仅在需要定位旋转更新时执行。

| 配对结果 | 获得的结论与下一步 |
|---|---|
| 本方法的留出位置响应、输出、答案一起改善 | 算子族目标带来超出普通输出拟合的有效增益；沿同一依赖任务保留并复核 |
| 本方法位置响应更好，但输出蒸馏的答案更好 | 当前分数拟合的取舍没有转化为更好的决策；使用输出路径的证据修改求解目标 |
| 两者能力相当，且都优于共同初始化 | 紧凑算子参数化可以有效恢复能力；本次没有分出两种拟合目标的优劣，选择成本更低或更稳定者 |
| 输出蒸馏在位置响应与答案上更好 | 保留其有效参数与结果，把它作为本方法更好的求解方式；当前分数目标不保留优越性结论 |
| 两者都差 | 检查共享表示、初始化或预算中哪个损失占主导；不把共同失败归给某个独有目标 |

`--init-from` 加载相同初始张量；每层记录初态哈希、真实输入/距离轨迹哈希和更新数。`compare-objectives` 检查配对条件，输出留出逐文档差值和同一答案的 NLL 差值。它不会用训练 loss 的数值比较两种不同目标，也不会把未优化初始化接受为这一归因对照。

机制读数应优先对应这个依赖场景的主query与有关记录。当前自然文本捕获用于拟合和留出泛化；不要把任意早期query的全体均值写成“已经解释了该回答为什么错”。`study.capture`、`load_record`、`responses` 提供逐记录/逐query张量接口，能够针对同一个实际问题查看计算；不需要另训一组模型。

### 唯一备选：同一问题的正确答案条件 NLL

若两者都答对，或少量0/1答案暂时分不出差异，使用**同一问题、同一正确答案**的条件 NLL：

\[
\mathrm{NLL}_{answer}=-\frac1{|y|}\log p(y\mid x),
\qquad
\Delta\mathrm{NLL}=\mathrm{NLL}_{method}-\mathrm{NLL}_{KD}.
\]

负差值直接表示本方法对正确答案分配了更高的序列概率。这是有明确含义的效果读数，不是无关代理；它比少量二元命中更容易看见模型决策的变化。当前 `generate --expected-answer` 已输出该值、严格 exact-match 和原始回答。

备选用于主测量**没有区分度**时，不用于把清楚的负结果换口径变成正结果。两者都答错而本方法答案概率提高，记为“正确答案概率改善，尚未恢复生成”；这可以帮助决定下一步，但不写成已经完成答案修复。

答案内容与格式分别判断。若只是多一个 `Answer:` 前缀，内容仍正确，就记录格式差异；当前 `answer_exact` 是严格字符串读数，原始输出会同时保留。研究关联检索时，不让纯格式差异代替内容判断。

### Core 失败后的处理

| 实际结果 | 判断 | 下一步 |
|---|---|---|
| 位置响应与真实回答一起改善 | 核心链条成立，当前方法有继续价值 | 保留当前配置，用同一种任务确认这个信号 |
| 位置响应改善，但静态内容/value或回答变差 | 当前联合目标/表示取舍没有形成有用的整体修复 | 停止把这次局部改善称为方法成功；下一候选针对损失的内容或输出路径 |
| 回答改善，位置响应没有改善 | 方法有实际收益，收益机制与原解释不同 | 保留有效方法，把解释转向内容/value或其他有证据的因素 |
| 正常求解后，位置响应和回答都没有改善 | 当前初始化、目标与求解配方没有显示价值 | 记录后转下一候选，不先加模型、任务或种子 |
| 两者在主读数上相同 | 当前读数未分开两者 | 用同一问题的答案 NLL，或复核同类实例；不自动称成功或失败 |
| 数值、接口或输入出错 | 实验没有测到目标效果 | 修复这一具体错误，再解释方法；不会据报错否定数学路线 |

## 3. 什么指标直接代表旋转/位置响应改善

对同一份真实 Q/K，分别计算原模型分数 `s_Δ` 与压缩分数 `ŝ_Δ`。保持内容、参数和因果顺序不变，只改变相对位置。定义：

\[
E_{pos}(\Delta)=\mathbb E\left[
\big((\hat s_\Delta-\hat s_0)-(s_\Delta-s_0)\big)^2\right].
\]

**它直接测量：距离变化引起的分数变化是否被保留。** 静态内容分支在差分中消去；value 重建变好也不会单独使这个指标变好。A/B 与频率都可以影响它，因此它衡量的是当前内容表示上的相对位置计算，而非只评频率数字。

当前 `diagnose` 已输出 `position_response_mse`、`relative_position_response_mse`，以及距离分桶内的同类误差。代码检查覆盖了“只改变静态内容分支，位置响应误差保持不变”。

同时保留静态内容误差和完整分数误差：

\[
E_0=\mathbb E(\hat s_0-s_0)^2,\qquad
E_{total}(\Delta)=\mathbb E(\hat s_\Delta-s_\Delta)^2.
\]

两者与 `E_pos` 一起告诉我们：是修复了距离响应、损坏了内容，还是确实改善了完整计算。由于存在交叉项，不能简单用 `E_total−E_0` 代替 `E_pos`。

训练 loss 则负责**求解方法**。当前默认 loss 是 score NMSE + value NMSE，它有用，因为需要同时保存两条计算路径；验证位置主张时读取 `E_pos`，不拿混合 loss 冒充位置响应结果。训练与验证各自承担清楚的作用。

## 4. 每项实验的实际用途与价值

### E1：拟合能否得到有用的紧凑表示

**入口：** `fit`；代码 `study.py::fit_layer`。

观察分开的 score/value/output 误差和优化过程。价值是检验当前求解能否找到有用表示，并得到可放回模型的参数。

成功意味着这份求解确实改善了目标计算；留出内容上的诊断再判断它是否只记住校准。失败可以淘汰**当前初始化、预算与求解配方**。单次没找到解不必被包装成整个表示类不可能成功，也不需要为了证明“不可能”无限调参。

首末训练日志可能来自不同文档和距离；比较效果时使用同一份诊断输入。默认总 loss 下降若主要来自value，就是value的改进，按它实际带来的价值记录。

### E2：位置响应与静态内容的分解

**入口：** `diagnose`；主读数是上节的 `E_pos`、`E_0` 和完整分数误差。

价值是定位改进/失败发生在哪里，并直接检验“保留相对位置计算”的主张。真实距离分桶提供现象；要隔离距离作用，可在同一捕获记录上只改一次 `position_scale`，无需重新训练。

明确的排除例子：A/B和内容分支固定时，任何频率都不能改变Δ=0的计算。因此静态误差大时，**纯调频不能修复这部分损失**。如果静态很好但远距离响应差，则当前旋转表示仍不适合这些依赖距离。

`generator_diagnostics` 的 leakage/生成元残差用于解释这种现象。`calibration_prediction` 字段表示在校准内容上测得的误差估计，和留出观测对读；本轮不靠字段名称宣称已经完成独立预测规律的研究。

### E3：位置/分数变化有没有保留真实 attention 输出

**入口：** 同一次 `diagnose`，复用已有 Q/K/V。

读取完整 softmax 输出误差、attention KL、value误差和关键竞争。价值是把算子变化与实际值聚合连接起来，避免只优化一个与输出脱节的量。

若 `E_pos` 和完整分数改善、输出却差，就排除“当前分数目标已经足以保留响应”的判断；下一候选应针对关键竞争或value路径。输出差异可以按

\[
\hat o-o=(\hat p-p)V+\hat p(\hat V-V)
\]

定位。现有张量接口支持这一针对性检查；无需扩展任务集。

全行分数的公共平移不改变softmax；相似values之间的权重交换也可能保持输出。此时较大的分数误差或KL本身不是失败。我们按所需计算有没有被保留来判断，而不是要求所有指标都变好。

### E4：完整模型的真实文本预测

**入口：** `evaluate`。

将全部层替换为当前方法，计算真实文档末尾目标token的NLL。价值是检查局部近似组合到整网后是否仍有效，并测量真实预测质量。

在同输入、同预算的简单对照下，配对NLL降低就是有效的模型质量证据。如果局部响应好而整网NLL明确变差，便排除**当前逐层局部拟合可直接完成整网替换**的实施路线，重点看跨层累积、内容迁移或目标取舍。

只拿本方法一个NLL数值无法比较高低，所以它不作为孤立的“成功实验”。在本轮Core中，已知答案条件NLL更贴近依赖任务；自然文本NLL用于判断整体语言建模代价，不自动加成第二套大任务评测。

### E5：单种真实依赖的生成

**入口：** `generate --expected-answer`，是Core的实际效果端。

同样上下文与问题，错误变正确就是答案恢复的直接证据。一个清楚的配对实例可以证明这种修复确实存在，足以支持方法定位继续推进；需要扩大适用范围时才在同类实例上复核。

若当前候选在对照能完成的依赖上持续出错，当前配置的保留能力不足。若只是格式差异，单独记录格式；若原模型本来就不会该依赖，则这条样例没有回答压缩保留问题。

### E6：缓存与运行成本

**入口：** `profile`。

读取实际cache storage字节、prefill/decode时间和峰值。价值是确认“同KV预算”真正成立，并判断方法收益所需的计算代价。

缓存确实缩小而decode没有更快，是内存收益与算子开销的取舍；排除的是“缓存减半必然提速一倍”的想法，而不是压缩价值本身。若storage未按布局缩小，就先修复当前缓存实现；若参考后端慢，也准确记录这个实现的成本。

当前profile使用随机token测成本；质量由E4/E5回答。之前按多组完整benchmark估计的数天/数周不用于本轮方法定位。

## 5. 单个针对性实验何时有必要

以下选项用于已有结果指向具体原因时，选择其中一项；不默认全部运行。

| 单次设置 | 回答的问题 | 失败能排除的路线 |
|---|---|---|
| `--freeze-projections --freeze-content` | 固定内容表示，仅改变旋转是否能修复远距响应 | 当前固定表示上的纯频率修复；Δ=0损失本来就不会改变 |
| `--freeze-frequency` | 固定旋转，重配内容是否已经足够 | 当前固定频率集合和求解下的投影修复 |
| `--freeze-content` | 保持共同内容/value路径，仅修改A/B/旋转是否有效 | 只修旋转分支足以挽回当前全部损失；固定value重建误差不会随之改变 |
| `--max-position-scale 1` | 当前方法是否需要额外的距离回放 | 若真实位置拟合好而远距失配，说明这份位置覆盖不足 |

失败记录写清实际改变了什么即可。一个可用方法不必先通过全部消融；一个无用候选也不必先耗完所有选项才允许停止。

## 6. 可直接执行的Core入口

准备与捕获命令见 [README](README.md)。初始化只计算和保存一次，两条拟合读取完全相同的张量：

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run fit \
  --capture work/capture --out work/initialization \
  --content-rank 192 --rotary-dim 64 --initialize-only --device cuda
```

先执行当前方法；归因对照是下一条单配置命令。两者各500次更新，除 score/output 目标权重之外参数一致：

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run fit \
  --capture work/capture --init-from work/initialization --out work/operator \
  --content-rank 192 --rotary-dim 64 --steps 500 --max-position-scale 8 \
  --score-weight 1 --value-weight 1 --output-weight 0 --seed 42 --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run fit \
  --capture work/capture --init-from work/initialization --out work/output_kd \
  --content-rank 192 --rotary-dim 64 --steps 500 --max-position-scale 8 \
  --score-weight 0 --value-weight 1 --output-weight 1 --seed 42 --device cuda
```

对两份checkpoint使用同一个prompt与答案：

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run generate \
  --model "$OPERATOR_MODEL" --factors work/operator \
  --prompt work/core/prompt.txt --chat --expected-answer f83a9144 --max-new-tokens 32 \
  --out work/core_method.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run generate \
  --model "$OPERATOR_MODEL" --factors work/output_kd \
  --prompt work/core/prompt.txt --chat --expected-answer f83a9144 --max-new-tokens 32 \
  --out work/core_output_kd.json --device cuda
```

上面使用已经冻结的真实实验输入：412条记录、8189个chat-template tokens、已知答案 `f83a9144`，不是前文的说明性 `blue` 示例。输入没有按模型回答挑选。已有输出保留原始答案、严格exact-match、条件答案NLL以及模型、token输入和生成配置。下面以最后一层、同一留出记录的相同位置为例复用诊断；需要解释某个答案时，诊断对应其实际输入与query。

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run diagnose \
  --capture work/capture --factors work/operator --layer 27 --position-scale 1 \
  --out work/diagnostic_method.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run diagnose \
  --capture work/capture --factors work/output_kd --layer 27 --position-scale 1 \
  --out work/diagnostic_output_kd.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run compare-objectives \
  --method work/operator --control work/output_kd \
  --method-diagnostic work/diagnostic_method.json --control-diagnostic work/diagnostic_output_kd.json \
  --method-generation work/core_method.json --control-generation work/core_output_kd.json \
  --out work/objective_attribution.json
```

比较器输出 method−control 的位置响应、完整输出、静态内容、value 和答案NLL差值，保留逐文档差值与原始回答。初态、数据/位置轨迹、更新次数或测试输入不匹配时，指出具体不一致，不生成有效归因结论。

## 7. 一次运行后的简短结论

- **观察：** 本方法与所选简单对照，在同一条件下具体改变了什么。
- **价值：** 支持位置响应、完整输出、实际答案或内存成本中的哪一项。
- **排除：** 这次结果否定的具体实施路线；不把“未达到完整论文要求”当作方法失败。
- **决定：** 保留并做一个有信息的复核，或结束当前候选转下一方法。

`prepare`、`capture` 和 `report` 是支撑步骤；小模型测试保证代码可运行。研究结论来自以上直接测量。已回看的历史结果另记在 [历史结论简记](RECENT_RESULTS_NOTE.md)，不因此恢复旧实验队列。
