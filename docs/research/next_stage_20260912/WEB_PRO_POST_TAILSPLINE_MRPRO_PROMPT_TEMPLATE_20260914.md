# 给Web Pro：TailSpline–MrRoPE终审后的唯一下一步

> 使用方法：这是Web端自包含提示词。`FINAL_RESULT_BLOCK`将在当前两臂实验完成后替换为
> 真实报告；替换前不要发送。GitHub连接只用于核查明确给出的路径，不能替代本文背景。

---

你现在扮演一名严格的ICLR领域审稿人、长上下文位置编码研究者和实验设计负责人。
请根据下面完整提供的研究背景与最新揭盲结果，决定Hybrid-RoPE项目**唯一最值得做的下一步**。

这不是开放式头脑风暴。你的任务是选择一个能够最大幅度提高论文可信度或贡献强度的动作，
并关闭其余低收益路线。不要再提出一组新频率曲线或宽参数搜索。

## 1. 信息边界

本提示词中的实验状态、结果数字和证据边界是当前权威快照。Web端虽然连接了GitHub，
GitHub读取是按查询检索，可能看不到未提交工作、远端GPU raw、大文件和没有被搜索命中的文档。
因此：

1. 先仅依据本文完成科学判断；
2. GitHub只用于核查本文明确列出的文件或实现，不得因搜索不到就判定证据不存在；
3. 如需核查外部工作，只使用原论文、官方实现或会议标准等一手来源；
4. 把theorem、CPU代数、proxy、开发结果、独立确认和完整benchmark严格分开；
5. 不得把历史近似配置的分数转移给精确TailSpline。

## 2. 论文真正的研究问题

Hybrid-RoPE研究的是：在RoPE高频与低频端点固定时，中间频率的allocation是否具有独立、
可利用的因果效应；这种基函数配置效应，和checkpoint训练后形成的频率—坐标使用方式，
能否通过干预分离。

论文不应只宣称“发现了一个z参数”或“非几何频率有用”。更有防御力的主线是：

1. 固定实际频率端点，只改变内部allocation，识别allocation的独立作用；
2. 通过范围重定向、权重×表交叉安装、固定频谱槽位置换，区分频率配置与learned compatibility；
3. 展示训练与成熟checkpoint部署中可利用的正向结果，同时保留反例和迁移失败；
4. 若精确TailSpline取得自己的统一主结果，再将其升级为论文的方法贡献。

即使TailSpline失败，上述科学识别主线仍然成立；不得把论文降格成“什么都没找到”，也不得
为了补偿失败而发明无依据的不可行性定理。

## 3. 已成立且必须保留的证据

### 3.1 固定支持的allocation因果效应

- 151.9M模型、固定实际端点、只改变30个内部频率、三训练种子；
- Cosh−Geo平均NLL差在2/4/8倍外推为`-0.281/-0.176/-0.146`；
- 三种子方向一致，但2倍的种子区间跨零；
- Native窗口内平均NLL增加约`0.0262`，因此不是无代价统治。

该证据支持allocation有独立作用，不支持任意几何proxy都能预测任务收益。

### 3.2 成熟checkpoint的冻结干预

- OLMo同端点、同gain、同权重比较中，uniform/ramp/derived约为
  `0.56/61.04/60.47%`；
- 内部形状对完整生成具有巨大作用；
- derived没有胜过简单ramp，因此复杂代理构造没有得到独特优越性。

### 3.3 learned compatibility

- 固定频谱置换后，OLMo NLL约`3.104→6.865`；
- Qwen 64K任务macro约`0.70→0`；
- 同时置换Q/K坐标和频率时，kernel存在精确等变恒等式。

这支持checkpoint使用特定频率—坐标匹配，但不等于已经识别最佳allocation规律。

### 3.4 已有构造性收益

- BM相对MrRoPE-Pro在五任务、631个超Native自然QA输入上，任务等权F1约
  `21.62%→25.44%`，差值区间约`[1.32,6.29]`百分点；
- 早期静态配置在untouched OLMo完整RULER-13的4/8/16K约为
  `71.40/66.71/49.86%`，明显优于对应YaRN配置；
- 这些分别属于不同构造和协议，不能拼接成TailSpline自己的成绩。

### 3.5 必须保留的反例

- fixed-u跨倍率迁移相对fixed-m的AUC约下降`5.15`百分点，区间
  `[-9.73,-0.93]`；
- Qwen mix075的开发累计优势在独立追加块中明显缩小且区间跨零；
- Llama S8中不同方法总分相近却发生tracking、FWE、multikey的大幅任务交换；
- OOD/SEP等几何proxy出现多次“几何支配但任务反转”；
- 因此漂亮的几何目标和局部代理不能替代真实PPL、NIAH和完整RULER。

## 4. 当前唯一候选：精确TailSpline

对canonical MrRoPE/YaRN 32圈/1圈band，令`n=d_h-d_l`、`q=0,...,n`：

\[
m_T(q)=\frac{q(3n^2+3n+1-q^2)}{n(n+1)(2n+1)},
\qquad \nu_k=\omega_kS^{-m_k}.
\]

增量为：

\[
\epsilon_q=\frac{3(n+q)(n-q+1)}{n(n+1)(2n+1)}.
\]

它是下列one-sided roughness的唯一解：

\[
J_{tail}=\sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2,
\qquad \sum_q\epsilon_q=1.
\]

有限网格上：

\[
m_T=(1-w_n)m_{BM}+w_nm_{front},
\qquad w_n=\frac{3n}{2(2n+1)}.
\]

Llama的`n=17`时`w_n=0.728571`。历史mix075只是它的近似开发先验，且历史band/gain
并非完全相同。

### 4.1 已确认的数学边界

若把高频入口和低频tail两端都惩罚：

\[
J_{sym}=\epsilon_1^2+\sum_q(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2,
\]

唯一解恰好是BM。因此TailSpline不能描述成“全局最平滑”。其真实任务假说是：

> 成熟长上下文模型更偏好尽早重新分配中频尺度，同时平滑接入fully-scaled低频tail，
> 即early transport + smooth tail landing。

CPU只证明给定边界目标后的唯一性，不能选择one-sided或symmetric，也不证明模型收益。

### 4.2 与MrPro的未分离因素

TailSpline和MrPro在当前S4比较中虽然band、gain、端点相同，但同时改变：

- allocation的前移/后移位置；
- 中频总减速量`sum(m)`；
- 更高阶曲线形状。

所以TailSpline即使获胜，也先构成经验方法结果，不能立即宣布one-sided tail roughness是收益原因。

## 5. 当前冻结终审合同

- 模型：`Meta-Llama-3-8B-Instruct`，Native窗口8192，冻结权重；
- 倍率：`S=4`，评测长度8K/16K/32K；
- band：canonical `[18,35]`；
- gain：两臂相同，`1+0.1 ln4 = 1.138629436111989`；
- 当前只比较精确TailSpline与exact MrRoPE-Pro；
- Full-13 RULER：13任务×3长度×10行，共390 prompts/臂；
- 六个单答案NIAH任务每个长度覆盖10/30/50/70/90%深度，各2行；
- PPL：46篇冻结自然长文，包括32篇ProofPile arXiv test和14篇PG19 test，
  每篇使用嵌套8/16/32K Llama token前缀，共138文档—长度格/臂；
- greedy decoder、prompt顺序、scorer、precision和静态表部署完全匹配；
- MrRoPE raw与全部测试资产进入专用复用库，后续身份匹配时禁止重跑baseline。

三个预注册family endpoint分别判断，不合成临时总分：

1. PPL log-length AUC，越低越好；
2. Passkey/NIAH task-equal log-length AUC，越高越好；
3. Full-13 task-equal log-length AUC，越高越好。

当前方法门：相对MrPro至少两个family方向获胜，才值得进入进一步确认。单任务或单长度
反转不自动否决总体结果，但必须完整报告。

## 6. FINAL_RESULT_BLOCK

以下区块必须由完成后的真实报告替换，不能根据运行中的partial推断：

```text
实验身份核验：<PASS/FAIL及失败原因>
TailSpline Full-13 AUC：<value>
MrPro Full-13 AUC：<value>
Delta Full-13 AUC：<value, 95% paired interval>

TailSpline NIAH/Passkey AUC：<value>
MrPro NIAH/Passkey AUC：<value>
Delta NIAH/Passkey AUC：<value>

TailSpline PPL AUC：<value>
MrPro PPL AUC：<value>
Delta PPL AUC：<value, 95% paired interval; negative is better>

8K/16K/32K分长度：<exact values and deltas>
任务族与最大反转：<retrieval/tracking/aggregation/QA>
EOS/cap/空输出：<exact values>
方法门：<3类赢几类，是否达到2/3>
MrRoPE baseline registry：<ready=true/false>
```

## 7. 两个已经冻结、但尚未做GPU验证的机制对照

### 7.1 TailSpline同总减速量反事实C

令`U_q=q/n`、`P_q=q(q+1)/(n(n+1))`、`F_q=2U_q-P_q`，并使用TailSpline的
`w_n=3n/[2(2n+1)]`：

\[
C_q=(1-w_n)U_q+w_nF_q.
\]

`C`与TailSpline共享band、端点、gain、`sum(m)`及增量重心，只改变更细的shape；数学上
TailSpline的`J_tail`更小，但任务排序未知。若TailSpline先赢MrPro，追加一个C臂即可判断
收益是否超出总减速量解释；TailSpline与MrPro现有raw均无需重跑。

### 7.2 YaRN–MrPro等剂量单交叉后移

对Llama canonical band的`n=17`，选择唯一倍率：

\[
S^\star=7.51324282212058,
\qquad \sum_qm_Y(q;S^\star)=\sum_qm_P(q)=19/3.
\]

目标窗口约61,548 tokens，共享gain约`1.20166671731257`。MrPro在`q=1,...,9`的累计
指数小于YaRN，在`q=10,...,16`大于YaRN，只有一次交叉。该对照在固定总log位移后检验
back-loading原则，但与“TailSpline能否成为主方法”是不同问题，作者已要求YaRN后置。

## 8. 你必须做出的唯一决策

根据`FINAL_RESULT_BLOCK`，从下面四类动作中只选择一个最优先动作。可以选择列表外动作，
但必须证明它在信息增益、论文价值和GPU成本上严格优于以下选项。

### A. 同总减速量反事实C

适用情形：TailSpline已形成有意义的方法优势，但one-sided机制仍被总剂量解释混杂。
只新增C一臂，复用当前TailSpline、MrPro和测试资产。

### B. YaRN–MrPro等剂量S*实验

适用情形：论文更需要识别MrRoPE的back-loading原则，且该理论问题比TailSpline归因更关键。
只跑YaRN与MrPro；但需要新增48K/60K冻结输入，作者当前将YaRN后置。

### C. TailSpline独立确认

适用情形：当前方法优势已经存在，但10行/格的不确定性、开发历史或单checkpoint限制，
比机制归因更可能导致审稿拒绝。应复用当前MrRoPE baseline，只新增独立TailSpline数据块
或一个独立checkpoint，不能重跑已匹配baseline。

### D. 停止TailSpline新增GPU并重组论文

适用情形：TailSpline没有形成方法优势，或结果只来自不可接受的任务/长度交换。论文回到
allocation因果效应、learned compatibility和已有构造性收益主线；不根据失败结果再造曲线。

## 9. 回答要求

请按以下结构回答，控制在1500中文词以内：

1. **唯一选择**：只给A/B/C/D之一，或一个被严格论证为更优的替代动作；
2. **对最新结果的准确判决**：区分点估计、区间、任务异质性和证据等级；
3. **最强竞争解释**：指出当前结果仍可能由什么造成；
4. **最小下一步合同**：模型、臂、可复用baseline、新增数据、长度、主要指标与估计成本；
5. **可证伪预测与停止条件**：结果如何时支持、削弱或关闭命题；
6. **论文影响**：下一步不同结果分别允许写什么、禁止写什么；
7. **被拒绝的路线**：简述为什么其余选项现在信息增益更低。

硬约束：

- 不提出新的frequency curve家族；
- 不调TailSpline系数、band、gain、depth或边界惩罚；
- 不训练模型；
- 不把CPU泛函、PPL或几何proxy冒充完整任务证据；
- 不重复运行身份匹配的MrRoPE baseline；
- 不按已揭盲的失败任务重新选择主任务；
- 给出一个能够执行并结束争论的动作，而不是多个并行方向。

