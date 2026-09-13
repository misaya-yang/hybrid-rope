# 区间最优固定 RoPE 表：研究总自查、充分理论与判决路径

更新：2026-09-13。本文在新服务器开机前，结合仓库当前结果 owner、成功与失败记录、
三接口计划和新实验实现，回答三个问题：为什么 YaRN→MrRoPE-Pro 能产生显著变化；
为什么后续长期只有部分改善；当前实验能否真正把“一张固定表最大化 `[L,SL]` 质量”
推进为审稿人可判断的方法结果。

本文严格区分：**Observed** 为已有结果 owner 记录的事实；**Supported inference** 为
这些事实共同支持的解释；**New hypothesis** 为尚待实验的构造。它不把代码完成写成
实验完成，也不等待一个能解释所有模型、任务和长度的完美理论。

## 1. 最终判断

**当前最短的成功路径不是继续发明曲线，而是确认已经冻结的 Llama Solver
`[14,32]`。** 它在三任务×五长度×每格4行的开发屏上，相对 BM/MrPro 的
log-length AUC 为 `+5.58/+5.15pp`，worst-length 为 `+12.50/+16.67pp`。
这是目前唯一同时明显改善区间均值和最弱点的强开发候选。现阶段缺的不是“再想一个
名字”，而是同 runner 的 Core-6 mini、Native 保留、BM/MrPro 主比较和 static YaRN
诊断；随后才是 medium/high 与自然任务确认。

若该候选在 Core-6 mini 仍保持有意义的 AUC/worst/Native 优势，论文已经取得实质
前进：方法可以定义为**在校准长度分布上求解的固定 exponent allocation**，理论负责
解释设计对象和约束，不需要先证明闭式全局最优。若它失效，则说明旧优势来自三项检索
或4行离散选择，当前方法尚未解决主问题；但 band、depth、shape 的匹配干预仍可定位
失败来自任务广度、端点预算还是 profile 交互。

今晚已经准备的流水线能推进主问题，但贡献不等价：Llama paper-confirm 是直接判决；
OLMo/Qwen/Llama depth 与 shape 是修正下一张表的机制证据；checkpoint replay 只有先
通过真实 runtime parity 和已有大效应排序，才有资格产生候选。low 机制结果本身不能
替代论文级确认。

## 2. 已经稳固的事实

### 2.1 内部 exponent allocation 不是无意义自由度

**Observed**：固定 checkpoint、support、端点和 gain 后改变内部 allocation，会在
成熟模型上产生大幅、结构化且可重复的任务变化。BM、MrPro、C42/C42V24、Solver 的
排序随长度和任务变化；PPL 常不跟随完整输出任务分数。

**Supported inference**：RoPE 表不是仅由最大波长决定的缩放常数。预训练权重实际使用
不同频率槽，`z`/内部 allocation 是部署时可被独立干预、会改变能力的变量。这是当前
论文最稳固的基础发现。

### 2.2 没有一张已有解析表普遍最优

**Observed**：

- OLMo Core-6 mini 上，BM/C42/MrPro 的三点 AUC 为 `68.20/69.12/25.43`；C42
  相对 BM 的优势仅 `+0.92pp` 且任务有反转。
- Llama S=8 小屏上，BM 偏强于1×/2×，MrPro 在旧四点口径的4×/8×略强；自然任务
  又不复现同一排序。
- Qwen S=2 的 `[22,39]` 当前64K点估计比 MrPro 高 `4.69pp`，但区间跨零，且该结果
  已参与理论形成。
- Gamma3、BM/MrPro exponent midpoint、Winding-Matched、严格 OOD-max+SEP-min
  路线都出现真实任务反转；降低 NLL 也可伴随检索归零或下游不升。

**Supported inference**：单个端点、总剂量、质心、winding、PPL、对角 Fisher 或两个
几何标量均不足以排序实际任务。已有反例否定的是这些“充分性”，不是否定整个固定表
空间存在更优解。

### 2.3 band 不是 YaRN 预先规定的模型常数

**Observed**：Llama S4 的局部好点 `[16,34]` 原样迁到 OLMo 明显退化；OLMo 当前
basin 为 low=14/high=31--32；Qwen 当前点在 `[22,39]`；Llama S8 中 band 效应又
与 shape 强交互。移动 band 同时改变 `sum(m)`，因此旧屏并非纯 placement 因果实验。

**Supported inference**：band 应是完整 profile 的可学习/可校准 support，而不是
YaRN“低/中/高”的固定定义。几何 winding 可产生候选邻域，checkpoint 和任务决定
该邻域内的实际代价。

## 3. 为什么 YaRN→MrRoPE-Pro 会显著

以下解释只针对同 checkpoint、同 support、同端点和同 gain 的匹配静态表；它不是
“MrPro 永远优于 YaRN”的定理。

令 `t∈[0,1]` 表示从 fast 到 slow 的 transition 位置。YaRN 的线性频率插值可写成

\[
\frac{\omega'_Y}{\omega^N}=1-(1-S^{-1})t,
\qquad
m_Y(t)=-\log_S\left[1-(1-S^{-1})t\right].
\]

MrPro 的离散累计 exponent 为

\[
m_P(q)=\frac{q(q+1)}{N(N+1)}\approx t^2.
\]

两者都保持 fast 端 `m=0`、slow 端 `m=1`，但 MrPro 在 transition 前中部通常给出
更小的 exponent，把更强压缩推迟到慢频侧。它因此保留更多接近 Native 的中高频相位
变化，同时仍由慢频平台承担目标端 `/S`。局部绑定和短距离分辨率主要受前者影响，
远距离 phase exposure 主要依赖后者；这给出 YaRN→MrPro 大幅改善的直接机制解释。

RoPE 在相对距离 `d` 上的槽贡献满足

\[
a(d)=\sum_k\operatorname{Re}(C_k e^{id\omega'_k}),
\qquad
\frac{\partial a}{\partial m_k}
=\log S\,d\omega'_k\operatorname{Im}(C_k e^{id\omega'_k}).
\]

所以看似不大的 exponent 改动会被长距离 `d` 放大，并经 softmax key competition、
层间反馈和自回归阈值变成大幅任务差异。这解释“为什么曲线只改一点，任务能差很多”。

但 `C_k` 随 checkpoint、层、头、内容和任务变化，导数还随相位改符号。OLMo 上 BM
远强于 MrPro、Llama 自然任务与检索排序不同，正是这套解释的边界：**MrPro 的成功
说明延后中频压缩是一个强先验，不说明它对所有长度分布和模型权重都是最优预算。**

## 4. 统一三个接口的更有用坐标

对任一槽定义

\[
r_k=S^{m_k}.
\]

当实际长度为 `r_k L` 时，该槽的目标表相位跨度等于 Native 表在 `L` 处的相位跨度：

\[
r_k L\,\omega_k^N S^{-m_k}=L\omega_k^N.
\]

因此 `r_k` 可解释为该槽的 **Native-equivalent crossing length**。一张固定表的本质
不是选一个“中频段”，而是把有限频率槽的 crossing lengths 分配到 `[1,S]`：

- band 决定哪些槽的 `r_k>1`、哪些槽到达远端；
- transition 决定 crossing lengths 在 log-length 上的密度；
- tail depth 决定最大的 crossing length 是否到达 `S`；
- gain 决定这些旋转槽进入 attention logits 的共同幅度。

这是恒等坐标，不是任务最优定理。它把三接口统一为一个真正与部署问题一致的对象：
**为给定使用长度分布，把不同 Native 频率槽分配到不同的等效覆盖尺度。** YaRN、
MrPro、BM、C42、Solver 都只是不同的 crossing-length distribution。

### 4.1 与大 Native 窗口的关系

随着 Native `L` 增大，实际工作负载通常不会等概率停在极限 `SL`；更多请求落在
Native附近和前几个倍增区。此时 `S` 仍是物理 horizon，但不应独占设计目标。令
`π(r)` 为事先声明的部署长度权重，合理目标是

\[
\max_{m,g}\;\mathbb E_{r\sim\pi,\tau}[Q_{\tau}(r;m,g)]
\]

同时分别报告 Native、endpoint、worst-length 和任务族 regret。`π` 取 log-uniform
时每次倍增等权；若产品分布已知可换成真实分布，但不能看测试成绩后调权重。

在一个仅作设计解释的独立槽近似中，可把 crossing length 看成 `π` 的加权分位点：

\[
m_k=\frac{\log r_k}{\log S},
\qquad r_k\approx F_{\pi}^{-1}(u_k),
\]

其中 `u_k` 由频率顺序和 checkpoint 对该槽的敏感度决定。真实 attention 有交叉项和
周期性，所以最终仍需联合校准；但这个坐标比“猜 Cosh/Beta 曲线”更直接地表达区间
目标，并自然解释为何 endpoint-only 表会在内部长度产生 regret。

## 5. 为什么我们此前推进艰难

### 5.1 目标清楚，不等于观测足以优化目标

我们定义了区间 AUC，但多数早期屏只有3个检索任务×每格4行，单个答案就改变8.33pp；
一些结论来自单端点、PPL或另一 benchmark。目标是宽任务区间质量，观测却常是高噪声
代理，优化自然出现赢家反复。

### 5.2 多个变量曾被一个名字捆在一起

band remap 同时改变位置、宽度和总压缩剂量；Solver 与 C42 比较有时同时改变 shape
和 gain；不同 S 又改变整个相位暴露。把这种整表差异归因给一个变量，会产生看似理论
命中、换模型即反转的现象。

### 5.3 任务能力是阈值系统，不是表空间上的光滑标量

Attention phase 是周期的，完整输出还需要检索、绑定、聚合、QA、EOS 和自回归稳定性
共同成立。频率表中的连续小步不保证任务分数连续；BM/MrPro midpoint 和 gamma3 的
64K坠落已经否定“两个强表之间平滑插值仍强”。

### 5.4 我们试图先得到普适理论，再确认手里的正结果

严格 OOD/SEP、winding、Fisher、replay 都可能解释局部，但尚未预测真实宽任务排序。
与此同时，Llama Solver 已经给出强开发信号，却长期没有先完成 Core-6 同口径确认。
这使很多工作增加了认识，却没有增加可写的主结果。

### 5.5 证据身份与测评口径曾漂移

官方 contains、完整 exact+EOS、PPL、自然QA F1、不同任务集合和不同样本量不能互相
替代；旧 Llama 64K parity 还出现 raw output/multivalue 差异。若不先固定同 runner、
同 prompt、同 table+gain、同 decoder/precision，更多运行只会增加不可比较数字。

## 6. 足够成文、无需完美理论的方法方向

当前最可信的方法不是新的万能闭式，而是把已有 Solver 正式化为
**Range-Calibrated Static Allocation**：

1. 用单调 exponent/crossing-length profile 和固定 gain 表示一张可安装表；保存完整
   64槽，不预设 YaRN band 必须正确。
2. 在与最终测试隔离的 calibration 长度和任务上，直接最小化 task-equal 多长度损失、
   source-counterfactual 绑定损失和 Native 保留项；每个提议用真实前向回验。
3. 表和 gain 在测试前冻结；同一表用于全部层和全部长度。
4. 在 fresh mini→medium→high 上与 BM/MrPro 比较；static YaRN 单列为零训练诊断。
5. 以 AUC、worst-length、Native、endpoint 和任务族共同呈现，不要求每个 cell 全胜，
   但不能隐藏明确深坑。

这不是从 benchmark test 直接调64个数。开发/确认必须分开，优化自由度应由单调增量、
少量 spline knots 或 trust region 控制。当前 Solver 恰好已经在 OLMo calibration 上
冻结，然后转到 Llama；新的 Core-6 Llama mini 因而是比同模型反复调 band 更有价值的
确认。若它成功，可以先形成“校准式固定表方法”；跨 checkpoint 的纯解析初始化是后续
增强，不是论文成立的前置条件。

### 新假说：Range-quantile allocation

可将后续统一算法写成：在单调可安装约束下，用少量 knots 表示
`log r_k=m_k log S`，由声明的长度分布 `π` 给初始分位点，再用 checkpoint calibration
调整 knots 与 gain。其可证伪预测是：相对 endpoint-oriented MrPro，它应减少内部
长度 regret；相对 BM，它应保留或改善目标端；收益应主要表现为 AUC/worst 提升，而
非要求每个任务逐点支配。

这条假说目前尚未由新 GPU 结果验证。它的价值是把大 Native 窗口、区间效用和三接口
放进同一可计算对象，而不是再猜一个外观更漂亮的 `m(x)`。

## 7. 当前实验逐项是否推进主问题

| 实验 | 直接问题 | 成功说明 | 失败说明 | 对最终结果的作用 |
|---|---|---|---|---|
| Llama Solver/BM/MrPro Core-6 mini | 冻结强候选能否扩展到宽任务三长度 | 已有方法候选可进入medium/high | 旧信号是窄任务/小样本选择 | **最高，直接判决方法** |
| Llama 8K Solver vs Native | Native 是否被平均AUC换掉 | 满足短窗保持或量化可接受代价 | 必须修正profile/gain或收窄claim | **主约束** |
| static YaRN同row | 零训练静态YaRN诊断 | 可报告相对静态YaRN差值 | 不影响BM/MrPro主envelope | 辅助；不冒充YaRN SFT |
| OLMo C42/BM同band depth×shape | depth作用是否依赖transition | 给出可迁移的交互方向 | soft depth或简单shape分解不足 | 机制修正，不单独成方法 |
| Qwen C42 mini+depth | 当前band信号是否扩展、S2是否同向 | 支持checkpoint/S条件规律 | 明确模型依赖；Qwen仍为开发证据 | 跨模型边界 |
| Llama C42 full/Dlog depth low | `/S`完整终值是否牺牲区间 | 给出当前候选的可改进方向 | full depth仍是强边界或全局缩放错对象 | 下一候选生成 |
| checkpoint capture/replay R0/R1 | 无标签Q/K代理是否有预测资格 | 可冻结一张solver候选 | 停用该proxy，不否定表空间 | 有条件的方法自动化 |

当前流水线刻意不设 score 自动门禁。首批结果出来后由研究者判断是优先扩展 Solver 到
medium/high，还是先利用三接口结果修正一张新表；不会由一次low跨零自动取消实验。

## 8. 还缺什么才算真正解决

1. **Core-6 mini 同口径确认**：这是开机后的第一优先级。
2. **冻结候选的 medium/high fresh rows**：最终至少需覆盖检索、tracking、aggregation、
   QA与中间长度；选择用过的mini和最终新增block分开报告。
3. **自然长上下文分面**：不与RULER硬合成总分，用来检查合成任务收益是否伴随严重
   自然能力退化。
4. **方法定义可复现**：完整表、gain、calibration split、求解规则和所有负对照入账。
5. **一个真正未参与构造的验证对象**：可以是新任务rows或新checkpoint；Qwen1.5
   不能再冒充新理论的prospective留出。

若 Solver mini 成功但 high 未完成，结论是“强候选通过首轮宽任务确认”；若 high 也
保持优势，则可以主张在声明 benchmark 身份上实现更好的 `[L,SL]` 固定表。只有多模型
规则也前瞻成功，才升级为跨checkpoint统一构造；这不是当前论文主结果的必要条件。

## 9. 开机后的研究纪律

- 第一批只运行已冻结合同，不根据中途分数改表。
- BM/MrPro在相同benchmark身份只运行一次；static YaRN不进入主baseline envelope。
- 64K单模型驻留、顺序多表；表hash必须连同gain、decoder和precision绑定。
- 每15分钟检查进度、显存、GPU利用率、输出prefix、磁盘和status；异常先判工程身份，
  不把OOM或未验收partial rows写成方法失败。
- 第一阶段完成立即人工审读逐任务/逐长度/EOS/cap和worst bootstrap；若已有论文级
  正信号，优先扩确认，不再用整晚追求更漂亮理论。

**一句话结论：** YaRN→MrPro 的显著改善来自更合理地安排各频率槽何时承担扩展，
但 checkpoint 与任务决定具体最优分配；我们应把 `r_k=S^{m_k}` 的 crossing-length
distribution 作为统一设计对象，用隔离校准求一张固定表，并以同口径宽任务区间曲线
确认。当前 Llama Solver 是最短的可验证突破口，今晚的第一阶段能够直接决定它是否
成为论文方法，而后三接口实验负责在失败或局部不足时生成下一张更好的表。
