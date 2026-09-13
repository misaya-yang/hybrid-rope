# 固定 RoPE 表三接口理论审计与验证合同

更新：2026-09-13。状态：**采纳为核心可证伪方向；数学修正与实验合同已锁定，
checkpoint replay 求解尚未运行。**

本文审计2026-09-13外部推导提出的三个设计接口：band、tail depth、transition。
它们共同决定一张全层、全会话固定的Native-relative RoPE表；attention gain在首轮
冻结，后续单独做交互控制。三个接口是有用的block-coordinate求解顺序，不是已被
识别为互相独立的因果模块。

## 1. 为什么这一方向比继续扫band更接近最终问题

现有YaRN、MrPro和BM通常同时固定了三项选择：哪些槽开始/停止移动、慢端是否完整
除以S、transition内的累计位移。当前实验只放开band，已经观察到跨模型、跨倍率、
任务和shape的强交互；因此“找到一个槽位区间”不是最终解。

新方向把候选写成：

\[
\theta=(l,h,c,\epsilon,g),\qquad
\epsilon_q\ge0,\quad\sum_q\epsilon_q=c\le1,
\]

其中band=`[l,h]`，`c`是tail depth，`epsilon`是transition增量，`g`是固定gain。
目标不是由benchmark分数反推参数，而是在隔离的无标签Native文本上用真实checkpoint
Q/K构造有限相位replay风险，再把冻结候选送入真实任务确认。

这给论文提供了一条完整桥梁：

```text
RoPE代数与checkpoint Q/K
  -> 无标签全区间replay目标
  -> band/depth/transition固定表
  -> 独立RULER/QA/PPL验证
```

## 2. 可直接保留的数学核心

以下部分经独立复算可保留：

1. frozen-Q/K条件下的频率一、二阶导数，以及单个score对不同频率的交叉二阶为零；
2. attention KL的局部`J^T(diag(p)-pp^T)J`，其中off-diagonal来自key competition，
   不能用逐槽对角Fisher替代；
3. 完整有限circular phase差和正弦上界；S=8不能默认处于Native Hessian近邻；
4. `/S`对slow substate的目标端dilation matching恒等式；
5. `a=2/(S+1)`在端点minimax、log-uniform平方误差，以及给定偶对称单调
   circular distortion条件下的精确结论；
6. `J_log'(a)=[F(aS-1)-F(a-1)]/[a log S]`；
7. BM的双端increment roughness解、异方差加权解、累计变量下
   `Q_epsilon=T^T H_m T`以及SPD equality-QP闭式。

这些是恒等式或带明确条件的最优解；它们没有证明任一具体checkpoint的最佳band、
tail depth或transition。

## 3. 必须加入的七项修正

### 3.1 Native必须是显式约束

`sup_rho D_rho`包含`rho=1`，不等于Native硬保护。远端风险可能主导最大值，使
Native风险在最大值以下任意上升。首轮至少使用：

\[
D_1(m)\le\epsilon_N
\]

并按layer/head/row组报告worst-group或CVaR；全局均值不能掩盖局部灾难。

### 3.2 连续区间只能作数值近似

circular目标关于`rho`可能多峰。没有Lipschitz上界或区间全局证书时，所谓
`sup_[1,S]`必须写成预注册、自适应补点的有限网格最大值，不能声称认证整个连续区间。

### 3.3 局部QP必须在正确坐标求step

在当前`epsilon_0`附近的Gauss--Newton子问题应写为：

\[
\min_d g^Td+\tfrac12d^TQd,
\quad\mathbf1^Td=0,
\quad\epsilon_0+d\ge0,
\quad\epsilon^*=\epsilon_0+d.
\]

若写成绝对`epsilon`坐标，必须显式给出线性项如何由`epsilon_0,g,Q`变换。每个
尺度的GN使用当前候选的attention分布；Native固定F不能冒充所有尺度的精确GN。

### 3.4 求解器必须允许真正的零增量

softmax参数化会让每个增量严格为正，无法从active zeros发现band或多个plateau。
发现support必须用允许simplex边界的active-set QP或等价投影法，并在看benchmark前
冻结数值/KKT零容差。

### 3.5 单一外包络不保证只有一段transition

`min(active)-1`与`max(active)`只给活跃增量的外包络。全谱解可能出现分离活跃区或
内部零增量；报告必须保存完整64槽增量，不能强行压成YaRN式唯一中段。

### 3.6 C1/C2/C3是顺序条件效应

- C0→C1：C0 shape/depth/gain条件下的band效应；band宽度变化时shape重采样规则锁定；
- C1→C2：C1 band、C0 normalized shape条件下的**整表深度**效应；`m=c b`
  同时移动transition和tail，不能称“只改低频”；
- C2→C3：C1 band、C2 depth、冻结replay corpus条件下的shape效应。

要估计band与shape主效应，最终depth/gain固定后至少补
`band={b0,b1} × shape={shape0,shape3}`的2×2。完整三因素归因需要8格，但不作为
首轮求解的前置成本。

### 3.7 运行几何必须逐项对齐

固定causal lag符号、split-half rotary pairing、GQA head重复、实际attention scale和
`gain^2`。`rho=1,m=0`必须重现零KL；数学replay还要与一个真实runtime attention row
对齐。失败属于实现无效，不产生方法结论。

## 4. 修正后的checkpoint replay目标

对隔离Native语料缓存pre-RoPE Q/K、完整可见keys、causal positions、非rotary项与
实际scale。用精确sin/cos和精确softmax KL定义：

\[
D_\rho(m)=\mathbb E_x\operatorname{KL}
\left(p_x^{Native}\Vert p_x^{replay}(\rho,m)\right).
\]

首轮数值问题为：

\[
\min_m\max_{\rho\in\mathcal R}D_\rho(m)
\quad\text{s.t.}\quad
0\le m_0\le\cdots\le m_{K-1}\le1,
\quad D_1(m)\le\epsilon_N,
\quad \operatorname{CVaR}_{groups}(D_1)\le\epsilon_G.
\]

`mathcal R`是预注册log-length网格并可按区间上界失败处补点。这个目标仍只是假说：
它假定Native内容关系在位置dilation后值得保持，不包含真实新distractors、前层
hidden-state feedback、V/O与自回归阈值。replay改善而真实生成下降，应否定该
transport metric，而不是否定band/depth/shape自由度。

## 5. 最小验证流水线

| 阶段 | 目的 | 成功说明 | 失败说明 | 与最终解距离 |
|---|---|---|---|---:|
| R0 runtime parity | 验证sign/layout/GQA/gain²和精确KL | replay实现可用 | 工程无效，不判理论 | 15% |
| R1 retrospective rank | 不调权重地排序Llama同shape/gain的`[14,32]`与`[18,35]` | replay至少解释已知大效应方向 | 否定当前transport predictor；保留强表 | 25% |
| R2 C1 band | 在C0 shape/depth上求band并冻结 | 得到checkpoint条件band候选 | 当前band求解器未带来任务收益 | 40% |
| R3 C2 depth | 在C1上求标量`c`；`2/(S+1)`只作条件定理control | 完整`/S`不是该区间目标的最佳depth | 否定该depth objective/近似 | 55% |
| R4 C3 transition | 固定band/depth做full-Q active-set shape | joint频率交互提供增量 | replay/局部GN不能预测任务 | 70% |
| R5 2×2与mini | 分离band×shape并测Core-6全区间 | 顺序候选在真实任务非支配 | 定位交互或代理失配 | 82% |
| R6 prospective checkpoint | 完整规则冻结后预测未参与构造的新checkpoint | 支持跨checkpoint预测初始化 | 收窄为模型内或校准式方法 | 92% |
| R7 medium/high与自然任务 | 对最终Pareto表做宽任务、EOS、PPL和独立样本 | 支撑声明范围内论文结论 | 保留反例并收窄claim | 100% |

当前OLMo/Llama/Qwen band小屏与mini属于R5前的现象发现/benchmark校准，不是R1--R4
replay求解已经完成。它们继续运行，为后续真实任务层提供冻结对照，不因新理论中断。

## 6. 独立性与模型安排

Qwen1.5B `[22,39]`已经开封并被新理论材料引用；它对早期winding假说可能曾是
前瞻结果，但对本replay理论只能作retrospective检查。若读取某模型的无标签Q/K来
构表，再在fresh任务行验证，准确名称是`checkpoint-conditioned calibration`，不是
geometry-only leave-one-model-out。

服务器已有另一个Qwen约3B checkpoint，Native geometry为32K/base1M/64槽。其
band-specific旧结果在冻结本理论的prospective预测前不解封；R6应先保存完整solver
输入、输出表和预测时间戳，再运行fresh任务行。若历史资产已经包含同一表，只有在
确认其未参与规则构造后才可作为事后复用证据。

## 7. 实现复用与新增边界

可复用：

- `scripts/analysis/native_attention_kl.py`：FP64有限相位、完整key competition与
  off-diagonal cancellation；
- `experiments/nongeometric_screen/pro_block_calibration.py`：全row pre-RoPE Q/K
  捕获、GQA、gain²及runtime-to-replay校验；
- `experiments/olmo_recovery_20260912/recovery_v2_eval.py`：冻结静态表真实生成；
- 当前永久mini panel、BM/MrPro/Native registry和task-equal log-AUC scorer。

首轮只新增通用流式capture/replay模块与CPU active-set求解驱动，不重建已有
fit/select或基线。捕获在Native窗口执行：OLMo 4K优先完成实现与parity；Llama只在
8K捕获，绝不在64K生成时挂全层Q/K hooks。CPU负责精确replay、rho补点、QP和表构造；
GPU只负责一次checkpoint捕获与冻结表的真实任务验证。

## 8. 论文允许的结论

若C1/C2/C3的replay排序与独立生成一致、2×2不发生破坏性反转、且prospective
checkpoint预测成功，可写：checkpoint-native attention replay provides a predictive
construction for the band, depth, and transition of a single fixed RoPE table over a declared
deployment interval。

若只有depth或band成功，分别写条件机制和轻量校准法。若replay全部失败，仍有明确
负贡献：真实固定表的部分进步不能由Native attention transport充分解释，必须转向
含hidden-state feedback或直接任务约束的full-spectrum minimax。有限实验不能证明
不存在任何通用规律。

