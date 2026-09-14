# TailSpline-RoPE方法与统一评测合同

更新：2026-09-14。状态：**精确有限网格实现与边界条件CPU审计完成；Llama-3-8B与
OLMo-2-1B的统一两臂判决均在Full-13、NIAH与PPL三个预注册方向上胜MrPro。**

本文吸收作者提供的`TailSpline_RoPE_final_answer_20260914.md`与
`MRROPE_TASK_REVERSALS_AND_METHOD_CLOSURE.md`，替代fixed-u、局部task repair和旧
mix075后继队列的执行优先级。冻结checkpoint、零训练、全层和全部输入长度共享一张
静态Native-relative表的部署合同不变。

## 1. 唯一候选

对canonical YaRN/MrRoPE 32圈/1圈边界`[d_l,d_h]`，令`n=d_h-d_l`、
`q=clip(k-d_l,0,n)`。TailSpline累计指数为

\[
m_q=\frac{q(3n^2+3n+1-q^2)}{n(n+1)(2n+1)},
\qquad \omega'_k=\omega_k S^{-m_k}.
\]

逐gap增量为

\[
\epsilon_q=\frac{3(n+q)(n-q+1)}{n(n+1)(2n+1)},\quad q=1,\ldots,n.
\]

高频前缀固定`m=0`，低频尾段固定`m=1`。方法不拟合checkpoint、不搜索band、gain、
depth或混合系数；gain固定为与canonical MrPro/static YaRN相同的
`1+0.1 ln S`。

## 2. 理论闭式与边界

TailSpline是以下one-sided离散roughness问题的唯一解：

\[
\min_{\epsilon}\sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2,
\quad \sum_q\epsilon_q=1.
\]

它平滑接入增量为0的完整`/S`尾段，但不额外强制高频起点增量为0。严格凸性、KKT、
正递减增量、单位总量和最优值
`6/[n(n+1)(2n+1)]`已经由
`tailspline_verification.py`独立核验。

这里的one-sided不是由全局smoothness自动推出的，而是一项结构先验：只惩罚进入完整
`/S`低频tail前的末端jump，不惩罚从Native高频区进入transition时的首端jump。若改为

\[
J_{sym}=\epsilon_1^2+\sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2,
\]

则唯一解恰好是BM，CPU误差为浮点精度量级。更一般地，为首端项赋任意非负权重会形成
一族严格凸、各有唯一闭式解的连续边界族。因此CPU能够证明“给定边界先验后的唯一性”，
却不能选择one-sided还是symmetric；不得把one-sided称为无条件全局最平滑，也不得把这条
连续族用于按benchmark调参。需要GPU检验的任务假说是：相较于MrPro的后置搬运和BM的
双边平滑，真实长上下文任务是否更偏好TailSpline的`early transport + smooth tail landing`。

具体地，对`J_lambda=lambda*epsilon_1^2+sum(Delta epsilon)^2+epsilon_n^2`，令

\[
B_\lambda=\frac{1+\lambda n(n+2)}{2(1+\lambda n)},\qquad
C_\lambda=\frac{(n+1)^2}{2}-(n+1)B_\lambda,
\]

则唯一解与`-q^2/2+B_lambda q+C_lambda`成比例并归一化到总和1。`lambda=0`退化为
TailSpline，`lambda=1`退化为BM；该闭式只用来暴露理论欠定性，不创建额外实验臂。

有限网格上它精确等于

\[
(1-w_n)m_{BM}+w_nm_{front},\qquad w_n=\frac{3n}{2(2n+1)}.
\]

`n=17/18`时`w_n=0.728571/0.729730`；历史mix075只是连续极限的近似prior。
CPU结果只证明上述各自声明目标的唯一最优表和三者的搬运几何排序，不证明任务准确率，
不允许把历史mix075分数记为精确TailSpline结果。

## 3. 为什么相对MrRoPE是新增步骤

MrPro使用递增radix增量，最大的gap改动出现在低频尾端前，再突然接到尾段零增量。
TailSpline把未被优化的tail-connection boundary condition写入离散目标，导出递减增量
与one-sided cubic累计曲线。新增主张是“最小局部log-gap弯曲并平滑接入fully
interpolated tail”，不是任意Q/K上的accuracy dominance定理。

任务反转的机制材料只用于解释为什么少量频率变化可以经signed logit、softmax指数重加权、
value聚合和EOS读出放大；它不用于根据VT/FWE或其他局部指标调表。

## 4. 当前Llama-3-8B S4诊断与主判决边界

统一经典两臂主判决已经完成，准确结果、raw身份与解释边界见
[TailSpline–MrPro Llama经典两臂结果](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)。
Full-13 AUC差为`+3.20pp`，95%配对区间`[+0.65,+5.79]pp`；NIAH为
`+3.75pp`，PPL AUC为`−0.00449`（越低越好），预注册门3/3通过。以下Core-6记录
继续只作历史诊断，不能替代该主结果。

第二checkpoint结果见[OLMo经典两臂结果](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md)：
Full-13差`+49.23pp`、NIAH差`+64.45pp`、PPL AUC差`−3.853`，三个方向均通过，
13个任务AUC差全部为正。两模型结果确立方法候选，但仍不识别总剂量与one-sided形状机制。

当前根目录：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_first/`。

- 模型：Llama-3-8B-Instruct，冻结权重；
- 实际数据：`llama_low108`原始面板是Core-6×8/32/64K×6行；当前命令筛选
  8/16/32K后实际只运行8K与32K，共72个prompt/arm；
- 四臂：TailSpline、MrPro、static YaRN、BM；
- 所有扩展方法共同canonical band`[18,35]`、gain`1.138629436111989`、prompt、decoder、
  batch1和prefill8192，只改变exponent allocation；
- TailSpline FP32表SHA256：
  `594b38f2669b8fe4bf00d6fd2ba61445ba5408acc3103e68386d197e498e6394`；
- 两臂全部完成前不读取分数；四臂完成后只生成明确标记为8K/32K diagnostic的
  task-equal两点摘要、逐长度/任务、EOS/cap与paired bootstrap报告。

此前Qwen S2精确TailSpline单臂只作已启动工作的归档，不继续启动Qwen对照；历史mix075
及旧OOM partial也只作为开发记录，均不并入Llama主比较。

这72行是已启动后发现面板长度不匹配而保留的配对诊断，不是原计划的108行三长度首判，
更不能冒充最终完整benchmark。真实主判决直接使用下节PPL、完整NIAH/passkey与Full-13；
无论诊断结果如何都不得据此更改闭式、band或gain。

## 5. 最终只保留三个交付块

### A. 现有raw错误账本

不新增生成。按task/length/method复算official score、空输出、答案前EOS、非空答案召回、
cap及可直接判定的错词/错链类型。它只约束失败解释，不进入构表。

### B. 统一经典评测

当前主判决只测试冻结TailSpline与MrPro；YaRN和BM后置，不进入当前继续条件。覆盖：

- 同语料、同token计权的长序列PPL曲线；
- passkey以及single/multi-key/multi-value/multi-query NIAH；
- 全部13个RULER任务的task-equal长度曲线和log-AUC。

主判决看PPL、NIAH/passkey和full-RULER三个family-level endpoint的整体方向，不要求每个
task或长度逐格获胜。Native、worst、EOS/cap和局部反转仍完整报告。

NIAH/passkey与Full-13若共享raw，其family读数是重叠视角，不算独立的重复确认。
Full-13表示任务覆盖，不自动表示样本未参与开发；结果注明样本暴露与追加选择过程。
同文档多长度PPL和同语义蓝图多长度生成存在相关性，区间按实际独立单位处理。

### C. 共享数据的最小归因

同gain、同band、同support比较当前TailSpline/MrPro两臂，识别整张allocation干预的作用。
TailSpline同时改变累计log位移总量与细形状，因此这一比较不能单独确认尾端边界条件是
收益原因。BM等后置对照不自动加入当前队列；官方默认配置另列实用比较。针对更强机制
主张的同总log位移对照C及其限制见[审计处理](../reviews/PRO_AUDIT_DISPOSITION_20260914.md)，当前仅为可选方案。

## 6. 停止的路线

fixed-u、coherence、Fisher、replay KL、CAL、OOD/SEP、局部margin和系数/band/gain搜索不再
生成候选。旧Llama局部析因与旧mix075 S4队列已被本合同取代。后续GPU只服务TailSpline
统一benchmark、必要强基线和身份完全匹配的归因控制。
