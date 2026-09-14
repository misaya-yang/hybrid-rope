# TailSpline-RoPE方法与统一评测合同

更新：2026-09-14。状态：**精确有限网格实现与CPU代数核验完成；Qwen2.5-3B首个统一GPU判决运行中；尚无性能结论。**

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

有限网格上它精确等于

\[
(1-w_n)m_{BM}+w_nm_{front},\qquad w_n=\frac{3n}{2(2n+1)}.
\]

`n=17/18`时`w_n=0.728571/0.729730`；历史mix075只是连续极限的近似prior。
CPU结果只证明上述声明目标的唯一最优表，不证明任务准确率，不允许把历史mix075分数
记为精确TailSpline结果。

## 3. 为什么相对MrRoPE是新增步骤

MrPro使用递增radix增量，最大的gap改动出现在低频尾端前，再突然接到尾段零增量。
TailSpline把未被优化的tail-connection boundary condition写入离散目标，导出递减增量
与one-sided cubic累计曲线。新增主张是“最小局部log-gap弯曲并平滑接入fully
interpolated tail”，不是任意Q/K上的accuracy dominance定理。

任务反转的机制材料只用于解释为什么少量频率变化可以经signed logit、softmax指数重加权、
value聚合和EOS读出放大；它不用于根据VT/FWE或其他局部指标调表。

## 4. 首个GPU判决：Qwen2.5-3B S2

当前根目录：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_unified_full324/`。

- 模型：Qwen2.5-3B-Instruct，冻结权重；
- 数据：Core-6，32/48/64K，每任务每长度18行，324个prompt/arm；
- 四臂：TailSpline、MrPro、static YaRN、BM；
- 所有扩展方法共同band`[23,40]`、gain`1.0693147180559945`、prompt、decoder、
  batch2和prefill8192，只改变exponent allocation；
- TailSpline FP32表SHA256：
  `bb55a80244758dc071953c5e557e521490f1e1a1ee4c601c8bfb8bd814af19ca`；
- 两臂全部完成前不读取分数；四臂完成后统一生成task-equal log-AUC、worst、逐长度/
  任务、EOS/cap与paired bootstrap报告。

此前Qwen S2 mix075在相同324行上已完成一臂，但band/gain与本合同不同，只作为近似开发
参考；旧batch4的114行OOM partial也继续保留，二者都不并入TailSpline主比较。

这个Core-6块是完整同prompt的快速方法判决，不冒充最终完整benchmark。若它没有相对
MrPro/YaRN的整体正信号，必须如实报告；不得看分数后更改闭式、band或gain。

## 5. 最终只保留三个交付块

### A. 现有raw错误账本

不新增生成。按task/length/method复算official score、空输出、答案前EOS、非空答案召回、
cap及可直接判定的错词/错链类型。它只约束失败解释，不进入构表。

### B. 统一经典评测

最终只测试冻结TailSpline，对照Native reference、YaRN、MrPro和BM。覆盖：

- 同语料、同token计权的长序列PPL曲线；
- passkey以及single/multi-key/multi-value/multi-query NIAH；
- 全部13个RULER任务的task-equal长度曲线和log-AUC。

主判决看PPL、NIAH/passkey和full-RULER三个family-level endpoint的整体方向，不要求每个
task或长度逐格获胜。Native、worst、EOS/cap和局部反转仍完整报告。

### C. 共享数据的最小归因

同gain、同band、同support比较TailSpline与MrPro/BM，确认收益来自新增allocation边界条件，
而不是部署参数。各方法官方默认配置只能另列实用附表，不与纯频率归因混合。

## 6. 停止的路线

fixed-u、coherence、Fisher、replay KL、CAL、OOD/SEP、局部margin和系数/band/gain搜索不再
生成候选。旧Llama局部析因与旧mix075 S4队列已被本合同取代。后续GPU只服务TailSpline
统一benchmark、必要强基线和身份完全匹配的归因控制。
