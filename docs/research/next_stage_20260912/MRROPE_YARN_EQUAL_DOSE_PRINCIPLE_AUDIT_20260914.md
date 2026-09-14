# MrRoPE–YaRN等剂量后移原则审计

更新：2026-09-14。状态：**有限网格CPU复算通过；构成可前瞻检验的理论对照，尚未运行模型，未进入当前GPU队列。**

## 1. 结论

从YaRN到MrRoPE可以提炼出一个有条件的设计假说：在checkpoint、band、两端、gain与
总log-frequency位移均固定时，把中频减速从较高频侧后移到较低频侧，是否能够保护
Native-compatible计算并改善长距离覆盖。

这不是已经成立的迁移定律，也不推出MrPro二次式全局最优。MrRoPE原文确认MrPro与YaRN
均为冻结模型的training-free扩展，并将主要方法差别定位在中频conversion strategy；原文
同时承认不同模型的最佳band不同，只把32圈/1圈作为强默认值：
[MrRoPE原文](https://arxiv.org/abs/2601.22181)。

## 2. CPU有限网格结果

对Llama-3-8B的canonical band `[18,35]`，令`n=17`、`q=0,...,17`：

\[
m_P(q)=\frac{q(q+1)}{17\cdot18},\qquad
m_Y(q;S)=-\log_S\left(1-\left(1-\frac1S\right)\frac q{17}\right).
\]

要求两者累计指数总量相等：

\[
\sum_qm_Y(q;S^\star)=\sum_qm_P(q)=\frac{19}{3}.
\]

独立CPU解得：

- `S*=7.51324282212058`；
- `8192*S*=61548.4852` tokens；
- 共享gain `1+0.1 ln S*=1.20166671731257`；
- 两侧`sum(m)`残差`8.9e-16`，总log位移残差`1.8e-15`；
- `q=1,...,9`上MrPro指数低于YaRN，`q=10,...,16`上高于YaRN；内部只有一次交叉，
  位于`q=9/10`之间。

这个解是唯一的。对任意`u in (0,1)`，令
`h(a)=-ln((1-u)+u*exp(-a))`、`a=ln S`，则`h(0)=0`且`h`严格凹，故`h(a)/a`
随`a`严格下降；有限网格YaRN剂量之和也严格下降，只能与固定MrPro剂量相交一次。

整数倍率比较确实混入剂量变化：`S=4`时YaRN的`sum(m)`比MrPro高`0.770853`，
`S=16`时反而低`0.810933`。因此现有YaRN–MrPro成绩不能单独识别“后移位置”效应。

可执行核验见
`experiments/fixed_rope_three_interfaces_20260913/dose_matched_yarn_mrpro_verification.py`。
该脚本不加载checkpoint、不读取任务分数，也不证明性能优势。

## 3. 与当前TailSpline判决的关系

当前主实验只回答`TailSpline是否胜MrPro`，保持原队列不变。二者在S4下既改变分配形状，
又改变总减速量，所以无论谁赢，都不能单独确认或否决等剂量后移原则：

- TailSpline胜出：说明前移或额外剂量可能有利，与“后移普遍更好”存在张力，但仍有剂量混杂；
- MrPro胜出：可能来自后移，也可能来自较小总剂量；仍需等剂量对照；
- 当前结果不触发对band、gain或TailSpline系数的修改。

## 4. 后续唯一对照，不自动启动

待当前`TailSpline vs MrPro`完成且作者要求继续YaRN理论线时，只运行两臂：

1. exact official static YaRN at `S*`；
2. exact MrPro at `S*`。

两臂固定同一Llama checkpoint、canonical `[18,35]`、gain、端点、prompt、decoder与scorer。
评测应覆盖8/16/32/48K及接近目标窗的60K；PPL、NIAH和Full-13分别报告，不合成临时总分。
当前8/16/32K资产可复用；48/60K输入必须另行冻结，不能把旧S8或64K聚合成绩拼入。

事前预测是：若等剂量后移原则成立，MrPro相对YaRN应改善短中窗兼容性且60K不劣。
若总体等效、长端明显退化或任务族强烈反转，则停止把back-loading称为可迁移原则，
不得依据同批结果改倍率或生成另一条曲线。

