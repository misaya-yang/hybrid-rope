# 全部相关理论与推导：身份、结论及入稿位置

本文件把现稿理论、三段式历史推导、独立CPU分析和两轮Pro意见放入同一框架。
它是[作者审核包](README.md)的理论底稿；“已证明”指声明条件下的数学命题，
“已观察”指相应实验合同下的结果。二者不互相代替。

## 1. 总体论证与采用层级

**研究对象：**RoPE内部频率配置z。

**现有论证：**固定范围的配置价值 → 完整位置结构 → 模型的学习使用 → 解析构造 → 目标范围内质量。

**本轮增量：**把“位置结构”与“距离调制内容坐标”的联系说清楚；用成功方法的rank—quality
观察提供实证；用倍率响应及有限相位量化构造的实际改动。

| 编号 | 对象 | 结论身份 | 本次安排 |
|---|---|---|---|
| T01 | 支持与内部z | 精确参数化 | 保留现稿 |
| T02 | 增量、总位移与质心 | 精确恒等式 | 保留现稿附录 |
| T03 | 完整pair主角度与重叠 | 标准工具应用及RoPE对象选择 | 保留主文 |
| T04 | r2与平均pair重叠 | 精确身份 | 保留主文 |
| T05 | 慢频子空间集中与幅度 | 条件渐近 | 保留主文/证明 |
| T06 | cosine-only排序反例与窗口依赖 | 明确构造与CPU验证 | 保留主文/附录 |
| T07 | 内容坐标与位置方向的区别 | 新整理的命题与构造性例子 | 主文解释、附录证明 |
| T08 | T/P rank—quality反向排序 | 新CPU值结合已有任务结果 | 主文观察、附录两行表 |
| T09 | 中心化pair特征值与二阶矩 | 条件渐近/标准恒等式 | 本次不扩写 |
| T10 | 两种相位二次目标退化为统一缩放 | 代数反例 | 研究判断，不入稿 |
| T11 | YaRN与固定指数的倍率响应 | 精确恒等式与CPU验证 | 新附录 |
| T12 | T/P逐通道排序与波长量级 | 精确闭式与公共网格计算 | 方法短句、新附录 |
| T13 | TailSpline/BM/Robin边界族 | 声明目标的唯一解 | 保留方法、附录补足解释 |
| T14 | T/C等位移形状交换 | 精确身份；任务排序另测 | 保留现稿 |
| T15 | Native/完整插值参照区间 | 有条件的结构比较 | 新附录 |
| T16 | 固定槽位有限相位代价 | 精确式、局部展开、CPU值 | 新附录 |
| T17 | 整数位置精确核等价性 | 标准谱相似推论 | 全文移至原证明附录，主文短引 |
| T18 | 最优正交换基与κ界 | 条件定理，已补齐证明 | 已整理，本次不应用 |
| T19 | 有限softmax/value及gain | 标准算子身份 | 保留现稿附录 |
| T20 | T/C路径积分与证据相位 | 可微/固定内容条件身份 | 保留现稿，新增部分不扩写 |
| T21 | 区间深度2/(S+1)与周期反例 | 条件最优与反例 | 保留现稿附录 |
| T22 | 功能区间与fixed-u | 条件构造；已有迁移负结果 | 保留研究材料 |
| T23 | NLL、margin与完整任务 | 不同目标的实验/数学反例 | 约束解释，不增加不足清单 |
| T24 | 内容关联×距离×干扰 | 探索性机制假说 | 不进入已完成结论 |
| T25 | Cosh密度泛函、逆CDF与端点锚定 | 声明目标的唯一解及方向结论 | 保留现稿辅助构造 |

## 2. T01–T02：从支持到配置，再到实际安装

写负log频率为xk，令

\[
\omega_k=e^{-x_k},\qquad x_k=a+Rz_k,\quad z_0=0,\ z_{K-1}=1.
\]

固定a、R即固定实际频率端点，内部z仍然自由。Native-relative安装为

\[
\nu_k=\omega_k^N S^{-m_k},\qquad x'_k=x_k^N+m_k\log S.
\]

若Native是几何网格、g0=log(b)/K、m0=0、m(K−1)=c，

\[
z'_k=\frac{kg_0+m_k\log S}{(K-1)g_0+c\log S}.
\]

固定支持后，总log位移是z的统计量。不能将T/P同时改变总位移与形状，描述成“没有比较z”。
T/C回答的是在这一统计量也匹配后，具体形状差异有没有额外作用。

对n个过渡增量εj、累计mq，有

\[
\sum_{q=1}^{n-1}m_q=\sum_{j=1}^n(n-j)\epsilon_j.
\]

固定总增量后，累计位移与增量一阶矩一一对应。严格递增多重集的同矩非平凡排列可能被
重排不等式排除，因此不能把不存在的排列当作未来实验。

来源：[现稿频率参数化](../../sections/02_exponents.tex)、
[TailSpline控制](../../appendix/a10_tailspline.tex)、
[三段理论](../../../docs/research/next_stage_20260912/THEOREM_FIRST_ROPE_DESIGN_20260913.md)。

## 3. T03–T06：完整位置结构

一个固定内容pair的旋转分量为C cos(ωd)+D sin(ωd)。因此对象是完整空间

\[
V_\omega=\operatorname{span}\{\cos(\omega d),\sin(\omega d)\}.
\]

以明确距离测度形成self-Gram Sω与cross-Gram Hων，白化后的

\[
Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}
\]

给出主角度余弦。pair重叠cων=||Qων||F²/2，完整白化Gram满足

\[
r_2(\Gamma)=\frac{2K}{1+(K-1)\bar c}.
\]

它是位置函数重叠的结构描述；主角度与frame potential工具本身属于已有数学。

慢频x=ωL→0时，[cos(xt),sin(xt)/x]趋向[1,t]，位置子空间趋同。
均匀测度下，现稿更细展开为

\[
2-\|Q_{x,y}\|_F^2=\frac{19}{12600}(x^2-y^2)^2+O(\epsilon^6).
\]

原始self-Gram弱特征值约x²/12；白化移除幅度差，因此不能由白化方向数直接推出可用响应能量。

现有明确反例：相同端点的四pair表，可以cosine-only重叠更低而完整pair r2更低；
完整pair排序也可随观察窗口改变。现稿中8个慢pair的r2=2.11是实际有限网格计算，
不依赖把所有通道当作ωL≪1。

**处理：全部保留。**这些内容说明我们选择了合适的位置对象；新内容解释它与内容使用的关系。

来源：[当前理论](../../sections/03_theory.tex)、[完整证明](../../appendix/a1_proofs.tex)、
[反例计算](../../figs/verify_explicit_geometry.py)。

## 4. T07：位置方向重复不等于内容坐标重复

对r个旋转pair，算子R_B(d)始终正交且秩为2r。位置函数接近同一二维空间，并不删除这些内容坐标。

设η=H max|ωk|，则对|d|≤H，

\[
|q_B^\top R_B(d)k_B-q_B^\top k_B|\le\eta\|q_B\|\|k_B\|.
\]

证明用||R(φ)−I||=2|sin(φ/2)|≤|φ|及块对角最大范数。
让2r类内容使用单位正交向量，query与正确key同向、错误key用其他方向，原始margin为1；
任意允许距离下margin≥1−2η，η<1/2时排序保留。

**新增价值：**这个例子把“位置几何与内容使用不同”说清楚。它不是声明真实QK或attention矩阵满秩。
Llama实际被修改的中频在8/16/32K上最小相位约1.95/3.90/7.79rad，不能套η<1/2解释整个中频收益。

**入稿：**主文几句话；[附录草稿](proposed_appendix.tex)包含完整证明。相关频率功能讨论引用
已有Round and Round，不主张首次发现低频可承载内容。

## 5. T08：成功方法的rank—quality观察

Llama公开网格：K64、base500000、Native8192、S4、band[18,35]。

| 范围 | T的r2 | P的r2 | T−P clean任务差 |
|---|---:|---:|---:|
| 16K | 8.283391 | 8.739887 | +3.3897pp |
| 32K | 10.077270 | 10.214888 | +11.7224pp |

连续/整数距离、FP32和因果位置对权重均保留rank排序；任务值来自原有650/2600对报告。
这反驳了“固定支持下只有r2提高才可能改善冻结任务”的必要条件。
它不证明降低r2导致收益，也不支持把r2反向当作新选表目标。

**入稿：**主文正面提出经验认识；完整数值表放附录，避免把r2混入主性能表。

来源：[独立复算](../../../experiments/iclr2027_three_track_sprint_20260915/web_pro_finite_window_checks.json)、
[clean16K](../../../experiments/iclr2027_three_track_sprint_20260915/reports/clean16k_tailspline_vs_mrpro.json)、
[clean32K](../../../experiments/iclr2027_three_track_sprint_20260915/reports/clean_ruler200_tailspline_vs_mrpro.json)。

## 6. T09–T10：两个有用但不应抢占主文的数学补充

### 中心化幅度与二阶矩

均匀[0,1]上，对sin/cos去均值后，两个特征值分别渐近于x²/12和x⁴/720。
后者可由二次多项式扣除一次方向后的残差积分得到，已用高精度核验。
这进一步说明把弱方向白化到单位能量需要幅度补偿。

对均值μ、协方差Σ的内容系数a，

\[
\mathbb E[(a^\top v)^2]=v^\top(\Sigma+\mu\mu^\top)v.
\]

只写Σ需要零均值。这个修正针对外部提案；当前论文没有该遗漏，故本轮不创建一项虚假的
“论文错误修复”。两项均保留在[核查报告](../../../docs/research/next_stage_20260912/WEB_PRO_FINITE_WINDOW_AUDIT_20260915.md)。

### 简单双目标为何不能生成三段结构

若Jk(r)=ωk²[A(r−1)²+B(Sr−1)²]且A、B对所有槽相同，

\[
r_*=(A+SB)/(A+S^2B),
\]

与ωk无关。它不能独立推出band；人为添加未知Ak/Bk会把选择信息移到这些输入中。
这是研究路线筛选的代数观察，本次不写入论文。

## 7. T11–T12：倍率响应与中频干预幅度

固定index-ramp坐标0<t<1时，YaRN波长倍数为

\[
A_Y=\frac1{1-t+t/S}\longrightarrow\frac1{1-t},\quad
\partial_{\log S}\log A_Y=\frac{t}{S(1-t)+t}.
\]

固定指数则有A=S^m，log导数为m。固定turn-ramp权重γ>0也存在波长上限1/γ；
γ=0的完全插值尾部没有这个固定上限。

T/P有限网格精确差为

\[
T_q-P_q=\frac{q(n-q)(3n+q+1)}{n(n+1)(2n+1)}\ge0.
\]

Llama S4最大波长比S^(T−P)=1.765612，S16为3.117385。
Y/P内部排序随S变化：S4大部分MrPro频率更快，S16后部更多MrPro频率更慢。
不能把原论文不同S和当前S4比较串成同一个逐槽方向的连续胜出证明。

**入稿：**方法正文一句量级，完整公式和表放附录。它说明实际改变了什么，不提供任务胜出保证。
也不以该数学比较替代尚未完成的同合同YaRN模型臂。

来源：[独立分析](../../../docs/research/next_stage_20260912/INDEPENDENT_MIDBAND_THEORY_ANALYSIS_20260915.md)、
[CPU数据](../../../experiments/iclr2027_three_track_sprint_20260915/midband_scale_analysis/midband_scale_response_checks.json)、
[YaRN算子](../../../scripts/lib/rope/official_yarn.py)。

## 8. T13–T14：边界目标与等位移交换

TailSpline目标J=Σ(Δε)²+εn²，Σ ε=1。正定三对角矩阵给出唯一解

\[
\epsilon_q=\frac{3(n+q)(n-q+1)}{n(n+1)(2n+1)},\quad
T_q=\frac{q(3n^2+3n+1-q^2)}{n(n+1)(2n+1)}.
\]

加入ε1²得到BM；加入λε1²得到一族边界问题。共同的端点和外侧频段不独立指定λ=0。
“自然边界条件”是所选泛函的变分结论，不是模型对入口没有代价的证据。

总位移匹配C=(1−w)U+wF，w=3n/[2(2n+1)]，有

\[
T-C=(1-w)(B-U),\qquad
(T-C)_q=\frac{q(n-q)(2q-n)}{2n(n+1)(2n+1)}.
\]

差分反对称、零总量且端点不动。n=1、2时T=C；n≥3才有非平凡残余形状。
classic E1的任务排序仍按原合同记录，clean C属于现有X4安排，不写成已完成。

**入稿：**保留既有推导，将目标明确称解析先验；旧诊断细节放回原附录，方法正文只说明控制目的。

## 9. T15–T16：两个参照与有限相位

对Native参照ω和完整插值参照ω/S，定义固定相位容忍θ的区间

\[
D_N=\theta/|\omega-\nu|,\qquad D_D=\theta/|\nu-\omega/S|.
\]

ω/S≤νT≤νP≤ω意味着D_D(T)≥D_D(P)，D_N(T)≤D_N(P)。
它们说明伸长参照与Native参照的取舍，不是context容量指标。

对指定内容分量a cos(νd−ωd0)、a>0、d=Sd0，如果0≤δT≤δP≤π，则T响应更高。
相位从2π减至π则得到反例；不能删除内容相位和单调区间条件。

整数窗口的有限差异由

\[
\rho_H(\delta)=\frac4H\sum_{d=0}^{H-1}\sin^2(d\delta/2)
\]

精确给出，近零主项为[(H−1)(2H−1)/6]δ²。固定槽位RMS是各频率ρ的平均平方根。
32K时T/C最大未绕回差10.672514rad、核RMS0.448329，故等位移不保证小相位。

**入稿：**新附录。主文不引入新的相位评分器，不宣布这些条件已经在真实模型中被定位。

## 10. T17–T18：精确与近似换基

### 精确整数核等价性：保留并搬到已有证明小节

在频率(0,π)内，AᵀRΩ(d)B=RΛ(d)对d=0、1成立，当且仅当频率多重集相同。
d=0给B=(Aᵀ)^{-1}，d=1给谱相似；无混叠区间识别频率。充分性由pair置换构造。
保留原label以使现有交叉引用自动继续工作。

### 有限窗口最优补偿：已完整整理，本次补丁不加入

定义cij=min{ρ_H(ωj−λi),ρ_H(ωj+λi)}，

\[
\mathfrak d_H^2=K^{-1}\min_\pi\sum_i c_{i\pi(i)},\quad
\mathcal E_H^2(S)=(2KH)^{-1}\sum_d\|S R_\Omega(d)S^{-1}-R_\Lambda(d)\|_F^2.
\]

则min_O正交 E²=d²；一般可逆S满足E²≥d²/κ₂(S)²。

证明关键：把实2×2块拆成对易/反对易两部分，使代价分解为ρ(ω−λ)、ρ(ω+λ)。
正交S的块平方范数形成双随机矩阵，pair置换/反射达到最优。
一般S的任意r×c块子矩阵在r+c>K时具有容量下界σ_min²(r+c−K)，
因此其归一化块平方范数逐项支配双随机矩阵；再用||ES||F≤σ_max||E||F得κ因子。

完整独立证明、实数pair可实现性和240组数值检查见
[核查报告§5](../../../docs/research/next_stage_20260912/WEB_PRO_FINITE_WINDOW_AUDIT_20260915.md)。
对独立A/B，必须有AᵀB=I才可直接改写为相似变换；否则需计入零距离误差。

**不加入的理由：**当前论文已有同步补偿身份，这个新界不额外决定TailSpline、band或模型收益。
它适合作为备选附录，未来讨论受限适配时再前置。不是数学被否决，也不是尚未检查。

## 11. T19–T20：attention、value与任务路径

固定当前层内容QKV，分数改变量η给出

\[
p'_j=p_j e^{\eta_j}/\mathbb E_p e^\eta,\quad
o'-o=\operatorname{Cov}_p(v,e^\eta)/\mathbb E_p e^\eta.
\]

gain只有在目标分数与原分数满足逐query的共同正仿射关系时才能精确吸收变化。
完整shape或排序变化一般不属于该情形。二者均为标准算子关系，现稿已保留。

T/C可微效用的路径身份

\[
\mathcal V(T)-\mathcal V(C)=\sum_{q<n/2}(-\delta_q)(G_{n-q}-G_q)
\]

需要整个路径的平均边际值。greedy accuracy、F1不直接满足可微前提。
Pro追加的证据组odds与相位Hq条件把它细分为可观测内容，但尚未解决怎样在结果未知时决定符号。

**处理：**保留现稿附录身份。新增证据组命题不重复入稿；内容标签和value方向未独立确认前，
不使用“已实证中介”表述。

## 12. T21–T24：旧推导和未决研究路线

- 区间深度2/(S+1)：在声明的log长度分布与单调对称失真下最优；周期失真有反例。
  已在现稿[区间章节](../../sections/05_threeband.tex)，本轮不重写。
- 功能区间：给定每槽合法相位/频率区间可解析求单调最小改动表；尚未从公开参数得到通用功能输入。
  它不能自动转成checkpoint捕获或校准任务。
- fixed-u：唯一迁移式是坐标控制，已有OLMo对照任务较差；不能由不变量推出性能。
- NLL/完整答案：已有不同目标响应反向的证据，平均margin改善也不保证逐行正确率。
  不把这些历史现象当作所有方法的统一失败机制。
- 内容关联×距离×干扰：本轮独立分析提出的可检验假说，可与Pro的clean C优先建议分别讨论。
  它尚未运行，也不是这次论文改稿成立的前提。

完整历史与数值见[三段式总结](../../../docs/research/next_stage_20260912/THREE_BAND_RESEARCH_SYNTHESIS_FOR_PRO_20260915.md)。
研究文件中的旧执行建议不覆盖当前作者意图。

## 13. T25：Cosh的辅助变分构造

保留现稿的密度/尾部目标。令ρ是[0,1]上的非负单位密度，尾部质量
T(u)=∫_u^1 ρ(v)dv，则T(0)=1、T(1)=0、T'=-ρ。声明的目标等价于

\[
J[T]=\frac12\int_0^1\{(T'(u))^2+\tau^2T(u)^2\}\,du.
\]

严格凸性与Euler方程T''=τ²T给出唯一解

\[
T(u)=\frac{\sinh(\tau(1-u))}{\sinh\tau},\qquad
\rho(u)=\frac{\tau\cosh(\tau(1-u))}{\sinh\tau}.
\]

其逆CDF为

\[
Q_\tau(v)=1-\frac{\operatorname{asinh}((1-v)\sinh\tau)}{\tau}.
\]

τ→0恢复均匀密度。将采样Q值作端点仿射锚定后，凸性使内部指数向快频方向运输。
该最优性相对于声明的泛函；它不是冻结模型任务风险的最优解。
现有固定支持、MLA与继续训练分别提供不同协议下的作用证据。

**本次处理：**正文/证明不变，保持辅助外推实例。不同的运输方向说明z具有更广的设计价值，
不写成“训练必需加速、冻结必需减速”的普遍法则。
来源：[现稿完整证明](../../appendix/a1_proofs.tex)、
[构造章节](../../sections/04_construction.tex)、[CPU核验](../../figs/allocation_design.py)。

## 14. 来源与数值核验入口

- [公开参数构造器](../../figs/allocation_design.py)：finite-grid身份与安装公式。
- [完整pair反例](../../figs/verify_explicit_geometry.py)：解析Gram与独立数值检查。
- [Pro有限窗口独立核验](../../../experiments/iclr2027_three_track_sprint_20260915/verify_web_pro_finite_window.py)：
  换基、rank、有限相位、渐近与内容公式。
- [其完整结果](../../../experiments/iclr2027_three_track_sprint_20260915/web_pro_finite_window_checks.json)。
- [中频倍率独立分析](../../../experiments/iclr2027_three_track_sprint_20260915/analyze_midband_scale_response.py)：
  构造器复用、倍率导数、T/P顺序、带条件响应、反例及任务重汇总。
- [其完整结果](../../../experiments/iclr2027_three_track_sprint_20260915/midband_scale_analysis/midband_scale_response_checks.json)。
- [两轮Pro与独立观点对比](../../../docs/research/next_stage_20260912/INDEPENDENT_MIDBAND_THEORY_ANALYSIS_20260915.md)。

本次作者审核包没有新模型数据。候选主张若被批准进入论文，再同步当前claim map与证据登记；
外部意见和审核稿不先变成论文证据。
