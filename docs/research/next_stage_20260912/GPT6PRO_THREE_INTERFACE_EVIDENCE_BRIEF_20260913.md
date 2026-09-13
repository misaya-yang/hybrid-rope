# Hybrid-RoPE 固定表三接口：全证据、理论边界与夜间实验简报

日期：2026-09-13

用途：本文件可直接上传给不具备仓库访问能力的外部推理模型。它整理当前固定
RoPE表研究的完整问题、关键正负证据、哪些结论已经可靠、哪些具体命题已被反驳、
三个核心方向的成熟度，以及下一台32GB GPU上一整晚可执行的实验队列。

本文件是研究状态与实验合同，不是论文最终结论。文中“失败/排除”只否定明确写出
的命题与已测协议，不把有限反例升级成整个方法族不可能。

2026-09-13的《三段频率设计：进展判定与三问题数学分析》作为二级理论审计材料
参与了本版修订：其中可复算的双边界假说、非对称tail-depth control和增量坐标
已吸收；线程口述但本地没有owner的`64K八任务59.48 vs 49.01`没有进入证据表；
从Winding-Matched失败推断真实checkpoint代价单调等跳跃也没有采纳。

---

## 0. 一页摘要

### 0.1 最终问题

给定一个已经预训练好的RoPE模型：

- Native上下文长度为`L`；
- Native频率表为`omega^N`；
- 最大部署长度为`H=S L`；
- 整个会话、全部层只能安装一张固定RoPE表，不能按当前输入长度动态换表。

希望构造：

\[
\omega'_k=\omega^N_k S^{-m_k},
\]

使模型保留Native能力，并在`[L,SL]`整个区间稳定有效，而不是只在某一端点强。

### 0.2 三个核心设计问题

一张固定表至少包含三个耦合接口：

1. **Band**：哪些频率保持Native，哪些进入变形，变形在哪里结束；
2. **Tail depth**：慢端究竟完整缩放`1/S`，还是只缩放到更软的终值；
3. **Transition allocation**：给定band和depth，中频累计位移怎样分配。

attention gain是第四个部署变量，但首轮固定；只有在表结构冻结后做最小交互控制。

### 0.3 当前成熟度

| 方向 | 当前成熟度 | 已有证据 | 最大缺口 |
|---|---|---|---|
| Band | 三者中最完善，但仍属早期 | Llama S2/S4/S8、OLMo S4、Qwen S2均有真实生成；已发现强非单调、跨模型与跨S交互 | 仍缺完整三模型mini、同shape/depth的跨S预测与真正prospective checkpoint |
| Transition | 已有强解析基线和受控形状证据 | MrPro、Uni、BM、C42/C42V24、Solver均有结果；BM是双端增量粗糙度最小解 | 仍缺在同一band、同一depth、同一gain下的full-Q/active-set候选与2×2归因 |
| Tail depth | 理论最有新意，实证最空 | `/S`端点性质、`2/(S+1)`条件定理已经推导；构表代码已支持`c<1` | **尚无任何真实模型任务实验直接比较full depth与soft depth** |

### 0.4 当前最重要判断

- 内部exponent allocation确实是一个能独立改变学习和成熟模型行为的真实变量；
- 没有证据支持BM、MrPro或某个固定band在所有模型、任务和倍率上统一最优；
- 部分进步不是偶然噪声的唯一解释：band、depth、shape、gain、checkpoint频谱使用、
  实际距离和任务族存在强交互；
- PPL、逐槽几何、attention exposure、单一endpoint相位同余都不能替代真实生成；
- 当前最值得优先做的新实验是tail depth，因为它是三接口中唯一尚未被真实任务直接
  干预的变量，并且直接对应“端点最优”与“全区间最优”的区别。

---

## 1. 精确优化对象与评价合同

实际运行长度写成：

\[
\ell=rL,\qquad r\in[1,S].
\]

`S`是频率表设计倍率，`r`是实际运行长度倍率；二者不能混用。每个候选在所有`r`
上使用同一张表。

主质量曲线为`Q_theta(r)`。区间平均使用log-length AUC：

\[
A(\theta)=\frac{1}{\log S}\int_1^S Q_\theta(r)\,d\log r.
\]

但AUC不能掩盖中间深坑，因此同时报告：

- Native点；
- endpoint；
- `min_r Q(r)`；
- task-family最差变化；
- EOS与cap-exhaustion；
- 相对同一张固定强基线的逐长度regret。

### 1.1 真实主分数

RULER使用官方task score：期望答案作为子串出现在完整输出中，大小写/空白按官方
规则处理；多答案任务按官方命中比例。完整字符串+EOS、whole-response F1、PPL、
空答和hit-cap单列，不能替代official score。

主聚合顺序固定：

1. task×length cell内先对row取平均；
2. 每个length对task等权；
3. 在实际`log(length)`上做梯形AUC。

逐行W/L只用于诊断。bootstrap按同一prompt配对；任务集合固定，不把任务当随机样本。

### 1.2 评价层级

| 层级 | 任务×长度×行数 | 总行数/arm | 用途 |
|---|---:|---:|---|
| low | Core-6×3×6 | 108 | 发现大效应、协议灾难和任务反转；不淘汰机制族 |
| mini | Core-6×3×18 | 324 | 第一次有统计趋势的候选确认 |
| medium | 9×4×22 | 792 | 冻结少数Pareto候选 |
| high | 13任务、5长度、混合18/22 | 1314 | 声明范围内的最终比较，约full RULER 40%成本 |

Core-6为`niah_single_2`、`niah_multikey_2`、`niah_multiquery`、`vt`、`fwe`、
`qa_1`。

### 1.3 永久基线规则

Native、BM、MrPro按以下完整benchmark身份只运行一次并永久复用：

```text
model/checkpoint + tokenizer/template + prompt rows/hashes
+ task/length + decoder/budget + scorer
+ baseline table/band/depth/gain + precision/arithmetic path
```

换GPU不重跑；换模型、表、prompt、scorer或解码合同则不是同一基线。MrPro只在
缺少可比coverage时补一次。官方YaRN零训练表不加入当前主对照：官方方法依赖全参
SFT，其训练合同与当前零训练固定表问题不匹配。

---

## 2. 三接口的数学对象

令rotary pair数为`K`，Native频率为：

\[
\omega^N_k=B^{-k/K}.
\]

用非负增量表示单调exponent：

\[
\epsilon_q=m_q-m_{q-1}\ge0,
\qquad \sum_q\epsilon_q=c\le1.
\]

### 2.1 Band

Band不是YaRN预定义的低/中/高区间，而是增量support的经验外包络。若增量出现多个
分离活跃区，不能强行解释成唯一连续中段。

当前候选几何坐标包括：

- raw slot；
- Native winding `t_k=L omega_k/(2pi)`；
- Native-window phase distortion；
- target-window phase distortion；
- checkpoint Q/K replay或phase sensitivity。

这些都是候选预测器，不是充分统计量。

两个值得直接证伪、但尚非结论的边界假说是：

1. **H-slow：约一圈覆盖锚。** 定义Native窗口内转数

   \[
   t_k^N=\frac{L\omega_k^N}{2\pi}.
   \]

   若慢端槽完整缩放`1/S`，则目标端有`t_k^H=t_k^N`。令`t_h^N=1`得到

   \[
   h_{1\rm turn}=\frac{K}{\log B}\log\frac{L}{2\pi}.
   \]

   它是与`S`无关的覆盖包络和high初始化，不是最佳high的定理；多槽联合码、
   checkpoint使用及shape都能把实测basin拉离一圈位置。

2. **H-fast：反事实Native相位预算。** low槽本身实际`m_l=0`、扰动为零；为给它
   一个与首个小增量无关的坐标，定义“若该槽被完整压缩”的反事实量：

   \[
   \bar D_N(k;S)=L\omega_k^N\left(1-\frac1S\right).
   \]

   若同一checkpoint的可承受预算`B_{f,M}`近似固定，则预测

   \[
   \Delta l=\frac{K}{\log B}
   \log\frac{1-1/S_2}{1-1/S_1}.
   \]

   当前`B_{f,M}`数值是从开发赢家反推的潜变量，不是已由Q/K独立测得的checkpoint
   常数。Llama S8的`[17,35]`是该规则的联合预测格；它没有理论依据预言具体AUC或
   32K分数，也不能单独分离fast与slow边界，必须结合high固定/移动的控制解释。

### 2.2 Tail depth

固定normalized profile `b_k in [0,1]`，令：

\[
m_k=c b_k,
\qquad a=S^{-c},
\qquad \omega'_k=\omega^N_k a^{b_k}.
\]

`c=1`对应慢端完整除以`S`。它对slow substate有精确端点dilation matching：

\[
SL(\omega/S)=L\omega.
\]

但该性质只服务端点/禁止unwrapped OOD，不自动最大化整个区间。

对isolated slow block定义signed有效距离误差`z=a r-1`及代价`F(z)`。log-uniform
区间目标满足精确导数：

\[
J_{\log}(a)=\frac1{\log S}\int_1^S F(ar-1)\,d\log r,
\qquad
J_{\log}'(a)=\frac{F(aS-1)-F(a-1)}{a\log S}.
\]

在`F`偶对称并随`|z|`**严格增大**的isolated slow-block目标下：

\[
a^*=\frac{2}{S+1},
\qquad
c^*=\log_S\frac{S+1}{2}.
\]

对应：

| S | `a=2/(S+1)` | `c` |
|---:|---:|---:|
| 2 | 0.666667 | 0.584963 |
| 4 | 0.400000 | 0.660964 |
| 8 | 0.222222 | 0.723308 |

这是条件定理和实验control，不是预先宣布的checkpoint最优值。真实candidate depth
最终应由完整checkpoint replay或真实任务决定。

一个可复算的非对称control是分段二次代价：

\[
F_\kappa(z)=
\begin{cases}
\kappa z^2,&z\ge0\quad\text{(overshoot)},\\
z^2,&z<0\quad\text{(pull-in)},
\end{cases}
\qquad
a^*(\kappa)=\frac{1+\sqrt\kappa}{1+S\sqrt\kappa}.
\]

只有`kappa>=1`时，`a*`才位于`[1/S,2/(S+1)]`；`kappa<1`时方向相反。
例如`S=4,kappa=10`给`a*=0.305,c≈0.857`，反向权重给`a*=0.581`。
这只是指定损失下的第三个control。Winding-Matched失败不能识别真实`kappa`，也不能
证明checkpoint的`F`单调。并且`2/(S+1)`优化的是区间目标，不预言soft depth在
`SL`端点优于完整`1/S`。

### 2.3 Transition allocation

给定band和depth后，`epsilon`决定中频预算分配。令`N=h-l,q=1,...,N`：

| 构造 | 增量坐标 | 精确含义 |
|---|---|---|
| MrRoPE-Uni | `epsilon_q=c/N` | 等权gap deformation |
| MrRoPE-Pro | `epsilon_q=2cq/[N(N+1)]` | 增量向慢端算术增大；不是已证明的checkpoint optimum |
| BM | `epsilon_q=6cq(N+1-q)/[N(N+1)(N+2)]` | 双端ghost-zero的increment roughness唯一最小解 |
| YaRN代码式 | `m(x)=-log[1-(1-1/S)x]/log S` | ratio空间线性；在`m`空间不是线性，增量向慢端增大 |
| C42/C42V24/Solver | 冻结的经验或局部求解profile | 需要受控任务结果，不由名字提供最优性 |

同band、同depth下，BM相对Pro的累计位移满足：

\[
m_q^{BM}-m_q^{Pro}
=c\frac{2q(q+1)(N-q)}{N(N+1)(N+2)}>0,
\qquad 0<q<N.
\]

因此BM在transition内更早累计压缩，但这个恒等式不预测哪个任务会赢。

若引入解析加权，必须区分两类目标：

\[
\min\frac12\sum_q w_q\epsilon_q^2
\Rightarrow
\epsilon_q=c\frac{w_q^{-1}}{\sum_t w_t^{-1}},
\]

而weighted roughness使用`R=D^T diag(kappa) D`并给
`epsilon*=c R^{-1}1/(1^T R^{-1}1)`。完整checkpoint二次模型需先做累计坐标变换：

\[
Q_\epsilon=T^T H_m T,
\]

再在simplex边界上用active-set求解；逐槽`H_{m,kk}`不能直接当gap权重。

BM是指定离散粗糙度目标的唯一解，不是任务最优性的定理。

### 2.4 三者不可独立宣称因果

C0→C1→C2→C3可以作为block-coordinate构表顺序：

- C1：在C0 shape/depth/gain条件下换band；
- C2：在C1 band和C0 normalized shape条件下改变全表depth；
- C3：在C1 band、C2 depth下改变transition。

这些差值是顺序条件效应。最终至少补`band={b0,b1}×shape={shape0,shape3}`的2×2，
否则不能把收益单独归因给band或shape。

`m=c b`改变的是整个profile深度，同时改变transition和tail，并非“只动低频”。若首个
depth对照出现足够大的任务信号，可条件追加一个匹配`sum(m)`的widen-companion，
判断收益更接近终值位置还是总压缩剂量；该companion改变band，只是二阶段归因控制，
不作为首次depth判决的前置成本。

---

## 3. 已经可靠的基础证据

### 3.1 内部exponent allocation确实有独立作用

151.9M scratch模型、三seed、固定端点/log-span/初始化/数据顺序/优化器/token预算，
只改变30个内部指数。Cosh相对uniform在512/1K/2K的NLL差为：

```text
-0.28073 / -0.17599 / -0.14571
```

3 seeds×3 OOD cells全部同向；Native 256附近代价约+0.026。这个结果证明内部铺点
本身是可识别变量。

### 3.2 多种shape都有价值，但没有唯一解析曲线

50.9M、12个配置×3 seeds、严格固定端点：reference Cosh改善7/12，预指定
1.25×Cosh改善10/12，deformation-matched exponential改善9/12。它证明“存在可
系统构造的shape”，不证明某一条曲线普适。

### 3.3 模型会学得使用训练时的位置基

weights×runtime-table crossing出现明显对角偏好：

- 50M PPL矩阵约为`[[7.14,76.20],[23.05,7.16]]`；
- 151.9M两seed在1K的tail NLL约为`[[3.426,5.776],[4.455,3.479]]`。

同一多重集只置换内部槽位，OLMo tail NLL可从3.10423恶化到6.86493，Qwen 64K
可从0.7000降到0。成熟checkpoint并不只读取频率集合，也读取其槽位/权重耦合。

### 3.4 相同总量和质心仍不能决定任务结果

OLMo C42与C42V24固定相同support、band、`sum(m)=42`和增量质心，350个开发提示
task score为43.6714与54.4000，相差+10.7286pp；16篇自然文本NLL约2.9417与
2.8309。这是transition shape有真实任务作用的最强受控证据之一。

### 3.5 BM有实际用途，但不是统一解

OLMo五任务自然QA中，BM相对原对照task-equal F1从21.62提高到25.44，差+3.82pp，
重采样区间[1.32,6.29]。另一72提示长层面板BM/Uni/officialYaRN/MrPro为
51.32/32.12/6.94/2.78。

这些结果建立BM的实用价值；后续Llama/Qwen反例证明它不能升级为统一最优。

---

## 4. Band实验：当前最完整的方向

### 4.1 Llama-3-8B，S=2，小开发屏

Native `L=8192`、base 500000、64槽。三项NIAH×每格4行；两篇PG19 PPL。

| 表 | 8K official | 16K official | 8K PPL | 16K PPL |
|---|---:|---:|---:|---:|
| BM | 100.00 | 100.00 | 16.9792 | 14.8385 |
| MrPro | 100.00 | 97.92 | 17.0178 | 15.2512 |
| C42 `[12,30]` | 100.00 | 97.92 | 17.0307 | 14.8697 |
| C42 `[14,32]` | 100.00 | **100.00** | 16.9891 | **14.8358** |
| C42 `[16,34]` | 100.00 | 87.50 | 16.9676 | 14.8381 |
| C42 `[18,36]` | 100.00 | 97.92 | 16.9657 | 14.8454 |

只移动两槽，检索可变化12.5pp，而PPL几乎不变；再移动两槽又恢复。band响应明显
非单调，PPL无法选择band。

### 4.2 Llama-3-8B，S=4，小开发屏

| 表 | 8K | 16K | 32K | 32K PPL |
|---|---:|---:|---:|---:|
| BM | 97.92 | 91.67 | 97.92 | 14.8988 |
| MrPro | 97.92 | 91.67 | 64.58 | 15.1773 |
| C42 `[14,32]` | 100.00 | 91.67 | 95.83 | 14.8120 |
| C42 `[16,34]` | 100.00 | 91.67 | **97.92** | **14.7715** |
| C42 `[18,36]` | 97.92 | 91.67 | **18.75** | 15.5017 |

当前小屏偏好`[16,34]`，但每格只有4行。`[18,36]`从16K健康到32K突然崩溃，是
当前最值得理论解释的单一现象之一。

### 4.3 Llama band直接迁移到OLMo失败

OLMo Native `L=4096`，base仍为500000、64槽。Llama S4赢家`[16,34]`原样安装：

| 表 | NIAH 4K/8K/16K | QA2 4K/8K/16K | tail-512 NLL 4K/8K/16K |
|---|---:|---:|---:|
| BM | 100/75/75 | 75/75/25 | 2.9551/2.9549/2.8621 |
| OLMo C42 `[14,32]` | 100/87.5/75 | 50/100/50 | 2.9219/2.9312/2.8301 |
| Llama `[16,34]`直迁 | 100/62.5/50 | 75/100/25 | 2.9484/2.9682/3.2420 |

16K NLL相对BM恶化+0.3800 nat、16/16文档同向。这个结果排除“相同base/槽数即可
原样迁移最佳band”，不排除checkpoint-conditioned预测。

### 4.4 OLMo局部band搜索

固定C42 normalized shape和S4 gain，局部搜索得到：

- 当前basin：low=14、high=31--32；
- `[14,31]`与`[14,32]`小屏生成完全相同，PPL差极小且区间跨零；
- low移到13损伤16K NIAH但可能提高4K QA；
- `[14,30]`或`[14,33]`均损伤16K；
- `[16,34]`明显更差。

### 4.5 OLMo唯一已完成Core-6 mini

6任务×4/8/16K×18行=324行，所有输出按prompt hash覆盖并从raw text统一重算。

| 表 | 4K | 8K | 16K | log-AUC | 最弱点 |
|---|---:|---:|---:|---:|---:|
| BM | **81.67** | 71.31 | 48.50 | 68.20 | 48.50 |
| MrPro | 42.28 | 25.63 | 8.18 | 25.43 | 8.18 |
| C42 `[14,31]` | 79.65 | **71.84** | **53.15** | **69.12** | **53.15** |

C42相对BM：AUC +0.92pp；16K和最弱点+4.65pp；4K -2.02pp。20,000次配对
bootstrap的AUC差区间[-2.48,+4.36]pp，`P(delta>0)=0.698`。

任务AUC不是同向：multikey、multiquery、FWE、single改善，VT -8.61pp，QA
-1.39pp。因此它是Pareto候选，不是已认证赢家；4行NIAH屏幕确实会遗漏重要损伤。

### 4.6 Llama S=8，4080固定五长度开发屏

此处面板为三项NIAH×每格4行，8/16/32/48/64K；不能与另一台5090上的不同prompt
panel拼接。

| 固定表 | 8K | 16K | 32K | 48K | 64K | 5点AUC | 最弱点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM | 95.83 | 85.42 | 97.92 | 95.83 | 72.92 | 91.33 | 72.92 |
| MrPro | 91.67 | 95.83 | 87.50 | 100.00 | 68.75 | 91.76 | 68.75 |
| Solver `[14,32]` | 95.83 | 97.92 | 100.00 | 95.83 | **85.42** | **96.91** | **85.42** |
| Solver `[16,34]` | 97.92 | 97.92 | 97.92 | 95.83 | 60.42 | 94.98 | 60.42 |
| Solver `[18,35]` | 97.92 | 97.92 | 97.92 | 87.50 | 45.83 | 92.58 | 45.83 |
| C42 `[14,32]` | 95.83 | 97.92 | 100.00 | 95.83 | 81.25 | 96.62 | 81.25 |
| C42 `[16,34]` | 97.92 | 97.92 | 87.50 | 95.83 | 81.25 | 93.67 | 81.25 |

关键修正：Solver shape下`[14,32]`与`[16,34]`的64K差25pp；匹配C42 shape后两者
64K相同，主要差异转到32K。此前25pp不能归因于band本身，证明shape×band交互很强。

### 4.7 Qwen2.5-1.5B，S=2

Native `L=32768`、base 1M、64槽。初始32/64K小屏：

| 固定表 | 32K | 64K | 32K NLL | 64K NLL |
|---|---:|---:|---:|---:|
| Native | 79.17 | 52.08 | 3.1995 | 2.8275 |
| BM | 79.17 | 83.33 | 3.2613 | 2.8340 |
| MrPro | 95.83 | 79.17 | 3.2546 | 2.8370 |
| C42 `[14,32]` | 95.83 | 81.25 | 3.2473 | 2.8296 |
| C42 `[16,34]` | 100.00 | 87.50 | 3.2449 | 2.8329 |
| C42 `[22,39]` | **100.00** | **91.67** | 3.2535 | **2.8247** |
| C42 `[23,40]` | 95.83 | 83.33 | 3.2510 | 2.8267 |

64K三NIAH任务扩至每任务16行后：

```text
Native 46.88
BM     80.73
MrPro  82.81
[16,34] 83.33
[22,39] 87.50
[23,40] 81.77
```

`[22,39]`相对BM +6.77pp、相对MrPro +4.69pp，但相对MrPro的配对区间
[-0.52,+11.46]pp，不能称唯一SOTA。

### 4.8 机器故障时的Qwen mini状态

Core-6×32/48/64K×18行mini已冻结。机器失联前最后确认GPU 100%、约24.8/32.8GB、
无OOM；四条逐行落盘作业的最低确认前缀为：

```text
C42 [22,39] mini gap: 104 / 270
BM mini gap:          104 / 270
MrPro mini gap:        70 / 270
C42 [20,38] low:       29 / 108
```

随后SSH在banner阶段超时。数据盘若保留，可按JSONL前缀继续；不得重复启动。四个
独立长窗进程可能增加驱动、主存和I/O压力，虽无证据证明是宕机原因，下一台改为
单驻留模型顺序多表，或最多双进程。

---

## 5. Transition与模型条件化求解证据

### 5.1 BM与MrPro描述不同“增量到达时序”

同band、同depth下，BM比MrPro更早在transition中累积位移；BM的解析差值在所有
内部点为正。这解释二者不同，但不预测哪个任务会赢。

### 5.2 Llama不同倍率与任务出现排序交叉

另一独立5090小panel上，Llama g8的BM/MrPro在1×/2×偏向BM，在4×/8×偏向MrPro；
自然任务又不复现完全相同排序。这个结果与4080 panel身份不同，不能合并成一条
曲线，但二者共同否定“BM总强”或“MrPro总远”。

### 5.3 OLMo模型条件化trust-region首轮

固定S4和`[14,32]`，对17个有效shape自由度+gain做真实answer/EOS NLL与
source-counterfactual的一阶trust-region：

- BM初始化接受0步；只否定本次局部提议，不否定BM或更大空间；
- C42V24初始化接受1个alpha=0.5步，fit最坏range regret 0.2273→0.2108，平均
  regret 0.0685→0.0288，endpoint差0.1534→0.0667，source hinge
  0.8798→0.7215；
- select/confirm没有统一赢家；
- 350行16K七任务：C42 43.67、C42V24 54.40、Solver 50.39；
- 391行自然QA：MrPro 25.92、BM 29.41、Solver 29.57，Solver相对BM统计未决。

shape×gain 2×2显示，在350行面板中主要损伤随allocation变化，gain贡献很小且有负
交互。该Solver在另一些长度改善，因此仍是Pareto，不是失败或赢家。

### 5.4 当前checkpoint replay代码状态

已经实现但尚未在真实checkpoint运行：

- exact finite circular attention-KL；
-独立Native/candidate gain与position dilation；
- GQA、split-half、causal lag符号；
- finite log-rho grid；
- Native mean与worst-group/CVaR约束；
- 允许exact-zero increments的closed-simplex局部PSD QP；
- capture receipt格式。

41项合成/回归测试通过。它只是代码准备，不是模型结果。真实首步必须先做
`rho=1,m=0 -> KL=0`和runtime attention row parity。

---

## 6. CPU理论与几何结果

### 6.1 权重engagement跨模型相似，位置相位核不相似

OLMo与Llama逐槽Q/K投影范数engagement相关`r=0.970`，slow plateau承载更强；但
短文本pre-RoPE Q/K激活的相位核对比度相关约`-0.025`。这支持：

- “有多少承载”可能跨模型稳定；
- “哪些槽承担位置差异”高度checkpoint相关；
- 纯配置几何可以给初始化，但未必给最终band。

### 6.2 单一暴露/裂隙标量不能排序真实分数

已有11臂数据出现多组反例：过暴露更小的表反而更差；MrPro的暴露和大于gamma3，
分数却更高。暴露日程可以解释某些差异在何长度开始出现，不能作为最终选表目标。

### 6.3 OOD-max + SEP-min严格几何路线失败

12张冻结表上，`task vs -OOD_max`相关0.378，`task vs SEP_min`为-0.713；原始
OOD+SEP Pareto有22个“几何支配但任务反转”。即使事后改成OOD-p95仍有9个反转。
因此该两标量路线不能继续生成候选；几何恒等式本身仍成立。

### 6.4 Winding-Matched端点同余构造失败

作者提出的逐槽整数绕圈表在实数域实现端点同余，并保持频率有序；但Llama实际
结果为：8K 89.58%，16/32/48/64K全部0。它直接否定“端点相位同余足以保证真实
长任务”，但不否定无band全谱优化。

---

## 7. LoRA结果：支持能力边界，但不是当前零训练主线

### 7.1 OLMo BM适配

BM_g4全层QKVO+FFN LoRA：8K200步约260秒、16K200步约516秒。16K NIAH从
frozen 68.75%提高到93.75%；NLL也下降。但32/64K简单检索仍为0/16并全部hit cap。

结论：适配能提升模型在训练附近的能力；PPL/NLL下降仍不保证更远检索。

### 7.2 Llama BM适配

BM_g4 8K-step100降低4--32K NLL，自然13格F1 33.23→35.01，检索8/16/32K为
100/100/97.92。再加10步16K时NLL继续下降但自然任务回落到34.58，故停止。

LoRA不是今晚三接口零训练实验的主变量；只有最终静态表成熟后，才考虑匹配适配。

---

## 8. 当前可以写成solid的结论

| 编号 | Solid结论 | 证据边界 |
|---|---|---|
| S1 | 内部exponent allocation在固定支持下能独立改变学习与外推 | 151.9M三seed配对及成熟冻结干预 |
| S2 | 模型学得使用其训练时的位置基，运行表与权重存在耦合 | weights×table crossing、同谱置换 |
| S3 | 相同band、总位移和质心仍不足以决定任务表现 | C42/C42V24受控对 |
| S4 | Band是高影响变量，响应可尖锐、非单调且与S交互 | Llama S2/S4及OLMo/Qwen屏幕 |
| S5 | raw slot最佳band不能直接跨checkpoint | Llama `[16,34]`→OLMo失败 |
| S6 | shape×band交互足以改变差异出现在哪个长度 | Llama S8 Solver/C42匹配对照 |
| S7 | BM有真实任务价值，但不是跨模型/倍率/任务统一赢家 | OLMo正结果与Llama/Qwen交叉 |
| S8 | PPL、NLL、几何量只能作代理，不能替代完整生成 | 多次PPL/检索解离与几何反转 |
| S9 | `/S`是slow-substate端点dilation matching解 | 数学恒等式；不是区间任务最优性 |
| S10 | BM是指定双端increment roughness目标的唯一离散解 | 数学结论；不是任务最优性 |
| S11 | C1→C2→C3只能报告顺序条件效应 | 变量耦合与现有factorial反例 |
| S12 | 当前没有一张经过足量跨模型×S×任务确认的统一固定表 | 三模型mini尚未全部完成 |

---

## 9. 已经失败、可以永久排除的具体命题

这里的“永久排除”表示：除非协议、模型或目标发生实质变化，不再花GPU重复同一命题。

| 编号 | 被排除的具体命题 | 直接反例 | 不应扩大成什么 |
|---|---|---|---|
| F1 | PPL/NLL下降足以证明远距检索或证据使用 | OLMo训练后32K NLL下降但检索0/16；band PPL近等而任务差巨大 | 不等于PPL毫无诊断价值 |
| F2 | 单一目标端点相位同余足以保证端点或全区间任务 | Winding-Matched 16--64K全0 | 不等于所有周期性/无band方法失败 |
| F3 | 一组raw slot band可在相同base/K的模型间直接迁移 | Llama `[16,34]`直迁OLMo显著退化 | 不等于checkpoint-conditioned band不存在 |
| F4 | S4小屏赢家无需S8复核即可宣称为S8赢家 | Llama matched-C42在S8的中间长度排序改变 | 不等于`[16,34]`已被充分样本否决，也不等于`D_N`等含S规则失败 |
| F5 | YaRN式机械重映射`[18,35]`是Llama S8部署赢家 | 64K 45.83、最差点45.83 | 它在8--32K仍是机制/Pareto控制，不删除raw |
| F6 | OOD-max+SEP-min可作为冻结成熟模型的选表泛函 | 22个几何支配/任务反转 | 不等于完整checkpoint-aware joint metric失败 |
| F7 | 对称gamma3或BM/MrPro中点沿表空间平滑改进Llama S8端点 | 64K分别60.42和56.25，均弱于BM | 只排除已测构造/配置，不排除其他transition |
| F8 | BM总比MrPro强，或MrPro总在远端更强 | 同模型不同S、任务和panel发生排序交叉 | 不排除二者作为永久强基线 |
| F9 | 只看逐槽engagement/exposure/Fisher对角即可排序任务表现 | 多组逐槽标量方向反转 | 不排除full-frequency、key-competition joint metric |
| F10 | 当前零训练问题需要跑official YaRN表作主训练对照 | official YaRN依赖全参SFT，合同不匹配 | 不否定YaRN方法本身 |
| F11 | softmax-positive increments可用于发现exact-zero band support | 数学上softmax每个增量严格正 | 不排除active-set或投影simplex |

---

## 10. 尚未解决、绝不能写成失败的方向

| 状态 | 问题 |
|---|---|
| U1 | 跨模型、跨S的统一band坐标是否存在；`D_N`、约一圈slow edge和Q/K谱修正均未完整验证 |
| U2 | Qwen `[22,39]`是否在完整Core-6 mini保持优势；机器故障时未完成 |
| U3 | Llama Solver/C42 `[14,32]`在324行mini及宽任务是否仍强 |
| U4 | OLMo `[14,31]`相对BM是微弱AUC优势还是任务重分配；当前区间含零 |
| U5 | tail depth `<1`是否改善全区间AUC并守住endpoint；没有任何真实任务实验 |
| U6 | checkpoint exact replay是否能排序已知band，或只是另一个失败代理；只完成合成测试 |
| U7 | full-Q active-set transition是否胜过BM/C42；尚无真实checkpoint候选 |
| U8 | band×depth×transition联合stationarity；顺序block-coordinate后必须回验 |
| U9 | 真正prospective新checkpoint预测；Qwen1.5结果已经污染新理论的独立性 |
| U10 | medium/high RULER、自然QA和独立seed上的最终部署价值 |

---

## 11. 为什么现有方法总是“部分进步”

### 11.1 同一位移在不同内容和距离上符号不同

冻结单层Q/K时：

\[
a(d)=\sum_k\operatorname{Re}(C_ke^{id\nu_k}),
\qquad
\frac{\partial a}{\partial m_k}
=\log S\,d\nu_k\operatorname{Im}(C_ke^{id\nu_k}).
\]

导数符号随距离、layer、head、token内容和checkpoint改变。因此一个shape可能修复
长距相位，却同时损伤局部绑定或另一任务的key competition。

### 11.2 三接口与gain耦合

移动band通常改变`sum(m)`；缩小tail depth会移动整个transition和tail；换shape又
改变每个槽何时累积位移；gain在完整rotary attention logit中按平方起作用。只改
一个名字而不做matched control会错误归因。

### 11.3 不同任务需要不同距离结构

single-key、multikey、multiquery、VT、FWE和QA对局部顺序、精确绑定、聚合与远距
读取的需求不同。OLMo mini中C42改善多项检索却损伤VT，就是直接例子。

### 11.4 代理目标与自回归生成存在断层

teacher-forced NLL、attention KL、source margin和几何距离都连续；greedy生成具有
离散token阈值、EOS和hit-cap。代理改善而完整输出不升，优先判为metric/transport
失配，不用更多优化步强行挽救同一代理。

---

## 12. 三接口checkpoint replay：可保留与必须修正

### 12.1 可保留

- frozen-Q/K一、二阶导数；
- attention KL局部`J^T(diag p-pp^T)J`和off-diagonal key competition；
- exact circular phase与softmax KL；
- `/S`端点性质和`2/(S+1)`条件定理；
- BM/weighted allocation/full-Q equality-QP数学；
- detached replay不是完整网络导数、不是任务正确attention的明确边界。

### 12.2 必须修正

1. `sup_rho D_rho`包含rho=1不等于Native硬保护；另加`D_1<=epsilon_N`和
   worst-group/CVaR；
2. circular目标关于rho可多峰；无Lipschitz证书时只能称有限预注册网格近似；
3. 局部GN必须在step坐标写`epsilon=epsilon0+d`，而不是混淆绝对坐标；
4. solver必须允许exact-zero increments；
5. active increment外包络不保证只有一个连续band；保存完整64槽数组；
6. C1/C2/C3只给顺序条件效应；最终补2×2并检查joint stationarity；
7. causal lag符号、split-half、GQA、gain平方必须和runtime逐token对齐。

---

## 13. 下一台机器的一整晚固定实验队列

下面按10--12小时设计。实际时间由新卡吞吐校准，但**顺序和科学合同不因中途点
估计改变**。只有分支条件触发预先写好的备用路线。

### 13.1 运行纪律

- 单模型驻留后顺序安装多张表；Llama 64K只允许一个GPU进程；
- OLMo/Qwen最多双进程，只有显存、主存、I/O和功耗稳定时使用；
- 不用假负载追求显存数字；追求稳定总吞吐、接近满GPU利用率；
- 每行立即写JSONL，summary/status原子更新；重启只从有效前缀继续；
- CPU并行做tokenization、coverage、bootstrap、replay/QP和下一模型准备；
- 数据盘只清理已汇总、可重建的失败临时缓存；模型、表、raw输出、baseline、
  capture receipt和Pareto候选不删。

### 13.2 Phase 0：恢复与身份覆盖（0:00--0:30）

1. 读取新SSH、GPU/CPU/RAM/磁盘；不做全盘hash；
2. 若旧数据盘存在，检查四条Qwen JSONL前缀并继续，不重跑；
3. 若数据盘丢失，从本地冻结panel、表构造代码和永久baseline登记重建；
4. 注册一个单驻留multi-table runner，禁止恢复旧四进程并发；
5. CPU同时生成三模型tail-depth固定表和结果manifest。

输出：恢复receipt、每臂真实coverage、剩余时间估计。

### 13.3 Phase 1：完成已有Qwen证据（0:30--1:45）

按前缀顺序完成：

1. C42 `[22,39]` mini；
2. BM mini缺口；
3. MrPro mini缺口；
4. retrospective `D_N`候选`[20,38]` low。

完成后立即计算Core-6三长度曲线、AUC、worst-length、family交互和20k bootstrap。
`[20,38]`是在Qwen结果已开封后提出，只能称retrospective probe。

结果分支：

- `[22,39]`在mini保持非支配：作为Qwen C0；
- 若相对BM/MrPro混合：保留Pareto，不再扫邻点；
- 若某任务族崩：记录band×task反例，仍继续tail-depth，因为它回答不同变量。

### 13.4 Phase 2：tail depth首次真实实验，OLMo优先（1:45--2:30）

固定OLMo C42 `[14,31]`、shape、S4 gain与同一low108 panel，只比较：

```text
C0: c=1.000000, slow a=1/4
Dlog: c=0.660964, slow a=2/5
```

不扫描第二个软深度。BM/MrPro/Native复用。若Dlog在AUC、endpoint、Native或任务族
之间出现混合结果，保留为Pareto并补到mini；只有协议无效才停止。

只有在R0/R1证明replay至少能回溯已知大效应、并从未看目标分数的capture中冻结了
明确`F_kappa`与`kappa`时，才追加非对称`c_kappa`第三臂。不得从WM失败、开发任务
赢家或端点分数反推`kappa`。若首轮depth出现大效应，再按§2.4加入matched-dose
companion做归因，而不是提前扩展网格。

这个实验无论正负都直接回答：端点完整插值是否阻碍全区间表现。

### 13.5 Phase 3：同band transition控制，OLMo（2:30--3:15）

在Phase 2得到的两个depth中，选择任务上非灾难且replay风险较低的一个作为固定
depth，同时保留C0。构造：

```text
C42 shape @ band[14,31]
BM shape  @ band[14,31]
Pro shape @ band[14,31]   # 机制control，不冒充MrPro永久baseline
```

全部使用同一depth、gain、rows。先low，C42/BM中非支配者补mini；Pro shape只需
完成low以定位增量到达方向，除非修复独特task family。

目的：把“BM表现”拆成roughness shape与其默认band/depth，而不是继续比较名字。

### 13.6 Phase 4：checkpoint replay R0/R1（CPU并行，GPU捕获约3:00--4:15）

R0先在OLMo Native 4K完成：

- `rho=1,m=0` exact KL为0；
- runtime attention row与数学replay逐token对齐；
- causal sign、split-half、GQA、gain²一致；
- capture只保留选定queries及完整可见keys，不物化全L²。

R1在不改损失/权重的前提下回溯排序已知大效应：

- Llama同Solver shape/gain：`[14,32]`应优于`[18,35]`；
- OLMo同C42 shape/gain：`[14,31]`应优于`[16,34]`；
- Qwen用于retrospective检查，不算独立留出。

分支：

- replay方向在至少两个明确对照上正确：进入C1/C2/C3求解；
- 方向反复相反：否定当前Native-attention transport predictor，停止用它生成GPU
  候选；保留analytic depth和same-band transition真实实验；
- runtime parity失败：只修实现，不产生理论结论。

### 13.7 Phase 5：Llama S8 band与depth理论探针（4:15--6:15）

固定C42 shape、S8 gain、现有8/16/32/48/64K输入。已有`[14,32]`与`[16,34]`复用，
新增的band规则只跑：

```text
B_DN: [17,35]  # 同checkpoint跨S的Native-window phase budget预测
B_DH: [23,41]  # target-window phase预测的反例control
```

先用8K/64K低成本panel，非支配表再补中间长度。随后在当前强C42 `[14,32]`上只新增：

```text
Dlog: c=0.723308, slow a=2/9
```

若replay已给出不同标量`c_replay`，再加一张C2；否则不凭task结果调第二个depth。
Solver `[14,32]`保留为强任务reference，BM/MrPro不重跑。

### 13.8 Phase 6：完成Llama Core-6 mini（6:15--8:15）

冻结候选最多三张：

1. 当前强reference：Solver或C42 `[14,32], c=1`；
2. Phase 5的band代表；
3. tail-depth代表。

每张补到Core-6×8/32/64K×18行；BM/MrPro只补永久coverage缺口。报告任务family，
不因某个NIAH或PPL单项淘汰。

### 13.9 Phase 7：Qwen tail depth与same-band transition（8:15--9:30）

Qwen C0取Phase 1冻结的非支配band，优先`[22,39]`。新增：

```text
Dlog: c=0.584963, slow a=2/3
BM-shape@same-band,same-depth
```

先跑low108；已有完整mini结果直接作为c=1对照。若Dlog或BM-shape修复独特任务族，
才补其余216行；不扫`c`网格或band邻点。

### 13.10 Phase 8：C1/C2/C3 checkpoint候选（若R1通过，9:30--10:45）

以OLMo为首个便宜闭环：

1. C1：固定C0 shape/depth/gain，以finite-grid replay选band；
2. C2：C1 band上求一维depth；
3. C3：C1/C2上用full-Q active-set求transition；
4. exact circular replay/backtracking；
5. C3后重新检查band/depth stationarity；只做一次回查，不无限迭代。

C0/C1/C2/C3先跑同一low；最终depth上补`band×shape`2×2。replay降低而任务下降，
明确判为transport metric失败，不以更多solver步挽救。

### 13.11 Phase 9：prospective checkpoint或medium扩展（10:45--12:00）

二选一，顺序预先固定：

- 若三接口形成一条完整、未看目标模型结果的规则：冻结后在服务器已有Qwen约3B
  checkpoint上做prospective low；在冻结预测前不解封其band-specific旧结果；
- 若尚未形成统一规则：把今晚最重要的两个Pareto候选升到medium，不假装已有跨模型
  闭式解。

旧Qwen1.5已经被新理论引用，不能再作为新replay理论的独立leave-one-model-out。

---

## 14. 今晚结束时必须交付的内容

最低交付：

1. Qwen1.5 mini完整结果或明确的数据盘丢失回执；
2. 至少OLMo和一个较大模型上的首个tail-depth真实对照；
3. 至少OLMo上的same-band transition控制；
4. checkpoint replay R0 parity和R1方向判决；
5. 所有候选的完整表、gain、depth、band、shape、raw生成、EOS/cap和GPU成本；
6. 按Solid/Falsified/Pareto/Unresolved更新一份结果报告；
7. 下一步只由今晚新增证据决定，不按GPU剩余时间临时造候选。

理想交付：

- OLMo C0/C1/C2/C3低成本闭环；
- Llama、Qwen至少一个soft-depth候选进入mini；
- 一个真正prospective checkpoint预测已冻结并运行。

---

## 15. 希望外部推理模型重点解决的问题

请不要只评价现有文档是否“合理”。真正需要回答：

1. band、tail depth、transition是否是最合适的三个坐标，或应使用更一般的全谱对象；
2. 怎样从checkpoint本身推导一张固定表，而不是在目标benchmark上扫参数；
3. Native attention replay是否是合理预测器；若不是，最小需要加入hidden-state、
   V/O、任务margin或别的结构是什么；
4. `/S`与soft depth的真实取舍应由什么目标决定；
5. full-Q transition怎样在非凸circular目标、Native保护和worst-length下稳定求解；
6. 哪一组最小实验可以在一晚内最大幅度地区分竞争解释；
7. 成功或失败分别能形成怎样诚实、有分量的论文贡献。
