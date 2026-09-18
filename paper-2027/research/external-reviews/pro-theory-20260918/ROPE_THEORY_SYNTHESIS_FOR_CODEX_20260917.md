# RoPE 内部配置 z：服务现有论文的理论推导、统一解释与 Codex 落地报告

**仓库基线：** `misaya-yang/hybrid-rope@efd31c3b0e4f04b0d839497caa81c0e906374361`  
**研究对象：** 原生窗口与部署工作区间内的 RoPE 内部频率配置。  
**交付范围：** 数学推导、已有理论聚合、理论与已有证据的对应、论文源码中的定点整合、CPU 数学验证。  
**本轮实际执行：** 读取 GitHub 源码及理论记录；独立推导；CPU 有理数、合成注意力与高精度数值检查。未运行模型，未修改远端仓库。

---

## 0. 这份报告需要解决的理论问题

论文的原问题保持不变：位置编码不仅要支持更大的最大外推倍率，还应改善给定部署范围 `[1, sL]` 中的实际上下文质量，并探索原生窗口本身的增强。内部配置 `z` 是贯穿冻结部署、原生干预和学习的共同设计自由度。

本报告不以新的损失函数、性能代理或边界消融计划替换上述问题。要完成的是以下数学连接：

\[
\boxed{
\text{内部配置 }z
\longrightarrow \text{逐通道距离响应}
\longrightarrow \text{目标与干扰的注意力竞争}
\longrightarrow \text{已有模型结果的准确解释}
}
\]

其中，前两步有精确的算子关系；第三步有精确的固定状态恒等式和条件性结论；完整多层模型的任务收益，由已完成的匹配实验支持。不能把某一步的恒等式自动提升为下一步的普遍性能定理。

### 0.1 最值得整合的六个结论

1. **固定倍率不固定配置。** `m`、`ε` 不是论文主角 `z` 之外的新变量体系，而是相对于 Native 表表达 `z` 的两种坐标。
2. **TailSpline 改变的是整个过渡段的距离伸缩。** 相对 MrPro，它在每个内部过渡通道都有更大的累计伸缩，不是只修最后一个边界。
3. **尾端目标具有直接的相位意义。** 最后若干中频通道到完全插值参照的相位残差，可以精确比较；在最后一个内部通道，TailSpline 的残差比例甚至不超过已有的终端 gap 比例 `3/(2n+1)`。
4. **不同依赖伸缩存在不同偏好。** 对明确的相位对齐参照，T/P 的逐通道偏好在一个可解析的依赖伸缩阈值处反转。这把“工作区间”从口号落实为距离响应问题，但阈值不等于 benchmark 的长度分界。
5. **频率几何不足以决定任务方向。** 目标—干扰 log-odds、value 聚合与真实训练梯度说明，需要考虑内容相位、竞争权重和 value 的下游作用；完整 pair 的位置有效秩不能代替这些量。
6. **Native 与学习具有同一自由度、不同适应条件。** 固定 `z` 训练权重并不保证该 `z` 对冻结模型已经驻点；Cosh 的变分最优、NCP 的参考风险最优和 Transformer 任务最优必须分清。

### 0.2 数学身份约定

- **[已有]**：当前论文或仓库理论文件已包含该公式、证明或同等结论。
- **[推论]**：在本报告中从已有公式展开、便于论文解释的推论；不据此声称首次提出数学工具。
- **[标准]**：softmax、变分法、线性代数或微分优化的标准结果，本报告给出需要的证明。
- **[条件]**：只有在显式的参照、内容、距离、mask 或可微性条件下成立。

阅读范围是当前主稿相关章节、compact A/B/C/E/F 附录、核心构造代码和理论综合记录，不是逐行审计整个仓库。历史理论记录中的旧附录路径及实验待完成状态，以当前 `main.tex` 的活动输入和正式结果为准。[R1–R12]

---

## 1. 统一坐标：z、log-frequency 位移与累计增量

### 1.1 基本定义与实际自由度

令旋转维度为 `d_rot=2K`，频率严格递减且为正：

\[
\omega_0>\omega_1>\cdots>\omega_{K-1}>0.
\]

定义

\[
x_k=-\log\omega_k,
\qquad
x_k=a+Rz_k,
\qquad
z_0=0,\ z_{K-1}=1.
\tag{1}
\]

其中 `a=x_0`、`R=x_{K-1}-x_0>0`。固定 `a,R` 后，仍有 `K-2` 个内部配置坐标。

标准 RoPE 的 `ω_k=b^{-k/K}` 对应

\[
a=0,\quad R=\frac{K-1}{K}\log b,
\quad z_k=\frac{k}{K-1}.
\tag{2}
\]

所以在标准几何族中，改变 base 改变范围，却不会改变归一化内部配置。将 `z` 明确分离出来，是识别不同配置效果的准确坐标化。[已有；R1]

这里 `z` 不是额外增加推理时的可学习参数：它可以是固定表的描述坐标，也可以由解析构造或学习产生。固定排序的表默认包含原来的频率—旋转坐标指派；把同一频谱换到不同 learned coordinates，是另一种干预，不能只根据排序后的 `z` 区分它。

### 1.2 Native-relative 表与 z 的精确联系

设 Native 表为

\[
x_k^N=a+R_Nz_k^N.
\]

部署时使用

\[
\nu_k=\omega_k^Ns^{-m_k},\qquad s>1,
\quad m_0=0,\ m_{K-1}=1.
\tag{3}
\]

于是

\[
x_k'=x_k^N+(\log s)m_k,
\qquad R'=R_N+\log s,
\]

从而

\[
\boxed{
z_k'=\frac{R_Nz_k^N+(\log s)m_k}{R_N+\log s}.
}
\tag{4}
\]

**证明：** 对式 (3) 取负对数，减去不变的最快端，再除以新的 sampled log-span 即得。[已有；R4]

两个共享 Native、范围和倍率的方法 A/B 满足

\[
\boxed{
z_k^A-z_k^B=
\frac{\log s}{R_N+\log s}(m_k^A-m_k^B).
}
\tag{5}
\]

因此，T/P 是完整的内部配置比较；T/C 再固定总位移，回答该配置差异是否还包含超出一个总体统计量的作用。不能因为 T/P 的总位移不同，就称它“没有比较 z”。

还有一个重要的量纲区别：式 (5) 中的归一化差可以很小，但物理 log-frequency 差仍是

\[
\Delta x_k=(\log s)\Delta m_k,
\quad \nu_k^A/\nu_k^B=s^{-\Delta m_k}.
\tag{6}
\]

不能用 `Δz` 数值小直接认定相位干预小。

### 1.3 s=1 不是 Native 配置空间的退化

当前 TailSpline 构造器在 `s=1` 时返回精确 Native FP32 表和单位 gain，这是该扩展构造的恒等分支。[R7]

但原生配置可以直接写成

\[
x_k'=x_k^N+u_k,
\quad u_0=u_{K-1}=0,
\qquad z_k'=z_k^N+u_k/R_N.
\tag{7}
\]

只要新频率保持有序，非零内部 `u` 完全可行。

**结论：** `m log s` 在 `s=1` 上只表达恒等干预，不意味着 RoPE 在 Native 没有内部设计自由度。论文应区分扩展构造的恒等分支与完整的原生配置空间。

### 1.4 m 与 ε：伸缩量及其分配

对 band `[l,h]`，记 `n=h-l`、局部 `q=0,...,n`：

\[
m_0=0,\quad m_n=1,
\quad \epsilon_q=m_q-m_{q-1},\quad
\sum_{q=1}^n\epsilon_q=1.
\tag{8}
\]

几何 Native 网格的间隔 `c=log b/K` 给出

\[
\boxed{
x'_{l+q}-x'_{l+q-1}=c+(\log s)\epsilon_q.
}
\tag{9}
\]

`m` 是累计的 log-wavelength 伸缩；`ε` 是新增 log-span 在相邻频率间隔上的分配。固定单位增量和就固定了新增跨度，但没有固定分配形状。

令完整表的总位移为

\[
D=\sum_k(x_k'-x_k^N)=(\log s)\sum_km_k.
\]

局部求和交换给出

\[
\sum_{q=1}^{n-1}m_q
=\sum_{j=1}^{n}(n-j)\epsilon_j
=n-\sum_{j=1}^{n}j\epsilon_j.
\tag{10}
\]

所以同 band、同外带、同倍率下，**总位移与增量质心不是两个独立控制**。匹配前者已经匹配后者。[已有；R4、R7]

**论文落点：** 式 (4) 宜前移至 `sections/02_exponents.tex` 或 `04_mature.tex`；式 (10) 保留附录 C。不要为同一自由度引入新的宏大命名。

---

## 2. 从配置到距离响应：必须区分 s、H、d 和 α

### 2.1 四个量分别是什么

- `L`：声明的 Native/reference length。
- `s`：静态部署表的构造倍率，目标上限通常为 `sL`。
- `H`：一次输入的实际长度，未必等于 `sL`。
- `d`：一次注意力比较的相对 token 距离，因果输入中通常 `0≤d≤H-1`。
- 后文 `α=d/d_0`：相对于某个明确内容依赖参照 `d_0` 的伸缩倍数。

特别地，`α` 不是自动等于 `H/L`。增加无关文本、改变证据位置和真正拉伸语义依赖，是不同操作。

### 2.2 单通道的精确相位等效

式 (3) 意味着

\[
\boxed{
d\nu_k=\omega_k^N\frac{d}{s^{m_k}}.
}
\tag{11}
\]

定义波长扩展因子

\[
A_k=s^{m_k}.
\]

则相同旋转因子可以理解为：Native 频率在相位等效距离 `d/A_k` 上的响应。

该式只对旋转因子作恒等变换。它没有假设隐藏状态、token 内容、attention 分布或多层网络也回到了 Native 分布。

### 2.3 两侧频段保留两种距离参照

高频外带 `m=0`：

\[
R_{\nu_k}(d)=R_{\omega_k^N}(d).
\]

完全插值尾带 `m=1`：

\[
R_{\omega_k^N/s}(sd_0)=R_{\omega_k^N}(d_0).
\tag{12}
\]

中频过渡是在这两种参照之间分配不同程度的伸缩。

注意：这里保留的是**旋转相位参照**。当前扩展方法还使用共同 gain `g=1+0.1 log s`；与 Native 比较时，旋转 Q/K 的幅度也会改变。不能把高频相位未变写成高频完整 logit 贡献相对 Native 完全未变。T/P 的 gain 相同，所以它不混淆两者的频率差。[R2、R7–R8]

### 2.4 全工作区间为什么不能由一个最远长度代表

即使一个输入长度为 `H`，一次完整注意力计算也会涉及不同的 `d`、不同的内容相位及不同竞争键。对于同一张表，式 (11) 在每个 `d` 都成立，但收益方向取决于该比较需要保持哪一种响应。

因此，理论真正支持的是：**设计 `z` 会改变一组逐距离、逐内容的响应，而不是只改变一个最大可用长度数字。** 这为论文的区间视角提供了直接算子依据，不需要另定义一个未经验证的“区间最优分数”。

---

## 3. TailSpline 的有限网格解：完整证明及含义

### 3.1 目标确实平滑什么

定义

\[
J_T(\epsilon)=
\sum_{q=1}^{n-1}(\epsilon_{q+1}-\epsilon_q)^2+\epsilon_n^2,
\qquad \mathbf1^T\epsilon=1.
\tag{13}
\]

由式 (9)，`(log s)^2 J_T` 正好是过渡段内部相邻 log-gap 的变化平方和，加上向完全插值尾端的 gap 连接误差。高频外带固定，但入口额外 gap 未纳入惩罚。[已有；R2–R3]

这是一个具体的边界优先级。端点条件并不唯一推出它；自然边界条件也不能证明真实模型对入口不敏感。

### 3.2 正定性与 Green 矩阵

令 `Dε=(ε_{q+1}-ε_q)_{q=1}^{n-1}`，定义

\[
H_T=D^TD+e_ne_n^T.
\]

若 `v^T H_Tv=0`，所有相邻差为零且 `v_n=0`，故 `v=0`。所以 `H_T` 正定，约束问题有唯一解。

更明确地，采用从 1 开始的索引：

\[
\boxed{(H_T^{-1})_{ij}=n+1-\max(i,j).}
\tag{14}
\]

**证明：** 对固定列 `j`，右侧随 `i` 在 `i≤j` 时为常数，在 `i>j` 时每步减一。用 `H_T` 的首行 `[1,-1]`、内部二阶差分和末行 `[-1,2]` 相乘，只在第 `j` 个位置留下 1。`n=1` 时同样成立。

等式约束的一阶条件是 `H_Tε=λ_0 1`。因此

\[
\epsilon^T=
\frac{H_T^{-1}\mathbf1}{\mathbf1^TH_T^{-1}\mathbf1}.
\tag{15}
\]

对式 (14) 按行求和：

\[
(H_T^{-1}\mathbf1)_q=
\frac{(n+q)(n-q+1)}2,
\quad
\mathbf1^TH_T^{-1}\mathbf1=
\frac{n(n+1)(2n+1)}6.
\]

得到

\[
\boxed{
\epsilon_q^T=
\frac{3(n+q)(n-q+1)}{n(n+1)(2n+1)},
\qquad
T_q=m_q^T=
\frac{q(3n^2+3n+1-q^2)}{n(n+1)(2n+1)}.
}
\tag{16}
\]

每个增量均为正，因此非负性约束自动不活跃，累计表单调。`n=1` 时 `ε_1=1`，无需例外套用连续式。[闭式为已有；Green 写法为等价推导]

### 3.3 最优值和完整误差恒等式

\[
J_T(\epsilon^T)=\frac6{n(n+1)(2n+1)}.
\tag{17}
\]

任意同单位和配置 `ε=ε^T+v` 满足 `1^Tv=0`，所以

\[
\boxed{
J_T(\epsilon)-J_T(\epsilon^T)=v^TH_Tv\ge0.
}
\tag{18}
\]

交叉项为 `2v^TH_Tε^T=2λ_0v^T1=0`。这比只报数值 KKT 残差更完整地给出全局最优性，但最优性对象仍仅是式 (13)。

### 3.4 它怎样分配伸缩

\[
\epsilon_{q+1}^T-\epsilon_q^T
=-\frac{6q}{n(n+1)(2n+1)}<0.
\tag{19}
\]

MrPro 为

\[
P_q=\frac{q(q+1)}{n(n+1)},
\quad \epsilon_q^P=\frac{2q}{n(n+1)}.
\]

因此 MrPro 的增量递增，TailSpline 的增量递减。逐点差为

\[
\boxed{
T_q-P_q=
\frac{q(n-q)(3n+q+1)}{n(n+1)(2n+1)}>0
\quad(0<q<n).
}
\tag{20}
\]

于是

\[
\omega_k^N/s\le\nu_k^T\le\nu_k^P\le\omega_k^N,
\quad A_k^T/A_k^P=s^{T_q-P_q}.
\tag{21}
\]

这是**整个过渡段的伸缩再分配**，不是只有末端差异。对任意单调递增的索引函数 `f`，求和分部还给出

\[
\sum_q\epsilon_q^Pf(q)-\sum_q\epsilon_q^Tf(q)
=\sum_{q=1}^{n-1}(T_q-P_q)[f(q+1)-f(q)]\ge0.
\tag{22}
\]

即 TailSpline 把单位增量质量更早分配。该关系描述频率构造，不把索引早晚直接等同于任务价值。

相应的精确统计量为

\[
\sum_q q\epsilon_q^T=
\frac{(n+2)(3n+1)}{4(2n+1)},
\]

\[
\sum_{q=1}^{n-1}T_q=
\frac{(n-1)(5n+2)}{4(2n+1)},
\quad
\sum_{q=0}^{n}(T_q-P_q)=
\frac{(n-1)(7n+2)}{12(2n+1)}.
\tag{23}
\]

### 3.5 BM 和连续极限的准确位置

在式 (13) 增加 `ε_1²`，得到对称构造

\[
\epsilon_q^{BM}=\frac{6q(n-q+1)}{n(n+1)(n+2)}.
\tag{24}
\]

BM 最小化包含两个入口/出口的对称粗糙度；TailSpline 最小化优先尾端的粗糙度。它们的数学优劣不能脱离各自目标比较，更不能由目标最小值判模型分数。

固定归一化坐标 `u=q/n`，TailSpline 的连续极限为

\[
T(u)=\frac{3u-u^3}{2}.
\tag{25}
\]

它也解 `min ∫(m''(u))²du`，条件为 `m(0)=0,m(1)=1,m'(1)=0`，自然边界为 `m''(0)=0`。**实际模型安装使用式 (16)，不是用式 (25) 替换有限表。**

**论文落点：** 保留当前有限解定理。式 (14)–(18) 可使附录证明更简洁；式 (20)–(21) 直接接逐通道响应。不要把 Green 矩阵、概率排序各自包装成新贡献。

---

## 4. 尾端连接的更直接含义：相位残差的有限网格抑制

这是从现有有限式进一步展开、最值得用来解释 TailSpline 设计的推论。

### 4.1 从终端 gap 到最后一个内部通道

完全插值的频率参照是 `ω_k^N/s`。对任一过渡通道定义无量纲残差

\[
E_q^M=s\nu_{l+q}^M/\omega_{l+q}^N-1
=s^{1-m_q^M}-1,\quad M\in\{T,P\}.
\tag{26}
\]

它是相对于完全插值频率的相对残差。相同距离 `d>0` 上的未绕回相位误差为

\[
\delta_q^M(d)=
 d\left(\nu_{l+q}^M-\frac{\omega_{l+q}^N}{s}\right)
 =\frac{d\omega_{l+q}^N}{s}E_q^M.
\tag{27}
\]

在最后一个内部通道 `q=n-1`：

\[
1-T_{n-1}=\epsilon_n^T=
\frac6{(n+1)(2n+1)},
\quad
1-P_{n-1}=\epsilon_n^P=\frac2{n+1}.
\]

于是

\[
\boxed{
\frac{\delta_{n-1}^T(d)}{\delta_{n-1}^P(d)}
=
\frac{s^{\epsilon_n^T}-1}{s^{\epsilon_n^P}-1}
\le
\frac{\epsilon_n^T}{\epsilon_n^P}
=\frac3{2n+1}.
}
\tag{28}
\]

`n≥2,s>1,d>0` 时分母为正。`d=0` 时两者均为零，讨论其连续比值即可。

**证明：** 对 `t>0`，函数 `(exp(t)-1)/t` 严格递增；再用 `0<ε_n^T≤ε_n^P`。共同的 `dω/s` 因子消去。

因此，当前论文的 terminal-gap 比例不仅是曲线光滑程度的指标：它也给出了**最后一个中频通道相对于尾端距离参照的相位残差比例上界**。

### 4.2 有限 band 的实际数值

数学网格、`s=4`：

| band 宽度 n | terminal log-gap 比例 `3/(2n+1)` | 最后内部通道相位残差比例 |
|---:|---:|---:|
| 13（当前 GLM 网格） | 0.11111111 | 0.10158527 |
| 17（当前 Llama/Qwen 网格） | 0.08571429 | 0.07980797 |
| 18（当前 OLMo 网格） | 0.08108108 | 0.07575629 |

这些是公共参数和既有构造的解析计算，不是模型观测，也不是效果拟合。比例不依赖该通道的绝对 `ω` 或比较距离 `d`；绝对误差大小仍依赖两者。

### 4.3 不只是最后一个点：低频侧的有限层结构

取离尾端 `j` 个间隔的内部通道 `q=n-j`，`1≤j≤n-1`。由式 (16) 精确相减：

\[
1-T_{n-j}=
\frac{j(j+1)(3n+1-j)}{n(n+1)(2n+1)},
\]

\[
1-P_{n-j}=
\frac{j(2n+1-j)}{n(n+1)}.
\tag{29}
\]

因此

\[
\boxed{
\frac{E_{n-j}^T}{E_{n-j}^P}
\le
\frac{1-T_{n-j}}{1-P_{n-j}}
=
\frac{(j+1)(3n+1-j)}{(2n+1)(2n+1-j)}.
}
\tag{30}
\]

固定 `j`、`n→∞` 时，TailSpline 的累计尾端残差为 `O(n^{-2})`，MrPro 为 `O(n^{-1})`。相应残差比为 `O(n^{-1})`。这说明尾端连接优先级如何作用于一个邻域，而不是一个孤立端点。

在连续极限，令 `v=1-u`：

\[
1-T(1-v)=\frac32v^2-\frac12v^3,
\quad 1-P(1-v)=2v-v^2.
\tag{31}
\]

前者二阶接近完全插值累计量，后者一阶接近。连续导数结论不能替代有限式 (29) 的离散误差项。

### 4.4 这个推论解释什么，不解释什么

它证明：**TailSpline 选定的尾端目标，确实降低了低频侧过渡通道对伸缩参照的相位错配。** 不是把任意光滑量与任务分数强行相连。

相位误差变小是否让某个内容贡献增加，还取决于其相位、目标与干扰的竞争，以及周期的所在分支。对于 `2|sin(δ/2)|` 这样的实际旋转误差，不能跨任意多周保持未绕回误差排序。第 5、9 节给出准确连接条件。

**建议入稿：** 主文一句解释，附录 B 一条推论加证明；不需要新增算法或 GPU 对照。式 (28) 比再增加一个纯粗糙度统计量更直接服务“设计为何合理”。

---

## 5. 从最远端扩展到区间：依赖伸缩的精确交叉条件

### 5.1 一个明确而有限的内容参照

设同一个内容分量在 Native 的依赖距离 `d_0>0` 对齐，其频率 `ω>0`，幅度 `A>0`：

\[
f_M(\alpha)=A\cos[\omega d_0(\alpha s^{-m_M}-1)].
\tag{32}
\]

这里实际依赖距离为 `d=αd_0`，括号内是相对于原有内容相位的残差；它不假设输入长度整体都按 `α` 缩放。[条件；与 R3 的指定内容例子相容]

记

\[
a_T=s^{-T_q},\qquad a_P=s^{-P_q},
\quad 0<a_T<a_P\le1.
\]

去掉共同的 `(ωd_0)²`，两种未绕回相位误差平方差是

\[
\boxed{
(\alpha a_T-1)^2-(\alpha a_P-1)^2
=
\alpha(a_T-a_P)\,[\alpha(a_T+a_P)-2].
}
\tag{33}
\]

### 5.2 逐通道伸缩阈值

由式 (33)：

\[
\boxed{
\alpha_q^*=\frac2{s^{-T_q}+s^{-P_q}}
=\frac{2A_q^TA_q^P}{A_q^T+A_q^P}.
}
\tag{34}
\]

它是两种波长扩展因子的调和平均，处于二者之间，内部通道严格满足 `1<α_q^*<s`。

- `α>α_q^*`：TailSpline 对该参照的相位误差更小。
- `α<α_q^*`：MrPro 对该参照的相位误差更小。
- `α=α_q^*`：误差大小相同。

若两个残余相位的绝对值均不超过 `π`，余弦在绝对相位上的单调性使该排序进一步成为式 (32) 的内容响应排序；若越过此条件，只保留平方残差结论，不能保证响应排序。

### 5.3 该结论怎样服务实际的区间问题

对 `n=17,s=4`：

| q | `T_q` | `P_q` | `A_T` | `A_P` | `α_q^*` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.085714 | 0.006536 | 1.126173 | 1.009102 | 1.064428 |
| 4 | 0.337255 | 0.065359 | 1.596054 | 1.094839 | 1.298768 |
| 8 | 0.638655 | 0.235294 | 2.423868 | 1.385674 | 1.763304 |
| 12 | 0.868347 | 0.509804 | 3.332707 | 2.027368 | 2.521093 |
| 16 | 0.990476 | 0.888889 | 3.947536 | 3.428976 | 3.670029 |

这些阈值随 `q` 递增，因为 `T_q,P_q` 均递增，式 (34) 的分母随之递减。它们描述一个有序的通道响应变化，而不是单一“最大长度”。

例如第 8 个过渡位置：若某个明确依赖由 `d_0` 拉伸到 `2d_0`，则 `2>1.7633`，TailSpline 的相位残差更小。在同一内容参照未经拉伸的 `α=1` 下，MrPro 反而更近。这不是证明 Native 必然变差，因为真实 Native 内容不必服从式 (32) 的单一对齐假设，模型还有其他通道、键和层。

**最重要的限制：** 不能把表中的 `α_q^*` 标为“16K/32K benchmark 的理论转折点”。知道 `H` 并不知道实际每个任务的 `d_0`、`α` 和内容相位。论文应将它作为解释区间依赖的条件分析，而不是由结果反推语义距离。

### 5.4 多通道参考误差也有精确交叉，但不要把它当作新选表目标

给定非负固定权重 `w_q`，且至少一个不同通道的权重为正，定义

\[
\mathcal E_M(\alpha)=\sum_qw_q(\alpha a_{M,q}-1)^2.
\]

将式 (33) 相加得到

\[
\boxed{
\bar\alpha^*=
\frac{2\sum_qw_q(a_{P,q}-a_{T,q})}
{\sum_qw_q(a_{P,q}^2-a_{T,q}^2)}.
}
\tag{35}
\]

`α>barα*` 时 `E_T<E_P`。式 (35) 可以写成各 `α_q^*` 的正权重平均，因此介于它们的最小值和最大值之间。

`w_q` 可以在某个声明的参考模型中吸收幅度或 `(ω_qd_0)²`，但这些权重不是当前公共配置规则天然知道的量。该式用来理解为什么不同内容组成可以具有不同偏好，不用于从任务输出拟合一张新表。

**论文落点：** 附录 B 的已有两参照分析可增加式 (33)–(34)，主文只保留一句“配置改变的是一族距离响应而非单一长度”。避免为了增加理论数量把整张阈值表塞进正文。

---

## 6. 固定倍率内的质量，与继续增大倍率，是两个不同导数

### 6.1 固定实际距离时

在 `m_k` 固定、band 不随 `s` 改变的条件下：

\[
\frac{\partial\log A_k}{\partial\log s}=m_k,
\qquad
\frac{\partial\log(d\nu_k)}{\partial\log s}=-m_k
\quad(d>0\text{ 固定}).
\tag{36}
\]

增大 `s` 会在相同的真实距离上减慢相位。这不等价于当目标距离同时变远时，相位也一定变小。

### 6.2 随目标上限移动时

取 `d=r sL`，固定 `r>0`、`L`，则

\[
d\nu_k=rL\omega_k^Ns^{1-m_k},
\qquad
\boxed{
\frac{\partial\log(d\nu_k)}{\partial\log s}=1-m_k.
}
\tag{37}
\]

对真正内部通道 `0<m_k<1`，目标距离上的相位仍随 `s` 增大；完全插值尾端 `m=1` 才恰好保持不变。由于 `T_q>P_q`，TailSpline 的该增长指数小于 MrPro。

这是一个有意义的结构性结论：**TailSpline 让更多过渡通道承担伸缩，因此在目标距离也增长时，通道相位增长得更慢。** 它不证明这些通道的完整任务效果永远提高，也不意味着无限增大 `s` 后所有相位仍处在有限训练区间。

`r sL` 是连续尺度分析；有限输入的最大整数距离为 `H-1`，数值复核时应使用真实整数而不是悄悄把长度当最大 lag。

### 6.3 与 YaRN 的精确算子比较

在固定的 index-ramp 坐标 `0<t<1` 上，官方 YaRN 方程为

\[
\nu_Y=\omega(1-t+t/s),
\quad A_Y=(1-t+t/s)^{-1},
\]

\[
\frac{\partial\log A_Y}{\partial\log s}
=\frac{t}{s(1-t)+t},
\quad \lim_{s\to\infty}A_Y=\frac1{1-t}.
\tag{38}
\]

因此对随目标增长的 `d=r sL`：

\[
d\nu_Y=rL\omega[(1-t)s+t].
\tag{39}
\]

YaRN 的固定内部混合权重对应线性增长的远端相位；固定指数规则为 `s^{1-m}`。这解释了二者不同的倍率响应，而不是一次未经运行的任务对比。[已有尺度公式的进一步解释；R3、R8、E2–E3]

必须保留实现身份：官方 YaRN 的 correction rounding 与 TailSpline 的 canonical band 不必逐槽相同。式 (38)–(39) 是固定混合权重下的算子比较；真实基线继续用已运行的官方实现。仓库 legacy `schedules.py` 中名为 `yarn` 的旧方法不能代替 `scripts/lib/rope/official_yarn.py`。[R8]

### 6.4 为什么一个简单“双距离平方损失”不能反过来推导 TailSpline

若尝试以同一权重平衡 Native 与 `s` 倍距离：

\[
J_k(r)=\omega_k^2[A(r-1)^2+B(sr-1)^2],
\quad A,B\ge0,
\]

则

\[
r_*=(A+sB)/(A+s^2B),
\tag{40}
\]

与 `k` 无关。它给出统一缩放，而不是三段式，更不是 TailSpline。

所以本报告的相位比较不能被倒写成“我们从一般最优区间损失推导出了三段结构”。32/1 边界、外带参照和尾端优先级各自有来源；承认构造先验不削弱其闭式正确性或模型效果。[已有反例；R10]

**论文落点：** 方法段短述式 (37)，附录保留 YaRN 的尺度响应；式 (40) 留在理论内部记录，用于避免下一轮再制造不成立的统一最优理论。

---

## 7. 不同 base、Native 长度和旋转维数：哪些量可以跨模型迁移

### 7.1 绝对频率不是唯一合适的比较坐标

相位可以写成

\[
d\nu_k=(d/L)(L\omega_k^N)s^{-m_k}.
\tag{41}
\]

其中 `Lω` 是无量纲的 Native 相位尺度。32/1 turn 规则依据的就是 `Lω/(2π)`，并非固定绝对频率。

对几何网格 `c=log b/K`，`l` 为最后一个超过 32 turns 的槽，`h` 为第一个低于 1 turn 的槽，`n=h-l`。假设两端都存在且未裁剪，定义

\[
\eta_+=\log\frac{L\omega_l}{64\pi},
\quad
\eta_-=\log\frac{2\pi}{L\omega_h},
\quad 0<\eta_\pm\le c.
\]

由 `ω_h=ω_l exp(-nc)`：

\[
nc=\log32+\eta_++\eta_-.
\tag{42}
\]

再令 `u=q/n`：

\[
\boxed{
L\omega_{l+q}=64\pi\,32^{-u}
\exp[(1-u)\eta_+-u\eta_-].
}
\tag{43}
\]

这表明，turn-selected band 的主要相位范围具有共享结构；base、K、L 的差异还通过取样宽度 `n`、取整误差和实际频率—内容指派体现。[已有；R3]

### 7.2 不能直接说“大 base 抑制 z 收益”

在固定槽位比较中，增大 base 往往降低该槽的绝对频率；但在 turn-selected 比较中，band 本身会移动。式 (43) 说明二者不是同一个干预。

同时，归一化式 (5) 的系数会随着 `R_N` 增大而变小，但物理的 `Δx=(log s)Δm` 不能因此被忽略。规范化坐标的缩小不是模型作用变弱的定理。

当前数学网格为：

| 模型 | K | base | L | band | n |
|---|---:|---:|---:|---|---:|
| Llama | 64 | 500000 | 8192 | [18,35] | 17 |
| OLMo | 64 | 500000 | 4096 | [14,32] | 18 |
| Qwen | 64 | 1000000 | 32768 | [23,40] | 17 |
| GLM partial RoPE | 32 | 10000 | 32768 | [17,30] | 13 |

同一公共规则的可迁移性，来自相位归一化和有限网格构造，而不是各模型的频率数值相同。**任务收益是否迁移仍由跨模型实验回答。**

只有 `c→0` 才是该公式的细网格极限；固定 `K` 而增大 base 会增大 `c`，不能叫连续近似更准确。

### 7.3 当前构造器的有效域必须保留

`allocation_design.py` 要求存在非空 32/1 过渡，否则报错。理论证明不能让 Codex 为不存在的边界静默补零、裁剪后继续声称同一规则。[R7]

对非几何 Native 表，`m` 的有限构造仍可定义，但式 (9) 的常数 `c` 应改成逐槽原始间隔；“目标恰为总 log-gap 曲率”的表述需要重新区分 Native 间隔与额外间隔。当前几何网格证据不能被悄悄推广成任意不规则 Native 表的等价定理。

**论文落点：** 附录 B 保留式 (43)；正文用一句说明公共 band 以 Native turns 为尺度。无需声称所有 base 下增益相等。

---

## 8. 等位移为什么仍能有形状效果：精确方向与路径公式

### 8.1 现有 C 的完整身份

令

\[
U_q=q/n,\quad F_q=2U_q-P_q,
\quad w_n=\frac{3n}{2(2n+1)}.
\]

已有控制为

\[
C_q=(1-w_n)U_q+w_nF_q,
\quad T_q=(1-w_n)BM_q+w_nF_q.
\]

因此

\[
\boxed{
\delta_q:=T_q-C_q
=\frac{q(n-q)(2q-n)}{2n(n+1)(2n+1)}.
}
\tag{44}
\]

它满足

\[
\delta_0=\delta_n=0,\quad
\delta_{n-q}=-\delta_q,\quad
\sum_q\delta_q=0.
\tag{45}
\]

`n=1,2` 时 `T=C`；`n≥3` 才出现非平凡对照。[已有；R3、R7]

T/P 的累计差单向为正；T/C 的差则在 band 中部换号。前者包含更多累计伸缩，后者在同总量下交换内部形状。二者回答的问题不同，均属于配置研究。

### 8.2 等总量没有取消相位变化

对同一槽：

\[
\nu_k^T-\nu_k^C=
\nu_k^C[\exp(-\delta_k\log s)-1].
\tag{46}
\]

相位差为 `d(ν_T-ν_C)`。即使 `Σδ=0`，逐槽差仍非零，且不同槽乘以不同 Native 频率和内容系数。总位移相消不能变成 logit 相消。

已有 Llama `s=4` 数学网格在 32K 的最大 T/C 未绕回相位差约 10.6725 rad，因此不能默认采用一阶小扰动解释该完整对照。[R3]

### 8.3 可微效用的精确有限路径

若 `V(m)` 在 T/C 连线邻域可微，定义

\[
m(t)=C+t(T-C),\quad
G_q=\int_0^1\partial_{m_q}V(m(t))\,dt.
\]

由微积分基本定理：

\[
V(T)-V(C)=\sum_q\delta_qG_q.
\]

利用式 (45)：

\[
\boxed{
V(T)-V(C)=
\sum_{q<n/2}(-\delta_q)(G_{n-q}-G_q).
}
\tag{47}
\]

所以等总位移下的收益对应的是：**沿整个路径，低频侧增加伸缩的边际价值，是否高于高频侧被减少伸缩的边际价值。** 这是真正的形状方向分析，而不是一个总剂量指标。

`G` 是解释量，当前构造不读取它来拟合表。公式适用于可微 NLL、固定状态注意力效用等；不直接适用于 greedy exact match、离散 F1。原有任务实验可以显示形状效应存在，但不能由此声称已测到所有 `G_q`。

**论文落点：** 等位移控制保留在识别/实验部分。式 (47) 作为可选附录解释；不以获取梯度或新增边界臂作为改稿条件。

---

## 9. z 怎样改变注意力：有限变化、梯度、value 与曲率

### 9.1 从完整旋转对出发，统一符号

固定某层某个 query 的 pre-RoPE Q/K 和 valid-key 集合，使用

\[
\ell_j(x)=b_j+
\operatorname{Re}\sum_k C_{jk}\exp(i d_j\nu_k),
\quad \nu_k=e^{-x_k}.
\tag{48}
\]

`C_jk` 包含完整 pair 的内容相位、幅度、共同 attention normalization 与固定 gain；`b_j` 可以包含不旋转子空间的贡献。

选择相反的 lag 或旋转符号时，`C` 的相位定义需同步变换。只要统一从式 (48) 推导，下列符号不会含混。

有限表变化的 logit 差为

\[
\eta_j=\operatorname{Re}\sum_k C_{jk}
(e^{id_j\nu_k'}-e^{id_j\nu_k}).
\tag{49}
\]

这是固定状态下的精确式，没有线性化误差。[已有；R5]

### 9.2 精确指数重加权与目标竞争

令 `p=softmax(ℓ)`、`p'=softmax(ℓ+η)`。则

\[
\boxed{p_j'=\frac{p_j e^{\eta_j}}{\mathbb E_p e^\eta}.}
\tag{50}
\]

对非空的目标键组 `G` 和竞争键组 `B`，两组分割有效键，分别以原 softmax 归一化为 `p_G,p_B`：

\[
\mathcal O=\log\sum_{j\in G}e^{\ell_j}-
\log\sum_{j\in B}e^{\ell_j}.
\]

于是

\[
\boxed{
\Delta\mathcal O=
\log\mathbb E_{p_G}e^{\eta_j}
-\log\mathbb E_{p_B}e^{\eta_j}.
}
\tag{51}
\]

证明只需从更新后的两组配分函数分别提出旧配分函数。`min_G η>max_B η` 是目标 attention share 严格上升的充分条件。[已有；R5]

这比“目标 logit 变大”准确。目标和干扰同时增加同一个常数时，softmax 完全不变；目标 logit 变大但干扰变得更多时，目标份额可能下降。

### 9.3 value 聚合给出下一层联系

固定 value `v_j`，记 `o=Σp_jv_j`、`o'=Σp'_jv_j`。由式 (50)：

\[
\boxed{
o'-o=
\frac{\operatorname{Cov}_p(v,e^\eta)}{\mathbb E_p e^\eta}
=
\frac{\sum_jp_j(v_j-o)e^{\eta_j}}{\mathbb E_p e^\eta}.
}
\tag{52}
\]

因此，即使某个目标组获得更多概率，最终输出作用仍取决于被重加权的 value 内容及其下游方向。频率不直接编码“正确答案”；它改变已有内容比较的距离依赖及聚合方式。[标准；仓库历史理论亦已整理，R10]

### 9.4 可用于区间分析的有限扰动界

旋转算子满足

\[
\|R(d\nu')-R(d\nu)\|_2
=2|\sin(d(\nu'-\nu)/2)|
\le\min\{2,|d(\nu'-\nu)|\}.
\]

从而

\[
|\eta_j|\le
\sum_k|C_{jk}|\min\{2,|d_j\Delta\nu_k|\}=:B_j.
\tag{53}
\]

若一个固定状态读的所有有效距离均不超过 `H-1`，可用 `H-1` 替换右侧距离得到该读上的共同上界；它仍含内容幅度，而不是只依赖 `s`。

设 logit 改变量的振幅 `r=max_jη_j-min_jη_j`，则

\[
\boxed{
\operatorname{TV}(p,p')\le\tanh(r/4),
\qquad
D_{KL}(p\|p')\le r^2/8.
}
\tag{54}
\]

**TV 界证明。** 令 `w=e^η∈[a,b]`、`Z=E_pw`。凸性给出

\[
\mathbb E|w-Z|\le\frac{2(b-Z)(Z-a)}{b-a}.
\]

所以 `TV≤(b-Z)(Z-a)/[Z(b-a)]`。在 `Z=√(ab)` 最大，得到 `(√b-√a)/(√b+√a)=tanh(r/4)`。两点分布可以取等号。

**KL 界证明。** 令 `K(t)=log E_p e^{tη}`。`K''(t)=Var_{p_t}(η)≤r²/4`，而

\[
D_{KL}(p\|p')=K(1)-K(0)-K'(0)
=\int_0^1(1-t)K''(t)dt\le r^2/8.
\]

若 value 集合在所用范数下直径为 `D_v`，正负概率差各有总质量 TV，故

\[
\|o'-o\|\le D_v\operatorname{TV}(p,p')
\le D_v\tanh(r/4).
\tag{55}
\]

这些界回答“给定状态和频率变化，注意力响应最多变化多少”。它们不判收益符号，不是 Native 无损或全模型鲁棒性证书。将全长换成最大距离只是单次读的上界；要推广到整网需要各层状态和误差传播控制。

### 9.5 z/m 的局部 Jacobian

从式 (48) 直接求导：

\[
J_{jk}:=\frac{\partial\ell_j}{\partial x_k}
=d_j\nu_k\operatorname{Im}(C_{jk}e^{id_j\nu_k}).
\tag{56}
\]

因此在固定支持下

\[
\frac{\partial\ell_j}{\partial z_k}=R J_{jk},
\qquad
\frac{\partial\ell_j}{\partial m_k}=(\log s)J_{jk}.
\tag{57}
\]

局部二阶导数仅在同频率索引上非零：

\[
B_{jk}:=\frac{\partial^2\ell_j}{\partial x_k^2}
=-d_j\nu_k\operatorname{Im}(C_{jk}e^{id_j\nu_k})
-(d_j\nu_k)^2\operatorname{Re}(C_{jk}e^{id_j\nu_k}).
\tag{58}
\]

以上的“非零仅对角”限定于固定 Q/K、固定 gain 的单次 logit。多层网络的完整 Hessian 不具有这个简化结构。

### 9.6 Fisher pullback 与真实损失 Hessian 必须分开

令

\[
F_p=\operatorname{diag}(p)-pp^T.
\]

对小的 log-frequency 变化 `v`：

\[
D_{KL}(p(x)\|p(x+v))
=\frac12v^T J^TF_pJv+O(\|v\|^3).
\tag{59}
\]

`J^T F_p J` 是该 attention 分布的局部 Fisher/广义 Gauss–Newton 度量。它描述局部响应敏感性、消去共同 logit 平移，但不能自动称为真实任务 Hessian。

例如固定目标分布 `y` 的 attention cross-entropy：

\[
\boxed{
\nabla_x^2[-y^T\log p]
=J^TF_pJ+\operatorname{diag}\left(\sum_j(p_j-y_j)B_{jk}\right)_{k}.
}
\tag{60}
\]

真实模型输出 NLL 还涉及 attention 之后的网络，不应直接等同此处的 attention-label cross-entropy。

**Codex 核查规则：** 历史文件若把 `J^T FJ` 写成完整任务 Hessian，应检查该段到底在定义 KL 的局部度量还是任务目标。只修发生混同的活动表述，不把历史分析目录中的不同定义当作主稿已出错。

### 9.7 全网络频率梯度：内容、竞争和下游作用同时出现

在一个 attention read 中，设 `g_o=∂ℒ/∂o`，则

\[
\frac{\partial\mathcal L}{\partial\ell_j}
=p_j g_o^T(v_j-o).
\tag{61}
\]

共享的 `x_k` 在所有层、head、query 的旋转操作中使用，正确反向传播为对所有直接使用位置求和：

\[
\boxed{
\frac{\partial\mathcal L}{\partial x_k}
=\sum_{\text{所有读}}\sum_j
p_j\,g_o^T(v_j-o)\,
 d_j\nu_k\operatorname{Im}(C_{jk}e^{id_j\nu_k}).
}
\tag{62}
\]

每个局部导数固定该读输入，反向传播的上游伴随 `g_o` 已包含后续层效应；不要再在右侧人工加一套重复的“隐藏状态间接项”。标准输出投影可吸收到 `g_o` 中。

该表达显示频率收益方向依赖距离、Q/K 相位和幅度、softmax 竞争以及 value 的下游方向。LeRoPE 已给出相关的频率梯度及按距离聚合的相位形式，本文应引用并衔接，而非声称首次推导这类梯度。[E4]

式 (62) 用于解释 `z` 的作用，不是授权从权重、激活或梯度拟合当前静态公共构造。

**论文落点：** 正文应优先保留式 (48) 的直观解释和式 (51) 的竞争意义。式 (52)–(62) 可在附录 A 按需要合并，避免把标准工具拆成多个新主定理。

---

## 10. 有限窗口的位置几何：保留它的价值，也划清它与内容能力的对象差异

### 10.1 为什么必须使用完整 sine–cosine pair

固定内容比较的单频贡献总是

\[
C\cos(\omega d)+D\sin(\omega d),
\]

因此位置函数对象是

\[
V_\omega=\operatorname{span}\{\cos(\omega d),\sin(\omega d)\}.
\tag{63}
\]

仅比较 cosine 会任意偏向内容相位，并遗漏 sine–cosine 交叉项。当前论文的反例已经证明 cosine-only 排序可以与完整 pair 的位置有效秩排序相反。[已有；R5–R6]

### 10.2 Cross-Gram 的闭式与不变性

设 `d~Unif[0,H]`，`a(t)=sin(t)/t`、`b_*(t)=(1-cos t)/t`，奇点取连续延拓。定义 `d_-=(ω-ν)H`、`d_+=(ω+ν)H`：

\[
H_{\omega\nu}=
\frac12
\begin{bmatrix}
a(d_-)+a(d_+) & b_*(d_+)-b_*(d_-)\\
b_*(d_+)+b_*(d_-) & a(d_-)-a(d_+)
\end{bmatrix}.
\tag{64}
\]

令 `S_ω=H_{ωω}`、`Q_ων=S_ω^{-1/2}H_ωνS_ν^{-1/2}`，假设 self-Gram 非奇异。`Q` 的奇异值是两个二维位置函数子空间的主角度余弦。

\[
c_{\omega\nu}=\tfrac12\|Q_{\omega\nu}\|_F^2\in[0,1].
\tag{65}
\]

若分别变换 pair 内基底 `x_j→x_j A_j`，则白化后的 `Q` 只发生左右正交变换，`c` 不变。这保留了相位选择无关性。[已有；主角度工具为标准]

### 10.3 完整 pair 的有效秩恒等式

拼接白化后的 pair，记完整 Gram 为 `Γ`。其对角块为 `I_2`。令

\[
\bar c=\frac1{K(K-1)}\sum_{i\ne j}c_{ij}.
\]

则

\[
\operatorname{tr}\Gamma=2K,
\quad
\operatorname{tr}(\Gamma^2)=2K[1+(K-1)\bar c],
\]

\[
\boxed{r_2(\Gamma)=\frac{2K}{1+(K-1)\bar c}.}
\tag{66}
\]

该式精确描述提供的位置函数方向重叠。它不是模型内容容量的等式，也不单独给出最优频率表。[已有；R5–R6]

### 10.4 慢频极限的完整展开

令 `t=d/H`、`x=ωH`，取稳定基底

\[
[\cos(xt),\sin(xt)/x]
=[1,t]+x^2[-t^2/2,-t^3/6]+O(x^4).
\tag{67}
\]

所以 `V_ω` 趋向 `span{1,t}`。这不等于在 `ω=0` 直接把 `[cos0,sin0]` 当成二维；极限二维空间使用了合法的非零频率重标度。

在 `L_2[0,1]` 中投影掉 `span{1,t}`，记

\[
p_2=t^2-t+1/6,
\qquad p_3=t^3-9t/10+1/5.
\]

基础 Gram 与横向扰动 Gram 为

\[
G_0=\begin{bmatrix}1&1/2\\1/2&1/3\end{bmatrix},
\quad
M=\begin{bmatrix}1/720&1/1440\\1/1440&1/2800\end{bmatrix}.
\]

`tr(G_0^{-1}M)=19/12600`，从而对 `ε=max(|x|,|y|)→0`：

\[
\boxed{
2-\|Q_{x,y}\|_F^2
=\frac{19}{12600}(x^2-y^2)^2+O(\epsilon^6).
}
\tag{68}
\]

这是当前论文已有的更细渐近式，值得保留其条件和系数，不需要重新构造另一个“collision”定义。[R5]

### 10.5 白化移除了能量，因此 rank 不等于有效信号幅度

均匀距离测度下，原始单 pair self-Gram 的两个特征值恰为

\[
\lambda_\pm(S_\omega)
=\frac{1\pm|\sin x/x|}{2},\quad x=\omega H.
\tag{69}
\]

因为 trace 为 1，而对应的二倍角平均复振幅模为 `|sin x/x|`。在慢频极限：

\[
\lambda_-\sim x^2/12.
\]

白化将弱方向也变为单位能量。若再先去均值，sine 主方向的能量为 `x²/12+O(x⁴)`，去掉其线性相关部分后的 cosine 弱方向为 `x⁴/720+O(x⁶)`。

因此，相同的白化子空间重叠不意味着相同的实际 Q/K 内容幅度和可用 logit 变化。该差异是对象不同，不是有效秩分析“没有价值”。

### 10.6 softmax 中心化进一步改变了相关的几何对象

由于 `F_p 1=0`，共同常数 logit 方向被消去。对固定 `p`，去均值后

\[
\frac{\sin(\omega d)-\mathbb E_p\sin(\omega d)}{\omega}
\to d-\mathbb E_pd,
\]

\[
-\frac{2[\cos(\omega d)-\mathbb E_p\cos(\omega d)]}{\omega^2}
\to d^2-\mathbb E_pd^2.
\tag{70}
\]

当距离支持至少包含三个适当不同点时，两条中心化多项式方向独立。原始位置 Gram、softmax 度量、实际下游 Hessian 是三个相关但不相同的对象。

### 10.7 位置函数趋同没有删除内容坐标

对 `r` 个旋转 pair，块旋转 `R_B(d)` 对每个距离始终正交且秩为 `2r`。若 `η=H max_B|ω_k|`，则

\[
|q_B^TR_B(d)k_B-q_B^Tk_B|
\le\eta\|q_B\|\|k_B\|,\quad |d|\le H.
\tag{71}
\]

令 `2r` 个内容符号分别用不同单位坐标表示，query 与正确 key 同向，其他 key 与之正交；未旋转 margin 为 1，旋转后 margin 至少 `1-2η`。因此即使多个慢频的位置子空间接近同一个二维空间，仍可以保留 `2r` 个独立内容坐标的比较。[已有；R5]

该例要求小相位，不能套到实际相位已经数 rad 的完整 TailSpline 中频上。它证明的是“位置冗余≠内容坐标被删除”，不是完整冻结任务的获益机制已被定位。

### 10.8 与已有真实结果的正确连接

当前 Llama T/P 在 16K、32K 上都是 TailSpline 的 full-pair 有效秩较低、任务成绩较高。该现象排除了“有效秩必须上升才可能改善任务”的必要条件，却不支持把降低 rank 反过来当选表原则。[R5、R11]

**论文落点：** 保留当前完整 pair、慢频集中和内容保留的逻辑；正文用一个具体对照连接真实模型即可。式 (69)–(70) 如增加，只放附录解释度量边界，不再新增一个主性能指标。

---

## 11. 有限距离测度：为什么全区间、稀疏读取和单端点不是同一个理论问题

### 11.1 对任意距离分布的精确旋转差异

给定一个声明的距离概率测度 `μ`，定义

\[
\rho_\mu(\delta)=
\mathbb E_{d\sim\mu}\,4\sin^2(d\delta/2).
\]

其特征函数 `φ_μ(δ)=E_μ exp(idδ)` 给出

\[
\boxed{\rho_\mu(\delta)=2-2\operatorname{Re}\varphi_\mu(\delta).}
\tag{72}
\]

若距离有界，则

\[
\rho_\mu(\delta)
=\mathbb E_\mu d^2\,\delta^2
-\frac{\mathbb E_\mu d^4}{12}\delta^4+O(\delta^6),
\quad
\rho_\mu(\delta)\le\mathbb E_\mu d^2\,\delta^2.
\tag{73}
\]

这既提供有限变化，也解释为什么相同频率变化在不同距离组成下不应被赋予相同的代理代价。

### 11.2 三个容易混用的距离分布

令整数输入长度为 `H`、`d=0,...,H-1`：

| 抽样对象 | 距离权重 | 二阶矩 |
|---|---|---|
| 距离值本身等权 | `1/H` | `(H-1)(2H-1)/6` |
| 所有因果位置对 `i≥j` 等权，含对角 | `2(H-d)/[H(H+1)]` | `H(H-1)/6` |
| 两个独立均匀位置的绝对距离 | `p_0=1/H; p_d=2(H-d)/H², d>0` | `(H²-1)/6` |

后二者连续极限都是三角权重 `2(1-t)`，但有限 `H` 下并不相同。更重要的是，它们都不是自动等于真实任务的证据距离分布或 attention 权重。

均匀 lag 的精确式为

\[
\rho_H(\delta)=2-
2\frac{\sin(H\delta/2)}{H\sin(\delta/2)}
\cos((H-1)\delta/2),
\tag{74}
\]

奇点连续延拓。式 (74) 是现有附录的有限相位计算。[R3]

### 11.3 稀疏注意力的联系应停在准确的数学层

对固定的有效键集合 `M_i`，式 (48)–(62) 只需把求和改为 `j∈M_i`；所有倾斜重加权、竞争和 value 恒等式保持成立。不同 mask 改变实际参与竞争的内容与 lag，而不是使位置编码不再重要。

如果 mask 来自会随频率变化的 top-k selector，则必须分开“固定支持内的响应改变”与“集合成员改变”。若 selector 原来的第 k 与第 k+1 分数间隔为 `γ>0`，所有 selector 分数改变量绝对值小于 `γ/2`，top-k 集合不变；证明是任意入选/未选对的排序差仍为正。这里必须针对**真正的 selector 分数**，不能默认它等于本文的 attention logits。

该论证说明为什么频率设计与稀疏读取是相关而非互相替代的问题；它不是已经完成真实稀疏系统提效的证据。无需让 Codex 因这一理论说明额外启动稀疏模型实验。

**论文落点：** 讨论中可用一句连接应用动机；详细距离测度用于附录或内部解释。不要把人为选定 `μ` 的最优配置宣布成所有 Agent 轨迹的通用最优配置。

---

## 12. Native 的理论空间：有设计自由度，不等于已有无数据通用最优表

### 12.1 权重训练最优不推出频率配置已经最优

写可微任务风险为 `F(W,z)`。在固定几何表 `z_N` 上训练得到 `W_*`，即使理想地达到

\[
\nabla_WF(W_*,z_N)=0,
\]

也没有数学条件要求

\[
\nabla_zF(W_*,z_N)=0.
\tag{75}
\]

这是两个不同参数方向。NCP 的固定支持原生结果和 LeRoPE 的学习结果，对这一设计空间分别提供不同协议下的证据；两者不能互相替代。[R9、R12、E4]

### 12.2 一个严格而有限的局部可达性结论

令 `z_N` 严格位于固定端点可行集的内部（内部 gap 有正余量），并令 `P` 表示只保留自由内部坐标。若冻结风险 `F(z)=F(W_*,z)` 可微且

\[
g=P\nabla_zF(z_N)\ne0,
\]

则足够小的 `t>0` 下，`z(t)=z_N-tg` 仍可行，并且

\[
F(z(t))=F(z_N)-t\|g\|^2+o(t)<F(z_N).
\tag{76}
\]

**证明：** 有限多个严格正 gap 给出足够小的可行邻域；一阶 Taylor 展开给出严格下降。

它证明了“冻结权重不排除原生配置改善”，而不证明一个只用 base、K、L 的固定方向就是 `-g`。当前静态构造限制不读取梯度；式 (76) 是存在条件，不是改变构造协议的执行指令。

### 12.3 多任务共同改善为什么既不能假定必然，也不能判定不可能

对多个可微风险的内部投影梯度 `g_1,...,g_M`，共同**严格一阶**下降方向存在，当且仅当

\[
0\notin\operatorname{conv}\{g_1,\ldots,g_M\}.
\tag{77}
\]

必要性：若 `Σλ_i g_i=0`，`λ_i≥0,Σλ_i=1`，不可能所有 `g_i^Tv<0`。

充分性：取梯度凸包中距离 0 最近的点 `g_*≠0`。投影最优性给出 `g_i^Tg_*≥||g_*||²`，选择 `v=-g_*` 即使每一项严格下降。

若存在活跃的排序/最小 gap 约束，应使用相应切锥，而不能套用式 (77) 的无附加边界版本。若式 (77) 不满足，也只说明不存在共同严格**一阶**下降方向，不排除二阶或有限幅度改善。

这个结论宜留在研究记录中：它拒绝把“不同任务当前有升降”上升为不可能性结论，也拒绝从一个代理分数下降推出所有任务必然提高。

### 12.4 NCP 的参考风险怎样得到 sinc 平方

当前 NCP 的公开参考比较是

\[
r(\theta)=\frac1{2\pi}\int_0^{2\pi}
\log(1+e^{\cos\psi-\cos\theta})\,d\psi,
\]

\[
\mathcal R(\phi)=2\int_0^1(1-t)r(\phi t)dt,
\quad\phi=(L-1)\omega.
\tag{78}
\]

`r` 是平滑偶周期函数，可写 Fourier 级数

\[
r(\theta)=a_0+\sum_{j\ge1}a_j\cos(j\theta).
\]

积分核满足

\[
2\int_0^1(1-t)\cos(xt)dt
=\frac{2(1-\cos x)}{x^2}
=\operatorname{sinc}^2(x/2),
\]

其中 `sinc(y)=sin y/y`。于是

\[
\boxed{
\mathcal R(\phi)=a_0+
\sum_{j\ge1}a_j\operatorname{sinc}^2(j\phi/2).
}
\tag{79}
\]

这解释了三角距离权重与 NCP 频率风险的联系，不需要新增“内容天然均匀随机”的模型假设。[已有；R9]

### 12.5 数值构造的严格凸性属于有限参考目标

对有限 32 模态目标，记 `R_32`，每槽优化

\[
F_k(u)=\mathcal R_{32}(\phi_ke^{-u})+\frac\lambda2u^2,
\quad\lambda=9/4.
\tag{80}
\]

令 `q(φ)=dR_32/d logφ`，内点条件为

\[
\boxed{\lambda u=q(\phi_ke^{-u}).}
\tag{81}
\]

对 `f(x)=sinc²(x/2)`：

\[
h(x)=(x\partial_x)^2f(x)
=2\cos x-6\frac{\sin x}{x}
+8\frac{1-\cos x}{x^2}.
\tag{82}
\]

有 `|h(x)|≤4`。对 `0≤x≤4`，利用式 (79) 的积分核直接两次 log 微分：

\[
|h(x)|\le x/3+x^2/6\le4.
\]

对 `x≥4`，由式 (82)：

\[
|h(x)|\le\sqrt{4+36/x^2}+16/x^2\le3.5.
\]

所以只要有限系数满足

\[
\sum_{j=1}^{32}|a_j|<0.554403,
\]

便有

\[
F_k''(u)\ge9/4-4\sum_{j=1}^{32}|a_j|>0.03238.
\tag{83}
\]

这给出严格凸性。一般 box 约束还需分别检查下界、内根和上界的 KKT 符号；不能在未经检查的新网格上自动假设根落在区间内部。

当前约束为固定端点、`0≤u≤2/9`、相邻新 log-gap 至少为原 gap 的一半。当前 OLMo 部署网格上，标量解满足 gap 条件且该条件不活跃，因此也是受约束有限目标解。对任意其他网格，如果逐槽解违反 gap，应求解耦合约束，而不是先独立解后静默覆盖。[R9]

本轮独立 quadrature/FFT 得到 32 模态系数绝对和约 `0.55440179595465`，对应式 (83) 下界约 `0.03239281618`。这是公式与数值量级的独立复核，不是重放已部署 FP32 表、也不是无限 Fourier 风险的截断误差证书。

### 12.6 NCP 应如何服务本文

它提供一个有公开输入、明确目标、可复算数值解的原生固定支持构造。已有同目标 NLL/额外历史利用结果支持其实际作用；当前真实 QA 结果尚不能支撑通用增强。

正确理论关系是：**原生配置空间确实存在；参考风险构造是一个可执行实例；实际收益由任务证据界定。** 不应把当前 QA 未完成可靠增强解释成 `z` 在 Native 没有价值，也不能用参考风险下降替代真实下游成绩。

**论文落点：** 主文保留 Native 目标与已证实的作用；完整式 (78)–(83) 放附录 E。式 (75)–(77) 优先作为内部理论说明，不需要再加一个抽象多任务优化章节。

---

## 13. Cosh、频率密度与学习适应：另一种配置，而不是相反原则

### 13.1 用尾质量把已有变分推导写得更清楚

设 `[0,1]` 上有非负单位密度 `ρ`，定义尾质量

\[
S(t)=\int_t^1\rho(v)dv,
\quad S(0)=1,\ S(1)=0,\ S'=-\rho.
\]

当前密度目标为

\[
\mathcal J[\rho]
=\frac{\lambda_\rho}{2}\int_0^1\rho^2dt
+\frac{\lambda_S}{2}\int_0^1S^2dt,
\quad\lambda_\rho>0,\lambda_S\ge0,
\quad\tau^2=\lambda_S/\lambda_\rho.
\tag{84}
\]

这里使用 `λ_ρ,λ_S` 表示目标权重，以区别 Native 端点、RoPE base 和第 5 节的依赖伸缩倍数；它们对应当前 Cosh 推导中的两项权重。

等价地

\[
\mathcal J[S]=\frac{\lambda_\rho}{2}\int_0^1[(S')^2+\tau^2S^2]dt.
\tag{85}
\]

在固定边界的 `H^1` 函数空间中，该泛函严格凸。Euler–Lagrange 方程为

\[
S''=\tau^2S.
\]

唯一解

\[
S_\tau(t)=\frac{\sinh[\tau(1-t)]}{\sinh\tau},
\quad
\rho_\tau(t)=\frac{\tau\cosh[\tau(1-t)]}{\sinh\tau}.
\tag{86}
\]

它满足 `ρ>0` 与单位质量，因此非负性约束被满足。`τ=0` 连续退化为 `S=1-t,ρ=1`。[已有；R3、R10、R12]

### 13.2 最优值和逆 CDF

由 `S''=τ²S`：

\[
\int_0^1[(S')^2+\tau^2S^2]dt
=[SS']_0^1=\tau\coth\tau.
\]

所以

\[
\boxed{\mathcal J_{\min}=\frac{\lambda_\rho}{2}\tau\coth\tau,}
\tag{87}
\]

`τ→0` 的极限为 `λ_ρ/2`。

积分再反演给出

\[
\boxed{
Q_\tau(u)=1-
\frac{\operatorname{asinh}((1-u)\sinh\tau)}{\tau}.
}
\tag{88}
\]

其导数为

\[
Q_\tau'(u)=
\frac{\sinh\tau}{\tau\sqrt{1+(1-u)^2\sinh^2\tau}}>0,
\quad Q_\tau''(u)\ge0.
\tag{89}
\]

所以 `Q` 单调且凸，连续端点为 0/1。对于均匀间隔的采样节点 `u_k`，仿射锚定后

\[
z_k=
\frac{Q(u_k)-Q(u_0)}{Q(u_{K-1})-Q(u_0)}
\le\frac{k}{K-1}.
\tag{90}
\]

这是凸函数在两端弦线以下的直接结果。它说明在同 sampled endpoints 下，内部频率被向较快侧运输。非锚定 midpoint/native-node 安装有自己的范围变化，不能与式 (90) 的固定支持身份混用。

### 13.3 Cosh 与 TailSpline 为何可以方向不同

Cosh 密度构造关注一个给定频率区间中的分配及随后权重对它的适应；TailSpline 则相对于已经训练的 Native 表，为一套声明的扩展范围重新分配距离伸缩。它们的参照、外带约束、范围和权重是否适应都不同。

因此，“Cosh 在相同支持下向快端移动”和“TailSpline 相对 MrPro 放慢过渡段”并不冲突。论文共同研究的是配置，而不是“频率越慢越好”或“越快越好”的统一法则。

### 13.4 学习与冻结的曲率差别：可选的局部解释

为明确适应的数学作用，考虑一个局部光滑风险 `F(W,z)`，在 `W_0,z_0` 有 `∇_W F=0` 且 `H_WW` 正定。此处假设只用于局部推导，不声称实际大模型 Hessian 全局正定。

由隐函数定理，局部最优权重 `W_*(z)` 满足

\[
\frac{dW_*}{dz}=-H_{WW}^{-1}H_{Wz}.
\]

适应后风险 `\bar F(z)=F(W_*(z),z)` 有

\[
\nabla_z\bar F=\nabla_zF,
\]

\[
\boxed{
\nabla_z^2\bar F
=H_{zz}-H_{zW}H_{WW}^{-1}H_{Wz}.
}
\tag{91}
\]

这是一条标准 Schur-complement 关系：**在同一个理想权重驻点处，冻结与可适应风险的一阶配置梯度相同，二阶响应不同。** 不能错误地说只要解冻权重，一阶方向就必然反转。

权重适应至少可以选择不动，因此局部最小化后的风险不高于相同 `z` 下固定 `W_0` 的风险；在二阶近似中，收益是非负的耦合二次项。这为为何学习配置与冻结部署不能简单互换提供了一个有条件的解释，而不保证任何特定训练算法都达到该局部解。

**论文落点：** Cosh 仍保留现有变分构造和学习证据。式 (91) 可留内部报告，不需要扩写主文；不能把它作为 Cosh 和 TailSpline 两条具体移动方向的唯一解释。

---

## 14. 频谱改变与坐标重排：哪些变化能被静态换基吸收

设频率位于无整数位置混叠区间 `(0,π)`，完整旋转核为

\[
\mathcal R_\Omega(d)=\operatorname{diag}_k R(\omega_kd).
\]

若位置无关可逆矩阵 `A,B` 满足

\[
A^T\mathcal R_\Omega(d)B=\mathcal R_\Lambda(d)
\quad\text{对 }d=0,1\text{ 成立},
\]

则 `d=0` 给出 `A^TB=I`，`d=1` 给出两旋转矩阵相似。其复特征值为 `e^{±iω_k}`，在 `(0,π)` 内唯一识别频率多重集，因此

\[
\boxed{\Omega\text{ 与 }\Lambda\text{ 的频率多重集必须相同。}}
\tag{92}
\]

反过来，若多重集相同，完整 pair 置换即可构造这样的正交换基，且对所有整数 `d` 成立。[已有；R5]

该命题针对所有内容向量上的完整双线性核，不是声称有限数据上的模型不能近似适应另一张频率表。频率相同但只重排频率、没有同步重排 learned Q/K，当然可能改变模型行为。

当前理论记录另有有限窗口最佳正交换基及条件数误差界。这些结果有数学价值，但不直接决定 TailSpline 的 band、边界目标或任务收益，本轮无需为了“深入”把它们重新塞进主文。[R10]

---

## 15. 理论与已有模型证据：只建立已经支持的连接

本节是证明的证据索引，不新增指标、不重新合并实验，也不将 CPU 几何计算伪装成模型测量。

| 现有证据 | 对应理论 | 支持的结论 | 不能越过的边界 |
|---|---|---|---|
| 固定 sampled support 的训练/冻结比较 | 式 (1)–(6) | 范围固定后内部配置仍影响质量 | 训练干预与冻结同权重干预仍有不同身份 |
| BM–Uni 与 clean T–C | 式 (10)、(44)–(47) | 总位移不能充分解释配置效果 | 不把全部差异单独归因于最后一个尾端点 |
| Llama 同一 s=4 表在 8K/16K/32K 的 T/P 结果 | 式 (11)、(20)、(33)–(39) | 相同倍率下，整个已测部署范围的质量可以不同 | 三个长度点不证明每一个未测长度；α 不等于 H/L |
| 四家族、partial RoPE 和 70B 表复用 | 式 (41)–(43) | 公共相位归一化规则具有实测迁移价值 | partial RoPE 不是已经验证 sparse attention |
| T 比 P 的 full-pair rank 更低但任务更好 | 式 (51)–(71) | 位置方向多样性不单独决定 learned content use | 不推出降低 rank 是有益机制 |
| Native NCP 的同目标 NLL/上下文利用 | 式 (7)、(75)、(78)–(83) | 不增加频率范围也可以改变原生上下文使用 | 参考风险下降不等于通用 QA 增强 |
| Cosh 三种子学习、续训、LoRA、下游适配 | 式 (84)–(91) | 配置也可以与权重适应共同提供价值 | 保留各组是否锚定端点、训练曝光和原生成本 |
| 高倍率检索、NLL、自然 QA 方向并不全相同 | 式 (33)、(37)、(51)–(62) | 不同终点涉及不同距离/内容/竞争，不能由单针替代 | 不把不同方向宣布成所有方法不可避免的代价 |

### 15.1 可以直接用于正文的理论—结果连接

> At a fixed extension factor, the allocation changes the distance response of each learned rotary channel. TailSpline assigns more dilation throughout the transition and approaches the fully interpolated tail with a smaller residual phase mismatch. This changes the evidence–distractor comparisons available to the frozen model without changing its outer frequency bands. The matched evaluations then establish the practical value of this allocation across the tested context lengths.

该段必须紧接真实比较身份：T/P gain、权重、band 与输入共享。它不声称已观测所有层的具体证据中介。

### 15.2 一句足以连接 Native 与学习的理论表述

> Frequency allocation is not exhausted by extension: fixing the native support leaves interior frequencies free, while allowing weight adaptation changes how a model can use a transported allocation.

这句保留原本立意；后续分别报告 NCP 已有的原生预测/上下文利用结果，以及 Cosh 已有的学习结果，不暗示一张静态表已经在所有 Native QA 上完成增强。

### 15.3 目前不应写成定理的结论

“TailSpline 的边界目标等于 Transformer 任务风险”；“所有 task 的收益都由 terminal gap 降低造成”；“单次 attention 的 margin/TV 界保证多层生成分数”；“大 base 使内部配置失效”；“在 `[1,sL]` 的每个长度和任务都稳定改善”；“Native 存在局部下降方向，所以公共零数据构造已经可直接找到它”。这些都超出当前推导。

避免这些跳跃，不等于把正文写成限制清单。正文应正面写清实际成立的结论，条件放在对应数学对象旁一次说明即可。

---

## 16. Codex 执行：把理论接入当前论文，而不是把整份报告搬进主文

### 16.1 执行目标与边界

以当前工作树为起点，先核对 HEAD 及作者未提交改动。本报告基于 `efd31c3`；若源码已更新，按 label 和章节内容合并，不覆盖作者新稿，不回退到历史 theory proposal。

交付要达到：读者从主文能够理解 `z→距离响应→TailSpline 构造→实测质量` 的关系；附录能够复核关键公式与条件；构造器、模型结果和比较协议保持原身份。

这次任务执行数学整理、源码定点修改和 CPU 复核。模型实验与线上任务队列保持不变。本报告含内部仓库溯源信息，留在研究目录；匿名提交稿只整合必要命题、证明与匿名验证代码，不直接带入本报告的 GitHub 用户名和内部来源链接。报告中的梯度、距离权重、多任务可达性是解释工具，不作为重新校准配置的操作步骤。

### 16.2 分级采用：正文需要哪些内容

**主文优先采用：**

- 式 (4)：`m` 与 `z` 的精确连接。
- 式 (11) 与式 (20)：配置改变什么距离响应，TailSpline 如何移动整个过渡段。
- 尾端相位残差的简短结论，完整式 (28) 与证明放附录。
- 当前已有的目标—干扰竞争解释；避免只用有效秩解释质量。
- 用既有固定倍率多长度结果验证完整配置的应用价值。

**附录优先采用：**

- 式 (14)–(18) 的有限解证明，作为现有证明的合并优化而非平行复制。
- 式 (28)–(31) 的 tail residual 推论。
- 式 (33)–(34) 的明确依赖伸缩交叉条件；明确 `α≠H/L`。
- 式 (50)–(60) 中当前稿尚缺且真正用到的 value、有限响应与曲率解释。
- NCP 已有的有限目标凸性和 Cosh 已有的完整变分证明。

**优先留研究记录：**

- 多通道参考权重交叉式 (35)。
- 标量双目标退化式 (40)。
- 多任务共同一阶下降条件式 (77)。
- 学习适应的 Schur complement 式 (91)。
- 通用 top-k selector 稳定性、最佳换基误差等扩展内容。

这些内容是有用的理论分析，但不是因为正确就必须进入九页主文。是否放入稿件由它是否帮助读者理解现有主张决定，而不是按公式数量计价值。

### 16.3 具体文件与修改职责

| 文件（均为当前活动路径） | 执行动作 | 完成条件 |
|---|---|---|
| `paper-2027/sections/02_exponents.tex` | 整合式 (4)，说明 s 与实际 H 不同 | `z,m,ε` 被明确识别为同一配置的不同表达；不新增独立“总位移自由度” |
| `paper-2027/sections/03_theory.tex` | 接入距离响应和内容竞争的短连接，保留 full-pair 结构 | 不从 rank 直接推出任务方向；不删除已有强结构结果 |
| `paper-2027/sections/04_mature.tex` | 在 T/P 逐点排序后解释过渡伸缩及尾端参照残差 | 有限公式、band、gain、安装方式完全不变 |
| `paper-2027/appendix/compact_b_constructions.tex` | 合并有限解证明；增加 tail residual 和依赖伸缩交叉推论 | n 的边界情况、距离条件、周期条件明确；不重复旧推导 |
| `paper-2027/appendix/compact_a_geometry.tex` | 保留几何证明；按需要补 value/有限变化/Hessian 对象区分 | 固定状态与整网反向传播区分；Fisher 不冒充完整任务 Hessian |
| `paper-2027/appendix/compact_c_identification.tex` | 与前移的 z–m 公式交叉引用；必要时保留式 (47) | 总位移/质心不重复计控制；不将 F1 当作可微效用 |
| `paper-2027/appendix/compact_e_native.tex` | 核对 NCP 目标、32 模态身份、box/gap 条件 | 数值参考最优性与任务结果分开，既有 Native 数据不改写 |
| `paper-2027/appendix/compact_f_learning.tex` | 核对理论连接与各学习协议身份 | anchored/unanchored、冻结/训练、相位曝光不混用 |
| `paper-2027/sections/04_experiments.tex` | 只补必要的理论—结果连接句 | 不新增未经测量的中介结论或改变已报 endpoint |
| `paper-2027/figs/allocation_design.py` | 复用现有构造器，扩展验证时保持构造函数不变 | s=1 精确恒等、非空 band 校验和 FP32 输出规则保留 |

报告表中数学编号用于本报告阅读；入稿使用现有或新增的语义 label，禁止照抄硬编码 `\tag{...}` 导致编号冲突。

### 16.4 可直接改写成 TeX 的两个附录命题

建议 label：`prop:tailspline-tail-residual`。

> **Residual phase near the interpolated tail.** Let T and P be the finite-grid TailSpline and MrPro profiles with transition width n≥2 and scale s>1. At the final interior channel, the ratio of their unwrapped phase deviations from the fully interpolated reference equals `(s^{ε_n^T}-1)/(s^{ε_n^P}-1)` and is at most `3/(2n+1)`. For the channel j steps before the tail, the corresponding bound is `[(j+1)(3n+1-j)]/[(2n+1)(2n+1-j)]`.

证明使用式 (29) 和 `(exp x-1)/x` 的单调性即可。明确 unwrapped phase deviation，不把它写成任意多周旋转矩阵范数的相同比例。

建议 label：`prop:allocation-dilation-crossover`。

> **Dilation-dependent phase matching.** For an aligned content reference at distance d₀, compare two interior exponents T>P at a fixed scale s. TailSpline has smaller squared residual phase at dependency stretch α precisely when `α>2/(s^{-T}+s^{-P})`. If both residual phases lie in `[-π,π]`, this also orders the positive-amplitude aligned cosine contribution.

证明直接展开平方差。命题前注明 `α` 描述指定依赖伸缩，不是一般输入长度比。

这两条命题是既有构造的进一步分析，不增加参数、不改变方法、不要求模型重新运行。

### 16.5 CPU 执行与数学验收

本交付包含 `verify_rope_theory.py` 和已执行的 `theory_verification.json`。脚本仅依赖 Python、NumPy、mpmath。

可将它们放到建议新建的研究目录，例如：

```text
paper-2027/research/theory_synthesis_20260917/
    THEORY_SYNTHESIS.md
    verify_rope_theory.py
    theory_verification.json
    INTEGRATION_NOTES.md
```

上面是**建议新增路径**，不是声称当前仓库已存在。

执行示例：

```bash
# 在放入交付脚本的目录中运行；不加载 checkpoint。
python verify_rope_theory.py --output theory_verification.json

# 在仓库根目录，复用已经存在的原构造验证。
python paper-2027/figs/allocation_design.py

# 完成 TeX 整合后，沿用项目当前编译入口。
cd paper-2027
bash compile.sh
```

本轮只实际运行了交付的独立脚本；上面两个仓库命令是交给 Codex 的执行步骤，不是本轮已在当前容器复跑成功的声明。

验收内容：

1. 有理数证明辅助：`n=1,...,128` 的 T/P/C 代数、单位质量、KKT、最优值、质心与退化 n。
2. Green 逆矩阵恒等式与尾端有限残差不等式。
3. 逐通道与多通道相位交叉、单调相位区间内的合成余弦响应。
4. 固定状态 softmax 倾斜、组 log-odds、value 协方差、TV/KL 界和 Jacobian。
5. **分别**验证 KL 的局部 pullback 和真实 attention CE 的额外 Hessian 项。
6. uniform lag、causal pair、iid absolute-lag 的有限矩与旋转差异闭式。
7. 四个当前公共数学网格的 band、z–m 关系和倍率导数。
8. 高精度 slow-pair 主要系数、Cosh 最优值和 NCP 有限参考系数。

严格区分两种复现：脚本的 public-grid 验证用数学 float64 表；当前安装器先接受 FP32 Native 表、再计算并回写 FP32。前者通过不意味着已复现后者的 table hash。涉及安装回归时，应在 Codex 环境复用原构造器单独确认，不替换当前函数的浮点路径。

### 16.6 最终交付给作者的内容

`INTEGRATION_NOTES.md` 应简洁记录实际采用了哪些命题、放入哪个活动文件，以及哪些内容留在研究记录。最终交付修改后的 TeX/PDF、CPU 验证结果和源码 diff；数字表与模型结果应能确认没有被本轮理论编辑改写。

作者判断这次理论增强是否成功，只需看三个问题：

- 能否从 z 和 m 的关系直接理解 TailSpline 改变了什么？
- 能否从距离响应、尾端残差和内容竞争理解设计为何有意义，而不是只看到一个可解泛函？
- 是否完整保留工作区间与 Native 的原始研究目标，同时让每个数学结论对准真实成立的对象？

---

## 17. 本轮实际数值核对结果与证据边界

完整机器可读结果见随附 `theory_verification.json`，总状态为 `PASS`。

| 检查 | 实际执行结果 |
|---|---|
| 有限网格代数 | n=1…128，精确有理数通过 |
| 尾端残差推论 | n=2…128、全部内部尾距、s∈{2,4,8,16} 通过；直接残差比与稳定公式最大差约 9.06e-15 |
| 相位交叉恒等式 | 12,792 个逐通道比较，最大代数差约 4.26e-14 |
| 合成 attention | 100 组随机有限状态；组 odds 最大差约 1.11e-15，value 协方差最大差约 4.44e-16 |
| 一阶导数 | 最大方向差分误差约 5.22e-10 |
| attention CE 的完整二阶导数 | 最大方向差分误差约 1.77e-7 |
| KL pullback | 最大方向二阶差分误差约 6.00e-8 |
| 四个公共数学网格 | band、z–m、外带、尺度导数通过 |
| slow-pair 系数 | 65 位计算复核 19/12600 领先系数 |
| Cosh 与 NCP | 192 点 quadrature 复核 Cosh 最优值；独立参考积分/FFT 复核 NCP 系数和曲率余量 |

其中高精度例 `x=0.05,y=0.10` 的 exact/leading 比例为约 `1.00057027245`。它与当前示例值 `1.00058` 仅有很小的末位差别；若保留该例，可在独立高精度复核后统一成 `1.00057` 或直接给相对偏差，不需要改变渐近命题，也不应把该小差异写成论文的核心理论问题。

这些检查是对公式、有限离散化、边界条件和数值实现的检验。它们不是新的 RULER、QA、NLL、训练或 attention 中介观测。共同一阶下降、Schur complement 等一般命题以本文给出的解析证明和假设为依据，不冒称也经过了真实模型验证。

---

## 18. 来源与回查索引

### 18.1 当前仓库活动来源

下列路径均相对于仓库根目录，统一固定在 HEAD `efd31c3b0e4f04b0d839497caa81c0e906374361`。Codex 应优先读活动源文件，再看历史综合文件。

| 编号 | 路径 | 本报告的使用 |
|---|---|---|
| R1 | `paper-2027/sections/02_exponents.tex` | z、范围、位移与 gain 的基本定义 |
| R2 | `paper-2027/sections/04_mature.tex` | TailSpline/MrPro 有限公式、目标与安装 |
| R3 | `paper-2027/appendix/compact_b_constructions.tex` | 有限解、C 控制、Cosh、尺度响应、相位参照与 turn normalization |
| R4 | `paper-2027/appendix/compact_c_identification.tex` | z–m 映射、位移—质心、识别与坐标干预 |
| R5 | `paper-2027/appendix/compact_a_geometry.tex` | 完整 pair、慢频展开、内容坐标、rank 反例与组 odds |
| R6 | `paper-2027/sections/03_theory.tex` | 当前主文理论职责与已采用的结构认识 |
| R7 | `paper-2027/figs/allocation_design.py` | 实际 FP32 TailSpline 构造及已有 CPU 验证 |
| R8 | `scripts/lib/rope/official_yarn.py` | 官方 YaRN 方程、gain 与 legacy 算子的区别 |
| R9 | `paper-2027/appendix/compact_e_native.tex` | NCP 参考风险、32 模态、约束与实际证据范围 |
| R10 | `paper-2027/research/theory_revision_proposal_20260915/THEORY_SYNTHESIS.md` | 已有推导库存；历史路径/状态不覆盖当前活动稿 |
| R11 | `paper-2027/sections/04_experiments.tex`; `paper-2027/appendix/compact_d_frozen.tex`; `paper-2027/tables/table_clean_length_main.tex` | 已有 frozen/区间/等位移证据，不进行新重汇总 |
| R12 | `paper-2027/sections/04_construction.tex`; `paper-2027/appendix/compact_f_learning.tex` | Cosh 构造和各自独立学习协议 |

固定版本入口：

`https://github.com/misaya-yang/hybrid-rope/tree/efd31c3b0e4f04b0d839497caa81c0e906374361`

### 18.2 外部原始资料：用于承认来源和解释关系

**E1. RoFormer: Enhanced Transformer with Rotary Position Embedding.** Su 等。原始 RoPE 旋转和相对位置核。  
`https://arxiv.org/abs/2104.09864`

**E2. YaRN: Efficient Context Window Extension of Large Language Models.** Peng 等。分段插值/外推与 amplitude scaling。本文涉及的实现由仓库固定到原始算子 revision。  
`https://arxiv.org/html/2309.00071v2`

**E3. MrRoPE: Mixed-radix Rotary Position Embedding.** Tian 等。混合进制及 Pro 的中频分配，是当前共享外带的直接参照。  
`https://arxiv.org/html/2601.22181v1`

**E4. LeRoPE: Learnable RoPE Frequencies Improve Language Modeling.** Karypis 等，2026。频率梯度、Q/K 相位和 value/下游加权；式 (62) 应与它的 §3.2/Appendix B 对照引用。本文不把这些通用梯度工具声明为自己的首创。  
`https://arxiv.org/html/2607.10134v1`

**E5. How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization.** Wu、Liu、Jadbabaie，2026。数据依赖尺度与频率使用的条件性分析。它支持研究明确的依赖伸缩，而不是由总输入长度直接假定语义距离；本报告不借用其特定效用的最优性来证明 TailSpline。  
`https://arxiv.org/html/2607.07678v1`

**E6. Round and Round We Go! What makes Rotary Positional Encodings useful?** Barbero 等。不同频率的已学习位置/内容使用，支持区别位置方向与内容坐标；不把某模型的频带现象推广成所有模型统一分工。  
`https://arxiv.org/abs/2410.06205`

完整 pair 的主角度、frame potential、softmax 曲率和 Schur complement 属于标准数学工具。现有 BibTeX 中的 Björck–Golub、Benedetto–Fickus 等来源继续保留；本轮增量在于对论文具体配置和距离响应的应用、整理及条件化解释。

---

## 最终执行摘要

先复核并整合式 (4)、(11)、(20)、(28)、(34)、(51)：它们分别回答“是不是同一个 z”“改变了什么距离响应”“TailSpline 怎样分配伸缩”“尾端目标改善了什么相位量”“为什么不同依赖伸缩有不同偏好”“怎样进入真实内容竞争”。

保留现有 full-pair、Cosh、NCP 理论，减少对象混同和重复证明。把更一般的敏感性、共同下降和权重适应结果留作附录或研究支撑，而不是把论文重新改造成优化理论大全。

**理论最终服务的仍是：给定实际可用的上下文范围，内部频率配置是一个可以被识别、被构造、并由已有实验验证其实际价值的设计自由度。**
