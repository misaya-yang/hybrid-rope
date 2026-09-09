# RefCarry 最近邻审查：公式覆盖与 compressed-reader 边界

日期：2026-09-09。任务在阅读 DeepSeek-V4 压缩-reader 后，按主任务新指令转向用户提供的 RefCarry 方案。
输入：`/Users/yang/.codex/attachments/07256e86-931d-4d41-bc44-b323f9d6e124/pasted-text.txt`。
范围：只读一手论文和已有本地材料、独立方程审查、本文写作；没有连接GPU、下载模型、训练或代理扩展。

## 判断

**RefCarry 可以作为受限的、带明确参考地址解释的位置状态旁路假说；当前不能称为已确立新意的独立理论家族，更不足以宣布“唯一值得solid-accept下注的主线”。** 最强近邻是TAPE的实际位置更新与读取算子，不只是“都做contextual PE”这种表面关联。RoVE覆盖位置加权消息及frame change的组成操作；RePo覆盖单一动态位置的旋转实现。该方案剩下的真实差别主要是单边query更新、原始key坐标冻结、概率矩约束，以及跨特定hybrid bridge的更新日程与实现成本。

这些区别可以研究，但需要先证明所需能力和独立效率价值；不能仅凭架构部署位置不同，把已有代数改写成新定理。主任务已发现softmax、规范化分布秩、oracle与任务泄漏方面的反例，本文不将“logit恒等式正确”升级成“完整寻址机制正确”。

## 1. TAPE：最强公式级重合，及必须保留的非等价细节

一手来源：[TAPE v1 §3.2 Eq.6–8、§3.3、Appendix B](https://arxiv.org/html/2501.00712v1)。

将接收query记为i、来源token记为a，统一转置约定后，TAPE的两项核心操作可以写成

\[
E'_{i,m}=\sum_a A^{(m)}_{ia}E_{a,m},\qquad
s_{ij}=\sum_m q_{i,m}^{T}\,\phi(E_{i,m}E_{j,m}^{T})\,k_{j,m}.
\]

论文实用选择包含 `B=L`、`phi=identity`。其后还有依赖token features的等变位置MLP，以及residual位置更新。

对每个rotary pair，令原始位置矩阵

\[
E_a=\rho(p_a)^T.
\]

如果只更新接收query的位置字段、令该次更新权重为RefCarry的实际writer分布 `mu_i(a)`，并冻结key侧原始字段，则

\[
E'_i=\sum_a\mu_i(a)\rho(p_a)^T=M_{\mu_i}^{T},\qquad E_j=\rho(p_j)^T.
\]

代入TAPE的bilinear token mixing，得到

\[
q_i^T E'_iE_j^Tk_j
=q_i^TM_{\mu_i}^{T}\rho(p_j)k_j
=(M_{\mu_i}q_i)^T\rho(p_j)k_j.
\]

**这正是RefCarry宣称的核心群矩重基准公式。** 该等价需要说明以下条件，不能粗暴写成“默认TAPE和RefCarry逐位相同”：

- TAPE Eq.7 默认使用每个分组m自己的 `softmax_a(alpha_{ia,m})`；RefCarry使用某个writer head的完整attention row，跨reader频率共享它。上面的映射需显式令分组权重共享同一个mu，或比较这一受限变体。
- 默认TAPE传播每个token的contextual位置字段，因此后续key字段通常也已更新；RefCarry保留原始key坐标。上面是**冻结key、单边更新query**的映射，不是默认双边TAPE完整模型的等价声明。
- TAPE还有位置MLP和residual；RefCarry将中间层的侧状态更新设为恒等，随后使用特定gate。等价比较须匹配这些更新日程。
- TAPE实际多头实现带head维，并允许跨head线性位置混合；RefCarry的行随机矩阵Pi是受约束的head mixing。非负、行和1使“仍是地址概率测度的矩”保持成立，这是有解释意义的限制，但不是全新线性操作。
- TAPE §3.3 已通过零初始化位置更新投影，使PEFT模型在初始化时与原模型一致。RefCarry `eta=0` 的native parity不能作为新贡献。

这里的“受限变体”指复用TAPE的位置读写与bilinear算子、改变Q/K字段的更新约束；不是证明仅设置默认实现的几个参数即可逐位得到RefCarry。特别是当前token的self-key也保持原坐标时，需要区分query侧contextual字段与key侧原始字段。

因此，“TAPE只是更新一个field，RefCarry则读取真实地址”这一二分过强：**TAPE Eq.7已经明确从attention read产生位置状态，并将其供后续token mixing使用。** 精简、冻结或跳过部分更新可能改变成本和归纳偏置；它不能消除上述算子覆盖。

默认双边字段还可写成更一般的关系：若query与key各有地址测度mu_i和mu_j，则

\[
q_i^TM_{\mu_i}^{T}M_{\mu_j}k_j
=\mathbb E_{a\sim\mu_i,b\sim\mu_j}
 q_i^T\rho(p_b-p_a)k_j.
\]

RefCarry的原始key地址是这里的特殊选择 `mu_j=delta_{p_j}`。这是对其在contextual-position算子族中位置的明确描述，不是说论文默认TAPE有意实现该特例。

## 2. RoVE：地址矩可作为旋转value通道的精确特殊读出

一手来源：[RoVE v1 §3 Eq.3 与 local-frame lens](https://arxiv.org/html/2606.11275v1)。

RoVE实际算子是

\[
y_i=\rho(p_i)^{-1}\sum_a A_{ia}\rho(p_a)v_a.
\]

它先把value转入公共坐标，attention加权，再转回query坐标。对一个额外的固定载荷通道 `v_a=e_1`，每个二维pair的输出为

\[
z_{i,k}=e^{-i\omega_kp_i}\sum_a A_{ia}e^{i\omega_kp_a}
=e^{-i\omega_kp_i}m_{i,k}.
\]

乘回已知的 `e^{i omega_k p_i}` 就得到RefCarry的地址moment。复数 `m=alpha+i beta` 对应矩阵

\[
\begin{pmatrix}\alpha&-\beta\\\beta&\alpha\end{pmatrix},
\]

所以每频率两个实数即可表示该旋转矩阵平均；不必携带四个独立矩阵元素。这也等价于把已知位置特征Phi(a)当作额外value做一次普通attention读出 `A Phi`。

但这不是默认RoVE完整模块与RefCarry相同：RoVE通常运输有语义内容的value，未规定把独立地址通道跨三个GDN层传送后再乘下一次query。准确的结论是：**attention加权位置moment的产生与局部/全局frame转换已有精确操作；新的旁路日程和使用场景仍需独立价值证据。**

## 3. RePo：point reference的重定位重合，soft moment并不等于单一位置

一手来源：[RePo v3 §3.2 Eq.4–7](https://arxiv.org/html/2512.14391v3)。

实际RePo为每token/head预测连续scalar

\[
z_i=(\operatorname{Swish}(h_iW_g)\odot h_iW_c)W_z,
\qquad s_{ij}=q_i^T\rho(z_j-z_i)k_j.
\]

RefCarry在 `mu=delta_a,g=1` 时就是

\[
s^{RC}_{ij}=q_i^T\rho(p_j-p_a)k_j.
\]

在logit层面，这对应query位置指定为 `z_i=p_a`、key仍取 `z_j=p_j` 的**单边位置assignment**。默认RePo同时为query和key预测位置，且由hidden state回归scalar，不以writer的实际attention分布作监督已知输入；不能说两个完整方法相同。

RefCarry的soft moment确实可超出单一scalar位置的表达类：一般

\[
m_k=\sum_a\mu(a)e^{i\omega_kp_a},\quad |m_k|<1,
\]

而任何标量位置旋转 `e^{i omega_k z}` 都是单位模长。多频率moment也未必对应一个共同z。这个区别不能被抹平。

但它也不能自动作为RefCarry优于RePo的能力论据：moment是地址分布的有限统计，模长会收缩；它精确生成的是**平均bilinear logit**，不是先对每个anchor做softmax读取再混合的完整结果。主任务已用双峰反例确认该差别具有实际数学后果。

对于RefCarry的gate和head mixing，可进一步统一为

\[
\nu_i=(1-g_i)\delta_{p_i}+g_i\sum_h\Pi_{ih}\mu_h,
\qquad \widetilde q_i=M_{\nu_i}q_i.
\]

这说明它是“Dirac当前位置与读出地址测度的凸混合”，不是一般纯旋转。应按这个实际函数类解释norm/phase干预。

## 4. “先按内容找地址、再相对该地址读取”并非2026年才出现的计算原语

一手来源：[Neural Turing Machines §3.3.1–3.3.2 Eq.5、7–9](https://arxiv.org/html/1410.5401v2)。

NTM已经显式保留上一时刻的地址权重，结合当前内容寻址，再做相对位移：

\[
w_t^g=g_tw_t^c+(1-g_t)w_{t-1},\qquad
\widetilde w_t(i)=\sum_j w_t^g(j)s_t(i-j).
\]

它明确支持“按内容跳到一个地址，然后访问其邻近位置”，以及跨recurrent controller步骤保留地址。对其循环memory的DFT频率，卷积对应地址Fourier矩的乘法。这里使用完整N维地址权重、显式位移和sharpening，当然不是RefCarry的native-RoPE query公式；但“读取地址成为可组合状态”作为原则并非空白。

因此，RefCarry可争取的是现代预训练hybrid中的一种低成本、特定频率接口与有效学习方式，不能宣称首次提出read-to-read地址状态或内容→位置的组合寻址。

## 5. 从已核对的DeepSeek-V4得到的实际适用边界

一手来源：[DeepSeek-V4 §2.3.1–2.3.3](https://arxiv.org/html/2606.19348v1)，以及固定revision `b5968e9190ef611bbf34a7229255be88a0e937c1` 的官方 `inference/model.py`、`kernel.py`。
本轮重新取得代码并核对SHA，与已有本地记录一致：

- model.py：`ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`
- kernel.py：`59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`

实际压缩是逐channel learned gate/APE加权聚合，再learned RMSNorm，再以块起点赋partial RoPE；CSA ratio4使用重叠源窗口，HCA ratio128不重叠。core使用共享K=V，稀疏kernel以同一KV既算score又算value输出；最后对输出末64维做query位置的inverse-RoPE。sink只有softmax分母贡献。

忽略量化、固定selection后，其一头的实际相对形式是

\[
u_b=\rho(p_b-p_i)c_b,\quad
\alpha_b=\frac{\exp(\tau q_i^Tu_b)}{\exp(z_{sink})+\sum_c\exp(\tau q_i^Tu_c)},\quad
y_i=\sum_b\alpha_bu_b.
\]

这不是普通固定V的Qwen reader。若在该后端只把query旋到one-hot参考a，却保留原输出inverse-RoPE，结果是

\[
\alpha_b^a\propto\exp(\tau q_i^T\rho(p_b-p_a)c_b),\quad
 y_i^{query-only}=\sum_b\alpha_b^a\rho(p_b-p_i)c_b.
\]

若要连value的输出frame也真正重基准为a，则应是

\[
y_a^{full-rebase}=\sum_b\alpha_b^a\rho(p_b-p_a)c_b,
\quad y_i^{query-only}=\rho(p_a-p_i)y_a^{full-rebase}.
\]

二者通常不同，输出投影也不会自动消掉此差别。所以RefCarry的“只改query”在非旋转value的Qwen reader中可以是清晰QK干预，不能不加限定地推广为所有compressed/shared-KV后端的完整frame-change。该边界直接来自RoVE/V4已存在的OV旋转操作，不是一项新的value-transport发明。

另外，压缩条目的块锚点不是其所有源token地址的可恢复清单。learned channel gates、APE和因果上游hidden states可能已经保存足够任务信息，因此不能仅看“一条压缩KV对应多个位置”断言真实reference loss。反过来，如果所有下游可见状态真的相同，post-hoc位置重标记不能恢复已丢失的source/payload绑定；有限维moment侧状态也不是所有压缩历史的无损表示。

[PPE §3.2 Eq.5–7](https://arxiv.org/html/2510.22936v1)已有将多个源位置ID分配给一个压缩token的不同频率段；它不是writer attention地址测度，也不执行后一次query重基准。这个差别真实存在，但不能将“保留压缩token来源位置”本身当作空白。

## 6. 对方案创新主张的最终分级

**已有明确操作覆盖，不能成立的首次主张：**

- attention read产生独立位置字段/位置moment：TAPE Eq.7直接覆盖。
- 用contextual位置矩阵参与下一次bilinear寻址：TAPE Eq.6直接覆盖其广义函数类。
- 旋转value聚合与reference-frame转换：RoVE直接覆盖。
- 根据context改变query/key位置：RePo直接覆盖；RefCarry单边、显式读地址的限制是区别。
- 内容寻址后按相对地址继续读、跨recurrent计算保持地址：NTM已经提供实际算法。
- 用zero-init保持原模型、或把线性特征期望称为“最小充分群状态”：前者已有TAPE-PEFT/残差适配先例，后者首先是有限函数族的线性因子化事实，不能仅改名构成理论新家族。主任务发现的simplex秩修正也必须先解决。

**可能有实现/归纳偏置价值，但尚未被证明的区别：**

- 使用实际writer分布、在接收reader固定原生频率上读取moment；
- key位置缓存完全冻结，仅更新query位置字段；
- row-stochastic head mixing与凸gate确保状态仍有地址测度解释；
- 在选定writer/reader之间旁路携带该状态，中间GDN不重写它；
- 相对于功能匹配的一侧TAPE变体，是否形成真实参数、侧状态、训练或延迟优势。

比较最近邻时，不能只让完整TAPE在每层更新全部位置字段，而将RefCarry的one-sided/skip-update成本优势全部归因于新寻址理论。应同时区分完整强实现和功能匹配的受限算子；若二者在相同约束下为同一forward，就应直说等价。

**对“有没有用”的结论：** 有可定义的adapter假说和明确的数学对象，值得作为既有contextual-position家族的受限设计来审查；现在没有足够依据把它封成新的主理论或承诺论文级成功。已有oracle反例意味着现有计划的强解释需要改写，不能用一组先验提升门槛替代方法正确性或新颖性。本文不提出GPU队列，也不替主任务的数学与泄漏审查作未经验证的能力结论。
