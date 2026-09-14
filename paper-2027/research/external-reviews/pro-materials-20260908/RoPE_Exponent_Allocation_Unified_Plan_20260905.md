# RoPE 指数分配：已有结果的解释、联合外推方案与统一论文主线

**研究对象：** `frequency = base ** exponent` 中的 exponent allocation；在明确采样端点、频率跨度、旋转维数和安装阶段后，研究非几何频率分配怎样改变模型计算。  
**实际目标：** 同一张静态频率表，尽量保持原生窗口能力，提高 4× 及更远长度的真实生成能力；分别研究冻结权重和有限 LoRA 适配。  
**资料范围：** 本次入口 TeX、同日完整稿件、历史零训练记录、含 ON 回填的最新实验报告，以及已检索核实的相关原始论文。实验数字引用现有报告；本轮没有执行模型训练或重新测量模型性能。

## 0. 本轮判断与优先行动

论文可以围绕下面这一条主线统一：

> **有限 RoPE 表中的指数分配决定相对位置如何调制内容匹配。训练过程使 Q/K 子空间与频率共同适应；安装阶段决定这种配对可以改变多少。因此，分配的价值应通过固定范围的因果干预，以及不同适配预算下的原生保留与长程能力来衡量。**

EVQ-Cosh、既有 coarse/derived 表、`log_s4` 都是这个研究对象的实例。三个实验路线可以使用不同的实例；需要统一的是干预变量、理论对象和归因方式。固定范围的训练结果不应被某个实例在成熟模型上的失败一并否定；成熟模型的成功也不能追认某个解析 surrogate 已经解释了它。

目前最有价值的新增工作，是利用已有 ON、ZF、ZC 适配器补齐运行表交叉评估，并查看旧训练样本与独立验证样本上的完整答案决策。它们直接回答两个尚未被现有聚合数字回答的问题：**ZF 的收益有多少依赖训练期间与新表共同适应？当前 4× 失败究竟首先表现为训练未拟合，还是拟合后的长度/实例泛化不足？**

这会决定下一笔训练开销。旧方案已经包含真实 8K/16K 输入、全答案与 EOS、最差 margin、全线性 LoRA、原模型 teacher 约束。再次把这些组件组合起来，并不构成新的解决方案。原协议对此有明确记录；实际执行细节仍要以 run 配置核对。fileciteturn9file0L24-L35 fileciteturn9file0L86-L114

**现有资料尚未确认“一张静态表、极低原生损失、强 4× 自然生成”的联合成功点。** 下文给出能够解释现有现象的计算关系、当前证据排除的解释、尚须测量的分歧，以及有明确启动条件的修正方案。公式成立与方案有效是两件需要分别验收的事。

---

## 1. 先把已经发生的事情讲清楚

### 1.1 固定范围内，内部分配确实具有独立行为效应

151.9M 三种子实验固定采样端点、跨度、初始化、数据顺序和训练预算，只移动 30 个内部频率。保持训练范围时，Cosh 实例相对几何分配的平均 NLL 差为：训练长度 +0.026，512/1K/2K 分别 −0.281/−0.176/−0.146。按目标长度重设运行范围后，同一对训练分支的排序反转。这个结果支持“分配效果以运行范围及其与已学权重的关系为条件”；它没有单独识别一个普遍的 support × allocation 交互定律。fileciteturn12file0L83-L96 fileciteturn12file0L118-L124

50.9M 的固定范围实验还包含与位移幅度匹配的 exponential 分配。它与 Cosh 的结果接近，有利于把研究结论放在非几何分配这一层；该实验训练预算很短、效应量也小，不能与 151.9M 的效应量混合。fileciteturn12file0L200-L236 fileciteturn12file0L294-L306

### 1.2 成熟模型的零训练收益，也有固定范围控制

稿件的 OLMo 16K unseen-nine RULER 中，固定相同端点、gain、模型和评测协议：uniform allocation 为 0.56%，coarse 为 61.04%，derived 为 60.47%。Qwen 64K 对应为 57.75%、64.00%、66.50%。这些是既有算术混合版本的成熟模型实验，不能直接当成后来 `log_s4` 的同一组结果。coarse 与 derived 没有被这些样本识别出稳定优劣。fileciteturn12file1L537-L567 fileciteturn12file1L624-L626

由此已经能够回答：在这些 checkpoint 和运行范围下，几何分配没有穷尽有效设计。还不能回答：哪一种非几何分配普遍最好，以及上述收益能否与极低原生代价同时实现。

### 1.3 单表的原生代价是真实问题

稿件中的旧长表强制用于 4K 时，NLL 从 2.7538 升至 2.8774；短请求精确保持来自 Native/long 路由。后来 `log_s4` 报告的 PG-19 retention 为 0.875302，也不能未经指标定义就解释成“仅损失某个百分比的全部能力”。fileciteturn12file1L640-L645 fileciteturn7file0L40-L44

这里应记录原始 NLL 差、任务准确率差和最终模型相对原 checkpoint 的输出 KL。多个来源中的 retention 比率不能直接平均，更不能把旧路由的原生保持计入单表结果。

### 1.4 同一表在检索和多跳问答上的差异，排除了简单的长度阈值解释

9 月 2 日备忘录记录：在 38 个 Native-short-correct HotpotQA 样本上，`log_s4` 的 8K F1 为 0.677，YaRN 为 0.656；差异未被该样本量明确识别。16K 时分别为 0.153 和 0.500。该表同时在另一组 16K RULER 上优于对应 YaRN。这里的 Hotpot 数字属于备忘录中的历史报告，原始逐条产物尚未在本轮独立复核。fileciteturn7file0L46-L61

同样长度能够出现相反的方法排序，说明“频谱覆盖到 16K”不足以描述任务计算需求。也不能从这些结果推出 4× 是所有静态表的数学极限。

### 1.5 最新 LoRA 数据直接否定了“只修终止”的解释

以下为含 ON 回填的报告。组级严格成功要求两个证据世界均完整回答并终止；64 行诊断则按单个世界计数。

| OLMo，Native 窗口 4K，far 为 16K | far 严格成功，组 | far EOS | far 宽松含答案，行 |
|---|---:|---:|---:|
| T0：原表，无训练 | 0/32 | 0.00 | 0/64 |
| Z0：静态表，无训练 | 1/32 | 0.62 | 26/64 |
| ZC：静态表，compact-only LoRA | 1/32 | 0.70 | 27/64 |
| ZF：静态表，全布局 LoRA | 4/32 | 0.91 | 29/64 |
| ON：原表，全布局 LoRA | 0/32 | 1.00 | 0/64 |

ZF 中“正常终止但没有正确答案”为 32/64 行。ZC→ZF 的逐行精确答案从 3 增至 14；因此 LoRA 改变了内容利用或选择。相近的“宽松含答案”数量，也不意味着表已经决定内容上限：它没有记录正确答案的排序、完整性和两个证据世界中的一致响应。fileciteturn7file1L256-L290

ON 与 ZF 的差异说明，两套完整训练/运行配置在这组任务上表现不同。训练时的表、最终权重以及运行时的表已经共同变化；若 gain、端点也不同，更不能把差异全部归为纯内部 $z$ 的效果。

### 1.6 Qwen 的适配成功有价值，但目前是窗内结果

Qwen2.5-1.5B 按报告的 32K 窗界，16K 测试属于窗内。静态表从 3/32 经 LoRA 提高到 23/32，compact-only 为 20/32，说明小规模适配能在这一条件下显著改善生成。它尚未验证同一方法能把原有能力迁移到模型没有见过的相位范围。Native 保留也仍是点值可行、区间未确认的状态。fileciteturn7file1L249-L254

这与 OLMo 的结果并不矛盾：两者的语义能力、原生窗界和相位暴露条件都不同。单凭跨模型比较，无法把差异唯一归因于模型大小。

### 1.7 已经失败的方案对下一步有什么约束

历史记录包含：小样本 64 维频率梯度在开发集改善、独立集失败；gain 在一个短任务上的最优值无法迁移；Q/K、Q/K/V/O、首 token 修补、位置蒸馏和后处理路线均有失败或不足。fileciteturn7file0L55-L65 fileciteturn8file0L21-L29

历史备忘录还说明，`log_s4` 的逐槽位数值可由原生几何确定，但 mask 幂次与采用 log law 的选择看过长任务结果。因此应称“已冻结构造在新样本上的验证”，不能追溯性地称为完全没有长任务选择的发现。fileciteturn7file0L68-L107

这些事实支持继续研究功能兼容性和有限适配；它们同时要求新训练具有一个能被记录的实际改动。将 CE、margin、Native KL 或全线性 LoRA 重新命名，信息增益为零。

---

## 2. 从 frequency = base ** exponent 到真正进入 attention 的对象

### 2.1 固定表示约定，才能识别指数分配

标准形式可写为

$$
\omega_k=b_0^{-e_k},\qquad e_k=\frac{k}{K},\qquad k=0,\ldots,K-1.
$$

给定一个正频率表，定义

$$
x_k=-\log\omega_k=a+Rz_k,\qquad z_0=0,\quad z_{K-1}=1.
$$

于是

$$
\omega_k=e^{-a}(e^R)^{-z_k}.
$$

$a$ 和 $R$ 确定采样范围，$z$ 决定其中 $K-2$ 个内部位置。标准实现的 base 为 $b_0$ 时，几何表的实际跨度是 $(K-1)\log b_0/K$，不能把 base 数字直接当成实际端点。

base 与未归一化 exponent 都自由时，存在重参数化自由。论文的独立性来自**固定允许改变的量**：在相同端点下，几何分配固定，而非几何分配仍有内部自由度。这是实验识别的基础；代数改写本身不承担主要新颖性。

每个实验先从最终 `inv_freq` 还原 $a,R,z$。例如

$$
\omega'_k=\omega_k s^{-m_k}
$$

通常也改变采样范围。与原表的比较是完整频率干预；与该新范围中的几何表比较，才隔离内部分配。保留同一个 `rope_theta` 字段，并不能证明实际范围相同。

### 2.2 纯位置 kernel 与内容条件化 kernel

只观察正弦余弦，可以定义

$$
\phi_\Omega(\Delta)=
(\cos\omega_0\Delta,\sin\omega_0\Delta,\ldots,
\cos\omega_{K-1}\Delta,\sin\omega_{K-1}\Delta),
$$

以及纯位置内积

$$
\phi_\Omega(\Delta)^\top\phi_\Omega(\Delta')
=\sum_k\cos\{\omega_k(\Delta-\Delta')\}.
$$

它能描述位置特征的分辨与重叠。模型使用的分数还包含 Q/K 的内容系数。

取 $\Delta=p_j-p_i$，$J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，$R(\theta)=I\cos\theta+J\sin\theta$。对一层的固定输入，RoPE 的精确展开为

$$
s_{ij}=\frac{\alpha}{\sqrt d}\sum_k
\left[C_{ij,k}\cos(\omega_k\Delta)+D_{ij,k}\sin(\omega_k\Delta)\right],
$$

$$
C_{ij,k}=q_{i,k}^{\top}k_{j,k},\qquad
D_{ij,k}=q_{i,k}^{\top}Jk_{j,k}.
$$

其中 $\alpha$ 为最终 attention logit 的倍率。RoPE 的原始推导给出了这种相对位置双线性结构。[R1] citeturn111172search0

在线性 Q/K 投影的情形，也可写成矩阵值函数

$$
\mathcal K_{W,\Omega}(\Delta)
=W_Q^\top\operatorname{diag}_k R(\omega_k\Delta)W_K,
\qquad s_{ij}=\frac{\alpha}{\sqrt d}h_i^\top\mathcal K_{W,\Omega}(\Delta)h_j.
$$

模型存在额外 Q/K normalization 时，以上矩阵值形式需要纳入该算子；用实际归一化后的 $q,k$ 表示的逐对展开仍成立。

**论文需要连接的是：$z$ 怎样改变 $\mathcal K_{W,\Omega}$ 的实际用途。** 纯位置 Gram 删除了内容系数、attention 竞争和后续 value 读出，不能独立预测全部模型行为。

### 2.3 低频位置函数接近常数，不会使相应内容分数自动被 softmax 消除

慢频率满足

$$
C_{ij,k}\cos(\omega_k\Delta)+D_{ij,k}\sin(\omega_k\Delta)
=C_{ij,k}+D_{ij,k}\omega_k\Delta+O((\omega_k\Delta)^2).
$$

其中 $C_{ij,k}$ 随候选 token $j$ 的内容变化。softmax 对一整行的共同常数不敏感，却对这组不同的 $C_{ij,k}$ 敏感。

即使若干频率都为零，相应通道仍贡献

$$
h_i^\top\left(\sum_k Q_k^\top K_k\right)h_j.
$$

这些位置因子全部相同，内容矩阵的秩仍可随通道数增加。于是“位置函数重叠”不能直接推出“可以无代价挪走这些通道”。低频内容用途在成熟模型中已有实证研究。[R3] citeturn478784view4

这为已有失败提供了具体解释方向：从头训练可重新分配内容系数，冻结换表要承受原配对产生的变化。它没有证明每个慢频率都必须保留。

### 2.4 ordered coupling 的正确理论地位

只交换频率槽位会改变各项 $C,D$ 与频率的对应，因此可以严重退化。若连同 Q/K 的相关旋转坐标一起做一致重编号，双线性计算可以精确保持。

所以有序配对的重要性属于已学系统的关系属性。joint relabeling 是一个很小的实现控制：先用 CPU 张量测试；仅在实际代码路径尚未通过时，补一条完整模型 parity。已有 151.9M 权重×表交叉已展示了实际共同适应，没必要重新训练一个模型再证明交换会退化。fileciteturn12file0L182-L199

### 2.5 多层传播必须进入解释

冻结参数不等于冻结每层的 $q,k$。第一层输入可相同；前面层的表变化会改变后续隐藏状态。固定 Q/K 的分数重算，只测该层在指定输入下的直接影响。它应与完整模型实际输出一起报告，不应被包装为全网络精确预测。

### 2.6 指数的梯度为何会平坦、翻转或在小样本上过拟合

固定 $a,R$ 和当前 Q/K，$\omega_k=e^{-(a+Rz_k)}$ 给出

$$
\frac{\partial s_{ij}}{\partial z_k}
=\frac{\alpha R\omega_k\Delta}{\sqrt d}
\left[C_{ij,k}\sin(\omega_k\Delta)-D_{ij,k}\cos(\omega_k\Delta)\right].
$$

最终目标对 $z_k$ 的梯度，还要乘上 $\partial\mathcal L/\partial s_{ij}$，再对所有层、查询与候选求和。同一共享 $z_k$ 在不同距离、内容和 head 上的贡献可以异号；短输入中某些慢频率的直接梯度还会随 $\omega_k\Delta$ 变小。这说明“指数只有几十个参数”并不意味着少量样本就能可靠估计其长任务方向。

这与已报告的 learnable-τ 信号平坦、小样本频率梯度未通过 holdout 相容。前者还可能受参数化、loss 与训练实现影响；本轮没有原始梯度，不能据此宣称已定位唯一原因。它们也不能推出所有频率学习都无效。fileciteturn7file1L304-L308 fileciteturn7file0L63-L65

正因共享分配汇总了这些不同计算需求，理论不能只要求“把平均几何误差减小”。目前先解释已冻结分配的成功和失败，比再次用开发集梯度给全部槽位赋值更有依据。

---

## 3. 零训练怎样兼顾原生保持与外推

### 3.1 要保持的是实际函数及正确决策

记原模型为 $f_0=f(W_0,\Omega_0,\alpha_0)$，部署模型为 $f=f(W_0,\Omega,\alpha)$。原生代价至少包含

$$
D_N(f)=\mathbb E_{c\sim\mu_N}
\operatorname{KL}\left[p_{f_0}(\cdot\mid c)\|p_f(\cdot\mid c)\right]
$$

和独立原生任务上的准确率差。$\mu_N$ 应覆盖自然文本及模型实际能做的指令任务。微小平均 KL 不能保证每条贪心轨迹相同。

“强外推”则需原生窗口以外的真实输入、完整自回归生成和有效扩展基线。新训练见过 16K 后在 16K 成功，属于相对原 checkpoint 的上下文扩展；只有更长且未暴露位置的测试，才检验相对适配阶段的长度外推。

### 3.2 有限换表的直接影响可以精确计算，不必依赖局部 Fisher

固定一行 Q/K，记原分数为 $s$，新分数为 $s+\eta$。有

$$
p'_j=\frac{p_j e^{\eta_j}}{\sum_rp_r e^{\eta_r}},
\qquad
\operatorname{KL}(p\|p')=
\log\mathbb E_p e^\eta-\mathbb E_p\eta.
$$

这是有限干预恒等式，允许相位跨越多个周期。它说明共同的分数平移没有影响，候选之间的相对扰动才重要。若所有 $\eta_j$ 落在宽度为 $w$ 的区间，Hoeffding 界还给出 $\operatorname{KL}(p\|p')\le w^2/8$；实际评估优先使用精确式，避免用很松的最坏情况界选表。

对单个 token 对，固定 Q/K、固定 gain 时：

$$
|s'_{ij}-s_{ij}|
\le\frac{2\alpha}{\sqrt d}\sum_k
\|q_{i,k}\|\|k_{j,k}\|
\left|\sin\frac{(\omega'_k-\omega_k)\Delta}{2}\right|.
$$

这个式子直接解释了 frequency MAE 很小仍然失效的可能性：距离、内容投影范数、相位位置和候选间抵消都参与实际误差。它是上界；不能反过来把上界偏大等同于模型一定退化。

### 3.3 scalar gain 能修正什么

在固定 Q/K 下，正 scalar 乘全部分数，不改变这一行的候选排序。它可以改变集中程度。对整个多层模型，后续表示会受影响，最终答案当然可能改变。

YaRN 本身结合了按频段处理与 attention 温度调整；其 $1+0.1\log s$ 对应 Q/K 一侧的幅度约定，不能自动当成最终 logit 倍率。[R2] citeturn478784view0

对当前全旋转模型，若 cos/sin 同时使 Q、K 乘以 $c$，最终倍率为 $c^2$。部分旋转时，旋转部分与非旋转部分可能受到不同作用，必须查代码。`gain=1.1026` 的名字本身不能确定它属于哪一种。

为判断既有表的原生退化有多少是幅度失配，可在固定原生 Q/K 上求一个全局标量：

$$
\alpha^*=\arg\min_{\alpha\in[\alpha_{\min},\alpha_{\max}]}
\sum_i w_i\operatorname{KL}
\left[p_{0,i}\|\operatorname{softmax}(\alpha u_i)\right].
$$

这里 $u_i$ 为新表在同一隐藏状态下的未乘倍率分数，$w_i$ 为预先固定的样本权重。其导数与二阶导数为

$$
J'(\alpha)=\sum_iw_i\left[
\mathbb E_{\operatorname{softmax}(\alpha u_i)}u_i-\mathbb E_{p_{0,i}}u_i\right],
$$

$$
J''(\alpha)=\sum_iw_i\operatorname{Var}_{\operatorname{softmax}(\alpha u_i)}(u_i)\ge0.
$$

可用有界二分求解，无需扫描长任务。它只给出固定激活的最优全局 gain；完整网络仍要独立验证，也不能以它替代输出 KL。已有成功 gain 已经冻结的确认实验，应保持原配置；这个计算产生的新标量属于新的原生校准方案。

若最优 gain 后局部分布仍明显失配，继续以同一个标量搜索多跳生成通常缺乏针对性。反过来，局部残差小也不足以保证全网络或长任务正确。

### 3.4 从头训练与冻结外推，为何可能需要相反的频率移动

从头训练时，增加某些可辨识相位尺度，可能帮助模型建立相应计算。冻结部署时，原来的低频内容配对已经存在，减慢其中部分频率可能有助于保持远端对齐。两种作用可以同时成立。

自然长输入中的关系也没有统一伸长倍数。插入远端背景通常保持句内邻接，却增加证据到问题的距离。统一压缩所有相位会把局部关系一起压缩；完全不变又会使某些远端关系暴露于未训练相位。按频率选择运动，提供了处理不同关系尺度的自由度；能否成功取决于内容用途如何分布到槽位。关于数据依赖尺度与插值条件，已有相关理论，应作为依据而非重新宣称首创。[R4] citeturn478784view5

这解释了三路线可以使用不同非几何表。它不能推出“高频一律不动、低频一律缩放”在任何 checkpoint 上都是最优规则。

### 3.5 对当前零训练路线的可执行判断

保留已有 `log_s4` 和已经注册的 coarse/derived 对照，不再从几十个新曲线中寻找偶然赢家。先核实实际张量，重用已有原生输出；缺少固定 Q/K 记录时，只补少量预定层与查询位置。

执行顺序是：**计算既有表的原生分数变化 → 检查独立 gain 能消除多少变化 → 在原生完整模型上验证总损失 → 冻结一个工作点 → 在独立长任务上确认。**

这一路线的成功条件是，同一张表在原生任务上保持足够小的功能变化，并在长任务上形成正确答案决策。现有频谱恒等式不能保证这两个条件必然有交集。

若既有候选都达不到极低原生代价，本轮零训练结论就是这些候选的已测性能边界。继续研究单表目标仍然合理；当前预算应先用于适配能否修复已知损害，而不能借用 Native 路由的数据宣布单表已经成功。

## 4. LoRA 怎样修复原生损害，并把修复迁移到长输入

### 4.1 它需要改变 Q/K 与频率的配对，同时控制整网功能变化

固定某层输入，频率与 Q/K 的一阶变化给出

$$
\delta s_{ij}=\frac{\alpha}{\sqrt d}
\left[(\delta q_i)^\top R_\Omega k_j+
q_i^\top R_\Omega\delta k_j+
q_i^\top(\delta R_\Omega)k_j\right]
+\frac{\delta\alpha}{\sqrt d}q_i^\top R_\Omega k_j+O(\|\delta\|^2).
$$

单独改表产生第三项；Q/K 适配可以通过前两项抵消部分原生误差，或在长输入中建立更有利的内容排序。V/O、MLP 适配还会影响写入残差流的内容与后续处理。能训练哪些模块，应由实际误差与已有对照决定。

不存在把两个不同频率集合在所有内容、所有相对位置上精确互换的固定可逆 Q/K 补偿，并不意味着有限数据分布上的近似适配失败。该精确结论的量词过强，不能用来关闭 LoRA。反过来，允许全线性 LoRA 也不保证有限训练会找到合适补偿。

目前全线性 r16 已经执行，Q/K-only 也有历史对照。新增一个“更广模块”的配方不再提供足够信息。旧 Q/K-only 实验还继承了不同父分支的 V/O，不能仅凭它较好就断言所有原生损害来自 V/O。fileciteturn9file0L39-L54 fileciteturn12file0L416-L435

### 4.2 原生保持的约束必须从换表后的非零误差开始

设候选表为 $\Omega_Z$，adapter 为 $v$。实际原生约束是

$$
D_N(v)=\mathbb E\operatorname{KL}
\left[p_{W_0,\Omega_0,\alpha_0}\;
\|\;p_{W(v),\Omega_Z,\alpha_Z}\right].
$$

在 $v=0$ 时一般已有 $D_N(0)>0$。对输出 logits 作局部展开：

$$
e_N(v)=e_N(0)+J_Nv+O(\|v\|^2).
$$

因此训练首先需要补偿既有表误差，而不只是让新 adapter “不要再退化”。将“表造成的 KL”和“adapter 相对换表起点的 KL”相加，不会得到最终部署函数相对原模型的 KL。原协议已经对此设有恢复阶段与实际函数约束。fileciteturn9file0L124-L147 fileciteturn9file0L167-L185

在小扰动条件下，可以明确描述 LoRA 能修复哪一部分。令 $A$ 是频率变化对原生输出的 Jacobian 经参考分布 Fisher 白化后的矩阵，$B$ 是允许的 adapter 参数变化对应的白化 Jacobian，则

$$
\min_v\|A\delta z+Bv\|^2
=\delta z^\top A^\top(I-BB^\dagger)A\delta z.
$$

推导是将 $A\delta z$ 正交投影到 $B$ 的列空间。它把原生误差分成可被当前适配方向抵消的部分与剩余部分。

**这个关系只解释局部可修复性。** 剩余原生误差小，不等于长任务会做对；补偿还可能同时抵消原有长程收益。有限相位变化、隐藏状态迁移和 LoRA 双因子的训练动态也会超出这个线性近似。当前方案不要求估计一个巨大的 Jacobian 或 Fisher；优先用已训练适配器的实际交叉行为检验这种兼容性。

### 4.3 已有结果首先留下了“没学会”与“没迁移”的分歧

现在掌握的是若干最终测试分数。它们不能替代以下两个读数：**同一训练样本上的完整长答案是否已经学会；学会后的修正是否迁移到独立实例和未见长度。**

| 训练集 far | 独立验证 far | 原生保持 | 最直接的诊断方向 |
|---|---|---|---|
| 仍明显失败 | 失败 | 可行 | 学习信号、优化、可用适配方向或任务容量 |
| 明显成功 | 失败 | 可行 | 语义/布局覆盖不足，或适配没有跨长度迁移 |
| 明显成功 | 成功 | 失败 | 现有修正与原生功能冲突，或约束未有效执行 |
| teacher-forced 规范轨迹全部 margin 为正，实际 greedy 却偏离 | — | — | 检查同一解码合同下的 cache、位置、模板、logit processor 与数值差异 |

最后一行有严格依据。对固定规范答案 $y_{1:T}$ 及真实终止 token，定义

$$
m_t=\ell(y_t\mid x,y_{<t})-\max_{v\ne y_t}\ell(v\mid x,y_{<t}).
$$

若所有 $m_t>0$，且实际贪心解码使用相同 logits、处理规则和起始输入，则逐 token 归纳保证生成该答案并终止。一般自然语言有多个有效答案，负 margin 只能说明这一条规范轨迹未被保证，不能自动判语义错误。

全答案 CE 和最差 margin 已在旧协议中。需要补的是它们在训练/验证、不同长度、正确与错误证据世界上的实际轨迹，而不是再发明一次相同的 loss。fileciteturn9file0L86-L116

### 4.4 为什么 EOS 可以学好，而远端内容仍学不好

对一行 attention，$o_i=\sum_jp_{ij}v_j$，设 $g_i=\partial\mathcal L/\partial o_i$，有

$$
\frac{\partial\mathcal L}{\partial s_{ij}}
=p_{ij}\,g_i^\top(v_j-o_i).
$$

LeRoPE 的频率梯度分析已经包含这条 attention 链式关系及其与相位导数的联系。[R5] citeturn478784view3

它提供一个具体、但尚未在当前 run 中识别的解释：**若某条本应使用的远端证据路径获得极小的 attention，而 value 范数与下游梯度有界，答案监督传到这条路径的 score 梯度也会很小。** 模型仍可能沿较容易的路径学会格式、终止和常见答案。

这与 ON 的 EOS=1、内容=0 相容，也与 ZF 部分恢复答案相容；但相容不是完成归因。一个 head 没有关注源 token，不代表信息没有经前层、其他 head 或中间 token 传递。应结合源证据干预、任务正确性以及少量真实梯度读数判断。

现有 ZF 已经使用全布局长输入，因此“缺少任何长输入训练”不能作为它的解释。真正还待核对的是：实际训练样本能否拟合、关键决策的梯度是否到达相关计算，以及原生约束是否阻碍这一更新。

### 4.5 为什么只扩大 position IDs 不足以模拟真实长输入

正确来源分数为 $s_E$，$n$ 个干扰来源分数为 $s_j$。记

$$
\ell_D=\log\left(\frac1n\sum_{j=1}^{n}e^{s_j}\right).
$$

则正确来源获得至少 $p_*$ 的 attention 质量，等价于

$$
s_E-\ell_D\ge\log n+\log\frac{p_*}{1-p_*}.
$$

在干扰分数的 log-mean-exp 近似不变时，干扰数扩大四倍需要额外 $\log4$ 的分数优势。真实文本不一定满足“干扰分布不变”，所以它是有条件的定量解释，不能将所有 4× 失败归因于这一个数。

跳跃位置的短输入改变相位，却没有加入相同数量和内容的竞争者。真实长输入还会改变后续隐藏状态。因此物理长度、最大位置编号、证据距离与干扰竞争应分别记录。旧训练中的短物理序列加长位置已经明确披露，不能把它描述为从未见过长相位的纯外推。fileciteturn12file0L334-L341

### 4.6 一项有启动条件的修正：让答案监督能够到达已确认的证据路径

本节只适用于以下观测同时成立时：训练 far 尚未拟合；原生约束仍可行；受控单证据任务已经确认目标来源；实际读数显示证据路径的 attention 与答案监督 score 梯度很小。否则不启动这项训练。

固定原生 compact teacher、证据集合 $E$ 和预先指定的查询位置。记 teacher 在该证据集合上的 attention 总质量为 $r$，长输入 student 对相同语义来源的质量为 $P=\sum_{j\in E}p_j$。可增加

$$
\mathcal L_E=\operatorname{KL}\{\operatorname{Bern}(r)\|\operatorname{Bern}(P)\}.
$$

对源内和源外 token 的 score 梯度分别为

$$
\frac{\partial\mathcal L_E}{\partial s_j}
=\begin{cases}
(P-r)p_j/P,&j\in E,\\
-(P-r)p_j/(1-P),&j\notin E.
\end{cases}
$$

当 $P$ 很小时，源内梯度趋于 $-r p_j/P$，不再整体乘上极小的 $P$。这是上述特定梯度不足的一种直接处理方式。实现时用源内、源外 log-sum-exp 计算二元 KL，避免先算极小概率再取对数。

它仍有明确假设：compact teacher 的证据访问应当是长输入需要保留的计算；更高的证据质量应当确实有助于当前错误决策。它不保证修复多跳组合，也不保证 attention mass 上升就等于正确生成。零训练配置完全不受此训练辅助项影响。

**执行限定如下。** 使用现有 r16 全线性适配器和完整答案目标，保持实际 Native 约束；只在已验证的单证据/绑定训练项上使用该辅助项。固定少量均匀间隔层，在答案开始前的查询位置重算所需 attention 行；不根据最终任务得分选择最佳 heads，不改标准 attention 前向或 KV 语义。teacher 的 $r$ 随 head 自身用途变化，不把全部 head 强制为检索头。辅助项平均方式固定，初始系数可设为 0.1，作为待测工程设置记录，不能称为理论最优值。基线与候选表使用相同的辅助监督和 token 预算。

历史上已经尝试位置蒸馏。执行者必须先核对旧 loss：若已在相同查询、来源集合、约束与训练条件下做过这一辅助项，且同一断点仍存在，则本节没有新的实验价值。这里的改动必须落实为“当前答案梯度不足的证据 + 有别于旧实现的监督路径”，不能只改方法名称。Native teacher 修复与位置蒸馏本身也已有 LongReD 等先例，不承担本文新颖性。[R6] citeturn328639search2

### 4.7 其他诊断结果对应怎样的行动

**训练 far 已学会、独立实例失败：** 下一次唯一改动应是独立语义实例和布局的覆盖。保持总 token、模块、rank、表和更新预算，减少同一实例的重复呈现，把预算给独立来源与反事实变化。需要检验的是固定表下的适配迁移。不能因错误发生在 16K 就继续增大 rank 或再修 EOS。

**训练与验证都改善，但原生约束失败：** 先核对实际部署 KL、分层约束、乘子轨迹及保存点。若只是约束实现偏离已声明协议，修复后按同一协议执行。若有效约束下始终冲突，报告该表及适配预算下的折中；只有另一张既有表已显示更低原生修复代价时，才有理由改用它进入新版本的配对比较。

**答案及证据路径的监督都无法拟合小规模训练集：** 这时才有依据考虑模型或可训练范围是否不够。结论仍依赖已测优化条件；不直接把它解释成整个模型族、所有 LoRA 或所有静态表的无解证明。

这三个分支与上一节互斥地确定一次试验的主要改动。先由旧 checkpoint 的读数确定分支，不能把全部分支跑完后挑一个最好结果。

---

## 5. 把三条路线写成同一篇论文

### 5.1 用“同一个变量、不同可调整权重集合”统一

定义模型结果为

$$
\mathcal R(W,a,R,z,\alpha;\mu,\text{decoder}).
$$

三个路线只需遵循同一套变量定义：

| 路线 | 权重如何得到 | 固定范围对照回答的问题 |
|---|---|---|
| From-scratch | $W_z=\mathcal A_B(W_{\rm init};z)$ | 分配如何影响权重学习及随后的长度泛化 |
| Zero-training | $W=W_0$ | 在已学内容配对下，仅改变内部分配能产生什么行为变化 |
| LoRA | $W_z=W_0+\Delta W_B(z)$ | 有限适配能否修复配对变化，并将分配收益转化为真实生成 |

$\mathcal A_B$ 表示固定数据与预算 $B$ 的训练算法，并不假设求到全局最优。LoRA 两臂最终权重不同，是训练时分配干预产生的后果；只要初始化、训练规则、数据、预算配对，便可研究分配的总训练效应。

**统一主线不要求三条路线使用同一条曲线。** From-scratch 中成立的初始化分配，未必适合直接安装到成熟 checkpoint；成熟模型中有效的分配，也未必是从初始化训练时最好的选择。这正是 Q/K-frequency coupling 使安装阶段具有科学意义的原因。

### 5.2 用有限交叉分解直接影响与训练响应

在同一个范围和 gain 下，记 $L_{ab}=L(W_a,z_b)$，其中 $a,b\in\{G,Z\}$。完整训练结果的差有两个同样精确的表达：

$$
L_{ZZ}-L_{GG}
=\underbrace{(L_{GZ}-L_{GG})}_{\text{在几何训练权重上换表}}
+\underbrace{(L_{ZZ}-L_{GZ})}_{\text{在新表下改变已学权重}},
$$

$$
L_{ZZ}-L_{GG}
=(L_{ZG}-L_{GG})+(L_{ZZ}-L_{ZG}).
$$

两种路径的组成项由交互量

$$
I=L_{ZZ}-L_{ZG}-L_{GZ}+L_{GG}
$$

联系。它们是四个完整模型评估值的有限恒等式，不使用小扰动假设，也不需要估计全模型 Jacobian。不能把某一路径的两项宣布为唯一、与顺序无关的因果贡献；报告四格及交互即可。

这个分解能够把一个常见现象写清楚：**$L_{GZ}-L_{GG}$ 可以很差，而 $L_{ZZ}-L_{GG}$ 可以很好。** 前者是冻结安装的兼容性，后者包含训练对新分配的响应。From-scratch 成功与直接换表失败，因而可以同时为真。LoRA 检验有限预算能够补上多大一部分这种响应。

对固定训练算法，还可将训练状态记为 $S_{t+1}=U_t(S_t,z)$。在可微且数据次序固定的条件下，

$$
B_{t+1}=\partial_SU_t\,B_t+\partial_zU_t,
\qquad B_t=\frac{\partial S_t}{\partial z},\quad B_0=0.
$$

测试结果随 $z$ 的变化包含直接频率项，以及通过 $B_T$ 产生的学习响应项。冻结路线没有训练响应；LoRA 限制响应可经过的参数；从初始化训练允许更广的共同适应。这条链式关系说明三种路线为何没有必须同号或同幅度的性能变化。本轮使用有限交叉观察实际响应，不为计算 $B_T$ 重跑训练或建立新的元学习工程。

### 5.3 每条路线都需要与其主张匹配的对照

记 $z_G$ 为某个给定范围内的几何分配，$z_Z$ 为该范围内的非几何分配。冻结效应为

$$
\Delta_0=
\mathcal R(W_0,z_Z)-\mathcal R(W_0,z_G).
$$

在相同适配预算下，效应为

$$
\Delta_B=
\mathcal R(W_B(z_Z),z_Z)-\mathcal R(W_B(z_G),z_G).
$$

$\Delta_B-\Delta_0$ 描述所测训练协议怎样改变分配效应。这是配对实验的差分，不能单凭它把每个中间机制唯一定位到某个 head 或模块。

原始 Native 表、YaRN 表、非几何新表通常有不同端点或 gain。它们的比较承担实际方法评价；若要把 LoRA 收益归到 $z$，至少要有一个**相同最终采样范围与共同 gain 的几何 LoRA 臂**。已有 N/Z/Y 不能自动替代这一控制。fileciteturn8file0L82-L90

从理想可实现集合看，允许零更新的 LoRA 包含冻结权重的情形，因而最优可实现折中不应因增加自由度而变差。但实际有限训练可能优化失败、过拟合或违反约束；从集合包含关系不能推出一次 LoRA 实验必胜。From-scratch 使用不同起点和学习历史，也不能直接接到这条集合包含关系后面，假定三个结果必须单调排序。

### 5.4 文章的三个核心论点及其证据

**论点一：内部分配具有独立效应。** 使用已有固定范围三种子训练，以及成熟模型固定范围的 uniform/coarse/derived 比较。数学分解提供控制变量，实验提供行为证据。

**论点二：效应依赖已经学到的内容配对。** 使用已有权重×表交叉、joint relabeling 的精确计算，以及最新适配器的运行表交叉。纯位置有效秩可以描述位置函数重叠；内容系数、softmax 和后续计算解释它为什么不足以排序模型。

**论点三：在明确预算下，这项自由度可以带来多大实际收益。** 用同表全时启用的原生–长程结果、匹配几何适配臂和有效外部基线回答。最有价值的新增正结果是：在独立测试中，原生代价很低，真实长生成有稳定收益，而且收益无法由同预算几何表适配完全解释。

这三个论点形成从设计变量到已学计算再到实际收益的完整叙事。若第三项尚未实现，前两项仍有研究价值，但摘要必须明确实际联合目标尚未解决，不能用代理指标或旧路由补齐。

### 5.5 EVQ-Cosh 在新版中的位置

保留它作为已经研究充分的解析实例，用于展示在固定范围内实现非几何分配，以及明确 surrogate 下的可解构造。不要让 Cosh 的最优性承担所有成熟模型结果的解释。

现有 exponential、coarse 与 derived 对照是正文主线的资产：它们帮助判断效应属于一类分配，还是只属于某一公式。不同实例效果接近，支持前一种解释，同时限制唯一最优曲线的主张。

有效秩、慢频率极限、精确移植限制可以保留，但只陈述其数学范围。此前提出的 Fisher、投影和最小二乘关系也只属于分析工具。它们能否提前预测新条件、帮助减少适配或改善真实结果，才决定是否值得占据正文的重要位置。

### 5.6 竞争力需要落在一个被证据支持的增量上

LeRoPE 已经研究非几何频率及权重共同适应；AdaRoPE 已经研究 head 间的频率与缩放差异。你们需要把贡献落在固定范围的独立归因、不同安装阶段的对照，以及受限适配下的实际联合结果。不能只以“没有学习频率参数”或“用了新的坐标名”作为差异。[R5、R7] citeturn478784view3turn328639academia24

目前与这些工作的区别可以成为清楚的研究问题，但尚不能仅凭区别宣布竞争力足够。**如果新增实验只能再证明非几何表比失效的原表好、仍无法保持原生或超过有效扩展基线，文章的实用贡献仍偏弱。** 如果得到一个稳定的联合工作点，并能用匹配范围与交叉干预解释它为何需要这种分配，已有训练资产就能与之组成更有说服力的论文。

建议标题直接围绕 exponent allocation、learned coupling 与 context generalization；摘要首先写固定范围的识别结论，再写经确认的实际结果。避免把全部路线写成“EVQ-Cosh 的多场景验证”。

## 6. 只保留四项实验，其中正式训练须由前面的结果决定

### 6.1 E0：把会改变结论的现有资产对齐，主要使用 CPU

本项只处理当前主 checkpoint、ON/ZF/ZC、对应频率表和正在使用的评测数据。一次性记录：模型与 tokenizer 标识、adapter 标识、最终频率张量、$a,R,z$、实际 logit 倍率、训练物理长度与最大 position ID、来源文档/语义实例 ID、输出 token、评分器版本、Native teacher 配置。

原协议、执行配置和结果报告是三种不同证据。报告写“同配方”时，仍应检查实际 loss 和 Native multiplier 是否按协议执行。缺失的训练日志标为缺失，不能从协议文本推断已经执行。

历史 RULER 存在同名配置记录 0.5486 与 0.49859 的冲突；只需追踪到会用于本文的主结果，找到实际行集合和 run 归属。不得取较大值、平均或凭日期合并。fileciteturn8file0L49-L53

已有算子 parity 不必每轮重做。只有修改了实现、首次引入观察钩子，或者遇到正 margin 与 greedy 输出矛盾时，检查 identity/cache parity。`inv_freq` 和实际 gain 应在不同长度请求中保持不变；每次换表重新 prefill，禁止复用另一张表旋转后的 KV。

**产物：** 一张当前实验 manifest 和可直接读取的逐条预测表。CPU 整理设一个工作日上限；不把追溯全部历史实验设成后续工作的前置条件。

### 6.2 E1：现有权重×运行配置交叉，同时读取训练是否拟合

这是本轮优先级最高的 GPU 支出。记 $T_0=(\Omega_0,\alpha_0)$，$T_Z=(\Omega_Z,\alpha_Z)$。使用已经训练完的 ON 和 ZF 权重：

| 固定权重 | 运行 $T_0$ | 运行 $T_Z$ |
|---|---|---|
| $W_{ON}$ | 已有 ON 对角结果 | **补测** |
| $W_{ZF}$ | **补测** | 已有 ZF 对角结果 |

先在已有诊断实例的 compact/near/far、两个合法证据世界上补齐缺少的两个格子。严格复用对角格的输入、评分和生成规则；若其原始数据无法对齐，则四格统一评估，不能拼接不同样本。

这个交叉暂时称为 **运行配置交叉**，因为 $T_0,T_Z$ 可能同时改变端点、内部分配和 gain。它测已学权重与部署配置之间的关系，纯 $z$ 归因由固定范围实验承担。

不同结果对应不同决策：

| 交叉结果 | 支持的解释 | 对下一次训练的影响 |
|---|---|---|
| $W_{ON}$ 换到 $T_Z$ 后接近 $W_{ZF},T_Z$ | 旧训练中存在可跨运行配置复用的改进；运行配置本身贡献较大 | 优先利用已有通用适配，检查单表原生代价，避免重复完整长训练 |
| $W_{ON},T_Z$ 仍明显差，只有 $W_{ZF},T_Z$ 改善 | 在所测训练预算下，与该运行配置共同适应有额外价值 | 继续研究表条件化适配，并保留匹配几何训练对照 |
| $W_{ZF}$ 回到 $T_0$ 后原生行为明显改变 | 适配已形成依赖新运行配置的补偿 | 原生验证必须始终使用最终部署配置 |
| 四格 far 都处于很低水平 | 交叉没有找到强能力工作点 | 依据训练/验证拟合分歧决定下一项修正，不扩展曲线或模型搜索 |

这些结果都不能单独定位唯一 head、证明低频是唯一瓶颈，或证明某个训练方式普遍必要。对连续指标，可报告

$$
I=[L(W_{ZF},T_Z)-L(W_{ZF},T_0)]
-[L(W_{ON},T_Z)-L(W_{ON},T_0)].
$$

它量化这两套固定权重与运行配置的交互；用语义实例配对，分别报告原生、near、far。

**同一次评估还要读取训练拟合。** 先取现存 step 0、32、128；有必要且 checkpoint 已保存，再读取 64/96。按固定语义 ID 取训练 far 与独立验证 far，记录完整答案 CE、规范轨迹最差 margin、实际 greedy 正确性、EOS，以及实际原生输出 KL。训练 loss 曲线不能替代训练生成正确性。

两组数据都采用完整样本与原生 compact 可解子集两种视角。不要因为方法在 compact 上损伤了某些样本，就把它们从该方法的分母中移除。已有 ZF 的 compact 27→25 必须继续显式报告。fileciteturn7file1L268-L275

**信息收益：** 同时决定收益是否需要表条件化训练、现有失败主要属于拟合还是迁移。它增加的是缺失的反事实格子与训练端读数，几乎不增加方法自由度。

### 6.3 E2：只有仍存在具体机制分歧时，补极少量函数读数

E1 足以选择后续动作时，跳过本项。最多补两类读数，并且不建立全 head 搜索项目。

**原生兼容性读数。** 固定 4 个均匀间隔层、固定查询位置，对现有表重算相同实际 Q/K 下的 score 变化、精确 attention KL 和单个全局 gain 能消除的部分。保存所需 attention 行或聚合量即可；逐样本流式处理，不保存整个模型的全部 $L\times L$ attention 矩阵。随后使用完整模型输出判断这些局部读数是否具有解释力。

**相位与竞争的分歧。** 对至多 16 个固定单证据实例，在已有 compact 与真实 far 之外构造一个“短内容、带间隔的位置”输入：保持各内容块内部位置差，把证据块和问题块放到真实 far 中对应的位置；保留相同语义内容，不补齐背景 token。整体统一加一个 position offset 不改变 RoPE 相对相位，不能用来模拟远置。

如果带间隔短输入成功而真实 far 失败，说明新增上下文、竞争或由其产生的隐藏状态变化值得优先研究。若两者都失败，相位/内容配对不兼容仍是候选因素。这个对照不能在多层模型中严格区分所有原因；固定 Q/K 重算只补充直接 score 证据。

若计划采用第 4.6 节的监督，额外记录对应来源集合的 $P$、答案 loss 对所观测 score 的梯度、compact teacher 的 $r$。预先限定查询和层，保留每个实例的结果。低 attention 只能与这些读数共同构成启动依据，不能自动转换为“证据完全不可达”。

**不为这些诊断挑一个新最优频率表。** 它们只回答既有方案在哪里发生功能变化、现有训练为什么没有修正。

### 6.4 E3：依据诊断选择一次有限修正，随后才做正式配对训练

第一阶段为固定成本的小试验。沿用当前模型、静态表、rank、模块、解码与实际 Native 约束；只改变 E1/E2 指向的一个因素：证据监督路径、独立训练实例覆盖，或已发现偏离协议的约束实现。

用固定的小训练集合与独立验证集合验证预测。该集合可以来自旧开发数据，但正式确认集必须没有参与本次选择。若加入第 4.6 节辅助项，明确预测为“远端来源的 score 梯度增强，并带来训练答案拟合改善”，随后检查验证生成和原生代价；只有 mass 上升、答案未改善时，判为没有实现需要的修正。

若既有训练已经拟合而选择改进语义覆盖，则预测应是“固定预算下独立验证改善”，训练分数进一步上升不是验收标准。小试验只用于方法可行性，不能当最终论文的无偏确认。

通过后，正式训练保留以下三臂：

| 臂 | 表及 gain | 作用 |
|---|---|---|
| G | 与 Z **完全相同的采样端点**，内部为几何分配；与 Z 共用 gain | 识别内部分配的训练效应 |
| Z | 冻结的既有非几何分配 | 目标方法 |
| Y | 适用于该模型的有效扩展基线，使用其明确的范围和倍率约定 | 判断实际方法收益 |

原始 Native checkpoint 的无训练结果保留为参考，不另外训练一个不能回答主要比较问题的 N 臂。YaRN 可以承担 Y，但需要核对官方频率和倍率实现；它的无训练低分不能替代同预算训练后的基线。[R2] citeturn478784view1

G/Z 从同一 $W_0$ 启动，数据与顺序、seed、可训练参数、optimizer、token presentations、实际位置暴露、Native replay 和选模规则全部配对。共同 gain 的因果比较是“在这个已声明 gain 下的分配效应”，不能据此声称超过所有 gain 下的最优几何方案。Y 使用其合理部署约定，并获得相同的适配与辅助监督预算。

先完成一个配对 seed；有明确联合信号后完成第二个。训练预算依据小试验测得的吞吐在正式比较前固定，不能让某一臂单独训练更久。已有 ON/ZF 只有在表、数据、loss、预算和选模合同完全一致时才能复用为正式格子；仅名称接近不够。

**本轮 from-scratch 不新增训练。** 使用已有三种子固定范围结果和权重×表交叉。1.485B 不同 trainer 的对照、MLA 和视频结果可作为注明限制的外部证据，不能替代主要因果控制。fileciteturn12file0L437-L447 fileciteturn12file1L746-L758

---

## 7. 最终评测要回答“是否同时做到”，而不仅是某项指标过线

### 7.1 原生保持采用绝对量，保留原来严格标准

沿用旧方案已声明的目标作为默认工程标准：两个自然文本领域各自 $\Delta\mathrm{NLL}\le0.03$ nats/token；原生任务宏平均损失不超过 2 个百分点；预设任务组损失不超过 5 个百分点；未正常终止比例增加不超过 2 个百分点。训练中的 KL 预算 0.02 是约束设置，不能单独等同于上述能力保持。fileciteturn9file0L118-L147

这些阈值不是 RoPE 理论推导出的常数。所谓“极低损失”应在论文里附上实际绝对差、区间和覆盖的任务。需要更严格目标时，应在新的方法结果出现前整体收紧合同，不能看到结果后改变允许代价。

可复用尚未用于训练或选模的现有原生样本。补新样本时，优先增加独立语义实例和来源文档，不把相邻 token 或同一实例的不同长度当成独立重复。

### 7.2 长任务同时报告绝对能力、对照差和来源依赖

最终主要长度为原生、2×、4×。模型和配置冻结后，只保留一个更远长度测试，例如 8×，作为边界；无需再铺开完整 32K/64K/128K 矩阵。

对标准自然任务使用官方指标和完整输出；对唯一答案的受控任务保留两世界组级严格成功。并列报告 compact、near、far，避免 compact 受损造成“长短差距缩小”的假改善。来源替换后答案应随真值改变；随机含有答案片段、只学会终止或 teacher-forced NLL 下降不替代真实生成。

“强外推”至少需要可用的绝对正确率，并在有效基线前有实质增量。0→4/32 不能因相对增长很大就命名为强外推。正式测试前，根据任务的 Native compact 水平与有效基线确定需要达到的实用水平，并冻结判断标准；同时公开绝对分数，不能只公开阈值结论。

若训练见过真实 16K，16K 结果报告为模型上下文扩展；8K 训练后的 16K、16K 训练后的 32K 才分别检验相对于该训练阶段的未见长度。每个 checkpoint 的最大训练 position ID 应与物理长度一起列出。

### 7.3 小样本无法证明非常小的原生损失

32 组样本即使没有发现一次新错误，也不能确认真实新增错误率低于 1% 或 2%。独立伯努利样本零次事件时，单侧 95% 上界为

$$
1-0.05^{1/n};
$$

$n=32$ 时约为 8.94%。这是对新增错误事件的示例，不等同于净准确率差的完整区间；它说明现有小确认池无法承担极精细的无损主张。

对方法差使用配对区间，按语义实例或来源文档聚类；同时呈现每个训练 seed。评测实例重采样的区间不覆盖全部训练随机性。两个 seed 是有限预算下的重复证据，不足以声称普遍稳定。相关性、$p>0.05$ 或区间重叠也不构成方法等价证明。

原生样本量依据所需精度决定，不能为了将某个 retention 下界抬过 0.88 而更换分层权重、删掉困难组或选择区间方法。高精度的原生保持确认，通常比继续新增一个稀疏小样本 benchmark 更值得投入。

---

## 8. 算力安排与停止规则

### 8.1 预算按尚可使用的 GPU 小时计算

历史约束是 RTX 5090 / RTX PRO 6000，总规模最多约 100 GPU 小时。本轮不假定这 100 小时仍全部剩余。以实际可用余额 $B$ 为上限，以下只是 $B=100$ 时的支出上限，不是运行时间预测：

| 工作 | 上限 | 支出条件 |
|---|---:|---|
| E0 资产与预测重算 | 0 GPU 小时 | CPU 完成；需要模型的部分计入 E1 |
| E1 运行表交叉、训练/验证拟合读数 | 6 小时 | 优先执行 |
| E2 少量 Q/K、gain、间隔位置诊断 | 4 小时 | E1 留下具体分歧时 |
| E3 有条件的小规模修正 | 6 小时 | 与旧失败有明确区别、具有具体预测 |
| 正式 G/Z/Y × 两个配对 seed | 60 小时 | 小试验有真实生成与原生保持信号 |
| 独立原生与长任务确认 | 16 小时 | 配置和选模冻结以后 |
| 故障与必要复核预留 | 8 小时 | 不用于新曲线搜索 |

先用实际吞吐估算正式训练与评测是否能装入余额。若不足，先取消 E2 的非必要读数、更远长度或第二模型；保留 G/Z 的因果配对与实际基线。若连这些比较都无法完成，应收缩论文的主张范围，不能用不匹配的旧结果拼凑成完整矩阵。

同样的 epoch 数、样本数或 step 数可能对应不同 token 与长度分布。表中训练预算需同时记录物理 token presentations、最大位置、训练步数和实际 GPU 时间。

### 8.2 本轮不投入的项目

新大模型从头训练、几十条新频率曲线、全模型 Fisher、用长任务 gradient 优化 64 维静态表、per-head/per-layer 新位置参数、路由系统、解码重排、NoPE/稀疏架构重构均不进入当前主实验。它们会改变研究对象或重开已低收益的搜索轴。

模型升级也需要具体证据。现有 8B 实验已有“概率和来源使用改善，但 16K 完整生成很低”的记录；参数量上升并没有自行消除问题。fileciteturn12file1L729-L743

### 8.3 四条停止规则

1. **E1 没有确定新训练的实际改动时，停止训练扩张。** 保留已有结果和未识别原因；CPU 分析或补一个缺失读数仍可继续。不得因希望得到正结果而直接重复旧 all-linear 配方。
2. **小试验只有代理指标改善时，不追加大预算。** 它必须改善所预测的拟合或迁移断点，并出现真实生成信号；Native 总损害继续受约束。
3. **正式比较失败时，保留全部已完成格子。** 不换 seed、不在正式测试上调 gain/曲线、不删除低分任务后宣布成功。
4. **阶段性失败限定到具体配置。** 它不能证明所有非几何分配、所有 LoRA、单表目标或整个论文问题不可解；也不能靠弱化成功定义将其改记为成功。

---

## 9. 执行规格与论文交付

### 9.1 最小机器可读配置

下面是需要执行者填入实际资产的任务规格；`REQUIRED` 必须由现有 run 读取。它不是声称仓库已经存在这些字段或新 runner。

```yaml
study: rope_exponent_allocation
model:
  checkpoint: REQUIRED
  tokenizer_revision: REQUIRED
  native_window: REQUIRED
  precision_and_backend: REQUIRED

interventions:
  T0:
    inv_freq_sha256: REQUIRED
    logit_scale_semantics: REQUIRED
  TZ:
    inv_freq_sha256: REQUIRED
    logit_scale_semantics: REQUIRED
  rules:
    same_table_for_all_lengths: true
    same_adapter_for_all_lengths: true
    rebuild_kv_after_table_change: true

existing_weights:
  ON: REQUIRED
  ZF: REQUIRED
  ZC: REQUIRED_IF_USED

first_job:
  operation: existing_weights_by_runtime_configuration
  cells: [ON_T0, ON_TZ, ZF_T0, ZF_TZ]
  reuse_cell_only_when_manifest_matches: true
  data: same_semantic_ids_and_counterfactual_worlds
  layouts: [compact, near, far]
  collect:
    - generated_token_ids
    - official_score_and_strict_pair_score
    - answer_ce_and_min_gold_prefix_margin
    - native_total_output_kl_and_task_delta
    - train_vs_independent_validation
  gpu_hour_cap: 6

training_authorization:
  requires:
    - observed_fit_or_transfer_failure
    - one_explicit_change_from_previous_protocol
    - predefined_prediction_and_validation_rule
    - fixed_native_retention_contract
    - remaining_budget_for_matched_controls
  frequency_table_optimization: false
  rank_or_module_sweep: false
  benchmark_driven_gain_tuning: false
```

新增观察器应与已验证的前向路径分离：它读取需要的 Q/K 或 logits，不替换模型 attention 计算。取值位置必须包含模型实际 Q/K normalization，明确旋转布局和 GQA/MHA 对应；观察器自身通过张量与少量模型 parity 后才解释结果。

### 9.2 正文只需要三组核心图表

**第一组：固定范围与内部分配。** 同一端点下的 Geo、非几何实例，以及现有三种子训练结果。附上实际 $a,R,z$，让读者看到改变了什么。

**第二组：已学配对与安装阶段。** 已有权重×表交叉，加上当前 ON/ZF 的运行配置交叉；用原生、近端和远端的绝对结果说明配对的作用。注明频率与 gain 是否同时变化。

**第三组：原生代价与真实长生成。** G/Z/Y 在无训练与相同适配预算下的结果，列各 seed、置信区间、训练暴露和更远长度。只有实际满足联合目标的格子才标为成功。

完整解析 surrogate、额外形状、视频/MLA、多种 rank 诊断放入附录并注明用途。新版主文的读者应能先看懂一个变量、三种安装条件和最终收益，再决定是否需要读某个 Cosh 实例的推导。

### 9.3 这份方案完成了什么，仍缺哪一项事实

现有成功和失败能够被同一个计算对象容纳：指数分配改变带有内容系数的相对位置分数；冻结权重承受原配对误差；适配改变这一配对与后续计算；从初始化训练则允许配对共同形成。低频内容用途、有限分数扰动、softmax 竞争和实际答案 margin，解释了为何纯谱指标、gain、NLL、EOS 不能单独决定完整生成。

当前尚未得到的事实是：**哪个具体、可复现的静态表与适配协议，已在独立测试中实现极低原生损失和强 4× 生成。** 本轮能够执行的最短路径是：补齐 ON/ZF 交叉与训练拟合读数，选择有证据支持的一项修正，再完成 G/Z/Y 的配对确认。得到正结果后，论文的主张跟随该结果；得到负结果后，明确哪一项计算条件仍未满足。


---

## 附录 A. 本轮实际完成的 CPU 代数检查

以下检查只使用随机小张量与明确公式，没有访问模型 checkpoint。它们检验实现与推导是否一致，不构成 attention 机制在当前模型上已被识别的证据，更不预测某项训练必然成功。

| 检查 | 本轮数值 |
|---|---:|
| 有限 attention KL 恒等式与直接 KL 的绝对误差 | $1.11\times10^{-16}$ |
| 构造 $u=2s+4$ 时恢复的全局 gain | 0.4999999935，理论值 0.5 |
| 二元来源质量 KL 梯度与有限差分的最大误差 | $4.70\times10^{-11}$ |
| 来源质量 $P\approx4.36\times10^{-9}$、teacher $r=0.8$ 时源内 score 梯度 | 约 −0.8 |
| 原生线性补偿的最小二乘与投影公式误差 | $5.33\times10^{-15}$ |
| Q/K/频率联合重编号后的分数误差 | $3.33\times10^{-16}$ |
| 固定 Q/K 的指数梯度与有限差分误差 | $2.12\times10^{-10}$ |

下面代码可从本 MD 中保存运行；依赖 NumPy 与 SciPy。它不调用 GPU，也不改任何项目资产。

```python
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize_scalar
rng = np.random.default_rng(20260905)
checks = {}
# Exact finite change of attention and KL.
s = rng.normal(size=11); e = rng.normal(size=11)
p = softmax(s); pp = softmax(s+e)
kl = np.dot(p, np.log(p)-np.log(pp))
closed = logsumexp(np.log(p)+e)-np.dot(p,e)
checks['attention_KL_identity_abs_error'] = abs(kl-closed)
checks['Hoeffding_bound_slack'] = float(np.ptp(e)**2/8-kl)
# Gain-only repair when scores differ by affine scaling.
s0 = rng.normal(size=13); p0=softmax(s0); u=2*s0+4
fn=lambda a: logsumexp(a*u)-a*np.dot(p0,u)+np.dot(p0,np.log(p0))
a = minimize_scalar(fn, bounds=(0.01,3), method='bounded',options={'xatol':1e-13})
checks['affine_gain_recovery'] = float(a.x)
checks['affine_gain_residual_KL'] = max(0.,float(a.fun))
# Binary evidence-mass KL derivative, including tiny source probability.
s = np.array([-18., 0., 0.6, -0.4]); r=0.8; p=softmax(s); pe=p[0]
grad = p.copy(); grad[0] = (pe-r)*p[0]/pe
grad[1:] = -(pe-r)*p[1:]/(1-pe)
def mass_loss(t):
    lp=t-logsumexp(t)
    logpe=lp[0]; lognot=logsumexp(lp[1:])
    return r*(np.log(r)-logpe)+(1-r)*(np.log(1-r)-lognot)
h=1e-5
fd=np.array([(mass_loss(s+np.eye(4)[j]*h)-mass_loss(s-np.eye(4)[j]*h))/(2*h) for j in range(4)])
checks['evidence_mass_gradient_max_error'] = float(np.max(np.abs(fd-grad)))
checks['evidence_mass_P'] = float(pe)
checks['evidence_mass_source_gradient'] = float(grad[0])
# Native linear compensation is an orthogonal projection after whitening.
A=rng.normal(size=(14,5)); B=rng.normal(size=(14,4)); x=rng.normal(size=5)
b = -np.linalg.pinv(B)@A@x
res = np.linalg.norm(A@x+B@b)**2
P=np.eye(14)-[REDACTED_EMAIL](B)
S=A.T@P@A
checks['native_projection_identity_abs_error'] = abs(res - x@S@x)
checks['native_projection_min_eigenvalue'] = float(np.linalg.eigvalsh((S+S.T)/2).min())
# Joint slot relabeling preserves the bilinear RoPE calculation.
q=rng.normal(size=(6,2)); k=rng.normal(size=(6,2)); w=np.geomspace(1,.001,6)
def score(q,k,w,t):
    c=np.cos(w*t); si=np.sin(w*t)
    kr=np.stack((c*k[:,0]-si*k[:,1],si*k[:,0]+c*k[:,1]),axis=-1)
    return float(np.sum(q*kr))
perm=rng.permutation(6)
checks['joint_permutation_abs_error'] = abs(score(q,k,w,17.3)-score(q[perm],k[perm],w[perm],17.3))
checks['frequency_only_permutation_difference'] = abs(score(q,k,w,17.3)-score(q,k,w[perm],17.3))
# Direct derivative of a normalized interior exponent, with fixed Q/K.
a0=0.; span=np.log(1000.); z=np.linspace(0,1,6); lag=17.3; idx=2
ww=np.exp(-(a0+span*z))
C=np.sum(q*k,axis=-1)
Jk=np.stack((-k[:,1],k[:,0]),axis=-1)
D=np.sum(q*Jk,axis=-1)
analytic=span*ww[idx]*lag*(C[idx]*np.sin(ww[idx]*lag)-D[idx]*np.cos(ww[idx]*lag))
eh=np.zeros_like(z); eh[idx]=1e-6
numeric=(score(q,k,np.exp(-(a0+span*(z+eh))),lag)-score(q,k,np.exp(-(a0+span*(z-eh))),lag))/(2e-6)
checks['exponent_gradient_abs_error'] = abs(float(analytic-numeric))
for k,v in checks.items(): print(f'{k}: {v:.12g}')
assert checks['attention_KL_identity_abs_error'] < 1e-12
assert checks['evidence_mass_gradient_max_error'] < 1e-7
assert checks['native_projection_identity_abs_error'] < 1e-10
assert checks['joint_permutation_abs_error'] < 1e-12
assert checks['exponent_gradient_abs_error'] < 1e-7
```

## 附录 B. 资料与来源

### 项目资料

**P0.** `main(20260905-171808).tex`：本次入口文件。正文通过 `sections/...` 引入；本文的逐节和数字核对使用下列完整稿件。

**P1.** `main(20260905-052024).pdf`：31 页完整版本。主要依据为固定范围实验及附录 C、成熟模型附录 E、MLA 附录 F。文中给出了精确文件引用和最小支持范围。重点查看表 9–16；训练配置、运行表、gain 与结果的因果范围以相应段落为准。fileciteturn12file0L68-L96 fileciteturn12file1L537-L567

**P2.** `SPECTRAL_BUDGET_LORA_20260905.md`：使用 87 行、含 ON 完整回填的版本，内部标题为“静态 RoPE 表 retrofit：数据与三个问题（2026-09-06）”。同名的较早文件仍有 ON 待回填和后来撤回的机制判断，本文未将其作为最新结论。fileciteturn7file1L231-L266 fileciteturn7file1L285-L291

**P3.** `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`：使用其已报告的现象、构造来源与失败记录。备忘录中的理论解释不因被多次引用而自动成为实验证实结论。fileciteturn7file0L29-L107

**P4.** `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md`：原执行方案，用于核对哪些 loss、模块、数据布局和约束已经被提出。具体 run 是否完整实施仍需执行配置验证。fileciteturn9file0L24-L54 fileciteturn9file0L167-L185

### 相关原始论文

**R1. RoFormer: Enhanced Transformer with Rotary Position Embedding.** arXiv:2104.09864。相对位置双线性结构的原始来源。citeturn111172search0

**R2. YaRN: Efficient Context Window Extension of Large Language Models.** arXiv:2309.00071。按频段处理及 attention 温度约定；比较时区分原始频率、插值频率与最终 logit 倍率。citeturn478784view0turn478784view1

**R3. Round and Round We Go! What Makes Rotary Positional Encodings Useful?** arXiv:2410.06205。成熟模型中频率的内容与位置用途。citeturn478784view4

**R4. How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization.** arXiv:2607.07678。数据依赖尺度、频率使用及长度泛化的条件。citeturn478784view5

**R5. LeRoPE: Learnable RoPE Frequencies Improve Language Modeling.** arXiv:2607.10134。频率学习、内容相关梯度及共同适应。本文使用的链式梯度不作为新颖性主张。citeturn478784view3

**R6. LongReD.** ACL 2025，ACL Anthology 标识 `2025.acl-long.524`。长上下文扩展中的短程功能恢复与蒸馏；Native teacher 修复本身已有相关工作。citeturn328639search2

**R7. AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally.** arXiv:2607.19363。head 间频率与缩放差异的相关工作。citeturn328639academia24
