# A6 digest：sol16–sol19 + recovered sol20/astra10

状态：挖掘与整理，不是推导。所有条目给出处；证据等级 [已验证] / [部分证据] / [假设] / [叙事-未验证]。
本地复核已做的部分单独标注「本地复核」并给出命令级依据。
权威文档对照：`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}`。

---

## 0. 一句话定位

这六份材料是一条**同构的收敛线**：sol16 给出「优化器坐标系」（把冻结 MrPro 面变成 16 维无约束欧氏变量），sol17 给出「目标泛函的有限精确形式 + 全局最优 DP」，sol18 给出「模型级判定协议 + 十个实现陷阱」，sol19 撤回历史闭合证书但保留可辩护边界，sol20/astra10 回传 OLMo 面板与两条待核对线索。四者拼起来 = KKT 问题的**变量、目标、判定、边界**四件套；缺口只有一个且四家一致：**带角色标签的签名矩从未被测**。

---

## 1. sol16 — 有限频率分配与冻结模型校准
源：`docs/research/rope_allocation_20260910/agents/sol16.md`（121 行，全文读完）

### 1.1 坐标与分配流形（F 的坐标声明件）

log 频率坐标与归一化正间隙（sol16:5-11）：

```
x_j = −log ω_j = x_0 + A z_j,   0 = z_0 < z_1 < … < z_{K−1} = 1
a_j = (x_j − x_{j−1})/A  → 单纯形
```
- 关键主张：该坐标**正确分离**三件事 —— 快端 x_0、支撑跨度 A、内部配比 a/A。与 6Pro 分解一致（引 `docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:40-55`）。[部分证据-引文未本地复核]
- EVQ = 指定连续密度后把逆 CDF 量化成这种间隙；MrRoPE = 在原生网格上的加性膨胀间隙。**注意这不是"最优调度"而是"分配流形"**——sol16 明确否认存在普适 score-最优调度（sol16:5）。

### 1.2 两个 regime 是两个优化问题（不可互换）

```
scratch:  min_{W,a} E_{P_train}[−log p_{W,a}(y|x)]
frozen:   min_a     E_{P_deploy}[−log p_{W_0,a}(y|x,p)],  W_0 固定
```
- 直接反证 = 协同适应表 `paper-2027/tables/table_coadapt.tex:6-21`。[已验证 —— 本地复核：Geo/Geo 7.14、EVQ/EVQ 7.16、Geo→EVQ 运行时 **76.20**、EVQ→Geo **23.05**，n=1,920 head-query 观测，两运行时表都用 midpoint 网格 u_k=(k+1/2)/K]
- **关键因果**：最差的那格（Geo 权重 + EVQ 运行时，PPL 76.20）恰恰是静态 Rényi-2 秩 r₂ 改善最多的一格（4.57 → 12.54）。故静态相位量可以是**构造先验**，不能选冻结部署表。[已验证-本地复核]

### 1.3 精确有限约定（Qwen2.5-3B，zero-based）

sol16:29-33：[已验证-本地复核 `scripts/lib/rope/schedules.py` 的 `geometric_inv_freq`]
- d_head=128，对数 K=64，base θ=10⁶，zero-based j=0..63
- ω_j^native = θ^{−2j/d} = θ^{−j/64}；末对是 θ^{−63/64}，**不是** θ^{−1}
- 安装侧出处：`transformers/models/qwen2/modeling_qwen2.py`（sol16 引 89-98；本地 transformers 4.57.6 中该类在 `:273-304`，行号随版本漂移，**实质成立**）

MrPro 过渡（sol16:33-44）：

```
m_j^Mr = 0                        j ≤ 23
       = q(q+1)/(17·18)          24 ≤ j ≤ 40,  q = j−23
       = 1                       j ≥ 40
ω_j = ω_j^native · 4^{−m_j}
ε_i^Mr = 2i/(17·18),  i = 1..17,  Σε = 1   ⇒ 约束后 16 个有效自由度
```
[已验证 —— 本地复核：`docs/research/rope_allocation_20260910/code/joint_mode_candidates.py:26-29` 的 `mrpro_increments(width)=2i/(width(width+1))`，宽度 17 完全一致；两种写法 2i/[17·18] 与 2i/(17·18) 同一式]
- sol16 明说：**"16 个学到的过渡参数"不能理解为 16 个无约束频率**——那会丢掉精确的总尺度或保序。

### 1.4 ★ softmax-Helmert 无约束参数化（本次任务重点，完整抄出）

公式（sol16:46-53）：

```
B : 固定的 17×16 正交零和（Helmert）基，张成 {x ∈ R^17 : Σx = 0}
ε(η) = softmax( log ε^Mr + B η )
m_{23+q} = Σ_{i=1..q} ε_i(η)
```
- η = 0 时在 float64 下**逐位精确等于 MrPro**。
- 对任意有限 η 自动保持：增量全正、累计指数严格递增、过渡总量恰为 1、外带固定、末表递减。
- 16 个存储标量全部可辨识；用 17 个自由 logit 会引入无用的常数平移零方向。

**参考实现（本地完整读，`docs/research/rope_allocation_20260910/code/sol16_frequency_calibration_reference.py`，143 行）**

Helmert 基构造（`:14-23`），本地逐列验算：
```python
out[:count, column] = 1.0 / sqrt(count*(count+1))
out[count,   column] = -count / sqrt(count*(count+1))     # count = column+1
```
- 逐列平方和 = count/(count(count+1)) + count²/(count(count+1)) = 1 → **列正交归一** [本地复核：代数验算]
- 逐列和 = count/√(count(count+1)) − count/√(count(count+1)) = 0 → **零和** [本地复核]
- **结构性质（sol16 未点明，对本问题有用）**：第 c 列（0-based）非零支撑只在第 0..c+1 行 → 基是**下三角嵌套模板**；η_c 只影响增量 ε_1..ε_{c+1}。即 16 个坐标天然按「过渡带前缀」分层，边界段可局部控制。

主模块 `MrProAllocation16`（`:32-78`）：
- 构造参数 `pair_count=64, head_dim=128, rope_theta=1e6, scale=4.0, low=23, high=40`；断言 `pair_count*2 == head_dim` 且 `0 ≤ low < high < pair_count`
- `eta = nn.Parameter(zeros(width−1=16, dtype=float64))`；buffer 存 `log_reference_increments`、`basis`、`native_inv_freq`
- `increments() = softmax(log_reference + basis @ eta, dim=0)`
- `exponents() = cat(zeros(24), cumsum(eps), ones(64−40−1=23))` → 长度 24+17+23 = 64，槽 0..23 恒 0、槽 40 恒 1 [本地复核：代码断言 `exponent[:24]==0`、`exponent[40:]==1`（float64, atol 2e-16）]
- `inv_freq() = native_inv_freq * exp(−log(scale)·exponents)`，即 ν_j = ω_j S^{−m_j} ✓

配套 `DifferentiableQwenRotaryEmbedding`（`:81-107`）：**去掉 `@torch.no_grad`** 的 drop-in rotary；FP32 相位 + 关 autocast；cos/sin 乘 `attention_scaling`（默认 1.0）后 cast 回 hidden dtype；无 dynamic-RoPE 变异。

**已做的 gradcheck（sol16:55 与代码 `self_check()`）** —— 逐条列清，因为强度差别很大：
1. `eps == mrpro_increments(17)`，rtol=0/atol=2e-16 → η=0 逐位等于 MrPro [强]
2. 形状/端点/严格递减断言 [强]
3. **真实 rotary 路径梯度流**：x=(1,8,128) fp32，positions=[0,1,17,257,4096,32767,65535,131071]，loss = cos/sin 与一组**人造线性权重**的内积，断言 `eta.grad` 有限且**逐元素非零** [中 —— 这是合成标量，不是任务 CE]
4. `torch.autograd.gradcheck` 只作用在 `softmax(log_ref + B@z)` 这一层（eps=1e-6, atol=1e-6, rtol=1e-4）[中 —— 只证 softmax 局部雅可比，不证全模型]
- **强度边界（我加）**：sol16 声称的 "checks … all-parameter gradient flow through the actual FP32 rotary computation, and autograd finite differences" 成立，但 finite-difference 只在 16 维 softmax 映射上做；全模型方向导数的对照被列在 §"强制实现修正"里作为**待跑的 control**（sol16:94），并非已做。

**用法（写给下一步推导）**：直接把 `MrProAllocation16.eta` 交给 Adam（sol16:79 明说 "Adam on 16 scalars is sufficient"），不需要投影；参数化已强制保序与总尺度。若要位移界，用对称的 `‖Bη‖_∞` 约束并以「撞界」为诊断，而不是候选网格。

### 1.5 三种 EVQ 约定必须分开（口径冲突源）

sol16:59-65，我逐条本地复核：
1. 原生/HF geometric 用 u_j = j/K —— **成立**（`scripts/lib/rope/schedules.py` 的 `geometric_inv_freq`，idx=arange(K)，1/base^(2idx/head_dim)）[已验证]
2. 规范 EVQ 默认 midpoint u_j=(j+1/2)/K；τ=0 时返回 midpoint geometric 而**不是**原生 HF 表（`schedules.py` 的 `evq_cosh_phi(..., midpoint=True)`，τ≈0 时直接 `return u`）[已验证-本地复核]
3. 某些实验取 midpoint EVQ 后首末 φ 仿射重锚到原生端点 —— 引 `scripts/core_text_phases/phase16_phase_allocation_budget_matrix_m4.py:84-93`。**本地该区间是另一段评分代码**，未见到所述重锚逻辑。[叙事-未验证（行号/位置不吻合，文件存在）]
4. 理论笔记用 u_k=k/N 却讨论 u=1 端点（引 `docs/theory/EVQ_COSH_THEORY.tex:49-66,277-292`）[叙事-未验证-未核]
5. `LearnableEVQRoPE` 用 midpoint（`scripts/lib/rope/learnable_evq.py:82-83`，本地复核确为 `u=(arange(N)+0.5)/N`，注释 "matches paper eq. 9"）而其散文说端点不动。[已验证-本地复核]
- **可执行结论**：冻结 Qwen 校准时，把频率写成**实际原生 FP32 表**的乘性膨胀，一次性绕开全部三种歧义（sol16:65）。

### 1.6 最小全模型「仅频率」校准协议（7 步，sol16:69-79）

1. 冻结 Qwen2.5-3B checkpoint，hash 全部权重文件 + 内存中非 rotary 状态；gain 在所有臂固定，不学 gain/位置重映射/adapter/权重
2. 只替换 `model.model.rotary_emb` 为上面的可微共享模块，初始化 η=0；`use_cache=False`、SDPA、`model.eval()` 关 dropout 但保留 autograd；答案在末尾时只要末尾 answer-span logits
3. 先冻结一小批 32K record 任务，按 record 身份切 fit / calibration-selection；prompt 不得含答案；仅在精确正确-答案 span 上开 label
4. 位置约定 p'_t = 4 p_t，p_t=0..L−1，L ≤ 32768 → 触达位置 131068；token/attention 图仍是 32K（**是位置拉伸校准分布，不是真 128K 行为证据**，仍需真 128K 评测）
5. 目标 = 完整前向下的 mean correct-answer token CE；每层、每 head、每个因果可见 key 都参与；不用 selected-key replay / selected-head 目标 / 注意力几何代理 / correct-count 不连续
6. **不引入惩罚权重网格**；保存每一步；在满足「native-position answer CE ≤ 初始 MrPro + 数值容差」的步中，选拉伸位置 CE 最低的一步；若无可行非初始步，结论为 MrPro 且结果负面
7. 选定表对 Native / MrPro / 现有最强冻结基线在**未触碰的真 128K** 任务上评一次；先报 CE 后报 EM/任务分；仅 32K 拉伸增益不是晋级证据

[部分证据-方案，未执行；与 sol18 的判定协议口径冲突见 §7 FLAG-5/FLAG-A]

### 1.7 强制实现修正与对照（可直接当 checklist）

sol16:83-98：
- **`Qwen2RotaryEmbedding.forward` 装 `@torch.no_grad()`** → 把 `inv_freq` 变参数、把可微表拷进 buffer、复用 stock forward **三种做法都会静默不学**。必须换成无 `no_grad` 的 forward。[已验证-本地复核：transformers 4.57.6 `modeling_qwen2.py:293` 是 `@torch.no_grad()`，`:303-304` 是 `cos/sin * attention_scaling` → 同时坐实 sol18 的 gain 陷阱]
- 梯度检查点只在替换后兼容；用非重入 `use_reentrant=False` + `enable_input_require_grads()`；`position_embeddings` 在 decoder 层循环**之前**算一次（4.57.6 `:381`），其图可累积所有 checkpoint 层的梯度；在 checkpoint on/off 平价被证明前禁用 `torch.compile`
- 九项失败对照：MrPro 平价 / 梯度范围（恰一个 16 向量、全体权重 requires_grad=False、16 个梯度有限非零、非 rotary 状态 hash 不变）/ 零效应对照（位置全 0 或 S=1 时 ∂/∂η 必须**恰为 0**）/ 方向导数（随机单位向 g 与中心差分在 tiny CPU 契约 + 一个 GPU batch 上一致）/ checkpoint 平价 / 精度平价（BF16 autocast 下相位保持 FP32）/ 无 cache/compile 变异 / 答案卫生（解码检查 label 边界、答案不在 prompt、只有移位后的答案 label 进 CE）

### 1.8 sol16 的证据边界与新颖性自评（重要，防止过度宣称）

- 被分配的 6,144 行 artifact = 2,048 对世界；relation 准确率 1.0（4096/4096，mean NLL 0.000773），content 准确率仅 0.1948（399/2048，mean NLL 2.2525）；512 对 dev 同样 content-limited（0.2012）。→ 证明「代理或地板受限的任务面板会错排表」，**不是**分配律的证据。[部分证据-本地 artifact 缺失，无法复核]
- Boundary-matched MrPro 是其**声明的相邻间隙粗糙度目标**的唯一极小点，但该定理**没有任务分数后果**。[已验证-本地复核：`scripts/analysis/build_boundary_matched_mrpro.py` 的 `theorem_scope='Unique minimizer of this discrete spectral objective. No theorem of task-score improvement.'`（本地读 `:100-105` 区）]
- Smooth_MrBudget 是决定性反例类：几何更好、任务更差。
- 安全可说的四句（sol16:112-115）：EVQ 提供联合训练的连续分配先验；MrPro 提供部署支撑/带结构与合理初始化；全模型冻结 CE 提供正确的 checkpoint 特异选择子；**log-gap 单纯形是共享数学坐标**。

---

## 2. sol17 — 尺度输运审计与保标签有限优化器
源：`docs/research/rope_allocation_20260910/agents/sol17.md`（153 行，全文读完）

### 2.1 旧优化器里什么幸存、什么死了

| 对象 | 判决 | 出处 |
|---|---|---|
| 源可观测性（条件唯一性标量→位移） | 是**约束**不是分配目标；无符号、只表；不能区分冗余 nuisance 与参与学得对消的"看起来冗余"方向 | sol17:13-15（引 `scripts/analysis/rope_transport/tables.py:177-219`、`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md:25-40`） |
| null band 直觉 | **不成立**：弱边缘可参与强拍/曲率/对消；重定时一个关系同时保正交载体可能要求**加速**，纯压缩 ramp 排除它 | sol17:19-21（引 Astra05 投影构造） |
| MGDA | 数学有效但只回答局部问题；兼容梯度给共下降方向、对抗梯度给零；自带回溯与保序检查 → 是**受检局部下降**，不是解析分配律 | sol17:23-25（引 `scripts/analysis/verify_general_rope_allocation.py:24-42,98-126,127-142`） |
| 旧 τ/输运矩 | 静态 τ 审计：纯碰撞/相干/条件目标偏好极端 τ≈11–14、长度指数近零；自洽代理只到 ~L^−0.17；后续 stiffness 笔记用**选择 f-散度指数或条件于期望经验标度**才拿到 ~−0.5 | sol17:27-29（引 Astra02：有限连续 lag 窗内精确 EVQ 测度最优是唯一有限原子平衡，不是修正 Cosh） |
| 几何改善 | 不足为证：Smooth_MrBudget 改善无符号几何与 Q/K 加权弱子空间判据，却在记录的 128K 开发行上更差；125M 诊断改善长 NLL 而检索崩塌 | sol17:31-33 |

### 2.2 ★ 共享数学对象（F 的签名 margin 件）

签名 Q/K 正交积分（sol17:39-42）：

```
A±_rj = q_rj,0 k±_rj,0 + q_rj,1 k±_rj,1
B±_rj = q_rj,1 k±_rj,0 − q_rj,0 k±_rj,1
```
候选频率 ν_j、签名 lag d±_r 下的**精确条件源 margin**（sol17:44-51）：

```
M_r(ν) = γ Σ_j [ A+_rj cos(ν_j d+_r) + B+_rj sin(ν_j d+_r)
              − A−_rj cos(ν_j d−_r) − B−_rj sin(ν_j d−_r) ]
```
（γ 内含固定 gain）。保留对消与源身份；投影 Frobenius 能量、边缘相位覆盖、无符号注意力位移都丢掉这些信息。

N 个竞争 key 的概率证书（sol17:53-65）——**无需 distractor 间独立性**：

```
若 E e^{D_t} ≤ e^{b_t + v_t/2} 且源分数确定 = μ，则
Pr[p_* < ρ] ≤ ρ/(1−ρ) · Σ_{t=1..N} e^{b_t + v_t/2 − μ}
```
- 若源本身随机，需要的对象是**联合 margin MGF** `E exp(D_t − S)`；用均值源分数替代无效。
- 卡点：均值/协方差只给 Cantelli；样本协方差**不是**总体 PSD 上界；Gaussian/sub-Gaussian 尾必须被声明并检验，**不能从"看起来高斯的 Q/K 激活乘积"推断**。

### 2.3 ★ 精确有限计数 DP（F 的水床形式，O(BK²) 全局最优）

设定（sol17:69-73）：B 个已分辨 log 频率 bin（宽 Δ）、K 个等幅旋转对、整数占用 n_i、尾计数 T_i = Σ_{l≥i} n_l，**显式声明的**协方差

```
C_il = (α/Δ)·1{i=l} + β·min(x_i, x_l),   α>0, β ≥ 0
```
= 共享 bin 噪声 + 共享嵌套慢尾噪声。**不是** iid 通道噪声（iid 会给密度线性贡献，无法还原 EVQ 的密度平方代价）。

设 h_i 为确定性有用信号贡献，最小化 softmax 质量界等价（至常数）于：

```
(1)  min_{n_i ∈ Z_+, Σ n_i = K}  Σ_i [ (α/(2Δ)) n_i² + (βΔ/2) T_i² − K h_i n_i ]
```

精确后向递推（sol17:85-90）：

```
F_i(t) = (βΔ/2) t² + min_{0 ≤ n ≤ t} { (α/(2Δ)) n² − K h_i n + F_{i+1}(t−n) }
F_{B+1}(0) = 0，否则 ∞
```
从 F_1(K) 回溯给出**声明模型下的全局最优整数计数**，时间 O(BK²)。占用上限与必需端点通过限制 n 进入。**重复频率是该精确模型的一部分；解完再去"摊开"就改变了问题。** [已验证-推导；本地未复算 DP]

- 常数 h → 线性项为常数 → 连续松弛下**还原离散 Cosh 递推**；振荡的、源对齐的 h_{j,i} → 产生 Mr 式的正远端偏好。
- **βΔT_i²/2 就是"水床写成守恒律"**：质量推向慢端 ⇒ 每个远尾碰撞负载上升（INTEGRATION §4.2 同判）。
- MrPro 的等差 radix 增量仍是启发式：**首零论证与 (1) 都不能在无额外假设下导出其精确二次累计调度**（sol17:100）。

### 2.4 冻结标签版（不能把 scratch 密度移植进无标签槽）

sol17:92-98：对冻结有序表，计数本身不可用。保留槽标签，把连续标签单调分配给 bin；`h_i n` 换成

```
Σ_{j=K−t}^{K−t+n−1} h_{j,i}
```
h_{j,i} = 原槽 j 被分到 bin i 时的**带符号**有用贡献。前缀和预计算保持同复杂度。逐标签原生约束可禁止「认证的 native 角色 margin 或 MGF 界低于冻结阈值」的分配。→ 对上特殊协方差是精确 DP；一般异质跨槽协方差该递推**失效**，须改用 Astra03 的认证有限 margin 锥求解器或如实报告模型不受支持。

### 2.5 六条反例检查（应直接进 F 的单元测试）

1. 全部签名均值消失 ⇒ 降协方差只改善一个**界**，不提供正源信号；优化器必须报"无认证检索方向"
2. iid 各向同性 sine/cosine distractor 系数 ⇒ 分数方差对频率不变，EVQ 碰撞项**消失**；任何保留它的推导都在用一个未声明的相干协方差
3. 两候选可共享均值/协方差而有不同非线性失败概率；无公共 coupling / 高斯模型 / 随机占优条件时 Cantelli 界排序 ≠ 精确性能排序
4. 更好的成对标准化 margin 不蕴含更好的多 key 联合成功（margin 间相关重要）；softmax MGF 界充分但可能松
5. 联合 Q/K-频率置换是恒等；**仅表置换不是**——任何丢标签的冻结优化器与此 gauge 检查矛盾
6. 真 128K prefill 若把上游激活推出校准包络，固定态证书失效，即使其局部有限相位算术精确

### 2.6 sol17 对 root 校准计划的评价（与 sol16 分层，见 §7 FLAG-5）

**测到**：监督式仅频率表能否在选定的稀疏位置拉伸干预下改善答案似然、且守 fitted native 损失；比固定态几何/注意力图 Frobenius 更强（完整冻结模型 + 请求的答案 token 参与）；方向能否从 fitting 行迁移到 holdout 行、再到真 128K。[部分证据]
**没测到**（sol17:123-129）：新角色条件 margin/MGF 理论（除非理论独立于 answer CE 产出预注册排序）；**稠密 128K 额外 key 竞争**（32K token 摊在 128K 坐标范围仍只有 32K key）；连续 128K prefill 的状态分布；通用冻结部署规则；scratch EVQ 的计数密度主张。
- 最接近的方法学描述 = "supervised frozen-backbone per-frequency calibration"，与 LeRoPE 同族。称其为推导或 EVQ/Mr 统一的直接检验是过度宣称。[sol17:131]
- **"精确损失约束"需限定范围**：逐候选精确算两个损失是好的，但不使约束成为总体级，也不隔离源路由与输出格式/解码器效应；native 与拉伸行必须与最终 holdout 不相交，且需要 source-swap / source-deletion 对来证明改善 CE 跟随合法源而非捷径。[sol17:133]

### 2.7 sol17 的「更小的实验」（先花便宜的判决）

sol17:137-145，对已有 MrPro / Smooth_MrBudget / P2 / E1 四表，在**同一条冻结源行**上算一条预注册机制诊断：全行 source-vs-distractor log odds、签名均值 margin、经验协方差、单位对数 MGF 代价，native 与拉伸位置分开、任务/lag 单元分开。**要求统计量先正确排序已知的 Smooth_MrBudget 损失**；若仍偏好 Smooth 就停下理论主张。通过才解一个冻结标签 DP 表（答案 CE 不用于选表），再在未触碰距离配对集上比 MrPro/理论表/镜像。

---

## 3. sol18 — 算子地图、标签 oracle 与精确模型级测试
源：`docs/research/rope_allocation_20260910/agents/sol18.md`（133 行，全文读完）
覆盖自述：101 个文件 1,107,673 字节全读；1,536 行开发 JSONL 分页读完；另读 Astra01–08 + Astra09 可执行规则。[叙事-未验证：本地无 sol18 coverage 回执，见 §8]

### 3.1 四个不可共享代理的计算对象（分离声明）

1. **静态稠密 RoPE 部署**（Qwen 保留所有 key、改每对频率）— 这才是 EVQ/MrRoPE 的分配问题
2. 稀疏块选择 + 不变 reader（原生稀疏实验，按物理页排序后读选中原始 post-RoPE K/V、单 softmax；exact-mass 变体是全扫的**诊断 oracle**，不是高效稀疏法）
3. 前缀-only KV 保留（PM-Keep）— 其 future-query sampler 是选择代理，不是频率分配器
4. 稠密低秩算子替换（`rope_operator_family`）— 仍读每个 token，改变表征容量/投影/values/cache 宽度

→ 唯一可用的精确模型级测试 = **在单一共享静态 Qwen 表下、源受控配对 record 任务上的 teacher-forced 全模型目标 CE**，继以按族不相交的完整生成。

### 3.2 精确对象（F 的逐槽件）

```
z_j(ν_j) = γ { C_j cos(d ν_j) + S_j sin(d ν_j) }
C_j = q_j k_j + q_{j+64} k_{j+64}
S_j = q_{j+64} k_j − q_j k_{j+64}
```
[sol18:22-28] 必须保持附属：槽索引、层、query head、KV head、key 身份、target/distractor 角色、签名距离、表 gain、完整 normalizer。频率多重集或投影范数会丢掉学得的关联。
- 与 sol17 §2.2 的 A±/B± 是同一对象的两种写法（sol17 保留源/干扰双集，sol18 保留单槽但强调标签全附属）。

**现成自然捕获的用途边界**（`experiments/nongeometric_screen/pro_block_calibration.py`）：用 Native 频率 + MrPro 公共 gain、只有最后一个自然文本 query、采样 4 个 head、全部 key；**无问题 query、无请求 record 标签** → 兼容性对照，不是分配目标。[已验证-本地复核：本地读 `:15-80` 见 `common_gain = worker.tables['MrPro']['gain']`、`source_table = dict(worker.tables['Native'], gain=common_gain)`、`selected_heads = index%4+4*i`、`qpos = length-1`，与 sol18 描述一致]

### 3.3 正确源受控标签（四格族）

用 `content_swap` × `query_ordinal_swap` 四格族；仓库已检查定义不变量：每个 history 有相同 record 清单、请求序号与值分配独立变化、对格可共享同答案、dev/test token 内容不相交。[已验证-本地复核：`experiments/nosa_position/test_data.py` 的 `row_for(...)` 产 `row_id=f"{split}_c{c}q{q}"`、`family_id`、`query_key`、`query_ordinal`、`content_swap`、`query_swap`；`fourway(split)` 生成 (c,q) ∈ {0,1}²]

从输入文本 + tokenizer offsets 定义：**(T)** 请求的 key+序号 record span、**(H)** 同 key 的其他出现、**(O)** 其他显式 record、**(B)** 其余背景/结构 token、**(Q)** 问题 token（含所查 key 的副本）。保留 key-token 与 value-token 子 span 分开。
- **PM locator 的序号逻辑优于 RULER 唯一 key locator**（重复 key 情形）。
- 不要只选 Full/MrPro 已正确的行；否则校准变成 outcome-conditioned headroom selection。
- 选在真实 Qwen tokenizer + native chat template 下为**单 token** 的值串；主目标 = 该值的全词表 next-token log 概率；同时 teacher-force 终止 EOS。
- 角色诊断报精确全行 log 质量 `log Z_T − log Z_T̄`、`log Z_T − log Z_H` 加敏感度加权值贡献，**不得替代目标 CE**；query 与 local key 留在 T̄。

### 3.4 模型级校准测试（协议骨架）

参数化 `ν_j = ω_j exp(−x_j)`，保槽序。MrPro 面：固定 j≤23 的 x_j=0、j≥40 的 x_j=log4，只优化槽 24–39 且频率单调；公共 MrPro gain 固定。**若开放端点，须命名为 range-plus-allocation 实验而非纯中间分配测试。** [sol18:78-84]

族平衡 teacher-forced 目标：

```
L_target(x) = (1/|F|) Σ_{f∈F} (1/4) Σ_{c,q∈{0,1}}
              [ −log P_x(y_fcq | prompt_fcq) − log P_x(EOS | prompt_fcq, y_fcq) ]
```
[sol18:90-95] 全部层、value 路径、残差、MLP、解码器词表竞争者、全部 128K key 参与。自然 32K 全行 KL 与配对 source-world CE 作**硬/参考约束**（不劣于 MrPro）。从 MrPro 出发做约束投影优化 + 精确损失回溯；**冻结后再验证**；解冻结前不得看验证集。

判定比较（族不相交真 128K holdout）：① MrPro（固定公共 gain）② 一个冻结的 CE 校准表 ③ **关于 MrPro 的等范数镜像方向**（仅按需缩放以保序/端点）。主读数：目标 token CE、teacher-forced 答案+EOS CE、完整 greedy 值+EOS。**镜像**用来区分有用方向与一般性扰动/拟合。[sol18:100-108]

### 3.5 ★ 十个隐藏实现陷阱（可直接当 checklist；带本地复核结论）

1. **gain 重复计数**：自然捕获用 Native 频率 + MrPro 比较 gain，然后 `module.scaling * common_gain**2` 重建 logits。若候选 cos/sin 已含注意力因子，再乘一次会把 logits 改变 g⁴。**记录 gain 是乘在 Q/K 表、logits、还是两者，Q 与 K 各恰好一次。** [已验证-本地复核：`experiments/nongeometric_screen/pro_block_calibration.py` 内 `exact_scores=(c*cos+s*sin).sum(-1)*module.scaling*common_gain**2`，且上游是 `source_table = dict(Native, gain=common_gain)`]
2. **可运行的 Native 是不同对照**：Native/gain1 ≠ Native-frequencies/MrPro-gain
3. **符号/布局错配**：仓库 Qwen 用 split-half 而非相邻维度；捕获记 d = key−query；反转 d 会翻 sine 项
4. **目标 token 边界**：文本 decode/encode 可改空白与 assistant header；用实际 chat template 拼一次答案，验证 prompt 是 prompt+answer 的 token 前缀；显式包含终止 EOS
5. **梯度被静默 detach**：既有表安装器把 NumPy/FP32 拷进 rotary buffer，许多 runner 用 `inference_mode`，缓存 cos/sin 可能 detached 或在 autograd 外重建
6. **表变后的陈旧 cache**：post-RoPE 缓存 key 属于形成它的那张表；换频率需完整 fresh prefill，**绝不复用 candidate-A 的 K/V 算 candidate-B 的 CE**
7. **假长位置**：稀疏 position ID 或块拉伸映射只在**小 key 集**上测相位输运；真 128K 引入约 4 倍 key 并改变每个上游状态
8. **mask/normalizer 截断**：source-only、T-vs-H、selected-page、original-key KL 目标都漏竞争者；目标 CE 测试需要真因果 mask 与全部 key
9. **outcome-selected 行**：两行 target oracle 故意选 Full-correct 例子；频率校准必须用预定的完整四格族并在族/材料级切分
10. **argmax 无训练梯度**：优化 teacher-forced CE；greedy 生成只在冻结表之后用

### 3.6 sol18 对统一理论可诚实宣称的范围

Astra01/03/06 给出兼容的条件律（角色条件签名均值/协方差、含有限 distractor 数的指数矩 softmax 界、特殊协方差下的精确有限标签分配）；它们解释 EVQ 为何能从「常数保护信号 + 相干 nuisance」产生、Mr 式正远端信号为何以相反符号进入；**但不提供 Qwen 的角色均值、协方差、MGF 或下游 head/value 标签**。Astra09 的可执行步是**提案过滤器**（其输出余切与 Q/K/V 冻结），不能替代全模型 CE；正确桥接 = 测其预测方向是否与同族精确 CE 梯度一致，分歧则以模型级目标为准并诊断哪个冻结态假设失败。[sol18:125-129]

---

## 4. sol19 — 撤回了什么
源：`docs/research/rope_allocation_20260910/agents/sol19.md`（171 行，全文读完）
覆盖自述：18 个文件全读；development_rows.jsonl 1,536 行（1,024 relation + 512 content）全解析；surrogate_boundary_scan.json 280 个几何网格点、218 个 flagged failures（77.86%）。[叙事-未验证-本地源文件缺失，见 §8]

### 4.1 四类撤回（逐条，含撤回理由）

**撤 1 — 普适不可辨识定理 → 只保留类内**
`worker_falsification_1/report.md:447-456` 的量词覆盖「任何结构度量」并给出三类不可辨识，证明不支持该量词。具体：
- `:462-482` 置换构造只证**置换不变**映射（多重集、谱、对称 frame potential）不能识别冻结行为；有序映射可以区分两表。更强：标量映射 `Φ(Ω)=R(Ω;M,D)` 按定义识别所选风险（依赖模型/数据，但它是普适量词的字面反例）
- `:486-498` 的 ULP 构造只反驳**不连续**轨道计数统计量，对连续有序映射无话可说
- `:456` 的连续 ε→0 族是**假设**的，不是构造或测量的
→ 正确边界：**没有已测的 model-blind 类成员是所观测干预下的充分冻结部署证书**。[叙事-未验证-本地源报告不在仓库]

**撤 2 — slot-19 的 Hessian/Fisher 主张**
记录反复把「C2 平方残差 81.2% 在槽 19」转成「曲率 81.2% 在槽 19」，两者不同（`worker_falsification_1/report.md:507-517`）。无 artifact 提供实测 Fisher 矩阵、Hessian 特征值、κ(F)>10⁴、λ₁/λ₂>10²，也没有其首特征向量与槽 19 对齐的证据。另有两条数学错误：`:506` 把经验 Fisher 等同于损失 Hessian（需正则性、正确设定、模型分布下的期望）；`:517` 把 movement MAE 0.001223 当 L1 范数并断言 ΔL=0.004615（MAE 与 L1 差因子 K，报告的 gate 分数不建立该二次分解）。补救记录正确退役了局部 Taylor gating（相位偏移 22.74 / 90.97 rad，`challenger_remediation_1/report.md:23-25,85-105`），但其替代路径积分**是恒等式**，仍需模型/数据计算，不是廉价事前预测器；有限旋转界安全但饱和后太松无法排序候选。

**撤 3 — 「充要的六个观测量」**
`worker_falsification_1/report.md:539` 无充分性证明。清单混了：局部导数（H, J, Fisher，不决定有限干预）、需分布假设的矩摘要、基于平均 distractor + LLN 的 softmax 近似（不控制相关极值）、`T(a,R,z)`（是输入参数化而非可观测量）。补救文档自己承认两点：把 H 与 J 统一为一个 Jacobian Gram 对象的收缩、把 T 重分类为坐标（`challenger_remediation_1/report.md:26-28`）。→ 六摘要只是诊断。

**撤 4 — 支持重定标「机制」：观察保留、解释撤回**
`worker_falsification_1/report.md:600-603` 称把每个区间乘同一放大 R 只「破坏」非线性分配、均匀间距保持相对谐波比。**但乘公共标量对均匀与非均匀 z 都保持所有相邻 log 区间的比**，代数不蕴含符号反转、也不偏袒均匀 z。有效主张：受控反转证明 (R,z) 与学习/训练行为存在**相互作用**；机制仍是经验的，可能包含训练支撑、系数协同适应、有限 lag 相位巧合、任务分布。

### 4.2 sol19 保留的窄结论与「不可信清单」

窄结论（sol19:5-11）：无序谱摘要不能认证冻结 checkpoint（旋转槽被学得的 Q/K 坐标标注）；离散尺度轨道计数不是稳定物理选择子；小的无权表误差不认证行为；固定支撑分配与支撑重定标实验上耦合；静态几何是诊断不是任务成功目标。
不可信证书：
- 历史 "VICTORY CONFIRMED" 审计**不可作为闭合证书**（自称零幻觉数学，却与上面的无支撑 Fisher 谱与普适定理共存）[原文 `auditor_victory_document_2/report.md:9-17,215-224`]
- `teamwork_preview_document_2/DOCUMENT_REVIEW_REPORT.md:860-868` 内部不一致（先说没有破数学，后依赖其自身批评未确立的容量守恒/KV 干涉机制）
- 同多重集坍缩是**冻结 checkpoint 标签槽**的决定性证据，不是 scratch 训练的证据
- FullLagP2 是 Qwen-1.5B@64K 的条件局部成功、别处混合/失败迁移；旧报告的「相位过早饱和」「架构打乱」是假设不是识别机制
- MrRoPE-Pro 是实测稳健的**启发式**；快/慢截断与渐进过渡不是推导最优；补救文档称其 "satisficing"（`challenger_remediation_1/report.md:121-145`）
- 手稿审计：rank 与 whitening 忽略学得系数幅度（`critic_r1/r1_reviewer2_audit.md:47-78`）；几何→损失链含无界近似步（`:141-164`）

### 4.3 sol19 的「一框架两 regime」与可执行件

签名对比（sol19:66-78）：
```
Z_r(ν) = Σ_{h,k} [ A^c_{rhk}(cos(ν_k Δ_s) − cos(ν_k Δ_d))
                 + A^s_{rhk}(sin(ν_k Δ_s) − sin(ν_k Δ_d)) ]
μ_r(ν)=E[Z_r],  Σ_r(ν)=Cov(feature contrasts),  ψ_r(t;ν)=log E exp(t(Z_r−μ_r))
(1) P(任一 distractor 胜过源) ≤ N_r exp[−t μ_r(ν) + ψ_r(t;ν)]
(2) 中心化 sub-Gaussian ψ ≤ t²v/2 下优化 t ⇒ J_r(ν) = μ_r(ν)²/(2 v_r(ν)) − log N_r
```
- (2) 是可直接进 F 的紧凑标量：同时含正签名源 margin、相干干涉、distractor 多重性；**失效暴露干净：零或错符号均值不能靠降协方差修复**。

scratch/协同适应分支（sol19:96-116）：
```
mu_b = n_b a_b;   v_b = n_b σ_b² + n_b² γ_b²
(3) SNR_b(n_b) = n_b a_b² / (σ_b² + n_b γ_b²)      ← 递增且凹，dSNR/dn = a_b²σ_b²/(σ_b²+n γ_b²)² > 0
(4) max Σ_b q_b SNR_b(n_b),  n_b ∈ Z_+,  Σ n_b = K
```
- 可分离离散凹效用下**单位贪心分配即精确**；一般有限 lag 协方差用 DP。连续极限下（常数 a、可交换通道、对角线 ridge + 嵌套尾/Green 分量的协方差算子）**还原 EVQ 逆协方差问题**；**Cosh 密度因此是条件松弛，不是普适律**。若 a_b 变化或签名均值不同，KKT 变成 **active-set 逆协方差规则，一般非 Cosh**。[部分证据-推导]
- 该 regime 是「无序密度推理可以合法」的 regime：槽身份可与学得权重一起被置换。

frozen 分支（sol19:120-138）：
```
(5) min_r J_r(ν) ≥ η_r;  P(native 损失/margin 违反 role r) ≤ α_r;
    ν_0 ≥ … ≥ ν_{K−1};  端点固定
(6) D[k,b,c] = cost(k,b,c) + min_{b' ≤ b, c'} D[k−1, b', c']     ← 单调分配 DP，c = 已用压缩/计数预算
```
MrRoPE 供给**安全结构化域**（保留快 native 槽、界住慢槽相位、把有限 log 压缩预算经过渡分配）——是约束/初始化，不是固定真理。**全稠密跨槽协方差破坏该可分离性**；此时用精确整数二次/锥问题，或把协方差充分统计量放进状态；**不要为了迁就 DP 而强行对角化**。

### 4.4 sol19 的判定-sufficient 下一步（与 sol17 §2.7 同构）

在**已生成的同一批开发输入**上，对 MrPro / Smooth_MrBudget / E1 slot28 / P2 分开算 `Z_r`、签名均值、联合有限 MGF（或正当的上包络）、`log N_r`，区分源依赖长行与 native/短行；冻结标签、前缀、解码、顺序、初始状态、表 hash；**不得用生成正确性去拟合矩**。规则只有在事前（看 holdout 生成标签前）满足以下四条才赢得一个冻结候选：否决 Smooth（无符号几何偏好它的地方）；保住 E1 的轻微长面板方向但不声称确认；表示 P2 的长/短权衡而非平均掉它；满足 (5) 的 native 角色约束。**若签名 Q/K margin 在失败行上仍把 Smooth 排到 MrPro 之上，就停止**——缺失机制在测量的下游（源标注、上游前缀形成、值输运、head/输出混合、自回归解码轨迹），再加一个频率形状代理无济于事。[sol19:150-161]

---

## 5. sol18 配套资产：29 个联合模式候选（joint mode retiming）
源（代码 + 证据 JSON，本地完整读）：`docs/research/rope_allocation_20260910/code/joint_mode_candidates.py`（162 行）、`docs/research/rope_allocation_20260910/evidence/joint_mode_candidates.json`

### 5.1 构造与公式

```
(核心式)  ν_c = ν_M + n · (nᵀ ω_native / S − nᵀ ν_M) / (nᵀ n)        # S=4
n ∈ {[1,−1] × 15 个起点} ∪ {[1,−2,1] × 14 个起点}   起点范围 FIRST..LAST = 24..39
```
[joint_mode_candidates.py:48-52,63-67,130] 每个候选从**实际 MrPro FP32 张量**出发，只改指定的相邻 pair/triple，把该 native 关系时钟除 4。n 零和 ⇒ 保 Σν、端点逐位固定（`endpoints_bitwise_equal`）、严格递减，且报告 29/29 全部为 `CPU_VALID_CANDIDATE`。[已验证-本地复核：`candidate_count=29, valid_candidates=29`，29 个互异 tensor_sha256]

### 5.2 关键实测（本地复算，逐条）

| 声称 | 本地复核结果 | 判定 |
|---|---|---|
| 29 个候选 | `candidate_count=29`，`orders=[1,2]`，`transition_slots_zero_based=[24,39]` | [已验证] |
| MrPro 对 29 个相邻关系时钟 **0/29** 做了 ×4 重定时 | `mr_already_retimes_native_by4` 求和 = **0** | [已验证] |
| **3/29** 候选把某槽推得比 MrPro 还快（框内允许） | `slots_faster_than_mr` 非空 = 29/29（每个候选都至少有一槽更快）；`slots_faster_than_native` 非空 = **3/29**；`compression_box_0_to1` all=True = **False** | **口径需澄清**（见 §7 FLAG-3） |
| Σm 质心漂移范围 | `delta_sum_m ∈ [−0.01826469, +0.00185344]`，MrPro `sum_m = 29.33333327` | [已验证]（与 INTEGRATION §1 R4 的 −0.01826…+0.00185 逐位吻合） |
| 相位改动幅度 | `max_abs_delta_phase_at_128k` 最大 **58.31 rad** | [已验证] |
| 保序性检查 | 全部候选 `strictly_decreasing=True`、`unchanged_outside_relation=True`、`exact_projection_identity_pass=True` | [已验证] |

- 文件自标状态 `CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION`，qualification 字段明写：「**不得仅凭时钟/幅度/几何/native 梯度选择。只在方向过滤中把实际全模型长/原生梯度符号当过滤器使用，然后用精确有限全模型损失与生成任务端点**」。limitations 里点名：正确的关系时钟**不**证明该模式承载有用计算；关系相位系数与上游隐状态在全模型里会变；**零和 n 保的是原始频率和，log-压缩指数之和不受约束**。[已验证-本地读]
- 与 KKT 的接口：这是「**carrier-preserving 的重定时打破 ν_j ≤ ω_j 的 compression-only 盒**」的可执行实例（INTEGRATION §4.2 同判）；也因此否掉 "null band 可以自由搬" 的直觉（sol17:19-21）。

---

## 6. 回传件 sol20 / astra10
源：`docs/research/rope_allocation_20260910/recovered/sol20.md`（23 行）、`recovered/astra10.md`（22 行）
状态：**两份原始报告未落盘**；`archive_manifest.json` 的 `missing_original_reports = ['sol20','astra10']`、`missing_original_read_receipts = ['sol18','sol20','astra10']`（本地已核）；`archive_status = '28_original_reports_plus_2_labeled_recoveries'`。这两份是**整合席位的整理件，不是代理原稿**，也未伪造补全。

### 6.1 sol20：OLMo BM 结果（已对一手来源逐项复核）

来源：`docs/research/ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.md` 与同名 JSON。模型 OLMo-2-0425-1B-Instruct，静态 S4，16K，7 任务各 50 条 = 350 条/臂。

| 任务 | MrPro | BM |
|---|---:|---:|
| niah_single_1 | 20.00% | 90.00% |
| niah_single_3 | 0.00% | 50.00% |
| niah_multikey_1 | 12.00% | 60.00% |
| niah_multikey_3 | 0.00% | 0.00% |
| niah_multivalue | 5.00% | 60.50% |
| cwe | 0.60% | 5.20% |
| qa_2 | 12.00% | 26.00% |

[已验证-本地复核 JSON]：`result.candidate.macro_accuracy(16384) = 0.416714…`，`result.baseline… = 0.070857…`，`macro_delta_by_length = 0.345857…`（+34.59pp），`paired_wins = 156`，`paired_losses = 9`，`candidate.eos_count = 119`、`baseline.eos_count = 197`，`rows = 350`。
- **不能据单来源收益宣称多重绑定已解决**：multikey_3 双方仍 0%。
- **EOS 口径警告（sol20 明写）**：公开召回评分与「完整字符串 + EOS」标准不同，不得混用；EOS 次数差（BM 119 vs MrPro 197）不能单独用来推断收益都来自格式/终止。[已验证-数字；解释=叙事-未验证]
- sol20 自评：这是有用的证据诊断，**不是已证明的频谱结构或新分配公式**。

### 6.2 astra10：active-set 修正与三条线索

**数学修正（可直接改 F 的建模）**：
1. 相关 C 下的非负最优解**不能**用 `[C⁻¹h]_+` 直接裁剪代替 **active-set 求解**；独立通道噪声与共享频率 bin 噪声必须区分；正均值/阈值条件不能从尾界里省略。[部分证据-与 INTEGRATION §4.2 一致，本文档已并入]
2. 自由系数规差**不能**自动导出唯一 Cosh 密度；密度、整数通道、幅度三者不可互换。
3. 实际注意力存在罕见高分 key 时，均值/协方差近似可能**错误排序**；精确 logsumexp 与二阶近似必须分开报告。
4. 已有「少文档 + 高维直接频率拟合」未建立稳健迁移；新的 whole-model CE 测量或拟合**不能仅因用了梯度就称为新的统一理论**。
5. **LeRoPE 先行工作定位**：频率损失梯度及 downstream cotangent 并非本轮首创；原回传引 https://arxiv.org/html/2607.10134v1 §3.2（归档保留为文献入口，未本地核）。[叙事-未验证-文献未核]
6. 对 Mr 锚定的联合模式投影：`/4` 的模式目标**仍是假设**；梯度仅可筛选方向，有限变化必须实际计算；固定 R、以 native 定义的投影半群性质**不能自动移植**到每次重新锚定 Mr 的操作。
7. 对新 YaRN/Mr 附件独立复算：公式、交叉槽位、局部扰动数值吻合；**尺度导数须固定 native 和边界**。

**四格判读法（可执行）**：F00=YaRN、F10=max、F01=min、F11=Mr；`min>Mr` 只改变 A 段；Mr 优于两个混合**不足以**证明协同；交互项必须看 `F11 − F10 − F01 + F00`，并注明指标尺度。[部分证据-与 STARTING_POINT §7 / NEXT_DERIVATION K6 同式]

**max_new_tokens 线索（本次任务点名的项）**：[已验证-本地复核，找到了具体不一致]
- `docs/research/ROPE_OLMO_BM_NATURAL_RESULT_20260908.json` 的顶层 `contract.generation_config.max_new_tokens = 16`，
- 但 `docs/research/ROPE_OLMO_BM_NATURAL_TRANSFER_20260908.md` 写「输出预算分别 32/32/128」，
- 而该 JSON 的 567 行里 `candidate_ids` 长度最大 **51**、`baseline_ids` 最大 **32**，直方图众数是 4（132 次）。
→ **顶层字段确实不可单独作为运行预算依据**；astra10 的提醒成立且可定位。这不是「新数据错误结论」，但报任何长度/截断相关口径前必须回到逐行运行来源核定。

---

## 7. 与权威文档的矛盾 / 口径不一致（逐条给两边出处）

**FLAG-A（最重要）— 32K 位置拉伸能否作校准分布：三份材料三种立场**
- sol16：**可以**，作为「位置拉伸校准分布」，但明确「不是真 128K 行为证据」（sol16:74）
- sol17：**只是 oracle 上限与方向发生器**，不测稠密 128K 额外 key 竞争、不测真 128K prefill 状态分布（sol17:123-129）
- sol18：**禁止**用于 128K 结论——「不要用拉伸/拼接 32K 捕获 K/V 制造 128K：既不保新增 key 竞争也不保上游状态形成」，必须用真实连续 128K prompt（sol18:76）
- 权威裁决：`INTEGRATION_20260910.md` FLAG-5 已判为**证据分层**（拉伸行 = 方向发生器与 oracle 上限；模型级判定必须真实连续 128K）。**本地实测支持分层**：`full_model_response_native.jsonl` 显示 32K 下梯度质量 76.4% 住在槽 0–6、过渡带 24–39 只占 3.35%，即「32K 响应住在被冻结的快槽」（sol18 实测结论，INTEGRATION §3 同判）。→ 三份材料不平均、写标签即可，但**任何把拉伸结果当模型级证据的引用都是错的**。

**FLAG-B — 16-DOF 的三种命名不是同一个东西（但自由度数目一致）**
- sol16 说 17 个正增量、16 个有效自由度（softmax-Helmert 坐标）
- `NEXT_DERIVATION_KKT_PROBLEM.md` §1.1 写 `Δ = (Δ_23,…,Δ_39) ∈ R¹⁷`（17 个），同文 §1.3 说「可行域 = 16 维单纯形（最后一个 Δ 由等式消去）」——自洽
- `INTEGRATION_20260910.md` §3 写「自由增量 Δ_j 在槽 24–39（**16 个**）」——**与上者的 17 个增量口径不一致**
- sol18 写「只优化槽 24–39」= 16 个槽值 → 与 16 DOF 一致
- 本地核算（`sol16_frequency_calibration_reference.py`）：`increments()` 17 个、`exponents()` 长 64（24 + 17 + 23），槽 40 恒 1 → 17 增量 / 16 自由 DOF / 16 个内点槽值，三者数目自洽；把「17 个增量」写成「16 个增量」是**命名口径错，不是数学错**。权威裁决 FLAG-6 只说「同单纯形不同坐标」，未点出这个口语差；引用时须写清「17 个过渡 gap / 16 维自由」。

**FLAG-C — sol16 的 phase16 重锚引文未能在本地复核**
sol16:63 引 `scripts/core_text_phases/phase16_phase_allocation_budget_matrix_m4.py:84-93` 为「midpoint EVQ 后首末 φ 仿射重锚到原生端点」。本地该文件存在（17,299 字节），但 `:82-95` 区间是 `score == "phase-isotropy" / "pair-volume"` 的密度-CDF 代码，未见所述重锚。→ 该条降为 [叙事-未验证]，**不影响** sol16 的主要建议（用原生 FP32 表的乘性膨胀绕开歧义）。

**FLAG-D — mol 与"统一"的措辞强度**
- sol19 最终边界（:171）明确：「**没有**历史证明构造性频率分配不可能；**有**强证据证明 model-blind 无序标量不能认证冻结部署」，并说统一是**条件性**的。
- 这与 INTEGRATION §4.1 引 sol15 的「这些受限归约中没有一条确立与 checkpoint 无关的 LM 最优曲线」一致。
- 但 `archive_manifest.json` 的 notes 更保守：「**No new Qwen allocation was selected or validated from the 29 CPU-created joint-mode tables**」「No new inference, training or server inspection was performed for this archival task」。→ 凡本批材料里出现「已闭合 / 已证明 / VICTORY」式措辞，一律按红线降级处理。

**FLAG-E — sol18 无 coverage 回执**
`archive_manifest.json` 的 `missing_original_read_receipts` 含 `sol18`；`docs/research/rope_allocation_20260910/coverage/` 下确实无 `sol18_coverage.json`（本地已列目录确认）。故 sol18 自述的「101 个文件 1,107,673 字节全读」是**自报**而非有回执的覆盖。[叙事-未验证-覆盖层]

---

## 8. 死路登记（本批材料新增/加固的"绝不能再试"）

1. **静态代理选冻结表**：协同适应表给出决定性一例——静态 Rényi-2 秩从 4.57 改善到 12.54 的那一格正是 PPL 崩到 76.20 的那一格（`paper-2027/tables/table_coadapt.tex:6-21`）。[已验证-本地] 与红线 R2 同向。
2. **null band 自由搬**：联合模式否掉——重定时一个关系同时保正交载体有时**要求加速**，纯压缩 ramp 排除它；29 候选里 3/29 把某槽推到比 native 还快（`compression_box_0_to1` 不全真）。[已验证-本地复算]
3. **逐槽可加性 / 逐槽列表法**：pair28_29 不可组合（INTEGRATION §4.3/sol07），slot 级收益非加性。
4. **把 scratch 密度移植到冻结槽**：无标签槽不可承受；`h_i n` 必须换成逐标签 `h_{j,i}` 前缀和形式（sol17:92-98）。
5. **MGDA 当统一优化器**：只回答局部问题，兼容/对抗两例都自带；不放源标签、不认证有限 S=4 端点（sol17:23-25）。
6. **Taylor/Jacobian 局部代理**：相位偏移 22.74 / 90.97 rad 下退役（sol19 §1.2）；有限旋转界饱和后太松无法排序。
7. **用 17 个自由 logit 参数化过渡**：引入无用常数平移零方向，16 个存储标量才全可辨识（sol16:53）。
8. **`[C⁻¹h]_+` 裁剪冒充 active-set**（astra10 修正 + INTEGRATION §4.2）；以及**为迁就 DP 强行对角化协方差**（sol19:138）。
9. **iid 通道噪声推导 EVQ 密度**：会给密度**线性**贡献，无法还原平方密度代价 → 水床要求显式声明的相干/嵌套协方差（sol17:71-73；sol19 反例 2）。
10. **argmax/生成正确性拟合矩**：argmax 无训练梯度；矩不得用生成正确性拟合（sol18 陷阱 10；sol19:152）。
11. **复用 candidate-A 的 K/V 算 candidate-B 的 CE**（sol18 陷阱 6）；**未替换 `@torch.no_grad` 的 forward 而以为在学频率**（sol16:85 + sol18 陷阱 5 的 #1）。
12. **gain 乘两次（g⁴）**（sol18 陷阱 1，本地已在 `pro_block_calibration.py` 复核该乘法的存在）。
13. **「YaRN 递减 vs MrPro 递增」叙事**：与 STARTING_POINT F1/F2 一致，本批无新增反例，红线维持。
14. **把 32K 拉伸/拼接冒充 128K**（sol18:76；FLAG-A）。

---

## 9. 未解问题（本批材料产生的）

1. **带角色标签的签名矩从未被测**——四家（sol16 §1.6 步骤 3、sol17 §2.7、sol18 §3、sol19 §4.4）一致指向同一缺口：pre-RoPE Q/K 上按 (native/far) × (source/hard-distractor) 拆分的带符号均值/协方差/单位对数 MGF，含跨槽协方差，保 layer/head/relation/lag 标签。
2. **预注册门槛未跑**：统计量必须先正确否决 Smooth_MrBudget（slot-28 符号反转）、并暴露 P2 的 +long/−short 权衡——**通过前不得出表**（sol17 §2.7、sol19 §4.4、INTEGRATION §8-1）。
3. **`C_il = (α/Δ)1{i=l} + β min(x_i,x_l)` 这个结构协方差是否足以刻画真实 Qwen nuisance**——未测（sol17 §7 明列 Unresolved）。
4. **一般异质跨槽协方差下的 DP 失效后走哪条路**：Astra03 认证有限 margin 锥求解器，或"报告模型不受支持"二选一，尚无实例。
5. **sol16 的 16 维校准表 vs sol17/sol19 的角色 margin 表**：若二者方向一致 → CE 优化器成为独立 oracle 对照；若不一致 → 需诊断哪个冻结态假设失败。**这条桥从未被跑过。**
6. **等范数镜像方向（sol18 §3 判定项 3）** 是否真的能区分"有用方向"与"一般扰动"——未验证。
7. **E3_BM 命名（Q7）** 仍未解决（INTEGRATION FLAG-7 同判）。
8. **max_new_tokens 顶层字段与实际运行预算的对应规则**：已定位一处不一致（NATURAL_RESULT），但**是哪些 run 覆盖了该字段、按什么规则**未定（astra10 保留为待核对线索）。
9. **LeRoPE（arXiv 2607.10134 §3.2）** 的先行边界未本地核（astea10 引用；本批无本地副本）。
10. **sol18 覆盖回执缺失**：其 101 文件全读自述无法从仓库侧验证。

---

## 10. 覆盖度与本地可验证性

**读完**：`agents/sol16.md`(121)、`agents/sol17.md`(153)、`agents/sol18.md`(133)、`agents/sol19.md`(171)、`recovered/sol20.md`(23)、`recovered/astra10.md`(22)，全部逐行。
**为复核而读的旁证**：`code/sol16_frequency_calibration_reference.py`(143 全文)、`code/joint_mode_candidates.py`(162 全文)、`evidence/joint_mode_candidates.json`(程序化解析)、`evidence/full_model_response_native.jsonl`(4 行全解析)、`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}`(全文)、`archive_manifest.json`、`coverage/{sol16,sol17,sol19}_coverage.json`、`docs/research/ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.{md,json}`、`docs/research/ROPE_OLMO_BM_NATURAL_RESULT_20260908.json`、`paper-2027/tables/table_coadapt.tex`、`scripts/lib/rope/{schedules.py,learnable_evq.py}`、`scripts/analysis/build_boundary_matched_mrpro.py`、`experiments/nongeometric_screen/pro_block_calibration.py`、`experiments/{pm_keep/prepare.py, nosa_position/test_data.py}`、`.venv/.../transformers/models/qwen2/modeling_qwen2.py`(4.57.6)。

**本地不可核（材料引用但仓库内不存在）**：
- `.agents/` 与 `.qoder/` 原件 —— sol19 的四条撤回**全部**指向 `worker_falsification_1/report.md`、`challenger_remediation_1/report.md`、`critic_r1/r1_reviewer2_audit.md`、`auditor_victory_document_2/report.md`、`teamwork_preview_document_2/DOCUMENT_REVIEW_REPORT.md`。本地 `.agents/` 不存在；`outputs/.agents_archive_20260906.tar.gz`（487KB，230 条目，09-06）**不含** `worker_falsification_*`（本地 `tar tzf | grep -c` = 0），只有另一代 `teamwork_preview_victory_auditor_*`。→ sol19 的撤回**无法从本地复核其一手出处**（只在 assignments/coverage 回执里留有路径与 sha256）。
- `artifacts/sparse_memory_20260908/eval_v1_tp/rows.jsonl`（sol16 的 6,144 行）、`development_rows.jsonl`（sol19 的 1,536 行）—— 本地缺失 → sol16 §1.8 与 sol19 §5 的两组面板数字（relation 1.0 / content 0.1948；1,024 relation + 512 content；surrogate 280 点/218 失败/77.86%）均 [部分证据-本地无法复核]。
- `experiments/rope_operator_family/results/20260909_gpu/REPORT.md`（sol18 的 2.6514 / 9.9899 / 3.9427 三 NLL）—— 本地缺失；同数字在 `docs/research/BRANCH_09_09_BRIEF_20260909.md:39`（只留 output-KD 3.942717）与 `agents/astra04.md:88` 有独立记载 → 交叉一致 [部分证据]。
- `docs/theory/EVQ_COSH_THEORY.tex`、`docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`、`scripts/analysis/rope_transport/tables.py`、`scripts/analysis/verify_general_rope_allocation.py`、`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md` —— 未读（不在本次分配范围），相关引文一律 [部分证据-引文未核]。

**未读（超出本次任务）**：`agents/` 其余 24 份（sol01–15、astra01–09）、`digests/`、`digests_codex/`、`codex` 理论核心文档、`.codex` 会话目录（严格只读且未进入）。

**未做**：未运行任何模型/GPU 任务；未修改仓库任何文件（除本输出文件）；未写入 `~/.codex`。
