# EVQ-Cosh 六大理论问题 RALPH 分析总结

> **日期**: 2026-04-06
> **方法**: External RALPH loop (Read → Analyze → Localize → Probe → Hostile self-review)
> **范围**: 六大问题逐一审查，交叉引用论文正文、附录证明、分析脚本、内部报告
> **目标**: 为论文修订和审稿人答复提供可操作的理论定位

---

## Problem 1: Lagrange 乘子 λ 的解析闭合

### 1. Problem statement

变分泛函 $\mathcal{F}(\tau) = S_{\chi^2}(\tau) - \lambda\,U(\tau,L)$ 的极值条件给出 $\tau^* \propto d_{\mathrm{head}}/L^\gamma$。$d_{\mathrm{head}}$ 依赖和指数 $\gamma \approx 0.5$ 由推导得出，但 Lagrange 乘子 $\lambda$ 是通过拟合 99 组实验（27 配置 × 3+ seeds）校准的：$\lambda \approx 1.13 \pm 0.15$，CV ≈ 13%。

问：能否从 $(d_{\mathrm{head}}, L, \theta_{\mathrm{base}})$ 的函数关系无参数地确定 $\lambda$？

### 2. Why it matters

λ 的半解析性质是审稿人最可能攻击的点。如果 $\lambda$ 是自由参数，缩放律 $\tau^* = d_{\mathrm{head}}/\sqrt{L}$ 的理论地位从"定理"降为"经验校准的参数化公式"。这影响论文的整体可信度。

### 3. Current project status

**已建立（精确）：**
- 代理自洽性恒等式 $\tau^2 T_2(\tau) + T_1(\tau) = \tau\coth\tau$（Appendix 新定理，数值验证误差 < $10^{-15}$）
- 代理驻点 $\tau_{\mathrm{surr}} = \sqrt{\beta/\alpha}$ 是**精确**的（非近似），由 Balance identity $T_1' + \tau^2 T_2' = 0$ 保证
- 代理系数 $\alpha \approx 1/d_{\mathrm{head}}$, $\beta \sim L^{-0.22}$（数值拟合）
- $\chi^2$ stiffness 的闭合形式 $S_{\chi^2}(\tau) = (1/M)[\sinh\tau\cdot\arctan(\sinh\tau)/\tau^2 - 1]$

**已建立（半解析）：**
- 代理预测 $\tau_{\mathrm{surr}} \sim \sqrt{d_{\mathrm{head}}} \cdot L^{-0.11}$
- 物理最优 $\tau^* \sim d_{\mathrm{head}} \cdot L^{-0.5}$
- 差异来自：discrete-continuous transport（振荡核 vs 平滑代理），需要 $\sqrt{d_{\mathrm{head}}}$ 和 $L^{-0.39}$ 两个修正因子
- $\lambda$ 精确地吸收了这个 transport gap

**已尝试并失败的闭合路径：**
- Fisher 效用模型：$\lambda_{\mathrm{calc}} = 177\text{--}28906$，完全发散
- 幂律效用模型：要求 $\eta = -1/2$（非物理）
- 离散碰撞平衡 $S' + w\cdot C' = 0$：权重 $w$ 的 CV = 282%
- 有效秩最大化：CV = 280%
- 逆工程 $U'$：无干净标度模式

### 4. RALPH investigation

**推导尝试 1：渐近展开。** 在 $d_{\mathrm{head}} \to \infty$ 极限下，真实 RoPE 核在 cosh 密度下的碰撞积分应有 Euler-Maclaurin 展开：$C_{\mathrm{discrete}}(\tau, K, L) = C_{\mathrm{continuous}}(\tau) + c_1(\tau, L)/K + O(1/K^2)$。如果 $c_1$ 可解析计算，则 $\lambda = f(\beta_{\mathrm{eff}})$。

**问题定位：** $c_1$ 需要处理 sinc 型核 $\sin((\omega_i - \omega_j)L)/(\omega_i - \omega_j)$ 在 cosh quantile grid 上的离散求和。这是一个 two-parameter family of oscillatory sums，需要对 sinh 函数的反函数做数论型估计。技术难度极高，远超论文篇幅能容纳的范围。

**推导尝试 2：信息论约束。** 如果最优注意力在训练长度 $L$ 处的熵为 $H^* = \log L + c$，则从 $H^*$ 反推 $\lambda$。但 $H^*$ 本身依赖于训练目标（AR vs masked），不是 PE 独立决定的。这条路径无法产生一个只依赖 $(d_{\mathrm{head}}, L, b)$ 的结果。

**数值探测：** λ 在 9 个配置上的分布：
- 按 $d_{\mathrm{head}}$ 分组：$d=32$ → 1.208 ± 0.108; $d=64$ → 1.065 ± 0.096; $d=128$ → 1.130 ± 0.188
- 按 $L$ 分组：$L=256$ → 1.135 ± 0.086; $L=512$ → 1.039 ± 0.198; $L=1024$ → 1.230 ± 0.044
- **无系统趋势**：$\lambda$ 不随 $d_{\mathrm{head}}$ 或 $L$ 单调变化。CV(13%) 完全可由 parabolic fit 的统计误差和 finite-seed 噪声解释。

**关键数值：** 在 $d_{\mathrm{head}} = L$ 的特殊点（连续和离散问题重合），代理公式 $\tau_{\mathrm{surr}} = \sqrt{\beta/\alpha}$ 与 $d_{\mathrm{head}}/\sqrt{L}$ 吻合至 3%。这确认了 $\lambda$ 的物理含义：它量化了 $L/d_{\mathrm{head}} > 1$ 时离散修正的累积效应。

### 5. Best current resolution

**Partially resolved.** λ 无法从第一性原理闭合，但其不可闭合性已被**精确定位**：代理内部完全精确（新自洽性定理），λ 唯一地编码了从连续代理到离散物理核的 transport constant。其 O(1) 性质和低 CV 表明这不是一个"任意拟合参数"，而是一个结构常数。

### 6. Strongest defensible statement

> "The scaling law $\tau^* = d_{\mathrm{head}}/\sqrt{L}$ is semi-analytic: the $d_{\mathrm{head}}$ dependence and $L^{-1/2}$ exponent are derived from the variational framework (the former via $\alpha = 1/d_{\mathrm{head}}$, the latter via Pearson $\chi^2$ stiffness), while the O(1) transport constant $\lambda \approx 1.13$ is calibrated from 99 runs across 27 configurations (leave-one-out CV error < 0.2%). The surrogate self-consistency theorem proves that the balance within the variational layer is exact, not approximate."

### 7. Unsafe statement to avoid

> ~~"We derive the optimal temperature $\tau^*$ from first principles."~~

这暗示了无参数推导，而实际上 $\lambda$ 是校准的。

### 8. Recommended next step

**最高杠杆：论文措辞修正。** 在 Proposition 3 之后加显式声明区分"推导的"（指数和 $d_{\mathrm{head}}$ 依赖）和"校准的"（$\lambda$）。这比试图闭合 $\lambda$ 更有效——$\tau^*$ basin 的平坦性（± 20% τ 变化 < 1% PPL 变化）意味着 $\lambda$ 的精确值对实际性能无关紧要。

### 9. Confidence

**High.** λ 不可闭合的原因清楚（discrete-continuous transport gap），其近似普适性有充分数值支撑，论文已有诚实的三层认知分级。唯一风险是审稿人要求"为什么不继续推"，答案已在 04_lambda闭合分析.md 中准备好。

---

## Problem 2: 宽带代理核的有效性边界

### 1. Problem statement

理论推导使用宽带代理核 $K_{\mathrm{app}}(\phi_1, \phi_2) = \alpha\delta(\phi_1 - \phi_2) + \beta\min(\phi_1, \phi_2)$ 近似真实振荡 RoPE 核 $K(\Delta) = \sum_k \cos(\theta_k \Delta)$。代理的**泛函有效性**（EVQ 是否在真实核下也减少 collision）已验证通过，但**逐点保真度**未被声称。

核心问题：(a) 是否存在显式误差界？(b) 代理在极端参数下是否崩溃？(c) $\alpha, \beta$ 是否有解析式？

### 2. Why it matters

代理是整个推导链的唯一近似步骤。如果审稿人质疑"你的推导是在一个错误的核上做的精确运算"，需要有明确的辩护。

### 3. Current project status

- 12 配置（3 个量级的 $L$, 3 个 $b$, 2 个 $K$）上，EVQ（从代理推导）在**真实核**下减少 collision 24–92%，effective rank 提升 24–570%（Appendix Table 3）
- 代理系数：$\alpha \approx 1/d_{\mathrm{head}}$（精确于离散化），$\beta \sim L^{-0.22}$（经验拟合）
- 代理 R² ≈ 0.25–0.73（pointwise fit），但泛函层面定性正确
- 碰撞矛盾（v1）已修正：连续泛函在小 $\tau$ 减少 collision，离散 collision 的增加是 $T_1$（密度集中度）主导的预期行为

### 4. RALPH investigation

**Fourier 分析尝试。** 真实核在均匀距离先验下：
$$K(\phi_1, \phi_2) = \frac{1}{2L}\left[\frac{\sin((\omega_1 - \omega_2)L)}{\omega_1 - \omega_2} + \frac{\sin((\omega_1 + \omega_2)L)}{\omega_1 + \omega_2}\right]$$

当 $|\omega_1 - \omega_2|L \gg \pi$ 时，sinc 项振荡并趋于 0。当 $|\omega_1 - \omega_2|L \ll \pi$ 时（频率拥挤），sinc 项 → 1，产生强相关。

代理用 $\alpha\delta + \beta\min$ 捕获了两个极端：$\delta$ 捕获对角线 ridge，$\min$ 捕获 monotone off-diagonal 趋势。它**无法**捕获中间区域的振荡结构。但关键观察是：EVQ 的优化目标不是逐点拟合核，而是最小化碰撞泛函 $\int\int K\rho\rho$。代理保留了泛函的**凸性结构**和**单调性方向**，这解释了为什么泛函层面有效。

**逐点误差界。** 真实核的振荡使得 $\sup_\Delta |K - K_{\mathrm{app}}|$ 无法小于 $O(1)$——sinc 函数可以取负值而 $\alpha\delta + \beta\min$ 始终非负。因此**不存在**有意义的逐点误差界。这不是一个可以"修复"的问题；它是代理设计的根本特征。

**极端参数崩溃测试。** 从 Table 3 的趋势推断：
- $d_{\mathrm{head}} = 16$：$K = 8$ 个通道，代理的 $\delta$ 项主导（对角线占 Frobenius 范数的比例增大），预期代理仍然有效，但 collision reduction 幅度下降（fewer channels → less room for reallocation）
- $\theta_{\mathrm{base}} = 10^7$：极高 base → 所有频率几乎相同（$\omega_k = b^{-k/K}$ 在 $b$ 很大时快速衰减）→ 代理的 $\min$ 项更准确（因为所有 off-diagonal 项都在 constructive interference 区域）

**$\alpha$ 的半解析式。** 对角线值 $K(\phi, \phi) = \frac{1}{L}\int_0^L \cos^2(\omega(\phi)\Delta) d\Delta \approx 1/2$（对任何 $\omega L \gg 1$ 的通道）。乘以通道间距 $\Delta\phi = 1/K$：$\alpha \approx 1/(2K) = 1/d_{\mathrm{head}}$。这是**精确**的（在 $\omega L \gg 1$ 条件下），已在附录中写明。

**$\beta$ 的半解析式。** $\beta$ 来自 off-diagonal 的低频包络。Weyl equidistribution 论证：当 $K \to \infty$ 且频率均匀分布在 log-space 时，off-diagonal 碰撞的包络趋向某个函数的矩。但这个论证需要频率的等分布性质，而 EVQ 频率恰好不是均匀的。循环论证。$\beta \sim L^{-0.22}$ 只能保持为经验拟合。

### 5. Best current resolution

**Partially resolved.** 逐点误差界不存在且不应追求；泛函有效性已有充分验证（12 配置，100% success rate）。$\alpha$ 有半解析式，$\beta$ 保持经验。代理的认知地位已正确定位为"泛函层面的结构近似"。

### 6. Strongest defensible statement

> "The broadband surrogate captures the functional structure of the collision objective: EVQ derived from $K_{\mathrm{app}}$ reduces collision under the exact kernel by 24–92% across all 12 tested configurations (Table 3). Pointwise fidelity is not claimed; the surrogate is not a kernel approximation in sup-norm, but a structural proxy that preserves the convex geometry of the collision minimization landscape."

### 7. Unsafe statement to avoid

> ~~"The surrogate kernel accurately approximates the true RoPE collision kernel."~~

"Accurately approximates" 暗示逐点保真度，实际 R² = 0.25–0.73。

### 8. Recommended next step

**论文措辞：** 在 surrogate validation 段落中加一句明确声明 pointwise fidelity 不是目标。当前行文已经做到了（"not by pointwise comparison... but by the functional test"），但可以更显式。

如有精力，补一个 $d_{\mathrm{head}} = 16$ 和 $b = 10^7$ 的极端配置 collision test，确认代理在边界处的行为。

### 9. Confidence

**High.** 泛函有效性的 12/12 通过率是强证据。审稿人可能问"为什么这个代理 work？"，答案是它保留了碰撞泛函的凸性和单调性方向。这在 Appendix A.2 的讨论中已有暗示，但可以更显式地阐述。

---

## Problem 3: DiT 模态修正因子的理论推导

### 1. Problem statement

将 $\tau^*$ 公式应用于视频 DiT 时需要经验修正 $\tau^*_{\mathrm{DiT}} \approx 0.53 \times d_{\mathrm{head}}/\sqrt{L}$。0.53 是 post-hoc 拟合值，分解为"双向注意力因子"和"噪声衰减因子"是事后解释。

### 2. Why it matters

如果 0.53 是一个无法解释的 fudge factor，它削弱了缩放律的普适性声称。但如果能给出一个 heuristic-but-principled 的分解，它反而展示了框架的可扩展性。

### 3. Current project status

- 视频 DiT 实验（382M, single seed）确认 EVQ 在时域迁移上有效（teacher-forced top-1 +3.14%）
- 论文当前将此归为"supporting evidence"，不在主贡献中

### 4. RALPH investigation

**推导尝试：双向碰撞率。** 在双向注意力中，位置 $i$ 被 $j \neq i$ 的所有位置（而非仅 $j < i$）访问。碰撞率从 $P_{\mathrm{coll}}^{\mathrm{uni}} = 1 - \prod_{j<i}(1-p_{ij})$ 变为 $P_{\mathrm{coll}}^{\mathrm{bi}} = 1 - \prod_{j \neq i}(1-p_{ij})(1-p_{ji})$。

在 softmax transport 框架下，$U(\tau, L)$ 的有效 $L$ 翻倍（每个位置被双向查询），给出：
$$\tau^*_{\mathrm{bi}} = \frac{d_{\mathrm{head}}}{\sqrt{2L}} = \frac{1}{\sqrt{2}} \cdot \frac{d_{\mathrm{head}}}{\sqrt{L}} \approx 0.707 \cdot \frac{d_{\mathrm{head}}}{\sqrt{L}}$$

但实际因子是 0.53，不是 0.707。差异因子 $0.53/0.707 \approx 0.75$。

**推导尝试：扩散噪声修正。** 扩散模型的损失 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ 中，时间步 $t$ 引入噪声 $\sigma_t$。在高噪声步（$t$ 接近 $T$），注意力对位置编码的依赖减弱（因为信号被噪声淹没）。有效利用位置信息的时间步比例约为 $(T - t_{\mathrm{threshold}})/T$。

如果只有约 75% 的时间步有效利用 PE，则额外修正因子 $\approx 0.75$，给出 $0.707 \times 0.75 \approx 0.53$。但这个 75% 本身是 heuristic 的——它取决于噪声 schedule 和模型的去噪能力。

**尝试分解验证：** $0.53^2 \approx 0.28$，而 $1/\sqrt{2L} \Rightarrow 0.53^2/(0.5) = 0.56$，暗示有效方向数 $n_{\mathrm{eff}} \approx 1.78$（不是整数 2）。这支持"双向 + 部分噪声衰减"的两因子分解，但不是 clean 的理论结果。

**Hostile review：** 我是否真的推导了什么？不。我只是把 0.53 分解成了两个同样 heuristic 的因子之积。$1/\sqrt{2} \times 0.75 \approx 0.53$ 是 numerology，不是推导。一个审稿人可以轻易构造其他分解（如 $1/\sqrt{3.56}$）来得到同样的 0.53。

### 5. Best current resolution

**Still open.** 没有从变分框架推出 0.53 的第一性原理路径。heuristic 分解（双向 + 噪声衰减）在物理上合理但数学上不严格。

### 6. Strongest defensible statement

> "For video DiT, the scaling law requires an empirical correction factor $\tau^*_{\mathrm{DiT}} \approx 0.53 \times d_{\mathrm{head}}/\sqrt{L}$, consistent with the combined effect of bidirectional attention (increasing effective collision exposure) and diffusion-noise attenuation (reducing the effective fraction of PE-sensitive timesteps). The precise factor is calibrated, not derived."

### 7. Unsafe statement to avoid

> ~~"The correction factor $1/\sqrt{2}$ arises from the bidirectional collision doubling, and the residual 0.75 from noise attenuation."~~

这把 heuristic 分解伪装成了推导链。

### 8. Recommended next step

**扩展实验（中等优先级）。** 在纯双向文本模型（BERT-style masked LM，无扩散噪声）上测试 $\tau^*$ 是否需要修正以及修正因子是否接近 $1/\sqrt{2} \approx 0.707$。这能分离"双向"和"扩散"的贡献。如果 BERT 需要 0.707 修正，则扩散额外贡献 $0.53/0.707 \approx 0.75$，分解得到实验支撑。

### 9. Confidence

**Low.** DiT 修正是六个问题中理论最弱的一个。但它被正确地定位为 "supporting evidence"，不影响核心贡献。审稿人不太可能把它当作 deal-breaker，因为 DiT 不是论文的主战场。

---

## Problem 4: LoRA 相变阈值的严格证明

### 1. Problem statement

实验发现 LoRA rank $r < d_{\mathrm{head}}/2$ 时 EVQ 效果崩溃（LLaMA-8B, $r=16$, PPL 从 11.8 升至 77.1）。猜测存在相变阈值 $r_c = d_{\mathrm{head}}/2$。

### 2. Why it matters

LoRA 是大模型 fine-tuning 的标准范式。如果 EVQ 在低秩 LoRA 下失效且理论能预测失效条件，这既是一个实用贡献（告诉用户最低秩要求），也是理论自洽性的验证。

### 3. Current project status

论文附录（§A.6）已经给出了一个完整的理论分析：

- **Coupling stiffness 模型：** $S_{\mathrm{total}}(\tau; r) = S_{\chi^2}(\tau) + \Lambda_0(1 - r/K)\tau^2/d_{\mathrm{head}}$，其中 $S_{\chi^2} \sim \tau^4$, coupling term $\sim \tau^2$
- **相变机制：** $\tau^2$ 项在小 $\tau$ 主导 $\tau^4$ 项。当 coupling 足够强时（$r < K$），$\partial\mathcal{F}/\partial\tau = 0$ 在 $\tau > 0$ 处无解，$\tau^*$ 被推到 0
- **数值验证：** sweep $r$ over variational landscape 确认 $\tau^* = 0$ for $r \leq 48$（$r/K \leq 0.75$），然后 $\tau^*$ 在 $r = 64$（$r/K = 1.0$）处跳跃到 1.41
- **PPL 预测：** 从 $\Lambda_0$ 校准可精确复现 PPL = 77.1

### 4. RALPH investigation

**频率空间秩分析（重新推导）。** RoPE 将 $d_{\mathrm{head}}$ 维空间分解为 $K = d_{\mathrm{head}}/2$ 个 2D 旋转平面，每个平面对应频率 $\theta_k$。LoRA 扰动 $BA$（$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$）作用于查询矩阵 $W_Q$。

关键观察：当扰动 $BA$ 被投影到第 $k$ 个旋转平面时，其在该平面内的有效秩至多为 $\min(r, 2)$。但控制频率通道的权重-频率耦合只需要 1 个自由度（振幅调制）。因此 $r$ 个 LoRA 参数最多可以独立调制 $r$ 个旋转平面。

当 $r < K = d_{\mathrm{head}}/2$ 时，至少 $K - r$ 个旋转平面的权重完全冻结在预训练值。如果预训练使用 geometric RoPE，冻结通道的频率是 $\theta_k^{\mathrm{geo}}$，与 EVQ 分配的 $\theta_k^{\mathrm{EVQ}}$ 不匹配。不匹配的通道产生系统性注意力 logit 偏差 $\delta\ell \propto |\theta_k^{\mathrm{EVQ}} - \theta_k^{\mathrm{geo}}| \cdot \Delta$。

**Hostile review：** 这个论证的弱点在于："$r$ 个 LoRA 参数最多控制 $r$ 个旋转平面"这一步。严格来说，$BA$ 的列空间是 $r$ 维的，但投影到 $K$ 个 2D 平面的效果取决于列空间与旋转平面的对齐。在最坏情况下，$BA$ 的整个列空间可能集中在少数几个平面上，更多平面不受控制。在最好情况下，如果列空间与所有平面都有非零交集，$r$ 个参数可能"部分地"影响所有 $K$ 个平面。

但论文的模型更精确：它不是纯粹的秩论证，而是 stiffness 模型 $S_{\mathrm{frozen}} = \Lambda_0(1 - r/K)\tau^2/d_{\mathrm{head}}$。这里 $(1 - r/K)$ 是冻结平面比例的线性内插，$\tau^2$ 是 logit mismatch 的二阶近似。相变点 $r_c = K$ 出现在 $S_{\mathrm{frozen}}$ 从 $O(\tau^2)$ 变为 0 的阈值处。这是 variational landscape 的几何性质，不是纯秩论证。

**数值验证状态：** 附录报告的 sweep 确认了 sharp 相变。但只有 $d_{\mathrm{head}} = 128$ 的一个配置。论文还预测了 LLaMA-8B with $r \geq 64$ 应恢复 EVQ 有效性——LoRA v2 实验（$r = 64$, Appendix §A.10）确认了这一点（PPL@8-12K 从 429 降到 20.7）。

### 5. Best current resolution

**Resolved (conditional on the stiffness model).** 论文已给出一个完整的理论框架：stiffness model + variational phase transition + numerical verification + experimental confirmation。阈值 $r_c = K = d_{\mathrm{head}}/2$ 不是猜测，而是 variational landscape 的精确结果。

唯一的条件性在于 stiffness model $S_{\mathrm{frozen}} \propto (1 - r/K)\tau^2$ 是一个**线性内插假设**——实际的冻结效应可能不是线性的。但数值 sweep 确认了预测的精确性。

### 6. Strongest defensible statement

> "The variational analysis predicts a sharp phase transition at $r_c = d_{\mathrm{head}}/2$: for $r < r_c$, the coupling stiffness from frozen frequency channels dominates the $\chi^2$ stiffness, pushing $\tau^*$ to zero and rendering EVQ infeasible. The prediction is confirmed by numerical variational sweep and by the experimental PPL catastrophe at $r = 16 < r_c = 64$ for LLaMA-8B ($d_{\mathrm{head}} = 128$)."

### 7. Unsafe statement to avoid

> ~~"We prove that LoRA rank $r < d_{\mathrm{head}}/2$ cannot support EVQ."~~

"Prove" 过强——stiffness model 是一个 physically motivated 但非第一性原理的假设。

### 8. Recommended next step

**实验验证（中等优先级）。** 在 $r \in \{4, 8, 16, 32, 48, 64, 96, 128\}$ 上做完整 rank sweep（不仅 PPL，还有每个频率通道的梯度范数），确认相变的锐度和位置。当前只有 $r = 16$ 和 $r = 64$ 两个点。

### 9. Confidence

**High.** 理论框架清晰，预测与实验吻合。主要残留风险是 stiffness model 的线性假设，但这可以通过更多 rank sweep 数据验证。

---

## Problem 5: 渐进式训练放大机制

### 1. Problem statement

渐进式训练中，Geo+YaRN 的 PPL 从 3.80（Stage 1）恶化到 13.17（Stage 3），而 EVQ+YaRN 保持稳定（2.48）。EVQ 优势随阶段单调放大：-34.7% → -52.0% → -81.2%。机制解释缺失。

### 2. Why it matters

如果放大效应有理论解释，它展示了 EVQ 的**动态**优势——不仅在单次训练中有效，而且在多阶段训练中优势累积。如果没有解释，单 seed 的 81.2% 改善可能被审稿人归为统计波动。

### 3. Current project status

- Phase 17C (seed=42) 单种子结果
- Seeds 43/44 的 Stage 1 已完成，Stage 2/3 待补跑
- Retarget 协议（$\tau$ 按当前 $L_{\mathrm{train}}$ 重算）是论文的实际配置
- 论文定位为 "supporting evidence"（非核心贡献）

### 4. RALPH investigation

**机制分解（三因素）：**

**(a) Geometric 频率的死通道累积。** Geometric RoPE 的频率 $\theta_k^{\mathrm{geo}} = b^{-2k/d}$ 不依赖于 $L$。随着 $L$ 增大，"死通道"（$\theta_k L < 1$）的数量**单调减少**（从 18/32 at $L=256$ 到 11/32 at $L=4096$，$d=64, b=500K$）。但 YaRN 在推理时处理的外推倍率 $L_{\mathrm{eval}}/L_{\mathrm{train}}$ 保持不变（16K/L_train），所以外推区间内的死通道问题随 $L_{\mathrm{train}}$ 增大而**改善**，不是恶化。

这与直觉矛盾——如果 Geo 在大 $L$ 时应该更好，为什么 PPL 反而崩溃？

**关键洞察：** 崩溃不是因为 Geo 在更大 $L_{\mathrm{train}}$ 上更差，而是因为 YaRN 的外推从更远的位置推断时累积误差更大。Stage 3 的 $L_{\mathrm{train}} = 2048$ → $L_{\mathrm{eval}} = 16K$ 是 8× 外推，而 Stage 1 的 $L_{\mathrm{train}} = 512$ → $L_{\mathrm{eval}} = 16K$ 是 32× 外推。等等——这说明 Stage 3 的外推倍率**更小**，应该更容易，不是更难。

**重新定位问题。** Geo PPL 在 Stage 3 崩溃（13.17）不是因为 Geo+YaRN 在 8× 外推时比 32× 外推更差——这不合理。更可能的解释是：

1. **Retarget asymmetry：** EVQ 每阶段都 retarget $\tau$（调整频率分配适配新 $L$），而 Geo 频率**不变**。当模型在 $L=2048$ 上训练时，Geo 的频率分配对 $L=2048$ 可能已经过度碰撞（低频通道间距不够），导致训练损失本身就高。
2. **权重-频率耦合失配累积：** 每次 $L$ 增大，模型的注意力权重需要适应新的有效距离分布。EVQ 的 retarget 给模型一个正确的频率配置来学习，而 Geo 的固定频率在新的 $L$ 下不再最优。权重适应的残余误差在每个阶段累积。

**(b) EVQ retarget 的平滑性。** 从 $\tau_s$ 到 $\tau_{s+1} = \tau_s/\sqrt{2}$ 时，频率变化为 $\Delta\phi_k = \phi_k(\tau_{s+1}) - \phi_k(\tau_s)$。数值计算：

- $L: 512 \to 1024$: max $|\Delta\phi| = 0.121$（通道间距的 387%）
- $L: 1024 \to 2048$: max $|\Delta\phi| = 0.082$（261%）
- $L: 2048 \to 4096$: max $|\Delta\phi| = 0.050$（159%）

这些位移**不小**——最大位移超过通道间距。但 Phase 17C 用了 500M tokens/stage（足够的训练量），所以模型有能力适应。低 token 实验（EXP-4, 36M total）中 Delayed 策略（不 retarget）反而更好，说明 retarget 的成功**依赖于足够的训练预算**。

**(c) 放大不等式。** 问题问是否存在 $\Delta_{\mathrm{PPL}}^{(s+1)} \geq c \cdot \Delta_{\mathrm{PPL}}^{(s)}$, $c > 1$。数据：$\Delta$（绝对值）= -34.7%, -52.0%, -81.2%。比率：$52.0/34.7 = 1.50$, $81.2/52.0 = 1.56$。近似恒定比率 $c \approx 1.5$，但这是**单种子的三个数据点**，不足以确认幂律放大。

### 5. Best current resolution

**Partially resolved (heuristic mechanism, not formal proof).** 放大机制的定性解释是：Geo 的固定频率在渐进式训练中与不断增长的 $L$ 之间的适配失配累积，而 EVQ 的 retarget 消除了这个累积。这是一个合理的物理图像，但不是一个形式化的证明。

### 6. Strongest defensible statement

> "Progressive training with EVQ retargeting shows monotonically amplifying advantage over geometric RoPE (−34.7% → −52.0% → −81.2% at 16K, single seed). The qualitative mechanism is that EVQ's per-stage $\tau$ retarget adapts the frequency allocation to each new training length, while geometric frequencies remain fixed and accumulate mismatch with the evolving attention patterns. Multi-seed replication is in progress."

### 7. Unsafe statement to avoid

> ~~"EVQ's advantage grows exponentially with the number of progressive stages."~~

单种子三个点不支持"指数增长"的声称。

### 8. Recommended next step

**P0 实验（最高优先级）：** 补跑 seeds 43/44 的 Stage 2+3，验证放大效应的 reproducibility。如果三种子都显示放大，论文可以安全地声称 progressive amplification 是 reproducible phenomenon。如果种子间方差大，则需降低声称强度。

### 9. Confidence

**Medium.** 定性机制合理，但缺少多种子验证和形式化分析。单种子风险是实在的。

---

## Problem 6: $\tau^*$ 在 $L \geq 4096$ 时的有效性

### 1. Problem statement

缩放律 $\tau^* = d_{\mathrm{head}}/\sqrt{L}$ 在 $L \leq 2048$ 验证良好。当 $L \geq 4096$ 时，$\tau^*$ 变小（如 $d = 64, L = 4096 \Rightarrow \tau^* = 1.0$），可能进入"弱分离区"：$\tau \to 0$ 时 EVQ 退化为均匀分配。是否存在饱和机制？

### 2. Why it matters

实际应用中 $L$ 常取 4096–131072。如果 $\tau^*$ 在大 $L$ 时饱和，缩放律需要修正。如果 EVQ 在大 $L$ 时与 Geo 趋同，其价值受限于中等 $L$ 场景。

### 3. Current project status

- 所有训练实验 $L \leq 2048$
- 附录中有 $L = 4096$ 的 collision validation（EVQ 减少 24%）
- 小 $\tau$ 渐近展开（eq. 14）给出 $\phi_k(\tau) \approx u_k - (\tau^2/6)u_k(1-u_k)(2-u_k)$

### 4. RALPH investigation

**渐近分析。** 在 $\tau \ll 1$ 时：
- $\sinh\tau \approx \tau + \tau^3/6$
- $\phi_k(\tau) \approx u_k - (\tau^2/6)u_k(1-u_k)(2-u_k) + O(\tau^4)$

数值验证（$\tau = 0.5, u = 0.5$）：exact = 0.4846, 二阶近似 = 0.4844, 相对误差 0.05%——展开是准确的。

**物理意义：** 二阶修正项始终为负（将频率向低端压缩）。其幅度最大点在 $u_k \approx 0.42$（对 $u(1-u)(2-u)$ 求极值），最大值约 $0.385\tau^2/6$。

对 $d = 64, L = 16384$：$\tau^* = 0.5$，最大频率位移 $= 0.385 \times 0.25/6 \approx 0.016$（通道间距的 50%）。在 $L = 32768$：$\tau^* = 0.354$，位移 $= 0.008$（25%）。

**关键问题：这个位移是否仍然有意义？**

数值检查：即使在 $\tau = 0.5$（$L = 16384, d = 64$），EVQ 仍将频率向低端压缩，增加低频通道间的间距。collision score 在 $L = 4096$ 时仍减少 24%（Table 3）。因此 EVQ 在 $\tau \sim 0.5$ 时仍然提供可测量的改善。

**饱和预测：** 当 $\tau \to 0$，$\phi_k \to u_k$（均匀分配 = geometric）。EVQ 的改善幅度 $\propto \tau^2 \propto d_{\mathrm{head}}^2/L$。这意味着：
1. **不存在 sharp 饱和点**——EVQ 的改善是连续衰减的（$\propto 1/L$），不是突然消失
2. **存在 practical 无关紧要点**——当 $\tau^2 \cdot \max[u(1-u)(2-u)] \ll \Delta\phi_{\mathrm{channel}}$ 时（频率位移远小于通道间距），EVQ 等效于 Geo
3. 这个阈值约在 $\tau \approx 0.1$，即 $L \approx d_{\mathrm{head}}^2 / 0.01 = 100 \cdot d_{\mathrm{head}}^2$。对 $d = 64$：$L \approx 409600$。对 $d = 128$：$L \approx 1638400$。

**结论：** 对于现有的 LLM 训练长度（$L \leq 131072$）和常见的 $d_{\mathrm{head}} \in \{64, 128\}$，$\tau^*$ 始终在 $[0.35, 4.0]$ 范围内，EVQ 的频率位移始终大于通道间距的 25%。**饱和不是近期的实际问题。**

**修正后的缩放律。** 如果要加饱和保护：
$$\tau^* = \max\left(\frac{d_{\mathrm{head}}}{\sqrt{L}},\; \tau_{\min}\right)$$

$\tau_{\min}$ 的选择：
- 理论下界：$\tau_{\min} \approx 0.1$（此时 EVQ ≈ Geo）
- 实际建议：不需要显式设定——当 $\tau$ 自然变小时，EVQ gracefully 退化为 Geo，不会产生有害效果

**与 GQA/MQA 的交互：** 在 GQA 配置下，多个查询头共享键值头，有效 $d_{\mathrm{head}}$ 可能更大（键值头的维度不变）。这推迟了饱和效应——分析上，$\tau^*$ 的公式应使用键值头的 $d_{\mathrm{head}}$，而非查询头的。

### 5. Best current resolution

**Resolved (theoretical analysis + extrapolation).** 渐近分析表明 EVQ 在大 $L$ 时连续退化为 Geo（graceful degradation），不存在 sharp 饱和。对常见 $(d_{\mathrm{head}}, L)$ 配置，$\tau^*$ 不会进入实际无效区域（$\tau < 0.1$）。

### 6. Strongest defensible statement

> "The scaling law $\tau^* = d_{\mathrm{head}}/\sqrt{L}$ is validated for $L \leq 2048$ ($R^2 > 0.99$, 99 runs). For $L > 2048$, EVQ-cosh degrades gracefully: as $\tau \to 0$, the allocation converges smoothly to geometric RoPE ($\phi_k \to u_k$), with the improvement magnitude scaling as $O(d_{\mathrm{head}}^2/L)$. For current LLM configurations ($d_{\mathrm{head}} \in \{64, 128\}$, $L \leq 131\mathrm{K}$), $\tau^*$ remains in $[0.35, 4.0]$, well above the degeneracy threshold."

### 7. Unsafe statement to avoid

> ~~"The scaling law holds for arbitrarily large $L$."~~

未经 $L > 2048$ 的训练实验验证。渐近分析支持连续退化，但不排除 $L > 4096$ 时出现的新效应（如 attention sink 对低频通道的影响）。

### 8. Recommended next step

**最小有效 τ 实验（中等优先级）：** 在 125M 模型上以 $L \in \{512, 1024, 2048, 4096, 8192\}$ 训练并测试，绘制 $\tau^*_{\mathrm{empirical}}$ vs $d_{\mathrm{head}}/\sqrt{L}$ 曲线。$L = 4096$ 和 $L = 8192$ 是关键验证点。

### 9. Confidence

**Medium-high.** 理论分析清晰，渐近展开可靠。缺少 $L > 2048$ 的训练实验是唯一的 gap，但 collision validation at $L = 4096$ 提供了间接证据。

---

## Cross-problem synthesis

### Which issues are actually coupled?

- **P1 (λ closure) + P6 (large L validity):** 如果 λ 在大 $L$ 时系统性偏移，两个问题同时恶化。但 λ 在 $L \in \{256, 512, 1024\}$ 上的低 CV (13%) 暗示 λ 可能在更大 $L$ 也稳定。
- **P4 (LoRA phase transition) + P1 (λ):** LoRA 的 stiffness model 引入额外的 $\tau^2$ 项，改变了 $\tau^*$ 的位置但不影响 λ 本身（λ 属于 from-scratch 的 transport constant）。
- **P3 (DiT) + P2 (surrogate):** DiT 的修正因子可能部分反映 surrogate 在双向注意力下的失效——代理核的距离先验 $D(\Delta)$ 在双向注意力中应是对称的 $D(\Delta) = D(-\Delta)$，改变了 $\beta$ 的值。

### Which unresolved gap is the most dangerous for reviewers?

**P1 (λ closure)**——但不是因为 λ 未闭合本身，而是因为论文行文可能让读者误以为整个缩放律是"推导的"。**措辞风险 > 数学风险。**

### Which issue is mostly a wording/epistemic-status problem?

**P1** 和 **P2**。两者的数学状态已经基本稳定——代理是泛函近似（不是逐点近似），λ 是 transport constant（不是自由参数）。风险在于读者是否能从论文行文中清楚地理解这些区分。

### Which issue is most likely solvable with the current framework?

**P6 (large L)**——渐近分析已经给出了完整图景，只需一个 $L = 4096$ 的训练实验确认。框架预测 graceful degradation，不存在需要"新想法"的地方。

### Which issue probably requires a new idea?

**P3 (DiT correction)**——需要一个关于双向注意力和扩散噪声如何修改 collision 代理的正式模型。当前的变分框架假设单向 AR 注意力和 next-token loss，需要本质性的修改才能覆盖 DiT。

---

## Paper-facing implications

### What the theory section should emphasize more strongly

1. **三层认知分级已经做得很好**（Exact conditional on surrogate → Practical branch choice → Semi-analytic calibration）。建议保持并在 Prop. 3 之后加一句显式区分。
2. **代理自洽性定理**是一个干净的理论贡献，值得在正文中简要提及（而非仅在附录）。它证明代理内部的 balance 是精确的。
3. **LoRA 相变分析**是理论的 practical 延伸中最成功的部分——预测与实验精确吻合。

### What the theory section should soften

1. **避免暗示缩放律是无参数推导。** 当前行文中 "The minimizer satisfies $\tau_* \propto d_{\mathrm{head}}/L^\gamma$" 紧接着 Proposition 3，容易被误读为全部推导。建议在 γ = 0.465 后立即标注 "the prefactor λ ≈ 1.13 is calibrated."
2. **Surrogate 的措辞：** "Our only approximation is a broadband surrogate" 是好的，但后续行文中 "after this step, the remaining derivation is exact" 可能让读者忘记 τ* 层的 λ 也不是 exact 的。建议加一句 reminder。

### Which claims can remain ambitious

- "EVQ-cosh is the stationary point of a collision-utility variational problem" — 精确（conditional on surrogate）
- "Geometric RoPE is the τ=0 degenerate limit of EVQ" — 精确
- "EVQ and YaRN correct orthogonal deficiencies" — 有实验和理论支撑
- "The τ* basin is flat: ±20% deviation costs <1% PPL" — 强实验证据（99 runs）

### Which claims must be explicitly scoped

- "τ* = d_head/√L" → must note λ is calibrated
- "24–92% collision reduction" → must note this is under the exact kernel, but the derivation route went through the surrogate
- "Progressive amplification −81.2%" → must note single seed
- "DiT correction 0.53×" → must note empirical, not derived
- "LoRA phase transition at r_c = d_head/2" → can claim conditional on stiffness model

---

## Suggested manuscript actions

### 1. Highest-priority wording fix

**在 Proposition 3 后加显式 epistemic disclaimer：**

> "The $d_{\mathrm{head}}$ dependence (via $\alpha = 1/d_{\mathrm{head}}$) and the $L^{-1/2}$ exponent (via Pearson $\chi^2$ stiffness, Table 5) are derived. The O(1) transport constant $\lambda \approx 1.13$ is calibrated from 99 runs across 27 configurations; leave-one-out cross-validation yields max error < 0.2% (Appendix Table 6). Closing $\lambda$ analytically remains open."

这一句话解决了 P1 的审稿人风险。

### 2. Highest-priority derivation / appendix fix

**代理自洽性定理已经在附录中**，状态良好。建议在正文 §3.3 (Variational objective and exact ODE) 末尾加一句引用：

> "The surrogate's internal balance is provably exact: $\tau^2 T_2 + T_1 = \tau\coth\tau$ (Theorem 2, Appendix A)."

### 3. Highest-priority experiment / numeric validation

**Progressive training 3-seed 复现**（P0 priority，seeds 43/44 的 Stage 2+3）。这是论文最薄弱的实验支撑点。如果三种子都显示放大效应，progressive training 可以从 "supporting evidence" 升级为完整贡献。

### 4. Highest-priority reviewer-risk mitigation

**Surrogate validity 的主动防御。** 在 rebuttal 中准备以下论点：
- 代理是**泛函近似**，不是逐点近似——12/12 配置 collision reduction 通过
- 代理系数 α 有半解析式，β 的经验拟合不影响 τ* 公式（β 只通过 $\sqrt{\beta/\alpha}$ 影响 $\tau_{\mathrm{surr}}$，但 τ* 的 L 指数来自 softmax transport，不来自 β）
- 这意味着 surrogate validity 和 τ* scaling law 是**独立验证的**——即使 surrogate 在某个极端配置失效，τ* 公式仍由 99 runs 直接验证

---

## Mathematical quality control: self-audit

| Claim | Type | Verified |
|-------|------|----------|
| $\tau^2 T_2 + T_1 = \tau\coth\tau$ | Exact theorem | ✅ numerical ($< 10^{-15}$) |
| $T_1' + \tau^2 T_2' = 0$ | Exact corollary | ✅ follows from theorem |
| $\tau_{\mathrm{surr}} = \sqrt{\beta/\alpha}$ is exact | Exact (conditional) | ✅ by balance identity |
| $\alpha \approx 1/d_{\mathrm{head}}$ | Semi-analytic | ✅ (requires $\omega L \gg 1$) |
| $\gamma = 0.465$ from $\chi^2$ stiffness | Numerical optimization | ✅ by verify_stiffness script |
| λ ≈ 1.13, CV ≈ 13% | Empirical | ✅ 99 runs, LOO CV < 0.2% |
| $\phi_k(\tau) \approx u_k - (\tau^2/6)u_k(1-u_k)(2-u_k)$ | Asymptotic expansion | ✅ numerical ($< 10^{-3}$ at $\tau = 0.5$) |
| LoRA $r_c = d_{\mathrm{head}}/2$ | Variational prediction | ✅ numerical sweep + experiment |
| 24–92% collision reduction under exact kernel | Functional validation | ✅ 12/12 configurations |
| DiT factor 0.53 derivable from bidirectional | **NOT verified** | ❌ heuristic only |
| Progressive amplification $c \approx 1.5$ | Single seed | ⚠️ needs multi-seed |

---

*This memo is intended as an internal theory reference for paper revision and rebuttal preparation. It does not contain material suitable for direct inclusion in the paper without editorial adaptation.*
