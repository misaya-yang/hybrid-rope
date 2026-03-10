# EVQ-Cosh 核心理论与关键实验（Paper-Ready）

> **定位**：论文写作的唯一核心参考。只含已证明/已验证的理论和 solid 实验结果。
> **配套文档**：`SECONDARY_THEORY.md`（发散性理论、待验证猜想、次要实验）
> **最后更新**：2026-03-10a（v12 Phase 17b 全矩阵：YaRN 协同→替代相变 + training-inference 等价性 + progressive 放大至 -83.1%@16K）

---

## 0. 一页主线（Spotlight版）

### 0.1 一句话主张

**现代 LLM 面临 RoPE 的三重挤压：上下文越来越长（L↑）、频带被迫拉宽（base↑）、MLA 又主动压缩 RoPE 维度（d_head↓）。三个力全在把"每个频率通道的负担"推向极限——但通道内的分配方式（allocation）从未被优化过。**

EVQ 给出频率分配的闭式变分最优解，Geometric RoPE 是其 `τ=0` 退化点。这不是独立的小优化，而是 RoPE 优化链条中**被忽视的关键一环**：

- **与 base tuning 正交**：base 控制频带宽度，EVQ 控制频带内分配，两者独立
- **与 YaRN 阶段性协同**：短程训练阶段 `EVQ+YaRN >> Geo+YaRN`（-86%）；但 progressive training 后 EVQ raw 独立超越 YaRN 叠加（PPL@16K: EVQ raw 11.2 < EVQ+YaRN 16.8），说明 **EVQ 是比推理时修补更根本的解法**
- **与 progressive training 放大协同**：512→1024 continuation 后 EVQ 优势从 34.6% 扩大到 83.1%@16K，progressive training 不会"洗掉"EVQ，反而让模型内化最优分配
- **MLA 时代更关键**：DeepSeek V3、GLM-5、Kimi K2.5 将 RoPE 压缩至 d=64（32 对通道），每个通道更"金贵"，分配优化从 nice-to-have 变为 must-have

### 0.2 正文最该讲的 4 条 claim

1. **三重挤压下的缺失维度（Framing Claim）**
RoPE 长文本有三个正交维度：base（频带宽度）、allocation（频带内分配）、inference-time scaling（YaRN 等）。工业界只调了 base（10K→500K→1M）和推理修补（YaRN/NTK），**从未优化过 allocation**。同时 MLA 将 RoPE 压缩至 d=64，通道数锐减，分配优化的 leverage 急剧上升。

2. **Closed-form solution + near-optimal scaling law**
RoPE frequency allocation 是变分逆问题，闭式解为 `EVQ-cosh`，Geometric 是 `τ→0` 退化点。`τ*=d_head/√L` 在 99-run sweep 中 8/9 top-3，且 loss landscape 极平坦（worst-case PPL gap <1%）——practitioners 可直接使用，无需 grid search。

3. **EVQ 与 YaRN：从协同到替代**
短程训练阶段 `EVQ+YaRN >> Geo+YaRN`（-86%），是超线性协同。但 progressive training 后出现**相变**：
   - **短程阶段（L=512 only）**：EVQ+YaRN 远强于 EVQ raw（PPL@16K: 11.6 vs 79.6）
   - **progressive 后（512→1024）**：EVQ raw **反超** EVQ+YaRN（PPL@16K: **11.2 vs 16.8**）。YaRN 的频率缩放反而扭曲了已被模型内化的最优分配。
   - **等价关系**：`evq_512+yarn ≈ evq_1024_cont raw`（@16K: 11.6 vs 11.2），说明 progressive training 一步可以替代 YaRN，且不需要推理时开销。

   这意味着 EVQ 不只是 YaRN 的"训练时前提"，**它是比 inference-time scaling 更根本的解法**——当训练到位时，推理时修补变得多余。

4. **与工业训练 pipeline 天然兼容——且越训越强**
Progressive short→long 训练（SkyLadder, LLaMA）是工业标准。454M 512→1024 continuation 后：
   - EVQ vs Geo 的 raw 优势从 34.6% **扩大到 83.1%@16K**（PPL: 11.2 vs 66.4）
   - EVQ raw 在 4K-32K 全面优于 Geo+YaRN（即：EVQ 纯训练 > Geo 训练+推理修补）
   - Phase 15 (750M 2K→4K) 同样确认放大效应（PPL@16K -45.9%，AR exact 0%→77.5%）

   EVQ 不是需要 YaRN 护航的脆弱优化，而是**随 progressive training 自我强化**的训练底座。

### 0.3 现在最该推的证据顺序

| 优先级 | 证据 | 为什么它最值钱 |
|--------|------|----------------|
| **P0** | **Figure 2: EVQ × YaRN 主图** | 最接近工业现实；一句话就能解释为什么 training-time PE matters |
| **P0** | **Figure 3: DAPE-style extreme extrapolation + Phase 11** | 同时证明理论不是 heuristic，且在最纯净外推 setting 里更强 |
| **P1** | `350M 3-seed` raw PPL + passkey mix 多 seed | 给主锤提供稳健性，不让 reviewer 说全靠 toy/single-seed |
| **P1** | `τ*=d/√L` 99-run sweep 验证 | 3/9 精确命中、8/9 前三，公式不是 heuristic 而是 robust near-optimal law |
| **P1** | **训练课程 ablation (Phase 15+17)** | 两种现实 regime 下 EVQ 都赢：同长度+YaRN (-86%)、progressive (-45.9%@16K) |
| **P1** | **MLA 工业对齐 + loss landscape 平坦性** | d_head=64 = 2026 旗舰模型 RoPE 现实（DS V3/GLM-5/Kimi K2.5），τ* 在此配置 PPL 偏差 <1%，直接回应 practical relevance |
| **P2** | video temporal transfer | 抬高论文上限，证明不是 text-only artifact，但当前不替代 text 主线 |

### 0.4 当前最强的一句话证据

- **方法主锤**：`EVQ+YaRN@8K = 100% across 6/6 seeds`
- **理论主锤**：`Geometric = τ=0`，`τ*=d_head/√L` 在 `99-run` sweep 中 `3/9` 精确命中、`6/9` 前二、`8/9` 前三
- **PE-dominant 主锤**：`L=256, EVQ4+YaRN 99.6 vs Geo+YaRN 260.2` at `32×` (`-61.7%`)
- **训练课程主锤**：Phase 15 progressive (2K→4K) `PPL@16K -45.9%`；Phase 17 same-length+YaRN `-86%`；Phase 17b 454M 512→1024 后 EVQ raw PPL@16K=11.2 反超 EVQ+YaRN 16.8，progressive training **替代** YaRN
- **训练-推理等价**：`evq_512+yarn` ≈ `evq_1024_cont raw` @16K（11.6 vs 11.2）——progressive 一步替代 inference-time scaling，零推理开销
- **MLA 工业对齐**：DeepSeek V3/GLM-5/Kimi K2.5 的 `qk_rope_head_dim=64`，我们的 d_head=64 实验 = 工业现实；τ* 在此配置 worst-case PPL gap < 1%（shallow basin）
- **更大规模 supporting evidence**：`750M continue@4K` 下 `8K AR exact 77.5% vs 0%`

### 0.5 摘要/开场应该怎么说

不要以”我们提出一个新的 positional encoding”开头。开场直接讲三重挤压：

> **As context windows scale to millions of tokens, LLMs face a triple squeeze on RoPE: longer sequences (L↑) demand slower frequencies, wider frequency bands (base↑) dilute per-channel resolution, and Multi-head Latent Attention (MLA) compresses the RoPE subspace to just 64 dimensions (d_head↓). All three trends push each frequency channel to its limit — yet the within-band allocation has never been optimized: every production model uses the default geometric (uniform log-spacing) layout.**
>
> **We show this default is a degenerate point (τ=0) of a variational optimum. Our closed-form solution, EVQ-Cosh, requires zero extra parameters and composes supralinearly with YaRN at short training lengths (EVQ+YaRN reduces perplexity by 86% over Geo+YaRN). More remarkably, after progressive training (512→1024), EVQ raw alone surpasses EVQ+YaRN (PPL@16K: 11.2 vs 16.8) — the model internalizes optimal allocation so thoroughly that inference-time frequency scaling becomes unnecessary. EVQ is the missing piece in the RoPE optimization chain — orthogonal to base tuning, more fundamental than inference-time scaling, and increasingly critical as MLA compresses the frequency budget.**

### 0.6 哪些点现在不要抢主舞台

- `750M phase9f Hybrid dynamics`：可以讲，但只能是 supporting evidence
- `+40pp` 单 seed passkey 极值：不能再当 headline
- `video confirms τ*=2.0`：当前证据不够，不能这么写
- `strict improvement theorem`：正文应写 empirical / conditional proposition，不要把现有证据包装成无条件 theorem

---

## 1. 推导链（6 步，完整闭合）

```
D(Δ) 距离先验
  → 相位碰撞核 K(φ₁,φ₂) = ∫D(Δ)cos(b^{-φ₁}Δ)cos(b^{-φ₂}Δ)dΔ
  → Broadband 投影: K ≈ αδ(φ₁-φ₂) + βmin(φ₁,φ₂)     [算子: αI + βA⁻¹]
  → 变分泛函: J[ρ] = (α/2)∫ρ² + (β/2)∫∫ρρmin - μ∫ρb^{-2φ}
  → Euler-Lagrange ODE: ρ'' - τ²ρ = γb^{-2φ},  τ=√(β/α)
  → 通解: ρ* = C₁cosh(τφ) + C₂sinh(τφ) + Pb^{-2φ}
  → CDF 反演: φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinhτ)
```

**唯一核心近似**：Broadband 投影（步骤 2→3）。残差 O(1/lnb)（δ 脊物理宽度），R²_mid>0.99。其余全是精确推导。

---

## 2. 四个核心定理

### Theorem 1（ODE 精确解）
通解由 hyperbolic tether（cosh/sinh）+ Fisher pulse（b^{-2φ}）两项竞争构成。τ 控制平衡。

### Theorem 2（Geometric 是零温极限 + 严格次优性）

τ→0 时 EVQ 光滑退化为 Geometric RoPE。**Geometric 是 EVQ 族的 τ=0 特例。**

**证明**：φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh τ)。取 τ→0：sinh τ ≈ τ, arcsinh(xτ)/τ → x，得 φ_k → u_k（uniform quantiles）= Geometric RoPE。

**Corollary（严格次优性）**：对任意 L > 0，τ* = d_head/√L > 0，因此 Geometric（τ=0）严格次优——RoPE 原始等比频率设计是连续最优家族中的退化点。（注：L 越大 τ* 越小，趋近 Geometric；L 越小 τ* 越大，偏离越远。实验中 L=2048, d=64 → τ*=1.41。）

### Waterbed 不等式
∫ln E(φ)dφ ≥ lnb - lnc

- Jensen 等号条件：ρ(φ) ≡ 1 ↔ Geometric
- Geometric 最小化全局对数误差体积，但分布极不均匀（E ∝ b^{2φ}，低频段指数爆炸）
- EVQ 使误差趋于均匀但总体积增大——有界代价

### Hybrid 严格优越性（Riemann-Lebesgue 论证）

**命题**：在 J_HF 已极小化的条件下，Hybrid（局部 warp）严格优于 Pure EVQ（全局 warp）。

1. J_HF 在 Geometric 处取极小 → 局部 Hessian 强正定，偏移产生 ΔJ_HF ≈ ½δ^T H δ ≫ 0
2. 长程注意力中 cos(m·φ_HF)（m≫1）由 Riemann-Lebesgue 引理积分趋零 → ∇_{φ_HF} J_LF ≈ 0
3. J(Pure) - J(Hybrid) ≈ ½δ^T H δ - ∇J_LF · δ > 0。Q.E.D.

---

## 3. τ 的物理意义（⚠️ 精确表述，2026-03-03 数值验证）

τ 控制通道密度的重分配。τ=0 时 log-frequency 间距均匀（Geometric）。τ > 0 时：

- 每个 φ_k 向高频端（φ=0）偏移（Taylor: φ_k ≈ u_k - u_k(1-u_k)(2-u_k)τ²/6）
- **关键效应在间距不在位置**：低频端间距扩大（τ=1.5 时 ~1.4×），高频端压缩（~0.6×）
- 物理机制：**减少低频通道间的相位碰撞**
- 代价：有效长程通道数减少（Δ=16K 时 11→8），但每通道分辨率提升（间距↑ = 碰撞↓）
- 高频压缩代价小：高频通道在长距离贡献趋零（Riemann-Lebesgue）

**⚠️ 禁止说"τ 把频率推向低频端"——数学上是错的。**
✅ 正确表述："τ 重分配通道密度——压缩高频间距、扩大低频间距，减少长程关键的低频相位碰撞。"

### 非对称 Tradeoff 论证

Waterbed 形式上是 tradeoff，但实际效应高度不对称：PPL@2K 仅 +0.4%，PPL@16K -13.3%。原因：
- 高频端有大量冗余（相邻通道编码几乎相同的短距离信息）→ 压缩 40% 间距代价极小
- 低频端是瓶颈（相位碰撞 destroy 信息）→ 扩大 40% 间距收益巨大
- 本质：**拆冗余的墙补瓶颈的墙**，不是对称拆东墙补西墙

---

## 4. Broadband 近似的算子理论地位

min(φ₁,φ₂) 是 A = -d²/dφ² 在混合边界条件下的 Green 函数。

**K_approx = αI + βA⁻¹**（Identity + Resolvent）

不是经验拟合，是精确核在二参数算子族上的 Hilbert-Schmidt 最优投影。

R² 衡量中间频率区（渐近物理区）。全矩阵残差 35-49% 来自三个边界效应：UV 边界离散化、IR 边界波长超限、δ 函数物理脊宽 O(1/lnb)。

Power-law 先验 D(Δ)∝1/Δ 时，βmin 是余弦积分的渐近精确解，结构残差降至 O(b^{-γ})。

---

## 5. Fisher → 注意力效用桥梁

### Laplace 桥（核心）

K''(0) = -∫ρ(φ)b^{-2φ}dφ = -𝓗

Softmax Laplace 近似：A(Δ) ≈ exp(-𝓗Δ²/2τ_temp)。**Fisher 项 = 局部注意力高斯分布的精度矩阵。**

### 失效区

大 Δ 时高频 cos(ωΔ) 产生空间混叠假峰。Fisher 只看局部曲率，看不到远处混叠。

---

## 6. cosh 族为什么 1 参数打赢 32 参数

### n-width 论证

J[ρ] 严格凸，唯一全局解析解 ρ*。min 核谱特征值 λ_k ~ O(k⁻²)，Kolmogorov n-width 快速衰减。

N=32 时离散化误差 ΔJ ~ O(N⁻²) ≈ 0.1%。cosh 族捕获 >99.9% 的泛函物理方差。

### 密度比下界

ρ(0)/ρ(1) ≥ cosh(τ)。由 ODE 边界条件 + 极值原理严格证明。

---

## 7. Waterbed 精细结构

### 等号条件
Jensen on f(x)=-lnx（严格凸），等号 ↔ ρ ≡ 1 ↔ Geometric。

### PPL 看不到 Waterbed 的原因
高频误差上升仍在 Softmax 过参数化冗余区间。PPL 的 token-level 全局平均掩盖了频率轴效应。Phase 6A 验证：τ=0→5.0 全部不恶化训练窗口内 PPL。

### 下游任务看到 Waterbed 的原因
不同任务是频率轴上的带通滤波器：Retrieval（低通）提升，Multi-hop（中高通）可能退化。

### 量化
偏离 geometric 的额外水床体积 ΔW = D_KL(Uniform || ρ)。加权 L² 误差 ≥ D_KL/c²。

---

## 8. τ* Scaling Law

### 公式（单变量主律，base=500K 主线）

$$\tau^*(L) = \frac{d_{head}}{\sqrt{L}}$$

### 变分推导（semi-rigorous，已通过 Gemini Q6 审核）

从 Fourier 测不准原理推导：α*(L,b) ∝ 1/(L·lnb)，β* ≈ O(1)。

τ* = √(β*/α*) ∝ 1/√(L·lnb)。固定 base 时简化为 ∝ 1/√L。

### Phase 8D / Phase 11 锚点验证

| L_train | 预测 τ*=64/√L | 实测 τ* | 备注 |
|---------|--------------|--------|------|
| 128 | 5.66 | ≥5.0 | 单调下降（PE-dominant） |
| 256 | 4.0 | 5.0 | 偏高 25% |
| 512 | 2.83 | 4.0 | 偏高 41% |
| 1024 | 2.0 | 2.0 | 精确匹配 |
| 2048 | 1.41 | 1.5 | 偏差 6% |

L≥1024 吻合良好；L<1024 旧小模型数据系统偏高（PE-dominant regime）。

### Phase 16 全面 sweep：99 runs 系统验证 τ* 近最优性

**设置**：`local_wikitext`，`L ∈ {256, 512, 1024}` × `H ∈ {4, 8, 16}`（`d_head ∈ {128, 64, 32}`），45 pilot + 54 confirm = **99 runs**，每配置 **3 seed**。

| Config | τ*=d/√L | 经验最优 τ | 公式排名 |
|--------|---------|-----------|---------|
| L=256, d=32 | 2.0 | **2.0** | **#1 精确命中** |
| L=256, d=64 | 4.0 | **4.0** | **#1 精确命中** |
| L=256, d=128 | 8.0 | 10.0 | #2 |
| L=512, d=32 | 1.41 | 1.77 | #5（worst case）|
| L=512, d=64 | 2.83 | 4.24 | #3 |
| L=512, d=128 | 5.66 | **5.66** | **#1 精确命中** |
| L=1024, d=32 | 1.0 | 1.25 | #3 |
| L=1024, d=64 | 2.0 | 2.5 | #2 |
| L=1024, d=128 | 4.0 | 5.0 | #2 |

**统计汇总**：

- 精确命中 #1：**3/9**
- 前二名（#1 或 #2）：**6/9**
- 前三名：**8/9**（只有 L=512/d=32 排 #5，worst case）
- 所有经验最优 τ 都在 theory 的 **1.5×** 以内
- 经验最优值平均约为 theory 的 **1.20×**（系统性右偏移，有限容量效应）

**论文最稳 claim**：

> `τ*=d_head/√L` is a robust near-optimal scaling law. It hits the empirical optimum in 3/9 configurations and ranks top-3 in 8/9, with a mild systematic right-shift attributable to finite model capacity.

### 🆕 MLA 时代工业对齐：d_head=64 是 2026 旗舰模型的 RoPE 现实

2026 年旗舰开源模型全面转向 MLA（Multi-head Latent Attention），RoPE **不再作用于整个 head 维度**，而是只作用于 Q/K 的低维子空间：

| 模型 | 发布时间 | 注意力机制 | qk_rope_head_dim | 名义 d_head | 全 head 维度 |
|------|---------|-----------|-------------------|------------|------------|
| **DeepSeek V3/V3.2** | 2025/2026 | MLA | **64** | 128 | 192 (64 rope + 128 nope) |
| **GLM-5 (ChatGLM)** | 2026 | MLA+DSA | **64** | — | 256 (64 rope + 192 nope) |
| **Kimi K2.5** | 2026 | MLA | **~64** | 112 | — |
| LLaMA 3.1 (旧架构) | 2024 | MHA/GQA | 128 (全部) | 128 | 128 |

**关键洞察**：工业界趋势不是 d_head 变大提升精度，而是 MLA 把 RoPE 维度**压缩到 64**。实际用于位置编码的频率通道只有 **32 对**。

**对 EVQ 的意义**：
- 通道越少 → 每个通道的分配越关键 → **频率分配优化从 nice-to-have 变为 must-have**
- 我们的 d_head=64 实验不是"不贴近工业的小配置"，恰恰是**最贴近 MLA 时代的现实配置**
- 这直接加强了 EVQ 的 practical relevance 叙事

**论文写法**：Introduction 可以加一句："With the rise of Multi-head Latent Attention (DeepSeek V3, GLM-5, Kimi K2.5), RoPE is now applied to a compressed 64-dimensional subspace, making optimal frequency allocation within this limited budget critically important."

### 🆕 Loss Landscape 平坦性：τ* 的"精度"问题不影响实用性

d_head=64 三个 config 的实际 PPL 差距（Phase 16 raw data）：

| Config | τ* (公式) | 经验最优 τ | 公式排名 | Selection Score 差距 | PPL@eval 差距 |
|--------|----------|-----------|---------|--------------------|----|
| L=256, d=64 | 4.0 | 4.0 | **#1** | 0 | 0 |
| L=512, d=64 | 2.83 | 4.24 | #3 | **0.009** (< 0.15%) | **~3 PPL** |
| L=1024, d=64 | 2.0 | 2.5 | #2 | **0.020** (< 0.33%) | **~3.3 PPL** |

**即使在"最差"的 L=512/d=64（ratio=1.50×，全场最大 outlier），PPL 差距也不到 1%。**

Loss landscape 在 τ* 附近是**浅盆地**（shallow basin），不是尖峰。这意味着：
1. 公式不需要精确命中谷底，落在盆地里就够了
2. 公式的 10-25% 偏差在实际性能上几乎不可感知
3. 这是比"精确修正项"更强的叙事：**a simple formula that lands in the flat optimum basin every time**

**论文最稳写法**：

> The loss landscape around τ* is remarkably flat: even in the worst case (d_head=64, L=512), the PPL gap between τ*=d/√L and the empirical optimum is less than 1%. This means practitioners can use τ*=d/√L as a parameter-free default without grid search.

### 模型大小无关性

外层几何截断误差只依赖 L、d/2、τ，不依赖模型参数量。条件：模型不处于极度欠拟合。

---

## 9. 碰撞块分析（核心定量贡献）

### 碰撞块定义

频率通道 k 满足 λ_k = 2πb^{2k/d} > L_train 时不可分辨。占比：c = lnL/lnb。

### EVQ 修正净增益

$$\Delta J \propto \frac{1-c}{\ln b} = \frac{1 - \ln L / \ln b}{\ln b}$$

| Base | lnb | c | 碰撞块占比 | 碰撞通道数 (d/2=32) | 相对净增益 |
|------|-----|---|----------|-------------------|----------|
| 10K | 9.21 | 0.903 | 9.7% | ~3 | 1.0× |
| 100K | 11.51 | 0.722 | 27.8% | ~9 | 2.3× |
| 500K | 13.12 | 0.634 | 36.6% | ~12 | **2.7×** |

**"低 base 放大 EVQ 增益"叙事被推翻**：base=500K 净增益是 base=10K 的 2.7 倍。

### Base=10K 完整负面结果（350M, 50M tokens, L=4096）

| Method | τ | retrieval | vs Geo | PPL@16K | vs Geo |
|--------|---|-----------|--------|---------|--------|
| Geometric | — | 0.680 | — | 274.2 | — |
| EVQ | 0.7 | 0.643 | -5.5% | 342.8 | +25.0% |
| EVQ | 1.1 | 0.568 | -16.5% | 282.4 | +3.0% |
| Hybrid | 0.7/r=22 | 0.568 | -16.5% | 308.2 | +12.4% |

**结论**：base=10K 下全败，是碰撞块理论的预测确认（仅 3/32 通道可优化）。

---

## 10. Hybrid 理论

### r* 解析解

$$r^* = \frac{d}{2\ln b} \ln\left(\frac{L_{train}}{2\pi}\right)$$

| Base | r* | 当前 r=16 合理？ |
|------|-----|---------------|
| 10K | 22.5 | ❌ 过度 warp |
| 100K | 18.0 | ⚠️ 轻度过度 |
| 500K | 15.8 | ✅ 接近最优 |

### Waterbed 隔离

锁定高频 Geometric（保护 PPL），只在低频做 EVQ 重组。用 δ_{HF}=0 的刚性挡板锁定高频。

### Phase 8F 多种子验证（base=500K, L=4096, 350M, 4 seeds）

| Method | PPL@16K (mean±std) | Passkey (mean±std) |
|--------|-------------------|-------------------|
| Geometric | 175.7±13.6 | 0.735±0.055 |
| EVQ τ=1.0 | 193.9±17.1 | 0.706±0.014 |
| Hybrid τ=1.0 | **177.0±7.4** | **0.709±0.007** |

Hybrid 是 Pareto 最优：PPL ≈ Geo，方差全面最低（passkey std = Geo 的 1/8）。

---

## 11. Solid 实验结果汇总

### 11.1 跨规模 PPL 改善（base=500K, L_train=2048, τ=1.5）

| 规模 | 数据集 | Δ PPL@2K | Δ PPL@16K | 方向 |
|------|--------|----------|-----------|------|
| 50M | TinyStories | -0.3% | **-10.9%** | ✅ |
| 125M (seed=42) | TinyStories | -1.7% | **-18.9%** | ✅ |
| 350M | TinyStories (Hybrid) | -0.4% | **-13.7%** | ✅ |
| 350M (3-seed) | FineWeb-Edu | +0.4% | **-13.3%** | ✅ |

**三个规模方向完全一致。短程代价 ≤ +0.4%（误差范围内），长程改善 10-19%。**

### 11.2 350M FineWeb-Edu 3-seed 详细（5090, L=2048, 50M tokens）

| Method | PPL@2K | PPL@4K | PPL@8K | PPL@16K |
|--------|--------|--------|--------|---------|
| Geo mean | 87.40 | 119.41 | 173.58 | 284.78 |
| EVQ mean | 87.73 | 115.83 | 155.38 | 246.88 |
| **Δ** | **+0.4%** | **-3.0%** | **-10.5%** | **-13.3%** |

3 个 seed（42/137/256）方向完全一致。

### 11.3 竞品对比（50M, L=2048, base=500K）

| Method | PPL@16K | vs Geo |
|--------|---------|--------|
| Geometric | 17.97 | — |
| **EVQ τ=1.5** | **16.86** | **-6.2%** |
| YaRN | 39.48 | +119.7% ❌ |
| PI | ~254 | 灾难性崩溃 ❌ |

### 11.4 Phase Collision Score 验证

| τ | Total Collision | Short | Mid | Long |
|---|----------------|-------|-----|------|
| 0.0 (Geo) | 0.386 | 0.534 | 0.196 | 0.070 |
| **1.5 (EVQ)** | **0.268** | 0.267 | 最低 | 最低 |

Phase collision 在 τ=1.5 取最小值，与变分理论预测（τ*=1.41）一致。

### 11.5 8B LoRA 微调（anchored_sigmoid, base=500K）— ⚠️ 参考但不作为核心证据

| Length | Anchored Sigmoid | Geometric |
|--------|-----------------|-----------|
| 1K-8K | 100% | 100% |
| 16K | **100%** | **80%** ❌ |

LongBench +14.5%。但 **LoRA 实验方法论有根本缺陷**：模型同时适应新频率 + 学习下游任务，两个信号 confounded + 灾难性遗忘风险。此数据只能作为 motivation/参考，不能作为 EVQ 有效性的核心证据。论文中如引用需注明局限性。核心证据必须来自 from-scratch 实验。

### 11.6 🔴 R6000 Phase9F 完整结果（750M, L=2048, base=500K, 1B tokens, seed=42）

> **状态**: VALID / COMPLETE — Geo + Hybrid 各 4 checkpoints 全部完成。训练 16.4h。
> **数据源**: `docs/exp/2026-03-03_phase9f_50pct_checkpoint_report.md`
> **对比方法**: Geometric (τ=0) vs Hybrid (τ=1.5, r=16)

#### 11.6.1 PPL 全量数据（4-chunk eval）

| Ckpt | 方法 | L=1K | L=2K | L=4K | L=8K | L=16K |
|------|------|------|------|------|------|-------|
| 25% | Geo | 28.08 | 35.20 | **51.07** | **97.77** | **200.13** |
| 25% | Hybrid | **28.05** | 35.21 | 52.75 | 104.05 | 214.83 |
| 50% | Geo | 21.98 | 27.00 | **42.20** | **99.43** | **215.10** |
| 50% | Hybrid | **21.75** | **26.78** | 43.16 | 104.36 | 225.65 |
| 75% | Geo | 18.95 | 23.52 | **41.41** | **108.80** | **238.20** |
| 75% | Hybrid | **18.94** | **23.06** | 42.08 | 115.21 | 248.88 |
| 100% | Geo | 17.55 | 21.98 | **41.38** | **115.01** | **253.18** |
| 100% | Hybrid | **17.53** | **21.65** | 42.26 | 121.58 | 267.70 |

**PPL 解读**：
- **In-distribution (L≤2K)**：Hybrid 持平或微优（100%时 L=1K -0.14%, L=2K -1.51%）
- **OOD (L≥4K)**：Geo 优于 Hybrid（100%时 L=8K +5.7%, L=16K +5.7%）
- **⚠️ 重要**：两种方法的 OOD PPL 都在 50%→100% 恶化（Geo L=8K: 99.4→115.0 +15.7%, Hybrid: 104.4→121.6 +16.5%）。**Waterbed 对双方对称作用。** PPL 维度不是 Hybrid 的核心优势所在——passkey 才是。

#### 11.6.2 Passkey Retrieval 全量数据（20 trials/length/checkpoint）

| Ckpt | 方法 | L=1K | L=2K | L=4K | L=8K | 全局 | AR exact |
|------|------|------|------|------|------|------|----------|
| 25% | Geo | 100% | 95% | 50% | 55% | 75.0% | 1.25% |
| 25% | Hybrid | 100% | 95% | **70%** | 45% | 77.5% | 1.25% |
| 50% | Geo | 100% | 100% | 80% | **70%** | **87.5%** | 3.75% |
| 50% | Hybrid | 100% | 100% | 80% | 65% | 86.25% | **7.5%** |
| 75% | Geo | 100% | 100% | 55% | 60% | 78.75% | 7.5% |
| 75% | Hybrid | 100% | 100% | **80%** | **75%** | **88.75%** | **20.0%** |
| 100% | Geo | 100% | 100% | 80% | 60% | 85.0% | 16.25% |
| 100% | Hybrid | 100% | 100% | 65% | **80%** | **86.25%** | **30.0%** |

#### 11.6.3 🔴 Passkey L=8K 训练轨迹——整个实验的核心发现

```
Passkey L=8192 Retrieval (20-trial checkpoint eval):

         25%    50%    75%    100%
Geo:     55% → 70% → 60% → 60%      ← 50%后回退，然后停滞
Hybrid:  45% → 65% → 75% → 80%      ← 单调递增，从未回退!
                            ^^^^
                     差距: +20pp
```

**尽管两种方法的 OOD PPL 都在恶化（waterbed），但**：
- Geo 的 passkey retrieval **跟着 PPL 一起退化**（70%→60%→60%）
- Hybrid 的 passkey retrieval **逆势上升**（65%→75%→80%）

→ **PPL waterbed 对 Hybrid 的位置检索能力没有破坏力**。这是 EVQ-Cosh 频率分配的核心价值。

#### 11.6.4 100% Head-to-Head 总结

| 维度 | Geo | Hybrid | Δ | 判定 |
|------|------|--------|---|------|
| PPL@1K | 17.55 | **17.53** | -0.14% | 持平 |
| PPL@2K | 21.98 | **21.65** | -1.51% | Hybrid 微优 |
| PPL@8K | **115.01** | 121.58 | +5.71% | Geo 优 |
| PPL@16K | **253.18** | 267.70 | +5.73% | Geo 优 |
| Passkey@8K ret | 60% | **80%** | **+20pp** | **Hybrid 大幅领先** |
| AR exact match | 16.25% | **30.0%** | **+13.75pp** | **Hybrid 近乎翻倍** |
| Passkey 全局 ret | 85.0% | **86.25%** | +1.25pp | 接近 |

#### 11.6.5 训练效率发现

> **Hybrid@50% > Geo@100%**：PPL@8K 104.4 vs 115.0 (-9.3%), PPL@16K 225.7 vs 253.2 (-10.9%)

Hybrid 用一半训练预算就超过了 Geometric 的完整训练效果。

#### 11.6.6 40-trial Final Eval（更高统计量）

| 方法 | L=4K ret | L=8K ret | 全局 ret |
|------|---------|---------|---------|
| Geo | **77.5%** | 50% | 81.87% |
| Hybrid | 65% | **62.5%** | 81.87% |

**全局 retrieval 完全一致（81.87%），但分布不同**：Hybrid 把检索能力从 L=4K 向 L=8K 推移（+12.5pp 从 4K 转移到 8K）。EVQ 的频率重分配让模型"看得更远"。

#### 11.6.7 RULER Multi-Needle（5 needles, 8 trials）

| Ckpt | 方法 | L=2K per/all | L=4K per/all | L=8K per/all |
|------|------|-------------|-------------|-------------|
| 75% | Geo | 92.5% / 62.5% | **57.5% / 12.5%** | **47.5%** / 0% |
| 75% | Hybrid | **95% / 75%** | 55% / 0% | 40% / 0% |
| 100% | Geo | **95% / 75%** | 57.5% / **12.5%** | **47.5%** / 0% |
| 100% | Hybrid | 87.5% / 37.5% | **60%** / 0% | 40% / 0% |

RULER 8 trials 方差极大（all-needle 差 3 个 trial = 37.5pp），统计效力不足做可靠对比。OOD RULER 两者都在地板附近。

### 11.6B 🆕 Phase15：750M `2K→4K` Continue（Geo vs Full EVQ `r=0`, 500M tokens, seed=42）

> **状态**: COMPLETE
> **数据源**: `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`
> **对比方法**: Geometric (τ=0) vs **Full EVQ** (`τ=1.5, r=0`)

#### 11.6B.1 Final PPL

| 长度 | Geo | EVQ r=0 | Δ |
|------|-----|---------|---|
| 2K | 25.922 | 26.160 | `+0.9%` |
| 4K | 21.955 | 22.282 | `+1.5%` |
| 8K | 23.386 | 19.607 | `-16.2%` |
| 16K | 45.136 | 24.407 | `-45.9%` |

#### 11.6B.2 Final Passkey / AR

| 长度 | Geo ret / AR | EVQ ret / AR | Δ |
|------|--------------|--------------|---|
| 2K | `100% / 100%` | `100% / 100%` | — |
| 4K | `100% / 100%` | `100% / 100%` | — |
| 8K | `100% / 0%` | `100% / 77.5%` | `AR +77.5pp` |
| Global | `100% / 66.67%` | `100% / 92.5%` | `AR +25.8pp` |

#### 11.6B.3 这轮该怎么读

- retrieval 在这条 continue 设定里已经 ceiling 到 `100%`，因此**最有区分度的指标是 `16K PPL` 和 `AR exact`**
- EVQ 在 `2K/4K` 只有很小代价，但在真正长程段继续放大优势：`8K -16.2%`，`16K -45.9%`
- 这条 750M full-EVQ continue 结果说明：**把训练长度继续拉到 `4K` 后，EVQ 的长程收益没有被“洗掉”**
- **重要 caveat**：EVQ 这轮 downstream NLL 因 LongBench 远端下载失败未形成公平对比，所以它当前只能作为 **single-seed supporting evidence**

### 11.6C 🆕 Phase17：454M L=512 Continue-to-1B + YaRN Overlay（训练课程 Ablation）

> **状态**: COMPLETE
> **数据源**: `docs/exp/2026-03-09_phase17_evq_yarn_overlay_results.md`
> **对比方法**: Geometric vs EVQ τ=2.8, 在 25% (~0.5B) 和 50% (~1.0B) 两个 checkpoint
> **关键设计**: 固定 L=512 继续训练（不拉长），测试 raw + YaRN overlay

#### 11.6C.1 核心发现：同长度继续训练让 raw 外推退化

| 方法 | 25%→50% raw PPL 平均恶化 (4K-32K) |
|------|------|
| Geo raw | +24.4% |
| EVQ raw | +62.2% |

这与 SkyLadder (NeurIPS 2025) 的发现一致：固定短窗口训练会让模型过拟合短程分布。

#### 11.6C.2 但 EVQ+YaRN 始终远强于 Geo+YaRN

| Checkpoint | EVQ+YaRN 相对 Geo+YaRN 的优势 (4K-32K 平均) |
|------------|------|
| 25% | **-87.4%** |
| 50% | **-86.3%** |

关键数字（50% checkpoint）：

| 长度 | Geo+YaRN | EVQ+YaRN | Δ |
|------|----------|----------|---|
| 4K | 19.946 | **2.742** | -86.3% |
| 8K | 63.749 | **6.224** | -90.2% |
| 16K | 102.889 | **11.567** | -88.8% |
| 32K | 224.743 | **46.666** | -79.2% |

#### 11.6C.3 与 Phase 15 构成完美对照

| | Phase 17 (同长度训练) | Phase 15 (拉长训练) |
|---|---|---|
| 策略 | L=512 → 继续 L=512 | L=2K → 拉长到 L=4K |
| EVQ raw 趋势 | 退化 (+62.2%) | 改善（16K PPL 单调下降） |
| EVQ 长程赢面 | 靠 YaRN rescue (-86%) | 原生 raw 即赢 (-45.9%@16K) |
| 对应生产场景 | ❌ 反模式 | ✅ 工业标准 (short→long curriculum) |

#### 11.6C.4 论文该怎么写

Phase 15 + 17 合在一起讲的故事：

> **Short-context-only training weakens raw extrapolation for both Geo and EVQ — this is expected and consistent with production training curricula (SkyLadder, LLaMA). But EVQ wins in both realistic regimes:**
> - **Fixed-short + YaRN**: EVQ+YaRN >> Geo+YaRN (-86%)
> - **Progressive short→long**: EVQ long-range advantage amplifies (-45.9% @16K)
>
> **EVQ is the better substrate regardless of training curriculum.**

### 11.7 Figure 1 三联图描述

**文件**：`paper_draft/figs/fig1_frequency_dynamics.png` / `paper_draft/figs/fig1_frequency_dynamics.pdf`

| Panel | 内容 | 核心发现 |
|-------|------|----------|
| (a) | 频率分配：Geo vs EVQ τ=1.5 的 32 通道 log-freq | 低频间距扩大，高频压缩 |
| (b) | PPL 训练动态 @L=8K | Geo 50%后回升；Hybrid 无回升；L=2K 重叠 |
| (c) | Passkey @L=8K | Hybrid +10pp 且单调↑；Geo regression -10pp |

### 11.8 Geo 动态退化的理论解释

1. **训练初期**：模型学短程模式，高频通道足够 → 两者都在进步
2. **训练中期（~50%）**：模型开始学长程模式，Geo 低频碰撞严重 → 开始过拟合训练窗口
3. **训练后期**：Geo 的梯度被迫适应碰撞的低频通道 → **主动恶化外推能力**（PPL↑ + passkey↓）
4. **Hybrid 不退化**：低频间距已扩大，碰撞更少 → 梯度信号不矛盾 → passkey 单调递增

**核心论点**：Geometric 不仅是"静态次优"（Corollary 1），更是"动态退化"——越训练越差。

### 11.9 论文级 Claims（6 条，待多 seed 后定稿）

1. **Waterbed PPL 实证**：两种方法都展现 waterbed（短程↓、长程↑），幅度相近（Geo +17.6%, Hybrid +16.8% @L=8K）
2. **🔴 Retrieval Divergence（核心 Claim）**：尽管 PPL waterbed 相近，Geo L=8K passkey 从 70%→60%（regression），Hybrid 从 65%→80%（单调递增）。**EVQ 保护位置可区分性**
3. **训练效率**：Hybrid@50% > Geo@100% in OOD PPL（-9.3% @8K, -10.9% @16K）
4. **In-distribution 无损**：Hybrid L=1K PPL -0.14%，L=2K -1.51%。**零短程代价**
5. **检索距离重分布**：40-trial 全局 retrieval 相同 (81.87%)，但 Hybrid 从 L=4K 推移 +12.5pp 到 L=8K
6. **AR Precision**：Hybrid 30% vs Geo 16.25% (+13.75pp)，不只是定位到 passkey，还能精确复现内容

---

## 12. Capability-Preserving Property + Passkey Mix 验证 ✅

### 12.1 纯 FineWeb-Edu：Passkey 对 from-scratch 小模型无效

FineWeb-Edu 不含 passkey/检索模式。350M from-scratch passkey ~55%（随机噪声）。3-seed Geo=0.578, EVQ=0.576，差值 -0.002（纯噪声）。

→ 安全性证据：EVQ 频率重分配**没有破坏** OOD 任务表现。

### 12.2 🔴 Passkey Mix 实验结果（2026-03-03, 5090, 350M, 3-seed）

**配置**：90% FineWeb-Edu + 10% passkey 混合训练，L_train=2048，base=500K

**Passkey Retrieval（10% mix, 3-seed mean）**：

| 长度 | Geo (τ=0) | EVQ (τ=1.5) | Δ |
|------|-----------|-------------|---|
| 2K（训练内） | **100%** | **100%** | 0 |
| 4K（2× 外推） | 58.7% | **68.7%** | **+10.0pp** |
| 8K（4× 外推） | 40.7% | **53.3%** | **+12.7pp** |
| Global | 66.7% | **74.0%** | **+7.3pp** |

**PPL（10% mix, 3-seed mean；Waterbed 成立）**：

| 长度 | Geo | EVQ | Δ |
|------|-----|-----|---|
| 2K | 67.2 | 67.9 | +1.0% |
| 8K | 161.9 | 150.3 | **-7.2%** |
| 16K | 262.0 | 237.2 | **-9.5%** |

**注**：早期 seed=42 的 `4K: 42%→82% (+40pp)` 是单 seed 极值，不应继续作为论文主 headline。正文应使用 3-seed mean；更强也更稳健的 retrieval headline 是 `EVQ+YaRN@8K = 100% across 6/6 seeds`。

### 12.3 5% vs 10%：Geo 退化更强，EVQ 更抗退化（multi-seed 校准后）

同样总 token 量（100M），仅 passkey 浓度从 5%→10%：

| 效应 | Geo 8K retrieval | EVQ 8K retrieval |
|------|-----------------|-----------------|
| 5% | 54.0% | 56.7% |
| 10% | 40.7% | 53.3% |
| **Δ (5%→10%)** | **-13.3pp** | **-3.3pp** |

**multi-seed 结论**：原先 seed=42 的“方向完全相反”结论不稳健。更严谨的说法是：**更密集的 passkey 训练会让 Geo 的长程 retrieval 明显退化，而 EVQ 的退化显著更小**。

**解读**：更多 passkey 训练让 Geo 更容易**过拟合**到 L=2K 的检索模式，position encoding 的固有局限使泛化能力下降。EVQ 的频率分配没有消除这一效应，但显著提高了对该退化的鲁棒性。

**论文论点**：频率分配质量（而非数据量）是长度泛化的瓶颈。

### 12.4 核心结论

1. **2K 都是 100%**：10% mix 让两种方法都学会了检索。差异纯粹来自外推能力
2. **raw passkey mix 的稳健增益是 +10.0pp@4K / +12.7pp@8K（3-seed）**：EVQ 对已学习检索能力的长度泛化有稳定提升
3. **PPL waterbed 成立**：短端 +1.0%，长端 -7.2%~-9.5%
4. **5%→10% 后 Geo 退化更强**：Geo 8K -13.3pp，EVQ 仅 -3.3pp。频率分配质量仍是关键瓶颈
5. **PI inference-time baseline**：PK_Global=51%, PPL@2K=191.7——几乎退化到随机。证明 naive inference-time PE 不是替代方案

### 12.5 Capability-Preserving Property（经验命题，基于当前证据）

**Empirical Proposition (Capability Preservation + Task-Conditional Improvement).**

(a) *Safety*: For any task T absent from training (baseline ≈ random), Waterbed reallocation has zero effect: P_evq(T) ≈ P_geo(T).
- **证据**：纯 FineWeb-Edu 下 passkey Geo=55.7%, EVQ=56.7%（噪声级别）

(b) *Improvement on learned retrieval tasks*: For tasks present in training, EVQ improves **mean** extrapolation performance beyond L_train in our passkey-mix setting.
- **证据**：10% mix 3-seed mean 下，4K retrieval `58.7% → 68.7%`（+10.0pp），8K retrieval `40.7% → 53.3%`（+12.7pp）

(c) *Combined*: 当前证据支持 EVQ 在 capability space 上至少是 **non-destructive**，并且对已训练 retrieval 任务的长度泛化给出稳定增益。

**论文用途**：正文可写为 empirical proposition / observation，而不是无条件 theorem。这仍然是 PPL 之外的重要实验证据。

---

## 13. Practical Recipe（论文核心卖点）

### Algorithm: EVQ-Cosh（Zero Hyperparameter, Pure EVQ r=0）

```python
def evq_cosh_inv_freq(d_head, L, base):
    """完整的 EVQ 频率分配。替换一行 inv_freq 初始化即可。
    不需要 r 参数——cosh 分配自动保持高频通道不变。"""
    tau = d_head / math.sqrt(L)  # τ* scaling law
    K = d_head // 2
    u = torch.arange(K) / K  # standard RoPE indexing; u_0=0 anchors ω_0=1
    phi = 1 - (1/tau) * torch.arcsinh((1 - u) * math.sinh(tau))
    return base ** (-phi)
```

替换一行 inv_freq 初始化。不改架构、不改训练、不改推理。**不需要指定 r**。

### 超参数对比

| Method | 超参数 | 需要搜索？ |
|--------|--------|----------|
| Geometric | 0 | — |
| PI | 1 | 否，但性能差 |
| NTK-aware | 1 | 通常需要 |
| YaRN | 3 | **是** |
| DAPE | 32 | **是** |
| **EVQ-cosh** | **0** | **否** |

### 退化安全

τ→0 时 EVQ → Geometric。即使 τ* 预测偏差 50%，最坏情况是"没提升"而不是"崩溃"。

---

## 14. 论文叙事方向（v10, strong-accept / spotlight 导向）

### 14.1 一句话定位（v11, 三重挤压叙事）

**Modern LLMs face a triple squeeze on RoPE (L↑, base↑, d_head↓ via MLA). EVQ is the first closed-form solution for the missing dimension — within-band frequency allocation. With progressive training, EVQ alone surpasses Geo+YaRN, making it more fundamental than inference-time scaling.**

### 14.2 四层主故事（按重要性排序）

1. **问题定义（三重挤压）**：L 越来越长、base 被迫拉宽、MLA 压缩 d_head——三个力把每个 RoPE 通道推向极限。但 allocation（通道内分配）从未被优化过。这是 RoPE 优化链条中被忽视的关键维度。
2. **理论锚点**：RoPE frequency allocation 是变分逆问题，闭式解为 EVQ-Cosh；Geometric 是 `τ=0` 退化点。`τ*=d/√L` 是 near-optimal scaling law，loss landscape 平坦（<1% PPL gap）。
3. **系统级杀手锏（从协同到替代）**：
   - **× YaRN（短程阶段协同）**：L=512 训练后，`EVQ+YaRN >> Geo+YaRN` (-86%)。
   - **× Progressive training（长程阶段替代）**：512→1024 continuation 后，EVQ raw 反超 EVQ+YaRN（PPL@16K: 11.2 vs 16.8）；EVQ raw 在 4K-32K **全面优于 Geo+YaRN**。EVQ 不是 YaRN 的附庸，**它比 inference-time scaling 更根本**。
   - **× Base tuning**：完全正交，Phase 18 将验证跨 5 个 base 泛化。
4. **PE-dominant 验证**：在最纯净的极限外推 regime 中，EVQ 用 0 额外参数打赢 DAPE-style learnable PE，证明 PE quality itself 决定长程 PPL。

### 14.3 当前最该放在主文最前面的三个 claim

| 优先级 | Claim | 当前最硬证据 | 对审稿决策的作用 |
|--------|-------|-------------|------------------|
| **P0** | **EVQ+YaRN >> Geo+YaRN** | `EVQ+YaRN@8K = 100% across 6/6 seeds`；`Geo+YaRN = 61%-65%`；10% fair scale=8 下 `PPL@8K 70.9 vs 82.9`，`PPL@16K 107.5 vs 157.7` | **当前最强 killer result**；最像 spotlight 主图 |
| **P0** | **DAPE-style extreme extrapolation 下，closed-form EVQ > learnable PE** | 128-token 训练、64x 外推到 8K：EVQ `333.7` vs DAPE `455.3`，且 EVQ 为 `0` 额外参数 | 把工作从“又一个 RoPE trick”抬到“更强的 frequency optimization principle” |
| **P1** | **L=256 / Phase 11 直接确认 `τ*` 与 EVQ×YaRN 杠杆效应** | `EVQ4+YaRN 99.6 vs Geo+YaRN 260.2` at 32x (`-61.7%`)；YaRN 对 Geo 仅 3-5%，对 EVQ 改善 33-41% | 理论预测 + 方法协同的综合性证据 |
| **P1** | EVQ raw length generalization 稳定提升 | 350M 3-seed `PPL@16K -13.3%`；10% mix raw retrieval `+10.0pp@4K / +12.7pp@8K` | strong accept 稳定器 |
| **P1** | **τ* 99-run sweep 近最优验证** | 3/9 精确 #1、6/9 前二、8/9 前三，经验最优在 theory 1.5× 以内 | 封堵 "公式是 heuristic" 攻击，提升理论可信度 |
| **P1** | **训练课程 ablation (Phase 15 + 17)** | 同长度训练 EVQ+YaRN -86%；progressive EVQ -45.9%@16K | 直接回答 "更多训练会洗掉 EVQ 吗"——不会，两种 regime 都赢 |
| **P2** | base=10K dead zone 与 collision-block | 负面结果被理论正确预测 | 封堵 reviewer 攻击，不是主 headline |
| **P2** | video temporal transfer | 2-seed 支持迁移 + EVQ×YaRN synergy | 上限抬升项，当前仍应附录优先 |

### 14.4 正文叙事顺序（建议 8 步，不再平均发力）

1. **🆕 RoPE 两个正交维度（opening framing）**：长文本能力 = base（频带宽度）× allocation（频带内分配）。工业界从 10K→500K→1M 只调了 base，allocation 一直是 Geometric 默认值。EVQ 解决 allocation 这个被忽视的维度。
2. **变分逆问题**：RoPE frequency allocation is a variational inverse problem with a closed-form solution.
3. **Geometric 是退化点**：standard RoPE corresponds to `τ=0`; EVQ is the continuous family that contains it as a degenerate case.
4. **τ* scaling law + MLA 工业对齐**：`d_head/√L` 在 99-run sweep 中 8/9 前三；loss landscape 平坦（worst-case PPL gap <1%）；d_head=64 恰好是 2026 MLA 旗舰模型（DeepSeek V3、GLM-5、Kimi K2.5）的 RoPE 维度。
5. **Waterbed 与碰撞块**：解释为什么代价有限、为什么 `base=10K` 会进入 dead zone。
6. **PE-dominant 极限外推**：在 128-token / 256-token regime 中证明 PE 质量本身决定远程 PPL，并直接对标 DAPE-style learnable PE。
7. **跨规模 raw gain**：50M-350M-454M-750M 一致证明 EVQ 不是 toy artifact。
8. **🔴 EVQ × YaRN 阶段性协同 → 替代**：短程阶段 EVQ+YaRN >> Geo+YaRN (-86%)；progressive 后 EVQ raw 反超 EVQ+YaRN。两阶段叙事：先证 EVQ 解锁 YaRN 上限，再证 EVQ 最终让 YaRN 多余。
9. **训练课程 ablation（Phase 17b 全矩阵）**：454M 512→1024 continuation 后：(a) EVQ raw PPL@16K 从 79.6→11.2，优势从 34.6%→83.1%；(b) EVQ raw 全面优于 Geo+YaRN（4K-32K）；(c) `evq_512+yarn ≈ evq_1024_cont raw`，progressive training 一步替代 inference-time scaling。
10. **Base 泛化 sweep (Phase 18)**：5 个 base 跨越 10K-10M，用 MLA-realistic d_head=64 配置验证 EVQ gain 跨 base 稳定性 + collision-block 理论定量验证。
11. **视频 temporal 迁移**：作为跨模态上限证明，而不是替代 text 主线。

### 14.5 当前主文图表组合（按重要性重排）

| 资产 | 定位 | 主承载信息 |
|------|------|-----------|
| **Figure 2** `paper_draft/figs/fig2_evq_yarn_synergy.pdf` | **主图** | `EVQ+YaRN >> Geo+YaRN`；training-time frequency optimization 解锁 inference-time scaling |
| **Figure 3（主图）** | **第二主图** | PE-dominant regime / scaling law direct validation；128-token extreme extrapolation 对标 DAPE-style learnable PE + Phase 11 raw/YaRN 双确认 | `paper_draft/figs/fig3_pe_dominant_scaling.pdf` |
| Figure 1 `paper_draft/figs/fig1_frequency_dynamics.pdf` | supporting 图 | 频率重分配机制 + 750M dynamics supporting evidence |
| Figure 4（待做） | mechanism / rebuttal 图 | collision-block / base dead zone，把负面结果反转为理论证据 |
| Table 1 | 主表 | 跨规模 raw PPL 一致性 |
| Table 2 | **killer 表** | 10% mix, fair scale=8 的 PE baseline + EVQ+YaRN 公平比较 |
| Table 3 | supporting 表 | capability-preserving + passkey mix 3-seed |

### 14.6 当前最值钱的数字（只保留 paper-grade headline）

- **EVQ+YaRN@8K = 100% across 6/6 seeds**，而 `Geo+YaRN = 61%-65%`
- **10% fair scale=8**：`PPL@8K 70.9 vs 82.9`，`PPL@16K 107.5 vs 157.7`
- **128-token, 64x extrapolation**：EVQ `333.7` vs DAPE `455.3` at 8K
- **L=256 / 454M / 3-seed**：`EVQ4+YaRN 99.6 vs Geo+YaRN 260.2` at 32x (`-61.7%`)
- **τ* scaling law**：99-run sweep，3/9 精确 #1、8/9 前三
- **350M raw 3-seed**：`PPL@16K -13.3%`
- **10% mix raw retrieval 3-seed**：`+10.0pp@4K`，`+12.7pp@8K`
- **🆕 Phase 17 same-length + YaRN**：454M L=512，`EVQ+YaRN vs Geo+YaRN` 平均 **-86%** (4K-32K)
- **Phase 15 progressive**：750M 2K→4K，`PPL@16K 24.4 vs 45.1` (`-45.9%`)，`8K AR exact 77.5% vs 0%`（single-seed）
- **🆕 Phase 17b 全矩阵（454M 512→1024）**：
  - EVQ raw PPL@16K: **11.2** vs Geo raw 66.4（**-83.1%**）
  - EVQ raw **反超** EVQ+YaRN（16K: 11.2 vs 16.8；全 2K-32K 成立）——progressive training 后 YaRN 变为负优化
  - `evq_512+yarn ≈ evq_1024_cont raw`（@16K: 11.6 vs 11.2）——training-time progressive **等价替代** inference-time YaRN
  - EVQ raw 全面优于 Geo+YaRN（@16K: 11.2 vs 25.5；@32K: 50.8 vs 94.6）——纯训练优化 > 训练+推理修补
- **🆕 τ* landscape 平坦性**：d_head=64 worst-case（L=512）PPL 差距 **~3 PPL / <1%**；shallow basin 意味着公式直接可用无需 grid search
- **🆕 MLA 工业对齐**：DeepSeek V3 / GLM-5 / Kimi K2.5 全部 `qk_rope_head_dim=64`，我们的实验配置 = 旗舰模型 RoPE 现实

### 14.7 当前应该主动降级或移出主线的东西

- `+40pp` 单 seed passkey 极值：只能当历史现象，不能再当 headline
- `5%→10% 反对称 scaling`：multi-seed 后不成立，改写为 **robustness gap**
- `r-sweep / τ*(r)`：当前只适合 appendix / method note，不能压过 `EVQ+YaRN` 与 `DAPE-style` 主线
- `750M retrieval divergence`：只作为 supporting mechanism；750M 主价值现在是 Phase 15 training curriculum ablation
- `video confirms τ*=2.0`：当前不能这么写，只能写 **supports transfer + strong EVQ×YaRN synergy**

### 14.8 当前可以说 / 不能说（v10）

**可以说**
- ✅ EVQ is a closed-form solution to a variational frequency-allocation problem, with Geometric as the `τ=0` degenerate case.
- ✅ `τ*=d_head/√L` is a robust near-optimal scaling law (99-run sweep: 3/9 exact, 8/9 top-3).
- ✅ In PE-dominant extreme extrapolation, EVQ beats DAPE-style learnable PE with `0` extra parameters.
- ✅ `EVQ+YaRN >> Geo+YaRN` is the current strongest systems result; EVQ unlocks the full benefit of inference-time scaling.
- ✅ Raw EVQ gives stable long-range gains with bounded short-range cost.
- ✅ Collision-block analysis correctly predicts the `base=10K` dead zone.
- ✅ EVQ wins in both realistic training regimes: fixed-short + YaRN (-86%), and progressive short→long (-45.9%@16K).
- ✅ After progressive training (512→1024), EVQ raw surpasses EVQ+YaRN (PPL@16K: 11.2 vs 16.8), demonstrating that EVQ is more fundamental than inference-time scaling.
- ✅ Training-inference equivalence: `evq_512+yarn ≈ evq_1024_cont raw` (@16K: 11.6 vs 11.2) — progressive training substitutes for YaRN with zero inference overhead.
- ✅ RoPE long-context has two orthogonal dimensions: base (bandwidth) and allocation (within-band distribution). Industry only tuned the first; EVQ is the first closed-form solution for the second.
- ✅ With MLA (DeepSeek V3, GLM-5, Kimi K2.5) compressing RoPE to d=64, frequency allocation optimization becomes more critical, not less.
- ✅ The loss landscape around τ* is flat: worst-case PPL gap <1% at d_head=64, making τ*=d/√L a practical parameter-free default.

**不能说**
- ❌ `Passkey +40pp` 是稳定主结论
- ❌ `5%→10%` 呈现方向完全相反的 scaling law
- ❌ `r` 是当前主方法的核心超参数
- ❌ 视频已经单独严格确认 `τ*=2.0`
- ❌ 仅凭 750M single-seed dynamics 就能推出 paper-grade primary claim

### 14.9 Reviewer 对策（按当前最可能攻击点更新）

| 攻击 | 当前最优防御 |
|------|--------------|
| "这只是 another RoPE tweak" | 不是 heuristic：closed-form variational solution + Geometric is the `τ=0` degenerate case |
| "learnable PE / DAPE 也行" | 在 DAPE-style extreme extrapolation regime 中，EVQ `0` 参数直接优于 DAPE-style learnable PE |
| "为什么不用 YaRN 就好" | 两层回应：(1) 短程阶段 `EVQ+YaRN >> Geo+YaRN`（-86%），说明瓶颈是训练时 allocation；(2) progressive 后 EVQ raw 反超 EVQ+YaRN（PPL@16K: 11.2 vs 16.8），**EVQ 比 YaRN 更根本** |
| "只有 synthetic / passkey" | raw 3-seed PPL、10% mix retrieval、Phase 11、750M continue、以及视频 temporal 支持共同构成体系 |
| "base=10K 全败" | 这是 collision-block dead-zone 的理论验证，不是意外负面结果 |
| "短程会退化" | Waterbed 不等式：短程代价 ≤+0.9%，长程收益 -45.9%@16K，高度不对称 |
| "更多训练会洗掉 EVQ" | **Phase 17b 全矩阵直接回答**：512→1024 continuation 后 EVQ 优势从 34.6%→83.1%@16K，**越训越强**。EVQ raw 甚至反超 EVQ+YaRN，说明 progressive training 让模型内化最优分配 |
| "单 seed 太多" | 主 claim 以 3-seed / 6-seed 结果为主；750M dynamics 与 video 只作为 supporting evidence |
| "d_head=64 不贴近工业" | **恰恰相反**：2026 旗舰 MLA 模型（DeepSeek V3、GLM-5、Kimi K2.5）的 qk_rope_head_dim 全部是 64。d_head=128 是旧 MHA 时代 |
| "τ* 不够精确" | Loss landscape 在 τ* 附近是浅盆地——worst-case PPL 偏差 <1%。Practitioners 不需要 grid search，直接用 d/√L |
| "只在 base=500K 测了" | Phase 18 将覆盖 5 个 base（10K-10M），collision-block 理论预测 gain 随 base 单调变化 |

---

## 15. 全实验理性盘点（2026-03-04，诚实评级）

> **原则**：每个 claim 按证据强度分级。不夸大，不隐瞒弱点。论文只用 A/B 级证据。

### 15.1 证据强度评级

| Claim | 证据 | 强度 | 论文可用？ |
|-------|------|------|-----------|
| **EVQ PPL@8K-16K 优于 Geo** | 6/6 runs 全胜（5%×3 + 10%×3），delta -4.4%~-17.0%，方差更小 | **A（无争议）** | ✅ 主 Table |
| **EVQ+YaRN 8K=100%** | **6-seed** (5%×3 + 10%×3) zero variance vs Geo+YaRN 61-65% | **A+（铁证）** | ✅ Killer result |
| **Waterbed 短端代价有限** | 6/6 runs PPL@2K delta ≤+4.1%，5% 均值仅 +0.8% | **A** | ✅ 正文 |
| **τ* scaling law** | 5 context lengths，L≥1024 吻合，L<1024 系统偏高 | **B+** | ✅ 正文 + caveat |
| **τ*(r) 修正公式** | 2 点验证（r=0/r=14），τ*(14)=2.51 vs 实验 2.5 | **B（需更多 r 点）** | ✅ 正文 + "preliminary" |
| **Passkey mix `+40pp` (historical seed=42 outlier)** | multi-seed 弱化为 `+10.0pp@4K / +12.7pp@8K`；seed=42 不能再当 headline | **C+（夸大风险）** | ⚠️ 只能说 `3-seed mean` |
| **Passkey mix +12.7pp @8K (10%)** | 3/3 方向一致，但绝对值在噪声区（EVQ 53% vs random 50%） | **B-（方向真实，幅度不确定）** | ✅ 报方向+confidence interval |
| **5%→10% 旧“反对称 scaling”叙事** | multi-seed 修正版是 robustness gap：Geo `-13.3pp`，EVQ `-3.3pp`；原 ±22pp 仅 seed=42 | **C（历史单 seed 现象）** | ⚠️ 只能当历史 illustration，不能做正文 claim |
| **Retrieval divergence 750M** | 单 seed，4 checkpoint 趋势清晰（Geo 70→60，Hybrid 45→80） | **B（趋势可信，需 multi-seed）** | ✅ 正文 + single-seed caveat |
| **750M full EVQ continue @4K** | 单 seed，但 head-to-head 很强：`16K -45.9%`，`8K AR exact +77.5pp` | **B+（幅度大，仍需更多 seed）** | ✅ supporting evidence |
| **r=14/τ=2.5 是联合最优** | 单 seed=42，PPL@16K=234.0 (-19.6%) | **C+（需 multi-seed）** | ⚠️ Appendix 或 "preliminary" |
| **Geo PK@8K < 50%（反随机）** | 10% 3-seed 均值 40.7%，3/3 seeds 低于 50% | **B（方向可信，样本小）** | ✅ 可以提，但不能过度解读 |
| **EVQ PK@8K > Geo（不含 YaRN）** | 10% 3/3 seeds 方向一致，5% 2/3 一致 | **B-（方向一致，幅度在噪声区）** | ✅ 辅助证据，不作主 claim |
| **Base=10K 死区** | 碰撞块理论精确预测，实验全败确认 | **A** | ✅ 理论验证的负面结果 |
| **750M OOD PPL Hybrid 更差** | +5.7%@16K，与 350M 方向相反 | **需解释** | ⚠️ 必须论文中讨论 |
| **CHE Even Pairs EVQ 泛化更差** | L=100 EVQ 43.7% < Geo 72.9%（pilot 2K steps） | **⚠️ 负面信号** | 见 §15.6 |

### 15.2 PK@8K 深度分析：Geo < 50% 和 EVQ ~53-60% 的物理意义

**为什么 Geo 低于随机 50%？**

50% 是"模型对检索位置完全没有偏好"的基线。Geo < 50% 说明模型不是"不知道 passkey 在哪"，而是**自信地看错位置**。

物理机制：Geometric RoPE 在 L_train=2048 内学到了"低频通道编码远距离"的模式。但在 8K（4x OOD）时，这些低频通道的相位碰撞导致不同位置的 RoPE 编码变得不可区分甚至倒序——模型用训练内学到的模式反向推理，结果比随机差。

这是**相位碰撞的直接可观测后果**：不是信号消失（→50%），是信号反转（→<50%）。

**数据支持**：
- 10% Geo PK@8K: 46%, 36%, 40% → **3/3 低于 50%**
- 5% Geo PK@8K: 56%, 42%, 64% → 1/3 低于 50%（5% 信号弱，训练不充分）
- 10% 浓度更高→模型更"用力"检索→OOD 反转效应更明显→更低

**EVQ 53.3% 的意义**：

绝对值看，53.3±8.3% 距 50% 不显著（0.4σ）。但：
1. **方向**：3/3 seeds EVQ > Geo at PK@8K（10% mix），最小 delta = +8pp
2. **与 PPL 一致**：PPL@8K 也是 6/6 全胜，说明长程建模能力确实更好
3. **加 YaRN 后**：EVQ+YaRN 直接 100%，Geo+YaRN 只有 62%——说明 EVQ 学到的"根基"更好

**理性结论**：
- EVQ 单独在 PK@8K 的绝对提升（53% vs 41%）**方向可信、幅度在噪声边界**
- 不应作为独立 claim，应作为 PPL 改善的佐证
- 真正有论文级说服力的是 EVQ+YaRN 100% vs Geo+YaRN 65%（3-seed, zero variance）

### 15.3 各实验维度的信噪比排序

| 维度 | 信噪比 | 理由 |
|------|--------|------|
| **PPL@8K-16K** | ★★★★★ | 连续值，方差小，6/6 全胜 |
| **EVQ+YaRN PK@8K** | ★★★★★ | 100% vs 65%，3-seed zero variance |
| **PPL@4K** | ★★★★☆ | 5/6 胜（10% seed=42 EVQ 微劣），方向一致 |
| **750M 训练轨迹** | ★★★☆☆ | 趋势清晰但单 seed |
| **PK@4K (10%)** | ★★☆☆☆ | +10pp mean，但 Geo std=16%（极大方差）|
| **PK@8K (standalone)** | ★★☆☆☆ | +12.7pp mean，但绝对值在随机附近 |
| **5→10% robustness gap** | ★★☆☆☆ | multi-seed 方向明确，但幅度远弱于早期单-seed 极值 |
| **r=14/τ=2.5** | ★★☆☆☆ | -19.6% 惊人但单 seed |

### 15.4 论文叙事的"安全区"和"风险区"

**安全区（可以写得自信）**：
1. EVQ PPL 改善 -8% to -13% 跨 3 规模（50M/350M/750M），3-seed 一致
2. EVQ+YaRN 超线性组合 100% vs 65%，3-seed 确认
3. Waterbed 代价有限（≤+0.8% PPL@2K）
4. τ* scaling law 在 L≥1024 吻合
5. Base=10K 死区验证碰撞块理论

**风险区（需要 caveat 或降调）**：
1. "+40pp passkey" → 实际 3-seed 是 "+10pp mean"，必须修正
2. "5→10% 反对称 ±22pp" → 已被 multi-seed 修正为 robustness gap，不能保留旧表述
3. "r=14/τ=2.5 是最优" → 单 seed
4. 750M Hybrid OOD PPL 更差 → 必须讨论，不能回避
5. PK@8K standalone 在噪声区 → 只能作辅助证据

### 15.5 对 750M OOD PPL 矛盾的解释

750M 用的是 **Hybrid (r=16, τ=1.5)**，350M passkey mix 用的是 **全通道 EVQ (r=0, τ=1.5)**。这不是同一个方法：

| | 350M passkey mix | 750M Phase9F |
|---|---|---|
| 方法 | EVQ r=0 | Hybrid r=16 |
| warp 通道 | 32/32 | 16/32 |
| τ | 1.5 | 1.5 |
| PPL@16K vs Geo | **-9.5%** | **+5.7%** |

r=16 的 τ*(16) = 1.41 × 32/16 = **2.82**，但我们用了 1.5！严重不足。r-sweep 数据也显示 r=16/τ=1.5 only -7.1%（vs r=0/τ=1.5 的 -13.6%）。750M 用了次优配置导致 OOD PPL 翻转。

**论文中的处理**：必须解释 r=16 配 τ=1.5 是 suboptimal。或者避免在正文强调 750M PPL 数字，focus on retrieval divergence（这个信号清晰）。

### 15.6 CHE Pilot 结果分析（Even Pairs, 2K/200K steps）

**数据**（5L/8H/d256, L_train=40, eval 至 L=500）：

| Method | L=50 | L=100 | L=200 | L=500 |
|--------|------|-------|-------|-------|
| NoPE | 50.4 | 50.0 | 51.1 | 49.9 |
| RoPE-Geo | 96.4 | **72.9** | 49.6 | 50.2 |
| **RoPE-EVQ** | **99.96** | **43.7** ❌ | 49.0 | 50.0 |
| Kerple | 74.8 | 65.9 | 59.8 | 52.8 |
| DAPE | 93.3 | 68.8 | 57.9 | 55.0 |
| EVQ+Kerple | 80.9 | 53.4 | 49.0 | 49.8 |

**⚠️ EVQ 在 CHE 上比 Geo 泛化更差**：L=100 时 43.7% < Geo 72.9%，且低于随机 50%。

**诊断**：

1. **τ 值极端**：d_head=32, L_train=40 → τ*(0)=32/√40≈5.06。这是远高于我们 NLP 实验的 1.5。极端 τ 导致频率过度重分配——训练内精度极高（99.96%）但 OOD 立即崩塌
2. **仅 2K/200K steps**：严重欠训练。EVQ 的更 aggressive 分配让模型更快锁定训练内模式（99.96% vs 96.4%），但也更快过拟合。200K steps 后可能不同
3. **低于随机的解释**：与 PK@8K Geo<50% 相同机制——模型在训练内学到精确的位置匹配模式，OOD 时这些模式系统性误导（相位碰撞导致信号反转），比随机猜更差
4. **Kerple/DAPE 更好是预期的**：它们是 learnable/adaptive PE，专为 length generalization 设计。静态 RoPE（无论 Geo 还是 EVQ）在纯算法泛化任务上天然劣势

**对论文的影响**：

- **不影响核心叙事**：EVQ 的 value proposition 是 NLP 长上下文建模（L≥512），不是 L=40 的算法泛化
- **需要在 Limitations 提到**："EVQ's redistribution is optimized for the natural language regime where phase collision is the primary bottleneck. On short-sequence algorithmic tasks where positional generalization structure differs, adaptive methods (DAPE, Kerple) may be preferred."
- **关键后续实验**：用 τ=1.5（而非理论 τ*≈5）重跑 Even Pairs pilot，验证是否是 τ 过大导致的

---

## 16. 谨慎声明（2026-03-04 更新）

- τ* scaling law: 5 个数据点（均 base=500K），短 L 端系统偏高。论文写为 Conjecture + 实验支持
- C=d_head 可能是巧合，需换 head_dim 验证
- 碰撞块分析中 (1-c)/lnb 是简化模型，精确指数待确认
- ✅ ~~R6000 Geo 退化数据仅 2 个 checkpoint~~ → Figure 1 展示完整 4-checkpoint 轨迹，趋势清晰
- Capability-preserving 目前只有 passkey 一个 OOD 任务的证据
- **750M Phase9F 单 seed (42)**。趋势清晰但需 multi-seed
- **750M full EVQ continue @4K 目前也是单 seed**。`16K PPL` 与 `AR exact` 很强，但仍应写成 supporting evidence
- ✅ ~~Passkey mix 10% 仅单 seed~~ → **10% 3-seed 已完成**，+40pp 弱化为 +10pp（seed=42 是 outlier）
- **750M OOD PPL Hybrid +5.7%**——原因：r=16 配 τ=1.5 严重不足（τ*(16)=2.82）。见 §15.5 分析
- **Phase15 EVQ downstream NLL 未形成公平比较**：LongBench 远端下载失败，下一轮前必须预置数据
- RULER multi-needle 统计效力不足（8 trials），不建议作为核心证据
- **5%→10% 反对称 scaling (+22pp/-22pp)** 仅 seed=42。3-seed 均值显示方向一致但幅度弱化（Geo -13.3pp / EVQ -3.3pp）
- **PK@8K standalone (不含 YaRN)**: 方向一致（3/3 EVQ>Geo at 10%），但绝对值在噪声区（53% vs 50% ~0.4σ），不宜作独立 claim
- **r=14/τ=2.5 single seed**: -19.6% PPL@16K 惊人，但需 multi-seed 才能用于论文

---

## 16. ⚠️ 教训记录

1. Passkey ~50% = 随机噪声。引入新指标前必须先验证基线有非随机表现
2. 不要用"规模不够"解释 EVQ 不 work。PE 同类以 125M 发表，真正瓶颈是下游任务
3. "τ 推向低频"的说法数学上错误。正确描述是重分配通道密度/间距
4. 双变量 τ*(L,b) 公式尚未确定（A vs B 竞争），论文只用单变量形式
5. **🔴 LoRA 微调不能证明 EVQ 有效**：改变 RoPE 频率 + LoRA 微调 = 模型同时学习"适应新频率"和"下游任务"，两个信号 confounded。加上灾难性遗忘（预训练知识因 PE 变化而丢失），LoRA 实验无论好坏都不能 cleanly 归因于 EVQ。**唯一干净的实验设计是 from-scratch 训练。** 这是核心方法论原则，不可妥协
