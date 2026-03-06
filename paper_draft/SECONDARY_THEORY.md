# EVQ 备选理论、猜想与次要实验

> **定位**：发散性理论思考、待验证猜想、Appendix 级别推导、次要实验结果。
> **与 CORE_THEORY.md 的关系**：一旦某个猜想获得实验验证或严格证明，升级到 CORE。
> **最后更新**：2026-03-06（Passkey Mix 多 seed 校准已同步至 CORE / PLAN 口径）

---

## A. 待验证猜想

### A.1 Vanishing Tradeoff Conjecture（2026-03-03 提出）

**猜想**：Let Δ_short(N) = PPL_evq@L_train - PPL_geo@L_train. Then Δ_short(N) → 0 as N → ∞, while Δ_long(N) remains bounded below by a positive constant for any extrapolation ratio α > 1.

**证据**：
- TinyStories 三规模（50M/125M/350M）Δ@2K 全部 ≤ 0（零代价或微赢）
- FineWeb-Edu 350M Δ@2K = +1.9%（单 seed）→ +0.4%（3-seed）
- 趋势：模型越大或数据越简单，短程代价越小

**验证方式**：R6000 750M + FineWeb-Edu 数据。如果 Δ_short 进一步缩小 → 支持。

**论文用途**：如验证，写为 Conjecture + supporting evidence。

### A.2 Learnability-Capacity 框架

**核心公式**：E_total(τ, N) = E_alloc(τ) + E_learn(τ, N)，其中 E_learn ~ f(τ)/N^γ

- Geometric (τ=0): E_alloc 最大，E_learn 最小
- EVQ (τ=τ*): E_alloc 最小，E_learn 最大
- 最优 τ_effective(N) ≤ τ*_theory

**可检验推论**：
1. 固定 base 和 L，更大模型下 EVQ vs Geo 差距应缩小（✅ 部分支持）
2. Base=10K + 小模型 = EVQ 必败（✅ 已确认）
3. Base=500K + 大模型 = EVQ 最可能胜出（⏳ 等 R6000）

**PPL@2K 跨规模退化分析**：

| 规模 | 数据集 | Δ@2K | Δ@16K |
|------|--------|------|-------|
| 50M | TinyStories | -0.3% | -10.9% |
| 125M | TinyStories | -1.7% | -18.9% |
| 350M | TinyStories | -0.4% | -13.7% |
| 350M | FineWeb-Edu | +1.9% | -15.4% |

短程退化仅出现在 FineWeb-Edu → E_learn 而非 E_alloc。

**状态**：Conjecture 级别。需要至少 750M 或 1B 数据点确认 γ 值。

### A.3 "频率注意力"假说（2026-03-03 讨论）

模型隐式学习在不同距离的 attention 中对不同频率通道赋予不同权重。EVQ 提供更好的工具箱，但模型能否用好取决于容量。

与 Learnability 框架统一：E_learn 本质上衡量模型学习"频率注意力"的能力。

**状态**：纯理论直觉，无直接实验验证方案。可在 Discussion 中一句话提及。

---

## B. Appendix 级别理论推导

### B.1 谱误差界（Heat Kernel 视角，Gemini Q1 验证通过）

真实核中 δ 被平滑化为热核 e^{-εA}（ε~1/lnb）：

K ≈ α·e^{-c(lnb)⁻¹·A} + β·A⁻¹

用 αI 近似时，低频模式误差极小（e^{-ελ_n}≈1），高频累积。
Weyl 摄动定理：前 k 个特征值残差 ≤ O(k²/lnb)。

**论文用途**：Appendix Remark。解释为什么宏观 scaling law 行为可信但全矩阵拟合残差高。

### B.2 WKB 变系数鲁棒性（Gemini Q3 验证通过）

**Reviewer 可能攻击**："α(φ),β(φ) 变化时 cosh 不再是解。"

**三层防御**：
1. **边界层分离**：驱动源 b^{-2φ} 衰减常数 2lnb≈18~28，仅在 φ∈[0,~0.05] 有效。等效常系数。
2. **WKB 外层解**：ρ_WKB ≈ C·(αβ)^{-1/4}·cosh(∫τ(s)ds)。光滑单调扭曲。
3. **Nyquist 吸收**：32 点离散化步长 Δφ≈1/N，WKB 扭曲被采样率吸收。

**结论**：变系数推广不改变离散频率分配的宏观行为。

**论文用途**：Appendix Proposition。正文一句话引用。

### B.3 Expected Contrastive Margin (ECM) 修正

用 U(ω) = E_Δ[1-cos(ωΔ)] = 1-sinc(ωL) 替代 b^{-2φ}：
- 低频：U ≈ ω² → 退化为 Fisher
- 高频：U → 1（饱和，消除紫外灾难）

**论文用途**：正文 Remark（1-2 句），Fisher 是 ECM 短距离特例。

### B.4 C = d_head 的几何解释（Gemini Q6 验证通过）

连续泛函 φ∈[0,1] → 离散通道 i∈[1,D]，映射 φ=i/D。
变换后 ODE 有效衰减率 κ_index = κ/D。

Landau-Pollak-Slepian 解释：长度 L 序列独立自由度 ~O(L)，d_head/2 通道捕获 O(L) 自由度时，τ* 是分配比例。C=d_head 是维度还原的必然结果。

**论文用途**：Appendix Remark。

---

## C. 双变量 τ*(L,b) 理论（未定，搁置）

### C.1 两套竞争公式

**公式 A**（简单，Phase 8D 标定）：
$$\tau^*(L, b) = \frac{d_{head}}{\sqrt{L}} \cdot \sqrt{\frac{\ln 500000}{\ln b}}$$
→ 预测 τ*(10K, 4096) ≈ 1.19。**已被 50M 实验否定**。

**公式 B**（Gemini 严格推导）：
$$\tau^*(L, b) \propto \frac{1}{\sqrt{L}} \cdot \frac{\sqrt{b}}{\ln b \sqrt{1-3c^2+2c^3}}$$
→ 预测 τ*(10K, 4096) ≈ 0.68。待验证。

**核心分歧**：α* 中是否含 b 因子（Q6: α∝1/(L·lnb) vs Gemini: α∝b/(L·(lnb)²)）。

### C.2 临界 Base b_c

$$b_c = L^{A/(P(\tau)+\epsilon)}$$

b_c 随 L 指数增长。Base=500K + L=4K 可能已接近 b_c，解释 Phase 8F 统计等价。

### C.3 EVQ 增益 1/lnb Scaling（原始版，已被碰撞块分析修正）

原始：ΔJ ∝ 1/lnb。
修正：ΔJ ∝ (1-c)/lnb（见 CORE §9）。
原始公式忽略了碰撞块收缩因子 (1-c)。

**状态**：双变量理论整体搁置。论文只用单变量 τ*(L) = d_head/√L。

---

## D. 跨学科联系（论文一句话级别）

### D.1 Waterbed ↔ Bode 灵敏度积分

| RoPE 概念 | 控制论对应 | 数学本质 |
|-----------|-----------|---------|
| Waterbed 不等式 | Bode ∫ln|S(jω)|dω ≥ 0 | 对数约束守恒 |
| Geometric 分配 | Constant-Q 小波 | 尺度不变 Δf/f=const |
| EVQ 偏移 | Gabor 变换 | Heisenberg Δt·Δf ≥ C |
| d/2 频率维度 | 系统极点数 | 信息容量上限 |

### D.2 信道注水 + 匹配滤波

- Shannon capacity 替代 Fisher → log(1+SNR·ρ) 的边际递减自动防高频堆积
- QK^T = matched filter，C_interf 是旁瓣能量惩罚

**论文用途**：Related Work 一句话（Bode），Conclusion 一句话（Heisenberg）。

---

## E. 信息论/贝叶斯解释

### 贝叶斯相变

- 短 L：低频通道梯度 SNR ≈ 0 → 需要强先验（大 τ）
- 长 L：长距离观测 O(L²) 爆炸 → 先验退火（小 τ）

τ* scaling law = "硬先验主导" → "数据驱动主导" 的贝叶斯相变临界方程。

**状态**：定性直觉正确，缺乏定量推导。Discussion 可提。

---

## F. 次要实验结果

### F.1 Phase 8F τ=1.0 在 base=500K L=4096 下不赢

4-seed 均值：EVQ PPL@16K=193.9 vs Geo=175.7。统计不显著但趋势偏差。
原因：τ*=64/√4096=1.0 在 base=500K 下可能接近 b_c（碰撞块太小）。

### F.2 Base=100K Smoke Test 失败

Retrieval 0.615 < Geo 0.71。原因：固定 τ=1.0 在 base=100K 下过大，需先标定 τ*(100K)。

### F.3 Phase 8D 短 L 端 τ* 偏高

L=128,256,512 的实测 τ* 系统高于 d_head/√L。PE-dominant regime（模型容量不足以利用所有频率通道）。

### F.4 Anchored Sigmoid 历史战绩（EVQ 前身）

from-scratch 预训练：
- 50M: PPL@16K 赢 Geo 7-11%
- 125M: 赢 6-19%
- 350M (L=2048): 赢 13.7%
- 8B LoRA: LongBench +14.5%, Passkey@16K 100% vs 80%
- **全部 L_train=2048**

注：anchored_sigmoid ≠ evq_cosh，用 sigmoid 函数而非 arcsinh。论文中需注明。

---

## G. 进行中实验（等结果后可能升级到 CORE）

### G.1 ✅ 5090 Passkey Mix — 已完成，已升级到 CORE §12

- 初始 seed=42 极值：4K retrieval 42%→82%（+40pp），PPL@16K -4.4%
- multi-seed 校准后，CORE 主文使用 10% mix 的 3-seed mean：raw retrieval +10.0pp@4K / +12.7pp@8K，并以 EVQ+YaRN@8K = 100% across 6/6 seeds 作为更强 headline
- Capability-Preserving 的正文表述已收紧为 empirical proposition / observation
- **详见 CORE_THEORY.md §12**

### G.2 R6000 Phase9F Hybrid Checkpoint Trajectory

- 目标：验证 EVQ/Hybrid 是否避免 Geo 的训练退化
- 配置：750M, L=2048, 1B tokens, Geo→Hybrid 自动衔接
- Geo 已显示退化：PPL@8K +9.4%, PPL@16K +10.7%（50%→75%）
- **如 Hybrid 不退化**：升级 Geo 退化为正文核心论据

### G.3 PE Baseline 对比（待做）

- YaRN/NTK-aware/Dynamic NTK 是 post-hoc inference-time 方法
- 可用已有 Geo checkpoint 直接 eval（零训练成本）
- 最高 ROI 实验之一

### G.4 Real Downstream Tasks（待做，冲 spotlight 关键）

- LongBench / SCROLLS / RULER
- 需要较大模型（≥750M）+ 足够训练 + 标准 eval pipeline
- 这是论文从 poster 升级 spotlight 的核心变量
