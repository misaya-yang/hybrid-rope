# Full-RoPE 有限频率分配审计 — 最终报告 (2026-08-19, v2)

可复算脚本: analysis/full_rope_audit/{verify_core, verify_onesided, verify_counterexamples,
verify_small_models}.py + finK_compare.py (agent) + finK_final.md + attention_interface.md。
证据分级: T1 定理级 / T2 命题级 / T3 数值事实 / 反例 / 已证伪。

## 验证状态总表
- T1 (双实现, 1e-14): Gram 2×2 块结构、canonical correlations = 白化 cross-Gram、
  logdet 恒等式、σ₁ = 相位最优化 cos 重叠、单边网格完整闭式、Fisher 闭式 I=Φᵀ(σ²I+ΦΣΦᵀ)⁻¹Φ、
  G_attn=(ΦᵀΦ)(Σ+θθᵀ)(ΦᵀΦ)。
- T2 (手推常数+数值一致): 塌缩定律 1−σ²=(ωL)⁴·{1/45, 1/525}, L-无关; 小 L 从下方收敛
  (L=8→64: 0.001392→0.001851; 0.016899→0.021691)。
- T3 (双实现交叉): effrank 表、finite-K 最优解、三类反例、anchor 偏离。
- 已证伪 (本审计自己的猜想): "过采样饱和于 2K*+2" (V-S 数据推翻); "格子=2πk/L"
  (修正为奇偶类格, 见 Q2/Q4)。
- 未做: checkpoint Σ 实测、训练动力学机制、过采样区 rank(K,L) 解析律。

## Q1: cosine-only kernel 漏掉了什么
真实 pair 贡献 C·cos(ωΔ)+D·sin(ωΔ) ∈ V_ω (2D)。论文 kernel (a1:334) = 单边均匀 prior
下完整 2×2 Gram 的 cos-cos 块 (verify_onesided.py)。漏掉: sin-sin 块、cos-sin 交叉块
(单边网格低频对 +0.82/−0.39; causal 网格上 {σ}≠{cos,sin重叠}, 最大偏差 0.87)、
相位不变性 (同子空间相位差 1.3rad 时 cos-only 重叠 0.270; σ₁=相位最优化重叠, T1)、
sin 方向塌缩更紧 (1/525 vs 1/45)、能量各向异性 (原始 cond 10¹¹–10¹⁴)。
仓库内已有同类批评 (EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md:357-388); 本审计独立推导并给出
精确度量。另: train.py:336 训练端 penalty 仍是带 TODO 的指数 toy kernel。

## Q2: full-RoPE collision 定义 + 构造性答案
定义 (T1): pairwise = canonical correlations σ₁≥σ₂ = svals(Q_aᵀQ_b) = 白化 cross-Gram
奇异值; C_full = Σ_{i<j}(σ₁²+σ₂²)/2。相位不变; 2-block logdet 恒等式 (T1);
K-block 逐对可加性 logdet ≈ Σself + Σlog(1−σ₁²)(1−σ₂²): 健康区 <1%, 塌缩区 K-body 主导
(K=16 lowcluster: 逐对 −2541 vs 真值 −941)。对称网格上 σ 精确 = (cos重叠, sin重叠) (3e-15)。

**构造性答案 (新, 小模型穷举验证)**:
- 单边网格上零交叉条件: 全部通道在同一奇偶类格 ω_k = πa_k/L (a_k 同奇偶)。
  容量 K* = ⌈floor(L/π)/2⌉。验证: K=3@L=16 (a={1,3,5}) er 6.0000/6; K=5@L=32 er 10.0000/10。
- K=2 全局最优性 (穷举 L=16/24/32): C_full 全局最小 = 3e-33 (精确 0), 仅在格点对。
- 混奇偶 → 秩损 (K=4@L=16: 7.60/8)。
- 端点钉死 (L=64, K=4, ω₁=1/512, ω₄=1): 最优 er 7.988/8, 边界层代价 ~0.15%。
- **过采样区 (K>K*)**: 最优解 = 最大奇偶类 + 分数偏移打包, 秩**连续退化**而非饱和
  (L=32: K=7 → 12.96/14; 最优偏移含 1/4, 3/4, 1/2 格单位)。本审计原猜想 "饱和于 2K*+2"
  被穷举证伪 (12.96 > 12)。
- 真实 RoPE: geometric 所有通道在无理偏移上, 永远不在格; 低频带 = 过采样溢出区。

## Q3: 低频塌缩 — 真实, 强, 现在有精确相图
定律 (T2): V_ω → span{1,Δ}, 主角度 ~(ωL)², 常数 1/45, 1/525。强度: b=5e5/L=4096/K=64
geometric er_whit 26.1/128 (24/64 通道 ωL≤1); EVQ τ=1.4/2.5 → 39.6/65.6。
训练模型证据: 22 个慢通道 = 1.05 logit 单位的 recency 核 (RESEARCH_MEMO) — 塌缩子空间
被实际使用, "dead zone" 图景不完整。
相图 (V-S): K≤K* → 精确 2K; K>K* → 连续退化; ωL≪1 → span{1,Δ}。

## Q4: 哪个 finite-K objective 最有数学依据 + cosh 的地位
- 静态: whitened-Gram logdet/entropy effrank (唯一精确处理 K-body); pairwise 和是健康区
  leading order。三个静态 optima (cos/full/logdet) 收敛到同一类"清空低频带"表
  (er 121–124/128, K=64/L=4096); uniform-in-ω 127.8 更高。
- 但静态 objective ≠ 训练目标 (7/8 anchors effrank 在 τ* 后仍升; 仓库: 静态选 τ≈11-14);
  effrank 随 L 上升 (causal +63%) — 静态可辨识性与外推难度方向相反。
- **cosh 非唯一 (用户纠正, 已采纳)**: cosh 只是 C_app 拟合 surrogate 的条件最小元
  (Thm 1 层次正确但对象错位); 真实 finite-K 问题有平坦平台: cosh 族 τ=6→40 全为
  122–128/128, 三个 objective 与 uniform 表形状互异而秩相近。静态解平凡:
  healthy-zone 铺开 → rank≈2K; 精确解是奇偶类格 (Q2)。
- attention 接口 (T1): G_attn 不是 Fisher; Fisher 首阶 = 静态 Gram (σ⁻²ΦᵀΦ),
  Σ 修正项 σ⁻⁴ΦᵀΦΣΦᵀΦ; G_attn 协方差部分 = −σ⁴×该修正。静态理论隐含 Σ∝I;
  Σ 是唯一能把"偏离格"翻译成"信号权衡"的输入。

## Q5: 推翻/约束当前思路的结果
1. 反例①: C_cos(A)<C_cos(B) 但 effrank(A)<effrank(B) — 对称网格 2–5% (1e-14 cross-check),
   causal 8.0%; 手设计例 C_cos 差 486× 而 effrank 相反。正面发现: C_full 在所有反例中
   排序从未出错。
2. 反例②: C_L 排序在 2L/4L 反转 — 15.5%/13.1%/4.1% (双 metric), 幅度最高 0.14。
3. 反例③: effrank 随 L 上升 — 静态几何与模型退化方向相反。
4. 静态 optimum ≫ τ* (7/8 anchors) — 训练不优化静态 collision; 论文自身只写
   "direction check" (a1:134)。
5. 逐对 metric 塌缩区失效; cos-opt 二维退化。
6. 本审计自证伪两例 (2K*+2 饱和猜想; 2πk/L 格不完整) — 奇偶类格 + 连续退化取代之。
判决: "lowering collision ⇒ better extrapolation" 被证伪; 静态 collision 至多是
训练窗口内可辨识性描述。可辨识性最大化的精确答案是奇偶类格, 不是任何 cosh/几何形状。

## Q6: 下一步 3 个动作
1. **过采样区 rank(K,L) 解析律** — 唯一剩下的理论硬核: 分数偏移打包的最优结构
   (数据给出 1/4, 3/4, 1/2 偏移; 猜想与 Dirichlet 核的相位选择有关, 待推导)。
2. **checkpoint Σ/G_attn 实测** (results/weekend_sweep/*.pt, 一次 forward pass):
   检验 Σ 各向异性、低频带使用强度, 把"训练表偏离格"翻译成信号权衡。
3. **论文理论节改写**: canonical-correlation metric + 奇偶类格定理 + (ωL)² 塌缩定律
   (常数 1/45, 1/525) + 相图; cosh 位置改为 "C_app 的条件最小元, 非分配问题的唯一解"。
