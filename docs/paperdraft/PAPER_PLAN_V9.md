# NeurIPS v9 论文写作规划（Definitive）

> **硬性目标**：正文 ≤ 9 页（Title/Abstract 到 Conclusion 末尾），References 不计，Appendix 不计
> **当前 v8 问题诊断**：正文约 13 页，Abstract 269 words（应 ≤150），段落冗余，自吹式语言（"This is one of the paper's strongest results"），bullet list 过多

---

## A. NeurIPS 格式硬约束

| 项 | 要求 |
|----|------|
| 正文 | ≤ 9 页（违反 = desk reject）|
| Abstract | ≤ 200 words（惯例 ~150 best）|
| References | 不计入页数 |
| Appendix | 不计入页数，在 References 后 |
| Broader Impact | 必须有，不计入页数 |
| Checklist | 必须有，不计入页数 |
| Column | single column, 5.5in × 9in text block |
| Font | 10pt Times |

---

## B. 叙事重构（v8 → v9 核心变化）

### B.1 v8 叙事（已过时）
- "EVQ beats YaRN/PI in from-scratch training"
- 问题：PE baselines 证明 YaRN PK@4K=100% > EVQ 82%，叙事站不住

### B.2 v9 叙事（新）

**一句话**：Training-time frequency optimization (EVQ) and inference-time length scaling (YaRN) are orthogonal; their combination achieves near-perfect extrapolation (98% at 4× context) that neither achieves alone.

**三层递进**：

1. **理论层**：RoPE frequency allocation 是变分逆问题 → ODE 闭式解 → 单参数 τ → Geometric 是 τ=0 退化点 → scaling law τ*=d/√L
2. **训练时实验层**：EVQ 单独在 4 个规模上改善 OOD PPL 10-19%，passkey +40pp，retrieval divergence（Geo 退化 vs EVQ 单调上升）
3. **🆕 组合层（killer result）**：EVQ + YaRN 8K retrieval = 98% vs Geo + YaRN = 62%（+36pp）。Training-time 和 inference-time 是正交优化维度，EVQ 提供更好的 foundation 让 inference-time PE 更有效

### B.3 要删除的内容
- ❌ 旧 Table 3（50M TinyStories competitor comparison）——对 YaRN 不公平的 from-scratch 对比
- ❌ Phase collision score verification table——移到 Appendix
- ❌ Affine calibration table——移到 Appendix
- ❌ 350M FineWeb 3-seed detail table——移到 Appendix，正文只引数字
- ❌ Section 4.5 "Implicit priors of existing methods"——融入 Related Work 一句话
- ❌ Physical interpretation 长段（Section 3.3 末尾）——移到 Appendix
- ❌ Corollary 1 后的 Taylor expansion 段——太细节，移 Appendix
- ❌ Section 5.3 中的 bullet list 分析——改为紧凑段落

### B.4 要新增的内容
- ✅ Table: PE baselines + EVQ+YaRN 组合（6-7 行，Section 5.3）
- ✅ 1-2 段论述 training-time vs inference-time 正交性 + 超线性叠加

---

## C. 正文页面预算（精确到 0.25 页）

NeurIPS 10pt single column，一页约 55-60 行正文 or 3 个 medium table。

| Section | 页数 | 行数约 | 表/图 |
|---------|------|--------|-------|
| Title + Authors + Abstract | 0.5 | 30 | — |
| 1. Introduction | 0.9 | 50 | — |
| 2. Related Work | 0.5 | 28 | — |
| 3. Joint Variational Framework | 2.0 | 110 | — |
| 4. Predictions | 0.8 | 44 | Tab affine calib |
| 5. Experiments | 3.3 | — | 4 Tab + 1 Fig |
| 6. Limitations | 0.5 | 28 | — |
| 7. Conclusion | 0.5 | 28 | — |
| **TOTAL** | **9.0** | | |

### Section 5 细分（3.3 页 = 战场）

| Subsection | 页数 | 内容 | 表/图 |
|------------|------|------|-------|
| 5.1 Setup | 0.15 | 1 段：base/L/τ/seeds/datasets | — |
| 5.2 PPL scaling (50M-750M) | 0.45 | Table 1 + 1 段 | Tab 1 |
| 5.3 Training-time vs Inference-time PE | 0.65 | **新 Table 2** + 2 段 | Tab 2 |
| 5.4 Passkey mix + capability preservation | 0.55 | Table 3 + antisymmetric + Prop 4 | Tab 3 |
| 5.5 750M retrieval divergence | 0.7 | Table 4 + Figure 1 + 2 段 | Tab 4, Fig 1 |
| 5.6 Theory validation | 0.4 | τ* table + collision dead zone（合并） | Tab 5 |
| 5.7 Pareto frontier (r-sweep) | 0.4 | r-sweep 结果 + r* 验证 | Tab 6 or Fig 2 |
| **Total** | **3.3** | | **4-5 Tab, 1-2 Fig** |

---

## D. 各 Section 写作规范

### D.1 Abstract（≤ 150 words，7 句）

| 句号 | 内容 | 约字数 |
|------|------|--------|
| S1 | Problem + gap | 20 |
| S2 | We formulate ... variational problem | 20 |
| S3 | Exact solution → EVQ, single parameter τ | 20 |
| S4 | Geometric RoPE = τ=0, strictly suboptimal | 15 |
| S5 | From-scratch 50M-750M: PPL -10-19%, passkey +40pp | 20 |
| S6 | **Killer**: EVQ + YaRN 8K=100% (3 seeds, zero variance) vs Geo+YaRN=65% | 25 |
| S7 | One-line replacement, zero hyperparameters | 15 |
| **Total** | | **~135** |

**禁止在 Abstract 中出现**：
- broadband decomposition 细节
- Brownian covariance / diagonal ridge
- antisymmetric scaling 细节
- 任何公式

### D.2 Introduction（0.9 页）

**Para 1**（动机，5 句）：RoPE → context extension bottleneck → existing methods heuristic → no principled objective → gap

**Para 2**（我们做什么，4 句）：variational formulation → ODE → exact solution → EVQ single parameter → Geometric is degenerate → τ* scaling law

**Para 3**（Contributions，4 条 itemize）：
1. Joint variational framework → governing ODE → exact solution (Theorem 1)
2. EVQ closed-form warp → Geometric is τ=0 degenerate (Theorem 2) → scaling law τ*=d/√L
3. From-scratch validation 50M-750M: PPL improvement + retrieval divergence + capability preservation
4. **Superlinear complementarity**: EVQ + YaRN achieves 100% 8K retrieval across 3 seeds (zero variance) vs 65% for Geo+YaRN (+35pp), establishing training-time and inference-time PE as orthogonal optimization dimensions

**风格**：
- 每条 contribution 以 "We derive / We prove / We validate / We demonstrate" 开头
- 不超过 2 句/条
- 不在 intro 重复 abstract

### D.3 Related Work（0.5 页，4 段）

| \paragraph | 内容 | 句数 |
|------------|------|------|
| Rotary and relative PE | RoPE, ALiBi, T5-bias, Kerple, FIRE | 3 |
| Context extension | PI, YaRN, NTK-aware, LongRoPE, sparse attn | 3 |
| Theoretical analyses | Prior work on PE theory + our contribution | 2 |
| Scaling and evaluation | Scaling laws, passkey, PPL eval | 2 |

**注意**：不要有 "How this work differs" 段——contribution 已在 intro 说完。

### D.4 Framework（2.0 页，5 subsections）

| Subsection | 核心内容 | 行数 |
|------------|---------|------|
| 3.1 RoPE as frequency allocation | ω(φ)=b^{-φ}, ρ(φ), S_ρ(Δ) | 8 |
| 3.2 Phase-collision energy | 公式 (1)-(3): C[ρ], kernel K, scale separation | 25 |
| 3.3 Joint objective | 公式 (4)-(6): J[ρ], stationarity, ODE. **Theorem 1** | 30 |
| 3.4 EVQ | 公式 (7)-(8): CDF inversion, EVQ warp. **Theorem 2 + Corollary** | 25 |
| 3.5 Structural theorems | Prop 1-3，每个 2 行 | 18 |

**删减**：
- Physical interpretation 段 → Appendix
- "When μ=0..." 段 → Appendix
- Corollary 1 后的 Taylor expansion + "high-frequency spacing" 分析 → 压缩为 1 句
- Eq numbers 只给被引用的公式

### D.5 Predictions（0.8 页，4 subsections）

| Subsection | 保留？ | 处理 |
|------------|--------|------|
| 4.1 Waterbed inequality | ✅ 保留 | 公式 + 3 句解释 |
| 4.2 Finite-base calibration | ⚠️ 精简 | 删 table，保留 1 句 "R²>0.99 confirms affine approximation" |
| 4.3 τ* scaling law | ✅ 保留 | 公式 + 2 句推导概要 |
| 4.4 Collision-block | ✅ 保留 | 公式 + dead zone prediction |
| ~~4.5 Implicit priors~~ | ❌ 删除 | 移 1 句到 Related Work |

### D.6 Experiments（3.3 页）

#### 5.1 Setup（1 段）
"All from-scratch experiments use base b=500K (matching LLaMA-3/Qwen-2), L_train=2048, EVQ τ=1.5 (predicted by Eq.X: 64/√2048≈1.41). No architecture or inference changes beyond one line of inv_freq initialization. Full configurations in Appendix H."

#### 5.2 PPL Scaling（Table 1 + 1 段）

Table 1: 5 行（50M/125M/350M/350M-FW/750M），列: Scale | Dataset | Seeds | Δ@2K | Δ@16K

段落：highlight consistency across scales, waterbed asymmetry, 750M Hybrid note

#### 5.3 Training-time vs Inference-time PE（**新，核心 section**）

**Table 2（最重要的表，论文 killer result）**：

用 5% 3-seed 数据（多 seed 确认，统计更强）：

| Method | Type | Seeds | PK@8K | PK@12K | PK@16K |
|--------|------|-------|-------|--------|--------|
| Geo (no PE) | baseline | 3 | 54±11% | 55±5% | 55±14% |
| Geo + YaRN | infer | 3 | 65±6% | 54±4% | 56±6% |
| EVQ τ=1.5 | train | 3 | 57±5% | 58±2% | 56±9% |
| **EVQ + YaRN** | **train+infer** | **3** | **100±0%** | **63±4%** | **70±14%** |

补充 10% single-seed 表（含更多 PE baselines，放正文或 Appendix）：

| Method | Type | PK@4K | PK@8K | PPL@4K | PPL@8K |
|--------|------|-------|-------|--------|--------|
| Geo (no PE) | baseline | 42% | 46% | 94.9 | 156.5 |
| PI | inference | 54% | 56% | 198.9 | 204.2 |
| Dynamic NTK | inference | 60% | 50% | 93.1 | 115.7 |
| NTK-aware | inference | 100% | 50% | 74.8 | 90.8 |
| YaRN | inference | 100% | 62% | 72.5 | 82.4 |
| **EVQ τ=1.5** | **training** | **82%** | **60%** | 95.3 | 152.5 |
| **EVQ + YaRN** | **train+infer** | **100%** | **98%** | 74.2 | 82.3 |
| **EVQ + NTK-aware** | **train+infer** | **100%** | **88%** | 73.7 | 96.8 |

Caption（3-seed 表）: "Training-time vs inference-time frequency optimization (350M, 5% passkey mix, base=500K, 3 seeds). EVQ + YaRN achieves 100% retrieval at 4× extrapolation (8K) across all three seeds with zero variance, compared to 65±6% for Geo + YaRN (+35pp). The effect extends to 8× extrapolation (16K): 70% vs 56% (+14pp). This demonstrates superlinear complementarity between training-time and inference-time optimization."

**Para 1**（结果概述）：
- YaRN/NTK-aware reach 100% at 4K (2×) through inference-time rescaling alone
- EVQ alone reaches 82% at 4K through training-time reallocation
- These operate at different stages and are complementary

**Para 2**（超线性叠加 = killer paragraph，用 3-seed 数据）：
- EVQ + YaRN at 8K: **100% across all 3 seeds, zero variance**, vs Geo+YaRN = 65±6%（+35pp）
- The effect persists at extreme extrapolation: 12K +9pp, 16K +14pp
- EVQ's expanded low-frequency spacing creates representations more amenable to inference-time rescaling
- This establishes training-time and inference-time PE as orthogonal optimization dimensions with superlinear interaction

#### 5.4 Passkey Mix + Capability Preservation（Table 3 + 段落 + Prop 4）

Table 3: 保持 v8 的 passkey mix table (4 行: 2K/4K/8K/16K × Geo/EVQ retrieval + PPL)

1 段：+40pp at 4K, antisymmetric 5%→10% scaling (1 句), Prop 4 statement (2 句)

#### 5.5 750M Retrieval Divergence（Table 4 + Figure 1）

Table 4: 4 checkpoints × PPL@8K / PK@8K / AR (6 columns, 保持 v8)

Figure 1: 三联图（保持 v8）

2 段：
- Para 1: Retrieval divergence description (Geo regresses, Hybrid improves monotonically)
- Para 2: Training efficiency (Hybrid@50% > Geo@100%) + zero in-dist cost

**不要写**：
- ❌ "This is the paper's most striking finding"
- ❌ Bullet list analysis
- ❌ "dynamically degenerative" 超过 1 次

#### 5.6 Theory Validation（合并 τ* + collision）

Table 5: τ* scaling 5 rows（保持 v8）

1 段：τ* matches for L≥1024, systematically higher at short L

Table 6: collision dead zone base=10K（保持 v8 但精简为 3 行）

1 段：dead zone confirms collision-block analysis, negative result validates theory

#### 5.7 Warp Boundary Pareto Frontier（r-sweep）

Table 7 或 Figure 2: r-sweep 9 points (r vs PPL@2K vs PPL@16K vs Δ@2K vs Δ@16K)

1 段：monotonic waterbed, r* formula first-order approximation, Pareto optimal near r=0-8

### D.7 Limitations（0.5 页，紧凑段落 NOT bullet list）

5 个要点写成 2 段连续散文：
1. Scale-separation approximation O(1/ln b) residual
2. Waterbed trade-off intrinsic (750M Hybrid OOD PPL +5.7%)
3. Model scale 50M-750M (PE papers publish at 125M)
4. τ* validated at 5 points, single base
5. Single seed for 750M and 10% passkey mix

**风格**：段落散文，不用 bullet list，不用 bold label，不自我贬低

### D.8 Conclusion（0.5 页，2 段）

**Para 1**（理论总结）：variational formulation → ODE → EVQ → Geometric degenerate → τ* scaling law

**Para 2**（实验总结 + impact）：
- PPL improvement 10-19% across 4 scales
- Retrieval divergence at 750M
- **EVQ + YaRN superlinear complementarity** (98% vs 62% at 8K)
- Training-time and inference-time optimization are orthogonal
- One-line replacement, zero hyperparameters
- Future: r* validation, layer-wise allocation, billion-scale

---

## E. 写作风格禁令

| 禁止 | 替代 |
|------|------|
| "This is one of the paper's strongest results" | （删掉，让数据说话） |
| "striking", "remarkably", "interestingly" | "we observe", "the data show", "notably" |
| "In this subsection we..." | （直接写内容） |
| Bullet list in main text analysis | 段落散文 |
| 重复 abstract 内容在 intro | （用不同角度描述） |
| "We leave X for future work" 超过 1 次 | 在 Conclusion 集中说 |
| 自我引用 "our method" | "EVQ" or "the proposed allocation" |
| Bold 行内数字超过每段 2 个 | 只 bold 最关键的 |

---

## F. 表格总览（正文 6-7 个）

| ID | 内容 | Section | 约行高 |
|----|------|---------|--------|
| Tab 1 | PPL scaling 50M-750M | 5.2 | 8 行 |
| Tab 2 | **PE baselines + EVQ+YaRN 组合** | 5.3 | 10 行 |
| Tab 3 | Passkey mix (Geo vs EVQ, retrieval+PPL) | 5.4 | 7 行 |
| Tab 4 | 750M training dynamics (4 ckpt) | 5.5 | 7 行 |
| Tab 5 | τ* scaling law (5 L values) | 5.6 | 8 行 |
| Tab 6 | Collision dead zone base=10K | 5.6 | 5 行 |
| Tab 7 | r-sweep Pareto frontier | 5.7 | 11 行 |

→ 7 tables. NeurIPS 惯例 5-8，可以。如果超页考虑合并 Tab 5+6 或移 Tab 7 到 Appendix。

## G. 图总览（正文 1-2 个）

| ID | 内容 | Section |
|----|------|---------|
| Fig 1 | 三联图：freq alloc + PPL dynamics + passkey dynamics | 5.5 |
| Fig 2 | r-sweep Pareto frontier（if space，否则用 Tab 7） | 5.7 |

---

## H. Appendix 规划（从正文移出的内容）

| Appendix | 内容 | 来源 |
|----------|------|------|
| A | Theorems 1,2 full proofs | 保持 v8 |
| B | Prop 1 proof (uniform prior) | 保持 v8 |
| C | Prop 2 proof (power-law prior) | 保持 v8 |
| D | Prop 3 proof (resonance) | 保持 v8 |
| E | Diagonal approximation + residual | 保持 v8 |
| F | Structural stability (Brownian, Fredholm, Waterbed proof) | 保持 v8 |
| G | Non-asymptotic + discretization guarantees | 保持 v8 |
| H | Experiment configs | 保持 v8 |
| **I** | **Physical interpretation of Theorem 1 solution** | **从 §3.3 移出** |
| **J** | **350M FineWeb 3-seed PPL detail** | **从 §5.2 移出** |
| **K** | **Phase collision score verification** | **从旧 §5.7 移出** |
| **L** | **Affine calibration table** | **从 §4.2 移出** |
| **M** | **5% 3-seed full data + PE baselines passkey depth detail** | **新** |

---

## I. 执行 Checklist

- [ ] Abstract ≤ 150 words → word count 验证
- [ ] 正文 ≤ 9 pages → pdflatex 编译后测量
- [ ] 每个 Table caption 自包含（独立可读）
- [ ] 每个数字引用来源（Appendix table or inline）
- [ ] Figure 1 referenced in text
- [ ] All theorems/propositions numbered consistently
- [ ] No "striking/remarkably/interestingly"
- [ ] No bullet list analysis in experiments
- [ ] Broader Impact present
- [ ] Checklist complete
- [ ] References complete（no missing citations）
