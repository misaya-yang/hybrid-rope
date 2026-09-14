# 未使用高价值资产清单

创建 2026-09-14，**2026-09-14 修订**（用户复核后重写）。只读盘点，不是执行队列，不构成编辑授权。

**评估口径**（按用户给定）：是否补上正文缺失的**论证或关键归因**；是否值得替换现有版面。
**不以「增加实验数量」为理由收录。**

---

## 更正记录（第一版错误，已修）

| # | 第一版的错 | 更正 |
|---|---|---|
| 1 | **OLMo 全表 Σm 少算 1**。我求的是 `m_0..m_{N−1}`，漏掉终端 `m_N=1` | BM/Uni **40.5**（非 39.5）；Pro **37.667**（非 36.667）。**差值 2.8333 不变，结论仍成立**；与 C42 实测 `sum_m=42` 自洽（质心 8 → 11+31=42） |
| 2 | 把 gain 因子判为「不加」，理由是「它显示 allocation 不是主导杠杆」 | **推理错**。它不否定 z 的独立价值，只要求分清**联合方法收益的来源**。改为「可加，但必须做归因切分」 |
| 3 | 写「TailSpline 正文 0 命中」 | **已过期**。本轮改稿已加入：`sections/04_experiments.tex:20-36` 有完整 TailSpline 小节，`appendix/a10_tailspline.tex` 已 `\input`，`fig_tailspline_main.pdf` 已用 |
| 4 | B13 的 Λ 比值序列 `1.000/1.188/1.421/1.610/1.631/1.336/0.588` | **不在 `evq_three_completions.tex` 里**（`1.631` 全仓找不到）。**已删除**，只保留该文件中确实存在的数字 |
| 5 | B15 写 `5.394245 / 5.394257 / 5.150449 / 5.150463`（6 位） | 源文件只有 **`5.3942 / 5.3943 / 5.1504 / 5.1505`**（4 位）。已改 |
| 6 | C13 把 6 个数字标为单一来源 | 数字真实但**散落在不同文件**（见下）。已逐条重新标注来源 |
| 7 | 多条把他文件数字并入被引文件 | 凡未逐条确认来源者，一律降级到文末「未核实线索」，**不给数字** |

**根因**：我把子代理报告的数字直接转录，只以 ⚠️ 标记当免责。那是转述，不是核验。

---

# 一、用户已采纳的三项（本轮核对无误）

### 1.1 BM–Uni 等总位移对照 ✅

**BM 与 MrRoPE-Uni 的内部累计和完全相同；BM 比 MrRoPE-Pro 多 (N−1)/6。**

按 `appendix/a9_recovered_design_evidence.tex` 自己的约定复算（transition 上的累计值为 `m_1..m_N`）：

| 配置 | transition Σm | 全表 Σm | 增量质心 |
|---|---|---|---|
| **BM** | **9.5000 = (N−1)/2** | **40.5000** | 9.5000 |
| **Uni** | **9.5000 = (N−1)/2**（与 BM 逐位相同） | **40.5000** | 9.5000 |
| **Pro** | **6.6667 = (N−1)/3** | **37.6667** | 12.3333 |

N=18：**BM − Pro = 17/6 = 2.8333 Σm ≈ ln4 × 2.8333 = 3.93 nats**。
N=17（Qwen `[23,40]`）：8/3 = 2.6667，≈ 3.70 nats。

**自洽性检验**：C42 两臂记录 `sum_m=42`、质心 8 ⇒ transition 和 = 19−8 = 11，加平台 31 = **42** ✓

**归因后果**：
- `sections/02_identification.tex` 的「BM improves 16K macro from 2.78% to 51.32% relative to MrRoPE-Pro」是**剂量混淆**的（BM 多 3.93 nats 总压缩）。
- 论文自己的表里已有剂量匹配对照：`a9` 的 `tab:four-method-control` 中 **MrRoPE-Uni 32.12 vs BM 51.32** 共享端点、band、gain **且 Σm**。+19.20pp 是固定总位移下的形状对照。

### 1.2 χ² 刚度精确闭式 ✅

`sections/04_construction.tex` 与 `appendix/a1_proofs.tex:539-541` 现用 `S_χ²(τ) = τ⁴/(45 d_head) + O(τ)`。精确值为

**S_χ²(τ) = (1/d_head)·[sinh τ·arctan(sinh τ)/τ² − 1]**

截断误差：τ=1.414 高估 **1.52×**；τ=2.0 高估 1.97×；τ=4.0 高估 **3.52×**。
而 `tab:allocation-protocols` 全部协议 τ ∈ [1.414, 4.0] —— **现行展开在论文每一个操作点上都无效**。全文 `arctan` 0 命中。

### 1.3 8B 未适配底模（`geo_base`）臂 ✅

`paper-2027/figs/llama_temporal_summary.json` 的 `source_manifest_sha256`
= `187c8ff86f0e52229b0ca6c6a55b506c4e536c959a95bb5bbe06399c1611879e`，
与三臂源文件 `rebuttal/pre_rebuttal/seed42_lora_eval_20260713/raw/legacy_results/temporal_three_arm_2026.json`
的 `collection_manifest_sha256` **逐字节相同**；后者 `arms = [evq_lora, geo_base, geo_lora]`，图的 JSON 只有两臂。

| 长度 | geo_base（未适配） | geo_lora（匹配 LoRA） | evq_lora |
|---|---|---|---|
| 8K | 2.0730 / **7.948** | 1.9195 / 6.817 | 2.3093 / 10.068 |
| 16K | 5.0143 / **150.545** | 4.6910 / 108.958 | 3.1809 / **24.068** |
| 32K | 7.3085 / **1492.915** | 6.8992 / 991.475 | 4.8513 / **127.911** |

相对同一底模 PPL 比：**匹配 LoRA 0.858 / 0.724 / 0.664；allocation 1.267 / 0.160 / 0.086**。
⇒ 16K/32K 上，**匹配适配只买到 28%/34%，allocation 买到 84%/91%**。

**⚠️ 源文件未被 git 跟踪**（worktree only），进论文前需先 commit。
**⚠️ 8K 窗口成本仍在**（base 7.948 < evq_lora 10.068），论文已披露，引用时须保留。

---

# 二、已核验、指向「机制解释 / 关键归因」的候选项

### 2.1 50M crossing 的因子分解（归因） ✅

`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md:391,400`

**E_T = +0.5991、E_W = −0.5965、I_{T×W} = −3.5367** ⇒ **交互项是大主效应的 5.90×**
95% CI：`[+0.331,+0.956]` / `[−0.896,−0.175]` / `[−5.165,−3.039]`

把「权重会共适应」从定性升级为**量化压倒主效应**。`table_coadapt.tex` 只带 4 个 PPL/r₂ 格。
**归因价值高于新增实验。** ⚠️ seed-42 only（报告自标 seed-43/44 未做）。

### 2.2 ε⁴ 塌缩指数在真实训练模型上被实测 ⚠️

`foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §3.3；脚本 `scripts/analysis/attention_fisher_50m_probe.py`
跨 4 个 weights/table 格、**1,920 个 head-query 观测**，chordal-deficit log-log 斜率 **4.009–4.011**，与 `O((ωΔ_max)⁴)` 一致。

这是唯一把论文解析指数拿到真实模型自身几何上验证的地方。⚠️ 验证的是**指数**不是行为；provenance 有缺口（报告引脚本 SHA `9e133a83…`，树里是 `9c5d79b0…`）。

### 2.3 先验敏感性 —— 已算好，但在孤儿文件里，且**双刃** ✅

`paper-2027/research/three_completions/evq_three_completions.tex`（59 KB）Part III "non-uniform separation priors"，
含 `tab:overload`、`prop:overload`、Prior-side ceiling lemma。核验脚本 `verify_three_completions.py` Parts B/C（**需 SciPy**）。

**该文件中确实存在的数字**：`Λ(Unif[0,1]) = 1.5079516e-3` vs `19/12600 = 1.5079365e-3` ✅
**该文件中的原句**（逐字）：
- 「The r₂=2.00 slow-collapse figure is **prior-robust**. But under a Zipf prior with tail index α≥1.5 the r₂(τ) curve acquires an interior maximum at τ≈3.5–6: **high-frequency overload is real**. It never reverses the Geo/EVQ ordering, but **it erases the margin**.」
- 「\evq{} ≥ Geo at every α tested, with the margin shrinking from **8.0×** (α=0, τ=4) to **2.9×** (α=1) to **1.17×** (α=2) to **1.11×** (α=2.5).」
- 「The mitigation of slow-frequency collapse survives under a Zipf prior; **the magnitude of the benefit does not.**」
- 「the static advantage quoted at α=0 is an **upper bound** on what a realistic prior would show」

**⚠️ 这不是纯资产**：同文件另有不利判定 —— `b=256` 固定支持对照「sits on the formula's degeneracy locus」、其部署 `τ=4` 是「**1.7× the model's optimum**」、「The rank analysis does **not** separate allocation from support」。
**整段搬入会连带搬入这些负面陈述。** 且仓库自己测得的距离先验约 `r^{−2.4}` → α≈2.5 → 优势仅 1.11×。
**要引必须逐句挑，并明确保留哪些。**

### 2.4 唯一性起点常数 `r_orth` + 论文自身 r₂ 的第三方重算 ✅

`paper-2027/research/attention-aware-retrofit/analysis/PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md` §V3–V5（行 88–110）

`r_orth := 1/(1−b^{−1/K})`，渐近 `= K/ln b + 1/2 + ln b/(12K) + O(K⁻²)`
**COMPUTED: OLMo exact 5.3942 vs expansion 5.3943；Qwen 5.1504 vs 5.1505**（原文精度）

V4 独立重实现：`r₂(native, U[0,4096]) = 8.067`；`U[0,16384] = 12.641`；23 塌缩对 `r₂ = 2.0001`（论文称 2.00）
V5：**99.16%** 的成对碰撞质量落在 `r<1` 带；两端都 `r≥1` 的对只占 **0.84%**

⚠️ V4 的 8.067/12.641 用的网格尺寸论文里没有，须写明 support/measure，否则「比较未定义」。

**同文件 V6 另有一条**（我方代理未报，我读到）：YaRN 阈值就是旋转圈数截断 ——
`idx(r) = 128·ln(L/(2πr))/(2·ln b)` 给出 OLMo `(⌊14.7⌋, ⌈31.6⌉) = (14,32)`、Qwen `(23,40)`；
且「The frozen protocol's (1,32) is exactly YaRN's (β_slow=1, β_fast=32) in rotation coordinates」。

---

# 三、已核验的其他候选项

| # | 内容 | 核验 |
|---|---|---|
| **3.1** | **A31–A34**（registry 标 `current manuscript claim not yet changed`）。A31 LlamaS4 AUC 80.95 vs BM 74.68（+6.27 [+3.10,+9.45]）；A34 QwenS2 AUC 79.72 vs BM 75.84（+3.88 [+0.81,+6.93]）、**Native@32K +6.51 [+1.45,+11.88]**；A32 LlamaS8 64K 57.45 vs MrPro 52.50；A33 OLMoS8 AUC 41.87 vs BM 20.28（+21.59 [+13.70,+29.59]）。注册表 `interpretation` 字段已写好限定语，沿用即可 | ✅ 数字对 owner 文档逐条核过 |
| **3.2** | **held-out base-1M / d_head=128 / 3 seed** —— `git show 79e52bb:rebuttal/rebuttal_0723/EXPERIMENT_REPORT_20260724.md` §4。512→16K ΔNLL：+0.0694 / −0.8018 [−0.9351,−0.6684] / −0.6640 / −0.4329 / −0.2871 / −0.2117。全部 CI 排除零，3 seed 同向 | ✅ |
| **3.3** | **151.9M anchor 级 CI**（已算好从未印出）。`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json`。fixed **512 [−0.7573,+0.1959] 跨零**；1024 [−0.2720,−0.0800]；2048 [−0.2303,−0.0611]。JSON 自带 `descriptive_with_n_3_no_generic_significance_claim` | ✅ |
| **3.4** | **τ 下限** `τ_floor(N,K) = 4√(N/K)·[1 + N/(2K) + (241/120)(N/K)² + O(K⁻³)]`，`scripts/theory_B_floor_higher_order.py:4` | ✅ 公式在源文件 |
| **3.5** | **区间预算改写** `J[h] = ½∫¹[α/h(u) + β(1−u)²h(u)]du`，闭式 gap profile。K=64,τ=2：等距 gap ≈**0.21587**，EVQ 首 gap ≈**0.10483**、末 ≈**0.39124**（`docs/research/rope_allocation_20260910/source_inputs/pro6_allocation_analysis.md:248`） | ✅ 数字在源文件 |
| **3.6** | **gauge no-go + Neyman 计数律** `ρ*(x) = σ(x)|w(x)|/∫σ|w|`，**不含 `α∫ρ²`**（`docs/research/rope_allocation_20260910/agents/astra07.md:197`）。⚠️ 负结果，作范围框定 | ✅ 原文在源文件 |
| **3.7** | **整数 DP + 离散 Cosh 递推** `p_i ∝ cosh[κ(B+½−i)]`（`astra06.md`）。B=16,K=64,α=1,β=4：variance 2.20953369140625 vs 均匀 2.46094。⚠️ 精确有限 K 结果仍是整数 DP，**不可把递推说成有限表最优** | ✅ 数字在源文件 |
| **3.8** | **BM 逐槽支配 Pro** `m^BM_q − m^Pro_q = 2q(q+1)(N−q)/[N(N+1)(N+2)] > 0` | ✅ |
| **3.9** | **fresh_72**（432 条生成记录在库，`score_mismatches: []`）。4K/16K macro：BM **77.50/48.23**、**official YaRN 51.04/14.06**、**MrRoPE-Pro 26.53/13.54** | ✅ |
| **3.10** | **C42 稳健性包**：bootstrap 95% CI **[+6.97,+14.63]**，0/20000 ≤0；剔 `niah_single_3` 后 +6.5167pp, t=+3.58 | ⚠️ |
| **3.11** | **剪枝因果**：`results/heldout_causal_pruning_s42_20260724/`，超 L_train 后 −0.2996 [−0.3558,−0.2429] @1024、−0.2025 @4096、−0.2829 @8192，训练长度处**完美零**。自身标 `paper_claim=false`、post-hoc、单 seed | ✅ flags 已核 |
| **3.12** | **逐对因果谱**：Spearman ρ **+0.760**（256）→ **−0.807**（8192）。⚠️ 须作边界/负结果 | ✅ |
| **3.13** | **固定支持剂量响应**（**未登记进 registry**）：128 篇新文档，冻结权重。λ=0.02 时 16K tail **−0.11528 [−0.13378,−0.09656]**。预注册 P1/P2/P3 pass、**P4 fail**、联合门 fail | ✅ |
| **3.14** | **b 轴剂量响应**：b=0(≡MrPro) → b=1(≡BM) 的 6 个点（0.07086…0.55871）+ **72 行 held-out：b=1 与 b=3 都是 0.53495 —— 样本外 0.00pp**。⚠️ 数字**散落在多个文件**（`olmo_b3/beta_b3p0_summary.json`、`holdout/beta_b{1,3}p0_summary.json`、`_reports/LEDGER_20260911.md`、`audit/pro_decision_20260911/check_results.json`），非单一来源 | ️ 逐文件确认存在 |
| **3.15** | **YaRN freq vs mscale 分量消融**：两架构一致，**mscale 单独中性到有害**（gap 0.206 vs 不做的 0.177） | ✅ 0 命中已核 |

---

# 四、未核实线索（**只给路径，不给数字**）

这些是子代理报告的，我**没有**逐条确认数字来源。**在核验之前不要引用任何数字。**
（第一版里它们带着数字，是这次更正删掉的主要部分。）

| 路径 | 声称内容 | 待查 |
|---|---|---|
| `docs/research/next_stage_20260912/5090_SOL_HANDOFF_20260912.md:116` | E3 980 行固定总位移确认 | raw 在**离线 5090**，仓库内无法复算；且是不同于 a9 的面板 |
| `results/weekend_sweep/`（63 个 GPU run） | 50M TinyStories τ×L 网格 | **其分析 JSON 被 bug 污染**（见 R1）；须先修正基线再重算 |
| `results/video_dit/BASE_SWEEP_REPORT.md` + `base_sweep_b*_summary.json` | 6 base × 2 臂跨模态 | 文件存在，数字归属未逐条核对 |
| `paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md` | 非循环 ramp 对照 | 未核 |
| `paper-2027/research/attention-aware-retrofit/evidence/QWEN_K64_MATCHED_S2_BASELINE_RECEIPT_20260901.json` | 同 checkpoint YaRN s=2 vs s=4 | 未核；**双向刃** |
| `paper-2027/research/foundations/..._20260819.md` §3.2 | 慢带原始能量秩与维度损失 | 未核 |
| `docs/research/next_stage_20260912/{LLAMA,OLMO}_S8_*`、`QWEN_S2_*` | 即 3.1 的 owner 文档 | 已核（见 3.1） |
| `analysis/p0_gradients/full_model_response.jsonl` | 真 128K 梯度带质量 | **反向证据，建议不加**（见 R4） |

---

# 五、取舍判断（修正后）

### 「不加」——修正后的理由

| 项 | 修正后的理由 |
|---|---|
| **gain 2×2 因子** | **理由改了**：不是「它证明 allocation 不重要」而排除。它是**归因工具** —— 要求把联合方法的收益拆成 gain 与 table 两部分。**可加，但必须做归因切分**，不能当作「z 无效」的证据。⚠️ 两个端点 raw 未在本机保存（1.0 与 1.1386 的 350 行文件缺失，1.20 只有 83/350） |
| MLA linear-vs-sqrt 稀释 | 前提数值不成立：推导要求带内质量随 τ 线性增长，实测**递减**（0.0527→0.0404） |
| softmax transport 指数匹配 | 已被 `optimization_notes.md` O3 取代；紧邻 `c_coll` 硬红线 |
| LoRA rank 相变 r_c≈K | Λ₀ 是拟合的；r=K=64 时理论预测满能力，与论文 8B 的 deficit 冲突 |
| `appendix/a4_supporting_experiments.tex` / 125M composition | **模型身份冲突**：`EXPERIMENT_ASSETS_TOP15_20260909.md:33` 记同一组 260.2/99.6 被标成 125M/350M/454M，判「暂不晋升」。orphan 化正确 |
| 稀疏记忆线 | `CLOSED/ASSAY_UNQUALIFIED`，dense 对照也失败 ⇒ 仪器失败 |
| `kappa_att` / `L_eff_probe` | 预注册 Tier-1 失败；L_eff 只有「预期形状」 |
| composition triple | owner 报告明说相对纯 FMRoPE **更差** |
| `rope_z_ood_collision` | 结构性 CPU 失败 |
| `checkpoint_attention_replay` | 从未运行 |
| **A28** full-z oracle | 已被执行负结果关闭（`scientific_passed: false`） |
| A36 fixed-u | 论文从未声称该迁移 |
| **A24** base-only | **已在论文里**（`a1_proofs.tex` "Approximate compatibility within a geometric family"） |

### 方针：**不靠「零命中」判定**

第一版大量用「正文 0 命中」当收录理由 —— 这既会过期（TailSpline 就是例子），也**不是**该用的判据。
正确判据是：**它是否补上正文缺失的论证或归因，以及是否值得替换现有版面。**

---

# 六、风险项

| # | 内容 |
|---|---|
| **R1** | **`weekend_sweep` 的两个 JSON 被 bug 污染，会产生「论文命题被证伪」的假象。** `scripts/m4_max_36gb/analyze_weekend_sweep.py:55` 用 `r.get("tau", 0.0)` 读 `results_checkpoint.json`，把三个 `tau: None` 的 PI 臂并进 τ=0 组。`analysis/tau_floor_check.json` 的 τ=0 基线因此是 29.9303 (n=6)，正确值 12.1217 (n=3)（**`analysis/summary.md:15` 的手写表 12.12 ± 0.87 n=3 是对的**）。该 JSON 的判决「invalidates Proposition 2」是**伪影**。**修一行即可** |
| **R2** | **τ anchor 脚本不可当验证引用**：`scripts/verify_tau_unified.py` 手输 15 个 anchor，实测 mean err 9.6%、**max 33.3%**、仅 10/15 在 15% 内，却因阈值设在**均值**上而打印 `PASS` |
| **R3** | τ selector 未通过验证（`historical_gate_pass=false`）；碰撞最优离散 τ 对每个 (K,L) 都是 5.0，与 τ=K/√L 冲突 |
| **R4** | 真 128K 梯度带质量是**反向证据**：`analysis/p0_gradients/full_model_response.jsonl` 显示快带占 ‖g‖² 的 91–95%，中带仅 0.005–0.014%。方向与主张 (1) 相反，且模型不匹配、slot 索引未统一 → **不建议加入** |
| **R5** | `scripts/validate_rebuttal_evidence_bundle.py` **正则损坏**：`[REDACTED_AUTHOR]` 是字符类，匹配任一 `c/t/a/e/d/r/h/u/o/_` ⇒ 假阳性。**守的是旧 bundle，不影响本次投稿** |
| **R6** | `ds_workspace/.../coverage_theory_results.json` 数值错：`turns_a2_b32` 记 39.6667，raw summary 是 **42.0** |
| **R7** | `main.tex` 仍有 ICML 残留引用 `venue_icml_fallback/` |
| **R8** | `tables/table_coadapt.tex` 钉了网格但**未钉分离先验**（R2 均匀先验重算 4.78/13.68 vs 正文 4.57/12.54） |
| **R9** | **8B `geo_base` 臂存在于图源 provenance 却不在表里** —— 披露缺口 |
| **R10** | `docs/theory/EVQ_COSH_THEORY.tex`（power-law 先验）与 `appendix/a1_proofs.tex`（uniform）先验不一致 |

---

# 七、变更记录

- 2026-09-14 创建（第一版，含错误）
- 2026-09-14 **修订**：修正 Σm off-by-one（结论不变）、gain 项理由、TailSpline 过期判断；
  删除 B13 中不属于该文件的 Λ 序列；修正 B15 精度；C13 改逐文件标注；
  **新增第四节「未核实线索」**，把未逐条确认来源的内容从正文移除；
  全文改为按「是否补上缺失论证/归因」而非「零命中」评判。